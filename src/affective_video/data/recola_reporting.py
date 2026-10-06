"""Private descriptive summaries and optional plots; never model performance."""

import csv
import json
from pathlib import Path

import numpy as np

from affective_video.data.adapters.recola import RecolaDatasetAdapter
from affective_video.data.adapters.recola_schema import ManifestRow
from affective_video.data.adapters.recola_tables import value_statistics
from affective_video.data.recola_splits import split_records


def write_manifest(path: Path, rows: list[ManifestRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(ManifestRow.model_fields)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in sorted(rows, key=lambda row: (row.participant_id, row.video_id)):
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True)
                    if isinstance(value, (dict, list))
                    else value
                    for key, value in row.model_dump(mode="json").items()
                }
            )


def dataset_summary(
    adapter: RecolaDatasetAdapter, rows: list[ManifestRow], split: dict | None = None
) -> dict:
    eligible = [row for row in rows if row.eligible]
    targets: dict[str, dict[str, list[float | None]]] = {}
    trace_summaries = {}
    disagreement = {}
    if adapter.mapping:
        for recording in adapter.mapping.recordings:
            for dimension in ("valence", "arousal"):
                individuals = []
                for spec in recording.annotations:
                    if spec.dimension != dimension:
                        continue
                    key = f"{recording.video_id}:{dimension}:{spec.kind}:{spec.annotator_id}"
                    if key not in adapter.traces:
                        continue
                    trace = adapter.traces[key]
                    trace_summaries[key] = {"timing": trace.timing, "statistics": trace.statistics}
                    frame = adapter.timestamp_traces.get(recording.video_id)
                    if frame and frame.timestamps and trace.timestamps:
                        trace_summaries[key]["video_coverage"] = {
                            "start_offset_seconds": trace.timestamps[0] - frame.timestamps[0],
                            "end_offset_seconds": trace.timestamps[-1] - frame.timestamps[-1],
                            "overlap_seconds": max(
                                0.0,
                                min(trace.timestamps[-1], frame.timestamps[-1])
                                - max(trace.timestamps[0], frame.timestamps[0]),
                            ),
                        }
                    if spec.kind == "consensus":
                        targets.setdefault(recording.participant_id, {}).setdefault(dimension, [])
                        targets[recording.participant_id][dimension].extend(trace.values)
                    else:
                        individuals.append(dict(zip(trace.timestamps, trace.values, strict=True)))
                shared: dict[float, list[float]] = {}
                for individual in individuals:
                    for timestamp, value in individual.items():
                        if value is not None:
                            shared.setdefault(timestamp, []).append(value)
                deviations = [
                    float(np.std(values)) for values in shared.values() if len(values) >= 2
                ]
                disagreement[f"{recording.video_id}:{dimension}"] = {
                    "individual_trace_count": len(individuals),
                    "exact_timestamp_multiannotator_count": len(deviations),
                    "mean_annotator_std": float(np.mean(deviations)) if deviations else None,
                    "method": "raw-scale std at exact shared timestamps; no interpolation",
                }
    participants = {}
    for participant in sorted({row.participant_id for row in rows}):
        selected = [row for row in rows if row.participant_id == participant]
        participants[participant] = {
            "annotated_duration_seconds": sum(row.annotated_duration_seconds for row in selected),
            "eligible_recordings": sum(row.eligible for row in selected),
            "targets": {
                dimension: value_statistics(tuple(targets.get(participant, {}).get(dimension, [])))
                for dimension in ("valence", "arousal")
            },
        }
    distribution = {
        dimension: value_statistics(
            tuple(
                value
                for participant in targets.values()
                for value in participant.get(dimension, [])
            )
        )
        for dimension in ("valence", "arousal")
    }
    partition_stats: dict[str, dict] = {}
    if split:
        for partition in ("train", "validation", "test"):
            selected_videos = {
                record.video_id for record in split_records(split) if record.partition == partition
            }
            partition_stats[partition] = {}
            for dimension in ("valence", "arousal"):
                values: list[float | None] = []
                if adapter.mapping:
                    for recording in adapter.mapping.recordings:
                        if recording.video_id not in selected_videos:
                            continue
                        for spec in recording.annotations:
                            if spec.dimension == dimension and spec.kind == "consensus":
                                key = ":".join(
                                    (
                                        recording.video_id,
                                        dimension,
                                        "consensus",
                                        str(spec.annotator_id),
                                    )
                                )
                                values.extend(adapter.traces[key].values)
                partition_stats[partition][dimension] = value_statistics(tuple(values))
    return {
        "eligible_participants": len({row.participant_id for row in eligible}),
        "eligible_recordings": len(eligible),
        "recording_count": len(rows),
        "eligible_annotated_duration_seconds": sum(
            row.annotated_duration_seconds for row in eligible
        ),
        "duration_definition": "participant-recording overlap spans; gaps are not subtracted",
        "target_statistics_scope": "raw consensus traces; not aligned M3 observations",
        "participants": participants,
        "targets": distribution,
        "partitions": partition_stats,
        "traces": trace_summaries,
        "frame_timing": {video: trace.timing for video, trace in adapter.timestamp_traces.items()},
        "annotator_disagreement": disagreement,
        "pathology_warnings": [
            f"{partition}:{dimension}:near-constant target"
            for partition, dimensions in partition_stats.items()
            for dimension, stats in dimensions.items()
            if stats["std"] is not None and stats["std"] < 1e-6
        ],
    }


def plot_dataset(
    adapter: RecolaDatasetAdapter,
    rows: list[ManifestRow],
    summary: dict,
    output: Path,
    split: dict | None = None,
) -> list[str]:
    """Plots are private derived data. Requires the optional inspection dependency."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    files = []

    def save(name: str) -> None:
        plt.tight_layout()
        for extension in ("png", "pdf"):
            path = output / f"{name}.{extension}"
            plt.savefig(path, dpi=300)
            files.append(path.name)
        plt.close()

    participants = sorted(summary["participants"])
    for metric, name in (("annotated_duration_seconds", "participant_annotation_duration"),):
        plt.figure(figsize=(8, 4))
        plt.bar(participants, [summary["participants"][p][metric] for p in participants])
        plt.ylabel("Annotation overlap span (seconds)")
        plt.xticks(rotation=90)
        save(name)
    for dimension in ("valence", "arousal"):
        traces = [
            (key, trace)
            for key, trace in sorted(adapter.traces.items())
            if f":{dimension}:consensus:" in key
        ]
        values = [value for _, trace in traces for value in trace.values if value is not None]
        plt.figure(figsize=(7, 4))
        plt.hist(values, bins=30)
        plt.xlabel(f"{dimension.capitalize()} (documented source scale)")
        plt.ylabel("Valid consensus observations")
        save(f"{dimension}_distribution")
        if traces:
            key, trace = traces[0]  # Lexicographically first available consensus video.
            plt.figure(figsize=(8, 3))
            plt.plot(
                trace.timestamps, [np.nan if value is None else value for value in trace.values]
            )
            plt.xlabel("Source time (seconds)")
            plt.ylabel(dimension.capitalize())
            plt.title(key)
            save(f"example_{dimension}_timeline")
        for field, name in (
            ("valid_count", "valid_annotation_count"),
            ("missing_fraction", "missing_annotation_rate"),
        ):
            plt.figure(figsize=(8, 4))
            plt.bar(
                participants,
                [
                    summary["participants"][p]["targets"][dimension][field]
                    if summary["participants"][p]["targets"][dimension][field] is not None
                    else np.nan
                    for p in participants
                ],
            )
            plt.ylabel(field.replace("_", " "))
            plt.xticks(rotation=90)
            save(f"{dimension}_{name}")
        if split:
            plt.figure(figsize=(7, 4))
            for partition in ("train", "validation", "test"):
                videos = {
                    record.video_id
                    for record in split_records(split)
                    if record.partition == partition
                }
                selected = [
                    value
                    for key, trace in traces
                    if any(key.startswith(video + ":") for video in videos)
                    for value in trace.values
                    if value is not None
                ]
                plt.hist(
                    selected, bins=30, histtype="step", label=partition, density=bool(selected)
                )
            plt.xlabel(dimension.capitalize())
            plt.ylabel("Density")
            plt.legend()
            save(f"{dimension}_partition_distributions")
    return files
