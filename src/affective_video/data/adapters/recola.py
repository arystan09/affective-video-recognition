"""Inspect an explicitly mapped local release, without decoding or target resampling."""

from collections import Counter
from pathlib import Path

from affective_video.data.adapters.recola_provenance import ChecksumCache, local_source
from affective_video.data.adapters.recola_schema import (
    AnnotationSpec,
    InspectionIssue,
    ManifestRow,
    ReleaseMapping,
)
from affective_video.data.adapters.recola_tables import ParsedTrace, parse_trace


class RecolaDatasetAdapter:
    """File roles and identities come only from a documented operator mapping."""

    def __init__(
        self, root: Path, mapping: ReleaseMapping | None = None, cache: ChecksumCache | None = None
    ):
        self.root = root.resolve()
        self.mapping = mapping
        self.cache = cache or ChecksumCache(self.root)
        self.issues: list[InspectionIssue] = []
        self.traces: dict[str, ParsedTrace] = {}
        self.timestamp_traces: dict[str, ParsedTrace] = {}
        self.manifest_rows: list[ManifestRow] = []
        self.inventory: list[dict] = []

    def discover(self) -> list[dict]:
        if not self.root.is_dir():
            raise FileNotFoundError("local release root does not exist or is not a directory")
        roles: dict[str, set[str]] = {}
        if self.mapping:
            for recording in self.mapping.recordings:
                roles.setdefault(recording.video_path, set()).add("video")
                if recording.timestamps:
                    roles.setdefault(recording.timestamps.path, set()).add("frame_timestamps")
                for annotation in recording.annotations:
                    roles.setdefault(annotation.path, set()).add("annotation")
                    if annotation.scale_evidence:
                        roles.setdefault(annotation.scale_evidence, set()).add("scale_evidence")
            for evidence_path in self.mapping.evidence_files:
                roles.setdefault(evidence_path, set()).add("release_evidence")
        self.inventory = []
        for path in sorted(self.root.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(self.root).as_posix()
            local_source(self.root, relative)  # Reject escaping symlinks.
            self.inventory.append(
                {
                    "path": relative,
                    "bytes": path.stat().st_size,
                    "roles": sorted(roles.get(relative, set())),
                    "candidate_type": (
                        "video"
                        if path.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv", ".mpg")
                        else "table"
                        if path.suffix.lower() in (".csv", ".arff", ".tsv")
                        else "documentation"
                        if path.suffix.lower() in (".md", ".txt", ".pdf")
                        else "unknown"
                    ),
                }
            )
        return self.inventory

    def _issue(
        self, code: str, detail: str, path: str | None = None, severity: str = "error"
    ) -> None:
        self.issues.append(
            InspectionIssue.model_validate(
                {
                    "code": code,
                    "detail": detail,
                    "path": path,
                    "severity": severity,
                }
            )
        )

    def build_manifest(self) -> list[ManifestRow]:
        self.issues, self.traces, self.timestamp_traces, self.manifest_rows = [], {}, {}, []
        self.discover()
        if self.mapping is None:
            self._issue("mapping_required", "File identities and roles have not been verified")
            return []
        for identifier, count in Counter(item.video_id for item in self.mapping.recordings).items():
            if count > 1:
                self._issue("duplicate_video_id", f"video_id {identifier!r} appears {count} times")
        rows = []
        for recording in sorted(
            self.mapping.recordings, key=lambda item: (item.participant_id, item.video_id)
        ):
            reasons: list[str] = []
            checksums = {}
            paths = [recording.video_path, *self.mapping.evidence_files]
            if recording.timestamps:
                paths.append(recording.timestamps.path)
            paths.extend(item.path for item in recording.annotations)
            paths.extend(
                item.scale_evidence for item in recording.annotations if item.scale_evidence
            )
            for relative in sorted(set(paths)):
                try:
                    source = local_source(self.root, relative)
                    if source.stat().st_size == 0:
                        raise ValueError("empty source file")
                    checksums[relative] = self.cache.checksum(relative)
                except (OSError, ValueError) as error:
                    reasons.append(f"source_unavailable:{relative}")
                    self._issue("source_unavailable", str(error), relative)
            frame_trace = None
            if recording.timestamps is None:
                reasons.append("frame_timestamps_missing")
            elif recording.timestamps.path in checksums:
                try:
                    frame_trace = parse_trace(
                        local_source(self.root, recording.timestamps.path), recording.timestamps
                    )
                    self._audit_trace(frame_trace, recording.timestamps.path, reasons)
                    self.timestamp_traces[recording.video_id] = frame_trace
                except (OSError, ValueError) as error:
                    reasons.append("frame_timestamps_malformed")
                    self._issue("malformed_table", str(error), recording.timestamps.path)
            consensus: dict[str, tuple[AnnotationSpec, ParsedTrace]] = {}
            parsed_annotations = []
            for spec in recording.annotations:
                if spec.path not in checksums:
                    continue
                try:
                    trace = parse_trace(local_source(self.root, spec.path), spec)
                    self._audit_trace(trace, spec.path, reasons)
                    key = f"{recording.video_id}:{spec.dimension}:{spec.kind}:{spec.annotator_id}"
                    if key in self.traces:
                        reasons.append("duplicate_annotation_identity")
                        self._issue("duplicate_annotation_identity", key, spec.path)
                    self.traces[key] = trace
                    parsed_annotations.append((spec, trace))
                    if spec.kind == "consensus":
                        if spec.dimension in consensus:
                            reasons.append("duplicate_consensus")
                            self._issue("duplicate_consensus", spec.dimension, spec.path)
                        consensus[spec.dimension] = (spec, trace)
                    if spec.scale_min is not None and spec.scale_max is not None:
                        if any(
                            value is not None and not spec.scale_min <= value <= spec.scale_max
                            for value in trace.values
                        ):
                            reasons.append("annotation_outside_documented_scale")
                            self._issue("scale_violation", spec.dimension, spec.path)
                except (OSError, ValueError) as error:
                    reasons.append("annotation_malformed")
                    self._issue("malformed_table", str(error), spec.path)
            available = {
                dimension: any(spec.dimension == dimension for spec, _ in parsed_annotations)
                for dimension in ("valence", "arousal")
            }
            for dimension in ("valence", "arousal"):
                if dimension not in consensus:
                    reasons.append(
                        f"{dimension}_consensus_missing:target_construction_not_approved"
                    )
            scale_verified = bool(
                len(consensus) == 2
                and all(
                    spec.scale_min is not None and spec.scale_evidence in checksums
                    for spec, _ in consensus.values()
                )
            )
            if not scale_verified:
                reasons.append("scale_unverified")
            intervals = []
            valid_counts = {dimension: 0 for dimension in ("valence", "arousal")}
            frequencies: dict[str, float | None] = {}
            for dimension, (_, trace) in consensus.items():
                valid_times = [
                    time
                    for time, value in zip(trace.timestamps, trace.values, strict=True)
                    if value is not None
                ]
                valid_counts[dimension] = len(valid_times)
                frequencies[dimension] = trace.timing["frequency_hz"]
                if valid_times:
                    intervals.append((valid_times[0], valid_times[-1]))
            start = max(item[0] for item in intervals) if len(intervals) == 2 else None
            end = min(item[1] for item in intervals) if len(intervals) == 2 else None
            duration = max(0.0, end - start) if start is not None and end is not None else 0.0
            if duration < self.mapping.minimum_annotation_seconds:
                reasons.append("insufficient_joint_annotation_span")
            if frame_trace and start is not None and end is not None:
                if (
                    not frame_trace.timestamps
                    or end < frame_trace.timestamps[0]
                    or (start > frame_trace.timestamps[-1])
                ):
                    reasons.append("annotation_video_coverage_disjoint")
                    duration = 0.0
                else:
                    if start < frame_trace.timestamps[0] or end > frame_trace.timestamps[-1]:
                        self._issue("partial_video_overlap", recording.video_id, severity="warning")
                    duration = max(
                        0.0,
                        min(end, frame_trace.timestamps[-1])
                        - max(start, frame_trace.timestamps[0]),
                    )
                    if duration < self.mapping.minimum_annotation_seconds:
                        reasons.append("insufficient_video_annotation_overlap")
            annotators = {spec.annotator_id for spec, _ in parsed_annotations if spec.annotator_id}
            rows.append(
                ManifestRow(
                    participant_id=recording.participant_id,
                    video_id=recording.video_id,
                    session_id=recording.session_id,
                    dyad_id=recording.dyad_id,
                    original_release_partition=recording.original_release_partition,
                    video_path=recording.video_path,
                    annotation_paths=tuple(sorted({item.path for item in recording.annotations})),
                    timestamp_path=recording.timestamps.path if recording.timestamps else None,
                    duration_seconds=(frame_trace.timestamps[-1] - frame_trace.timestamps[0])
                    if frame_trace and len(frame_trace.timestamps) >= 2
                    else None,
                    frame_count=len(frame_trace.timestamps) if frame_trace else None,
                    annotation_start=start,
                    annotation_end=end,
                    annotation_frequency=frequencies,
                    number_of_annotators=len(annotators) if annotators else None,
                    valence_available=available["valence"],
                    arousal_available=available["arousal"],
                    valid_counts=valid_counts,
                    annotated_duration_seconds=duration,
                    scale_verified=scale_verified,
                    file_checksums=checksums,
                    eligible=not reasons,
                    validation_status="excluded" if reasons else "eligible",
                    exclusion_reason=tuple(sorted(set(reasons))),
                )
            )
        self.cache.save()
        self.manifest_rows = rows
        return rows

    def _audit_trace(self, trace: ParsedTrace, path: str, reasons: list[str]) -> None:
        if (
            len(trace.timestamps) < 2
            or not trace.timing["monotonic"]
            or (trace.timing["negative_count"] > 0)
        ):
            reasons.append(f"invalid_timestamp_integrity:{path}")
            self._issue("timestamp_integrity", str(trace.timing), path)
        if trace.timing["large_gap_count"] or trace.timing["irregular_intervals"]:
            self._issue("irregular_timestamps", str(trace.timing), path, "warning")

    def validate_release(self) -> dict:
        return {
            "recognized_files": [item for item in self.inventory if item["roles"]],
            "unrecognized_files": [item for item in self.inventory if not item["roles"]],
            "issues": [issue.model_dump(mode="json") for issue in self.issues],
            "critical_failure": any(issue.severity == "error" for issue in self.issues),
            "explicit_mapping_supplied": self.mapping is not None,
            "missing_or_invalid_components": [
                {"video_id": row.video_id, "exclusion_reasons": list(row.exclusion_reason)}
                for row in self.manifest_rows
                if row.exclusion_reason
            ],
        }
