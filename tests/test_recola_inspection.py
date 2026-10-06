"""Fabricated file trees exercise format contracts, not an actual RECOLA release."""

import hashlib
import json
import runpy
from pathlib import Path

import pytest
import yaml

from affective_video.data.adapters.recola import RecolaDatasetAdapter
from affective_video.data.adapters.recola_provenance import ChecksumCache, local_source
from affective_video.data.adapters.recola_schema import AnnotationSpec, ReleaseMapping, TableSpec
from affective_video.data.adapters.recola_tables import (
    parse_trace,
    timestamp_audit,
    value_statistics,
)
from affective_video.data.manifest import ParticipantRecord
from affective_video.data.recola_reporting import dataset_summary, plot_dataset, write_manifest
from affective_video.data.recola_splits import (
    audit_split,
    freeze_split,
    generate_internal_split,
    manifest_hash,
    split_records,
)
from affective_video.data.splits import validate_splits


@pytest.fixture
def miniature_release(tmp_path):
    root = tmp_path / "fabricated-release"
    root.mkdir()
    (root / "fixture-notes.txt").write_text(
        "SYNTHETIC ONLY: scale [-2,2], seconds, 4 Hz, dyads pair adjacent IDs.\n",
        encoding="utf-8",
    )
    recordings = []
    for index in range(10):
        video = f"clip-{index}"
        (root / f"{video}.mp4").write_bytes(f"not a real video: {index}".encode())
        (root / f"{video}-time.csv").write_text(
            "frame;elapsed\n" + "".join(f"{i};{i / 4}\n" for i in range(9)),
            encoding="utf-8",
        )
        (root / f"{video}-ratings.csv").write_text(
            "seconds;v;a\n"
            + "".join(
                f"{i / 4};{'?' if i == 4 else (i - 4) / 4};{(4 - i) / 4}\n" for i in range(9)
            ),
            encoding="utf-8",
        )
        recordings.append(
            {
                "participant_id": f"subject-{index}",
                "video_id": video,
                "session_id": f"session-{index // 2}",
                "dyad_id": f"dyad-{index // 2}",
                "video_path": f"{video}.mp4",
                "original_release_partition": "fixture-only",
                "timestamps": {
                    "path": f"{video}-time.csv",
                    "format": "csv",
                    "delimiter": ";",
                    "timestamp_column": "elapsed",
                    "timestamp_unit": "seconds",
                },
                "annotations": [
                    {
                        "path": f"{video}-ratings.csv",
                        "format": "csv",
                        "delimiter": ";",
                        "timestamp_column": "seconds",
                        "timestamp_unit": "seconds",
                        "value_column": column,
                        "dimension": dimension,
                        "kind": "consensus",
                        "scale_min": -2,
                        "scale_max": 2,
                        "scale_evidence": "fixture-notes.txt",
                    }
                    for dimension, column in (("valence", "v"), ("arousal", "a"))
                ],
            }
        )
    mapping = ReleaseMapping.model_validate(
        {
            "evidence_files": ["fixture-notes.txt"],
            "recordings": recordings,
        }
    )
    return root, mapping


def test_discovery_and_manifest(miniature_release):
    root, mapping = miniature_release
    adapter = RecolaDatasetAdapter(root, mapping)
    rows = adapter.build_manifest()
    assert len(rows) == 10 and all(row.eligible for row in rows)
    assert all(row.annotation_frequency == {"valence": 4.0, "arousal": 4.0} for row in rows)
    assert rows[0].frame_count == 9
    assert rows[0].valid_counts == {"valence": 8, "arousal": 9}
    assert rows[0].number_of_annotators is None  # Consensus alone does not reveal rater count.
    assert not adapter.validate_release()["critical_failure"]
    assert len(adapter.validate_release()["recognized_files"]) == 31


def test_inventory_without_mapping_does_not_invent_ids(miniature_release):
    root, _ = miniature_release
    adapter = RecolaDatasetAdapter(root)
    assert adapter.build_manifest() == []
    assert len(adapter.validate_release()["unrecognized_files"]) == 31
    assert adapter.validate_release()["critical_failure"]


def test_unknown_file_reported(miniature_release):
    root, mapping = miniature_release
    (root / "mystery.bin").write_bytes(b"unknown")
    adapter = RecolaDatasetAdapter(root, mapping)
    adapter.build_manifest()
    assert adapter.validate_release()["unrecognized_files"][0]["path"] == "mystery.bin"
    assert not adapter.validate_release()["critical_failure"]


def test_missing_component_excluded(miniature_release):
    root, mapping = miniature_release
    (root / "clip-0-time.csv").unlink()
    adapter = RecolaDatasetAdapter(root, mapping)
    rows = adapter.build_manifest()
    assert not rows[0].eligible
    assert any("source_unavailable" in reason for reason in rows[0].exclusion_reason)
    assert sum(row.eligible for row in rows) == 9


def test_duplicate_identifier(miniature_release):
    root, mapping = miniature_release
    mapping = mapping.model_copy(
        update={"recordings": (*mapping.recordings, mapping.recordings[0])}
    )
    adapter = RecolaDatasetAdapter(root, mapping)
    adapter.build_manifest()
    assert any(issue.code == "duplicate_video_id" for issue in adapter.issues)


def test_missing_consensus_not_averaged(miniature_release):
    root, mapping = miniature_release
    raw = mapping.model_dump()
    for spec in raw["recordings"][0]["annotations"]:
        spec.update(kind="individual", annotator_id="rater-1")
    adapter = RecolaDatasetAdapter(root, ReleaseMapping.model_validate(raw))
    row = adapter.build_manifest()[0]
    assert not row.eligible and row.number_of_annotators == 1
    assert row.valence_available
    assert any("target_construction_not_approved" in reason for reason in row.exclusion_reason)


def test_scale_requires_evidence_and_bounds(miniature_release):
    root, mapping = miniature_release
    raw = mapping.model_dump()
    for spec in raw["recordings"][0]["annotations"]:
        spec.update(scale_min=None, scale_max=None, scale_evidence=None)
    row = RecolaDatasetAdapter(root, ReleaseMapping.model_validate(raw)).build_manifest()[0]
    assert not row.scale_verified and "scale_unverified" in row.exclusion_reason


def test_malformed_annotation_and_scale_violation(miniature_release):
    root, mapping = miniature_release
    path = root / "clip-0-ratings.csv"
    path.write_text("seconds;v;a\n0;3;0\n1;0;0\n", encoding="utf-8")
    adapter = RecolaDatasetAdapter(root, mapping)
    assert "annotation_outside_documented_scale" in adapter.build_manifest()[0].exclusion_reason
    path.write_text("seconds;v;a\n0;broken;0\n", encoding="utf-8")
    assert "annotation_malformed" in adapter.build_manifest()[0].exclusion_reason


@pytest.mark.parametrize(
    "times,field",
    [
        ((0.0, 0.1, 0.1), "duplicate_count"),
        ((0.2, 0.1), "nonincreasing_intervals"),
        ((-0.1, 0.1), "negative_count"),
        ((0.0, float("nan")), "nonfinite_count"),
        ((0.0, 0.1, 0.2, 4.0), "large_gap_count"),
    ],
)
def test_timestamp_diagnostics(times, field):
    assert timestamp_audit(times)[field] > 0


def test_disjoint_timestamp_coverage(miniature_release):
    root, mapping = miniature_release
    (root / "clip-0-time.csv").write_text("frame;elapsed\n0;20\n1;21\n", encoding="utf-8")
    row = RecolaDatasetAdapter(root, mapping).build_manifest()[0]
    assert "annotation_video_coverage_disjoint" in row.exclusion_reason


def test_dense_arff_milliseconds_and_missing(tmp_path):
    path = tmp_path / "ratings.arff"
    path.write_text(
        "@relation fixture\n@attribute elapsed numeric\n@attribute rating real\n"
        "@data\n0,-1\n250,?\n500,1\n",
        encoding="utf-8",
    )
    spec = AnnotationSpec(
        path="ratings.arff",
        format="arff",
        timestamp_column="elapsed",
        timestamp_unit="milliseconds",
        dimension="valence",
        value_column="rating",
        kind="consensus",
    )
    trace = parse_trace(path, spec)
    assert trace.timestamps == (0, 0.25, 0.5)
    assert trace.values == (-1, None, 1)
    assert trace.statistics["missing_fraction"] == pytest.approx(1 / 3)


@pytest.mark.parametrize(
    "content",
    [
        "",
        "@relation fixture\n",
        "@attribute t numeric\n@data\n{0 1}\n",
        "@attribute t string\n@data\n0\n",
    ],
)
def test_invalid_arff_rejected(tmp_path, content):
    path = tmp_path / "invalid.arff"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError):
        parse_trace(
            path,
            TableSpec(
                path=path.name, format="arff", timestamp_column="t", timestamp_unit="seconds"
            ),
        )


def test_stable_manifest_and_checksums(miniature_release, tmp_path):
    root, mapping = miniature_release
    rows = RecolaDatasetAdapter(root, mapping).build_manifest()
    reversed_mapping = mapping.model_copy(
        update={"recordings": tuple(reversed(mapping.recordings))}
    )
    reverse = RecolaDatasetAdapter(root, reversed_mapping).build_manifest()
    assert rows == reverse
    assert manifest_hash(rows) == manifest_hash(reverse)
    path = tmp_path / "manifest.csv"
    write_manifest(path, rows)
    first = path.read_bytes()
    write_manifest(path, list(reversed(rows)))
    assert path.read_bytes() == first
    assert str(root).encode() not in first
    assert (
        rows[0].file_checksums["clip-0.mp4"]
        == hashlib.sha256((root / "clip-0.mp4").read_bytes()).hexdigest()
    )


def test_checksum_cache_invalidation(miniature_release, tmp_path):
    root, _ = miniature_release
    path = tmp_path / "checksums.json"
    cache = ChecksumCache(root, path)
    before = cache.checksum("clip-0.mp4")
    cache.save()
    assert ChecksumCache(root, path).checksum("clip-0.mp4") == before
    (root / "clip-0.mp4").write_bytes(b"changed source bytes")
    assert ChecksumCache(root, path).checksum("clip-0.mp4") != before


def test_source_containment(tmp_path):
    with pytest.raises(ValueError):
        local_source(tmp_path, "../outside.csv")
    with pytest.raises(ValueError):
        local_source(tmp_path, str(tmp_path / "absolute.csv"))


def test_grouped_split_is_deterministic_and_leakage_safe(miniature_release):
    root, mapping = miniature_release
    rows = RecolaDatasetAdapter(root, mapping).build_manifest()
    one, two = generate_internal_split(rows), generate_internal_split(list(reversed(rows)))
    assert one["split_hash"] == two["split_hash"]
    assert one["partitions"] == two["partitions"]
    records = split_records(one)
    validate_splits(records)
    for field in ("participant_id", "video_id", "session_id", "dyad_id", "source_checksum"):
        destinations = {}
        for record in records:
            destinations.setdefault(getattr(record, field), set()).add(record.partition)
        assert all(len(partitions) == 1 for partitions in destinations.values())
    audit = audit_split(one, rows)
    assert audit["passed"] and all(not values for values in audit["intersections"].values())
    assert [
        audit["partitions"][p]["participant_count"] for p in ("train", "validation", "test")
    ] == [6, 2, 2]


def test_duplicate_source_recordings_stay_grouped(miniature_release):
    root, mapping = miniature_release
    (root / "clip-8.mp4").write_bytes((root / "clip-0.mp4").read_bytes())
    rows = RecolaDatasetAdapter(root, mapping).build_manifest()
    records = split_records(generate_internal_split(rows))
    assert next(r.partition for r in records if r.video_id == "clip-8") == next(
        r.partition for r in records if r.video_id == "clip-0"
    )
    validate_splits(records)


def test_transitive_sessions_and_dyads(miniature_release):
    root, mapping = miniature_release
    raw = mapping.model_dump()
    raw["recordings"][2]["session_id"] = raw["recordings"][0]["session_id"]
    rows = RecolaDatasetAdapter(root, ReleaseMapping.model_validate(raw)).build_manifest()
    records = split_records(generate_internal_split(rows))
    assert (
        len(
            {r.partition for r in records if r.video_id in ("clip-0", "clip-1", "clip-2", "clip-3")}
        )
        == 1
    )


def test_too_few_groups_rejected(miniature_release):
    root, mapping = miniature_release
    rows = RecolaDatasetAdapter(root, mapping).build_manifest()[:4]
    with pytest.raises(ValueError, match="three independent"):
        generate_internal_split(rows)


def test_frozen_split_never_overwritten(miniature_release, tmp_path):
    root, mapping = miniature_release
    rows = RecolaDatasetAdapter(root, mapping).build_manifest()
    path = tmp_path / "frozen.yaml"
    first = freeze_split(path, rows)
    content = path.read_bytes()
    assert freeze_split(path, rows) == first and path.read_bytes() == content
    with pytest.raises(ValueError, match="frozen split conflicts"):
        freeze_split(path, rows, seed=43)
    with pytest.raises(ValueError, match="frozen split conflicts"):
        freeze_split(path, rows[:-1])


def test_split_tampering_detected(miniature_release):
    root, mapping = miniature_release
    split = generate_internal_split(RecolaDatasetAdapter(root, mapping).build_manifest())
    split["partitions"]["train"][0]["source_checksum"] = "tampered"
    with pytest.raises(ValueError, match="hash mismatch"):
        split_records(split)


def test_reporting_statistics_and_empty_edges(miniature_release):
    root, mapping = miniature_release
    adapter = RecolaDatasetAdapter(root, mapping)
    rows = adapter.build_manifest()
    summary = dataset_summary(adapter, rows, generate_internal_split(rows))
    assert summary["eligible_participants"] == 10
    assert summary["eligible_annotated_duration_seconds"] == 20
    assert summary["targets"]["valence"]["valid_count"] == 80
    assert summary["targets"]["valence"]["missing_fraction"] == pytest.approx(1 / 9)
    assert summary["targets"]["arousal"]["min"] == -1
    assert summary["targets"]["arousal"]["max"] == 1
    assert value_statistics(())["mean"] is None
    assert value_statistics((None,))["missing_fraction"] == 1
    assert dataset_summary(RecolaDatasetAdapter(root), [])["eligible_recordings"] == 0


def test_plotting_synthetic_only(miniature_release, tmp_path):
    pytest.importorskip("matplotlib")
    root, mapping = miniature_release
    adapter = RecolaDatasetAdapter(root, mapping)
    rows = adapter.build_manifest()
    split = generate_internal_split(rows)
    summary = dataset_summary(adapter, rows, split)
    names = plot_dataset(adapter, rows, summary, tmp_path / "plots", split)
    assert "example_valence_timeline.png" in names
    assert "arousal_partition_distributions.pdf" in names
    assert all((tmp_path / "plots" / name).stat().st_size > 0 for name in names)


def test_cli_blocked_without_data(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "data": {
                    "dataset_name": "recola",
                    "manifest_path": "private/manifest.csv",
                    "split_path": "private/split.yaml",
                },
                "experiment": {"name": "inspection"},
            }
        ),
        encoding="utf-8",
    )
    cli = runpy.run_path(str(Path(__file__).parents[1] / "scripts/inspect_dataset.py"))
    code, result = cli["inspect"](config, None)
    assert code == 2 and result["status"] == "BLOCKED_ON_RECOLA_ACCESS"
    assert result["eligible_participants"] is None
    assert not (tmp_path / "private/manifest.csv").exists()
    assert not (tmp_path / "private/split.yaml").exists()
    assert (
        json.loads((tmp_path / "private/recola_inspection.json").read_text())["status"]
        == result["status"]
    )


def test_cli_mapped_fixture_and_frozen_split(miniature_release, tmp_path):
    root, mapping = miniature_release
    path = tmp_path / "mapping.yaml"
    path.write_text(yaml.safe_dump(mapping.model_dump(mode="json")), encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "data": {
                    "dataset_name": "recola",
                    "manifest_path": "private/manifest.csv",
                    "split_path": "private/split.yaml",
                },
                "experiment": {"name": "inspection"},
            }
        ),
        encoding="utf-8",
    )
    cli = runpy.run_path(str(Path(__file__).parents[1] / "scripts/inspect_dataset.py"))
    code, result = cli["inspect"](config, root, path, make_split=True)
    assert code == 0 and result["status"] == "AUDITED"
    assert (tmp_path / "private/manifest.csv").exists()
    assert (tmp_path / "private/recola_split_audit.md").exists()
    assert cli["inspect"](config, root, path)[1]["manifest_hash"] == result["manifest_hash"]


def test_annotator_disagreement_without_target_construction(miniature_release):
    root, mapping = miniature_release
    raw = mapping.model_dump()
    individual = dict(raw["recordings"][0]["annotations"][0])
    individual.update(kind="individual", annotator_id="rater-a")
    second = dict(individual)
    second["annotator_id"] = "rater-b"
    raw["recordings"][0]["annotations"] = (individual, second)
    adapter = RecolaDatasetAdapter(root, ReleaseMapping.model_validate(raw))
    rows = adapter.build_manifest()
    summary = dataset_summary(adapter, rows)
    stats = summary["annotator_disagreement"]["clip-0:valence"]
    assert stats["individual_trace_count"] == 2
    assert stats["exact_timestamp_multiannotator_count"] == 8
    assert stats["mean_annotator_std"] == 0
    assert not rows[0].eligible


def test_invalid_frame_order_excludes_recording(miniature_release):
    root, mapping = miniature_release
    (root / "clip-0-time.csv").write_text("frame;elapsed\n0;0.5\n1;0.25\n", encoding="utf-8")
    adapter = RecolaDatasetAdapter(root, mapping)
    row = adapter.build_manifest()[0]
    assert not row.eligible
    assert any(issue.code == "timestamp_integrity" for issue in adapter.issues)


def test_empty_video_excluded(miniature_release):
    root, mapping = miniature_release
    (root / "clip-0.mp4").write_bytes(b"")
    row = RecolaDatasetAdapter(root, mapping).build_manifest()[0]
    assert not row.eligible


def test_audit_rejects_manifest_identity_tampering(miniature_release):
    from affective_video.utils.run_metadata import split_hash

    root, mapping = miniature_release
    rows = RecolaDatasetAdapter(root, mapping).build_manifest()
    split = generate_internal_split(rows)
    split["partitions"]["train"][0]["participant_id"] = "replacement"
    records = [
        ParticipantRecord.model_validate(record)
        for partition in ("train", "validation", "test")
        for record in split["partitions"][partition]
    ]
    split["split_hash"] = split_hash(records)
    with pytest.raises(ValueError, match="identifiers/checksums"):
        audit_split(split, rows)


@pytest.mark.parametrize(
    "name",
    [
        "data/manifests/private.csv",
        "data/raw/labels.txt",
        "clip.mp4",
        "reports/figures/dataset/timeline.pdf",
        "ratings.arff",
    ],
)
def test_data_safety_detects_obvious_private_files(tmp_path, name):
    script = Path(__file__).parents[1] / "scripts/check_data_safety.py"
    check = runpy.run_path(str(script))["violations"]
    assert check(tmp_path, [name])


def test_data_safety_permits_public_scaffolding(tmp_path):
    script = Path(__file__).parents[1] / "scripts/check_data_safety.py"
    check = runpy.run_path(str(script))["violations"]
    assert (
        check(
            tmp_path,
            [
                "data/README.md",
                "data/manifests/.gitkeep",
                "docs/schemas/recola_mapping.schema.json",
            ],
        )
        == []
    )
