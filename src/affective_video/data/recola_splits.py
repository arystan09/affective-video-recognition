"""Frozen, label-independent connected-component splits using existing M1 checks."""

import random
from datetime import UTC, datetime
from pathlib import Path

import yaml

from affective_video.data.adapters.recola_schema import ManifestRow
from affective_video.data.manifest import ParticipantRecord, Partition
from affective_video.data.splits import validate_splits
from affective_video.utils.run_metadata import canonical_hash, split_hash

ALGORITHM = "connected-components-participant-balanced-v1"
PARTITIONS: tuple[Partition, ...] = ("train", "validation", "test")


def manifest_hash(rows: list[ManifestRow]) -> str:
    return canonical_hash(
        [
            row.model_dump(mode="json")
            for row in sorted(rows, key=lambda row: (row.participant_id, row.video_id))
        ]
    )


def generate_internal_split(rows: list[ManifestRow], seed: int = 42) -> dict:
    eligible = sorted((row for row in rows if row.eligible), key=lambda row: row.video_id)
    if len({row.video_id for row in eligible}) != len(eligible):
        raise ValueError("duplicate eligible video identifiers")
    parent = list(range(len(eligible)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    seen: dict[tuple[str, str], int] = {}
    for index, row in enumerate(eligible):
        fields = {
            "participant": row.participant_id,
            "video": row.video_id,
            "session": row.session_id,
            "dyad": row.dyad_id,
            "checksum": row.file_checksums.get(row.video_path),
        }
        if not fields["checksum"]:
            raise ValueError("eligible recordings require source-video checksums")
        for field, value in fields.items():
            if value is None:
                continue
            key = (field, value)
            if key in seen:
                parent[find(index)] = find(seen[key])
            seen[key] = index
    components: dict[int, list[ManifestRow]] = {}
    for index, row in enumerate(eligible):
        components.setdefault(find(index), []).append(row)
    groups = sorted(components.values(), key=lambda group: min(row.video_id for row in group))
    if len(groups) < 3:
        raise ValueError("at least three independent interaction components are required")
    random.Random(seed).shuffle(groups)
    groups.sort(key=lambda group: -len({row.participant_id for row in group}))
    total = len({row.participant_id for row in eligible})
    desired = [total * fraction for fraction in (0.6, 0.2, 0.2)]
    counts = [0, 0, 0]
    records = []
    for index, group in enumerate(groups):
        size = len({row.participant_id for row in group})
        if index < 3:
            destination = index  # Guarantee nonempty partitions; largest group goes to train.
        else:
            destination = min(
                range(3),
                key=lambda candidate: sum(
                    (counts[position] + (size if position == candidate else 0) - desired[position])
                    ** 2
                    for position in range(3)
                ),
            )
        counts[destination] += size
        for row in group:
            records.append(
                ParticipantRecord(
                    participant_id=row.participant_id,
                    video_id=row.video_id,
                    session_id=row.session_id,
                    dyad_id=row.dyad_id,
                    partition=PARTITIONS[destination],
                    source_path=row.video_path,
                    source_checksum=row.file_checksums[row.video_path],
                )
            )
    records.sort(key=lambda record: (record.partition, record.participant_id, record.video_id))
    validate_splits(records)
    return {
        "name": "recola_internal_v1",
        "seed": seed,
        "algorithm": ALGORITHM,
        "grouping": "connected_components_of_participant_video_session_dyad_checksum",
        "manifest_hash": manifest_hash(rows),
        "split_hash": split_hash(records),
        "created_at": datetime.now(UTC).isoformat(),
        "component_count": len(groups),
        "missing_session_records": sum(row.session_id is None for row in eligible),
        "missing_dyad_records": sum(row.dyad_id is None for row in eligible),
        "counts": {
            partition: {
                "participants": len(
                    {record.participant_id for record in records if record.partition == partition}
                ),
                "recordings": sum(record.partition == partition for record in records),
            }
            for partition in PARTITIONS
        },
        "partitions": {
            partition: [
                record.model_dump(mode="json")
                for record in records
                if record.partition == partition
            ]
            for partition in PARTITIONS
        },
    }


def split_records(split: dict) -> list[ParticipantRecord]:
    records = [
        ParticipantRecord.model_validate(record)
        for partition in PARTITIONS
        for record in split["partitions"][partition]
    ]
    for partition in PARTITIONS:
        if not split["partitions"][partition] or any(
            record["partition"] != partition for record in split["partitions"][partition]
        ):
            raise ValueError("split partitions must be nonempty and match record assignments")
    validate_splits(records)
    if split_hash(records) != split["split_hash"]:
        raise ValueError("split hash mismatch: assignments were changed")
    return records


def freeze_split(path: Path, rows: list[ManifestRow], seed: int = 42) -> dict:
    """Never overwrite an existing split; reject changes to data, seed, or assignments."""
    if path.exists():
        existing = yaml.safe_load(path.read_text(encoding="utf-8"))
        split_records(existing)
        if existing["manifest_hash"] != manifest_hash(rows) or existing["seed"] != seed:
            raise ValueError(
                "frozen split conflicts with current manifest/seed; create a new version"
            )
        if existing["algorithm"] != ALGORITHM:
            raise ValueError("frozen split uses an unsupported algorithm version")
        expected = generate_internal_split(rows, seed)
        if existing["split_hash"] != expected["split_hash"]:
            raise ValueError("frozen split differs from its deterministic generation rule")
        if existing["counts"] != expected["counts"]:
            raise ValueError("frozen split counts do not match its assignments")
        return existing
    result = generate_internal_split(rows, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        yaml.safe_dump(result, handle, sort_keys=True)
    return result


def audit_split(split: dict, rows: list[ManifestRow]) -> dict:
    records = split_records(split)
    if split["manifest_hash"] != manifest_hash(rows):
        raise ValueError("split refers to a different manifest")
    by_video = {row.video_id: row for row in rows}
    if {record.video_id for record in records} != {row.video_id for row in rows if row.eligible}:
        raise ValueError("split does not contain exactly the eligible manifest recordings")
    for record in records:
        row = by_video[record.video_id]
        if (record.participant_id, record.session_id, record.dyad_id, record.source_checksum) != (
            row.participant_id,
            row.session_id,
            row.dyad_id,
            row.file_checksums.get(row.video_path),
        ):
            raise ValueError("split identifiers/checksums do not match manifest")
    result: dict = {"passed": True, "partitions": {}, "intersections": {}}
    for partition in PARTITIONS:
        selected = [record for record in records if record.partition == partition]
        result["partitions"][partition] = {
            "participants": sorted({record.participant_id for record in selected}),
            "videos": sorted(record.video_id for record in selected),
            "sessions": sorted({record.session_id for record in selected if record.session_id}),
            "dyads": sorted({record.dyad_id for record in selected if record.dyad_id}),
            "participant_count": len({record.participant_id for record in selected}),
            "recording_count": len(selected),
            "annotated_duration_seconds": sum(
                by_video[record.video_id].annotated_duration_seconds for record in selected
            ),
            "valid_annotation_counts": {
                dimension: sum(
                    by_video[record.video_id].valid_counts[dimension] for record in selected
                )
                for dimension in ("valence", "arousal")
            },
        }
    for field in ("participant_id", "video_id", "session_id", "dyad_id", "source_checksum"):
        sets = [
            {
                getattr(record, field)
                for record in records
                if record.partition == partition and getattr(record, field) is not None
            }
            for partition in PARTITIONS
        ]
        result["intersections"][field] = sorted(
            (sets[0] & sets[1]) | (sets[0] & sets[2]) | (sets[1] & sets[2])
        )
    return result
