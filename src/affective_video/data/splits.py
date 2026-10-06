"""Fail closed on identity, recording, and session leakage."""

from collections.abc import Sequence

from affective_video.data.manifest import ParticipantRecord


class SplitLeakageError(ValueError):
    """A scientific grouping identifier appears in multiple partitions."""


def validate_splits(records: Sequence[ParticipantRecord]) -> None:
    """Check all available identifiers and report every crossing in one exception."""
    if not records:
        raise ValueError("split manifest must not be empty")
    leaks = []
    for field in ("participant_id", "video_id", "session_id", "dyad_id", "source_checksum"):
        seen: dict[str, set[str]] = {}
        for record in records:
            value = getattr(record, field)
            if value is not None:
                seen.setdefault(value, set()).add(record.partition)
        for value, partitions in sorted(seen.items()):
            if len(partitions) > 1:
                leaks.append(
                    f'{field} "{value}" occurs in partitions {", ".join(sorted(partitions))}'
                )
    if leaks:
        raise SplitLeakageError("Split leakage detected:\n" + "\n".join(leaks))
