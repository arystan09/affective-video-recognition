"""Immutable validated records; missing values are never neutral labels."""

import math
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Partition = Literal["train", "validation", "test"]
Identifier = Annotated[str, Field(min_length=1)]
Timestamp = Annotated[float, Field(ge=0, allow_inf_nan=False)]


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def validate_target(value: float | None, valid: bool, name: str) -> None:
    if value is not None and (not math.isfinite(value) or not -1 <= value <= 1):
        raise ValueError(f"{name} must be finite and in [-1, 1], or None")
    if valid and value is None:
        raise ValueError(f"valid {name} requires a value")


class ParticipantRecord(Record):
    """One participant/video association, not necessarily one row per person."""

    participant_id: Identifier
    video_id: Identifier
    session_id: Identifier | None = None
    dyad_id: Identifier | None = None
    partition: Partition
    source_path: str | None = None
    source_checksum: Identifier | None = None


class TimestampedObservation(Record):
    participant_id: Identifier
    video_id: Identifier
    timestamp: Timestamp
    valence: float | None = None
    arousal: float | None = None
    valence_valid: bool = False
    arousal_valid: bool = False
    face_valid: bool = False
    track_id: Identifier | None = None

    @model_validator(mode="after")
    def check_targets(self) -> "TimestampedObservation":
        validate_target(self.valence, self.valence_valid, "valence")
        validate_target(self.arousal, self.arousal_valid, "arousal")
        return self


class SequenceExample(Record):
    """Unpadded causal context; all tuple positions refer to the same sample grid."""

    participant_id: Identifier
    video_id: Identifier
    timestamps: tuple[Timestamp, ...]
    observation_mask: tuple[bool, ...]
    track_id: Identifier | None = None
    target_timestamp: Timestamp
    target_valence: float | None = None
    target_arousal: float | None = None
    target_valid_mask: tuple[bool, bool]

    @model_validator(mode="after")
    def check_context(self) -> "SequenceExample":
        if not self.timestamps or len(self.timestamps) != len(self.observation_mask):
            raise ValueError("timestamps and observation_mask must be nonempty and equal length")
        if any(b <= a for a, b in zip(self.timestamps, self.timestamps[1:], strict=False)):
            raise ValueError("timestamps must be strictly increasing")
        if self.timestamps[-1] != self.target_timestamp:
            raise ValueError("endpoint timestamp must equal target_timestamp")
        validate_target(self.target_valence, self.target_valid_mask[0], "valence")
        validate_target(self.target_arousal, self.target_valid_mask[1], "arousal")
        return self


class PredictionRecord(Record):
    participant_id: Identifier
    video_id: Identifier
    timestamp: Timestamp
    target_valence: float | None = None
    target_arousal: float | None = None
    target_valid_mask: tuple[bool, bool] = (True, True)
    predicted_valence: float | None = None
    predicted_arousal: float | None = None
    valid: bool = True
    uncertainty_valence: Annotated[float, Field(ge=0, allow_inf_nan=False)] | None = None
    uncertainty_arousal: Annotated[float, Field(ge=0, allow_inf_nan=False)] | None = None

    @model_validator(mode="after")
    def check_values(self) -> "PredictionRecord":
        for index, dimension in enumerate(("valence", "arousal")):
            validate_target(
                getattr(self, f"target_{dimension}"), self.target_valid_mask[index], dimension
            )
            validate_target(getattr(self, f"predicted_{dimension}"), self.valid, dimension)
        return self
