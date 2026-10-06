"""Explicit release mapping. No RECOLA filenames, scale, cadence, or units assumed."""

from typing import Literal

from pydantic import Field, model_validator

from affective_video.data.manifest import Record


class TableSpec(Record):
    path: str = Field(min_length=1)
    format: Literal["csv", "arff"]
    timestamp_column: str = Field(min_length=1)
    timestamp_unit: Literal["seconds", "milliseconds"]
    delimiter: str = Field(default=",", min_length=1, max_length=1)
    encoding: str = "utf-8-sig"
    missing_tokens: tuple[str, ...] = ("", "?", "NA", "NaN", "nan")


class AnnotationSpec(TableSpec):
    dimension: Literal["valence", "arousal"]
    value_column: str = Field(min_length=1)
    kind: Literal["consensus", "individual"]
    annotator_id: str | None = None
    scale_min: float | None = Field(default=None, allow_inf_nan=False)
    scale_max: float | None = Field(default=None, allow_inf_nan=False)
    scale_evidence: str | None = None

    @model_validator(mode="after")
    def check_annotation(self) -> "AnnotationSpec":
        if self.kind == "individual" and not self.annotator_id:
            raise ValueError("individual traces require an explicit annotator_id")
        if (self.scale_min is None) != (self.scale_max is None):
            raise ValueError("scale_min and scale_max must both be supplied or omitted")
        if self.scale_min is not None and self.scale_max is not None:
            if self.scale_min >= self.scale_max:
                raise ValueError("scale_min must be less than scale_max")
        return self


class RecordingSpec(Record):
    participant_id: str = Field(min_length=1)
    video_id: str = Field(min_length=1)
    video_path: str = Field(min_length=1)
    session_id: str | None = Field(default=None, min_length=1)
    dyad_id: str | None = Field(default=None, min_length=1)
    original_release_partition: str | None = None
    timestamps: TableSpec | None = None
    annotations: tuple[AnnotationSpec, ...] = ()


class ReleaseMapping(Record):
    schema_version: Literal[1] = 1
    evidence_files: tuple[str, ...] = ()
    minimum_annotation_seconds: float = Field(default=1.0, gt=0, allow_inf_nan=False)
    recordings: tuple[RecordingSpec, ...] = ()


class ManifestRow(Record):
    participant_id: str
    video_id: str
    session_id: str | None
    dyad_id: str | None
    original_release_partition: str | None
    video_path: str
    annotation_paths: tuple[str, ...]
    timestamp_path: str | None
    duration_seconds: float | None
    frame_count: int | None
    annotation_start: float | None
    annotation_end: float | None
    annotation_frequency: dict[str, float | None]
    number_of_annotators: int | None
    valence_available: bool
    arousal_available: bool
    valid_counts: dict[str, int]
    annotated_duration_seconds: float
    scale_verified: bool
    file_checksums: dict[str, str]
    eligible: bool
    validation_status: Literal["eligible", "excluded"]
    exclusion_reason: tuple[str, ...]


class InspectionIssue(Record):
    severity: Literal["error", "warning"]
    code: str
    path: str | None = None
    detail: str
