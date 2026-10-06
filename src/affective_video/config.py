"""Strict YAML configuration; paths resolve relative to the configuration file."""

from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

PositiveInt = Annotated[int, Field(strict=True, gt=0)]
PositiveFloat = Annotated[float, Field(gt=0, allow_inf_nan=False)]
Target = Literal["valence", "arousal"]


class StrictConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DataConfig(StrictConfig):
    dataset_name: str = Field(min_length=1)
    manifest_path: Path | None = None
    split_path: Path | None = None
    sample_rate_hz: PositiveFloat = 5.0
    sequence_seconds: PositiveFloat = 3.0
    stride: PositiveInt = 5
    target_dimensions: tuple[Target, ...] = ("valence", "arousal")

    @model_validator(mode="after")
    def validate_grid(self) -> "DataConfig":
        count = self.sample_rate_hz * self.sequence_seconds
        if count < 1 or abs(count - round(count)) > 1e-8:
            raise ValueError("sample_rate_hz * sequence_seconds must be a positive integer")
        if not self.target_dimensions or len(set(self.target_dimensions)) != len(
            self.target_dimensions
        ):
            raise ValueError("target_dimensions must be nonempty and unique")
        return self


class ModelConfig(StrictConfig):
    type: Literal["synthetic", "mean", "frame_mlp", "pooling", "gru"] = "synthetic"
    hidden_size: PositiveInt = 128
    dropout: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)] = 0.2


class TrainingConfig(StrictConfig):
    seed: Annotated[int, Field(strict=True, ge=0, le=2**32 - 1)] = 42
    batch_size: PositiveInt = 64
    learning_rate: PositiveFloat = 0.001
    epochs: PositiveInt = 60
    weight_decay: Annotated[float, Field(ge=0, allow_inf_nan=False)] = 0.0001
    deterministic: bool = False


class EvaluationConfig(StrictConfig):
    primary_metric: Literal["ccc"] = "ccc"
    participant_macro_average: bool = True


class ExperimentConfig(StrictConfig):
    name: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
    output_dir: Path = Path("../runs")


class ProjectConfig(StrictConfig):
    data: DataConfig
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    evaluation: EvaluationConfig = EvaluationConfig()
    experiment: ExperimentConfig


def load_config(path: str | Path) -> ProjectConfig:
    """Load safe YAML and resolve paths without relying on the caller's cwd."""
    source = Path(path).resolve()
    with source.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    config = ProjectConfig.model_validate(raw)
    paths = {}
    for key in ("manifest_path", "split_path"):
        value = getattr(config.data, key)
        paths[key] = (source.parent / value).resolve() if value is not None else None
    return config.model_copy(
        update={
            "data": config.data.model_copy(update=paths),
            "experiment": config.experiment.model_copy(
                update={"output_dir": (source.parent / config.experiment.output_dir).resolve()}
            ),
        }
    )


def save_config(config: ProjectConfig, path: Path) -> None:
    """Persist every default and resolved path, not just the source YAML."""
    path.write_text(
        yaml.safe_dump(config.model_dump(mode="json"), sort_keys=True), encoding="utf-8"
    )
