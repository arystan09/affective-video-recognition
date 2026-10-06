from pathlib import Path

import pytest
from pydantic import ValidationError

from affective_video.config import ProjectConfig, load_config, save_config


def test_valid_config_loads():
    config = load_config(Path(__file__).parents[1] / "configs/data.yaml")
    assert config.data.sample_rate_hz == 5
    assert config.training.seed == 42
    assert config.experiment.output_dir.is_absolute()


@pytest.mark.parametrize(
    "section,field,value",
    [
        ("model", "dropout", -0.1),
        ("model", "dropout", 1.1),
        ("model", "hidden_size", 0),
        ("data", "sample_rate_hz", 0),
        ("data", "sample_rate_hz", float("nan")),
        ("data", "sequence_seconds", -1),
        ("data", "sequence_seconds", 0.01),
        ("data", "stride", 0),
        ("data", "target_dimensions", ["anger"]),
        ("data", "target_dimensions", ["valence", "valence"]),
        ("data", "target_dimensions", []),
        ("training", "batch_size", 0),
        ("training", "seed", True),
        ("model", "unknown", 1),
    ],
)
def test_invalid_config(config, section, field, value):
    raw = config.model_dump()
    raw[section][field] = value
    with pytest.raises(ValidationError):
        ProjectConfig.model_validate(raw)


def test_resolved_config_roundtrip(config, tmp_path):
    path = tmp_path / "resolved.yaml"
    save_config(config, path)
    assert load_config(path) == config


def test_paths_resolve_from_yaml(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        "data:\n  dataset_name: synthetic\n  manifest_path: manifest.json\n"
        "experiment:\n  name: test\n  output_dir: runs\n",
        encoding="utf-8",
    )
    assert load_config(path).data.manifest_path == tmp_path / "manifest.json"


def test_nonmapping_yaml_rejected(tmp_path):
    path = tmp_path / "invalid.yaml"
    path.write_text("- not a mapping\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config(path)
