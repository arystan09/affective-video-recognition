import json
import runpy
import shutil
import subprocess
from pathlib import Path

import pytest

from affective_video.config import save_config
from affective_video.utils.run_metadata import canonical_hash, create_run, save_json, split_hash


@pytest.mark.skipif(shutil.which("git") is None, reason="Git executable not installed")
def test_git_provenance_and_dirty_state(config, manifest, tmp_path):
    repository = tmp_path / "repository"
    repository.mkdir()

    def git(*args):
        return subprocess.run(
            ["git", "-C", str(repository), *args],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()

    git("init")
    (repository / "sample.txt").write_text("fixture", encoding="utf-8")
    git("add", "sample.txt")
    git(
        "-c",
        "user.name=Test Fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "-c",
        "commit.gpgsign=false",
        "commit",
        "-m",
        "fixture",
    )
    metadata = create_run(config, manifest, repository)[1]
    assert metadata.git_commit == git("rev-parse", "HEAD")
    assert metadata.git_dirty is False
    (repository / "sample.txt").write_text("changed", encoding="utf-8")
    assert create_run(config, manifest, repository)[1].git_dirty is True


def test_run_artifacts(config, manifest, tmp_path):
    directory, metadata = create_run(config, manifest, tmp_path)
    save_json(directory / "metrics.json", {"ccc": None, "infrastructure_only": True})
    assert directory.is_dir()
    for filename in ("config.yaml", "metadata.json", "metrics.json", "split_manifest.json"):
        assert (directory / filename).is_file()
    saved = json.loads((directory / "metadata.json").read_text())
    assert saved["seed"] == config.training.seed
    assert saved["config_hash"] == canonical_hash(config.model_dump(mode="json"))
    assert saved["git_commit"] is None
    assert saved["git_dirty"] is None
    assert saved["environment"]["python"].startswith("3.11")
    assert metadata.split_hash == split_hash(manifest)
    assert create_run(config, manifest, tmp_path)[0] != directory


def test_hash_stability(manifest):
    assert canonical_hash({"b": 1, "a": 2}) == canonical_hash({"a": 2, "b": 1})
    assert split_hash(manifest) == split_hash(list(reversed(manifest)))
    changed = [manifest[0].model_copy(update={"partition": "test"}), *manifest[1:]]
    assert split_hash(manifest) != split_hash(changed)


def test_json_refuses_nan(tmp_path):
    with pytest.raises(ValueError):
        save_json(tmp_path / "bad.json", {"value": float("nan")})


@pytest.mark.parametrize("irregular", [False, True])
def test_synthetic_end_to_end(config, tmp_path, capsys, irregular):
    path = tmp_path / "config.yaml"
    save_config(config, path)
    script = Path(__file__).parents[1] / "scripts/synthetic_run.py"
    driver = runpy.run_path(str(script))
    driver["run"](path, irregular=irregular)
    output = json.loads(capsys.readouterr().out)
    assert output["infrastructure_only"] is True
    assert output["participants"] == 6
    assert output["sequences"] == 96
    directory = Path(output["output_dir"])
    metrics = json.loads((directory / "metrics.json").read_text())
    assert set(metrics["partitions"]) == {"train", "validation", "test"}
    assert (directory / "config.yaml").exists()
    assert (directory / "metadata.json").exists()
