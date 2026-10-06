"""Small local run artifacts with canonical hashes and honest Git provenance."""

import hashlib
import json
import platform
import subprocess
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any
from uuid import uuid4

import torch
from pydantic import Field

from affective_video.config import ProjectConfig, save_config
from affective_video.data.manifest import ParticipantRecord, Record


class RunMetadata(Record):
    run_id: str
    experiment_name: str
    seed: int
    timestamp: str
    git_commit: str | None
    git_dirty: bool | None
    config_hash: str
    split_hash: str
    environment: dict[str, Any] = Field(default_factory=dict)
    hardware: dict[str, Any] = Field(default_factory=dict)


def canonical_hash(value: Any) -> str:
    """SHA-256 of sorted, finite JSON; callers must normalize list ordering."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def split_hash(records: list[ParticipantRecord]) -> str:
    ordered = sorted(
        (record.model_dump(mode="json") for record in records),
        key=lambda item: json.dumps(item, sort_keys=True),
    )
    return canonical_hash(ordered)


def _git_state(repository: Path) -> tuple[str | None, bool | None]:
    def git(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", "-C", str(repository), *args],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            return result.stdout.strip() if result.returncode == 0 else None
        except (OSError, subprocess.TimeoutExpired):
            return None

    root = git("rev-parse", "--show-toplevel")
    if root is None or Path(root).resolve() != repository.resolve():
        return None, None
    commit = git("rev-parse", "HEAD")
    status = git("status", "--porcelain", "--untracked-files=normal")
    return commit, bool(status) if status is not None else None


def create_run(
    config: ProjectConfig, records: list[ParticipantRecord], repository: Path
) -> tuple[Path, RunMetadata]:
    """Create a unique run without overwriting; timestamps and filename times are UTC."""
    now = datetime.now(UTC)
    run_id = (
        f"{now:%Y%m%d-%H%M%S}_{config.experiment.name}_{config.training.seed}_{uuid4().hex[:8]}"
    )
    commit, dirty = _git_state(repository)
    gpu_names = (
        [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
        if torch.cuda.is_available()
        else []
    )
    metadata = RunMetadata(
        run_id=run_id,
        experiment_name=config.experiment.name,
        seed=config.training.seed,
        timestamp=now.isoformat(),
        git_commit=commit,
        git_dirty=dirty,
        config_hash=canonical_hash(config.model_dump(mode="json")),
        split_hash=split_hash(records),
        environment={
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {
                package: version(package)
                for package in (
                    "affective-video-recognition",
                    "numpy",
                    "pydantic",
                    "PyYAML",
                    "torch",
                )
            },
            "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        },
        hardware={
            "machine": platform.machine(),
            "cpu": platform.processor() or "unknown",
            "cuda_available": torch.cuda.is_available(),
            "gpu_names": gpu_names,
            "torch_cuda_version": torch.version.cuda,
        },
    )
    directory = config.experiment.output_dir / run_id
    directory.mkdir(parents=True, exist_ok=False)
    save_config(config, directory / "config.yaml")
    save_json(directory / "metadata.json", metadata.model_dump(mode="json"))
    save_json(
        directory / "split_manifest.json", [record.model_dump(mode="json") for record in records]
    )
    return directory, metadata


def save_json(path: Path, value: Any) -> None:
    """Fail rather than emit nonstandard NaN/Infinity JSON."""
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
