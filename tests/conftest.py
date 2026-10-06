"""Small reusable fixtures; no private recordings are loaded."""

import pytest

from affective_video.config import ProjectConfig
from affective_video.data.manifest import TimestampedObservation
from affective_video.data.synthetic import generate_synthetic


@pytest.fixture
def config(tmp_path):
    return ProjectConfig.model_validate(
        {
            "data": {"dataset_name": "synthetic"},
            "experiment": {"name": "test", "output_dir": str(tmp_path / "runs")},
        }
    )


@pytest.fixture
def manifest():
    return generate_synthetic(42)[0]


@pytest.fixture
def observations():
    return [
        TimestampedObservation(
            participant_id="P",
            video_id="V",
            timestamp=index / 5,
            valence=index / 100,
            arousal=-index / 100,
            valence_valid=True,
            arousal_valid=True,
            face_valid=index != 7,
            track_id="T",
        )
        for index in range(30)
    ]
