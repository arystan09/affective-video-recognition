import pytest
from pydantic import ValidationError

from affective_video.data.manifest import PredictionRecord, TimestampedObservation


@pytest.mark.parametrize(
    "update",
    [
        {"timestamp": -1},
        {"timestamp": float("nan")},
        {"valence": 2},
        {"valence": float("nan")},
        {"valence_valid": True, "valence": None},
        {"participant_id": ""},
    ],
)
def test_observation_rejects_invalid_contract(update):
    with pytest.raises(ValidationError):
        TimestampedObservation.model_validate(
            {
                "participant_id": "P",
                "video_id": "V",
                "timestamp": 0,
                **update,
            }
        )


def test_invalid_prediction_may_abstain():
    item = PredictionRecord(
        participant_id="P",
        video_id="V",
        timestamp=0,
        valid=False,
        target_valid_mask=(False, False),
    )
    assert item.predicted_valence is None


def test_valid_prediction_requires_values():
    with pytest.raises(ValidationError):
        PredictionRecord(
            participant_id="P", video_id="V", timestamp=0, target_valid_mask=(False, False)
        )
