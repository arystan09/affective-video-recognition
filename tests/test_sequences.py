import pytest
from pydantic import ValidationError

from affective_video.config import DataConfig
from affective_video.data.manifest import SequenceExample, TimestampedObservation
from affective_video.data.sequences import build_sequences


def test_length_initial_context_and_endpoint(observations):
    examples = build_sequences(observations, DataConfig(dataset_name="synthetic", stride=1))
    assert len(examples) == 30
    assert len(examples[0].timestamps) == 1
    assert len(examples[14].timestamps) == len(examples[-1].timestamps) == 15
    assert examples[-1].target_valence == observations[-1].valence
    assert examples[-1].target_arousal == observations[-1].arousal
    for example in examples:
        assert max(example.timestamps) == example.target_timestamp
        assert all(time <= example.target_timestamp for time in example.timestamps)


def test_missing_faces_preserved(observations):
    examples = build_sequences(observations, DataConfig(dataset_name="synthetic", stride=1))
    assert len(examples[7].timestamps) == 8
    assert examples[7].observation_mask[7] is False
    assert examples[7].target_valid_mask == (True, True)


def test_video_and_participant_isolation(observations):
    other = [item.model_copy(update={"video_id": "W"}) for item in observations]
    third = [item.model_copy(update={"participant_id": "Q"}) for item in observations]
    examples = build_sequences(observations + other + third, DataConfig(dataset_name="synthetic"))
    assert len(examples) == 18
    for participant, video in (("P", "V"), ("P", "W"), ("Q", "V")):
        assert (
            len(
                next(
                    item
                    for item in examples
                    if item.participant_id == participant and item.video_id == video
                ).timestamps
            )
            == 1
        )


def test_stride(observations):
    examples = build_sequences(observations, DataConfig(dataset_name="synthetic", stride=5))
    assert [item.target_timestamp for item in examples] == [0, 1, 2, 3, 4, 5]


def test_absent_grid_observation_keeps_mask(observations):
    del observations[7]
    examples = build_sequences(observations, DataConfig(dataset_name="synthetic", stride=1))
    assert len(examples) == 30
    assert examples[7].observation_mask[-1] is False
    assert examples[7].target_valid_mask == (False, False)
    assert examples[7].target_valence is None


def test_irregular_future_sample_never_used():
    observations = [
        TimestampedObservation(
            participant_id="P",
            video_id="V",
            timestamp=time,
            valence=value,
            valence_valid=True,
            face_valid=True,
        )
        for time, value in [(0, 0), (0.21, 0.9), (0.4, 0.4)]
    ]
    examples = build_sequences(observations, DataConfig(dataset_name="synthetic", stride=1))
    assert examples[1].target_timestamp == 0.2
    assert examples[1].target_valence is None
    assert examples[2].target_valence == 0.4


def test_future_changes_cannot_change_earlier_windows(observations):
    config = DataConfig(dataset_name="synthetic", stride=1)
    first = build_sequences(observations, config)
    changed = [
        item.model_copy(update={"valence": -1}) if item.timestamp > 2 else item
        for item in observations
    ]
    second = build_sequences(changed, config)
    assert first[:11] == second[:11]


def test_rate_and_duration(observations):
    examples = build_sequences(
        observations,
        DataConfig(dataset_name="synthetic", sample_rate_hz=2, sequence_seconds=2, stride=1),
    )
    assert len(examples[-1].timestamps) == 4
    assert examples[-1].timestamps[-1] - examples[-1].timestamps[0] == 1.5


def test_track_switch_resets_history(observations):
    observations = [
        item.model_copy(update={"track_id": "NEW"}) if index >= 10 else item
        for index, item in enumerate(observations)
    ]
    examples = build_sequences(observations, DataConfig(dataset_name="synthetic", stride=1))
    assert len(examples[10].timestamps) == 1
    assert examples[10].track_id == "NEW"


def test_duplicate_timestamp_rejected(observations):
    with pytest.raises(ValueError, match="duplicate timestamp"):
        build_sequences(observations + [observations[0]], DataConfig(dataset_name="synthetic"))


def test_empty_input():
    assert build_sequences([], DataConfig(dataset_name="synthetic")) == []


def test_sequence_contract_rejects_future_context():
    with pytest.raises(ValidationError, match="endpoint"):
        SequenceExample(
            participant_id="P",
            video_id="V",
            timestamps=(0, 1),
            observation_mask=(True, True),
            target_timestamp=0,
            target_valid_mask=(False, False),
        )
