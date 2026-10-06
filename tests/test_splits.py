import pytest

from affective_video.data.splits import SplitLeakageError, validate_splits


def test_valid_split(manifest):
    validate_splits(manifest)


@pytest.mark.parametrize(
    "field",
    [
        "participant_id",
        "video_id",
        "session_id",
        "dyad_id",
        "source_checksum",
    ],
)
def test_every_leak_type(manifest, field):
    train = manifest[0]
    validation_index = 4
    manifest[validation_index] = manifest[validation_index].model_copy(
        update={field: getattr(train, field)}
    )
    with pytest.raises(SplitLeakageError, match=field) as error:
        validate_splits(manifest)
    assert "train" in str(error.value) and "validation" in str(error.value)


def test_missing_optional_ids_do_not_leak(manifest):
    validate_splits(
        [
            item.model_copy(update={"session_id": None, "dyad_id": None, "source_checksum": None})
            for item in manifest
        ]
    )


def test_all_leaks_reported(manifest):
    manifest[4] = manifest[4].model_copy(
        update={
            "participant_id": manifest[0].participant_id,
            "video_id": manifest[0].video_id,
        }
    )
    with pytest.raises(SplitLeakageError) as error:
        validate_splits(manifest)
    assert "participant_id" in str(error.value) and "video_id" in str(error.value)


def test_empty_split_rejected():
    with pytest.raises(ValueError, match="empty"):
        validate_splits([])
