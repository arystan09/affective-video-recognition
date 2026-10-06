import random

import numpy as np
import pytest
import torch

from affective_video.data.synthetic import generate_synthetic
from affective_video.utils.reproducibility import set_global_seed


def test_global_seed():
    set_global_seed(12)
    first = (random.random(), np.random.random(), torch.rand(3))
    set_global_seed(12)
    second = (random.random(), np.random.random(), torch.rand(3))
    assert first[:2] == second[:2]
    assert torch.equal(first[2], second[2])


@pytest.mark.parametrize("irregular", [False, True])
def test_synthetic_repeats(irregular):
    assert generate_synthetic(42, irregular=irregular) == generate_synthetic(
        42, irregular=irregular
    )
    assert generate_synthetic(42, irregular=irregular) != generate_synthetic(
        43, irregular=irregular
    )


def test_synthetic_covers_edge_cases():
    manifest, observations = generate_synthetic(42)
    assert len(manifest) == 12
    assert len({item.participant_id for item in manifest}) == 6
    assert any(not item.face_valid for item in observations)
    assert any(not item.valence_valid for item in observations)
    assert any(not item.arousal_valid for item in observations)
    assert {
        item.valence for item in observations if item.participant_id == "P05" and item.valence_valid
    } == {0.2}


def test_deterministic_flag():
    try:
        set_global_seed(1, deterministic=True)
        assert torch.are_deterministic_algorithms_enabled()
    finally:
        set_global_seed(1)
    assert not torch.are_deterministic_algorithms_enabled()


@pytest.mark.parametrize("seed", [-1, 2**32, True, 1.5])
def test_invalid_seed(seed):
    with pytest.raises(ValueError):
        set_global_seed(seed)
