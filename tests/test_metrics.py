import numpy as np
import pytest

from affective_video.data.manifest import PredictionRecord
from affective_video.evaluation.metrics import evaluate_predictions, regression_metrics


def test_perfect_prediction():
    result = regression_metrics([-1, 0, 1], [-1, 0, 1])
    assert result.ccc == pytest.approx(1)
    assert result.pearson == pytest.approx(1)
    assert result.mae == result.rmse == 0


def test_inverted_prediction():
    result = regression_metrics([-1, 0, 1], [1, 0, -1])
    assert result.ccc == pytest.approx(-1)
    assert result.pearson == pytest.approx(-1)
    assert result.mae == pytest.approx(4 / 3)
    assert result.rmse == pytest.approx(np.sqrt(8 / 3))


def test_offset_ccc_uses_population_moments():
    # Variance=2/3, covariance=2/3, squared mean difference=1.
    result = regression_metrics([-1, 0, 1], [0, 1, 2])
    assert result.ccc == pytest.approx(4 / 7)
    assert result.pearson == pytest.approx(1)
    assert result.mae == result.rmse == 1


@pytest.mark.parametrize("targets,predictions", [([0, 0], [0, 0]), ([0, 0], [1, 1])])
def test_both_constant_undefined(targets, predictions):
    result = regression_metrics(targets, predictions)
    assert result.ccc is None
    assert result.pearson is None
    assert "constant" in result.undefined["ccc"]


@pytest.mark.parametrize("targets,predictions", [([0, 0], [-1, 1]), ([-1, 1], [0, 0])])
def test_one_constant(targets, predictions):
    result = regression_metrics(targets, predictions)
    assert result.ccc == 0
    assert result.pearson is None


def test_masks_and_nans():
    result = regression_metrics(
        [-1, 0, 1, np.nan, 100], [-1, 0, 1, 0, np.inf], [True, True, True, True, False]
    )
    assert result.valid_count == 3
    assert result.nonfinite_count == 1
    assert result.total_count == 5
    assert result.ccc == pytest.approx(1)


@pytest.mark.parametrize(
    "targets,predictions,mask",
    [
        ([], [], None),
        ([1], [1], [False]),
        ([np.nan], [1], None),
        ([1], [np.inf], None),
    ],
)
def test_empty_valid_set(targets, predictions, mask):
    result = regression_metrics(targets, predictions, mask)
    assert result.valid_count == 0
    assert result.ccc is result.mae is result.rmse is result.pearson is None


def test_single_pair():
    result = regression_metrics([0], [0.5])
    assert result.mae == result.rmse == 0.5
    assert result.ccc is result.pearson is None


def test_tiny_nonconstant_variance_not_epsilon_biased():
    result = regression_metrics([0, 1e-10], [0, 1e-10])
    assert result.ccc == pytest.approx(1)


def test_repeated_decimal_is_constant_despite_mean_roundoff():
    result = regression_metrics([0.2] * 17, [0.2] * 17)
    assert result.ccc is result.pearson is None
    result = regression_metrics([0.2] * 17, np.linspace(-1, 1, 17))
    assert result.ccc == 0
    assert result.pearson is None


def test_independent_reference():
    y, p = np.array([-0.8, -0.2, 0.4, 0.7]), np.array([-0.6, 0.1, 0.2, 0.9])
    result = regression_metrics(y, p)
    covariance = np.cov(y, p, ddof=0)
    reference = (
        2 * covariance[0, 1] / (covariance[0, 0] + covariance[1, 1] + (y.mean() - p.mean()) ** 2)
    )
    assert result.ccc == pytest.approx(reference)
    assert result.pearson == pytest.approx(np.corrcoef(y, p)[0, 1])


@pytest.mark.parametrize(
    "targets,predictions,mask",
    [
        ([1], [1, 2], None),
        ([[1]], [[1]], None),
        ([1], [1], [1]),
        ([1], [1], [True, False]),
    ],
)
def test_bad_shapes(targets, predictions, mask):
    with pytest.raises(ValueError):
        regression_metrics(targets, predictions, mask)


def prediction(participant, index, target, predicted, **kwargs):
    return PredictionRecord(
        participant_id=participant,
        video_id=participant + "_V",
        timestamp=index,
        target_valence=target,
        target_arousal=target,
        predicted_valence=predicted,
        predicted_arousal=predicted,
        **kwargs,
    )


def test_participant_macro_not_weighted_by_frames():
    records = [prediction("A", i, y, y) for i, y in enumerate([-1, 0, 1])]
    records += [prediction("B", i, y, -y) for i, y in enumerate([-0.5, 0.5])]
    result = evaluate_predictions(records)
    assert result.macro["valence"]["ccc"] == pytest.approx(0)
    assert result.pooled["valence"].ccc == pytest.approx(0.6)
    assert result.macro_contributors["valence"]["ccc"] == 2
    assert result.valid_counts["valence"] == 5


def test_partial_dimension_masks_and_coverage():
    records = [prediction("A", 0, 0, 0)]
    records.append(
        PredictionRecord(
            participant_id="A",
            video_id="A_V",
            timestamp=1,
            target_valence=None,
            target_arousal=1,
            target_valid_mask=(False, True),
            predicted_valence=0,
            predicted_arousal=1,
        )
    )
    records.append(prediction("A", 2, 1, 0, valid=False))
    result = evaluate_predictions(records)
    assert result.valid_counts == {"valence": 1, "arousal": 2}
    assert result.coverage["valence"] == pytest.approx(0.5)
    assert result.coverage["arousal"] == pytest.approx(2 / 3)
    assert result.macro_contributors["valence"]["ccc"] == 0
    assert result.macro["valence"]["ccc"] is None


def test_empty_evaluation_and_macro_switch():
    result = evaluate_predictions([], participant_macro_average=False)
    assert result.macro == {}
    assert result.coverage["valence"] is None


def test_duplicate_prediction_rejected():
    item = prediction("A", 0, 0, 0)
    with pytest.raises(ValueError, match="duplicate prediction"):
        evaluate_predictions([item, item])


@pytest.mark.parametrize("dimensions", [[], ["anger"], ["valence", "valence"]])
def test_bad_dimensions(dimensions):
    with pytest.raises(ValueError):
        evaluate_predictions([], dimensions)
