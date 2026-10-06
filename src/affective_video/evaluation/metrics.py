"""Population-moment metrics; undefined results serialize as JSON null with reasons."""

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import ArrayLike

from affective_video.data.manifest import PredictionRecord


@dataclass(frozen=True)
class MetricResult:
    ccc: float | None
    mae: float | None
    rmse: float | None
    pearson: float | None
    valid_count: int
    total_count: int
    nonfinite_count: int
    undefined: dict[str, str]


def regression_metrics(
    targets: ArrayLike, predictions: ArrayLike, mask: ArrayLike | None = None
) -> MetricResult:
    """Require equal 1-D shapes. Exclude masked/nonfinite pairs and count exclusions.

    MAE/RMSE need one pair. CCC/Pearson need two. One constant trace gives CCC=0
    if the denominator is positive; two constant traces make CCC undefined, even
    if they differ. Pearson is undefined if either trace is constant. No epsilon
    is added to metric denominators: undefined agreement is not perfect agreement.
    """
    y = np.asarray(targets, dtype=np.float64)
    p = np.asarray(predictions, dtype=np.float64)
    if y.ndim != 1 or p.shape != y.shape:
        raise ValueError("targets and predictions must have identical one-dimensional shapes")
    valid = np.ones(y.shape, dtype=bool) if mask is None else np.asarray(mask)
    if valid.size == 0 and valid.shape == y.shape:
        valid = valid.astype(bool)
    if valid.shape != y.shape or valid.dtype != np.bool_:
        raise ValueError("mask must be a boolean array with the same shape as targets")
    finite = np.isfinite(y) & np.isfinite(p)
    nonfinite = int(np.sum(valid & ~finite))
    y, p = y[valid & finite], p[valid & finite]
    n = len(y)
    undefined = {}
    if not n:
        return MetricResult(
            None,
            None,
            None,
            None,
            0,
            len(valid),
            nonfinite,
            dict.fromkeys(("ccc", "mae", "rmse", "pearson"), "no valid finite pairs"),
        )
    error = y - p
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error**2)))
    ccc = pearson = None
    if n < 2:
        undefined = dict.fromkeys(("ccc", "pearson"), "fewer than two valid pairs")
    else:
        yc, pc = y - y.mean(), p - p.mean()
        # Repeated decimal values can have a rounded mean unequal to that value.
        # Check constancy directly rather than interpret roundoff as signal variance.
        y_constant, p_constant = bool(np.all(y == y[0])), bool(np.all(p == p[0]))
        vy = 0.0 if y_constant else float(np.mean(yc**2))
        vp = 0.0 if p_constant else float(np.mean(pc**2))
        covariance = 0.0 if y_constant or p_constant else float(np.mean(yc * pc))
        if vy == 0 and vp == 0:
            undefined["ccc"] = "both traces are constant"
        else:
            denominator = vy + vp + float((y.mean() - p.mean()) ** 2)
            ccc = float(np.clip(2 * covariance / denominator, -1, 1))
        if vy == 0 or vp == 0:
            undefined["pearson"] = "at least one trace is constant"
        else:
            pearson = float(np.clip(covariance / np.sqrt(vy * vp), -1, 1))
    return MetricResult(ccc, mae, rmse, pearson, n, len(valid), nonfinite, undefined)


@dataclass(frozen=True)
class EvaluationResult:
    per_participant: dict[str, dict[str, MetricResult]]
    pooled: dict[str, MetricResult]
    macro: dict[str, dict[str, float | None]]
    macro_contributors: dict[str, dict[str, int]]
    valid_counts: dict[str, int]
    coverage: dict[str, float | None]

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate_predictions(
    records: Sequence[PredictionRecord],
    dimensions: Sequence[str] = ("valence", "arousal"),
    *,
    participant_macro_average: bool = True,
) -> EvaluationResult:
    """Evaluate full participant traces, not an average of batch correlations.

    Macro scores exclude undefined participant metrics and expose contributor counts.
    Coverage denominator is the number of valid reference labels per dimension.
    Duplicate participant/video/timestamp predictions are rejected.
    """
    if not dimensions or len(set(dimensions)) != len(dimensions):
        raise ValueError("dimensions must be nonempty and unique")
    if any(dimension not in ("valence", "arousal") for dimension in dimensions):
        raise ValueError("unknown target dimension")
    seen = set()
    grouped: dict[str, list[PredictionRecord]] = defaultdict(list)
    for record in records:
        key = (record.participant_id, record.video_id, record.timestamp)
        if key in seen:
            raise ValueError(f"duplicate prediction: {key}")
        seen.add(key)
        grouped[record.participant_id].append(record)

    def summarize(items: Sequence[PredictionRecord], dimension: str) -> MetricResult:
        index = ("valence", "arousal").index(dimension)
        return regression_metrics(
            [getattr(item, f"target_{dimension}") for item in items],
            [getattr(item, f"predicted_{dimension}") for item in items],
            [item.valid and item.target_valid_mask[index] for item in items],
        )

    per_participant = {
        participant: {dimension: summarize(items, dimension) for dimension in dimensions}
        for participant, items in sorted(grouped.items())
    }
    pooled = {dimension: summarize(records, dimension) for dimension in dimensions}
    macro: dict[str, dict[str, float | None]] = {}
    contributors: dict[str, dict[str, int]] = {}
    if participant_macro_average:
        for dimension in dimensions:
            macro[dimension], contributors[dimension] = {}, {}
            for name in ("ccc", "mae", "rmse", "pearson"):
                values = [
                    value
                    for scores in per_participant.values()
                    if (value := getattr(scores[dimension], name)) is not None
                ]
                macro[dimension][name] = float(np.mean(values)) if values else None
                contributors[dimension][name] = len(values)
    coverage = {}
    for dimension in dimensions:
        index = ("valence", "arousal").index(dimension)
        reference_count = sum(record.target_valid_mask[index] for record in records)
        coverage[dimension] = (
            pooled[dimension].valid_count / reference_count if reference_count else None
        )
    return EvaluationResult(
        per_participant,
        pooled,
        macro,
        contributors,
        {dimension: pooled[dimension].valid_count for dimension in dimensions},
        coverage,
    )
