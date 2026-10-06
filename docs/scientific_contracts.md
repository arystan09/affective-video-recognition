# Scientific contracts

## Records and missingness

All records are frozen Pydantic models with unknown fields rejected. Identifiers
are nonempty. Timestamps are finite, nonnegative seconds within a video/segment.
Targets are finite [-1, 1] or `None`. A valid target must have a value. Missing labels
must use a false target mask; zero is a valid affect coordinate, not missingness.

`ParticipantRecord` is a participant/video association with partition and optional
session, dyad, path, and checksum. `TimestampedObservation` separates face validity
from valence/arousal validity. `SequenceExample` stores unpadded sample-grid timestamps,
face masks, endpoint labels, and dimension-specific target masks. `PredictionRecord`
adds dimension-specific reference masks to the requested overall prediction-valid
flag; this is needed to evaluate annotation gaps independently. Uncertainty fields
are nonnegative variability values, not guaranteed confidence probabilities.

Contract objects reject NaNs as labels. Low-level metric functions still accept
arrays containing NaNs, exclude nonfinite pairs, and expose their exclusion counts.

## Split independence

Split checks reject every available participant, video, session, dyad, or source
checksum occurring in multiple partitions. Repeated identifiers within one partition
are permitted. Missing optional identifiers cannot prove session independence and
are not treated as shared IDs. Validation reports all detected leaks at once.
Empty split manifests are rejected. A caller must additionally verify that every
observation joins an audited manifest record before evaluating real data.

No random frame-level train/test mixing. Future local RECOLA splits will be grouped
by participant and dyad/session when available. They must be called internal splits,
not official challenge results. Train-only statistics and no test-set tuning apply
to preprocessing selection, normalization, early stopping, smoothing, and uncertainty
thresholds.

## Time alignment and causal windows

Generic construction groups by participant/video, sorts timestamps, rejects duplicate
timestamps, and resets context when a new known track ID appears. Unknown IDs during
missing observations do not themselves signal an identity switch. Track loss handling
belongs to M3 and must explicitly mark resets when identity cannot be established.

The sample grid starts at each segment's first timestamp and ends at its last complete
grid point. At time t it may use only the latest source observation at or before t,
and only if it is at most half a sample period old. Nearest-future matching is
prohibited. More than one source observation in a bin uses the latest causal one.
This is a conservative M1 resampling policy, not a RECOLA annotation-interpolation
algorithm. Unmatched bins carry false masks and `None` targets; annotations are
not interpolated. Irregular observations after the final grid point are not extrapolated.

Default: 5 Hz, 3 seconds, 15 observations, stride 5 grid endpoints. The first window
has one observation, so initial histories are shorter. Fifteen samples span 2.8
seconds between first and endpoint. Prediction is sequence-to-one at the endpoint.
No future timestamp or cross-video history is allowed. Missing grid positions are
preserved. Padding is not materialized in M1; any future batch collator must distinguish
padding from genuinely missing input with a separate padding mask.

## Metrics and numerical conventions

Use identical one-dimensional float64 target/prediction arrays per dimension with a
boolean mask. Pair eligibility requires a true mask and finite target and prediction.
Record total, valid, and unmasked-nonfinite counts. Valence and arousal are independent.

- MAE = mean absolute error.
- RMSE = square root of mean squared error.
- Pearson = covariance divided by the product of standard deviations.
- CCC = 2 covariance / (target variance + prediction variance + squared mean difference).

Use population moments consistently (ddof=0). Correlations are clipped only for
floating-point excursions outside [-1, 1]. Inputs from contracts are bounded [-1, 1].
No artificial denominator epsilon is added to reported metrics.

| Case | MAE/RMSE | CCC | Pearson |
|---|---|---|---|
| No valid finite pairs | undefined | undefined | undefined |
| One valid pair | defined | undefined | undefined |
| One trace constant, other variable | defined | 0 | undefined |
| Both traces constant, equal or unequal | defined | undefined | undefined |
| Both traces variable | defined | defined | defined |

Both-constant CCC is conservatively undefined: no dynamic agreement can be established.
Although the algebra gives zero for distinct constant means, this project intentionally
flags both-constant cases. Every undefined value is `None`/JSON `null` with a reason.

Compute per-participant metrics across all their supplied recordings, pooled metrics
across observations, and equal-participant macro averages. Macro excludes undefined
values **per metric** and reports contributor counts. An empty contributor set yields
null. Pooled and macro CCC answer different questions; pooled agreement can incorporate
between-participant offsets. Do not average batch CCC. Duplicate prediction keys
(participant, video, timestamp) are rejected.

Coverage is valid paired predictions divided by valid reference labels per dimension.
Overall prediction validity must reflect current face availability. Missing-face
observations must not be scored as successful predictions. For real comparisons the
caller must enforce a common eligible timeline across models.

## Runs and provenance

Each run has resolved YAML, finite JSON metadata and metrics, plus split and prediction
snapshots. Config hashes use sorted canonical JSON; split hashes also sort records,
making them input-order independent. Resolved absolute paths are included in config
hashes, so relocation changes the hash intentionally. Unique UTC IDs include experiment,
seed, and a collision-resistant suffix. Same-seed metrics can repeat while run IDs differ.
Git metadata is captured only for the supplied repository root; unavailable values
are null. Dirty state includes untracked files. Hardware and installed package versions
are recorded. No hosted experiment service is required.

## Interpretation

Human annotations operationalize perceived affect. Facial behavior does not establish
internal feelings, mental health, honesty, or diagnosis. Synthetic trajectories and
label-plus-noise predictions validate infrastructure only. They are not evidence of
emotion-recognition performance. No test-set tuning or performance claim is permitted.
