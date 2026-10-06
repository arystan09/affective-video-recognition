# Approved Phase 0 research design

Title: Video-Based Facial Affect Recognition: Valence–Arousal Estimation with Temporal
Modeling and Uncertainty Analysis.

## Questions and scope

Primary question: On participant-independent RECOLA partitions, does a small causal
GRU using three seconds of frozen facial-image embeddings improve valence and arousal
concordance compared with a frame-independent MLP using the same encoder and preprocessing?

Secondary questions: Does learned temporal structure outperform pooling/smoothing?
How do one-, three-, and five-second contexts compare? Does MC dropout variability
identify larger absolute errors and support useful selective prediction?

Facial video only; two bounded continuous targets. No required classification,
multimodal fusion, Transformer, or full encoder comparison. Negative results remain
valid scientific outcomes. M1 implements only infrastructure.

## Dataset

Primary: RECOLA, chosen for spontaneous interactions, continuous annotations, and
manageable compute. Current overview describes 23 accessible training/development
participants; official test labels are withheld. Annotation intervals are 40 ms and
source video timestamps are supplied. Use the annotated first five minutes only.
Confirm actual completeness and release format in M2.

The access request requires a signed EULA and a permanent academic's institutional
request. Ask the supervisor early. References:
https://recola.human-ist.ch/ ; https://recola.human-ist.ch/modules.html ;
https://recola.human-ist.ch/download.html

Use the documented consensus target where available, otherwise a fixed arithmetic
mean of valid annotator ratings. Validate scale from release documentation. Internal
splits approximately 60/20/20, grouped by participant and session/dyad where possible;
do not call them official challenge splits. Develop with synthetic fixtures while
awaiting access. Optional AFEW-VA adapter checks do not become a second required study.

## Future pipeline and models

Source timestamps → 5 Hz sampling → YuNet target-face tracking → five-landmark
alignment → 224×224 RGB crops → frozen ImageNet ResNet-18 → 512-D cached embeddings.
Retain crops for auditability, masks and quality flags for failures, and hashes for
cache invalidation. Never redetect faces during routine cached-feature training.

Comparisons: training mean; current-frame MLP with 128 hidden units; causal smoothing
of frame outputs; valid-history mean pooling concatenated with current embedding;
one-layer unidirectional GRU with hidden size 128 and a 64-unit dropout regression head.
Default dropout 0.2; tanh outputs. Three-second windows contain 15 observations;
training stride 5, evaluation stride 1. Initial histories are shorter, track switches
reset context, and endpoint supervision is dimension-masked.

Train-only normalization; encoder/BatchNorm frozen. AdamW lr=0.001, weight decay=0.0001,
batch=64, max=60 epochs, early stopping patience=10, clipping=1.0. Optimize
0.5 MSE + 0.5 mean(1−CCC) across dimensions. Select checkpoints by validation mean CCC.
All models share cached representation and eligible timestamps.

## Evaluation and uncertainty

Primary: participant-macro CCC separately for valence/arousal. Secondary: pooled CCC,
MAE, RMSE, Pearson, coverage. Save full predictions, report three seeds, paired
participant differences, and later cluster-bootstrap intervals. Small test populations
limit certainty; correlated frames are not independent subjects.

Thirty MC dropout passes in the regression head, with encoder/GRU/BatchNorm in eval
mode. Report mean, standard deviation, and empirical variability bands. These are not
automatically calibrated predictive intervals. Assess within-participant Spearman
association, uncertainty bins, and risk–coverage versus random and face-quality rejection.

Required matrix: mean, frame, smoothing, pooling, GRU, and GRU MC inference. Required
ablations: context 1/3/5 seconds; MSE-only versus mixed loss. Three seeds, about 18
trained configurations including ablations. Optional blur, brightness, and lower-face
occlusion tests rerun the affected pipeline and report detection coverage.

## Engineering, budget, and outputs

Python 3.11, PyTorch, YAML, local JSON tracking, one dependency lock. GPU 6–8 GB and
RAM 16 GB recommended; reserve 20–30 GB initially, verify in M2/M3. Cached-feature
training is small. Budget approximately six to eight weeks. Estimates are not measured
runtime guarantees. Every run saves config, split/cache hashes, versions, Git state,
checkpoints, and metrics. CPU CI uses synthetic fixtures.

Required report figures: both timelines, errors, uncertainty–error, model comparison,
learning curves, risk–coverage, optional VA trajectory. Export vector and 300-DPI PNG
from saved predictions. Local Streamlit MP4 demo follows experiments. Delete temporary
uploads and show observable-affect disclaimer. No accuracy is shown for unlabeled uploads.

Report: abstract, introduction, related work, affect representation, dataset,
methodology, models, experimental setup, results, ablations, discussion, limitations
and ethics, conclusion, references. Discuss cultural variation, subjectivity, demographic
bias, privacy/surveillance risks, domain shift, lighting/pose, and incomplete uncertainty.
Maintain decision records for oral defense. Dataset and model licenses remain separate.

Milestones M0–M11 are listed in README. M2 cannot begin until M1 is verified and the
next milestone is authorized.

## M2 evidence update (2026-10-06)

M1 is verified and M2 was authorized. **M2 remains BLOCKED ON RECOLA ACCESS.** No local
authorized root was supplied, and no release files were found in the checked repository,
Downloads, course, or Codex locations. This does not prove absence elsewhere on the device.

The Phase 0 statements about approximately 23 available participants, 40-ms annotation
cadence, a five-minute subset, and possible consensus were planning assumptions informed
by external documentation. They are not verified local counts/format/scale/targets.
No accepted split or final preprocessing observation count exists.

Access-independent tooling now inventories explicit mappings, parses CSV/dense numeric
ARFF, audits raw timestamps and missingness, hashes sources, proposes connected-component
splits, and generates private summaries/plots when sources are actually present.
Individual-only ratings remain ineligible until target construction is documented.
No normalization, temporal alignment, downsampling, face preprocessing, or models are added.
See the dataset card, inspection workflow, and proposed split decision. The original
Phase 0 design above is preserved to expose these unresolved assumptions.
