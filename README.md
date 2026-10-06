# Video-Based Facial Affect Recognition

**Valence–Arousal Estimation with Temporal Modeling and Uncertainty Analysis**

University research prototype. **M1 foundation is verified; M2's real-data audit is
blocked on RECOLA access.** No RECOLA preprocessing, face detection, encoder, neural
training, or demo is implemented. No model performance has been established.

**M2 update: BLOCKED ON RECOLA ACCESS.** Access-independent inspection tooling is
implemented, but no real release, eligible participant count, verified scale/cadence,
manifest, or frozen split is claimed. M1 remains intact. See
[dataset card](docs/dataset_card.md) and [inspection workflow](docs/recola_inspection_workflow.md).
Use `configs/recola.yaml` and a CLI `--root` (or `RECOLA_ROOT`); `configs/data.yaml`
continues to support the working M1 synthetic run. Dataset plots require
`uv sync --locked --extra inspection`. No dataset is downloaded by any command.

Valence represents negative-to-positive affect; arousal represents calm-to-activated
affect. Both targets use [-1, 1]. Continuous targets preserve variation over time
without forcing every observation into a discrete emotion category.

The planned system samples facial video at 5 Hz, aligns crops, caches frozen
ResNet-18 embeddings, and compares a frame MLP, temporal pooling, and a small causal
GRU using three seconds of context. MC dropout will later investigate uncertainty.
See [approved design](docs/research_design.md) and
[scientific contracts](docs/scientific_contracts.md).

## Scientific interpretation

Predictions reflect model estimates of observable affective cues and should not be
interpreted as ground-truth internal emotional states. Labels are subjective,
context-dependent operational approximations. This is not a diagnostic or
surveillance product. Cultural variation, dataset bias, privacy, annotation delay,
and domain shift constrain conclusions.

## Installation

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run these
commands **from the repository root** in a fresh clone:

```sh
uv python install 3.11
uv sync --locked --python 3.11
```

The authoritative environment is `pyproject.toml` plus `uv.lock`. Python is restricted
to 3.11.x for M1. Torch comes from its explicit CPU wheel index; CUDA is unnecessary.
The uv version used to produce the lock and in CI is 0.12.23. M1 runtime dependencies
are only NumPy, Pydantic, PyYAML, and Torch. NumPy supplies the small metric calculations;
pandas, SciPy, and scikit-learn are unnecessary at this milestone.

## Verification

```sh
uv run --locked python -m pytest
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked python -m mypy
uv run --locked python scripts/validate_config.py --config configs/data.yaml
uv run --locked python scripts/synthetic_run.py --config configs/data.yaml
uv run --locked python scripts/synthetic_run.py --config configs/data.yaml --irregular
```

The synthetic run checks six independent participants, two videos each, causal
windows, gaps, masks, metrics, and artifact serialization. **It fabricates predictions
by adding noise to reference labels. It is an infrastructure check, not a trained
baseline or scientific result.** Regular data contain 480 observations and produce
96 windows at stride 5. A JSON summary prints the run directory and valid counts.

Each unique UTC-named run contains `config.yaml`, `metadata.json`, `metrics.json`,
`split_manifest.json`, and `predictions.json`. Config paths resolve relative to the
YAML file, not the invoking directory. The resolved configuration includes defaults.
Git commit/dirty state are null when the supplied repository has no accessible Git
metadata; unavailable provenance is never invented.

The synthetic driver honors sampling, context, stride, target dimensions, macro
evaluation, output location, seed, and deterministic flag. Model/training fields are
validated future experiment contracts; the driver does not instantiate a model.
Synthetic input is generated at 5 Hz; changing the requested sampling rate exercises
resampling rather than changing its source rate. No private dataset is needed.

Tests keep temporary artifacts under `.pytest-tmp` so the standard test command
also works in environments with restricted system temporary directories.
Module invocation of pytest/mypy avoids platform-specific console-launcher behavior.

## Layout

- `src/affective_video/config.py`: typed YAML configuration.
- `src/affective_video/data/`: manifests, split checks, windows, synthetic generation.
- `src/affective_video/evaluation/metrics.py`: masked and participant-level metrics.
- `src/affective_video/utils/`: seeds and run provenance.
- `scripts/`: config validation and synthetic end-to-end driver.
- `tests/`: analytical, edge-case, and integration checks.
- `docs/`: approved research plan and methodological decisions.
- `.github/workflows/ci.yml`: Python 3.11 CPU lint, typing, tests, and smoke runs.

Raw data and runtime artifacts are ignored by default. Inspect staged files before
publication: `.gitignore` is a safeguard, not a license or privacy guarantee.

## Planned milestones

| Milestone | Purpose |
|---|---|
| M0 | Approved research design |
| M1 | Foundation and scientific contracts (this repository) |
| M2 | Inspect authorized RECOLA release and freeze internal splits |
| M3 | Timestamped face preprocessing and versioned caching |
| M4 | Training-mean baseline and complete timeline evaluation |
| M5 | Frozen encoder and frame baseline |
| M6 | Pooling and causal GRU |
| M7 | Paired participant evaluation and report plots |
| M8 | MC dropout and risk–coverage analysis |
| M9 | Context/loss ablations and optional robustness |
| M10 | Local MP4 demo |
| M11 | Final report assets, cards, and defense notes |

## Reproducibility limits

Seeds cover Python, NumPy's global generator, and Torch. Synthetic data also use an
explicitly seeded local NumPy generator. Deterministic Torch mode is optional and
raises if an unsupported operation is used. GPU libraries, hardware, threading,
hash randomization, and library versions can prevent cross-platform bitwise identity.
Set `PYTHONHASHSEED` before process startup if hash-order control is required; CUDA
may require startup settings such as `CUBLAS_WORKSPACE_CONFIG` in later milestones.
Three-seed comparisons are planned; deterministic repetition is not evidence of
statistical robustness.

Original code is MIT licensed. External dataset/model terms remain separate; see
[third-party notices](THIRD_PARTY_NOTICES.md).
