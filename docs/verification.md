# M1 verification

Verified locally on 2026-10-06 with Python 3.11.17, Windows x86-64, and
Torch 2.14.1+cpu. No private data or GPU was used.

## Results

- Locked dependency installation succeeded in a fresh source copy and new `.venv`.
  Downloads were reused from a local uv cache; the source copy had no preexisting
  environment or private dataset.
- `python -m pytest -q --tb=short`: **89 passed** (20.62 seconds in the clean copy).
- `ruff check .`: all checks passed.
- `ruff format --check .`: all 29 Python files already formatted.
- `python -m mypy`: no issues in 12 source files.
- Config validation succeeded with resolved defaults and absolute paths.
- Regular and irregular CLI synthetic runs completed and wrote finite JSON artifacts.
- Git ignore checks confirmed private manifests, raw data, videos, checkpoints,
  environments, run artifacts, and pytest temporary files are excluded.

The checked-in GitHub Actions workflow runs locked CPU installation, Ruff, formatting,
mypy, pytest, config validation, and both synthetic smoke runs on Python 3.11 Linux.
**A hosted GitHub Actions execution is not verified:** this local repository has no
remote or published commit. Local Windows CPU verification is not a claim that a
hosted Linux job has already passed. The first push/PR will produce that evidence.

## Regular synthetic example

```json
{
  "infrastructure_only": true,
  "participants": 6,
  "videos": 12,
  "observations": 480,
  "sequences": 96,
  "test_valid_counts": {"valence": 24, "arousal": 24}
}
```

Actual outputs are under ignored `runs/<run_id>/`: resolved config, metadata, metrics,
split manifest, and predictions. The irregular run used the same 480 observations
and 96 endpoints with 12 valid test valence pairs and 10 valid arousal pairs. Its lower
coverage reflects deliberately conservative causal matching, not a trained model's
performance.

Run IDs use UTC and a unique suffix. Git commit was null and dirty state true in the
initialized, uncommitted deliverable repository. Dedicated tests verify clean/dirty
state and actual commit capture using a temporary test repository.

## Scope and delivery

The source archive excludes virtual environments, Git internals, runtime caches, and
generated run artifacts. M1 contains no dataset-specific parser, detector, visual
encoder, GRU, training routine, bootstrap analysis, uncertainty estimator, or demo.
The configuration fields for future architectures describe contracts only.

Tests invoke the end-to-end driver in-process; CLI entry points were separately
smoke-tested. Use `uv run --locked python -m pytest` and `python -m mypy` to avoid
platform-specific Windows console launcher behavior. Temporary test files are rooted
directly under the ignored `.pytest-tmp` directory so a pristine checkout needs no
preexisting pytest cache directory.
