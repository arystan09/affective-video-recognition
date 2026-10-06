# M2 access-independent verification

**Milestone status: BLOCKED ON RECOLA ACCESS. M2 is not complete.**

The M1 repository was inspected before changes; its working core modules and existing
tests were left unchanged. Searches of the repository data directory, Downloads,
checked course folder, and Codex directories found no authorized RECOLA release.
No root was supplied. No dataset was downloaded.

## Implemented and verified

- Narrow explicit-mapping adapter, inventory, missing/unknown/duplicate reporting.
- Recording-level manifest contract and public JSON schemas.
- CSV/dense numeric ARFF inspection with declared columns/units/missing tokens/scales.
- Timestamp order, duplicates, negativity, cadence, irregularity, gaps, and overlap offsets.
- SHA-256 cache, source containment, deterministic manifest ordering.
- Consensus preservation; no individual averaging. Exact-time disagreement summaries.
- Connected-component split generation, frozen split validation, M1 leakage audit,
  manifest membership verification, counts and descriptive partition statistics.
- Optional private PNG/PDF plots and label-independent example selection.
- Local inspection CLI, root override/environment support, blocked exit behavior.
- Dataset card, workflow, decision record, privacy ignores, and Git safety check.

The full optional-inspection environment passes **131 tests**: 89 original M1 tests and
42 additional M2 cases. Ruff lint/format checks pass; mypy passes on 19 source files.
Fixtures are fabricated, including non-video placeholder bytes. No real eligibility,
cadence, scale, duration, missingness, plots, or split audit is claimed.

The fixture cadence is 4 Hz with explicitly documented [-2, 2] bounds, demonstrating
that release format assumptions are not fixed to 25 Hz or [-1, 1]. Fixture split
membership is not used as a research split. Plot generation is exercised only in
temporary test directories. CLI behavior without data was checked: exit code 2,
null eligibility counts, and no dataset manifest or frozen split creation.

Matplotlib is an optional `inspection` extra. Without it, the plot test is skipped;
the default M1 runtime dependencies remain unchanged. Mypy tolerates the absent optional
Matplotlib import while checking it when installed. CPU CI installs the extra to exercise
the full suite and includes the tracked-file safety check. Hosted CI was not run locally.

## Output currently present

Only ignored `data/manifests/recola_inspection.json` records the real-world blocker.
There is no `recola_manifest.csv`, no `recola_internal_v1.yaml`, and no real dataset figures.
Privacy ignore rules were checked for manifests, split IDs, local mappings, and plots.
Source-only packaging excludes these artifacts and generated environments/test trees.

## Remaining work after legitimate access

1. Inventory actual sources and inspect accompanying agreement/release documentation.
2. Verify paths, IDs, metadata hierarchy, columns, units, missing tokens, scale, and
   official consensus. Adapt unsupported formats rather than force the fixture layout.
3. If only individual traces exist, document and approve a configurable target rule
   before implementing averaging. Existing tooling intentionally excludes these targets.
4. Establish video timing/integrity, timestamp coverage, and real eligibility. Current
   structural checks and supplied timestamps do not certify valid video containers.
5. Generate and review the real manifest, source hashes, first frozen split, leakage
   audit, participant/partition statistics, and dataset figures.
6. Update dataset card/design/decision with measured facts and accept the real split.

M3 remains unstarted. No decoding, resampling, detection, tracking, alignment, features,
model, training, uncertainty estimator, or demo was introduced.
