# Decision 001: minimal, auditable M1

**What:** Python 3.11, uv lock, CPU Torch, frozen Pydantic records, safe YAML, NumPy
metrics, local JSON runs, unpadded generic windows.

**Why:** Type/shape/missingness errors and leakage are higher-priority risks than
model complexity. Small dependencies and synthetic CPU tests allow fresh-clone verification.

**Alternatives:** dataclasses would need separate validation; hosted trackers add setup
and privacy exposure; pandas/SciPy/sklearn are unnecessary for these calculations;
full dataset classes and batch padding belong to later milestones.

**Assumptions:** video-relative seconds, bounded labels, one selected target track,
known IDs, separately auditable split manifests, explicit reference validity.

**Limitations:** IDs cannot reveal undocumented real-world identity duplication. A
half-step causal resampling tolerance can reject delayed irregular samples; this is
deliberate and must be revisited separately from dataset-specific label alignment.
No feature tensors, annotation interpolation, neural model, bootstrap, or uncertainty
estimator is implemented. Macro contributor counts expose undefined constant traces.

**Deviations:** added per-dimension prediction masks, coverage, track reset, synthetic
generator module, and saved split/prediction artifacts because they support the approved
scientific contracts. Requirements export is omitted because one lock is authoritative.
Only CPU Torch is installed; later GPU work needs an explicit environment decision.
