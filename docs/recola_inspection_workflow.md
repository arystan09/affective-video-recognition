# Local RECOLA inspection workflow

M2 supplies unverified format tooling, not a guessed filename convention. No data are
downloaded. CSV and dense numeric ARFF are supported. Actual unsupported formats require
a documented extension after release inspection; sparse/mixed-type ARFF fails clearly.

## Commands from the repository root

```sh
uv sync --locked --extra inspection --python 3.11
uv run --locked python scripts/inspect_dataset.py --config configs/recola.yaml
```

Without a supplied root this writes an ignored status JSON and exits 2 with
`BLOCKED_ON_RECOLA_ACCESS`. It creates no fabricated manifest or split.

Inventory actual files first:

```sh
uv run --locked python scripts/inspect_dataset.py --config configs/recola.yaml --root /authorized/local/release
```

Alternatively set `RECOLA_ROOT`. Inventory-only mode exits 2: file identities and roles
remain unverified. Review ignored `data/manifests/recola_inspection.json`. Candidate
extensions do not establish roles; unmapped files are reported. Outputs must be outside
the release root. Use the release folder, not a parent containing the repository.

## Private release mapping

Keep mapping in `configs/recola-release.local.yaml` or an external private path.
Schema: [recola_mapping.schema.json](schemas/recola_mapping.schema.json). Use relative,
contained paths with `/` separators. IDs, grouping, units, scale, and consensus status
must be verified against actual files/documentation, not inferred from similar names.

**Illustrative fragment only: no paths, IDs, or columns below are real RECOLA facts.**

```yaml
schema_version: 1
evidence_files: [release-documentation.txt]
minimum_annotation_seconds: 1.0
recordings:
  - participant_id: VERIFIED_ID
    video_id: VERIFIED_RECORDING_ID
    session_id: null
    dyad_id: null
    original_release_partition: null
    video_path: verified/video-path.mp4
    timestamps:
      path: verified/frame-times.csv
      format: csv
      delimiter: ";"
      timestamp_column: VERIFIED_TIME_COLUMN
      timestamp_unit: seconds
    annotations:
      - path: verified/annotation.csv
        format: csv
        delimiter: ";"
        timestamp_column: VERIFIED_TIME_COLUMN
        timestamp_unit: seconds
        missing_tokens: ["", "?"]
        dimension: valence
        value_column: VERIFIED_VALUE_COLUMN
        kind: consensus
        scale_min: null
        scale_max: null
        scale_evidence: null
```

Add arousal, actual bounds/evidence, and every possessed recording. Use consensus only
when the release identifies an official consensus. For individual traces specify
`kind: individual` and `annotator_id`. Consensus alone does not establish rater count.
Seconds/milliseconds are explicit; unsupported units are a blocker. Missing-token
defaults are parser conveniences, not verified release behavior: specify actual tokens.
Unknown sentinels outside documented bounds cause scale violations.

## Review and freeze

```sh
uv run --locked python scripts/inspect_dataset.py --config configs/recola.yaml --root /authorized/local/release --mapping configs/recola-release.local.yaml
uv run --locked python scripts/inspect_dataset.py --config configs/recola.yaml --root /authorized/local/release --mapping configs/recola-release.local.yaml --make-split --rehash --report
```

Inspect unknown modules, exclusions, duplicate/malformed tables, frame timing, and target
statistics before freezing. Partial eligible/excluded rows remain inspectable. Critical
source errors prevent finalization; review an explicit possessed subset rather than
silently ignore errors. Without a split the command exits nonzero.

Outputs: private manifest CSV, inventory/statistics JSON, internal split YAML, leakage
audit JSON/Markdown, and PNG/PDF dataset figures. Repeated runs retain the existing
split bytes/timestamp. Changes to manifest, seed, algorithm, or assignments fail;
new versions require deliberate paths and acceptance. No label-driven split regeneration.

Durations are overlap spans, not gap-subtracted exposure. Raw consensus counts are not
aligned 5-Hz paired labels or face-valid samples. Container integrity is not established
by file existence. Video duration/frame count use supplied timing rows; actual video
timing/integrity verification remains part of the real-release review. No video probe,
frame decoding, label normalization, or resampling is implemented here.

Plots remain ignored derived data. Example traces use the lexicographically first
available consensus video per dimension, independently of values. Missing fractions
remain unknown when no reference trace exists. Source hashes can require one full read
of large videos; cached stat fingerprints avoid repeats. Rehash before a final freeze.

## Verification

```sh
uv run --locked python -m pytest
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked python -m mypy
uv run --locked python scripts/check_data_safety.py --include-untracked
```

Tests create fabricated source trees, including explicitly non-video placeholder bytes.
Git safety scans tracked paths (optionally nonignored untracked paths) for private data
locations, video/ARFF files, and files over 10 MiB. This is a safeguard, not a license review.
