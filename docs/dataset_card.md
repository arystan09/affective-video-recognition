# RECOLA dataset card — M2 pending access

**Status: BLOCKED ON RECOLA ACCESS.** No authorized local root was supplied. No
RECOLA recordings, annotations, or release metadata were found in the repository
data directory, Downloads, checked course folder, or checked Codex directories
(excluding generated environments and fixtures). These searches do not establish
that no release exists elsewhere on the computer.

Local eligibility, annotation duration, cadence, scale, missing rates, and split
membership are **not established**. No real manifest, split, or dataset plot was
generated. Synthetic file-tree tests are not dataset evidence.

## Original facts from external documentation

The original collection describes 46 French-speaking participants in collaborative
dyadic interactions, 9.5 hours of multimodal recordings, and annotations of the first
five minutes. The current site describes 23 accessible training/development subjects
and withheld official test annotations. [RECOLA overview](https://recola.human-ist.ch/),
checked 2026-10-06.

The module page describes six annotators, 40-ms affect ratings, and separately supplied
frame timestamps. It does not verify a local layout, columns, scale, missing tokens,
consensus availability, or locally available session/dyad IDs.
[RECOLA modules](https://recola.human-ist.ch/modules.html).

Academic access requires a signed EULA and a permanent-academic institutional request.
A separate commercial licensing route exists. No signed local agreement was available
to review, so redistribution permissions for annotations, metadata, figures, crops,
embeddings, and weights remain unverified.
[Access requirements](https://recola.human-ist.ch/download.html).

## This project's internal decisions

- Facial video only; no audio/biosignal analysis.
- Preserve official partitions as metadata. Internal splits are not official AVEC results.
- Inventory first; assign identities/roles only through a private, reviewed release mapping.
- No timestamp alignment, interpolation, normalization, or 5-Hz resampling in M2.
- Preserve official consensus if present. Inspect individual traces and exact-timestamp
  disagreement; do not average them without a documented later target-construction decision.
- Scale evidence means manually reviewed release bounds plus an existing evidence file
  and numeric conformance. Code does not semantically verify the documentary claim.
- Structural eligibility requires nonempty video/timestamp files, two consensus traces,
  nonnegative strictly increasing timing, documented scales, and the configured minimum
  overlap span (one second by default, a provisional sanity threshold).
- No face-quality exclusions. A nonempty video is not proof of container integrity.
  Duration/frame count are inferred from supplied timestamp rows, not decoded video.
- Cache SHA-256 using path/size/mtime/ctime; use `--rehash` before final source freeze.
- Proposed seed 42, label-independent 60/20/20 allocation of connected components across
  all known participant/video/session/dyad/video-checksum relationships. Unknown grouping
  metadata limit the independence claim and must be reviewed explicitly.

## Bias, limitations, and privacy

Language, interaction task, population, annotator perspectives, and capture conditions
limit generalization. Observer ratings operationalize perceived affective cues; they
do not establish internal emotional truth. Do not infer demographics from faces or
make clinical, psychological diagnostic, honesty, or surveillance claims.

**DO NOT COMMIT RECOLA DATA.** Raw ratings, private metadata, inventories, manifests,
split IDs, checksum caches, and dataset plots remain ignored. Relative source paths
do not automatically make a manifest public. Publish only reviewed sanitized aggregate
summaries where the signed agreement permits. Do not upload the release to external services.

## Evidence still needed

Obtain authorized local files and accompanying documentation/agreement; verify actual
layout/columns/units/tokens/scale/consensus; audit source-video timing and integrity;
review eligibility; create the real manifest and frozen split; run real leakage checks,
partition statistics, and figures; update this card with measured facts and approve
the split before M3.
