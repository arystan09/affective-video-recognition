# Proposed RECOLA internal split — not frozen on real data

**Status:** BLOCKED ON RECOLA ACCESS. No real grouping hierarchy or membership accepted.

**WHAT:** Join transitive connected components across equal participant, video, session,
dyad, and source-video checksum IDs. Sort groups, shuffle with Python `Random(42)`,
stably sort by decreasing participant count, and assign the first three groups to
train/validation/test. Allocate each remaining group to minimize squared deviation from
60/20/20 participant counts; ties follow train/validation/test order. No labels are used.

**WHY:** Participant-only splitting can separate partners; a single grouping-field
fallback misses transitive relationships. Preserve every available link and prioritize
scientific grouping over exact percentages. Nonempty partitions require at least three
independent components.

**ALTERNATIVES:** Official partitions offer benchmark comparability but official test
labels are withheld. Participant-only grouping is weaker. Grouped cross-validation
expands the later budget. Random frame/clip splitting is rejected.

**ASSUMPTIONS:** IDs/checksums and eligible sources are verified from the actual release.
Known sessions/dyads identify meaningful shared interactions. Missing IDs are explicitly
reported and do not establish true interaction independence.

**LIMITATIONS:** Components can be unbalanced or collapse. Target distributions can be
narrow; report rather than repeatedly regenerate the split. Undocumented identity
duplicates or re-encoded duplicate recordings cannot be detected by exact checksums.

**FREEZING:** Name, seed, algorithm/version, UTC creation time, manifest/assignment hashes,
component/missing-group counts, and recording assignments are saved. Refuse overwrite on
source/seed/algorithm/assignment changes. M1 audits every available identifier; additionally
compare assignments against the exact eligible manifest.

**NEXT:** Actual grouping, participant counts, durations, distributions, and license
permissions remain unknown. Approve a real-data decision before creating the accepted
`recola_internal_v1` or advancing to M3.
