# Data policy

**DO NOT COMMIT RECOLA DATA.** Raw videos/annotations, private metadata, inventories,
source checksum caches, participant manifests, split assignments, and derived dataset
figures remain ignored. No release was available during M2. See
`docs/dataset_card.md` and `docs/recola_inspection_workflow.md` for access blockers and
the local-only inspection process. Review the signed agreement before publishing even
aggregate derived artifacts. The Git data-safety CLI adds a small tracked-file safeguard.

M1 needs no private data. Fixtures are generated deterministically in memory.
Future raw videos, annotations, crops, embeddings, and private manifests must remain
outside version control. Only this file and empty manifest/split directory markers
are allowed by the default data ignore rules. Explicitly allowlist public artifacts
only after reviewing their contents and applicable access agreement.

RECOLA is available by application with a signed EULA; the current access page asks
for a permanent academic's institutional request. Arrange this through a supervisor:
https://recola.human-ist.ch/download.html

No download or dataset-specific parsing is implemented in M1. Source checksums are
required for future auditability where available, and every known participant,
video, session, and dyad must be isolated across experimental partitions.
