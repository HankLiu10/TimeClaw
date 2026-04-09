## Distillation Threshold Backup

Backup time: 2026-04-09 16:42 JST

Current effective numeric distillation thresholds before this change:

- `timeclaw/memory/hierarchical.py`
  - `_DISTILL_L1_TO_L2 = 10`

Important implementation note:

- Under the current codebase, `Notes -> MEMORY` is the only remaining numeric
  thresholded distillation stage.
- `MEMORY -> TOOLS / SKILLS / cross_domain` no longer uses independent numeric
  thresholds. Those stages rebuild when the upstream MEMORY fingerprint changes.
