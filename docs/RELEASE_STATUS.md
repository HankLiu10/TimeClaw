# Current source protocol — version 0.1.2

**Publication update — 2026-11-17:** the repaired 0.1.2 source is now public in
[HankLiu10/TimeClaw](https://github.com/HankLiu10/TimeClaw). All 297 files in the
initial source publication were independently fetched and matched against the
reviewed source hashes; its
[GitHub Actions run](https://github.com/HankLiu10/TimeClaw/actions/runs/44044567643)
passed. Earlier statements below about pending publication describe historical
audit stages, not the current repository. The later README refresh changes
documentation and its file manifest only; the validated runtime is unchanged.

The default task inventory now matches the public paper: `paper30_original`,
17 tasks / 30 primary metric cells. Weather max/min/diff prediction and all six
MSE/MAE summaries are included. `timeclaw18` remains an explicitly named extension;
learning, inference and baseline identities cannot mix task sets.

Single-execution learning is now supported after multiple branches were attempted
and the exploration budget ends with one valid scored path. It records observed
tools/output/score, without winner fields or comparative evidence. Single Notes
count toward the existing ten-Note distillation trigger, but cannot add support
to relative tool-selection/failure rules, including by merging existing rules.
Zero candidates, missing scores, unsupported labels and contradictory scores
cannot produce single-execution Notes. Normal comparative exploration remains.

Current source-package validation: 1,223 passed / 4 optional-package skips / zero
failures after independent extraction. Installed-wheel imports, all 17 task
entrypoints, a real four-step CPU ETS prediction and dependency compatibility
also pass.
See VALIDATION.json for the final distribution receipt. Tests use scripted model
responses and blocked external networking; this update made zero paid requests,
used no GPU, and does not certify numerical paper reproduction or every optional
neural model environment.

The following sections retain earlier 0.1.1 audit details. Their statements that
the task registry or native core was unchanged describe those earlier revisions;
they are superseded by the 0.1.2 protocol above.

---

# TimeClaw1 source delivery verification — 2026-11-17

**The repaired main-experiment source passes the documented software checks.**
Official publication readiness is a separate assessment: the public-paper task
scope, independent-corpus availability and existing remote history remain open. This acceptance concerns source delivery, not exact historical paper
numbers. Old checkpoints, Notes, learned memory, predictions and weights are
excluded; providing them is not a condition for this source package. Historical
paper-provenance findings remain separately recorded in PAPER_ALIGNMENT.md.
No repaired source upload has been performed. The later authorized cleanup
replaced normal remote branch history while preserving a complete private backup.

## Prior 0.1.1 verification

- Linux / isolated Python 3.10.9, without host system-site packages:
  **1187 passed, 4 skipped, 0 failed**. Skips are four uninstalled optional
  signal packages. Test networking is blocked except loopback fixtures.
- One shared roster owns all 18 tasks and 26 primary metric columns. All 18
  inference preparations reject training supervision and are invariant to
  changed evaluator answers/future numeric values at fixed public horizons.
  Independent-source isolation, sealed learning, immutable inference and exact
  result identity remain enforced by actual loader/CLI regressions.
- Workspace selection now honors explicit CLI > environment > config. Omitted
  CLI flags cannot silently redirect training or standalone distillation to
  `.timeclaw`. `--all` with `--train`/`--distill` is rejected before execution.
- Creation, evidence reads and Note trace references share the trace module's
  configured path owner. Tests put conflicting same-ID traces in two directories
  and verify the current workspace alone is used; switching workspaces cannot
  adopt the previous trace. Test-process artifact bindings are restored per case.
- Executor, executor prompt, tool dropout and PipelineRunner remain byte-identical
  to the prior accepted source. In hierarchical memory, only `Notes._source_lines`
  changes its trace path reference; the rest of its AST is identical. No new
  selection policy, fallback or model substitution is introduced.
- Result files enforce one writer/run and one row per frozen sample identity.
  Recovery retains failures, reconstructs failure sidecars from authoritative
  rows and saves admission before the first prediction. Exact ownership prevents
  clearing `main` from deleting a different `main_retry` run.

- Shared reader/exclusive writer OS access now covers the entire learning,
  distillation or inference job, including cleanup. Eight actual-entrypoint
  regressions expose the former race and verify exclusion, reader sharing and
  release after failure. No change to prediction/selection policy.

## Final submission repairs and independent setup checks

- Removed three unused alternate sample-batch training helpers that referenced
  absent text-preparation functions. Active per-task training and the complete
  native controller/memory remain unchanged. Critical static checks report no
  undefined names/syntax errors across runtime, scripts and tests, and run in CI.
- CSV quality rejection, missing prediction or returned execution error cannot
  exit successfully; the result and cleanup are preserved. Genuine zero output
  is admitted. Sundial no longer advertises unsupported imputation/anomaly tasks
  or loads weights for an unimplemented operation.
- Declare direct aiohttp/requests dependencies. Actual pinned MTBench setup
  succeeded from an empty directory; failed-fetch/reuse/dirty-checkout tests use
  local Git origins. Public metadata access verified all eight pinned dataset
  revisions/file counts without credentials or raw-data downloads. This is not
  a full new dataset/agent benchmark.

## Scope and supporting checks

Dataset construction, synthetic demonstrations, ablations, case studies,
unrelated toy runners and TimeClaw2 research code are excluded. Main runtime,
exploration/dropout, complete hierarchical memory, reusable tools/model adapters,
main baseline, scorer, setup/download helpers and regression tests remain.

Prior read-only input checks loaded 7,573 physical rows across eight local
archived evaluation datasets and prepared all 18 tasks without model predictions.
The prior small live native-core check used artificial fixtures: two training
samples/four branches, two Notes, finalized memory and one unique inference
sample across two successful executions. Its retained failures and 54 total
transport attempts are historical engineering evidence. Current workspace/trace
repairs are verified offline, not a new live or full-benchmark result.

This source-delivery audit made **zero model requests**, used no GPU and did not
train or package any weights. Final source archive/wheel content and extraction
checks are recorded separately alongside the release artifacts.

## Explicit limits

Source-delivery acceptance does not certify every optional model's GPU loading,
upstream environment or population-level accuracy. Historical table numerical
reproduction is not accepted and is outside this delivery's acceptance criteria.
Use independently admitted inputs and a fresh workspace for new learning.

Building a clean source archive does not itself remove old remote history.
The separately authorized normal-branch cleanup is described below; repaired
source publication remains a separate action and has not been performed.

## Official publication prerequisites

The canonical [public paper](https://arxiv.org/abs/2605.10038) is verified and
CITATION.cff contains its actual authors and DOI. Its v1 scope is 17 tasks /
30 metric views, whereas this release follows 18 / 26. See PAPER_ALIGNMENT.md.
No public independent-training-corpus download has been verified.

Authorized cleanup now removed the old development branch and replaced master
with a standalone root preserving the same README. Git fetch and GitHub REST
readbacks confirm the change. The managed PR #1 reference still retains
credential-style literals, and revocation/rotation has not been confirmed.
A support draft is prepared but has not been sent; credentials were not used or
validity-tested. A clean normal branch is not complete removal from GitHub.

The source-time defect in 117 raw finance-long rows is now resolved through an
explicit, audited metadata attachment, with all numeric values/news/labels and
all 1,575 financial long-window task prompts preserved. All eight actual pools
and 7,573 records pass source checks. The 160-record portable review subset is
admitted against all eight pools; source-group identity is proven, not guessed.
See DATA_READINESS.md and OBSERVATION_TIME_EVIDENCE.md. Full independent-corpus
coverage/public hosting remains separate from software delivery.

This follow-up adds one generic metadata reader, an explicit config field and
13 actual-entrypoint regressions. The complete current source suite passes
1187 tests / 4 optional skips / zero failures. Original Executor, dropout,
PipelineRunner, hierarchical memory, task roster and scoring are unchanged.
The source-metadata helpers/search and raw inputs are not distributed as
construction/ablation/case-study code in the main source package.

## 2026-11-17 repeated source review

The prior 1,159 checks passed again before changes. New counterexamples exposed
silent cohort changes in explicit sample selection and competing baseline output
writers. The repaired selection rejects invalid/missing/duplicate indices, and
learning admits its requested cohort before service startup or lineage mutation.
The baseline holds an OS-backed exclusive output owner through cleanup. A real
child-process regression verifies exclusion and release after failure.

The complete suite now passes 1,187 checks with the same four optional signal
skips; 28 checks were added. Native controller, dropout, hierarchical memory,
scorers, source-isolation guards and the 18-task roster are unchanged. Citation
order follows the maintainer correction, with Renhe Jiang last. This review made
no model requests and used no GPU. Packaging verification is recorded with the
new archive, not inferred from the older archive acceptance.
