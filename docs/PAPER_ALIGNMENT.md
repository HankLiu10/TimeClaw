# Paper alignment audit

Version 0.1.2 defaults to the paper **17-task / 30-metric** inventory. A code audit
checks implementation and protocol; it does not establish the correctness of
historical paper numbers or substitute for a full controlled reproduction.

| Paper mechanism / requirement | Release implementation | Verification / limit |
|---|---|---|
| Explore → Compare → Distill → Reinject | Original Executor, training evaluator and hierarchical memory | Native lifecycle regressions plus a prior small real API chain |
| Exploratory evidence | Normal comparative contract seeks two valid scored branches; terminal one-path evidence is explicitly non-comparative | Main-loop, score-binding, Note and distillation regressions; zero/unscored evidence cannot create single-execution Notes |
| Task-aware tool dropout | `(1+n_min)/(1+n_i)` raised to alpha; protected tools retained | Probability/monotonicity/protected-tool and answer-capability regressions |
| Hierarchical experience outside weights | Notes → scoped MEMORY → tool/skill guidance | Batch trigger, final draining and derived-state tests; no model-weight training |
| Frozen inference without test learning | Native run_inference, test-mode supervision rejection, immutable seals | Both public task-set preparations are invariant to answers; actual CLI seal checks before/after execution |
| Independent exploration corpus | Separate loader and declared source admission | Alias/copy/partial-window/article-source and lineage regressions |
| Finance/weather short/long tasks | One versioned registry; default 17 paper tasks | Both entrypoints and public sample API use it; current task preparation checked offline, including weather indicators |
| Main metrics | Shared numerical/label scorer and summaries | 30 primary cells supported; arithmetic and threshold boundary tests against pinned upstream code |

The delivery scope is the repaired main-experiment **source**, not an old
training-product bundle. Historical checkpoints, Notes, memory, weights and
predictions are excluded and are not prerequisites for source delivery. The
items below concern numerical/paper claims separately; they remain recorded
and do not require distributing or relabeling those old products.

## Public paper and release scope

The canonical paper is [TimeClaw: A Time-Series AI Agent with Exploratory
Execution Learning](https://arxiv.org/abs/2605.10038), by Hangchen Liu, Dongyuan Li,
Jiewen Deng, Weiwei Ye, Yoshihide Sekimoto and Renhe Jiang. The release citation
uses the maintainer's latest author-order correction (Renhe Jiang last); the
public v1 snapshot may retain its earlier author order. The arXiv v1 record
was submitted on 2026-05-11 and is labelled "Under review". Its Section 5.1 and
Appendix B.3 explicitly define 17 tasks / 30 metric views.

| Scope | Public arXiv v1 | This repaired release |
|---|---|---|
| Finance tasks | 10 | 10 |
| Weather forecast | 2 | 2 |
| Weather indicator | 1 | 1 |
| Weather future/past trend | 2, short settings | 2, short settings |
| Weather MCQA | 2 | 2 |
| Total | 17 tasks / 30 metric views | 17 tasks / 30 metric cells |

The previous `timeclaw18` inventory is retained as an explicit extension only.
Single-execution learning now records one actual scored path after a multi-branch
attempt ends at its budget, without selecting a comparative winner. Its Notes
are not winner-frequency evidence and cannot support relative tool rules.
Zero valid paths still record failure evidence; no usable prediction is invented.
A complete independent-learning-corpus download is not supplied. Source-time
attachments retain original values, news and labels; see OBSERVATION_TIME_EVIDENCE.md.
These remain limits for full numerical reproduction.

## Items not certified as paper-consistent

1. The 17-task / 30-cell inventory now matches public v1; the alternative
   18-task / 26-cell table is explicitly an extension. Historical tables below
   remain archived evidence, not results certified by this source update.
2. Read-only cell tracing confirms that both model rows in the archived
   17-task/30-metric main manuscript match the processed CSV: all 30 TimeClaw
   cells match `timeclaw_merged`, and all 30 GPT-5.4 cells match
   `mtbench_planned`. Both archived 18-task/26-metric table versions likewise
   match those two columns in all 26 cells. The generator reads these fields;
   preparation code selects better runs using ground truth and plans baseline
   row replacements. This is more than the presence of unused scripts. It is
   evidence about local archived tables, not a verified external final manuscript.

   The July arithmetic reanalysis was independently reproduced from 22,884
   saved predictions: all 30 means and effective sample counts agree with its
   legacy scorer. However, its input inherited earlier score-conditioned
   deduplication. The earlier 18,860 attempts contain 5,031 duplicate groups;
   938 reported predictions differ from the first attempt, including 216 with
   lower numeric error and 170 changed from incorrect to correct. These counts
   span the historical 20 internal scopes, not just the public paper inventory.
   Recomputing selected inputs does not establish an unselected fixed run.
   Neither selection procedure is implemented in this release.
3. Historical model settings can now be identified: the archived config SHA
   matches the recorded run fingerprint; its six roles use `openai/gpt-5.4`,
   medium effort, max_tokens 4096 and timeout 600 seconds. Missing config fields
   do not mean explicitly disabled capabilities. The public default is a
   software profile, not this historical environment; credentials are not
   distributed. Exact admissible training inputs, frozen paper memory, cohort
   and optional-model environments remain uncertified or undistributed.

   Independent data directories exist, but the saved training state has no
   lineage receipt. Across 12 JSON task scopes, 3,608 training references name
   files found in the official pool and absent from the independent pool;
   the archived training loader also routes `train` to that official pool.
   Filename evidence alone is not a full record-content comparison. Finance
   Parquet virtual row IDs alone cannot prove content overlap. One current
   training checkpoint and the training manifest are later engineering fixtures,
   not a complete paper snapshot. The release's corrected independent loader
   cannot retroactively certify historical learning or repair historical scores.
4. Single-execution evidence is now explicitly supported after a multi-branch
   attempt ends at its budget: one real prediction and score, no comparative winner.
   Zero valid candidates remain failure records and cannot fabricate a normal Note.
   These offline protocol checks do not establish improved population accuracy.
5. Upstream has score filters and a weather-long prompt/evaluator discrepancy.
   The release discloses filtered versus all-valid summaries, target-gap
   normalization and stricter label admission. See SCORING.md. Identical task
   names or prompts alone do not establish identical scoring.

The original archived code, manuscript and historical evidence are preserved.
This audit changes the isolated release source only. It does not replace old
scores with new values or implement an oracle to make outputs match a table.
