# Train/evaluation boundary

The benchmark defaults to the paper 17-task / 30-metric roster; `timeclaw18`
is an explicitly selected extension. Official MTBench is evaluation-only;
training always reads the independent loader. Old official train aliases,
legacy task-set entrypoints fail explicitly. Official train/test manifest
generators are not shipped. Source admission runs before workspace rebinding and services.

## What admission enforces

`benchmark.evaluation_datasets` declares the pools that will be evaluated; null
means all eight paper datasets. The list cannot be empty or contain unknown/
duplicate names. Missing or empty declared pools block learning. A smaller experiment
explicitly declares its evaluation datasets rather than claiming all were checked. Download those pools before independently collecting/using data.

The loader rejects directory/file aliases into official roots and identical
observed input windows, including renamed/cross-format copies. It also compares
the complete input-plus-answer time support. Overlapping intervals within one
domain are rejected when they belong to the same source group or the group is
unknown. Only explicitly distinct physical `source_group` identities establish
distinct series for coincident dates; do not invent groups to bypass overlap.
Declared shared `text_source_id` values also reject reused article sources.
Source group/article IDs come from record fields, not opaque filenames. Missing
or unaligned time support cannot be filled with guessed timestamps.

## Memory and checkpoint lineage

Public learning/distillation jobs hold exclusive workspace access until cleanup
and sealing finish. Inference holds shared read access for its whole request/job.
This closes the race between a receipt check and another cooperating job's write;
multiple frozen-inference readers remain allowed. Busy access is an explicit
error, not a reason to adopt a different memory or skip validation.

Before learning, `training_lineage.json` freezes the task roster, ordered training
material (including supervision) and declared visible evaluation inputs. A
workspace containing old learned files without a receipt is rejected. Existing
receipts must match file hashes and source bindings before resume. Changing a
sample under the same index, replacing a checkpoint, adding unknown Notes or
switching the task set or planned evaluation inputs cannot reuse the old learning state.

Learning and standalone distillation mark an operation active and seal their
owned memory/checkpoint files after finalization. Normal failed cleanup is
recorded as interrupted state; matching state can resume learning, but cannot
serve frozen inference until finalized. Force-killed or externally edited state
with a broken seal is rejected rather than silently adopted. Keep old evidence
and explicitly audit/rebuild it in a new workspace if its origin is unknown.

Inference verifies the seal before agent creation and after shutdown, checks the
declared evaluation inputs, disables memory draining, and scores outside the
agent. An empty workspace is ordinary untrained ReAct. No no-memory ablation
entrypoint or lineage-check bypass is provided. Test mode cannot accept `ground_truth`.
Prepared inference/pruning requests in both public task sets are invariant to changes of evaluator
answers and future numeric values at fixed public output lengths/timestamps.

## Evidence and limits

The current core-suite receipt is in VALIDATION.json. Its regressions exercise
partial input/answer-window overlap, missing pools, unknown entities, shared
article IDs, old memory, copied/changed checkpoints, changed training material,
standalone distillation, immutable test memory and exact task-scope parity.
A retained reproduction admits a shifted shared window in the old code and
rejects it after repair. No fallback/model substitution was introduced.

Source identities must be truthful. These checks do not manufacture publication-
version evidence absent from the original news dataset, detect a deliberately
forged provenance declaration, or certify completeness of an arbitrary local
snapshot. Describe given-text offline evaluation accurately; a real prospective
claim needs its own source-time evidence. Quarantine markers remain enforced.
Raw data, old memory and historical failed evidence are preserved outside the
public source tree, never relabeled as a clean run.
