# Changelog

## Prior 0.1.1 source review — 2026-11-17

- Reject malformed, duplicate, missing-dataset and out-of-range explicit sample
  selections instead of silently changing the requested cohort. Validate the
  full learning selection before service startup or lineage mutation.
- Protect the direct-LLM baseline output directory with one OS-backed writer
  owner through cleanup; competing launches cannot overwrite results.
- Keep native execution, tool selection/dropout, memory and scoring unchanged.
- Put Renhe Jiang last in CFF and BibTeX according to the maintainer correction.

## 0.1.2

- Default to the paper's 17 tasks / 30 metrics; restore short weather-indicator
  entrypoints, scalar MSE/MAE and six-field aggregation. Keep `timeclaw18` explicit.
- Bind selected task sets through learning, baselines, inference and sealed lineage.
- Record one valid scored path at exploration termination as single-execution
  evidence, without comparative winner fields or relative tool-rule support.
  Preserve normal multi-branch exploration and reject unscored/zero-path Notes.

## 0.1.1

### Learning and inference

- Preserve native ReAct, comparative exploratory branches, tool dropout and the
  complete Note/MEMORY/TOOLS/SKILLS lifecycle.
- Read training inputs through the independent-corpus loader. Reject official
  aliases, overlapping source support and unknown or modified learned state.
- Freeze memory for inference, score outside the agent and reject test-mode
  supervision. Shared/exclusive workspace access protects complete job lifecycles.
- Honor configured artifact roots and use one trace-path owner. Reject duplicate
  learning aliases and remove unused alternate training wrappers.

### Evaluation and outcomes

- Use one 18-task roster and shared scorer for native inference and the direct
  LLM baseline. Preserve raw precision, nonzero-target MAPE and unambiguous labels.
- Report filtered/all-valid summaries and failed counts. Bind resume to code,
  config, complete evaluation inputs and frozen memory; reject duplicate rows.
- Keep one writer per output run, exact output ownership and admission snapshots
  before the first prediction. Retain failures across interruptions.
- Preserve rejected CSV payloads while returning failure. Correct Sundial's
  forecast-only capability metadata and avoid loading unimplemented operations.

### Installation and maintenance

- Supply secret-free config, a tested Python 3.10 core lock, explicit direct
  dependencies and pinned external data/code identifiers.
- Stage external dependency checkout before publishing its destination.
- Include core CI, critical static checks, source/wheel integrity checks,
  contribution instructions, software citation and issue templates.
- Distribute main-experiment source only. Old learned artifacts, raw datasets,
  weights and auxiliary experiments are excluded. Historical numerical-paper
  reproduction remains separate from software verification.

- Bind declared source metadata to exact official files; recover all117 finance-long
  time correspondences from unique raw historical sequences and verify NOAA
  station identities, preserving all prices/news/labels and native prediction policy.
