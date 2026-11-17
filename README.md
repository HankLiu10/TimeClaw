# TimeClaw

**A Time-Series AI Agent with Exploratory Execution Learning**

[Paper (arXiv)](https://arxiv.org/abs/2605.10038) ·
[Code](https://github.com/HankLiu10/TimeClaw)

TimeClaw combines time-series tools and language-model reasoning with reusable
experience. A native ReAct controller chooses tools, executes them and reads the
observations. During learning it compares executed candidate branches, records
Notes and distills hierarchical memory. Inference reuses frozen memory.
This learning updates experience files, not model weights.

The default benchmark is **`paper30_original`: 17 tasks / 30 metric cells**,
matching the main tables of arXiv v1. A metric cell means one reported score for
one task; for example, weather indicators report six scores, not six tasks.

| Domain | Task settings | Count |
|---|---|---:|
| Finance | Forecast, MACD, trend, news–price correlation, MCQA; each short/long | 10 |
| Weather | Forecast and MCQA short/long; indicator, future trend and past trend short | 7 |

The earlier `timeclaw18` software inventory remains an explicit extension
(18 tasks / 26 cells): it omits weather indicators and adds long future/past
weather trends. Set `benchmark.task_set` in the config for learning and baselines;
inference also accepts `--task-set`. Every stage must use the same selection.
Changing task set requires a fresh workspace/output; existing receipts are not relabeled.
Task/metric alignment does not establish reproduction of historical scores;
see [paper alignment](docs/PAPER_ALIGNMENT.md).

[Task definitions](docs/PAPER_TASKS.md) · [Architecture](docs/ARCHITECTURE.md) ·
[Scoring](docs/SCORING.md) · [Validation](docs/RELEASE_STATUS.md)

## Installation

The verified core environment is **Linux / Python 3.10**. From the source root:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-py310.lock
python -m pip install -e '.[dev]'
python -m pip check
python scripts/setup_mtbench.py
```

The setup helper installs the pinned external MTBench prompt/metric code at
revision `394a9d274403e93a08b24c2a0b7af461615ce302`; it does not download datasets.
For a custom installation, use `--destination PATH` and set
`TIMECLAW_MTBENCH_ROOT=PATH`. See [third-party notices](THIRD_PARTY_NOTICES.txt).
Benchmark setup and run commands are distributed in the source checkout;
the wheel contains the agent library.

The core environment includes statistical tools. Neural forecasting models need
separate dependencies and downloaded weights: see [model setup](docs/MODELS.md).
Listing a tool does not certify that its optional model environment is installed.

The following checks require no model credentials or weights:

```bash
timeclaw tools --list
python -m ruff check timeclaw scripts tests --select E9,F821,F822,F823
LITELLM_LOCAL_MODEL_COST_MAP=True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python -m pytest -q --timeout=20
```

## Data and configuration

**Verified source metadata:** the supplied profile explicitly loads
`configs/mtbench_observation_times.json`. It restores the actual observation-time
correspondence for 117 finance-long records, using unique exact matches to the
pinned raw series, and supplies NOAA-verified weather station identities.
The original data files, price/temperature values, news and labels remain
unchanged. All 7,573 evaluation records pass the source-time checks under this
profile. See [the evidence protocol](docs/OBSERVATION_TIME_EVIDENCE.md) and
[data readiness](docs/DATA_READINESS.md).

Download the official evaluation inputs before learning:

```bash
python scripts/download_mtbench.py --list
python scripts/download_mtbench.py
```

Eight dataset repositories and immutable revisions are listed in
`configs/datasets.lock.json`. Download receipts record hashes for verified reuse.
A smaller download can use `--dataset weather_aligned_short`.

Learning requires **separately sourced independent training data** under
`data_preparation/independent/splits/<dataset>/`. Independent training data and
collection scripts are not included in this source distribution; prepare records
according to [the data schema](docs/DATA.md). A portable 160-record
independent review bundle has been prepared separately with this source delivery;
it covers four dataset pools, not every task. There is no verified public hosted
URL for the complete independent corpus, so full historical paper learning cannot
yet be reproduced from public downloads alone. The loader never substitutes the
official evaluation pool for missing training data. Source paths, observed
windows, complete source-time support and declared article sources are checked
before learning. See [train/evaluation boundaries](docs/DATA_BOUNDARIES.md).

Create a local config and provide your credential through the process environment:

```bash
cp configs/default.yaml configs/local.yaml
export TIMECLAW_CONFIG="$PWD/configs/local.yaml"
export OPENAI_API_KEY='YOUR_OWN_KEY'
```

Configure your provider/model and optional services in `configs/local.yaml`.
`benchmark.observation_time_evidence` is an explicit source-metadata attachment,
resolved under `benchmark.repo_root` unless absolute. Keep it available when using
a custom data root. Setting it to null selects unmodified upstream metadata; that
raw view retains the upstream source-time defects and fails full-source learning
admission. There is no automatic repair or dataset exclusion on failure.
The supplied profile uses `gpt-6.1-sol`/low, no automatic model substitution,
no output-token cap, a 3600-second attempt timeout and transient-error retries
every 30 seconds. Quota failures and content refusals remain errors. Model
requests may incur provider charges. `.env.example` is a documentation template;
`.env` files are not loaded automatically.

`benchmark.repo_root` contains the inputs; `benchmark.training_root` is resolved
under it. By default all eight evaluation pools are declared. A deliberately
smaller experiment must explicitly set `benchmark.evaluation_datasets`; missing
or empty declared pools are errors. Keep those evaluation inputs fixed when
using the learned memory. Workspace priority is explicit `--workspace-dir`,
then `TIMECLAW_WORKSPACE_DIR`, then `workspace_dir` in the config.

## Learn and infer

For a small run on independently collected weather data:

```bash
python -m timeclaw.benchmark.run_learning --train \
  --dataset weather_aligned_short --task trend_past --max-samples 2 \
  --concurrency 1 --per-task-concurrency 1

python scripts/run_timeclaw_mtbench_inference.py \
  --config configs/local.yaml --task-set paper30_original \
  --task weather_trend_past_short --samples-per-task 2 \
  --per-task-concurrency 1 --llm-max-inflight-requests 1 \
  --output-root results/main --run-id weather_past_short
```

A **Note** stores one training experience. **MEMORY** stores distilled rules;
TOOLS/SKILLS guidance is derived from that store. Exploration seeks at least
two valid, executed and scored branches. If the exploration budget ends after
attempting multiple branches and only one valid path survives, its actual score
can support a **single-execution Note** (`single_execution`), without a winner or
comparative signal. An unscored/zero-candidate run cannot create such a Note.
For example, one failed branch plus one forecast with measured MAE can record
that forecast and MAE, but cannot establish that its tool beats the failed one.
These Notes are excluded from winner counts and relative tool-rule support.
Ten new scoped Notes
trigger batch distillation; training finalization drains the remainder. Thirty
controls history snapshots. Training already includes finalization; `--all` is
its alias and cannot be combined with `--train` or `--distill`. Standalone
`--distill` explicitly refreshes an existing admitted learning workspace.

An explicit `--sample-index-file` must map every requested dataset to a nonempty
list of unique, nonnegative integer indices into the loaded source list. Missing
dataset keys, booleans, strings, duplicates and out-of-range indices are errors;
they never mean "run everything" or "drop the invalid entries". For example,
`{"weather_aligned_short": [2, 0]}` selects original indices 2 then 0. Training
validates this selection before creating services or changing learning lineage.

When `--max-samples` is used, the existing training policy additionally excludes
articles longer than 70,000 characters and considers at most the first
`max_samples + 300` candidates to fill the limit. This is a text-length policy,
not a score-based choice; it can leave fewer rows and is reported in the log.
Use the realized sample manifest to identify the actual learning cohort.

For the full task set, remove training dataset/task filters, then remove the
inference task filter and use `--samples-per-task 0`. Only do so after preparing
all intended independent corpora and evaluation pools.

`training_lineage.json` binds learning inputs, evaluation inputs, task definitions
and hashes of learned files. Keep it with copied memory. Inference checks that
seal before execution and after cleanup, and cannot learn from test answers.
An empty workspace runs untrained native ReAct; learned-agent results require
the intended finalized learning workspace.

One learning/distillation job owns write access to a workspace. Frozen-inference
jobs may read it concurrently; learning cannot modify it while they run. A busy
workspace is reported before model startup. Use separate processes for separate
workspaces: artifact roots are bound per process.

Each output run has one writer. `--resume` requires matching code, config, full
evaluation content/targets, task selection and frozen memory. Existing recorded
failures remain outcomes; duplicate or foreign rows are rejected. A fresh launch
refuses an existing run. `--clear-output` explicitly removes only that run's
outputs and cannot be combined with `--resume`; preserve old evidence first.
Failed or empty batches exit nonzero after retaining their records.

## Baseline and evaluation

The direct-LLM baseline receives the same task's visible prompt and uses the
same scorer as TimeClaw, with one direct model call per sample/task:

```bash
python -m timeclaw.benchmark.baseline_eval \
  --config configs/local.yaml --dataset weather_aligned_short --task trend_past \
  --max-samples 2 --concurrency 1 --output-dir results/direct_baseline
```

One direct-baseline job owns its output directory through admission, requests,
and shutdown. A competing launch on the same directory fails before another
backend is created; use distinct output directories for separate concurrent runs.

Use the same frozen samples for both methods. Both preserve raw predictions and
report upstream threshold-filtered and all-valid summaries, plus failure and
exclusion counts. No score-based choice between different runs is implemented.
Read [the scoring protocol](docs/SCORING.md), including the documented
weather-long prompt/evaluator discrepancy, before comparing results.

This distribution contains main-experiment code and regression tests. Raw data,
old checkpoints, learned memory, experimental outputs and weights are not
included. Software validation is separate from exact numerical reproduction of
historical tables. The implementation/evidence limits are recorded in
[paper alignment](docs/PAPER_ALIGNMENT.md) and [release validation](docs/RELEASE_STATUS.md).

## Citation, contribution and license

This is the TimeClaw repository at
[HankLiu10/TimeClaw](https://github.com/HankLiu10/TimeClaw).
[CITATION.cff](CITATION.cff) identifies the paper and follows the maintainer's
latest author-order correction, with Renhe Jiang last. The cited arXiv v1 record
may retain its earlier author order and labels the manuscript "Under review".

```bibtex
@article{liu2026timeclaw,
  title={TimeClaw: A Time-Series AI Agent with Exploratory Execution Learning},
  author={Liu, Hangchen and Li, Dongyuan and Deng, Jiewen and Ye, Weiwei and Sekimoto, Yoshihide and Jiang, Renhe},
  journal={arXiv preprint arXiv:2605.10038},
  year={2026},
  doi={10.48550/arXiv.2605.10038}
}
```

Contributions and bug reports are welcome through [the contribution guide](CONTRIBUTING.md) and
[GitHub Issues](https://github.com/HankLiu10/TimeClaw/issues).

Project code uses the [MIT license](LICENSE). External datasets, model weights
and dependencies retain their own terms; see [third-party notices](THIRD_PARTY_NOTICES.txt).
