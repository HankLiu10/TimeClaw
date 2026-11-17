# Data roles and layout

Read [current data readiness](DATA_READINESS.md) first. The supplied profile
explicitly declares verified source-time metadata for the upstream finance-long
correspondence defect and NOAA station identities. All eight evaluation pools
now pass source checks; complete independent public corpus coverage remains
incomplete.

No raw dataset, article corpus, model weight, test outcome or learned memory is
included in this source release. The pinned MTBench **code** checkout is
separate from `data/processed`, which holds its evaluation **samples**.

Official evaluation data are downloaded by `scripts/download_mtbench.py` into
the registry layout. The eight repository identifiers and immutable revisions
are in `configs/datasets.lock.json`. Use `--list` without downloading, or
`--dataset weather_aligned_short` to select one dataset. Existing downloads are
reused only if their separate `data/download_receipts/` hashes match; unverified
existing inputs are not overwritten. Use a separate `--root` for a fresh copy.
The downloader does not convert official test inputs into independent training.
The registry layout is:

```text
data/processed/weather/aligned_in7days_out1days/*.json
data/processed/weather/aligned_in14days_out3days/*.json
data/processed/weather/QAshort/*.json
data/processed/weather/QAlong/*.json
data/processed/finance/aligned_in7days_out1days/**/*.parquet
data/processed/finance/aligned_in30days_out7days/**/*.parquet
data/processed/finance/QAshort/*.json
data/processed/finance/QAlong/*.json
```

Independent training uses `benchmark.training_root`, resolved under
`benchmark.repo_root`:

```text
data_preparation/independent/splits/
  weather_aligned_short/*.json
  weather_aligned_long/*.json
  weather_qa_short/*.json
  weather_qa_long/*.json
  finance_aligned_short/data/part.parquet
  finance_aligned_short/train_files.json
  finance_aligned_long/data/part.parquet
  finance_aligned_long/train_files.json
  finance_qa_short/*.json
  finance_qa_long/*.json
```

For aligned finance, each task directory contains exactly one materialized
Parquet table. `train_files.json` selects the table's row identifiers in the
form `00000.parquet_row`, `00001.parquet_row`, etc.; these refer to zero-based
rows of that one materialized table and are not indices into
an unrelated upstream source pool. Weather/QA JSON manifests such as
`train_files.json` are metadata and are excluded from sample parsing. Consult
`timeclaw/benchmark/mtbench_adapter.py` for the complete schema and registry.

A minimal weather record has `input_window`, `input_timestamps`, visible `text`,
`output_window` and `output_timestamps`. Weather QA records carry top-level
`question` and `answer`; finance QA uses `MCQA: {"question": ..., "answer": ...}`
and `news_price_correlation` for the correlation label. Aligned finance uses a
`trend` object with `output_bin_label` and a `technical` object with `in_macd`,
`out_macd` and any other supplied indicators (JSON-encoded object columns are
supported). The loader derives normalized labels from these fields. Outputs/answers are training supervision or external test targets,
not online policy features. Use separate source groups/time windows for learning,
development and formal testing; separate directories alone do not prove isolation.

Dataset construction scripts are excluded from this main-experiment release.
Supply independently collected, schema-compatible records with truthful source
metadata. Known target-derived weather text and retrospective-only records are
rejected. In particular, NOAA Storm Events narratives are post-event reports;
removing a quarantine marker does not make them prospective evidence. Finance
QA collection must not consult official test answer-label proportions.

`DATA_QUARANTINE.json` marks known inadmissible datasets or learned ancestry;
their rejection must be resolved by rebuilding admissible evidence. The
maintained checks catch observed disclosures, not all possible leakage. Where
the official dataset lacks a publication-version timestamp, describe given-text
offline evaluation honestly; this package cannot manufacture historical evidence.

## Required source contract

Download the declared evaluation pools before learning. Null
`benchmark.evaluation_datasets` declares all eight paper datasets; a smaller
experiment must explicitly list its planned evaluation datasets. Missing pools
fail rather than skip the comparison. Frozen-memory inference must use the same
evaluation inputs. The benchmark contains the default 17 paper tasks (30 metrics) and an explicit 18-task extension in PAPER_TASKS.md.

Every numeric record needs aligned `input_timestamps` and `output_timestamps`.
All input and answer timestamps participate in source-overlap checks. Optional
`source_group` identifies the actual physical station/asset/series; use a stable,
truthful string, not a sample ID. Optional `text_source_id` identifies a particular
article/source document. JSON fields live at the record top level; Parquet uses
columns with the same names. Without entity identity, overlapping source-time
support is rejected rather than assumed independent. Disjoint time ranges need
no invented entity metadata. See DATA_BOUNDARIES.md for seals and resume behavior.
