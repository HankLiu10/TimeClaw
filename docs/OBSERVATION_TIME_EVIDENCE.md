# Observation-time source evidence

An **observation-time evidence attachment** is versioned input metadata: a JSON
file binding complete source files to independently audited time correspondences
and physical-series identities. It contains no forecasting policy, learned
memory, test answers or news text. `benchmark.observation_time_evidence` explicitly
selects it; relative paths resolve under `benchmark.repo_root`.

## Why the pinned finance-long data need it

The upstream sampler selects timestamps by stride, then retrieves prices by
membership in that selected timestamp list. Repeated source timestamps can
therefore expand the prices without expanding the selected timestamp vector.
The raw source has actual repeated observations; interpolating or truncating
would change the task rather than recover its provenance.

A complete search of the 282 pinned raw-stock shards found exactly one original
historical input-series match for each of the 117 affected processed rows.
Selection used observed inputs only. Future prices were checked only after the
unique historical source had been selected; every check matched exactly.
The attachment restores those actual repeated source timestamps, retaining the
same distinct time set and endpoints. It never invents a new time or changes
prices, news, labels, order, horizon length or the original Parquet file.

Weather identities are normalized to `noaa:wban:<five-digit identifier>` from
NOAA station documentation and inventory. Each GHCN station identifier is present
in the actual context as well as the inventory; independent ISD identifiers are
checked against their USAF/WBAN inventory. These fields are physical provenance,
not sample identifiers used as prediction rules.

## Runtime contract

`schema: observation_time_evidence_v1` contains per-dataset source-file hashes,
total row count and entries. Each entry identifies observed inputs plus the
public future-time range, declares its match count and cites raw-source evidence.
It may supply actual `input_timestamps`/`output_timestamps` or a `source_group`.
The loader rejects changed files, unknown/duplicate keys, wrong match counts,
unsupported fields, conflicting source identities, nonfinite/misaligned times
and altered distinct-time support or endpoints. Changes are applied only after
all entries validate. It never catches an invalid attachment and adopts another
one. Datasets outside its explicitly declared scope remain unmodified.

Lookup does not use future numerical values, evaluator labels, filenames or row
indices. Raw row references in the attachment are provenance only. Regression
checks change future answers and reorder/rename records, and check the actual
Parquet loader's input/prompt invariance. The source metadata are not injected as
an agent decision policy; the original ReAct/dropout/memory system is unchanged.

The public source-file checksums and the attachment are part of the release's
input version. All financial long-window task prompts (525 rows times three
families) are exactly unchanged. Do not adopt existing memory with different
source bindings: use a fresh admitted workspace. Raw inputs and rejected old
learning artifacts remain preserved. Raw profile selection does not bypass the
existing strict source gate.

## Primary sources

- [Pinned raw stock dataset](https://huggingface.co/datasets/GGLabYale/MTBench_finance_stock/tree/0c1a656a9611846e76c35a96ec85404829763022)
- [Pinned upstream sampler](https://github.com/Graph-and-Geometric-Learning/MTBench/blob/394a9d274403e93a08b24c2a0b7af461615ce302/data_preparation/finance/utils.py)
- [NOAA station identity documentation](https://www.ncei.noaa.gov/oa/global-historical-climatology-network/hourly/doc/ghcnh_DOCUMENTATION.pdf)

Full raw-recovery helpers and private audit records are not dataset-construction
entrypoints in this main-experiment source. The runtime reads the declared,
reviewed metadata attachment. This does not certify arbitrary forged provenance,
all independent QA labels, historical body-version availability or old paper scores.
