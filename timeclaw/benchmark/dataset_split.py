"""Build and load **named sample lists** for official MTBench processed data.

Writes JSON under ``data/splits/<dataset>/`` (``train_files.json``,
``test_files.json``).  Names are historical: this module **time-partitions**
the official processed pool into two filename lists (~70–75% earlier years
vs remainder).  That is **not** the same thing as the project's **disjoint
independent training corpus** under ``data_preparation/independent/``.

- ``load_split(..., "test")`` → standard benchmark evaluation subset.
- ``load_split(..., "train")`` → subset used by default ``train_loop`` on
  **official** ``data/processed/``, not the independent corpus unless you
  rewire paths.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from timeclaw.core.config import TimeClawConfig
from timeclaw.benchmark.mtbench_adapter import (
    DATASET_REGISTRY,
    MTBenchSample,
    benchmark_splits_dir,
    load_dataset,
)

# Legacy default (cwd-relative).  Prefer ``benchmark_splits_dir(config)`` / YAML ``benchmark.splits_dir``.
SPLITS_DIR = Path("data/splits")


def _find_split_year(samples: list[MTBenchSample], target_train_ratio: float = 0.73) -> int:
    """Find the split year that gives closest to target_train_ratio for train."""
    years = sorted(set(s.start_year for s in samples if s.start_year > 0))
    if not years:
        return 2020

    best_year = years[len(years) // 2]
    best_diff = float("inf")

    for y in years:
        n_train = sum(1 for s in samples if s.start_year < y)
        ratio = n_train / len(samples) if samples else 0
        diff = abs(ratio - target_train_ratio)
        if diff < best_diff:
            best_diff = diff
            best_year = y

    return best_year


def split_dataset(
    dataset_name: str,
    split_year: int | None = None,
    sample_ratio: float = 1.0,
    seed: int = 42,
    config: TimeClawConfig | None = None,
) -> dict[str, Any]:
    """Time-partition one official dataset into two filename lists (train/test JSON).

    Args:
        sample_ratio: Fraction of samples to keep in each list (0, 1].
                      Applied *after* the temporal boundary so the early/late
                      partition is unchanged.  A ratio of 0.2 means each
                      list keeps 20 % of its original samples.
        seed: Random seed for reproducible sub-sampling.

    Returns dict with train/test sample lists and metadata.
    """
    samples = load_dataset(dataset_name, config=config)
    if not samples:
        raise ValueError(f"No samples loaded for {dataset_name}")

    if split_year is None:
        split_year = _find_split_year(samples)

    train = [s for s in samples if s.start_year < split_year]
    test = [s for s in samples if s.start_year >= split_year]

    if 0 < sample_ratio < 1.0:
        rng = np.random.RandomState(seed)
        n_train = max(1, int(len(train) * sample_ratio))
        n_test = max(1, int(len(test) * sample_ratio))
        if len(train) > n_train:
            idx = rng.choice(len(train), n_train, replace=False)
            train = [train[i] for i in sorted(idx)]
        if len(test) > n_test:
            idx = rng.choice(len(test), n_test, replace=False)
            test = [test[i] for i in sorted(idx)]
        logger.info(
            f"[Split] {dataset_name}: sub-sampled {sample_ratio:.0%} → "
            f"train {n_train}, test {n_test}"
        )

    if not train:
        logger.warning(
            f"[Split] {dataset_name}: no train samples with year < {split_year}. "
            f"Year range: {min(s.start_year for s in samples)}-{max(s.start_year for s in samples)}"
        )
    if not test:
        logger.warning(
            f"[Split] {dataset_name}: no test samples with year >= {split_year}."
        )

    return {
        "dataset": dataset_name,
        "split_year": split_year,
        "train": train,
        "test": test,
        "train_count": len(train),
        "test_count": len(test),
        "total": len(samples),
        "sample_ratio": sample_ratio,
    }


def save_split(
    dataset_name: str,
    train: list[MTBenchSample],
    test: list[MTBenchSample],
    split_year: int,
    config: TimeClawConfig | None = None,
) -> dict[str, Path]:
    """Persist split file lists to JSON."""
    out = benchmark_splits_dir(config) / dataset_name
    out.mkdir(parents=True, exist_ok=True)

    train_files = [s.file for s in train]
    test_files = [s.file for s in test]

    meta = {
        "dataset": dataset_name,
        "split_year": split_year,
        "train_count": len(train_files),
        "test_count": len(test_files),
        "domain": train[0].domain if train else (test[0].domain if test else "unknown"),
        "task_types": train[0].task_types if train else (test[0].task_types if test else []),
    }

    train_path = out / "train_files.json"
    test_path = out / "test_files.json"
    meta_path = out / "split_meta.json"

    train_path.write_text(json.dumps(train_files, indent=2))
    test_path.write_text(json.dumps(test_files, indent=2))
    meta_path.write_text(json.dumps(meta, indent=2))

    logger.info(
        f"[Split] {dataset_name}: train={len(train_files)}, test={len(test_files)}, "
        f"split_year={split_year}"
    )
    return {"train": train_path, "test": test_path, "meta": meta_path}


def load_split(
    dataset_name: str,
    split: str = "train",
    *,
    config: TimeClawConfig | None = None,
) -> list[MTBenchSample]:
    """Load samples for a given split of a dataset."""
    split_dir = benchmark_splits_dir(config) / dataset_name
    list_path = split_dir / f"{split}_files.json"

    if not list_path.exists():
        raise FileNotFoundError(
            f"{list_path} not found. Run split_all_datasets() first."
        )

    filenames = set(json.loads(list_path.read_text()))
    all_samples = load_dataset(dataset_name, config=config)
    return [s for s in all_samples if s.file in filenames]


def split_all_datasets(
    split_years: dict[str, int] | None = None,
    sample_ratio: float = 1.0,
    config: TimeClawConfig | None = None,
) -> dict[str, dict[str, Any]]:
    """Split all available MTBench datasets.

    Args:
        split_years: Optional per-dataset split year overrides.
        sample_ratio: Fraction of samples to keep per split (0, 1].
    """
    overrides = split_years or {}
    results = {}

    for name in DATASET_REGISTRY:
        try:
            year = overrides.get(name)
            result = split_dataset(
                name, split_year=year, sample_ratio=sample_ratio, config=config,
            )
            save_split(
                name, result["train"], result["test"], result["split_year"], config=config,
            )
            results[name] = {
                "split_year": result["split_year"],
                "train": result["train_count"],
                "test": result["test_count"],
            }
        except Exception as e:
            logger.error(f"[Split] Failed to split {name}: {e}")

    return results


def main() -> None:
    results = split_all_datasets()
    total_train = sum(r["train"] for r in results.values())
    total_test = sum(r["test"] for r in results.values())
    logger.info(f"All splits done: {len(results)} datasets, {total_train} train, {total_test} test")
    for name, r in results.items():
        print(f"  {name}: train={r['train']}, test={r['test']}, split_year={r['split_year']}")


if __name__ == "__main__":
    main()
