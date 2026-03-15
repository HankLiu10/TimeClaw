"""Split MTBench datasets into train/test by time (no overlap).

Supports all 8 MTBench datasets (weather + finance).  Each dataset is
split independently based on its own time distribution.  The split
point is chosen so that roughly 70-75 % of samples go to train.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from timeclaw.benchmark.mtbench_adapter import (
    DATASET_REGISTRY,
    MTBenchSample,
    load_dataset,
)

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
) -> dict[str, Any]:
    """Split a single dataset by time.

    Returns dict with train/test sample lists and metadata.
    """
    samples = load_dataset(dataset_name)
    if not samples:
        raise ValueError(f"No samples loaded for {dataset_name}")

    if split_year is None:
        split_year = _find_split_year(samples)

    train = [s for s in samples if s.start_year < split_year]
    test = [s for s in samples if s.start_year >= split_year]

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
    }


def save_split(
    dataset_name: str,
    train: list[MTBenchSample],
    test: list[MTBenchSample],
    split_year: int,
) -> dict[str, Path]:
    """Persist split file lists to JSON."""
    out = SPLITS_DIR / dataset_name
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
) -> list[MTBenchSample]:
    """Load samples for a given split of a dataset."""
    split_dir = SPLITS_DIR / dataset_name
    list_path = split_dir / f"{split}_files.json"

    if not list_path.exists():
        raise FileNotFoundError(
            f"{list_path} not found. Run split_all_datasets() first."
        )

    filenames = set(json.loads(list_path.read_text()))
    all_samples = load_dataset(dataset_name)
    return [s for s in all_samples if s.file in filenames]


def split_all_datasets(
    split_years: dict[str, int] | None = None,
) -> dict[str, dict[str, Any]]:
    """Split all available MTBench datasets.

    Args:
        split_years: Optional per-dataset split year overrides.
    """
    overrides = split_years or {}
    results = {}

    for name in DATASET_REGISTRY:
        try:
            year = overrides.get(name)
            result = split_dataset(name, split_year=year)
            save_split(name, result["train"], result["test"], result["split_year"])
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
