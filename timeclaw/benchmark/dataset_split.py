"""Evaluation-only compatibility API for official MTBench data.

The official pool is exclusively for evaluation. There is no official training
alias or generated train manifest. Independent learning uses the separate
``mtbench_adapter.load_independent_dataset`` owner.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from timeclaw.core.config import TimeClawConfig
from timeclaw.benchmark.mtbench_adapter import MTBenchSample, load_dataset

SPLITS_DIR = Path("data/splits")


def load_split(dataset_name: str, split: str = "train", *,
               config: TimeClawConfig | None = None) -> list[MTBenchSample]:
    """Read only explicit evaluation aliases; old default/train calls fail."""
    normalized = str(split or "").strip().lower()
    if normalized == "train":
        raise ValueError("Official MTBench is evaluation-only; use load_independent_dataset for independent training.")
    if normalized not in {"test", "all", "full"}:
        raise ValueError(f"Unsupported evaluation alias: {split!r}; expected test, all or full.")
    return load_dataset(dataset_name, config=config)








