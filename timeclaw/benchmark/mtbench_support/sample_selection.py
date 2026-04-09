"""Helpers for selecting explicit sample subsets by index into ``load_split`` output."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_sample_indices(path: str | Path | None) -> dict[str, list[int]] | None:
    """Load a dataset -> indices-into-loaded-sample-list mapping from JSON."""
    if not path:
        return None
    raw = json.loads(Path(path).read_text())
    return {
        str(ds_name): [int(i) for i in indices]
        for ds_name, indices in raw.items()
        if isinstance(indices, list)
    }


def select_samples_by_index(
    samples: list[Any],
    dataset_name: str,
    sample_indices: dict[str, list[int]] | None,
) -> list[Any]:
    """Apply index-based sample selection for one dataset."""
    if not sample_indices or dataset_name not in sample_indices:
        return samples

    selected: list[Any] = []
    for idx in sample_indices[dataset_name]:
        if 0 <= idx < len(samples):
            selected.append(samples[idx])
    return selected
