"""Helpers for selecting explicit sample subsets by index into the explicitly loaded source list."""

from __future__ import annotations

import json
from pathlib import Path
from numbers import Integral
from typing import Any


def _validate_sample_indices(raw: Any) -> dict[str, list[int]]:
    """An explicit selection is a nonempty dataset -> unique integer list map.

    An absent selection (None) means the full source pool. A supplied selection
    must never become full-pool execution through coercion or missing keys.
    """
    if not isinstance(raw, dict) or not raw:
        raise ValueError("Sample selection must be a nonempty dataset-to-indices mapping")
    for name, indices in raw.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Sample selection dataset names must be nonempty strings")
        if not isinstance(indices, list) or not indices:
            raise ValueError(f"Sample selection for {name!r} must be a nonempty list")
        if any(isinstance(idx, bool) or not isinstance(idx, Integral) or idx < 0 for idx in indices):
            raise ValueError(f"Sample indices for {name!r} must be nonnegative integers, not booleans")
        if len(set(indices)) != len(indices):
            raise ValueError(f"Duplicate sample indices for {name!r} would repeat an experiment row")
    return raw


def load_sample_indices(path: str | Path | None) -> dict[str, list[int]] | None:
    """Read an explicit cohort; only an omitted path means no subset filter."""
    if path is None:
        return None
    return _validate_sample_indices(json.loads(Path(path).read_text()))


def select_samples_by_index(
    samples: list[Any],
    dataset_name: str,
    sample_indices: dict[str, list[int]] | None,
) -> list[Any]:
    """Apply the same strict cohort contract as index-preserving selection."""
    return [sample for _, sample in select_samples_with_original_indices(samples, dataset_name, sample_indices)]


def select_samples_with_original_indices(
    samples: list[Any],
    dataset_name: str,
    sample_indices: dict[str, list[int]] | None,
) -> list[tuple[int, Any]]:
    """Preserve caller order and original source-list indices exactly.

    Invalid, duplicate or out-of-range indices and an omitted dataset key are
    errors. They cannot silently drop requested rows or select a full pool.
    """
    if sample_indices is None:
        return list(enumerate(samples))
    mapping = _validate_sample_indices(sample_indices)
    if dataset_name not in mapping:
        raise ValueError(f"Explicit sample selection is missing dataset {dataset_name!r}")
    indices = mapping[dataset_name]
    if any(idx >= len(samples) for idx in indices):
        raise ValueError(f"Sample index out of range for {dataset_name!r}: source has {len(samples)} rows")
    return [(idx, samples[idx]) for idx in indices]


def select_samples_with_text_char_cap(
    samples_with_indices: list[tuple[int, Any]],
    *,
    max_samples: int | None,
    text_char_cap: int,
    lookahead_extra: int = 0,
) -> list[tuple[int, Any]]:
    """Filter oversized-text samples while allowing bounded lookahead.

    New definition:
        When callers want the first ``max_samples`` usable items, we inspect at
        most the first ``max_samples + lookahead_extra`` candidates, drop any
        sample whose ``text`` length exceeds ``text_char_cap``, then keep the
        earliest surviving ``max_samples``. This matches the policy "skip very
        long texts, but only refill from a limited lookahead window instead of
        scanning arbitrarily far into the split".
    """
    if text_char_cap <= 0:
        return list(samples_with_indices[:max_samples]) if max_samples is not None else list(samples_with_indices)

    horizon = len(samples_with_indices)
    if max_samples is not None:
        horizon = min(horizon, max(0, int(max_samples)) + max(0, int(lookahead_extra)))
    window = list(samples_with_indices[:horizon])
    filtered = [
        (idx, sample)
        for idx, sample in window
        if len(str(getattr(sample, "text", "") or "")) <= int(text_char_cap)
    ]
    if max_samples is not None:
        return filtered[: max(0, int(max_samples))]
    return filtered
