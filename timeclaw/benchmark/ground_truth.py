"""Shared MTBench ground-truth normalization helpers.

New definition:
    A numeric ground truth is considered ``repairable`` when it can be
    deterministically restored by linear interpolation over finite points.
    Edge gaps are filled by the nearest finite neighbor, matching NumPy's
    interpolation behavior. Sequences with fewer than two finite points are
    treated as invalid instead of being guessed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class NumericGroundTruthNormalization:
    """Result of normalizing one numeric ground-truth sequence."""

    value: list[float] | None
    valid: bool
    repaired: bool
    reason: str | None = None


def normalize_numeric_ground_truth(value: Any) -> NumericGroundTruthNormalization:
    """Normalize a numeric sequence into a finite 1D float list.

    Valid cases:
    - already all finite
    - contains only NaN-like gaps and has at least two finite points, so
      interpolation is well-defined

    Invalid cases:
    - cannot be parsed as a numeric sequence
    - empty sequence
    - contains +/-inf
    - fewer than two finite points when repair would be required
    """

    try:
        arr = np.asarray(value, dtype=np.float64).ravel()
    except (TypeError, ValueError):
        return NumericGroundTruthNormalization(
            value=None,
            valid=False,
            repaired=False,
            reason="unparseable",
        )

    if arr.size == 0:
        return NumericGroundTruthNormalization(
            value=None,
            valid=False,
            repaired=False,
            reason="empty",
        )

    finite_mask = np.isfinite(arr)
    if bool(finite_mask.all()):
        return NumericGroundTruthNormalization(
            value=arr.tolist(),
            valid=True,
            repaired=False,
        )

    if bool(np.isposinf(arr).any() or np.isneginf(arr).any()):
        return NumericGroundTruthNormalization(
            value=None,
            valid=False,
            repaired=False,
            reason="contains_inf",
        )

    finite_idx = np.flatnonzero(finite_mask)
    if finite_idx.size < 2:
        return NumericGroundTruthNormalization(
            value=None,
            valid=False,
            repaired=False,
            reason="insufficient_finite_points",
        )

    repaired = np.interp(np.arange(arr.size, dtype=np.float64), finite_idx.astype(np.float64), arr[finite_mask])
    return NumericGroundTruthNormalization(
        value=repaired.tolist(),
        valid=True,
        repaired=True,
        reason="linear_interpolation",
    )


def _weather_trend_3way_label(sample: Any) -> str:
    """3-way weather trend: increasing / decreasing / stable.
    
    Based on mean temperature difference between last input day
    and first output day, threshold ±0.5.
    """
    if sample.output_window is None or len(sample.input_window) < 24:
        return "stable"
    last_day_mean = float(np.mean(sample.input_window[-24:]))
    first_pred_day_mean = float(np.mean(sample.output_window[:24]))
    diff = first_pred_day_mean - last_day_mean
    if diff >= 0.5:
        return "increasing"
    elif diff <= -0.5:
        return "decreasing"
    return "stable"


def _weather_trend_past_3way_label(sample: Any) -> str:
    """Official MTBench past-trend label for weather.

    Aligns with the official protocol: compute daily means from the
    input window, fit a linear trend, then threshold the slope at
    +/-0.25.
    """
    if len(sample.input_window) < 24:
        return "stable"
    arr = np.asarray(sample.input_window, dtype=np.float64).ravel()
    usable = (len(arr) // 24) * 24
    if usable < 24:
        return "stable"
    daily_mean = arr[:usable].reshape(usable // 24, 24).mean(axis=1)
    if len(daily_mean) < 2:
        return "stable"
    x = np.arange(len(daily_mean), dtype=np.float64)
    slope = float(np.polyfit(x, daily_mean, 1)[0])
    if slope >= 0.25:
        return "increasing"
    elif slope <= -0.25:
        return "decreasing"
    return "stable"

