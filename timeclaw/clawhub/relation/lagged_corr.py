"""Lagged correlation between time series."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from timeclaw.clawhub.tool_base import (
    ClawTool,
    ToolCategory,
    ToolIOType,
)


def _to_2d(ts_data):
    if ts_data is None:
        raise ValueError("ts_data is required")
    if isinstance(ts_data, pd.DataFrame):
        return ts_data.values.astype(float)
    arr = np.asarray(ts_data, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


class LaggedCorrelationTool(ClawTool):
    """Compute lagged correlation between channels."""

    name: str = "lagged_correlation"
    category: ToolCategory = ToolCategory.RELATION
    description: str = (
        "Compute the lagged cross-correlation between two time series channels, revealing the "
        "time delay at which they are most strongly correlated. Input: multivariate time series "
        "with at least 2 channels (uses the first two columns). Output: dict with 'lags' "
        "(array from -max_lag to +max_lag), 'correlations' (correlation at each lag), "
        "'best_lag' (lag with strongest absolute correlation), and 'best_correlation'. Use when "
        "you need to find the time offset between two related signals, detect lead-lag "
        "relationships, or align channels that are shifted in time. Key param: max_lag (int, "
        "default 20) — maximum number of time steps to shift in each direction."
    )
    io_type: ToolIOType = ToolIOType.XK_TO_R
    parameters_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "max_lag": {"type": "integer", "default": 20},
        },
        "required": [],
    }
    is_stub: bool = False

    async def _run(self, ts_data: Any, **params: Any) -> Any:
        data = _to_2d(ts_data)
        if data.shape[1] < 2:
            raise ValueError("Need at least 2 columns for lagged correlation")

        max_lag: int = int(params.get("max_lag", 20))
        x = data[:, 0]
        y = data[:, 1]

        n = len(x)
        x_centered = x - x.mean()
        y_centered = y - y.mean()
        norm = np.sqrt(np.sum(x_centered ** 2) * np.sum(y_centered ** 2))
        if norm == 0:
            norm = 1.0

        lags = list(range(-max_lag, max_lag + 1))
        correlations = []
        for lag in lags:
            if lag >= 0:
                c = np.sum(x_centered[:n - lag] * y_centered[lag:]) / norm
            else:
                c = np.sum(x_centered[-lag:] * y_centered[:n + lag]) / norm
            correlations.append(float(c))

        best_idx = int(np.argmax(np.abs(correlations)))
        return {
            "lags": lags,
            "correlations": correlations,
            "best_lag": lags[best_idx],
            "best_correlation": correlations[best_idx],
        }


tool = LaggedCorrelationTool()
