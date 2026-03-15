"""Transfer entropy between time series."""

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


def _histogram_entropy(*arrays, bins):
    hist, _ = np.histogramdd(np.column_stack(arrays), bins=bins)
    p = hist / hist.sum()
    p = p[p > 0]
    return -float(np.sum(p * np.log(p)))


def _compute_te(x, y, lag, bins):
    n = len(x)
    y_future = y[lag:]
    y_past = y[:n - lag]
    x_past = x[:n - lag]

    h_yf_yp_xp = _histogram_entropy(y_future, y_past, x_past, bins=bins)
    h_yp_xp = _histogram_entropy(y_past, x_past, bins=bins)
    h_yf_yp = _histogram_entropy(y_future, y_past, bins=bins)
    h_yp = _histogram_entropy(y_past, bins=bins)

    return (h_yf_yp + h_yp_xp) - h_yf_yp_xp - h_yp


class TransferEntropyTool(ClawTool):
    """Compute transfer entropy between time series channels."""

    name: str = "transfer_entropy"
    category: ToolCategory = ToolCategory.RELATION
    description: str = (
        "Compute the transfer entropy between two time series channels, measuring the directed "
        "information flow from one channel to another. Unlike correlation (symmetric), transfer "
        "entropy is asymmetric and can reveal causal directionality. Input: multivariate time "
        "series with at least 2 channels (uses the first two columns). Output: dict with "
        "'transfer_entropy_xy' (X → Y), 'transfer_entropy_yx' (Y → X), and "
        "'dominant_direction' ('X -> Y', 'Y -> X', or 'symmetric'). Use when you need to "
        "determine which channel drives the other, or to quantify information-theoretic causal "
        "influence. Key params: lag (int, default 1) — time lag for the transfer; bins (int, "
        "default 10) — histogram bins for entropy estimation."
    )
    io_type: ToolIOType = ToolIOType.XK_TO_R
    parameters_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "lag": {"type": "integer", "default": 1},
            "bins": {"type": "integer", "default": 10},
        },
        "required": [],
    }
    is_stub: bool = False

    async def _run(self, ts_data: Any, **params: Any) -> Any:
        data = _to_2d(ts_data)
        if data.shape[1] < 2:
            raise ValueError("Need at least 2 columns for transfer entropy")

        lag: int = int(params.get("lag", 1))
        bins: int = int(params.get("bins", 10))

        x = data[:, 0]
        y = data[:, 1]

        te_xy = _compute_te(x, y, lag, bins)
        te_yx = _compute_te(y, x, lag, bins)

        if te_xy > te_yx:
            direction = "X -> Y"
        elif te_yx > te_xy:
            direction = "Y -> X"
        else:
            direction = "symmetric"

        return {
            "transfer_entropy_xy": float(te_xy),
            "transfer_entropy_yx": float(te_yx),
            "dominant_direction": direction,
        }


tool = TransferEntropyTool()
