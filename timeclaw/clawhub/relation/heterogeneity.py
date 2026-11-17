"""Heterogeneity score between time series."""

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


class HeterogeneityScoreTool(ClawTool):
    """Compute heterogeneity score between time series channels."""

    name: str = "heterogeneity_score"
    category: ToolCategory = ToolCategory.RELATION
    description: str = (
        "Compute a heterogeneity score that measures how different multivariate time series "
        "channels are from each other at each time step, using the mean coefficient of "
        "variation across rows. Input: multivariate time series with at least 2 channels "
        "(2-D array or DataFrame). Output: dict with 'heterogeneity_score' (higher means more "
        "diverse channels), 'channel_means', 'channel_stds', and 'n_channels'. Use when you "
        "need to assess whether channels behave similarly or diverge, e.g., before deciding "
        "to aggregate channels or to flag disparate behavior. No required parameters."
    )
    io_type: ToolIOType = ToolIOType.XK_TO_R
    parameters_schema: dict[str, Any] = {
        "type": "object",
        "properties": {},
        "required": [],
    }
    is_stub: bool = False

    async def _run(self, ts_data: Any, **params: Any) -> Any:
        data = _to_2d(ts_data)
        n_channels = data.shape[1]
        if n_channels < 2:
            raise ValueError("Need at least 2 channels for heterogeneity score")

        channel_means = np.mean(data, axis=0)
        channel_stds = np.std(data, axis=0, ddof=1)

        row_means = np.mean(data, axis=1)
        row_stds = np.std(data, axis=1, ddof=1)
        safe_means = np.where(np.abs(row_means) < 1e-12, 1.0, row_means)
        cv_per_step = np.abs(row_stds / safe_means)
        heterogeneity_score = float(np.mean(cv_per_step))

        return {
            "heterogeneity_score": heterogeneity_score,
            "channel_means": channel_means.tolist(),
            "channel_stds": channel_stds.tolist(),
            "n_channels": n_channels,
        }


tool = HeterogeneityScoreTool()
