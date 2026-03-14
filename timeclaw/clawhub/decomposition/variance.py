"""Variance decomposition."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from timeclaw.clawhub.tool_base import (
    ClawTool,
    ToolCategory,
    ToolIOType,
)


def _to_array(ts_data):
    if ts_data is None:
        raise ValueError("ts_data is required")
    if isinstance(ts_data, pd.DataFrame):
        arr = ts_data.iloc[:, 0].values
    elif isinstance(ts_data, pd.Series):
        arr = ts_data.values
    else:
        arr = np.asarray(ts_data).ravel()
    return arr.astype(float)


class VarianceDecomposeTool(ClawTool):
    """Variance decomposition of time series."""

    name: str = "variance_decompose"
    category: ToolCategory = ToolCategory.DECOMPOSITION
    description: str = "Decompose variance contributions (e.g., between-group vs within-group)."
    io_type: ToolIOType = ToolIOType.X_TO_R
    parameters_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "window_size": {"type": "integer", "default": None},
        },
        "required": [],
    }
    is_stub: bool = False

    async def _run(self, ts_data: Any, **params: Any) -> Any:
        if ts_data is None:
            raise ValueError("ts_data is required")

        if isinstance(ts_data, pd.DataFrame) and ts_data.shape[1] > 1:
            return self._multivariate(ts_data)

        arr = _to_array(ts_data)
        window_size = params.get("window_size") or max(len(arr) // 10, 2)
        return self._univariate(arr, window_size)

    @staticmethod
    def _multivariate(df: pd.DataFrame) -> dict[str, Any]:
        data = df.values.astype(float)
        grand_mean = np.mean(data)
        total_variance = float(np.var(data, ddof=0))

        channel_means = np.mean(data, axis=0)
        n_rows = data.shape[0]
        between_variance = float(np.sum((channel_means - grand_mean) ** 2 * n_rows) / data.size)
        within_variance = float(total_variance - between_variance)
        ratio = float(between_variance / total_variance) if total_variance > 0 else 0.0

        return {
            "total_variance": total_variance,
            "between_variance": between_variance,
            "within_variance": within_variance,
            "variance_ratio": ratio,
        }

    @staticmethod
    def _univariate(arr: np.ndarray, window_size: int) -> dict[str, Any]:
        n = len(arr)
        total_variance = float(np.var(arr, ddof=0))

        kernel = np.ones(window_size) / window_size
        trend = np.convolve(arr, kernel, mode="same")
        residual = arr - trend

        trend_variance = float(np.var(trend, ddof=0))
        residual_variance = float(np.var(residual, ddof=0))
        ratio = float(trend_variance / total_variance) if total_variance > 0 else 0.0

        return {
            "total_variance": total_variance,
            "between_variance": trend_variance,
            "within_variance": residual_variance,
            "variance_ratio": ratio,
        }


tool = VarianceDecomposeTool()
