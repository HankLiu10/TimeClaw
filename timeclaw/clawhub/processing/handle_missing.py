"""Handle missing values in time series."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from timeclaw.clawhub.tool_base import (
    ClawTool,
    ToolCategory,
    ToolIOType,
)


class HandleMissingTool(ClawTool):
    """Handle missing values in time series."""

    name: str = "handle_missing"
    category: ToolCategory = ToolCategory.PROCESSING
    description: str = (
        "Clean NaN or missing values in a time series using one of several imputation "
        "strategies. Input: a numeric array, DataFrame, or Series that may contain NaN entries. "
        "Output: dict with 'data' (cleaned values as list) and 'missing_count' (number of NaN "
        "values that were filled). This is an essential preprocessing step — run it before any "
        "analysis, prediction, or profiling tool that cannot handle missing data. "
        "Parameters (required): strategy ('ffill' forward-fill | 'bfill' backward-fill | "
        "'interpolate' linear interpolation | 'mean' column-mean imputation)."
    )
    io_type: ToolIOType = ToolIOType.X_TO_X
    parameters_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "strategy": {
                "type": "string",
                "enum": ["ffill", "bfill", "interpolate", "mean"],
                "description": "Strategy: ffill (forward fill), bfill (backward fill), interpolate, or mean",
            },
        },
        "required": ["strategy"],
    }
    is_stub: bool = False

    _STRATEGY_ALIASES: dict[str, str] = {
        "ffill": "ffill", "forward_fill": "ffill", "forward": "ffill", "forwardfill": "ffill",
        "bfill": "bfill", "backward_fill": "bfill", "backward": "bfill", "backwardfill": "bfill",
        "interpolate": "interpolate", "linear": "interpolate", "interp": "interpolate",
        "mean": "mean", "average": "mean", "avg": "mean",
    }

    async def _run(self, ts_data: Any, **params: Any) -> Any:
        if ts_data is None:
            raise ValueError("ts_data is required")

        raw = params.get("strategy", "ffill")
        strategy = self._STRATEGY_ALIASES.get(str(raw).lower().strip(), "ffill")

        if isinstance(ts_data, (pd.DataFrame, pd.Series)):
            missing_count = int(ts_data.isna().sum().sum()) if isinstance(ts_data, pd.DataFrame) else int(ts_data.isna().sum())
            filled = self._fill_pandas(ts_data, strategy)
            data = filled.values.tolist()
        else:
            arr = np.asarray(ts_data, dtype=float)
            missing_count = int(np.isnan(arr).sum())
            filled = self._fill_array(arr, strategy)
            data = filled.tolist()

        return {"data": data, "missing_count": missing_count}

    def _fill_pandas(self, data: pd.DataFrame | pd.Series, strategy: str) -> pd.DataFrame | pd.Series:
        if strategy == "ffill":
            return data.ffill()
        if strategy == "bfill":
            return data.bfill()
        if strategy == "interpolate":
            return data.interpolate(method="linear")
        if strategy == "mean":
            if isinstance(data, pd.DataFrame):
                return data.fillna(data.mean())
            return data.fillna(data.mean())
        raise ValueError(f"Unknown strategy: {strategy}")

    def _fill_array(self, arr: np.ndarray, strategy: str) -> np.ndarray:
        result = arr.copy()
        if arr.ndim == 1:
            return self._fill_1d(result, strategy)
        for col in range(result.shape[1]):
            result[:, col] = self._fill_1d(result[:, col], strategy)
        return result

    def _fill_1d(self, arr: np.ndarray, strategy: str) -> np.ndarray:
        mask = np.isnan(arr)
        if not mask.any():
            return arr

        if strategy == "ffill":
            idx = np.where(~mask, np.arange(len(arr)), 0)
            np.maximum.accumulate(idx, out=idx)
            return arr[idx]
        if strategy == "bfill":
            idx = np.where(~mask, np.arange(len(arr)), len(arr) - 1)
            rev = idx[::-1]
            np.minimum.accumulate(rev, out=rev)
            return arr[idx]
        if strategy == "interpolate":
            valid = np.where(~mask)[0]
            if len(valid) == 0:
                return arr
            arr[mask] = np.interp(np.where(mask)[0], valid, arr[valid])
            return arr
        if strategy == "mean":
            arr[mask] = np.nanmean(arr)
            return arr
        raise ValueError(f"Unknown strategy: {strategy}")


tool = HandleMissingTool()
