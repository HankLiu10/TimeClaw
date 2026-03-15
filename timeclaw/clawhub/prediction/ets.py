"""Exponential smoothing forecast."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing

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


class ETSForecastTool(ClawTool):
    """Exponential smoothing (ETS) forecast."""

    name: str = "ets_forecast"
    category: ToolCategory = ToolCategory.PREDICTION
    description: str = (
        "Forecast a univariate time series using Holt-Winters Exponential Smoothing (ETS) "
        "with configurable error, trend, and seasonal components. "
        "Input: 1-D numeric array. "
        "Output: dict with 'forecast' (array of predicted values), 'horizon', and 'fitted_values' (last 10 in-sample fits). "
        "Strengths: very fast, robust baseline method that works well on smooth, low-noise series "
        "with clear trend and/or seasonal patterns. Falls back to simple exponential smoothing if full model fails. "
        "Key params: 'horizon' — steps ahead (default 10); 'seasonal_periods' — cycle length; "
        "'trend' — trend type ('add' or 'mul', default 'add'); 'seasonal' — seasonal type ('add', 'mul', or None)."
    )
    io_type: ToolIOType = ToolIOType.X_TO_X
    parameters_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "horizon": {"type": "integer", "default": 10, "description": "Number of steps to forecast."},
            "seasonal_periods": {"type": "integer", "default": None, "description": "Number of periods in a seasonal cycle."},
            "trend": {"type": "string", "default": "add", "description": "Trend component type."},
            "seasonal": {"type": "string", "default": None, "description": "Seasonal component type."},
        },
        "required": [],
    }
    is_stub: bool = False

    async def _run(self, ts_data: Any, **params: Any) -> Any:
        arr = _to_array(ts_data)
        horizon: int = params.get("horizon", 10)
        seasonal_periods = params.get("seasonal_periods", None)
        trend: str | None = params.get("trend", "add")
        seasonal: str | None = params.get("seasonal", None)

        try:
            model = ExponentialSmoothing(
                arr,
                trend=trend,
                seasonal=seasonal,
                seasonal_periods=seasonal_periods,
            )
            fit = model.fit(optimized=True)
        except Exception:
            model = SimpleExpSmoothing(arr)
            fit = model.fit(optimized=True)

        forecast = fit.forecast(horizon)
        fitted = fit.fittedvalues

        return {
            "forecast": forecast.tolist(),
            "horizon": horizon,
            "fitted_values": fitted[-10:].tolist(),
        }


tool = ETSForecastTool()
