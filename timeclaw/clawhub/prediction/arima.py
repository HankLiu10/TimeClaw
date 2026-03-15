"""ARIMA forecast."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from timeclaw.clawhub.tool_base import (
    ClawTool,
    ToolCategory,
    ToolIOType,
)


class ARIMAForecastTool(ClawTool):
    """ARIMA forecast using statsmodels."""

    name: str = "arima_forecast"
    category: ToolCategory = ToolCategory.PREDICTION
    description: str = (
        "Forecast a univariate time series using the ARIMA (AutoRegressive Integrated Moving Average) model. "
        "Input: 1-D numeric array (minimum 10 observations). "
        "Output: dict with 'forecast' (array of predicted values), 'horizon', and 'order'. "
        "Strengths: fast, lightweight, and effective on stationary or near-stationary series with linear dynamics. "
        "Best when the series has no strong seasonal component or when differencing can remove the trend. "
        "Key params: 'order' — tuple (p, d, q) controlling autoregressive, differencing, and moving-average terms "
        "(default (1,1,1)); 'horizon' — number of steps ahead to forecast (default 10)."
    )
    io_type: ToolIOType = ToolIOType.X_TO_X
    parameters_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "order": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "ARIMA order (p, d, q), default (1, 1, 1)",
            },
            "horizon": {
                "type": "integer",
                "description": "Forecast horizon (number of steps ahead), default 10",
            },
        },
        "required": [],
    }
    is_stub: bool = False

    async def _run(self, ts_data: Any, **params: Any) -> Any:
        from statsmodels.tsa.arima.model import ARIMA

        order = params.get("order", (1, 1, 1))
        if isinstance(order, list):
            order = tuple(order)
        horizon = params.get("horizon", 10)

        arr = self._to_array(ts_data)
        if arr.ndim > 1:
            arr = arr[:, 0]

        if len(arr) < 10:
            raise ValueError("ts_data must have at least 10 observations for ARIMA")

        model = ARIMA(arr, order=order)
        fitted = model.fit()
        forecast = fitted.forecast(steps=horizon)

        return {
            "forecast": forecast.tolist(),
            "horizon": horizon,
            "order": list(order),
        }

    def _to_array(self, ts_data: Any) -> np.ndarray:
        if ts_data is None:
            raise ValueError("ts_data is required")
        if isinstance(ts_data, pd.DataFrame):
            return ts_data.values
        return np.asarray(ts_data, dtype=float)


tool = ARIMAForecastTool()
