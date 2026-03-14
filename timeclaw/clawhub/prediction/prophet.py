"""Prophet forecast."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from timeclaw.clawhub.tool_base import (
    ClawTool,
    ToolCategory,
    ToolIOType,
)

try:
    from prophet import Prophet

    _PROPHET_AVAILABLE = True
except ImportError:
    _PROPHET_AVAILABLE = False


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


class ProphetForecastTool(ClawTool):
    """Prophet forecast for time series with seasonality."""

    name: str = "prophet_forecast"
    category: ToolCategory = ToolCategory.PREDICTION
    description: str = "Forecast using Facebook Prophet."
    io_type: ToolIOType = ToolIOType.X_TO_X
    parameters_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "horizon": {"type": "integer", "default": 10, "description": "Number of steps to forecast."},
        },
        "required": [],
    }
    is_stub: bool = not _PROPHET_AVAILABLE

    async def _run(self, ts_data: Any, **params: Any) -> Any:
        if not _PROPHET_AVAILABLE:
            return {
                "note": "Prophet is not installed. Install with: pip install prophet",
                "is_stub": True,
            }

        arr = _to_array(ts_data)
        horizon: int = params.get("horizon", 10)

        df = pd.DataFrame({
            "ds": pd.date_range(start="2000-01-01", periods=len(arr), freq="D"),
            "y": arr,
        })

        model = Prophet()
        model.fit(df)

        future = model.make_future_dataframe(periods=horizon)
        prediction = model.predict(future)

        forecast_values = prediction["yhat"].iloc[-horizon:].tolist()
        trend_values = prediction["trend"].iloc[-horizon:].tolist()

        return {
            "forecast": forecast_values,
            "horizon": horizon,
            "trend": trend_values,
        }


tool = ProphetForecastTool()
