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


def _detect_dominant_period(arr: np.ndarray, min_period: int = 3) -> int | None:
    """Use FFT to find the dominant period in the data."""
    if len(arr) < 2 * min_period:
        return None
    detrended = arr - np.linspace(arr[0], arr[-1], len(arr))
    fft_vals = np.abs(np.fft.rfft(detrended))
    fft_vals[0] = 0  # ignore DC
    freqs = np.fft.rfftfreq(len(arr))
    valid = freqs > 0
    fft_vals = fft_vals[valid]
    freqs = freqs[valid]
    if len(fft_vals) == 0:
        return None
    peak_idx = np.argmax(fft_vals)
    period = int(round(1.0 / freqs[peak_idx]))
    if period < min_period or period > len(arr) // 2:
        return None
    return period


class ProphetForecastTool(ClawTool):
    """Prophet forecast for time series with seasonality."""

    name: str = "prophet_forecast"
    category: ToolCategory = ToolCategory.PREDICTION
    description: str = (
        "Forecast a univariate time series using Facebook Prophet, an additive regression model "
        "that automatically detects changepoints and models multiple seasonalities. "
        "Input: 1-D numeric array. "
        "Output: dict with 'forecast' (array of predicted values), 'horizon', and 'trend' (trend component array). "
        "Strengths: handles piecewise-linear trend shifts and periodic patterns without manual tuning; "
        "best for moderate-length series (100–1000 points) with clear seasonal structure. "
        "Slower than ARIMA/ETS but more flexible for complex, non-linear patterns. "
        "Key params: 'horizon' — number of steps ahead to forecast (default 10)."
    )
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

        dominant_period = _detect_dominant_period(arr)

        if dominant_period is not None and dominant_period >= 3:
            minutes_per_point = int(round(7 * 24 * 60 / dominant_period))
            minutes_per_point = max(minutes_per_point, 1)
            freq = f"{minutes_per_point}min"
        else:
            freq = "D"

        df = pd.DataFrame({
            "ds": pd.date_range(start="2000-01-01", periods=len(arr), freq=freq),
            "y": arr,
        })

        model = Prophet(
            yearly_seasonality=False,
            weekly_seasonality=(dominant_period is not None),
            daily_seasonality=False,
            changepoint_prior_scale=0.1,
        )
        model.fit(df)

        future = model.make_future_dataframe(periods=horizon, freq=freq)
        prediction = model.predict(future)

        forecast_values = prediction["yhat"].iloc[-horizon:].tolist()
        trend_values = prediction["trend"].iloc[-horizon:].tolist()

        return {
            "forecast": forecast_values,
            "horizon": horizon,
            "trend": trend_values,
        }


tool = ProphetForecastTool()
