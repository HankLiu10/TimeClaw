"""Individual SBM foundation model forecast tools.

Each model (TimesFM, Chronos, Timer, Moirai) is exposed as a separate ClawTool
so the agent can learn which model works best for which data type.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from timeclaw.clawhub.tool_base import ClawTool, ToolCategory, ToolIOType


def _sbm_forecast(model_name: str, ts_data: Any, horizon: int, num_samples: int = 20) -> dict[str, Any]:
    """Shared forecast logic: call SBM server for a specific model."""
    from timeclaw.sbm.registry import get_global_registry, auto_register

    registry = get_global_registry()
    if not registry.list_available():
        auto_register()

    try:
        adapter = registry.get(model_name)
    except KeyError:
        return _statistical_fallback(ts_data, horizon, model_name)

    context = np.asarray(ts_data, dtype=np.float32)
    if context.ndim == 0:
        raise ValueError("ts_data must be at least 1D")

    try:
        output = adapter.forecast(context, prediction_length=horizon, num_samples=num_samples)
        result: dict[str, Any] = {
            "model": model_name,
            "horizon": horizon,
            "point_forecast": output.point_forecast.tolist()
            if output.point_forecast is not None
            else None,
        }
        if output.quantile_forecast:
            result["quantiles"] = {
                str(q): v.tolist() for q, v in output.quantile_forecast.items()
            }
        return result
    except Exception as e:
        result = _statistical_fallback(ts_data, horizon, model_name)
        result["sbm_error"] = str(e)
        return result


def _statistical_fallback(ts_data: Any, horizon: int, model_name: str) -> dict[str, Any]:
    arr = np.asarray(ts_data, dtype=np.float32).ravel()
    if len(arr) < 2:
        forecast = np.full(horizon, arr[-1] if len(arr) > 0 else 0.0)
    else:
        trend = (arr[-1] - arr[-min(10, len(arr))]) / min(10, len(arr))
        forecast = arr[-1] + trend * np.arange(1, horizon + 1)
    return {
        "model": f"{model_name}_fallback",
        "horizon": horizon,
        "point_forecast": forecast.tolist(),
    }


_PARAM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "horizon": {
            "type": "integer",
            "default": 96,
            "description": "Number of steps to forecast.",
        },
    },
    "required": [],
}


class TimesFMForecastTool(ClawTool):
    name: str = "timesfm_forecast"
    category: ToolCategory = ToolCategory.PREDICTION
    description: str = (
        "Forecast a univariate time series using Google TimesFM, a decoder-only transformer foundation model "
        "pre-trained on 100 billion real-world time points. "
        "Input: 1-D numeric array. "
        "Output: dict with 'point_forecast' (array), 'horizon', 'model', and optional 'quantiles' for uncertainty. "
        "Strengths: captures long-range dependencies and multi-scale patterns without fine-tuning; "
        "best for medium to long series (200+ points). Runs on GPU. "
        "Falls back to a linear-trend statistical forecast if the model server is unavailable. "
        "Key params: 'horizon' — steps ahead (default 96)."
    )
    io_type: ToolIOType = ToolIOType.X_TO_X
    parameters_schema: dict[str, Any] = _PARAM_SCHEMA
    is_stub: bool = False

    async def _run(self, ts_data: Any, **params: Any) -> Any:
        return _sbm_forecast("timesfm", ts_data, params.get("horizon", 96))


class ChronosForecastTool(ClawTool):
    name: str = "chronos_forecast"
    category: ToolCategory = ToolCategory.PREDICTION
    description: str = (
        "Forecast a univariate time series using Amazon Chronos, a language-model-based time series model "
        "that tokenizes values into bins and generates probabilistic forecasts. "
        "Input: 1-D numeric array. "
        "Output: dict with 'point_forecast' (array), 'horizon', 'model', and optional 'quantiles' for uncertainty. "
        "Strengths: excellent uncertainty quantification, robust to outliers; "
        "handles short to medium series well (50–500 points). Runs on GPU. "
        "Falls back to a linear-trend statistical forecast if the model server is unavailable. "
        "Key params: 'horizon' — steps ahead (default 96)."
    )
    io_type: ToolIOType = ToolIOType.X_TO_X
    parameters_schema: dict[str, Any] = _PARAM_SCHEMA
    is_stub: bool = False

    async def _run(self, ts_data: Any, **params: Any) -> Any:
        return _sbm_forecast("chronos", ts_data, params.get("horizon", 96))


class TimerForecastTool(ClawTool):
    name: str = "timer_forecast"
    category: ToolCategory = ToolCategory.PREDICTION
    description: str = (
        "Forecast a univariate time series using Timer, a generative pre-trained transformer "
        "that uses next-token prediction on continuous values with autoregressive decoding. "
        "Input: 1-D numeric array. "
        "Output: dict with 'point_forecast' (array), 'horizon', 'model', and optional 'quantiles' for uncertainty. "
        "Strengths: captures local temporal dynamics and abrupt changes well; "
        "suitable for short to medium series (50–300 points). Runs on GPU. "
        "Falls back to a linear-trend statistical forecast if the model server is unavailable. "
        "Key params: 'horizon' — steps ahead (default 96)."
    )
    io_type: ToolIOType = ToolIOType.X_TO_X
    parameters_schema: dict[str, Any] = _PARAM_SCHEMA
    is_stub: bool = False

    async def _run(self, ts_data: Any, **params: Any) -> Any:
        return _sbm_forecast("timer", ts_data, params.get("horizon", 96))


class MoiraiForecastTool(ClawTool):
    name: str = "moirai_forecast"
    category: ToolCategory = ToolCategory.PREDICTION
    description: str = (
        "Forecast a univariate time series using Salesforce Moirai, a masked encoder-based universal forecasting model "
        "trained on 27 billion observations across 9 domains with any-variate attention. "
        "Input: 1-D numeric array. "
        "Output: dict with 'point_forecast' (array), 'horizon', 'model', and optional 'quantiles' for uncertainty. "
        "Strengths: strong at multi-step forecasting, adapts to varying frequencies without fine-tuning; "
        "best for medium to long series (100+ points). Runs on GPU. "
        "Falls back to a linear-trend statistical forecast if the model server is unavailable. "
        "Key params: 'horizon' — steps ahead (default 96)."
    )
    io_type: ToolIOType = ToolIOType.X_TO_X
    parameters_schema: dict[str, Any] = _PARAM_SCHEMA
    is_stub: bool = False

    async def _run(self, ts_data: Any, **params: Any) -> Any:
        return _sbm_forecast("moirai", ts_data, params.get("horizon", 96))


# Register all four as separate tools
tool_timesfm = TimesFMForecastTool()
tool_chronos = ChronosForecastTool()
tool_timer = TimerForecastTool()
tool_moirai = MoiraiForecastTool()
