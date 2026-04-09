"""Sundial Base 128M forecast tool — generative forecasting transformer for time series."""

from __future__ import annotations

import threading
from typing import Any

import numpy as np
from loguru import logger

from timeclaw.clawhub.tool_base import ClawTool, ToolCategory, ToolIOType

_load_lock = threading.Lock()


def _run_timer(ts_data: Any, horizon: int, num_samples: int = 20) -> dict[str, Any]:
    from timeclaw.sbm.registry import configured_default_config, get_global_registry

    registry = get_global_registry()
    if "sundial_base_128m" not in registry.list_available():
        try:
            from timeclaw.sbm.timer_adapter import TimerAdapter, TIMER_DEFAULT_CONFIG
            registry.register(
                "sundial_base_128m",
                TimerAdapter,
                configured_default_config("sundial_base_128m", TIMER_DEFAULT_CONFIG),
            )
        except ImportError:
            raise RuntimeError("sundial_base_128m not installed")

    try:
        adapter = registry.get("sundial_base_128m")
    except KeyError:
        raise RuntimeError("sundial_base_128m adapter not available")

    context = np.asarray(ts_data, dtype=np.float32)
    if context.ndim == 0:
        raise ValueError("ts_data must be at least 1D")

    if not adapter.is_loaded:
        with _load_lock:
            if not adapter.is_loaded:
                try:
                    adapter.load()
                    logger.info("[sundial_base_128m_forecast] model loaded")
                except Exception as e:
                    logger.error(f"[sundial_base_128m_forecast] load failed: {e}")
                    raise RuntimeError(f"timer load failed: {e}") from e

    try:
        output = adapter.forecast(context, prediction_length=horizon, num_samples=num_samples)
        result: dict[str, Any] = {
            "model": "thuml/sundial-base-128m",
            "horizon": horizon,
            "point_forecast": output.point_forecast.tolist() if output.point_forecast is not None else None,
        }
        if output.quantile_forecast:
            result["quantiles"] = {str(q): v.tolist() for q, v in output.quantile_forecast.items()}
        return result
    except Exception as e:
        logger.error(f"[sundial_base_128m_forecast] inference failed: {e}")
        raise RuntimeError(f"timer inference failed: {e}") from e


_PARAM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "horizon": {"type": "integer", "default": 96, "description": "Number of steps to forecast."},
    },
    "required": [],
}


class TimerForecastTool(ClawTool):
    name: str = "sundial_base_128m_forecast"
    category: ToolCategory = ToolCategory.PREDICTION
    description: str = (
        "Forecast a univariate time series using THU Sundial Base 128M, a newer generative forecasting model "
        "from the THU Timer/Sundial line. "
        "Input: 1-D numeric array. "
        "Output: dict with 'point_forecast' (array), 'horizon', 'model', and optional 'quantiles' for uncertainty. "
        "Strengths: captures local temporal dynamics and abrupt changes well; "
        "suitable for short to medium series (50-300 points). Runs on GPU. "
        "Fails explicitly if Sundial is unavailable or the input length is incompatible. "
        "Key params: 'horizon' (steps ahead, default 96)."
    )
    io_type: ToolIOType = ToolIOType.X_TO_X
    parameters_schema: dict[str, Any] = _PARAM_SCHEMA
    is_stub: bool = False

    async def _run(self, ts_data: Any, **params: Any) -> Any:
        return _run_timer(ts_data, params.get("horizon", 96))


tool_timer = TimerForecastTool()
