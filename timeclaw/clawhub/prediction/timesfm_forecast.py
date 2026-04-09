"""TimesFM 2.0 forecast tool — Google's decoder-only transformer foundation model."""

from __future__ import annotations

import threading
from typing import Any

import numpy as np
from loguru import logger

from timeclaw.clawhub.tool_base import ClawTool, ToolCategory, ToolIOType

_load_lock = threading.Lock()


def _run_timesfm(ts_data: Any, horizon: int, num_samples: int = 20) -> dict[str, Any]:
    from timeclaw.sbm.registry import configured_default_config, get_global_registry

    registry = get_global_registry()
    if "timesfm2" not in registry.list_available():
        try:
            from timeclaw.sbm.timesfm_adapter import TimesFMAdapter, TIMESFM_DEFAULT_CONFIG
            registry.register(
                "timesfm2",
                TimesFMAdapter,
                configured_default_config("timesfm2", TIMESFM_DEFAULT_CONFIG),
            )
        except ImportError:
            raise RuntimeError("timesfm2 not installed")

    try:
        adapter = registry.get("timesfm2")
    except KeyError:
        raise RuntimeError("timesfm2 adapter not available")

    context = np.asarray(ts_data, dtype=np.float32)
    if context.ndim == 0:
        raise ValueError("ts_data must be at least 1D")

    if not adapter.is_loaded:
        with _load_lock:
            if not adapter.is_loaded:
                try:
                    adapter.load()
                    logger.info("[timesfm2_forecast] model loaded")
                except Exception as e:
                    logger.error(f"[timesfm2_forecast] load failed: {e}")
                    raise RuntimeError(f"timesfm load failed: {e}") from e

    try:
        output = adapter.forecast(context, prediction_length=horizon, num_samples=num_samples)
        result: dict[str, Any] = {
            "model": "timesfm-2.0-500m-pytorch",
            "horizon": horizon,
            "point_forecast": output.point_forecast.tolist() if output.point_forecast is not None else None,
        }
        if output.quantile_forecast:
            result["quantiles"] = {str(q): v.tolist() for q, v in output.quantile_forecast.items()}
        return result
    except Exception as e:
        logger.error(f"[timesfm2_forecast] inference failed: {e}")
        raise RuntimeError(f"timesfm inference failed: {e}") from e


_PARAM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "horizon": {"type": "integer", "default": 96, "description": "Number of steps to forecast."},
    },
    "required": [],
}


class TimesFMForecastTool(ClawTool):
    name: str = "timesfm2_forecast"
    category: ToolCategory = ToolCategory.PREDICTION
    description: str = (
        "Forecast a univariate time series using Google TimesFM 2.0 "
        "(`google/timesfm-2.0-500m-pytorch`). "
        "Input: 1-D numeric array. "
        "Output: dict with 'point_forecast' (array), 'horizon', 'model', and optional 'quantiles' for uncertainty. "
        "Strengths: captures long-range dependencies and multi-scale patterns without fine-tuning; "
        "best for medium to long series (200+ points). Runs on GPU. "
        "Fails explicitly if TimesFM 2.0 is unavailable. "
        "Key params: 'horizon' (steps ahead, default 96)."
    )
    io_type: ToolIOType = ToolIOType.X_TO_X
    parameters_schema: dict[str, Any] = _PARAM_SCHEMA
    is_stub: bool = False

    async def _run(self, ts_data: Any, **params: Any) -> Any:
        return _run_timesfm(ts_data, params.get("horizon", 96))


tool_timesfm = TimesFMForecastTool()
