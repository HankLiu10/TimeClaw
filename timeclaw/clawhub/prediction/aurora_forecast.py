"""Aurora forecast tool — Decision Intelligence Aurora forecasting model."""

from __future__ import annotations

import threading
from typing import Any

import numpy as np
from loguru import logger

from timeclaw.clawhub.tool_base import ClawTool, ToolCategory, ToolIOType

_load_lock = threading.Lock()


def _run_aurora(
    ts_data: Any,
    horizon: int,
    num_samples: int = 20,
    inference_token_len: int = 48,
) -> dict[str, Any]:
    from timeclaw.sbm.registry import configured_default_config, get_global_registry

    registry = get_global_registry()
    if "aurora" not in registry.list_available():
        try:
            from timeclaw.sbm.aurora_adapter import AuroraAdapter, AURORA_DEFAULT_CONFIG
            registry.register(
                "aurora",
                AuroraAdapter,
                configured_default_config("aurora", AURORA_DEFAULT_CONFIG),
            )
        except ImportError:
            raise RuntimeError("aurora-model==0.2.0 not installed")

    try:
        adapter = registry.get("aurora")
    except KeyError:
        raise RuntimeError("aurora adapter not available")

    context = np.asarray(ts_data, dtype=np.float32)
    if context.ndim == 0:
        raise ValueError("ts_data must be at least 1D")

    if not adapter.is_loaded:
        with _load_lock:
            if not adapter.is_loaded:
                try:
                    adapter.load()
                    logger.info("[aurora_forecast] model loaded")
                except Exception as e:
                    logger.error(f"[aurora_forecast] load failed: {e}")
                    raise RuntimeError(f"aurora load failed: {e}") from e

    try:
        output = adapter.forecast(
            context,
            prediction_length=horizon,
            num_samples=num_samples,
            inference_token_len=inference_token_len,
        )
        result: dict[str, Any] = {
            "model": str(output.metadata.get("model", "aurora")),
            "horizon": horizon,
            "point_forecast": output.point_forecast.tolist() if output.point_forecast is not None else None,
        }
        if output.quantile_forecast:
            result["quantiles"] = {str(q): v.tolist() for q, v in output.quantile_forecast.items()}
        return result
    except Exception as e:
        logger.error(f"[aurora_forecast] inference failed: {e}")
        raise RuntimeError(f"aurora inference failed: {e}") from e


_PARAM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "horizon": {"type": "integer", "default": 96, "description": "Number of steps to forecast."},
        "num_samples": {
            "type": "integer",
            "default": 20,
            "description": "Number of forecast samples to draw before aggregating to point/quantiles.",
        },
        "inference_token_len": {
            "type": "integer",
            "default": 48,
            "description": "Aurora inference token length used during generation.",
        },
    },
    "required": [],
}


class AuroraForecastTool(ClawTool):
    name: str = "aurora_forecast"
    category: ToolCategory = ToolCategory.PREDICTION
    description: str = (
        "Forecast a univariate time series using Aurora from Decision Intelligence. "
        "Input: 1-D numeric array. "
        "Output: dict with 'point_forecast' (array), 'horizon', 'model', and optional 'quantiles' for uncertainty. "
        "Strengths: modern generative forecasting with sample-based uncertainty estimation; "
        "useful as an additional foundation-model baseline beside Chronos-2, Moirai 1.1 R-base, TimesFM 2.0, and Sundial Base 128M. "
        "Fails explicitly if Aurora is unavailable. "
        "Key params: 'horizon' (steps ahead), 'num_samples', 'inference_token_len'."
    )
    io_type: ToolIOType = ToolIOType.X_TO_X
    parameters_schema: dict[str, Any] = _PARAM_SCHEMA
    is_stub: bool = False

    async def _run(self, ts_data: Any, **params: Any) -> Any:
        return _run_aurora(
            ts_data,
            params.get("horizon", 96),
            num_samples=params.get("num_samples", 20),
            inference_token_len=params.get("inference_token_len", 48),
        )


tool_aurora = AuroraForecastTool()
