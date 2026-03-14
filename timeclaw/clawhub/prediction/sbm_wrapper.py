"""SBM foundation model forecast wrapper -- routes to registered SBM adapters."""

from __future__ import annotations

from typing import Any

import numpy as np

from timeclaw.clawhub.tool_base import ClawTool, ToolCategory, ToolIOType


class SBMForecastTool(ClawTool):
    """Forecast using SBM foundation models (Chronos, Moirai, TimesFM, Timer).

    Routes to the appropriate SBM adapter via the global SBMRegistry.
    """

    name: str = "sbm_forecast"
    category: ToolCategory = ToolCategory.PREDICTION
    description: str = "Forecast using SBM foundation models (Chronos, Moirai, TimesFM, Timer)."
    io_type: ToolIOType = ToolIOType.X_TO_X
    parameters_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "model_name": {
                "type": "string",
                "enum": ["chronos", "moirai", "timesfm", "timer", "auto"],
                "description": "SBM model to use. 'auto' selects best available.",
            },
            "horizon": {
                "type": "integer",
                "default": 96,
                "description": "Number of steps to forecast.",
            },
            "num_samples": {
                "type": "integer",
                "default": 20,
                "description": "Number of probabilistic samples.",
            },
        },
        "required": [],
    }
    is_stub: bool = False

    async def _run(self, ts_data: Any, **params: Any) -> Any:
        from timeclaw.sbm.registry import get_global_registry, auto_register

        model_name = params.get("model_name", "auto")
        horizon = params.get("horizon", 96)
        num_samples = params.get("num_samples", 20)

        registry = get_global_registry()
        if not registry.list_available():
            auto_register()

        if model_name == "auto":
            model_name = self._select_best(registry)

        try:
            adapter = registry.get(model_name)
        except KeyError:
            available = registry.list_available()
            if not available:
                return self._statistical_fallback(ts_data, horizon)
            model_name = available[0]
            adapter = registry.get(model_name)

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
            result = self._statistical_fallback(ts_data, horizon)
            result["sbm_error"] = str(e)
            result["attempted_model"] = model_name
            return result

    def _select_best(self, registry: Any) -> str:
        available = registry.list_available()
        preference = ["chronos", "moirai", "timesfm", "timer"]
        for name in preference:
            if name in available:
                return name
        return available[0] if available else "chronos"

    def _statistical_fallback(self, ts_data: Any, horizon: int) -> dict[str, Any]:
        """Naive fallback when no SBM is available."""
        arr = np.asarray(ts_data, dtype=np.float32).ravel()
        if len(arr) < 2:
            forecast = np.full(horizon, arr[-1] if len(arr) > 0 else 0.0)
        else:
            trend = (arr[-1] - arr[-min(10, len(arr))]) / min(10, len(arr))
            forecast = arr[-1] + trend * np.arange(1, horizon + 1)
        return {
            "model": "statistical_fallback",
            "horizon": horizon,
            "point_forecast": forecast.tolist(),
        }


tool = SBMForecastTool()
