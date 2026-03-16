"""Chronos foundation model forecast tool."""

from __future__ import annotations

from typing import Any

from timeclaw.clawhub.prediction._sbm_common import PARAM_SCHEMA, sbm_forecast
from timeclaw.clawhub.tool_base import ClawTool, ToolCategory, ToolIOType


class ChronosForecastTool(ClawTool):
    name: str = "chronos_forecast"
    category: ToolCategory = ToolCategory.PREDICTION
    description: str = (
        "Forecast a univariate time series using Amazon Chronos, a language-model-based time series model "
        "that tokenizes values into bins and generates probabilistic forecasts. "
        "Input: 1-D numeric array. "
        "Output: dict with 'point_forecast' (array), 'horizon', 'model', and optional 'quantiles' for uncertainty. "
        "Strengths: excellent uncertainty quantification, robust to outliers; "
        "handles short to medium series well (50-500 points). Runs on GPU. "
        "Falls back to a linear-trend statistical forecast if the model server is unavailable. "
        "Key params: 'horizon' (steps ahead, default 96)."
    )
    io_type: ToolIOType = ToolIOType.X_TO_X
    parameters_schema: dict[str, Any] = PARAM_SCHEMA
    is_stub: bool = False

    async def _run(self, ts_data: Any, **params: Any) -> Any:
        return sbm_forecast("chronos", ts_data, params.get("horizon", 96))


tool_chronos = ChronosForecastTool()
