"""Timer foundation model forecast tool."""

from __future__ import annotations

from typing import Any

from timeclaw.clawhub.prediction._sbm_common import PARAM_SCHEMA, sbm_forecast
from timeclaw.clawhub.tool_base import ClawTool, ToolCategory, ToolIOType


class TimerForecastTool(ClawTool):
    name: str = "timer_forecast"
    category: ToolCategory = ToolCategory.PREDICTION
    description: str = (
        "Forecast a univariate time series using Timer, a generative pre-trained transformer "
        "that uses next-token prediction on continuous values with autoregressive decoding. "
        "Input: 1-D numeric array. "
        "Output: dict with 'point_forecast' (array), 'horizon', 'model', and optional 'quantiles' for uncertainty. "
        "Strengths: captures local temporal dynamics and abrupt changes well; "
        "suitable for short to medium series (50-300 points). Runs on GPU. "
        "Falls back to a linear-trend statistical forecast if the model server is unavailable. "
        "Key params: 'horizon' (steps ahead, default 96)."
    )
    io_type: ToolIOType = ToolIOType.X_TO_X
    parameters_schema: dict[str, Any] = PARAM_SCHEMA
    is_stub: bool = False

    async def _run(self, ts_data: Any, **params: Any) -> Any:
        return sbm_forecast("timer", ts_data, params.get("horizon", 96))


tool_timer = TimerForecastTool()
