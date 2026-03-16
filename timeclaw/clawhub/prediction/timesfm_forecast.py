"""TimesFM foundation model forecast tool."""

from __future__ import annotations

from typing import Any

from timeclaw.clawhub.prediction._sbm_common import PARAM_SCHEMA, sbm_forecast
from timeclaw.clawhub.tool_base import ClawTool, ToolCategory, ToolIOType


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
        "Key params: 'horizon' (steps ahead, default 96)."
    )
    io_type: ToolIOType = ToolIOType.X_TO_X
    parameters_schema: dict[str, Any] = PARAM_SCHEMA
    is_stub: bool = False

    async def _run(self, ts_data: Any, **params: Any) -> Any:
        return sbm_forecast("timesfm", ts_data, params.get("horizon", 96))


tool_timesfm = TimesFMForecastTool()
