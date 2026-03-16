"""Moirai foundation model forecast tool."""

from __future__ import annotations

from typing import Any

from timeclaw.clawhub.prediction._sbm_common import PARAM_SCHEMA, sbm_forecast
from timeclaw.clawhub.tool_base import ClawTool, ToolCategory, ToolIOType


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
        "Key params: 'horizon' (steps ahead, default 96)."
    )
    io_type: ToolIOType = ToolIOType.X_TO_X
    parameters_schema: dict[str, Any] = PARAM_SCHEMA
    is_stub: bool = False

    async def _run(self, ts_data: Any, **params: Any) -> Any:
        return sbm_forecast("moirai", ts_data, params.get("horizon", 96))


tool_moirai = MoiraiForecastTool()
