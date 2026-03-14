"""Select specific channels from multivariate time series."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from timeclaw.clawhub.tool_base import (
    ClawTool,
    ToolCategory,
    ToolIOType,
)


class SelectChannelsTool(ClawTool):
    """Select specific channels from multivariate time series."""

    name: str = "select_channels"
    category: ToolCategory = ToolCategory.PROCESSING
    description: str = "Select specific channels (columns) from multivariate time series."
    io_type: ToolIOType = ToolIOType.X_TO_X
    parameters_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "channels": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "List of channel indices to select (0-based)",
            },
        },
        "required": ["channels"],
    }
    is_stub: bool = False

    async def _run(self, ts_data: Any, **params: Any) -> Any:
        if ts_data is None:
            raise ValueError("ts_data is required")

        channels = params.get("channels", [])
        if not channels:
            raise ValueError("channels list must not be empty")

        if isinstance(ts_data, pd.DataFrame):
            selected = ts_data.iloc[:, channels]
            data = selected.values.tolist()
        elif isinstance(ts_data, pd.Series):
            raise ValueError("Cannot select channels from a 1D Series")
        else:
            arr = np.asarray(ts_data)
            if arr.ndim < 2:
                raise ValueError("Cannot select channels from 1D data; need 2D array")
            selected = arr[:, channels]
            data = selected.tolist()

        return {"data": data, "selected_channels": channels}


tool = SelectChannelsTool()
