"""Detect anomalies using z-score method."""

from __future__ import annotations

from typing import Any

import numpy as np

from timeclaw.clawhub.tool_base import (
    ClawTool,
    ToolCategory,
    ToolIOType,
)


class DetectAnomalyTool(ClawTool):
    """Detect anomalies using z-score method."""

    name: str = "detect_anomaly"
    category: ToolCategory = ToolCategory.DETECTION
    description: str = (
        "Detect anomalies in a time series using the z-score method. "
        "Identifies data points that deviate significantly from the mean. "
        "Input: 1D numeric array or array-like. "
        "Output: dict with 'anomaly_indices' (list of int positions), 'count' (number of anomalies), "
        "and 'threshold' used. "
        "Use for finding outliers, unusual events, or data quality issues before further analysis. "
        "Param 'threshold' (float, default 3.0): z-score cutoff — lower values flag more points as anomalous."
    )
    io_type: ToolIOType = ToolIOType.X_TO_C
    parameters_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "threshold": {
                "type": "number",
                "description": "Z-score threshold for anomaly detection (default: 3.0)",
                "default": 3.0,
            },
        },
        "required": [],
    }
    is_stub: bool = False

    async def _run(self, ts_data: Any, **params: Any) -> Any:
        threshold = float(params.get("threshold", 3.0))
        arr = np.asarray(ts_data, dtype=float).flatten()
        if arr.size < 2:
            return {"anomaly_indices": [], "count": 0, "threshold": threshold}
        mean = np.nanmean(arr)
        std = np.nanstd(arr)
        if std == 0:
            return {"anomaly_indices": [], "count": 0, "threshold": threshold}
        z_scores = np.abs((arr - mean) / std)
        anomaly_indices = np.where(z_scores > threshold)[0].tolist()
        return {"anomaly_indices": anomaly_indices, "count": len(anomaly_indices), "threshold": threshold}


tool = DetectAnomalyTool()
