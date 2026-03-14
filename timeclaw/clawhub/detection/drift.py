"""Detect distribution drift using CUSUM and PSI."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from timeclaw.clawhub.tool_base import (
    ClawTool,
    ToolCategory,
    ToolIOType,
)


def _to_array(ts_data):
    if ts_data is None:
        raise ValueError("ts_data is required")
    if isinstance(ts_data, pd.DataFrame):
        arr = ts_data.iloc[:, 0].values
    elif isinstance(ts_data, pd.Series):
        arr = ts_data.values
    else:
        arr = np.asarray(ts_data).ravel()
    return arr.astype(float)


def _cusum(arr: np.ndarray, threshold: float) -> list[int]:
    mean = arr.mean()
    cumsum_pos = np.zeros(len(arr))
    cumsum_neg = np.zeros(len(arr))
    drift_points = []
    for i in range(1, len(arr)):
        cumsum_pos[i] = max(0, cumsum_pos[i - 1] + (arr[i] - mean))
        cumsum_neg[i] = min(0, cumsum_neg[i - 1] + (arr[i] - mean))
        if cumsum_pos[i] > threshold or cumsum_neg[i] < -threshold:
            drift_points.append(int(i))
            cumsum_pos[i] = 0
            cumsum_neg[i] = 0
    return drift_points


def _psi(arr: np.ndarray, n_bins: int = 10) -> float:
    mid = len(arr) // 2
    ref, cur = arr[:mid], arr[mid:]
    eps = 1e-6
    breakpoints = np.linspace(arr.min(), arr.max() + eps, n_bins + 1)
    ref_counts = np.histogram(ref, bins=breakpoints)[0].astype(float)
    cur_counts = np.histogram(cur, bins=breakpoints)[0].astype(float)
    ref_pct = ref_counts / ref_counts.sum() + eps
    cur_pct = cur_counts / cur_counts.sum() + eps
    psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
    return psi


class DetectDriftTool(ClawTool):
    """Detect distribution drift using CUSUM and PSI."""

    name: str = "detect_drift"
    category: ToolCategory = ToolCategory.DETECTION
    description: str = "Detect distribution drift using CUSUM and PSI."
    io_type: ToolIOType = ToolIOType.X_TO_C
    parameters_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "threshold": {
                "type": "number",
                "description": "CUSUM threshold for drift detection.",
                "default": 5.0,
            },
        },
    }
    is_stub: bool = False

    async def _run(self, ts_data: Any, **params: Any) -> Any:
        arr = _to_array(ts_data)
        threshold = float(params.get("threshold", 5.0))
        cusum_drift_points = _cusum(arr, threshold)
        psi_score = _psi(arr)
        drift_detected = len(cusum_drift_points) > 0 or psi_score > 0.2
        return {
            "cusum_drift_points": cusum_drift_points,
            "psi_score": psi_score,
            "drift_detected": drift_detected,
        }


tool = DetectDriftTool()
