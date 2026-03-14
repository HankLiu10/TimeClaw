"""Temporal image alignment tool: align timestamped images to time-series windows."""

from __future__ import annotations

from typing import Any

import numpy as np

from timeclaw.clawhub.tool_base import ClawTool, ToolCategory, ToolIOType


class TemporalAlignImageTool(ClawTool):
    """Align timestamped images to corresponding time-series windows."""

    name: str = "temporal_align_image"
    category: ToolCategory = ToolCategory.IMAGE
    description: str = "Align images by timestamp to corresponding time-series windows."
    io_type: ToolIOType = ToolIOType.IX_TO_R
    parameters_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "window_size": {
                "type": "integer",
                "description": "Window size in time steps",
            },
            "strategy": {
                "type": "string",
                "enum": ["nearest", "window"],
                "description": "Alignment strategy",
            },
        },
        "required": [],
    }
    is_stub: bool = False

    async def _run(self, ts_data: Any = None, **params: Any) -> Any:
        images = params.get("images")
        if images is None or not isinstance(images, list):
            raise ValueError(
                "images parameter is required (list of dicts with 'timestamp' and 'path')"
            )

        window_size = params.get("window_size", 10)
        strategy = params.get("strategy", "nearest")

        ts_length = 0
        if ts_data is not None:
            arr = np.asarray(ts_data)
            ts_length = arr.shape[0] if arr.ndim >= 1 else 0

        ts_timestamps = np.arange(ts_length) if ts_length > 0 else np.array([])

        aligned = []
        for i, item in enumerate(images):
            if not isinstance(item, dict) or "timestamp" not in item:
                continue

            ts = item["timestamp"]
            entry: dict[str, Any] = {
                "image_idx": i,
                "timestamp": ts,
                "path": item.get("path", ""),
            }

            if len(ts_timestamps) == 0:
                entry["ts_idx"] = None
            elif strategy == "nearest":
                idx = int(np.argmin(np.abs(ts_timestamps - ts)))
                entry["ts_idx"] = idx
                entry["distance"] = float(abs(ts_timestamps[idx] - ts))
            elif strategy == "window":
                mask = np.abs(ts_timestamps - ts) <= window_size / 2
                entry["ts_window"] = np.where(mask)[0].tolist()
                entry["window_size"] = int(mask.sum())

            aligned.append(entry)

        return {
            "aligned": aligned,
            "n_matched": len(aligned),
            "strategy": strategy,
        }


tool = TemporalAlignImageTool()
