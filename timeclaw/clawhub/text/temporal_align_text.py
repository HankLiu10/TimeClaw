"""Temporal text alignment tool: align text events to time-series windows."""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from timeclaw.clawhub.tool_base import ClawTool, ToolCategory, ToolIOType


def _parse_timestamp(raw: Any) -> int | None:
    """Best-effort parse of a timestamp into an integer index.

    Supports: int/float, "day_35", "t42", "step_7", "35", ISO dates (as hash).
    Returns None if unparseable.
    """
    if isinstance(raw, (int, float)) and not np.isnan(raw):
        return int(raw)
    if isinstance(raw, (np.integer, np.floating)):
        return int(raw)
    if not isinstance(raw, str):
        return None
    s = raw.strip()
    m = re.search(r'(\d+)', s)
    if m:
        return int(m.group(1))
    return None


class TemporalAlignTextTool(ClawTool):
    """Align timestamped text data to time-series windows for cross-modal analysis."""

    name: str = "temporal_align_text"
    category: ToolCategory = ToolCategory.TEXT
    description: str = (
        "Maps text documents with timestamps to specific time windows in a time-series. "
        "Input: requires both text data (with timestamps) and time-series data in context. "
        "Output: dict with aligned text-to-window mappings showing which text corresponds to which time window. "
        "Use when you need to associate news, reports, or events with the matching segment of a time-series for joint analysis. "
        "Params: window_size (duration of each alignment window), strategy (alignment method)."
    )
    io_type: ToolIOType = ToolIOType.TX_TO_R
    parameters_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "window_size": {"type": "integer", "description": "Window size in time steps"},
            "strategy": {
                "type": "string",
                "enum": ["nearest", "window", "interpolate"],
                "description": "Alignment strategy",
            },
        },
        "required": [],
    }
    is_stub: bool = False

    async def _run(self, ts_data: Any = None, **params: Any) -> Any:
        text_data = params.get("text")
        if text_data is None:
            raise ValueError("text parameter is required (list of dicts with 'timestamp' and 'text')")

        if isinstance(text_data, str):
            import json
            try:
                text_data = json.loads(text_data)
            except (json.JSONDecodeError, TypeError):
                raise ValueError("text must be a JSON list of dicts with 'timestamp' and 'text' keys")

        if not isinstance(text_data, list):
            raise ValueError("text must be a list of dicts with 'timestamp' and 'text' keys")

        window_size = params.get("window_size", 10)
        strategy = params.get("strategy", "nearest")

        ts_length = 0
        if ts_data is not None:
            arr = np.asarray(ts_data)
            ts_length = arr.shape[0] if arr.ndim >= 1 else 0

        ts_timestamps = np.arange(ts_length) if ts_length > 0 else np.array([])

        if strategy == "nearest":
            aligned = self._nearest_align(text_data, ts_timestamps)
        elif strategy == "window":
            aligned = self._window_align(text_data, ts_timestamps, window_size)
        else:
            aligned = self._nearest_align(text_data, ts_timestamps)

        return {"aligned": aligned, "n_matched": len(aligned), "strategy": strategy}

    def _nearest_align(
        self, text_data: list[dict], ts_timestamps: np.ndarray
    ) -> list[dict[str, Any]]:
        if len(ts_timestamps) == 0:
            return [
                {"text_idx": i, "ts_idx": None, "text": self.extract_text_content(d)}
                for i, d in enumerate(text_data)
            ]

        aligned = []
        for i, item in enumerate(text_data):
            raw_ts = item.get("timestamp", 0)
            ts_val = _parse_timestamp(raw_ts)
            if ts_val is None:
                aligned.append({
                    "text_idx": i,
                    "ts_idx": None,
                    "timestamp_raw": raw_ts,
                    "text": self.extract_text_content(item),
                    "distance": None,
                })
                continue
            ts_val = int(np.clip(ts_val, 0, len(ts_timestamps) - 1))
            idx = int(np.argmin(np.abs(ts_timestamps - ts_val)))
            aligned.append({
                "text_idx": i,
                "ts_idx": idx,
                "timestamp_raw": raw_ts,
                "timestamp_parsed": ts_val,
                "text": self.extract_text_content(item),
                "distance": int(abs(ts_timestamps[idx] - ts_val)),
            })
        return aligned

    def _window_align(
        self, text_data: list[dict], ts_timestamps: np.ndarray, window_size: int
    ) -> list[dict[str, Any]]:
        if len(ts_timestamps) == 0:
            return []

        aligned = []
        for i, item in enumerate(text_data):
            raw_ts = item.get("timestamp", 0)
            ts_val = _parse_timestamp(raw_ts)
            if ts_val is None:
                continue
            mask = np.abs(ts_timestamps - ts_val) <= window_size / 2
            indices = np.where(mask)[0].tolist()
            if indices:
                aligned.append({
                    "text_idx": i,
                    "ts_window": indices,
                    "timestamp_raw": raw_ts,
                    "timestamp_parsed": ts_val,
                    "text": self.extract_text_content(item),
                    "window_size": len(indices),
                })
        return aligned


tool = TemporalAlignTextTool()
