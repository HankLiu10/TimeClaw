"""ModalityAligner: align text and image data to time-series windows by timestamp."""

from __future__ import annotations

from typing import Any

import numpy as np


class ModalityAligner:
    """Align heterogeneous modality data to a common temporal axis.

    Given a time-series with N time steps, aligns text documents and images
    (each carrying a timestamp) to the nearest time step or window.
    """

    def __init__(self, tolerance: float = float("inf")):
        """
        Args:
            tolerance: Maximum allowed distance between a modality timestamp
                       and a time-series step. Items beyond this are dropped.
        """
        self.tolerance = tolerance

    def align(
        self,
        ts_timestamps: np.ndarray | list | None = None,
        text_data: list[dict[str, Any]] | None = None,
        images: list[dict[str, Any]] | None = None,
        window_size: int = 1,
    ) -> dict[str, Any]:
        """Align text and image data to time-series timestamps.

        Args:
            ts_timestamps: Array of time-series timestamps (or integer indices).
            text_data: List of dicts with at least 'timestamp' and 'text' keys.
            images: List of dicts with at least 'timestamp' and 'path'/'data' keys.
            window_size: Number of time steps in each alignment window.

        Returns:
            Dict with 'text_aligned', 'image_aligned', and 'timeline' keys.
        """
        if ts_timestamps is None:
            ts_timestamps = np.array([])
        ts_ts = np.asarray(ts_timestamps, dtype=float)

        text_aligned = self._align_items(ts_ts, text_data or [], window_size, "text")
        image_aligned = self._align_items(ts_ts, images or [], window_size, "image")

        timeline = self._build_timeline(ts_ts, text_aligned, image_aligned)

        return {
            "text_aligned": text_aligned,
            "image_aligned": image_aligned,
            "timeline": timeline,
            "n_text_matched": len(text_aligned),
            "n_image_matched": len(image_aligned),
        }

    def _align_items(
        self,
        ts_ts: np.ndarray,
        items: list[dict[str, Any]],
        window_size: int,
        modality: str,
    ) -> list[dict[str, Any]]:
        if len(ts_ts) == 0 or len(items) == 0:
            return [
                {**item, "ts_idx": None, "distance": None}
                for item in items
            ]

        aligned = []
        half_w = window_size / 2
        for item in items:
            ts = item.get("timestamp")
            if ts is None:
                aligned.append({**item, "ts_idx": None, "distance": None})
                continue

            ts_val = float(ts)
            distances = np.abs(ts_ts - ts_val)
            nearest_idx = int(np.argmin(distances))
            dist = float(distances[nearest_idx])

            if dist > self.tolerance:
                continue

            if window_size <= 1:
                entry = {**item, "ts_idx": nearest_idx, "distance": dist}
            else:
                mask = distances <= half_w
                indices = np.where(mask)[0].tolist()
                entry = {**item, "ts_window": indices, "distance": dist}

            aligned.append(entry)

        return aligned

    def _build_timeline(
        self,
        ts_ts: np.ndarray,
        text_aligned: list[dict[str, Any]],
        image_aligned: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Build a unified timeline showing which modalities are available at each step."""
        if len(ts_ts) == 0:
            return []

        timeline: list[dict[str, Any]] = []
        for i, t in enumerate(ts_ts):
            entry: dict[str, Any] = {
                "step": i,
                "timestamp": float(t),
                "modalities": ["timeseries"],
            }

            text_at_step = [
                a for a in text_aligned
                if a.get("ts_idx") == i or (i in a.get("ts_window", []))
            ]
            if text_at_step:
                entry["modalities"].append("text")
                entry["n_texts"] = len(text_at_step)

            img_at_step = [
                a for a in image_aligned
                if a.get("ts_idx") == i or (i in a.get("ts_window", []))
            ]
            if img_at_step:
                entry["modalities"].append("image")
                entry["n_images"] = len(img_at_step)

            timeline.append(entry)

        return timeline
