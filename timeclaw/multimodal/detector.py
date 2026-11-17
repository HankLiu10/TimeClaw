"""ModalityDetector: automatically detect which modalities are present in input data."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


class ModalityDetector:
    """Detect active modalities from heterogeneous input data.

    Recognizes three modalities:
    - timeseries: numpy arrays, pandas DataFrames, or numeric lists
    - text: strings, list of strings, or list of dicts with 'text' key
    - image: file paths to image files, or dicts with 'path'/'data' keys
    """

    IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp"}

    def detect(
        self,
        ts_data: Any = None,
        text_data: Any = None,
        images: Any = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Detect modalities and return a structured report.

        Returns:
            Dict with keys: modalities (list[str]), details (dict per modality)
        """
        modalities: list[str] = []
        details: dict[str, Any] = {}

        if self._has_timeseries(ts_data):
            modalities.append("timeseries")
            details["timeseries"] = self._ts_info(ts_data)

        if self._has_text(text_data):
            modalities.append("text")
            details["text"] = self._text_info(text_data)

        if self._has_images(images):
            modalities.append("image")
            details["image"] = self._image_info(images)

        return {
            "modalities": modalities,
            "n_modalities": len(modalities),
            "is_multimodal": len(modalities) > 1,
            "details": details,
        }

    def _has_timeseries(self, data: Any) -> bool:
        if data is None:
            return False
        if isinstance(data, np.ndarray) and data.size > 0:
            return True
        try:
            import pandas as pd
            if isinstance(data, (pd.DataFrame, pd.Series)) and len(data) > 0:
                return True
        except ImportError:
            pass
        if isinstance(data, (list, tuple)) and len(data) > 0:
            first = data[0]
            if isinstance(first, (int, float, np.integer, np.floating)):
                return True
        return False

    def _has_text(self, data: Any) -> bool:
        if data is None:
            return False
        if isinstance(data, str) and len(data.strip()) > 0:
            return True
        if isinstance(data, list) and len(data) > 0:
            first = data[0]
            if isinstance(first, str):
                return True
            if isinstance(first, dict) and any(k in first for k in ("text", "content", "source")):
                return True
        return False

    def _has_images(self, data: Any) -> bool:
        if data is None:
            return False
        if isinstance(data, (str, Path)):
            return Path(data).suffix.lower() in self.IMAGE_EXTENSIONS
        if isinstance(data, list) and len(data) > 0:
            first = data[0]
            if isinstance(first, str):
                return Path(first).suffix.lower() in self.IMAGE_EXTENSIONS
            if isinstance(first, dict):
                return "path" in first or "data" in first
        return False

    def _ts_info(self, data: Any) -> dict[str, Any]:
        arr = np.asarray(data) if not hasattr(data, "shape") else data
        shape = tuple(arr.shape) if hasattr(arr, "shape") else (len(data),)
        return {"shape": shape, "dtype": str(getattr(arr, "dtype", "unknown"))}

    def _text_info(self, data: Any) -> dict[str, Any]:
        if isinstance(data, str):
            return {"n_documents": 1, "total_chars": len(data)}
        if isinstance(data, list):
            n = len(data)
            total = sum(
                len(d) if isinstance(d, str) else len(d.get("text", d.get("content", "")))
                for d in data
            )
            has_timestamps = any(
                isinstance(d, dict) and "timestamp" in d for d in data
            )
            return {
                "n_documents": n,
                "total_chars": total,
                "has_timestamps": has_timestamps,
            }
        return {"n_documents": 0}

    def _image_info(self, data: Any) -> dict[str, Any]:
        if isinstance(data, (str, Path)):
            return {"n_images": 1, "paths": [str(data)]}
        if isinstance(data, list):
            paths = []
            for item in data:
                if isinstance(item, str):
                    paths.append(item)
                elif isinstance(item, dict):
                    paths.append(item.get("path", "<embedded>"))
            has_timestamps = any(
                isinstance(d, dict) and "timestamp" in d for d in data
            )
            return {
                "n_images": len(data),
                "paths": paths[:10],
                "has_timestamps": has_timestamps,
            }
        return {"n_images": 0}
