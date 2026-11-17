"""Confidence normalization helpers.

New definition:
    ``confidence`` is treated as a numeric score only when it can be safely
    interpreted as a finite float. Free-text levels such as ``"high"`` are not
    silently reinterpreted into arbitrary numbers; they are normalized to
    ``None`` instead so the pipeline keeps running without inventing precision.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def coerce_confidence(value: Any) -> float | None:
    """Return a finite float confidence or ``None`` when the value is non-numeric."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        numeric = float(value)
        return numeric if np.isfinite(numeric) else None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            numeric = float(text)
        except ValueError:
            return None
        return numeric if np.isfinite(numeric) else None
    return None
