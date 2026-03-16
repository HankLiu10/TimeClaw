"""Built-in heartbeat tasks for drift monitoring."""

from __future__ import annotations

import time
from typing import Any

from loguru import logger


async def drift_check_task(
    data_sources: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Check for distribution drift on monitored data sources.

    Uses CUSUM or ADWIN drift detection tools if available.
    """
    results: dict[str, Any] = {"checked_at": time.time(), "drift_detected": False}

    if not data_sources:
        results["status"] = "no_data_sources"
        return results

    try:
        from timeclaw.clawhub.tool_registry import get_tool_instance
        detector = get_tool_instance("detect_drift")

        for source_name, source_data in data_sources.items():
            result = await detector.execute(source_data)
            if result.success and result.data:
                drift_info = result.data
                if drift_info.get("drift_detected", False):
                    results["drift_detected"] = True
                    results[f"drift_{source_name}"] = drift_info
                    logger.warning(f"[DriftCheck] Drift detected in '{source_name}'")
    except (KeyError, ImportError):
        results["status"] = "drift_tool_unavailable"

    return results
