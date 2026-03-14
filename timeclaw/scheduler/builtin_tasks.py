"""Built-in heartbeat tasks for drift monitoring, experience cleanup, and vector index maintenance."""

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


async def experience_cleanup_task(
    experience_pool: Any = None,
    max_age_days: int = 90,
) -> dict[str, Any]:
    """Clean up expired experience entries older than max_age_days."""
    if experience_pool is None:
        return {"status": "no_pool"}

    cutoff = time.time() - max_age_days * 86400
    before_size = experience_pool.size

    expired = [
        e for e in experience_pool._entries
        if e.timestamp < cutoff
    ]

    for e in expired:
        experience_pool._entries.remove(e)

    after_size = experience_pool.size
    cleaned = before_size - after_size

    if cleaned > 0:
        logger.info(f"[ExperienceCleanup] Removed {cleaned} expired entries")

    return {
        "before": before_size,
        "after": after_size,
        "cleaned": cleaned,
    }


async def vector_index_rebuild_task(
    experience_pool: Any = None,
) -> dict[str, Any]:
    """Rebuild the vector index from current experience entries."""
    if experience_pool is None or not experience_pool._enable_vector:
        return {"status": "vector_disabled"}

    pool = experience_pool
    if pool._vector_store:
        pool._vector_store.clear()
        count = 0
        for entry in pool._entries:
            pool._index_entry(entry)
            count += 1
        logger.info(f"[VectorRebuild] Re-indexed {count} entries")
        return {"reindexed": count}

    return {"status": "no_vector_store"}
