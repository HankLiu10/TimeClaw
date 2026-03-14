"""Fingerprint extractor: combines profiling tools to build a DataFingerprint."""

from __future__ import annotations

from typing import Any

import numpy as np
from loguru import logger

from timeclaw.clawhub.tool_base import ToolResult
from timeclaw.clawhub.tool_registry import get_tool_instance
from timeclaw.fingerprint.schema import (
    BasicStatsFingerprint,
    DataFingerprint,
    StationarityFingerprint,
    TrendFingerprint,
)


class FingerprintExtractor:
    """Extracts a DataFingerprint by running a suite of profiling tools."""

    async def extract(self, data: Any) -> DataFingerprint:
        fp = DataFingerprint()

        try:
            arr = np.asarray(data, dtype=float)
            fp.data_shape = tuple(arr.shape)
            fp.length = arr.shape[0]
            fp.num_channels = arr.shape[1] if arr.ndim > 1 else 1
            fp.missing_ratio = float(np.isnan(arr).sum() / arr.size) if arr.size > 0 else 0.0
        except (ValueError, TypeError):
            pass

        fp.basic_stats = await self._extract_basic_stats(data)
        fp.stationarity = await self._extract_stationarity(data)
        fp.trend = await self._extract_trend(data)
        fp.anomaly_count = await self._extract_anomaly_count(data)

        return fp

    async def _extract_basic_stats(self, data: Any) -> BasicStatsFingerprint:
        try:
            tool = get_tool_instance("basic_stats")
            result: ToolResult = await tool.execute(data)
            if result.success and isinstance(result.data, dict):
                return BasicStatsFingerprint(**{
                    k: result.data.get(k) for k in BasicStatsFingerprint.model_fields
                })
        except Exception as e:
            logger.debug(f"basic_stats extraction failed: {e}")
        return BasicStatsFingerprint()

    async def _extract_stationarity(self, data: Any) -> StationarityFingerprint:
        try:
            tool = get_tool_instance("stationarity_test")
            result: ToolResult = await tool.execute(data, test="both")
            if result.success and isinstance(result.data, dict):
                adf = result.data.get("adf", {})
                kpss = result.data.get("kpss", {})
                return StationarityFingerprint(
                    adf_statistic=adf.get("statistic"),
                    adf_p_value=adf.get("p_value"),
                    kpss_statistic=kpss.get("statistic"),
                    kpss_p_value=kpss.get("p_value"),
                    conclusion=result.data.get("conclusion", "unknown"),
                )
        except Exception as e:
            logger.debug(f"stationarity extraction failed: {e}")
        return StationarityFingerprint()

    async def _extract_trend(self, data: Any) -> TrendFingerprint:
        try:
            tool = get_tool_instance("detect_trend")
            result: ToolResult = await tool.execute(data)
            if result.success and isinstance(result.data, dict):
                return TrendFingerprint(
                    direction=result.data.get("direction", "unknown"),
                    slope=result.data.get("slope"),
                    p_value=result.data.get("p_value"),
                )
        except Exception as e:
            logger.debug(f"trend extraction failed: {e}")
        return TrendFingerprint()

    async def _extract_anomaly_count(self, data: Any) -> int:
        try:
            tool = get_tool_instance("detect_anomaly")
            result: ToolResult = await tool.execute(data)
            if result.success and isinstance(result.data, dict):
                indices = result.data.get("anomaly_indices", [])
                return len(indices)
        except Exception as e:
            logger.debug(f"anomaly extraction failed: {e}")
        return 0
