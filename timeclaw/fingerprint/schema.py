"""Data fingerprint schema: structured representation of time series characteristics."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class BasicStatsFingerprint(BaseModel):
    mean: float | None = None
    std: float | None = None
    min: float | None = None
    max: float | None = None
    skewness: float | None = None
    kurtosis: float | None = None


class StationarityFingerprint(BaseModel):
    adf_statistic: float | None = None
    adf_p_value: float | None = None
    kpss_statistic: float | None = None
    kpss_p_value: float | None = None
    conclusion: str = "unknown"  # stationary, non_stationary, trend_stationary, difference_stationary


class TrendFingerprint(BaseModel):
    direction: str = "unknown"  # up, down, flat
    slope: float | None = None
    p_value: float | None = None


class DataFingerprint(BaseModel):
    """Complete fingerprint of a time series dataset."""

    data_shape: tuple[int, ...] | None = None
    num_channels: int = 1
    length: int = 0

    basic_stats: BasicStatsFingerprint = Field(default_factory=BasicStatsFingerprint)
    stationarity: StationarityFingerprint = Field(default_factory=StationarityFingerprint)
    trend: TrendFingerprint = Field(default_factory=TrendFingerprint)

    seasonality_period: int | None = None
    anomaly_count: int = 0
    drift_detected: bool = False
    heterogeneity_score: float = 0.0

    missing_ratio: float = 0.0
    dominant_frequency: float | None = None
    hurst_exponent: float | None = None

    raw_features: dict[str, Any] = Field(default_factory=dict)

    def to_summary(self) -> str:
        lines = [
            f"Shape: {self.data_shape}, Channels: {self.num_channels}, Length: {self.length}",
            f"Stats: mean={self.basic_stats.mean}, std={self.basic_stats.std}",
            f"Stationarity: {self.stationarity.conclusion}",
            f"Trend: {self.trend.direction} (slope={self.trend.slope})",
            f"Drift: {self.drift_detected}, Heterogeneity: {self.heterogeneity_score:.3f}",
            f"Anomalies: {self.anomaly_count}, Missing: {self.missing_ratio:.2%}",
        ]
        return "\n".join(lines)
