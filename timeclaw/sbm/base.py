"""SBMBase: abstract base class for all Small Base Model adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

import numpy as np
from pydantic import BaseModel, Field


class SBMTask(str, Enum):
    FORECAST = "forecast"
    ANOMALY = "anomaly"
    CLASSIFICATION = "classification"
    IMPUTATION = "imputation"


class SBMConfig(BaseModel):
    """Configuration for an SBM adapter."""

    model_name: str
    model_path: str | None = None
    device: str = "cuda"
    context_length: int = 512
    prediction_length: int = 96
    batch_size: int = 32
    num_samples: int = 20
    quantile_levels: list[float] = Field(
        default_factory=lambda: [0.1, 0.25, 0.5, 0.75, 0.9]
    )
    extra: dict[str, Any] = Field(default_factory=dict)


class SBMOutput(BaseModel):
    """Standardized output from any SBM model."""

    model_config = {"arbitrary_types_allowed": True}

    point_forecast: Any = None  # np.ndarray (T,) or (T, C)
    quantile_forecast: dict[float, Any] | None = None  # quantile -> np.ndarray
    samples: Any = None  # np.ndarray (num_samples, T) for probabilistic models
    anomaly_scores: Any = None
    class_logits: Any = None
    class_label: str | None = None
    imputed: Any = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_numpy(self) -> np.ndarray | None:
        if self.point_forecast is not None:
            return np.asarray(self.point_forecast)
        return None


class SBMBase(ABC):
    """Abstract base class for Small Base Model adapters.

    Each adapter wraps a specific foundation model (Chronos, Moirai, TimesFM, Timer)
    behind a unified interface.
    """

    name: str = ""
    supported_tasks: list[SBMTask] = []
    requires_gpu: bool = False

    def __init__(self, config: SBMConfig):
        self.config = config
        self._model: Any = None

    @abstractmethod
    def load(self) -> None:
        """Load model weights into memory."""
        ...

    @abstractmethod
    def forecast(
        self,
        context: np.ndarray,
        prediction_length: int | None = None,
        **kwargs: Any,
    ) -> SBMOutput:
        """Generate point and/or probabilistic forecasts.

        Args:
            context: Historical time series, shape (T,) or (T, C).
            prediction_length: Steps to forecast. Defaults to config value.

        Returns:
            SBMOutput with point_forecast and optionally quantile_forecast/samples.
        """
        ...

    def detect_anomaly(self, data: np.ndarray, **kwargs: Any) -> SBMOutput:
        """Detect anomalies using reconstruction error or likelihood."""
        raise NotImplementedError(f"{self.name} does not support anomaly detection")

    def classify(self, data: np.ndarray, **kwargs: Any) -> SBMOutput:
        """Classify a time series segment."""
        raise NotImplementedError(f"{self.name} does not support classification")

    def impute(self, data: np.ndarray, mask: np.ndarray, **kwargs: Any) -> SBMOutput:
        """Fill in missing values indicated by mask."""
        raise NotImplementedError(f"{self.name} does not support imputation")

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def unload(self) -> None:
        """Release model from memory."""
        self._model = None

    def get_profile(self) -> dict[str, Any]:
        """Return a profile describing this model's capabilities for routing."""
        return {
            "name": self.name,
            "tasks": [t.value for t in self.supported_tasks],
            "requires_gpu": self.requires_gpu,
            "context_length": self.config.context_length,
            "prediction_length": self.config.prediction_length,
            "is_loaded": self.is_loaded,
        }

    def __repr__(self) -> str:
        loaded = "loaded" if self.is_loaded else "not loaded"
        return f"<SBM {self.name} ({loaded})>"
