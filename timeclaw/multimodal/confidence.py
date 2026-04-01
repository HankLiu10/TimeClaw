"""Modality Confidence Estimator: evaluates per-modality execution quality.

Computes reliability scores from:
- Tool success rate
- Observation richness (non-empty results)
- Prediction plausibility (numeric range, format compliance)
"""

from __future__ import annotations

from typing import Any

import numpy as np
from loguru import logger


class ModalityConfidenceEstimator:
    """Estimate the confidence/reliability of a single modality's execution."""

    def __init__(self, confidence_floor: float = 0.1):
        self.confidence_floor = confidence_floor

    def estimate(
        self,
        execution_log: list[dict[str, Any]],
        prediction: Any = None,
    ) -> float:
        """Compute a confidence score [0, 1] for a modality's execution.

        Factors:
        - tool_success_rate (40%): fraction of tools that succeeded
        - observation_richness (30%): fraction of tools with non-empty results
        - prediction_quality (30%): whether prediction exists and is well-formed
        """
        if not execution_log:
            return self.confidence_floor

        tool_success = self._tool_success_rate(execution_log)
        obs_richness = self._observation_richness(execution_log)
        pred_quality = self._prediction_quality(prediction)

        confidence = 0.4 * tool_success + 0.3 * obs_richness + 0.3 * pred_quality
        confidence = max(self.confidence_floor, min(1.0, confidence))

        logger.debug(
            f"[ConfidenceEstimator] success={tool_success:.2f}, "
            f"richness={obs_richness:.2f}, pred_q={pred_quality:.2f} → {confidence:.3f}"
        )
        return round(confidence, 4)

    @staticmethod
    def _tool_success_rate(execution_log: list[dict[str, Any]]) -> float:
        if not execution_log:
            return 0.0
        successes = sum(1 for e in execution_log if e.get("success", False))
        return successes / len(execution_log)

    @staticmethod
    def _observation_richness(execution_log: list[dict[str, Any]]) -> float:
        if not execution_log:
            return 0.0
        rich = 0
        for entry in execution_log:
            if not entry.get("success", False):
                continue
            observation = entry.get("observation")
            if observation is None and entry.get("output") is not None:
                observation = entry.get("output")
            if observation is not None and len(str(observation)) > 10:
                rich += 1
        return rich / len(execution_log)

    @staticmethod
    def _prediction_quality(prediction: Any) -> float:
        if prediction is None:
            return 0.0

        if isinstance(prediction, (int, float, np.integer, np.floating)):
            return 0.8 if not (np.isnan(prediction) or np.isinf(prediction)) else 0.2

        if isinstance(prediction, (list, np.ndarray)):
            try:
                arr = np.asarray(prediction, dtype=float)
                if arr.size == 0:
                    return 0.1
                nan_ratio = np.isnan(arr).sum() / arr.size
                return max(0.2, 1.0 - nan_ratio)
            except (ValueError, TypeError):
                return 0.3

        if isinstance(prediction, str):
            return 0.7 if len(prediction.strip()) > 0 else 0.1

        if isinstance(prediction, dict):
            return 0.6 if prediction else 0.2

        return 0.5


def estimate_all_confidences(
    per_modality_results: dict[str, dict[str, Any]],
    confidence_floor: float = 0.1,
) -> dict[str, float]:
    """Estimate confidence for all modalities at once."""
    estimator = ModalityConfidenceEstimator(confidence_floor=confidence_floor)
    return {
        modality: estimator.estimate(
            result.get("execution_log", []),
            result.get("prediction"),
        )
        for modality, result in per_modality_results.items()
    }
