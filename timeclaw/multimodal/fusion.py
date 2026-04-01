"""ModalityFusion: multi-strategy fusion engine for multimodal evidence.

Strategies:
- fixed: Static weights (original behavior)
- task_adaptive: Weights adjusted by task type
- confidence_weighted: Weights based on per-modality execution confidence
- llm_arbitrated: LLM acts as final arbiter to combine evidence

All strategies produce a unified result dict with prediction, confidence, and metadata.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Any

import numpy as np
from loguru import logger

if TYPE_CHECKING:
    from timeclaw.llm.base import LLMBackend

DEFAULT_FIXED_WEIGHTS = {"timeseries": 0.5, "text": 0.3, "image": 0.2}

DEFAULT_TASK_WEIGHTS = {
    "forecast": {"timeseries": 0.7, "text": 0.2, "image": 0.1},
    "classification": {"timeseries": 0.5, "text": 0.3, "image": 0.2},
    "anomaly_detection": {"timeseries": 0.6, "text": 0.2, "image": 0.2},
    "qa": {"timeseries": 0.2, "text": 0.6, "image": 0.2},
    "imputation": {"timeseries": 0.8, "text": 0.1, "image": 0.1},
}


class ModalityFusion:
    """Multi-strategy multimodal evidence fusion engine.

    Supports: "fixed", "task_adaptive", "confidence_weighted", "llm_arbitrated".
    """

    VALID_STRATEGIES = ("fixed", "early", "late", "task_adaptive", "confidence_weighted", "llm_arbitrated")

    def __init__(
        self,
        strategy: str = "confidence_weighted",
        weights: dict[str, float] | None = None,
        task_weights: dict[str, dict[str, float]] | None = None,
        llm: LLMBackend | None = None,
        confidence_floor: float = 0.1,
    ):
        if strategy == "late":
            strategy = "fixed"
        if strategy == "early":
            pass
        elif strategy not in self.VALID_STRATEGIES:
            raise ValueError(f"Unknown fusion strategy: {strategy}. Valid: {self.VALID_STRATEGIES}")

        self.strategy = strategy
        self.weights = weights or DEFAULT_FIXED_WEIGHTS.copy()
        self.task_weights = task_weights or DEFAULT_TASK_WEIGHTS.copy()
        self.llm = llm
        self.confidence_floor = confidence_floor

    def fuse(
        self,
        evidence: dict[str, Any],
        task_type: str = "forecast",
        modality_confidences: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Fuse multimodal evidence using the configured strategy.

        Args:
            evidence: Dict mapping modality -> {prediction, confidence, label, features, ...}
            task_type: Current task type for adaptive weighting
            modality_confidences: Pre-computed per-modality confidence scores (for confidence_weighted)
        """
        if self.strategy == "early":
            return self._early_fuse(evidence)
        elif self.strategy == "fixed":
            return self._weighted_fuse(evidence, self.weights)
        elif self.strategy == "task_adaptive":
            weights = self.task_weights.get(task_type, self.weights)
            return self._weighted_fuse(evidence, weights, strategy_name="task_adaptive")
        elif self.strategy == "confidence_weighted":
            weights = self._compute_confidence_weights(evidence, task_type, modality_confidences)
            return self._weighted_fuse(evidence, weights, strategy_name="confidence_weighted")
        elif self.strategy == "llm_arbitrated":
            return self._weighted_fuse(evidence, self.weights, strategy_name="llm_arbitrated")
        else:
            return self._weighted_fuse(evidence, self.weights)

    async def fuse_async(
        self,
        evidence: dict[str, Any],
        task_type: str = "forecast",
        modality_confidences: dict[str, float] | None = None,
        per_modality_results: dict[str, dict[str, Any]] | None = None,
        task_description: str = "",
    ) -> dict[str, Any]:
        """Async fusion — required for llm_arbitrated strategy."""
        if self.strategy == "llm_arbitrated" and self.llm is not None:
            return await self._llm_arbitrated_fuse(
                evidence, task_type, per_modality_results or {}, task_description
            )
        return self.fuse(evidence, task_type, modality_confidences)

    # ── Early Fusion ──

    def _early_fuse(self, evidence: dict[str, Any]) -> dict[str, Any]:
        all_features: list[np.ndarray] = []
        sources: list[str] = []

        for modality, data in evidence.items():
            if isinstance(data, np.ndarray):
                all_features.append(data.ravel())
                sources.append(modality)
            elif isinstance(data, dict) and "features" in data:
                all_features.append(np.asarray(data["features"]).ravel())
                sources.append(modality)
            elif isinstance(data, list):
                all_features.append(np.asarray(data, dtype=float).ravel())
                sources.append(modality)

        if not all_features:
            return {"fused_features": [], "dim": 0, "sources": [], "strategy": "early"}

        fused = np.concatenate(all_features)
        return {
            "fused_features": fused.tolist(),
            "dim": len(fused),
            "sources": sources,
            "strategy": "early",
        }

    # ── Weighted Fusion (shared by fixed, task_adaptive, confidence_weighted) ──

    def _weighted_fuse(
        self,
        evidence: dict[str, Any],
        weights: dict[str, float],
        strategy_name: str = "fixed",
    ) -> dict[str, Any]:
        predictions: dict[str, Any] = {}
        confidences: dict[str, float] = {}
        labels: dict[str, str] = {}

        for modality, data in evidence.items():
            if not isinstance(data, dict):
                continue
            if "prediction" in data:
                predictions[modality] = data["prediction"]
            if "confidence" in data:
                confidences[modality] = float(data["confidence"])
            if "label" in data:
                labels[modality] = data["label"]

        fused_pred = None
        if predictions:
            numeric_preds = {}
            for m, p in predictions.items():
                if isinstance(p, (int, float, np.integer, np.floating)):
                    numeric_preds[m] = float(p)
                elif isinstance(p, (list, np.ndarray)):
                    try:
                        numeric_preds[m] = np.asarray(p, dtype=float)
                    except (ValueError, TypeError):
                        pass

            if numeric_preds:
                fused_pred = self._weighted_average(numeric_preds, weights)
            else:
                fused_pred = self._majority_vote(predictions)

        total_weight = sum(weights.get(m, 0.1) for m in confidences)
        if total_weight > 0 and confidences:
            fused_confidence = sum(
                confidences[m] * weights.get(m, 0.1) for m in confidences
            ) / total_weight
        else:
            fused_confidence = 0.0

        return {
            "prediction": fused_pred if not isinstance(fused_pred, np.ndarray) else fused_pred.tolist(),
            "confidence": round(fused_confidence, 4),
            "weights_used": {m: round(weights.get(m, 0.1), 4) for m in evidence},
            "per_modality": {
                m: {"prediction": predictions.get(m), "confidence": confidences.get(m)}
                for m in set(list(predictions.keys()) + list(confidences.keys()))
            },
            "strategy": strategy_name,
        }

    # ── Confidence-Weighted Fusion ──

    def _compute_confidence_weights(
        self,
        evidence: dict[str, Any],
        task_type: str,
        modality_confidences: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """Compute weights: base_weight[task_type] * confidence * reliability."""
        base_weights = self.task_weights.get(task_type, self.weights)
        modalities = list(evidence.keys())

        if not modality_confidences:
            modality_confidences = {m: 0.5 for m in modalities}

        raw_weights: dict[str, float] = {}
        for m in modalities:
            base = base_weights.get(m, 0.1)
            conf = modality_confidences.get(m, 0.5)
            conf = max(self.confidence_floor, conf)
            raw_weights[m] = base * conf

        total = sum(raw_weights.values())
        if total > 0:
            return {m: w / total for m, w in raw_weights.items()}
        return {m: 1.0 / len(modalities) for m in modalities}

    # ── LLM-Arbitrated Fusion ──

    async def _llm_arbitrated_fuse(
        self,
        evidence: dict[str, Any],
        task_type: str,
        per_modality_results: dict[str, dict[str, Any]],
        task_description: str,
    ) -> dict[str, Any]:
        """Use LLM to arbitrate between modality predictions."""
        if self.llm is None:
            logger.warning("[Fusion] LLM not available for arbitration, falling back to task_adaptive")
            weights = self.task_weights.get(task_type, self.weights)
            return self._weighted_fuse(evidence, weights, strategy_name="llm_arbitrated_fallback")

        summaries = []
        for modality, data in evidence.items():
            pred = data.get("prediction") if isinstance(data, dict) else data
            conf = data.get("confidence", "unknown") if isinstance(data, dict) else "unknown"
            summaries.append(f"- {modality}: prediction={str(pred)[:5000]}, confidence={conf}")

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a multimodal fusion arbiter. Given predictions from different "
                    "modality-specialized analysis branches, determine the best final prediction.\n\n"
                    "Consider the reliability and relevance of each modality for this specific task.\n"
                    "Return JSON:\n"
                    '{"prediction": <final answer>, "confidence": float 0-1, '
                    '"weights": {modality: weight}, '
                    '"reasoning": "why you weighted modalities this way"}'
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Task type: {task_type}\n"
                    f"Description: {task_description}\n\n"
                    f"Modality predictions:\n" + "\n".join(summaries)
                ),
            },
        ]

        try:
            response = await self.llm.complete(messages, response_format="json")
            parsed = response.parsed or {}
            return {
                "prediction": parsed.get("prediction"),
                "confidence": parsed.get("confidence", 0.5),
                "weights_used": parsed.get("weights", {}),
                "per_modality": {m: {"prediction": evidence[m].get("prediction") if isinstance(evidence[m], dict) else evidence[m]} for m in evidence},
                "strategy": "llm_arbitrated",
                "llm_reasoning": parsed.get("reasoning", ""),
            }
        except Exception as e:
            logger.warning(f"[Fusion] LLM arbitration failed: {e}")
            weights = self.task_weights.get(task_type, self.weights)
            result = self._weighted_fuse(evidence, weights, strategy_name="llm_arbitrated_fallback")
            result["llm_error"] = str(e)
            return result

    # ── Helpers ──

    @staticmethod
    def _weighted_average(numeric_preds: dict[str, Any], weights: dict[str, float]) -> Any:
        total_weight = 0.0
        result = None
        for m, p in numeric_preds.items():
            w = weights.get(m, 0.1)
            val = np.asarray(p, dtype=float) * w
            result = val if result is None else result + val
            total_weight += w
        if result is not None and total_weight > 0:
            result = result / total_weight
        return result

    @staticmethod
    def _majority_vote(predictions: dict[str, Any]) -> Any:
        str_preds = [str(p) for p in predictions.values()]
        if not str_preds:
            return None
        counter = Counter(str_preds)
        return counter.most_common(1)[0][0]
