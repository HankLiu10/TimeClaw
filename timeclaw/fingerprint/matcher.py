"""Fingerprint matcher: routes to best model/checkpoint based on data characteristics.

Supports multimodal-aware routing and SBM model profile matching with OOD fallback.
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from timeclaw.fingerprint.schema import DataFingerprint


class FingerprintMatcher:
    """Matches a DataFingerprint to the best available model or strategy.

    Supports:
    - SBM model profile registration and matching
    - Multimodal-aware routing (text/image modalities influence tool chain)
    - OOD fallback path when no profile matches well
    """

    OOD_THRESHOLD = 0.3

    def __init__(self) -> None:
        self._profiles: dict[str, dict[str, Any]] = {}
        self._sbm_profiles: dict[str, dict[str, Any]] = {}

    def register_profile(self, name: str, profile: dict[str, Any]) -> None:
        """Register a model/checkpoint profile for matching."""
        self._profiles[name] = profile

    def register_sbm_profile(self, name: str, profile: dict[str, Any]) -> None:
        """Register an SBM model profile from SBMBase.get_profile()."""
        self._sbm_profiles[name] = profile

    def register_sbm_from_registry(self) -> None:
        """Auto-register all SBM profiles from the global registry."""
        try:
            from timeclaw.sbm.registry import get_global_registry

            registry = get_global_registry()
            for profile in registry.get_profiles():
                self._sbm_profiles[profile["name"]] = profile
        except ImportError:
            logger.debug("SBM registry not available")

    def match(
        self,
        fingerprint: DataFingerprint,
        modalities: list[str] | None = None,
        task_type: str = "forecast",
    ) -> str:
        """Find the best matching model/strategy for the given fingerprint.

        Args:
            fingerprint: Data characteristics fingerprint.
            modalities: Active modalities (e.g., ["timeseries", "text", "image"]).
            task_type: Task type (forecast, anomaly, classification, imputation, qa).

        Returns:
            Strategy/model name string.
        """
        modalities = modalities or ["timeseries"]

        if self._is_multimodal_qa(modalities, task_type):
            return self._multimodal_route(modalities, task_type)

        sbm_match = self._match_sbm(fingerprint, task_type)
        if sbm_match:
            return sbm_match

        if self._profiles:
            best_name, best_score = self._match_profiles(fingerprint)
            if best_score >= self.OOD_THRESHOLD:
                logger.info(f"Matched fingerprint to '{best_name}' (score={best_score:.3f})")
                return best_name

        logger.info("No good match found, entering OOD fallback path")
        return self._ood_fallback(fingerprint, modalities)

    def _match_sbm(self, fingerprint: DataFingerprint, task_type: str) -> str | None:
        """Try to match to an SBM model based on task and data characteristics."""
        if not self._sbm_profiles:
            self.register_sbm_from_registry()

        candidates = []
        for name, profile in self._sbm_profiles.items():
            tasks = profile.get("tasks", [])
            if task_type in tasks:
                score = self._sbm_score(fingerprint, profile)
                candidates.append((name, score))

        if not candidates:
            return None

        candidates.sort(key=lambda x: x[1], reverse=True)
        best_name, best_score = candidates[0]

        if best_score >= self.OOD_THRESHOLD:
            logger.info(f"SBM match: '{best_name}' (score={best_score:.3f})")
            return f"sbm:{best_name}"

        return None

    def _sbm_score(self, fingerprint: DataFingerprint, profile: dict[str, Any]) -> float:
        """Score an SBM profile against the fingerprint."""
        score = 0.5  # base score for any matching SBM

        ctx_len = profile.get("context_length", 512)
        if fingerprint.length <= ctx_len:
            score += 0.2
        else:
            score += 0.1

        if fingerprint.num_channels == 1:
            score += 0.1
        elif fingerprint.num_channels <= 8:
            score += 0.05

        return min(score, 1.0)

    def _match_profiles(self, fingerprint: DataFingerprint) -> tuple[str, float]:
        """Match against registered checkpoint profiles."""
        best_name = ""
        best_score = -1.0

        for name, profile in self._profiles.items():
            score = self._compute_score(fingerprint, profile)
            if score > best_score:
                best_score = score
                best_name = name

        return best_name, best_score

    @staticmethod
    def _compute_score(fingerprint: DataFingerprint, profile: dict[str, Any]) -> float:
        score = 0.0
        total = 0.0

        if "stationarity" in profile:
            total += 1.0
            if fingerprint.stationarity.conclusion == profile["stationarity"]:
                score += 1.0

        if "trend" in profile:
            total += 1.0
            if fingerprint.trend.direction == profile["trend"]:
                score += 1.0

        if "min_length" in profile:
            total += 1.0
            if fingerprint.length >= profile["min_length"]:
                score += 1.0

        if "max_channels" in profile:
            total += 1.0
            if fingerprint.num_channels <= profile["max_channels"]:
                score += 1.0

        return score / total if total > 0 else 0.0

    @staticmethod
    def _is_multimodal_qa(modalities: list[str], task_type: str) -> bool:
        return task_type == "qa" or (len(modalities) > 1 and "text" in modalities)

    @staticmethod
    def _multimodal_route(modalities: list[str], task_type: str) -> str:
        """Route multimodal tasks to appropriate tool chains."""
        if task_type == "qa":
            if "image" in modalities:
                return "multimodal_qa:text+image+timeseries"
            return "multimodal_qa:text+timeseries"

        if "image" in modalities and "text" in modalities:
            return "multimodal_fusion:all"
        if "text" in modalities:
            return "multimodal_fusion:text+timeseries"
        if "image" in modalities:
            return "multimodal_fusion:image+timeseries"

        return "sbm_forecast"

    @staticmethod
    def _ood_fallback(fingerprint: DataFingerprint, modalities: list[str]) -> str:
        """OOD fallback: SBM zero-shot + statistical tools + logic constraints.

        When no profile matches well, we fall back to a safe pipeline:
        1. SBM zero-shot inference (if available)
        2. Traditional statistical tools (ARIMA, ETS)
        3. Hard logic constraints via SafeAuto
        """
        components = []

        components.append("sbm_forecast")

        if fingerprint.stationarity.conclusion == "stationary":
            components.append("arima_forecast")
        else:
            components.append("stl_decompose")
            components.append("arima_forecast")

        components.append("fol_check")

        if len(modalities) > 1:
            if "text" in modalities:
                components.append("sentiment")
                components.append("text_to_event")
            if "image" in modalities:
                components.append("image_feature")

        return "ood_fallback:" + "+".join(components)
