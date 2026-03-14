"""Model router: per-agent model assignment, fallback chains, and load balancing."""

from __future__ import annotations

from typing import Any

from loguru import logger

from timeclaw.core.config import LLMConfig, LLMModelConfig
from timeclaw.llm.base import LLMBackend


class ModelRouter:
    """Routes LLM requests to the appropriate model based on agent role."""

    def __init__(self, config: LLMConfig | None = None):
        self.config = config or LLMConfig()
        self._backends: dict[str, LLMBackend] = {}

    def get_backend(self, role: str) -> LLMBackend:
        """Get or create an LLMBackend configured for the given agent role."""
        if role in self._backends:
            return self._backends[role]

        model = self._resolve_model(role)
        backend = LLMBackend(
            default_model=model,
            fallback_model=self.config.fallback,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            max_retries=self.config.max_retries,
            timeout=self.config.timeout,
            api_base=self.config.api_base,
            api_key=self.config.api_key,
        )
        self._backends[role] = backend
        logger.info(f"Router: {role} -> {model} (fallback: {self.config.fallback})")
        return backend

    def get_text_backends(self) -> dict[str, LLMBackend] | None:
        """Return {"light": ..., "heavy": ...} backends for text tools, or None."""
        models = self.config.models
        if not models.text_light and not models.text_heavy:
            return None
        result: dict[str, LLMBackend] = {}
        if models.text_light:
            result["light"] = self.get_backend("text_light")
        if models.text_heavy:
            result["heavy"] = self.get_backend("text_heavy")
        return result if result else None

    def _resolve_model(self, role: str) -> str:
        """Resolve the model name for a given role from config."""
        models: LLMModelConfig = self.config.models
        model_map = {
            "manager": models.manager,
            "analyzer": models.analyzer,
            "executor": models.executor,
            "reasoner": models.reasoner,
            "text_light": models.text_light or models.analyzer,
            "text_heavy": models.text_heavy or models.manager,
        }
        return model_map.get(role, models.manager)

    def list_assignments(self) -> dict[str, str]:
        """Return current role -> model assignments."""
        models = self.config.models
        assignments = {
            "manager": models.manager,
            "analyzer": models.analyzer,
            "executor": models.executor,
            "reasoner": models.reasoner,
            "fallback": self.config.fallback or "(none)",
        }
        if models.text_light:
            assignments["text_light"] = models.text_light
        if models.text_heavy:
            assignments["text_heavy"] = models.text_heavy
        return assignments
