"""Tests for the LLM model router."""

from __future__ import annotations

from timeclaw.core.config import LLMConfig, LLMModelConfig
from timeclaw.llm.router import ModelRouter


class TestModelRouter:
    def test_default_assignments(self):
        router = ModelRouter()
        assignments = router.list_assignments()
        assert "manager" in assignments
        assert "analyzer" in assignments
        assert "fallback" in assignments

    def test_custom_config(self):
        config = LLMConfig(
            models=LLMModelConfig(
                manager="gpt-4o",
                analyzer="gpt-4o-mini",
                executor="gpt-4o-mini",
                reasoner="gpt-4o",
            ),
            fallback="claude-sonnet-4-20250514",
        )
        router = ModelRouter(config)
        assignments = router.list_assignments()
        assert assignments["manager"] == "gpt-4o"
        assert assignments["fallback"] == "claude-sonnet-4-20250514"

    def test_get_backend_caching(self):
        router = ModelRouter()
        b1 = router.get_backend("manager")
        b2 = router.get_backend("manager")
        assert b1 is b2

    def test_different_roles_different_backends(self):
        config = LLMConfig(
            models=LLMModelConfig(manager="model-a", analyzer="model-b")
        )
        router = ModelRouter(config)
        b_manager = router.get_backend("manager")
        b_analyzer = router.get_backend("analyzer")
        assert b_manager.default_model == "model-a"
        assert b_analyzer.default_model == "model-b"
