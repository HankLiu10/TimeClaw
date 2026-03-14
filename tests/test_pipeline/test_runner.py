"""Integration test for the full pipeline runner with mock LLM."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from timeclaw.core.config import LLMConfig, LLMModelConfig, TimeClawConfig
from timeclaw.llm.base import LLMResponse
from timeclaw.pipeline.runner import PipelineRunner


def _mock_llm_response(*args, **kwargs):
    """Return a plausible mock LLM response."""
    return LLMResponse(
        text='{"task_type": "forecast", "sub_goals": ["predict"], "accepted": true, "reasoning": "ok"}',
        parsed={"task_type": "forecast", "sub_goals": ["predict"], "accepted": True, "reasoning": "ok"},
        model="mock",
    )


@pytest.mark.asyncio
async def test_pipeline_end_to_end():
    """Test the full 6-step pipeline with mocked LLM calls."""
    config = TimeClawConfig(
        llm=LLMConfig(
            models=LLMModelConfig(
                manager="mock-model",
                analyzer="mock-model",
                executor="mock-model",
                reasoner="mock-model",
            ),
            fallback=None,
        ),
        workspace_dir="/tmp/timeclaw_test",
    )

    runner = PipelineRunner(config)

    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
        mock_acompletion.side_effect = _make_mock_litellm_response

        np.random.seed(42)
        data = np.arange(100, dtype=float) + np.random.randn(100) * 2

        result = await runner.run(
            data=data,
            task_type="forecast",
            description="Forecast next 10 steps",
            horizon=10,
        )

        assert "trace_id" in result
        assert "timeline" in result
        assert "total_ms" in result
        assert result["total_ms"] > 0

        timeline = result["timeline"]
        assert "plan" in timeline
        assert "diagnose" in timeline
        assert "execute" in timeline
        assert "verify" in timeline
        assert "deliver" in timeline


def _make_mock_litellm_response(*args, **kwargs):
    """Create a mock litellm response object."""
    from unittest.mock import MagicMock

    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = (
        '{"task_type": "forecast", "sub_goals": ["predict"], '
        '"accepted": true, "reasoning": "ok", '
        '"fingerprint": {"basic_stats": {}, "stationarity": "stationary", '
        '"trend": "up", "drift_detected": false}, '
        '"tool_chain": [{"tool": "arima_forecast", "args": {"horizon": 10}}]}'
    )
    response.choices[0].message.tool_calls = None
    response.usage = MagicMock()
    response.usage.prompt_tokens = 100
    response.usage.completion_tokens = 50
    response.usage.total_tokens = 150
    return response
