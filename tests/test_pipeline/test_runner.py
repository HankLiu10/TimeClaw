"""Integration test for the full pipeline runner with mock LLM."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from timeclaw.core import trace_logger as trace
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


def test_pipeline_runner_uses_loaded_config_when_none_provided(monkeypatch):
    loaded = TimeClawConfig(
        llm=LLMConfig(
            models=LLMModelConfig(
                manager="loaded-manager",
                analyzer="loaded-analyzer",
                executor="loaded-executor",
                reasoner="loaded-reasoner",
                text_light="loaded-light",
                text_heavy="loaded-heavy",
            ),
            fallback=None,
        ),
        workspace_dir="/tmp/timeclaw_loaded_cfg",
    )

    monkeypatch.setattr(TimeClawConfig, "load", classmethod(lambda cls: loaded))

    runner = PipelineRunner(disable_memory=True)

    assert runner.config is loaded
    assert runner.router.get_backend("executor").default_model == "loaded-executor"
    assert runner.router.get_backend("text_light").default_model == "loaded-light"


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


@pytest.mark.asyncio
async def test_pipeline_cancellation_writes_run_end(tmp_path, monkeypatch):
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
        workspace_dir=str(tmp_path / "workspace"),
    )
    runner = PipelineRunner(config, disable_memory=True)

    monkeypatch.setattr(trace, "_trace_dir", tmp_path / "traces")
    trace.configure(verbose=False, full_dump=False)

    class _FakeTask:
        def __init__(self) -> None:
            self.metadata = {}

    class _FakeEnvelope:
        def __init__(self) -> None:
            self.task = _FakeTask()

        def update_environment(self, **kwargs) -> None:
            return None

        def with_agent(self, agent_name: str):
            return self

    class _BlockingExecutor:
        async def run(self, envelope):
            await asyncio.sleep(3600)

    async def _fake_preprocess(*args, **kwargs):
        return {
            "fingerprint": {"length": 3},
            "modalities": ["timeseries"],
            "profiling_results": {},
        }

    monkeypatch.setattr(runner, "_ensure_tools", AsyncMock())
    monkeypatch.setattr(runner, "_build_envelope", lambda **kwargs: _FakeEnvelope())
    monkeypatch.setattr(runner, "_preprocess", _fake_preprocess)
    monkeypatch.setattr(runner, "_create_executor", lambda: _BlockingExecutor())

    task = asyncio.create_task(
        runner.run(
            data=[1.0, 2.0, 3.0],
            task_type="forecast",
            description="cancel me",
        )
    )
    await asyncio.sleep(0.05)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    trace_files = list((tmp_path / "traces").glob("*.jsonl"))
    assert len(trace_files) == 1
    records = [json.loads(line) for line in trace_files[0].read_text().splitlines()]
    assert records[-1]["event"] == "run_end"
    assert records[-1]["payload"]["status"] == "cancelled"


def test_sanitize_reasoning_clause_keeps_evidence_but_drops_framework_preface():
    cleaned = PipelineRunner._sanitize_reasoning_clause(
        "The required hinted tool was used: entropy returned sample_entropy 0.778, "
        "which suggests the series is predictable enough for a smooth continuation."
    )

    assert "required hinted tool" not in cleaned.lower()
    assert cleaned.startswith("entropy returned sample_entropy 0.778")
    assert "predictable enough" in cleaned


def test_sanitize_reasoning_clause_rewrites_first_person_tool_use():
    cleaned = PipelineRunner._sanitize_reasoning_clause(
        "I used value_at on the input series with index -1 and obtained the latest observed price: 3.95."
    )

    assert "i used value_at" not in cleaned.lower()
    assert cleaned.startswith("value_at was used on the input series")
    assert "latest observed price: 3.95" in cleaned


def test_sanitize_reasoning_clause_strips_required_by_branch_tail():
    cleaned = PipelineRunner._sanitize_reasoning_clause(
        "I used spectral_density as required by the branch."
    )

    assert "i used spectral_density" not in cleaned.lower()
    assert "required by the branch" not in cleaned.lower()
    assert cleaned == "spectral_density was used"


def test_sanitize_reasoning_clause_rewrites_i_ran_tool():
    cleaned = PipelineRunner._sanitize_reasoning_clause(
        "I ran hurst_exponent on the full 130-point series."
    )

    assert "i ran hurst_exponent" not in cleaned.lower()
    assert cleaned == "hurst_exponent was run on the full 130-point series"
