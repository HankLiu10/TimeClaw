"""Integration test for the full pipeline runner with mock LLM."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from timeclaw.core import trace_logger as trace
from timeclaw.core.config import LLMConfig, LLMModelConfig, TimeClawConfig
from timeclaw.core.evidence import EvidenceLog
from timeclaw.clawhub.tool_base import ToolResult
from timeclaw.llm.base import LLMResponse
from timeclaw.pipeline import runner as runner_mod
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

    forecast_values = [float(i) for i in range(10)]
    llm_calls = {"count": 0}

    def _make_executor_response(*args, **kwargs):
        del args, kwargs
        llm_calls["count"] += 1
        if llm_calls["count"] == 1:
            payload = {
                "thought": "Reason: need a direct forecast baseline first. Action: run arima_forecast.",
                "done": False,
                "tool": "arima_forecast",
                "args": {"horizon": 10},
            }
        else:
            payload = {
                "thought": "Reason: arima_forecast already produced the 10-step forecast. Action: finish now.",
                "done": True,
                "answer": forecast_values,
                "answer_type": "forecast",
                "confidence": 0.8,
                "reasoning": (
                    "1. `arima_forecast` produced the final 10-step forecast vector.\n"
                    "2. I return that exact tool output unchanged."
                ),
                "execution_context": {
                    "tool_chain": ["arima_forecast"],
                    "public_reasoning": [
                        "`arima_forecast` produced the final 10-step forecast vector.",
                    ],
                    "evidence_summary": "The final answer reuses the exact `arima_forecast` output.",
                },
                "control_context": {"completion_notes": ""},
            }
        return _make_mock_chat_completion(payload)

    async def _fake_execute_tool(self, tool_name, tool_args, data, text_data, images):
        del self, tool_args, data, text_data, images
        assert tool_name == "arima_forecast"
        return ToolResult(
            success=True,
            data={"forecast": forecast_values},
            error=None,
            metadata={},
        )

    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
        mock_acompletion.side_effect = _make_executor_response
        with patch("timeclaw.agents.executor.ExecutorAgent._execute_tool", new=_fake_execute_tool):

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
            assert "preprocess" in timeline
            assert "execute" in timeline
            assert "postprocess" in timeline
            assert "deliver" in timeline
            assert result["answer"] == forecast_values


def _make_mock_chat_completion(payload: dict) -> SimpleNamespace:
    """Create a minimal chat-completions style response object."""
    response = SimpleNamespace()
    response.choices = [
        SimpleNamespace(
            message=SimpleNamespace(
                content=json.dumps(payload, ensure_ascii=False)
            )
        )
    ]
    response.usage = SimpleNamespace(
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150,
    )
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


@pytest.mark.asyncio
async def test_pipeline_shutdown_closes_router_even_without_memory(monkeypatch):
    runner = PipelineRunner(disable_memory=True)
    runner.router.shutdown = AsyncMock()

    await runner.shutdown()

    runner.router.shutdown.assert_awaited_once()


@pytest.mark.asyncio
async def test_pipeline_runner_accepts_numpy_array_executor_answer(monkeypatch):
    runner = PipelineRunner(disable_memory=True)

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

    class _ArrayExecutor:
        def __init__(self) -> None:
            self.evidence_log = EvidenceLog()

        async def run(self, envelope):
            return {
                "result": {
                    "answer": np.array([1.0, 2.0, 3.0], dtype=float),
                    "answer_type": "forecast",
                    "confidence": 0.9,
                    "reasoning": "array answer",
                    "execution_log": [],
                }
            }

    async def _fake_preprocess(*args, **kwargs):
        return {
            "fingerprint": {"length": 3},
            "modalities": ["timeseries"],
            "profiling_results": {},
        }

    monkeypatch.setattr(runner, "_ensure_tools", AsyncMock())
    monkeypatch.setattr(runner, "_build_envelope", lambda **kwargs: _FakeEnvelope())
    monkeypatch.setattr(runner, "_preprocess", _fake_preprocess)
    monkeypatch.setattr(runner, "_create_executor", lambda: _ArrayExecutor())

    result = await runner.run(
        data=[1.0, 2.0, 3.0],
        task_type="forecast",
        description="array answer path",
        horizon=3,
    )

    assert result["answer_type"] == "forecast"
    assert np.allclose(np.asarray(result["prediction"], dtype=float), np.array([1.0, 2.0, 3.0]))


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


@pytest.mark.asyncio
async def test_rewrite_learning_summary_from_analysis_replaces_control_flow_wording(monkeypatch):
    runner = PipelineRunner(disable_memory=True)

    async def _fake_generate(**_kwargs):
        return (
            {
                "sample_conditions": ["recent window is flat with low volatility"],
                "winner_evidence": ["llm_reasoning_classification matched the flat window better than detect_trend"],
                "failure_signals": ["detect_trend over-read a weak uptick"],
                "reusable_rule": ["When the recent window is flat, consider llm_reasoning_classification before detect_trend."],
            },
            "ignored",
        )

    monkeypatch.setattr(runner_mod, "generate_learning_analysis", _fake_generate)

    rewritten, structured, _summary_text = await runner._rewrite_learning_summary_from_analysis(
        {
            "insight": "The strongest next step is to reuse the task-valid answer.",
            "recommendation": "Return the saved payload unchanged.",
            "winner_candidate": "llm_reasoning_classification",
        },
        execution_log=[
            {
                "tool": "spawn_subagent",
                "output": [
                    {
                        "success": True,
                        "result": {
                            "answer": "stable",
                            "execution_log": [
                                {"tool": "llm_reasoning_classification", "observation": {"answer": "stable"}}
                            ],
                        },
                    },
                    {
                        "success": True,
                        "result": {
                            "answer": "up",
                            "execution_log": [
                                {"tool": "detect_trend", "observation": {"trend": "up"}}
                            ],
                        },
                    },
                ],
            },
            {
                "tool": "evaluate_batch_against_gt",
                "observation": {
                    "results": [
                        {"name": "llm_reasoning_classification", "metrics": {"accuracy": 1.0}},
                        {"name": "detect_trend", "metrics": {"accuracy": 0.0}},
                    ],
                },
            },
        ],
        task_type="classification",
        description="trend classification task",
        domain="finance",
        scope_keys=["finance_aligned_short_trend"],
        fingerprint={"length": 96, "trend": "flat"},
        profiling_results={},
        horizon=None,
        reasoning="1. Saved answer already exists.",
    )

    assert structured is not None
    combined = f"{rewritten['insight']} {rewritten['recommendation']}".lower()
    assert "task-valid" not in combined
    assert "saved payload" not in combined
    assert "recent window is flat" in combined


@pytest.mark.asyncio
async def test_rewrite_learning_summary_from_analysis_handles_no_validated_winner(monkeypatch):
    runner = PipelineRunner(disable_memory=True)

    async def _fake_generate(**_kwargs):
        return (
            {
                "sample_conditions": ["daily means sit close to the stability threshold"],
                "winner_evidence": [],
                "failure_signals": [
                    "explored classification paths all stayed too close to the threshold and flipped labels"
                ],
                "reusable_rule": [
                    "When all explored label paths stay near the decision threshold, verify the aggregation method before trusting any candidate."
                ],
            },
            "ignored",
        )

    monkeypatch.setattr(runner_mod, "generate_learning_analysis", _fake_generate)

    rewritten, structured, _summary_text = await runner._rewrite_learning_summary_from_analysis(
        {
            "insight": "Both explored paths looked reasonable, but GT showed they were wrong.",
            "recommendation": "Treat this as a failure pattern rather than a reusable win.",
            "winner_candidate": "",
            "winner_prediction": None,
            "winner_tool_chain": [],
            "winner_metrics": {},
            "winner_state": "none",
        },
        execution_log=[
            {
                "tool": "spawn_subagent",
                "output": [
                    {
                        "success": True,
                        "result": {
                            "answer": "stable",
                            "execution_log": [
                                {"tool": "segment"},
                                {"tool": "llm_reasoning_classification"},
                            ],
                        },
                    },
                    {
                        "success": True,
                        "result": {
                            "answer": "decreasing",
                            "execution_log": [
                                {"tool": "llm_reasoning_classification"},
                            ],
                        },
                    },
                ],
            },
            {
                "tool": "evaluate_against_gt",
                "args": {"prediction": "stable"},
                "observation": {"metrics": {"accuracy": 0.0, "correct": False}},
            },
            {
                "tool": "evaluate_against_gt",
                "args": {"prediction": "decreasing"},
                "observation": {"metrics": {"accuracy": 0.0, "correct": False}},
            },
        ],
        task_type="classification",
        description="trend classification task",
        domain="weather",
        scope_keys=["weather_aligned_short_trend"],
        fingerprint={"length": 96, "trend": "flat"},
        profiling_results={},
        horizon=None,
        reasoning="1. Both explored labels were wrong.",
    )

    assert structured is not None
    assert rewritten["winner_candidate"] == ""
    assert rewritten["winner_prediction"] is None
    assert rewritten["winner_tool_chain"] == []
    assert rewritten["winner_metrics"] == {}
    assert rewritten["winner_state"] == "none"
    assert "No explored candidate was validated against ground truth" in rewritten["insight"]
    assert rewritten["recommendation"].startswith(
        "When all explored label paths stay near the decision threshold"
    )
    assert structured["winner_evidence"] == []


def test_sanitize_reasoning_clause_drops_packaging_only_language():
    cleaned = PipelineRunner._sanitize_reasoning_clause(
        "Returning the corrected final JSON with unchanged saved label and explicit numbered tool-by-tool reasoning."
    )

    assert cleaned == ""


def test_sanitize_reasoning_clause_drops_execution_meta_language():
    cleaned = PipelineRunner._sanitize_reasoning_clause(
        "A task-valid indicator answer already exists, but the latest instruction requires an immediate tool call before yielding."
    )

    assert cleaned == ""


@pytest.mark.asyncio
async def test_pipeline_runner_coerces_non_numeric_confidence_before_result_message(monkeypatch):
    config = TimeClawConfig(workspace_dir="/tmp/timeclaw_confidence_test")
    runner = PipelineRunner(config, disable_memory=True)

    async def _fake_ensure_tools():
        return None

    async def _fake_preprocess(_envelope, _data, _text_data, _images):
        return {
            "fingerprint": {},
            "modalities": ["timeseries"],
            "profiling_results": {},
        }

    class _FakeExecutor:
        def __init__(self):
            self.evidence_log = SimpleNamespace(summary=lambda: "execution summary")

        async def run(self, _envelope):
            return {
                "result": {
                    "answer": "up",
                    "answer_type": "classification",
                    "confidence": "high",
                    "reasoning": "1. `llm_reasoning_classification` returned up.",
                    "execution_context": {
                        "tool_chain": ["llm_reasoning_classification"],
                        "public_reasoning": [
                            "`llm_reasoning_classification` returned up.",
                        ],
                        "evidence_summary": "The classifier output was up.",
                    },
                    "control_context": {"completion_notes": ""},
                    "execution_log": [
                        {"tool": "llm_reasoning_classification", "success": True},
                    ],
                }
            }

    monkeypatch.setattr(runner, "_ensure_tools", _fake_ensure_tools)
    monkeypatch.setattr(runner, "_preprocess", _fake_preprocess)
    monkeypatch.setattr(runner, "_create_executor", lambda: _FakeExecutor())
    monkeypatch.setattr(
        runner._quality_gate,
        "check",
        lambda prediction, *_args, **_kwargs: SimpleNamespace(
            accepted=True,
            rejections=[],
        ),
    )

    result = await runner.run(
        data=np.asarray([1.0, 2.0, 3.0]),
        task_type="classification",
        description="Classify the direction",
    )

    assert result["answer"] == "up"
    assert result["confidence"] is None
    assert result["result_message"].confidence is None
