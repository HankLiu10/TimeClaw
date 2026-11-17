"""Forecast delivery must preserve every requested time step and channel."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from timeclaw.agents.executor import ExecutorAgent
from timeclaw.clawhub import tool_registry
from timeclaw.clawhub.prediction.arima import ARIMAForecastTool
from timeclaw.clawhub.tool_base import ToolResult
from timeclaw.core.tea_protocol import EnvironmentSnapshot, TaskEnvelope, TaskSpec


def _tool_log(output):
    return [{"tool": "arima_forecast", "success": True, "output": output}]


@pytest.mark.parametrize("length", [95, 97, 103])
def test_forecast_requires_exact_requested_time_length(length):
    answer = list(np.linspace(1, 2, length))
    valid, reason = ExecutorAgent._validate_final_answer("forecast", answer, horizon=96)
    assert not valid
    assert str(length) in reason and "96" in reason and "exactly" in reason
    assert ExecutorAgent._normalize_answer_for_comparison("forecast", answer, horizon=96) is None


def test_observed_103_value_answer_cannot_be_backed_by_96_value_prefix():
    forecast = np.linspace(1, 2, 96)
    answer = np.round(forecast, 4).tolist() + [2.1] * 7
    assert not ExecutorAgent._is_task_answer_backed_by_tool_output(
        "forecast", answer, _tool_log(forecast), horizon=96,
    )
    # The converse must also fail: a tool with an oversized result is invalid.
    assert not ExecutorAgent._is_task_answer_backed_by_tool_output(
        "forecast", forecast, _tool_log(answer), horizon=96,
    )


@pytest.mark.parametrize("wrap", [lambda x: x, lambda x: {"forecast": x}, json.dumps])
def test_exact_rounded_forecast_remains_valid_and_tool_backed(wrap):
    forecast = np.linspace(-1, 2, 96)
    answer = wrap(np.round(forecast, 4).tolist())
    assert ExecutorAgent._is_valid_final_answer("forecast", answer, horizon=96)
    assert ExecutorAgent._is_task_answer_backed_by_tool_output(
        "forecast", answer, _tool_log({"forecast": forecast}), horizon=96,
    )


def test_multivariate_forecast_preserves_time_axis_and_all_channels():
    forecast = np.arange(96 * 3, dtype=float).reshape(96, 3)
    assert ExecutorAgent._is_valid_final_answer("forecast", forecast, horizon=96)
    normalized = ExecutorAgent._normalize_answer_for_comparison("forecast", forecast, horizon=96)
    np.testing.assert_array_equal(normalized, forecast)
    assert ExecutorAgent._answers_match_semantically("forecast", forecast.tolist(), forecast, horizon=96)
    changed = forecast.copy()
    changed[-1, -1] += 1
    assert not ExecutorAgent._answers_match_semantically("forecast", changed, forecast, horizon=96)
    assert not ExecutorAgent._is_valid_final_answer("forecast", forecast.T, horizon=96)
    assert not ExecutorAgent._is_valid_final_answer("forecast", forecast.ravel(), horizon=96)


def test_single_channel_column_and_vector_are_equivalent():
    forecast = np.linspace(1, 2, 96)
    column = forecast[:, None]
    assert ExecutorAgent._is_valid_final_answer("forecast", column, horizon=96)
    assert ExecutorAgent._answers_match_semantically("forecast", column, forecast, horizon=96)
    assert ExecutorAgent._is_task_answer_backed_by_tool_output(
        "forecast", column, _tool_log(forecast), horizon=96,
    )


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("channels", [1, 3])
def test_nonfinite_forecasts_fail_validation_and_backing(bad_value, channels):
    forecast = np.ones((96, channels))
    forecast[-1, -1] = bad_value
    valid, reason = ExecutorAgent._validate_final_answer("forecast", forecast, horizon=96)
    assert not valid and "finite" in reason
    assert ExecutorAgent._normalize_answer_for_comparison("forecast", forecast, horizon=96) is None
    assert not ExecutorAgent._is_task_answer_backed_by_tool_output(
        "forecast", forecast, _tool_log(forecast), horizon=96,
    )


@pytest.mark.parametrize("answer", [1.0, [], np.empty((96, 0)), np.ones((96, 2, 2))])
def test_forecast_rejects_missing_time_axis_or_unsupported_shape(answer):
    assert not ExecutorAgent._is_valid_final_answer("forecast", answer, horizon=96)
    assert ExecutorAgent._normalize_answer_for_comparison("forecast", answer, horizon=96) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("subexecutor", [False, True])
async def test_existing_self_correction_reports_length_reason_in_both_loops(monkeypatch, subexecutor):
    monkeypatch.setattr(tool_registry, "_TOOL_INSTANCES", {"arima_forecast": ARIMAForecastTool()})
    executor = ExecutorAgent(llm=SimpleNamespace())
    forecast = np.linspace(1, 2, 96)
    calls = 0
    captured = []

    class CorrectionObserved(Exception):
        pass

    async def request(conversation, **_kwargs):
        nonlocal calls
        calls += 1
        thought = (
            "Reason: obtain a forecast from the tool. Check: use its observed output. "
            "Self-check: no future labels are available. Update: preserve the observed result. "
            "Action: call the tool or return its forecast."
        )
        if calls == 1:
            return None, {"done": False, "tool": "arima_forecast", "args": {},
                          "thought": thought}
        if calls == 2:
            return None, {"done": True, "answer_type": "forecast", "confidence": 0.8,
                          "answer": forecast.tolist() + [2.1] * 7,
                          "thought": thought,
                          "reasoning": "1. The forecasting tool supplied the prediction."}
        captured.extend(conversation)
        raise CorrectionObserved

    async def execute(*_args, **_kwargs):
        return ToolResult(success=True, data=forecast, error=None)

    monkeypatch.setattr(executor, "_request_structured_decision", request)
    monkeypatch.setattr(executor, "_execute_tool", execute)
    monkeypatch.setattr(executor, "_should_enforce_stepwise_reasoning", lambda _mode: False)
    plan = {
        "candidates": ["arima_forecast"], "task_type": "forecast",
        "task_description": "Forecast 96 time steps.", "tool_names": "arima_forecast",
        "tool_notes_section": "", "skills_section": "", "system_memory": "",
        "modality_hint": "", "profiling_section": "", "subagent_slot_section": "",
        "subagent_slots": [], "fingerprint": {}, "horizon": 96, "domain": "weather",
        "reanalysis_section": "", "_mode": "test", "_defer_learning_record": False,
        "_ground_truth": None, "indicator_output_keys": [], "modality": "timeseries",
        "tool_notes": {},
    }
    envelope = TaskEnvelope(
        task=TaskSpec(task_id="forecast-contract", task_type="forecast", description="Forecast 96 time steps."),
        environment=EnvironmentSnapshot(data=[1.0, 2.0, 3.0], modalities=["timeseries"]),
    )
    with pytest.raises(CorrectionObserved):
        if subexecutor:
            await executor._run_sub_executor(
                idx=0, task_description=plan["task_description"], tools_hint="arima_forecast",
                branch_metadata={}, plan=plan, envelope=envelope,
                slot_id=0, slot_spec={"slot_id": 0, "visible_candidates": ["arima_forecast"]},
            )
        else:
            await executor.act(plan, envelope)
    messages = [item["content"] for item in captured if item["role"] == "user"]
    assert any("103" in text and "96" in text and "exactly" in text for text in messages)
    assert calls == 3
