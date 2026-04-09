from types import SimpleNamespace

import pytest

from timeclaw.benchmark.dataset_split import load_split
from timeclaw.benchmark.mtbench_support import build_task_description
from timeclaw.agents.prompts import executor_prompt
from timeclaw.core.config import TimeClawConfig
from timeclaw.pipeline.runner import PipelineRunner


class _FakeExecutor:
    def __init__(self):
        self.evidence_log = SimpleNamespace(summary=lambda: "ok")
        self.seen_metadata = None

    async def run(self, envelope):
        self.seen_metadata = dict(envelope.task.metadata)
        return {
            "result": {
                "answer": [1.0],
                "answer_type": "json",
                "confidence": 0.9,
                "reasoning": "ok",
                "execution_log": [{"tool": "compute_macd", "success": True}],
            }
        }


@pytest.mark.asyncio
async def test_pipeline_injects_multi_scope_memory_tool_and_skill(monkeypatch):
    runner = PipelineRunner(TimeClawConfig())
    fake_executor = _FakeExecutor()
    seen = {}

    async def fake_ensure_tools():
        return None

    async def fake_preprocess(envelope, data, text_data, images):
        return {
            "fingerprint": {"kind": "test"},
            "modalities": ["timeseries"],
            "profiling_results": {},
        }

    def fake_create_executor():
        return fake_executor

    def fake_get_tool_notes(domain="", task="", tool_names=None, scope_keys=None):
        seen["tool_scope_keys"] = scope_keys
        return {"compute_macd": "scope aware"}

    def fake_get_skill(domain, task, scope_keys=None):
        seen["skill_scope_keys"] = scope_keys
        return "skill text"

    monkeypatch.setattr(runner, "_ensure_tools", fake_ensure_tools)
    monkeypatch.setattr(runner, "_preprocess", fake_preprocess)
    monkeypatch.setattr(runner, "_create_executor", fake_create_executor)
    monkeypatch.setattr(runner._quality_gate, "check", lambda prediction, evidence, constraints: SimpleNamespace(accepted=True, rejections=[]))
    monkeypatch.setattr(runner.memory, "get_tool_notes", fake_get_tool_notes)
    monkeypatch.setattr(runner.memory, "get_skill", fake_get_skill)

    result = await runner.run(
        data=[1.0, 2.0, 3.0],
        task_type="indicator",
        description="test scope injection",
        domain="finance",
        scope_keys=[
            "finance_aligned_short_indicator_macd",
            "finance_aligned_short_indicator_upper_bb",
        ],
    )

    assert result["prediction"] == [1.0]
    assert seen["tool_scope_keys"] == [
        "finance_aligned_short_indicator_macd",
        "finance_aligned_short_indicator_upper_bb",
    ]
    assert seen["skill_scope_keys"] == [
        "finance_aligned_short_indicator_macd",
        "finance_aligned_short_indicator_upper_bb",
    ]
    assert fake_executor.seen_metadata["memory_l1"] == ""
    assert fake_executor.seen_metadata["tool_notes"] == {"compute_macd": "scope aware"}
    assert fake_executor.seen_metadata["skill_prompt"] == "skill text"


def test_modality_filtered_tool_names_respect_required_modalities():
    from timeclaw.clawhub.tool_registry import discover_and_register

    discover_and_register()
    text_tools = set(executor_prompt.get_all_tool_names(["timeseries", "text"]))
    ts_only_tools = set(executor_prompt.get_all_tool_names(["timeseries"]))

    assert "llm_reasoning_forecast" in text_tools
    assert "llm_reasoning_forecast" in ts_only_tools
    assert "llm_reasoning_indicator" in ts_only_tools
    assert "temporal_align_text" in text_tools
    assert "temporal_align_text" not in ts_only_tools


def test_executor_tool_catalog_includes_descriptions_and_required_args():
    from timeclaw.clawhub.tool_registry import discover_and_register

    discover_and_register()
    catalog = executor_prompt.build_tool_name_text(
        ["window_stats", "spawn_subagent", "evaluate_against_gt", "evaluate_batch_against_gt"]
    )

    assert "- window_stats:" in catalog
    assert "Required args:" in catalog
    assert "start<integer> required" in catalog
    assert "end<integer> required" in catalog
    assert "- spawn_subagent:" in catalog
    assert "tasks<array> required" in catalog
    assert "- evaluate_against_gt:" in catalog
    assert "task_type<string> required" in catalog
    assert "- evaluate_batch_against_gt:" in catalog
    assert "framework will auto-fill them" in catalog


def test_build_task_description_uses_full_prompt_template():
    sample = load_split("weather_aligned_short", "test")[0]
    description = build_task_description(sample, "forecast")

    assert description.startswith("User prompt:\nYou are a weather forecasting AI.")
    assert "The input time series represents temperature readings" in description
    assert "Predicted Temperatures: value1, value2, ..., valueN." in description


def test_build_task_description_finance_indicator_tasks_are_split():
    sample = load_split("finance_aligned_long", "test")[0]
    macd_description = build_task_description(sample, "indicator_macd")
    bb_description = build_task_description(sample, "indicator_upper_bb")

    assert "Predict the future Moving Average Convergence Divergence (MACD) values" in macd_description
    assert "estimated upper Bollinger Band (BB) values values" not in macd_description
    assert "estimated upper Bollinger Band (BB) values values" not in bb_description
    assert "estimated upper Bollinger Band (BB) values" in bb_description
    assert "Predict the future Moving Average Convergence Divergence (MACD) values" not in bb_description
