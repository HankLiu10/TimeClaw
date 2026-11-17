"""Dropout must leave a way to deliver the task's required answer artifact."""

from types import SimpleNamespace

import pytest

from timeclaw.agents.executor import ExecutorAgent
from timeclaw.clawhub import tool_registry
from timeclaw.clawhub.generic_llm.llm_reasoning import (
    LLMReasoningQATool, LLMReasoningCorrelationTool, LLMReasoningIndicatorTool,
)
from timeclaw.clawhub.processing.window_stats import WindowStatsTool
from timeclaw.clawhub.tool_base import ToolResult
from timeclaw.core.tea_protocol import EnvironmentSnapshot, TaskEnvelope, TaskSpec


CASES = [
    ("qa", LLMReasoningQATool, "B", "weather_qa_short_mcqa"),
    ("correlation", LLMReasoningCorrelationTool, "Moderate Positive Correlation", "finance_qa_short_correlation"),
    ("indicator", LLMReasoningIndicatorTool, {"max": [21.0], "min": [10.5], "diff": [10.5]}, "weather_aligned_long_indicator"),
]


@pytest.mark.parametrize("task,tool_class,answer,scope", CASES)
@pytest.mark.asyncio
async def test_hot_answer_tool_remains_usable_in_two_real_dropout_branches(monkeypatch, task, tool_class, answer, scope):
    """Exercise slot sampling -> branch visibility -> real executor -> contract.

    The provider response and tool computation are local stubs; scheduling,
    dropout, answer validation and branch execution use the production code.
    """
    tool = tool_class()
    monkeypatch.setattr(tool_registry, "_TOOL_INSTANCES", {
        tool.name: tool, "window_stats": WindowStatsTool(),
    })
    executor = ExecutorAgent(llm=SimpleNamespace())
    monkeypatch.setattr(executor, "_should_enforce_stepwise_reasoning", lambda mode: False)
    candidates = [tool.name, "window_stats", "slice", "basic_stats"]
    slots = ExecutorAgent._precompute_subagent_slots(
        trace_id="75b4ee890d1a", candidates=candidates, task_type=task,
        run_mode="train", scope=scope,
        scope_counts={tool.name: 999, "window_stats": 999, "slice": 0, "basic_stats": 0},
    )
    # Frequent exploratory tools still drop. No exemption for all tools.
    assert any("window_stats" not in s["visible_candidates"] for s in slots)
    envelope = TaskEnvelope(
        task=TaskSpec(task_id="dropout-real-loop", task_type=task, description="Return the requested task answer."),
        environment=EnvironmentSnapshot(data=[1.0, 2.0, 3.0], modalities=["timeseries"]),
    )
    plan = dict(candidates=candidates, task_type=task, task_description=envelope.task.description,
                tool_names=", ".join(candidates), tool_notes_section="", skills_section="", system_memory="",
                modality_hint="", profiling_section="", subagent_slot_section="", subagent_slots=slots,
                fingerprint={}, horizon=None, domain="weather", reanalysis_section="", _mode="train",
                _defer_learning_record=False, _ground_truth=None,
                indicator_output_keys=["max", "min", "diff"] if task=="indicator" else [],
                modality="multimodal", tool_notes={}, scope_keys=[scope])
    results = []
    for slot in slots[:2]:
        calls = 0

        async def request(conversation, **kwargs):
            nonlocal calls
            calls += 1
            thought = ("Reason: resolve the task from observed evidence. Check: use the tool answer. "
                       "Self-check: no future labels are available. Update: preserve the result. "
                       "Action: call the tool or return its answer.")
            if calls == 1:
                assert tool.name in kwargs["ordinary_tool_names"], "Dropout removed the only typed answer tool"
                return None, {"done": False, "tool": tool.name, "args": {"question": "Decide from evidence"}, "thought": thought}
            assert calls == 2, "A tool-backed task answer should not enter repair loops"
            return None, {"done": True, "answer_type": task, "answer": answer, "confidence": 0.8,
                          "thought": thought, "reasoning": f"1. {tool.name} supplied the observed answer.",
                          "execution_context": {"tool_chain": [tool.name],
                              "public_reasoning": [f"{tool.name} supplied the observed answer."],
                              "evidence_summary": f"{tool.name} supplied the observed answer."}}

        async def execute(*args, **kwargs):
            return ToolResult(success=True, data={"answer": answer}, error=None)

        monkeypatch.setattr(executor, "_request_structured_decision", request)
        monkeypatch.setattr(executor, "_execute_tool", execute)
        result = await executor._run_sub_executor(
            idx=slot["slot_id"], task_description=plan["task_description"], tools_hint=tool.name,
            branch_metadata={}, plan=plan, envelope=envelope, slot_id=slot["slot_id"], slot_spec=slot,
        )
        assert result["answer"] == answer
        assert calls == 2
        results.append({"success": True, "result": result})
    log = [{"tool": "spawn_subagent", "success": True, "output": results},
           {"tool": "evaluate_against_gt", "success": True, "observation": {"metrics": {"accuracy": 1.0}}}]
    assert ExecutorAgent._training_contract_status(
        log, task_type=task, indicator_output_keys=plan["indicator_output_keys"], scope_keys=[scope],
    )["satisfied"]


@pytest.mark.parametrize("label", ["up", "down", "positive", "neutral"])
def test_correlation_output_cannot_be_a_trend_or_sentiment_label(label):
    assert not ExecutorAgent._is_valid_final_answer(
        "correlation", label, scope_keys=["finance_qa_short_correlation"],
    )
