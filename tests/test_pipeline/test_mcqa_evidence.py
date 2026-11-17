"""A news paragraph is evidence, not a completed multiple-choice branch."""

import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from timeclaw.agents.executor import ExecutorAgent
from timeclaw.pipeline.runner import PipelineRunner
from timeclaw.core.errors import ExecutionContractError


SCOPE = ["weather_qa_short_mcqa"]
NEWS = "A February 2019 flood near the station damaged roads."


def branches(answers):
    return [{"tool": "spawn_subagent", "success": True, "output": [
        {"success": True, "result": {
            "slot_id": i, "answer_type": "qa", "answer": answer,
            "execution_log": [{"tool": "llm_reasoning_qa", "success": True,
                               "output": {"answer": answer}, "observation": {"answer": answer}}],
        }} for i, answer in enumerate(answers)
    ]}, {"tool": "evaluate_against_gt", "success": True,
         "args": {"prediction": "A"},
         "observation": {"metrics": {"accuracy": 1.0}, "ground_truth": "A"}}]


@pytest.mark.parametrize("answer", [NEWS, "Archived event context: flood", {"summary": NEWS}, [1, 2]])
def test_mcqa_rejects_evidence_instead_of_option(answer):
    assert not ExecutorAgent._is_valid_final_answer("qa", answer, scope_keys=SCOPE)
    assert not ExecutorAgent._task_answer_status(task_type="qa", candidate=answer, scope_keys=SCOPE)["available"]
    assert PipelineRunner._normalize_inference_prediction(
        answer, task_type="qa", domain="weather", scope_keys=SCOPE,
    ) is None


def test_scoped_training_contract_does_not_count_news_as_two_answers():
    assert not ExecutorAgent._training_contract_status(
        branches([NEWS, NEWS]), task_type="qa", scope_keys=SCOPE,
    )["satisfied"]
    assert ExecutorAgent._training_contract_status(
        branches(["A", "B"]), task_type="qa", scope_keys=SCOPE,
    )["satisfied"]


def test_freeform_qa_still_accepts_text_and_mcqa_accepts_explicit_option():
    assert ExecutorAgent._is_valid_final_answer("qa", NEWS)
    for answer in ("A", "B", "C", "D", {"answer": "B"}):
        assert ExecutorAgent._is_valid_final_answer("qa", answer, scope_keys=SCOPE)


def test_mcqa_memory_admission_rejects_unfinished_branches():
    with pytest.raises(ExecutionContractError, match="MCQA"):
        PipelineRunner._validate_mcqa_branch_evidence(branches([NEWS, NEWS]))


def test_wrong_but_real_mcqa_answers_remain_admissible():
    log = branches(["B", "C"])
    log[-1]["args"]["prediction"] = "B"
    log[-1]["observation"]["metrics"] = {"accuracy": 0.0}
    PipelineRunner._validate_mcqa_branch_evidence(log)


def test_mcqa_evaluation_cannot_score_an_answer_missing_from_execution():
    with pytest.raises(ExecutionContractError, match="evaluat"):
        PipelineRunner._validate_mcqa_branch_evidence(branches(["B", "C"]))


@pytest.mark.asyncio
async def test_invalid_mcqa_evidence_never_reaches_memory_or_analysis(monkeypatch):
    runner = PipelineRunner(disable_memory=True)
    runner.memory = SimpleNamespace(record=AsyncMock())
    analysis = AsyncMock(side_effect=AssertionError("Unvalidated evidence reached the LM"))
    monkeypatch.setattr("timeclaw.pipeline.runner.generate_learning_analysis", analysis)
    with pytest.raises(ExecutionContractError, match="MCQA"):
        await runner._save_learning_outcome(
            answer_type="learning_summary", raw_answer={"insight": "A was validated"},
            prediction=None, execution_log=branches([NEWS, NEWS]), task_type="qa",
            description="Select A/B/C/D", domain="weather", scope_keys=SCOPE,
            reasoning="", fingerprint={}, profiling_results={}, horizon=None,
            source_metadata={}, trace_id="bad-mcqa-17", ground_truth="A",
        )
    runner.memory.record.assert_not_awaited()
    analysis.assert_not_awaited()


@pytest.mark.asyncio
async def test_real_subexecutor_requests_option_after_reading_news(monkeypatch):
    from timeclaw.clawhub import tool_registry
    from timeclaw.clawhub.text.read_raw_text import ReadRawTextTool
    from timeclaw.clawhub.tool_base import ToolResult
    from timeclaw.core.tea_protocol import EnvironmentSnapshot, TaskEnvelope, TaskSpec

    monkeypatch.setattr(tool_registry, "_TOOL_INSTANCES", {"read_raw_text": ReadRawTextTool()})
    executor = ExecutorAgent(llm=SimpleNamespace())
    calls = 0
    captured = []

    class CorrectionObserved(Exception):
        pass

    async def request(conversation, **kwargs):
        nonlocal calls
        calls += 1
        thought = (
            "Reason: obtain evidence from the tool. Check: use its observed output. "
            "Self-check: no future labels are available. Update: preserve the observed result. "
            "Action: call the tool or return its answer."
        )
        if calls == 1:
            return None, {"done": False, "tool": "read_raw_text", "args": {}, "thought": thought}
        if calls == 2:
            return None, {"done": True, "answer_type": "qa", "answer": NEWS,
                          "confidence": 0.8, "thought": thought,
                          "reasoning": "1. Reuse the read tool output."}
        captured.extend(conversation)
        raise CorrectionObserved

    async def execute(*args, **kwargs):
        return ToolResult(success=True, data=NEWS, error=None)

    monkeypatch.setattr(executor, "_request_structured_decision", request)
    monkeypatch.setattr(executor, "_execute_tool", execute)
    monkeypatch.setattr(executor, "_should_enforce_stepwise_reasoning", lambda mode: False)
    plan = dict(candidates=["read_raw_text"], task_type="qa", task_description="Choose A, B, C, or D.",
                tool_names="read_raw_text", tool_notes_section="", skills_section="", system_memory="",
                modality_hint="", profiling_section="", subagent_slot_section="", subagent_slots=[],
                fingerprint={}, horizon=None, domain="weather", reanalysis_section="", _mode="train",
                _defer_learning_record=False, _ground_truth="A", indicator_output_keys=[],
                modality="multimodal", tool_notes={}, scope_keys=SCOPE)
    envelope = TaskEnvelope(
        task=TaskSpec(task_id="mcqa-contract", task_type="qa", description=plan["task_description"]),
        environment=EnvironmentSnapshot(data=[1.0, 2.0, 3.0], modalities=["timeseries"]),
    )
    with pytest.raises(CorrectionObserved):
        await executor._run_sub_executor(
            idx=0, task_description=plan["task_description"], tools_hint="read_raw_text",
            branch_metadata={}, plan=plan, envelope=envelope,
            slot_id=0, slot_spec={"slot_id": 0, "visible_candidates": ["read_raw_text"]},
        )
    assert calls == 3
    assert any("MCQA answer must" in m["content"] for m in captured if m["role"] == "user")
