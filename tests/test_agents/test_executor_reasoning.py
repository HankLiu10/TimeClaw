import json
from types import SimpleNamespace

import pytest

import timeclaw.agents.executor as executor_module
from timeclaw.agents.executor import ExecutorAgent
from timeclaw.agents.prompts import executor_prompt
from timeclaw.clawhub.tool_base import ToolResult
from timeclaw.core.tea_protocol import EnvironmentSnapshot, TaskEnvelope, TaskSpec
from timeclaw.llm.base import LLMResponse


def _decision_response(parsed: dict) -> LLMResponse:
    if parsed.get("done", False):
        args = {
            "thought": parsed.get("thought", ""),
            "answer_type": parsed.get("answer_type", ""),
            "answer_json": json.dumps(parsed.get("answer"), ensure_ascii=False),
            "confidence": parsed.get("confidence"),
            "reasoning": parsed.get("reasoning", ""),
        }
        return LLMResponse(
            text="",
            parsed=None,
            tool_calls=[{"name": "emit_final_answer", "arguments": args}],
            model="mock",
        )
    return LLMResponse(
        text="",
        parsed=None,
        tool_calls=[{
            "name": "emit_tool_call",
            "arguments": {
                "thought": parsed.get("thought", ""),
                "tool": parsed.get("tool", ""),
                "args": parsed.get("args", {}),
            },
        }],
        model="mock",
    )


async def _complete_with_tools_via_complete(llm, *args, **kwargs) -> LLMResponse:
    resp = await llm.complete(*args, **kwargs)
    if isinstance(resp, LLMResponse) and resp.tool_calls:
        return resp
    parsed = getattr(resp, "parsed", None)
    return _decision_response(parsed)


def test_has_stepwise_reasoning_requires_numbered_tool_steps():
    execution_log = [
        {"tool": "chronos2_forecast"},
        {"tool": "consistency_check"},
    ]
    reasoning = (
        "1. chronos2_forecast was chosen because the task needed a 33-step forecast; "
        "it produced a stable downward path, so I had a concrete candidate to validate.\n"
        "2. consistency_check was chosen to verify plausibility; it showed the forecast "
        "remained continuous with history, so I kept that candidate as the final answer."
    )
    assert ExecutorAgent._has_stepwise_reasoning(reasoning, execution_log) is True


def test_coerce_llm_decision_rejects_pseudo_tool_transcript_after_json():
    raw = (
        '{"thought":"Reason: call tool. Action: run it","done":false,"tool":"llm_reasoning_qa","args":{"question":"q"}}'
        'assistant to=llm_reasoning_qa {"question":"q"}'
    )

    decision = ExecutorAgent._coerce_llm_decision(
        parsed=None,
        text=raw,
        fallback_thought="parse failed",
    )

    assert decision["_parse_failed"] is True
    assert decision["thought"] == "json_followed_by_pseudo_tool_transcript"


def test_invalid_json_self_correction_mentions_field_mixing_for_multiple_json():
    msg = ExecutorAgent._invalid_json_self_correction_message(
        "multiple_json_objects_in_one_reply"
    )
    assert "Do not append a second JSON object" in msg
    assert "Do not mix tool-call fields" in msg
    assert "`answer_type`" in msg


def test_build_state_update_message_includes_current_space_rule():
    current_state = ExecutorAgent._initial_series_state([1.0, 2.0, 3.0])
    latest_state = current_state

    msg = ExecutorAgent._build_state_update_message(
        current_series_state=current_state,
        latest_artifact_state=latest_state,
    )

    assert "## Indexing Rule" in msg
    assert "current_series_state.coordinate_system.current_space" in msg
    assert "Treat `origin_space` as provenance only" in msg


def test_window_stats_description_mentions_origin_space_example():
    from timeclaw.clawhub.processing.window_stats import tool as window_stats_tool

    assert "origin_space=[144,168)" in window_stats_tool.description
    assert "start=0, end=24" in window_stats_tool.description


def test_initial_series_state_uses_general_coordinate_schema():
    state = ExecutorAgent._initial_series_state([1.0, 2.0, 3.0, 4.0])

    assert state["artifact_type"] == "series"
    assert state["producer_tool"] == "initial_input"
    assert state["shape"] == {"length": 4}
    assert state["coordinate_system"] == {
        "indexing": "positional",
        "current_space": [0, 4],
        "origin_space": [0, 4],
        "relative_to": "original_input",
    }
    assert state["derivation"] == {"operation": "identity", "args": {}}
    assert "current_index_range" not in state
    assert "absolute_index_range_in_original" not in state


def test_slice_artifact_state_uses_origin_space_instead_of_slice_specific_fields():
    previous_state = ExecutorAgent._initial_series_state(list(range(128)))

    state = ExecutorAgent._artifact_state_from_result(
        tool_name="slice",
        result_data=[79, 80, 81],
        downstream_data=list(range(48)),
        previous_series_state=previous_state,
        tool_args={"start": 79, "end": 127},
    )

    assert state["artifact_type"] == "series"
    assert state["producer_tool"] == "slice"
    assert state["parent_artifact_type"] == "series"
    assert state["shape"] == {"length": 48}
    assert state["coordinate_system"] == {
        "indexing": "positional",
        "current_space": [0, 48],
        "origin_space": [79, 127],
        "relative_to": "original_input",
    }
    assert state["derivation"] == {
        "operation": "slice",
        "args": {"start": 79, "end": 127},
    }
    assert "slice_relative_to_current_before_slice" not in state
    assert "current_index_range" not in state
    assert "absolute_index_range_in_original" not in state


def test_has_stepwise_reasoning_rejects_brief_summary():
    execution_log = [
        {"tool": "chronos2_forecast"},
        {"tool": "consistency_check"},
    ]
    reasoning = (
        "The forecast looked reasonable and passed consistency checks, so I used it."
    )
    assert ExecutorAgent._has_stepwise_reasoning(reasoning, execution_log) is False


def test_has_stepwise_reasoning_accepts_candidate_based_main_reasoning():
    execution_log = [
        {
            "tool": "spawn_subagent",
            "output": [
                {
                    "success": True,
                    "result": {
                        "execution_log": [{"tool": "arima_forecast"}],
                    },
                },
                {
                    "success": True,
                    "result": {
                        "execution_log": [{"tool": "ets_forecast"}],
                    },
                },
            ],
        },
        {"tool": "evaluate_batch_against_gt"},
    ]
    reasoning = (
        "1. I compared the arima_forecast and ets_forecast candidate paths because they "
        "captured different decline shapes for the target indicator.\n"
        "2. ETS matched the realized trajectory more closely than ARIMA, so I selected "
        "the ETS-based summary and recommendation as the final result."
    )
    assert ExecutorAgent._has_stepwise_reasoning(reasoning, execution_log) is True


def test_has_stepwise_reasoning_accepts_candidate_aliases_not_exact_tool_ids():
    execution_log = [
        {
            "tool": "spawn_subagent",
            "output": [
                {
                    "success": True,
                    "result": {
                        "execution_log": [{"tool": "arima_forecast"}],
                    },
                },
                {
                    "success": True,
                    "result": {
                        "execution_log": [{"tool": "ets_forecast"}],
                    },
                },
                {
                    "success": True,
                    "result": {
                        "execution_log": [{"tool": "llm_reasoning_indicator"}],
                    },
                },
            ],
        },
        {"tool": "evaluate_batch_against_gt"},
    ]
    reasoning = (
        "1. ARIMA-based forecasting produced a path that flattened too quickly.\n"
        "2. ETS-based forecasting preserved the smooth downward regime more faithfully.\n"
        "3. Direct multimodal LLM indicator reasoning stayed too elevated relative to the technical trend.\n"
        "4. The decisive evidence was that ETS matched the realized downward shape most closely."
    )
    assert ExecutorAgent._has_stepwise_reasoning(reasoning, execution_log) is True


def test_has_stepwise_reasoning_rejects_orchestration_tool_names():
    execution_log = [
        {
            "tool": "spawn_subagent",
            "output": [
                {
                    "success": True,
                    "result": {
                        "execution_log": [{"tool": "arima_forecast"}],
                    },
                },
                {
                    "success": True,
                    "result": {
                        "execution_log": [{"tool": "ets_forecast"}],
                    },
                },
            ],
        },
        {"tool": "evaluate_batch_against_gt"},
    ]
    reasoning = (
        "1. I used spawn_subagent to create candidates.\n"
        "2. I used evaluate_batch_against_gt to choose the winner."
    )
    assert ExecutorAgent._has_stepwise_reasoning(reasoning, execution_log) is False


def test_count_training_candidates_reads_spawn_result_list():
    execution_log = [
        {
            "tool": "spawn_subagent",
            "output": [
                {
                    "success": True,
                    "result": {
                        "answer": "No Correlation",
                        "execution_log": [{"tool": "llm_reasoning_correlation"}],
                    },
                },
                {
                    "success": True,
                    "result": {
                        "answer": "Moderate Positive Correlation",
                        "execution_log": [{"tool": "window_stats"}],
                    },
                },
            ],
        }
    ]
    assert ExecutorAgent._count_training_candidates(execution_log, task_type="correlation") == 2


def test_count_training_candidates_reads_spawn_result_dict_wrapper():
    execution_log = [
        {
            "tool": "spawn_subagent",
            "output": {
                "n_tasks": 2,
                "results": [
                    {
                        "success": True,
                        "result": {
                            "answer": "No Correlation",
                            "execution_log": [{"tool": "llm_reasoning_correlation"}],
                        },
                    },
                    {
                        "success": True,
                        "result": {
                            "answer": "Moderate Positive Correlation",
                            "execution_log": [{"tool": "window_stats"}],
                        },
                    },
                ],
            },
        }
    ]
    assert ExecutorAgent._count_training_candidates(execution_log, task_type="correlation") == 2


def test_count_training_candidates_keeps_distinct_branches_even_if_tool_names_match():
    execution_log = [
        {
            "tool": "spawn_subagent",
            "output": [
                {
                    "success": True,
                    "result": {
                        "slot_id": 0,
                        "tools_hint": "llm_reasoning_classification",
                        "sub_task_description": "branch a",
                        "answer": "<-4%",
                        "execution_log": [],
                    },
                },
                {
                    "success": True,
                    "result": {
                        "slot_id": 1,
                        "tools_hint": "sentiment, llm_reasoning_classification",
                        "sub_task_description": "branch b",
                        "answer": "<-4%",
                        "execution_log": [{"tool": "llm_reasoning_classification"}],
                    },
                },
            ],
        }
    ]

    assert ExecutorAgent._count_training_candidates(
        execution_log,
        task_type="classification",
    ) == 2


def test_count_training_candidates_includes_top_level_task_valid_tools():
    qa_execution_log = [
        {
            "tool": "llm_reasoning_qa",
            "success": True,
            "observation": {"answer": "A"},
        },
        {
            "tool": "window_stats",
            "success": True,
            "observation": {"min": 1.0, "max": 2.0},
        },
    ]

    assert ExecutorAgent._count_training_candidates(
        qa_execution_log,
        task_type="qa",
    ) == 1

    correlation_execution_log = [
        {
            "tool": "llm_reasoning_correlation",
            "success": True,
            "observation": {"answer": "No Correlation"},
        },
        {
            "tool": "window_stats",
            "success": True,
            "observation": {"min": 1.0, "max": 2.0},
        },
    ]
    assert ExecutorAgent._count_training_candidates(
        correlation_execution_log,
        task_type="correlation",
    ) == 1


def test_count_training_candidates_includes_top_level_indicator_payloads():
    execution_log = [
        {
            "tool": "timesfm2_forecast",
            "success": True,
            "observation": {
                "upper_bb": [140.097, 140.161, 140.28],
            },
        },
        {
            "tool": "window_stats",
            "success": True,
            "observation": {"min": 1.0, "max": 2.0},
        },
    ]

    assert ExecutorAgent._count_training_candidates(
        execution_log,
        task_type="indicator",
        horizon=3,
        indicator_output_keys=["upper_bb"],
    ) == 1


def test_count_training_candidates_keeps_distinct_top_level_paths_even_if_answers_match():
    execution_log = [
        {
            "tool": "llm_reasoning_indicator",
            "success": True,
            "observation": {"upper_bb": [1.0, 2.0, 3.0]},
        },
        {
            "tool": "prophet_forecast",
            "success": True,
            "observation": {"upper_bb": [1.0, 2.0, 3.0]},
        },
        {
            "tool": "arima_forecast",
            "success": True,
            "observation": {"upper_bb": [0.8, 1.8, 2.8]},
        },
    ]

    assert ExecutorAgent._count_training_candidates(
        execution_log,
        task_type="indicator",
        horizon=3,
        indicator_output_keys=["upper_bb"],
    ) == 3


def test_count_training_candidates_pending_eval_adds_only_new_answer_signatures():
    execution_log = [
        {
            "tool": "llm_reasoning_indicator",
            "success": True,
            "observation": {"upper_bb": [1.0, 2.0, 3.0]},
        },
        {
            "tool": "prophet_forecast",
            "success": True,
            "observation": {"upper_bb": [1.0, 2.0, 3.0]},
        },
    ]

    assert ExecutorAgent._count_training_candidates(
        execution_log,
        task_type="indicator",
        horizon=3,
        indicator_output_keys=["upper_bb"],
        pending_predictions=[
            {"upper_bb": [1.0, 2.0, 3.0]},
            {"upper_bb": [0.9, 1.9, 2.9]},
        ],
    ) == 3


def test_collect_batch_eval_predictions_includes_top_level_indicator_payloads():
    execution_log = [
        {
            "tool": "spawn_subagent",
            "output": [
                {
                    "success": True,
                    "result": {
                        "slot_id": 1,
                        "answer": {"upper_bb": [140.132, 140.133, 140.133]},
                        "execution_log": [{"tool": "arima_forecast"}],
                    },
                },
                {
                    "success": True,
                    "result": {
                        "slot_id": 9,
                        "answer": {"upper_bb": [139.917, 140.431, 141.46]},
                        "execution_log": [{"tool": "chronos2_forecast"}],
                    },
                },
            ],
        },
        {
            "tool": "timesfm2_forecast",
            "success": True,
            "observation": {
                "upper_bb": [140.097, 140.161, 140.28],
            },
        },
    ]

    preds = ExecutorAgent._collect_batch_eval_predictions(
        execution_log,
        task_type="indicator",
        horizon=3,
        indicator_output_keys=["upper_bb"],
    )

    assert preds == [
        {"name": "slot_1", "prediction": {"upper_bb": [140.132, 140.133, 140.133]}},
        {"name": "slot_9", "prediction": {"upper_bb": [139.917, 140.431, 141.46]}},
        {"name": "top_level::timesfm2_forecast", "prediction": {"upper_bb": [140.097, 140.161, 140.28]}},
    ]


def test_training_completion_contract_satisfied_requires_eval_and_three_candidates():
    execution_log = [
        {
            "tool": "spawn_subagent",
            "output": {
                "results": [
                    {
                        "success": True,
                        "result": {
                            "answer": "No Correlation",
                            "execution_log": [{"tool": "llm_reasoning_correlation"}],
                        },
                    },
                    {
                        "success": True,
                        "result": {
                            "answer": "Moderate Positive Correlation",
                            "execution_log": [{"tool": "window_stats"}],
                        },
                    },
                    {
                        "success": True,
                        "result": {
                            "answer": "Weak Positive Correlation",
                            "execution_log": [{"tool": "basic_stats"}],
                        },
                    },
                ],
            },
        },
        {
            "tool": "evaluate_batch_against_gt",
            "success": True,
            "observation": {
                "results": [
                    {"metrics": {"accuracy": 1.0}},
                    {"metrics": {"accuracy": 0.0}},
                    {"metrics": {"accuracy": 0.0}},
                ],
            },
        },
    ]
    assert ExecutorAgent._training_completion_contract_satisfied(
        execution_log,
        task_type="correlation",
    ) is True


def test_build_main_budget_message_warns_when_training_contract_satisfied():
    execution_log = [
        {
            "tool": "spawn_subagent",
            "output": {
                "results": [
                    {
                        "success": True,
                        "result": {
                            "answer": "No Correlation",
                            "execution_log": [{"tool": "llm_reasoning_correlation"}],
                        },
                    },
                    {
                        "success": True,
                        "result": {
                            "answer": "Moderate Positive Correlation",
                            "execution_log": [{"tool": "window_stats"}],
                        },
                    },
                    {
                        "success": True,
                        "result": {
                            "answer": "Weak Positive Correlation",
                            "execution_log": [{"tool": "basic_stats"}],
                        },
                    },
                ],
            },
        },
        {
            "tool": "evaluate_batch_against_gt",
            "success": True,
            "observation": {
                "results": [
                    {"metrics": {"accuracy": 1.0}},
                    {"metrics": {"accuracy": 0.0}},
                    {"metrics": {"accuracy": 0.0}},
                ],
            },
        },
    ]
    msg = ExecutorAgent._build_main_budget_message(
        step=9,
        execution_log=execution_log,
        plan={
            "_mode": "train",
            "task_type": "correlation",
            "horizon": None,
            "indicator_output_keys": [],
        },
    )
    assert "training completion contract is already satisfied" in msg


def test_collect_batch_eval_predictions_uses_branch_identity_names():
    execution_log = [
        {
            "tool": "spawn_subagent",
            "output": [
                {
                    "success": True,
                    "result": {
                        "slot_id": 0,
                        "sub_task_description": "branch a",
                        "answer": "<-4%",
                        "execution_log": [],
                    },
                },
                {
                    "success": True,
                    "result": {
                        "slot_id": 1,
                        "sub_task_description": "branch b",
                        "answer": "<-4%",
                        "execution_log": [{"tool": "llm_reasoning_classification"}],
                    },
                },
            ],
        }
    ]

    predictions = ExecutorAgent._collect_batch_eval_predictions(
        execution_log,
        task_type="classification",
    )

    assert predictions == [
        {"name": "slot_0", "prediction": "<-4%"},
        {"name": "slot_1", "prediction": "<-4%"},
    ]


@pytest.mark.asyncio
async def test_executor_raises_if_max_steps_end_without_finalized_answer(monkeypatch):
    class _Resp:
        def __init__(self, parsed):
            self.parsed = parsed
            self.text = json.dumps(parsed, ensure_ascii=False)

    class _LLM:
        def __init__(self):
            self.calls = 0

        async def complete(self, *_args, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                return _Resp(
                    {
                        "done": True,
                        "thought": "Reason: I want to finish early. Action: return a learning summary.",
                        "answer": {
                            "insight": "premature",
                            "recommendation": "needs evaluation first",
                        },
                        "answer_type": "learning_summary",
                        "confidence": 0.5,
                        "reasoning": "1. I tried to finish before any evaluation.",
                    }
                )
            return _Resp(
                {
                    "done": False,
                    "thought": "Reason: I still need to act. Action: run evaluation.",
                    "tool": "evaluate_against_gt",
                    "args": {
                        "task_type": "correlation",
                        "prediction": "No Correlation",
                    },
                }
            )

        async def complete_with_tools(self, *args, **kwargs):
            return await _complete_with_tools_via_complete(self, *args, **kwargs)

    async def _fake_execute_tool(*_args, **_kwargs):
        return ToolResult(
            success=True,
            data={
                "metrics": {
                    "accuracy": 1.0,
                    "correct": True,
                    "predicted_label": "No Correlation",
                    "ground_truth_label": "No Correlation",
                },
                "ground_truth": "No Correlation",
            },
            error=None,
            metadata={},
        )

    executor = ExecutorAgent(llm=_LLM())
    monkeypatch.setattr(executor, "_execute_tool", _fake_execute_tool)

    plan = {
        "candidates": ["evaluate_against_gt"],
        "task_type": "correlation",
        "task_description": "test correlation task",
        "tool_names": "evaluate_against_gt",
        "tool_notes_section": "",
        "skills_section": "",
        "system_memory": "",
        "modality_hint": "",
        "profiling_section": "",
        "fingerprint": {},
        "horizon": None,
        "domain": "finance",
        "reanalysis_section": "",
        "_mode": "train",
        "_defer_learning_record": False,
        "_ground_truth": "No Correlation",
        "indicator_output_keys": [],
        "modality": "multimodal",
    }
    envelope = TaskEnvelope(
        task=TaskSpec(task_id="test", task_type="correlation", description="test"),
        environment=EnvironmentSnapshot(data=[1.0], modalities=["timeseries", "text"]),
    )

    with pytest.raises(RuntimeError, match="max steps without producing a finalized answer"):
        await executor.act(plan, envelope)


@pytest.mark.asyncio
async def test_executor_rejects_last_step_learning_summary_when_training_contract_unsatisfied(monkeypatch):
    class _Resp:
        def __init__(self, parsed):
            self.parsed = parsed
            self.text = json.dumps(parsed, ensure_ascii=False)

    class _LLM:
        async def complete(self, *_args, **_kwargs):
            return _Resp(
                {
                    "done": True,
                    "thought": "Reason: finish now. Action: return a learning summary.",
                    "answer": {
                        "insight": "premature summary",
                        "recommendation": "should not pass",
                    },
                    "answer_type": "learning_summary",
                    "confidence": 0.4,
                    "reasoning": "1. I tried to finish without satisfying the training contract.",
                }
            )

        async def complete_with_tools(self, *args, **kwargs):
            return await _complete_with_tools_via_complete(self, *args, **kwargs)

    executor = ExecutorAgent(llm=_LLM())
    monkeypatch.setattr(executor, "_count_training_candidates", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(executor, "_has_training_evaluation", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(executor, "_has_stepwise_reasoning", lambda *_args, **_kwargs: True)

    plan = {
        "candidates": ["spawn_subagent", "evaluate_against_gt"],
        "task_type": "qa",
        "task_description": "test qa task",
        "tool_names": "spawn_subagent, evaluate_against_gt",
        "tool_notes_section": "",
        "skills_section": "",
        "system_memory": "",
        "modality_hint": "",
        "profiling_section": "",
        "subagent_slot_section": "",
        "subagent_slots": [{"slot_id": 0}, {"slot_id": 1}],
        "fingerprint": {},
        "horizon": None,
        "domain": "finance",
        "reanalysis_section": "",
        "_mode": "train",
        "_defer_learning_record": False,
        "_ground_truth": "A",
        "indicator_output_keys": [],
        "modality": "multimodal",
    }
    envelope = TaskEnvelope(
        task=TaskSpec(task_id="test", task_type="qa", description="test"),
        environment=EnvironmentSnapshot(data=[1.0], modalities=["timeseries", "text"]),
    )

    with pytest.raises(RuntimeError, match="max steps without producing a finalized answer"):
        await executor.act(plan, envelope)


@pytest.mark.asyncio
async def test_executor_blocks_training_eval_until_two_candidates_exist(monkeypatch):
    class _Resp:
        def __init__(self, parsed):
            self.parsed = parsed
            self.text = json.dumps(parsed, ensure_ascii=False)

    class _LLM:
        def __init__(self):
            self.calls = 0

        async def complete(self, *_args, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                return _Resp(
                    {
                        "done": False,
                        "thought": "Reason: I want immediate feedback. Action: evaluate a direct candidate first.",
                        "tool": "evaluate_against_gt",
                        "args": {
                            "task_type": "correlation",
                            "prediction": "No Correlation",
                        },
                    }
                )
            if self.calls == 2:
                return _Resp(
                    {
                        "done": False,
                        "thought": "Reason: evaluation was premature. Action: explore three branches first.",
                        "tool": "spawn_subagent",
                        "args": {
                            "tasks": [
                                {"slot_id": 0, "task": "branch a"},
                                {"slot_id": 1, "task": "branch b"},
                                {"slot_id": 2, "task": "branch c"},
                            ]
                        },
                    }
                )
            if self.calls == 3:
                return _Resp(
                    {
                        "done": False,
                        "thought": "Reason: I now have enough candidates. Action: batch-evaluate them.",
                        "tool": "evaluate_batch_against_gt",
                        "args": {"task_type": "correlation"},
                    }
                )
            return _Resp(
                {
                    "done": True,
                    "thought": "Reason: exploration and evaluation are complete. Action: return the learning summary.",
                    "answer": {
                        "insight": "Comparing two concrete branches before GT evaluation produces a cleaner training trace.",
                        "recommendation": "Spawn first, then run GT evaluation once two valid candidates exist.",
                    },
                    "answer_type": "learning_summary",
                    "confidence": 0.9,
                    "reasoning": (
                        "1. I first explored two distinct candidate branches.\n"
                        "2. I then evaluated the candidate set against ground truth.\n"
                        "3. I summarized the reusable lesson after the comparison was complete."
                    ),
                }
            )

        async def complete_with_tools(self, *args, **kwargs):
            return await _complete_with_tools_via_complete(self, *args, **kwargs)

    async def _fake_spawn_subagents(*_args, **_kwargs):
        return {
            "n_tasks": 3,
            "n_success": 3,
            "results": [
                {
                    "success": True,
                    "result": {
                        "slot_id": 0,
                        "answer": "No Correlation",
                        "answer_type": "correlation",
                        "execution_log": [{"tool": "window_stats"}],
                        "tool_chain": ["window_stats"],
                    },
                },
                {
                    "success": True,
                    "result": {
                        "slot_id": 1,
                        "answer": "Moderate Positive Correlation",
                        "answer_type": "correlation",
                        "execution_log": [{"tool": "llm_reasoning_correlation"}],
                        "tool_chain": ["llm_reasoning_correlation"],
                    },
                },
                {
                    "success": True,
                    "result": {
                        "slot_id": 2,
                        "answer": "Weak Positive Correlation",
                        "answer_type": "correlation",
                        "execution_log": [{"tool": "basic_stats"}],
                        "tool_chain": ["basic_stats"],
                    },
                },
            ],
            "observation": "spawn complete",
        }

    executed_tools = []

    async def _fake_execute_tool(tool_name, *_args, **_kwargs):
        executed_tools.append(tool_name)
        return ToolResult(
            success=True,
            data={
                "results": [
                    {"name": "slot_0", "metrics": {"accuracy": 1.0}},
                    {"name": "slot_1", "metrics": {"accuracy": 0.0}},
                    {"name": "slot_2", "metrics": {"accuracy": 0.0}},
                ]
            },
            error=None,
            metadata={},
        )

    executor = ExecutorAgent(llm=_LLM())
    monkeypatch.setattr(executor, "_spawn_subagents", _fake_spawn_subagents)
    monkeypatch.setattr(executor, "_execute_tool", _fake_execute_tool)
    monkeypatch.setattr(executor, "_has_stepwise_reasoning", lambda *_args, **_kwargs: True)

    plan = {
        "candidates": ["spawn_subagent", "evaluate_against_gt", "evaluate_batch_against_gt"],
        "task_type": "correlation",
        "task_description": "test correlation task",
        "tool_names": "spawn_subagent, evaluate_against_gt, evaluate_batch_against_gt",
        "tool_notes_section": "",
        "skills_section": "",
        "system_memory": "",
        "modality_hint": "",
        "profiling_section": "",
        "subagent_slot_section": "",
        "subagent_slots": [{"slot_id": 0}, {"slot_id": 1}, {"slot_id": 2}],
        "fingerprint": {},
        "horizon": None,
        "domain": "finance",
        "reanalysis_section": "",
        "_mode": "train",
        "_defer_learning_record": False,
        "_ground_truth": "No Correlation",
        "indicator_output_keys": [],
        "modality": "multimodal",
    }
    envelope = TaskEnvelope(
        task=TaskSpec(task_id="test", task_type="correlation", description="test"),
        environment=EnvironmentSnapshot(data=[1.0], modalities=["timeseries", "text"]),
    )

    result = await executor.act(plan, envelope)

    assert result["answer_type"] == "learning_summary"
    assert [entry["tool"] for entry in result["execution_log"]] == [
        "spawn_subagent",
        "evaluate_batch_against_gt",
    ]
    assert executed_tools == ["evaluate_batch_against_gt"]


@pytest.mark.asyncio
async def test_executor_reserves_final_step_once_training_contract_is_satisfied(monkeypatch):
    class _Resp:
        def __init__(self, parsed):
            self.parsed = parsed
            self.text = json.dumps(parsed, ensure_ascii=False)

    class _LLM:
        def __init__(self):
            self.calls = 0

        async def complete(self, *_args, **_kwargs):
            self.calls += 1
            if self.calls < 10:
                return _Resp(
                    {
                        "done": False,
                        "thought": "Reason: continue exploring. Action: run a tool.",
                        "tool": "basic_stats",
                        "args": {},
                    }
                )
            return _Resp(
                {
                    "done": True,
                    "thought": "Reason: the contract is already satisfied. Action: finish with a learning summary.",
                    "answer": {
                        "insight": "enough evidence collected",
                        "recommendation": "stop exploring near the step limit",
                    },
                    "answer_type": "learning_summary",
                    "confidence": 0.9,
                    "reasoning": "1. I compared the explored candidates and stopped before wasting the last step.",
                }
            )

        async def complete_with_tools(self, *args, **kwargs):
            return await _complete_with_tools_via_complete(self, *args, **kwargs)

    tool_calls = {"count": 0}

    async def _fake_execute_tool(*_args, **_kwargs):
        tool_calls["count"] += 1
        return ToolResult(
            success=True,
            data={"mean": 1.0},
            error=None,
            metadata={},
        )

    executor = ExecutorAgent(llm=_LLM())
    monkeypatch.setattr(executor, "_execute_tool", _fake_execute_tool)
    monkeypatch.setattr(
        executor,
        "_training_completion_contract_satisfied",
        lambda *args, **kwargs: True,
    )
    monkeypatch.setattr(executor, "_count_training_candidates", lambda *_args, **_kwargs: 3)
    monkeypatch.setattr(executor, "_has_training_evaluation", lambda *_args, **_kwargs: True)

    plan = {
        "candidates": ["basic_stats"],
        "task_type": "correlation",
        "task_description": "test correlation task",
        "tool_names": "basic_stats",
        "tool_notes_section": "",
        "skills_section": "",
        "system_memory": "",
        "modality_hint": "",
        "profiling_section": "",
        "fingerprint": {},
        "horizon": None,
        "domain": "finance",
        "reanalysis_section": "",
        "_mode": "train",
        "_defer_learning_record": False,
        "_ground_truth": "No Correlation",
        "indicator_output_keys": [],
        "modality": "multimodal",
    }
    envelope = TaskEnvelope(
        task=TaskSpec(task_id="test", task_type="correlation", description="test"),
        environment=EnvironmentSnapshot(data=[1.0], modalities=["timeseries", "text"]),
    )

    result = await executor.act(plan, envelope)

    assert result["answer_type"] == "learning_summary"
    assert tool_calls["count"] == 8


@pytest.mark.asyncio
async def test_executor_allows_second_spawn_round_only_when_candidates_are_insufficient(monkeypatch):
    class _Resp:
        def __init__(self, parsed):
            self.parsed = parsed
            self.text = json.dumps(parsed, ensure_ascii=False)

    class _LLM:
        def __init__(self):
            self.calls = 0

        async def complete(self, *_args, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                return _Resp(
                    {
                        "done": False,
                        "thought": "Reason: first exploration round. Action: spawn three branches.",
                        "tool": "spawn_subagent",
                        "args": {
                            "tasks": [
                                {"slot_id": 0, "task": "branch a"},
                                {"slot_id": 1, "task": "branch b"},
                                {"slot_id": 2, "task": "branch c"},
                            ]
                        },
                    }
                )
            if self.calls == 2:
                return _Resp(
                    {
                        "done": False,
                        "thought": "Reason: one branch failed, so I need one fresh candidate. Action: spawn one more branch.",
                        "tool": "spawn_subagent",
                        "args": {
                            "tasks": [
                                {"slot_id": 3, "task": "branch d"},
                            ]
                        },
                    }
                )
            if self.calls == 3:
                return _Resp(
                    {
                        "done": False,
                        "thought": "Reason: I now have three valid candidates. Action: batch-evaluate them.",
                        "tool": "evaluate_batch_against_gt",
                        "args": {"task_type": "correlation"},
                    }
                )
            return _Resp(
                {
                    "done": True,
                    "thought": "Reason: exploration and evaluation are complete. Action: return the learning summary.",
                    "answer": {
                        "insight": "A second spawn round is only justified when the first round leaves too few valid candidates.",
                        "recommendation": "Use one fresh slot to top up the missing candidate, then evaluate immediately.",
                    },
                    "answer_type": "learning_summary",
                    "confidence": 0.9,
                    "reasoning": (
                        "1. The first spawn round produced only two valid candidates because one branch failed.\n"
                        "2. A second spawn round added one fresh candidate using an unused slot.\n"
                        "3. After reaching three valid candidates, I evaluated them and summarized the lesson."
                    ),
                }
            )

        async def complete_with_tools(self, *args, **kwargs):
            return await _complete_with_tools_via_complete(self, *args, **kwargs)

    spawn_calls: list[dict[str, Any]] = []

    async def _fake_spawn_subagents(tool_args, *_args, **_kwargs):
        spawn_calls.append(tool_args)
        if len(spawn_calls) == 1:
            return {
                "n_tasks": 3,
                "n_success": 2,
                "results": [
                    {
                        "success": True,
                        "result": {
                            "slot_id": 0,
                            "answer": "No Correlation",
                            "answer_type": "correlation",
                            "execution_log": [{"tool": "window_stats"}],
                        },
                    },
                    {
                        "success": True,
                        "result": {
                            "slot_id": 1,
                            "answer": "Weak Positive Correlation",
                            "answer_type": "correlation",
                            "execution_log": [{"tool": "basic_stats"}],
                        },
                    },
                    {
                        "success": False,
                        "result": {
                            "slot_id": 2,
                            "execution_log": [{"tool": "llm_reasoning_correlation"}],
                        },
                    },
                ],
                "observation": "first spawn complete",
            }
        return {
            "n_tasks": 1,
            "n_success": 1,
            "results": [
                {
                    "success": True,
                    "result": {
                        "slot_id": 3,
                        "answer": "Moderate Positive Correlation",
                        "answer_type": "correlation",
                        "execution_log": [{"tool": "llm_reasoning_correlation"}],
                    },
                }
            ],
            "observation": "second spawn complete",
        }

    executed_tools = []

    async def _fake_execute_tool(tool_name, *_args, **_kwargs):
        executed_tools.append(tool_name)
        return ToolResult(
            success=True,
            data={
                "results": [
                    {"name": "slot_0", "metrics": {"accuracy": 1.0}},
                    {"name": "slot_1", "metrics": {"accuracy": 0.0}},
                    {"name": "slot_3", "metrics": {"accuracy": 0.0}},
                ]
            },
            error=None,
            metadata={},
        )

    executor = ExecutorAgent(llm=_LLM())
    monkeypatch.setattr(executor, "_spawn_subagents", _fake_spawn_subagents)
    monkeypatch.setattr(executor, "_execute_tool", _fake_execute_tool)
    monkeypatch.setattr(executor, "_has_stepwise_reasoning", lambda *_args, **_kwargs: True)

    plan = {
        "candidates": ["spawn_subagent", "evaluate_batch_against_gt"],
        "task_type": "correlation",
        "task_description": "test correlation task",
        "tool_names": "spawn_subagent, evaluate_batch_against_gt",
        "tool_notes_section": "",
        "skills_section": "",
        "system_memory": "",
        "modality_hint": "",
        "profiling_section": "",
        "subagent_slot_section": "",
        "subagent_slots": [
            {"slot_id": 0},
            {"slot_id": 1},
            {"slot_id": 2},
            {"slot_id": 3},
        ],
        "fingerprint": {},
        "horizon": None,
        "domain": "finance",
        "reanalysis_section": "",
        "_mode": "train",
        "_defer_learning_record": False,
        "_ground_truth": "No Correlation",
        "indicator_output_keys": [],
        "modality": "multimodal",
    }
    envelope = TaskEnvelope(
        task=TaskSpec(task_id="test", task_type="correlation", description="test"),
        environment=EnvironmentSnapshot(data=[1.0], modalities=["timeseries", "text"]),
    )

    result = await executor.act(plan, envelope)

    assert result["answer_type"] == "learning_summary"
    assert [entry["tool"] for entry in result["execution_log"]] == [
        "spawn_subagent",
        "spawn_subagent",
        "evaluate_batch_against_gt",
    ]
    assert executed_tools == ["evaluate_batch_against_gt"]
    assert [
        [task["slot_id"] for task in call["tasks"]]
        for call in spawn_calls
    ] == [[0, 1, 2], [3]]


def test_precompute_subagent_slots_builds_fixed_slot_pool():
    slots = ExecutorAgent._precompute_subagent_slots(
        trace_id="trace-1",
        candidates=[
            "chronos2_forecast",
            "timesfm2_forecast",
            "llm_reasoning_forecast",
            "spawn_subagent",
            "evaluate_batch_against_gt",
        ],
        task_type="forecast",
        run_mode="train",
        scope="finance/forecast",
        scope_counts={
            "chronos2_forecast": 10,
            "timesfm2_forecast": 20,
            "llm_reasoning_forecast": 999,
        },
    )

    assert len(slots) == 10
    for idx, slot in enumerate(slots):
        assert slot["slot_id"] == idx
        assert "spawn_subagent" not in slot["visible_candidates"]
        assert "evaluate_batch_against_gt" not in slot["visible_candidates"]
        assert "llm_reasoning_forecast" in slot["visible_candidates"]
        assert slot["keep_probabilities"]["llm_reasoning_forecast"] == 1.0
        assert slot["never_drop"] == ["llm_reasoning_forecast"]


def test_validate_spawn_subagent_tasks_requires_valid_distinct_slot_ids():
    slots = [
        {"slot_id": 0},
        {"slot_id": 1},
    ]

    duplicate_error = ExecutorAgent._validate_spawn_subagent_tasks(
        [
            {"slot_id": 0, "task": "path a"},
            {"slot_id": 0, "task": "path b"},
        ],
        slots,
    )
    invalid_error = ExecutorAgent._validate_spawn_subagent_tasks(
        [{"slot_id": 3, "task": "path a"}],
        slots,
    )
    ok = ExecutorAgent._validate_spawn_subagent_tasks(
        [
            {"slot_id": 0, "task": "path a"},
            {"slot_id": 1, "task": "path b"},
        ],
        slots,
    )

    assert "different `slot_id`" in duplicate_error
    assert "slot_id=3 is invalid" in invalid_error
    assert ok is None


def test_validate_spawn_subagent_tasks_enforces_training_minimum_when_requested():
    slots = [
        {"slot_id": 0},
        {"slot_id": 1},
        {"slot_id": 2},
    ]

    error = ExecutorAgent._validate_spawn_subagent_tasks(
        [
            {"slot_id": 0, "task": "path a"},
            {"slot_id": 1, "task": "path b"},
        ],
        slots,
        min_tasks=3,
    )

    assert "at least 3 distinct branches" in error


def test_build_subagent_candidates_prefers_precomputed_slot_visibility():
    executor = ExecutorAgent(llm=SimpleNamespace())
    visible = executor._build_subagent_candidates(
        idx=0,
        task_description="forecast with slot",
        tools_hint="timesfm2_forecast",
        plan={
            "_mode": "train",
            "_tool_dropout_scope": "finance/forecast",
            "_tool_dropout_counts": {
                "chronos2_forecast": 10,
                "timesfm2_forecast": 20,
                "llm_reasoning_forecast": 30,
            },
            "task_type": "forecast",
            "candidates": [
                "chronos2_forecast",
                "timesfm2_forecast",
                "llm_reasoning_forecast",
                "spawn_subagent",
                "evaluate_batch_against_gt",
            ],
        },
        envelope=SimpleNamespace(trace_id="trace-1"),
        slot_id=2,
        slot_spec={
            "slot_id": 2,
            "visible_candidates": ["timesfm2_forecast", "llm_reasoning_forecast"],
            "keep_probabilities": {
                "chronos2_forecast": 0.1,
                "timesfm2_forecast": 0.2,
                "llm_reasoning_forecast": 1.0,
            },
            "never_drop": ["llm_reasoning_forecast"],
        },
    )

    assert visible == ["timesfm2_forecast", "llm_reasoning_forecast"]


def test_build_visible_tool_guidance_includes_spawn_dropout_and_evaluate_rules():
    guidance = executor_prompt.build_visible_tool_guidance(
        [
            "chronos2_forecast",
            "spawn_subagent",
            "evaluate_batch_against_gt",
        ],
        defer_learning_record=False,
    )

    assert "prefer using `spawn_subagent`" in guidance
    assert "post-dropout visible tool subsets" in guidance
    assert "do not adjust, rewrite, or refine the candidate output" in guidance
    assert '`answer_type` must be `"learning_summary"`' in guidance


def test_build_visible_tool_guidance_is_empty_without_spawn_or_evaluate():
    guidance = executor_prompt.build_visible_tool_guidance(
        ["chronos2_forecast", "llm_reasoning_forecast"],
        defer_learning_record=False,
    )

    assert guidance == ""


def test_derive_tools_hint_from_slot_projects_onto_visible_candidates():
    derived = ExecutorAgent._derive_tools_hint_from_slot(
        "chronos2_forecast, timesfm2_forecast",
        slot_spec={
            "slot_id": 2,
            "visible_candidates": ["timesfm2_forecast", "llm_reasoning_forecast"],
        },
        task_type="forecast",
    )

    assert derived == ["timesfm2_forecast"]


def test_build_subagent_slot_section_keeps_each_slot_separate():
    text = ExecutorAgent._build_subagent_slot_section(
        [
            {
                "slot_id": 0,
                "visible_candidates": ["timesfm2_forecast", "llm_reasoning_forecast"],
            },
            {
                "slot_id": 1,
                "visible_candidates": ["timesfm2_forecast", "llm_reasoning_forecast"],
            },
            {
                "slot_id": 2,
                "visible_candidates": ["window_stats"],
            },
        ]
    )

    assert "slot_id=0: visible_tools=timesfm2_forecast, llm_reasoning_forecast" in text
    assert "slot_id=1: visible_tools=timesfm2_forecast, llm_reasoning_forecast" in text
    assert "slot_id=2: visible_tools=window_stats" in text
    assert text.count("timesfm2_forecast, llm_reasoning_forecast") == 2


def test_derive_tools_hint_from_slot_rejects_incompatible_explicit_hint():
    derived = ExecutorAgent._derive_tools_hint_from_slot(
        "chronos2_forecast",
        slot_spec={
            "slot_id": 2,
            "visible_candidates": ["timesfm2_forecast", "llm_reasoning_forecast"],
        },
        task_type="forecast",
    )

    assert derived == []


def test_derive_tools_hint_from_slot_generates_distinct_paths_for_same_visible_tools():
    slot_spec = {
        "slot_id": 2,
        "visible_candidates": ["timesfm2_forecast", "llm_reasoning_forecast"],
    }

    first = ExecutorAgent._derive_tools_hint_from_slot(
        "",
        slot_spec=slot_spec,
        task_type="forecast",
        used_signatures=set(),
    )
    second = ExecutorAgent._derive_tools_hint_from_slot(
        "",
        slot_spec=slot_spec,
        task_type="forecast",
        used_signatures={tuple(first)},
    )

    assert first == ["timesfm2_forecast"]
    assert second == ["llm_reasoning_forecast"]


def test_derive_tools_hint_from_slot_avoids_fragile_singleton_tools():
    slot_spec = {
        "slot_id": 2,
        "visible_candidates": ["window_stats", "slice", "llm_reasoning_forecast"],
    }

    first = ExecutorAgent._derive_tools_hint_from_slot(
        "",
        slot_spec=slot_spec,
        task_type="forecast",
        used_signatures=set(),
    )
    second = ExecutorAgent._derive_tools_hint_from_slot(
        "",
        slot_spec=slot_spec,
        task_type="forecast",
        used_signatures={tuple(first)},
    )

    assert first == ["slice"]
    assert second == ["llm_reasoning_forecast"]


@pytest.mark.asyncio
async def test_executor_terminal_path_review_can_wrap_earlier_bare_payload(monkeypatch):
    class _Resp:
        def __init__(self, parsed):
            self.parsed = parsed
            self.text = json.dumps(parsed, ensure_ascii=False)

    class _LLM:
        async def complete(self, *_args, **_kwargs):
            return _Resp(
                {
                    "done": True,
                    "thought": "Reason: an earlier reply already contained the valid MACD payload. Action: resend that same payload inside the final envelope.",
                    "answer": {"macd": [0.1, 0.2, 0.3]},
                    "answer_type": "indicator",
                    "confidence": 0.9,
                    "reasoning": "1. An earlier reply already contained the valid MACD payload.\n2. After budget exhaustion, I reviewed the prior path and wrapped that same payload in the required final envelope.",
                }
            )

    executor = ExecutorAgent(llm=_LLM())
    decision = await executor._request_terminal_path_review(
        conversation=[
            {"role": "assistant", "content": '{"macd": [0.1, 0.2, 0.3]}'},
            {"role": "user", "content": "tool observation"},
        ],
        expected_type="indicator",
        trace_phase="test_terminal_review",
    )

    assert decision["done"] is True
    assert decision["answer"] == {"macd": [0.1, 0.2, 0.3]}
    assert decision["answer_type"] == "indicator"


def test_build_main_output_contract_section_includes_learning_summary_contract_for_training():
    text = ExecutorAgent._build_main_output_contract_section(
        is_train=True,
        defer_learning_record=False,
        indicator_output_keys=[],
        allow_finish_current_turn=True,
    )

    assert '`answer_type` exactly `"learning_summary"`' in text
    assert "`insight`" in text
    assert "`recommendation`" in text
    assert "Do not write about framework/orchestration mechanics" in text
    assert "assigned path" in text


def test_build_main_output_contract_section_omits_learning_summary_contract_outside_training():
    text = ExecutorAgent._build_main_output_contract_section(
        is_train=False,
        defer_learning_record=False,
        indicator_output_keys=["macd"],
        allow_finish_current_turn=True,
    )

    assert "learning_summary" not in text
    assert "macd" in text


def test_is_valid_learning_summary_answer_requires_non_empty_insight_and_recommendation():
    assert ExecutorAgent._is_valid_learning_summary_answer(
        {"insight": "a", "recommendation": "b"}
    ) is True
    assert ExecutorAgent._is_valid_learning_summary_answer(
        {"insight": "", "recommendation": "b"}
    ) is False
    assert ExecutorAgent._is_valid_learning_summary_answer("bad") is False


def test_learning_summary_validation_message_explains_textual_recommendation_requirement():
    text = ExecutorAgent._learning_summary_validation_message(
        {"insight": "winner is slot_2", "recommendation": {"macd": [1, 2, 3]}}
    )

    assert "`recommendation` must be a non-empty string" in text
    assert "not the task prediction object" in text
    assert "do not mention framework wording" in text


def test_hinted_tools_accepts_plain_tool_names_without_underscores():
    hinted = ExecutorAgent._hinted_tools(
        "sentiment, detect_trend, llm_reasoning_qa"
    )

    assert hinted == ["sentiment", "detect_trend", "llm_reasoning_qa"]


def test_hinted_tools_accepts_list_hint():
    hinted = ExecutorAgent._hinted_tools(
        ["window_stats", "value_at", "llm_reasoning_qa"]
    )

    assert hinted == ["window_stats", "value_at", "llm_reasoning_qa"]


def test_shared_first_turn_template_uses_objective_observation_decision_structure():
    text = executor_prompt.REACT_FIRST_TURN.format(
        task_prompt="Task body",
        task_boundary_section="## Task Boundary\nBoundary body\n\n",
        execution_context_section="## Execution Context\nContext body\n\n",
        output_contract_section="## Output Contract\nContract body\n\n",
        fingerprint="{}",
        profiling_section="## Profiling Results\n- x\n",
        support_section="## Support\nSupport body\n\n",
        decision_section="Choose exactly one action for this turn:\n1. Call exactly one tool.",
        tool_names="- chronos2_forecast: ...",
    )

    assert "## Objective" in text
    assert "### Task Prompt" in text
    assert "## Task Boundary" in text
    assert "## Execution Context" in text
    assert "## Output Contract" in text
    assert "## Observation" in text
    assert "### Sample Fingerprint" in text
    assert "## Support" in text
    assert "## Decision" in text
    assert "### Available Tools" in text


def test_build_tool_observation_message_uses_structured_sections():
    text = ExecutorAgent._build_tool_observation_message(
        step=2,
        tool_name="slice",
        tool_args={"start": 10, "end": 20},
        success=True,
        observation_payload=[1, 2, 3],
    )

    assert "## New Observation" in text
    assert "- Step: Step 2" in text
    assert "- Tool: slice" in text
    assert "- Status: success" in text
    assert "### Tool Result" in text


def test_build_state_update_message_uses_current_working_state_header():
    text = ExecutorAgent._build_state_update_message(
        current_series_state={"artifact_type": "series"},
        latest_artifact_state={"producer_tool": "slice"},
    )

    assert text.startswith("## Current Working State\n")
    assert '"current_series_state"' in text
    assert '"latest_artifact_state"' in text


def test_build_task_boundary_section_forecast_is_task_specific():
    text = ExecutorAgent._build_task_boundary_section("forecast", horizon=24)

    assert "predict a future numeric time-series sequence" in text
    assert "JSON numeric array" in text
    assert "trend labels" in text


def test_build_task_boundary_section_indicator_mentions_exact_keys():
    text = ExecutorAgent._build_task_boundary_section(
        "indicator",
        indicator_output_keys=["macd", "upper_bb"],
    )

    assert "predict derived indicator values" in text
    assert "exactly these indicator keys: macd, upper_bb" in text
    assert "raw forecast arrays" in text


def test_build_task_boundary_section_qa_mentions_answer_text_or_option():
    text = ExecutorAgent._build_task_boundary_section("qa")

    assert "answer the question" in text
    assert "final answer text itself" in text
    assert "final option label" in text


def test_build_parent_branch_metadata_section_renders_short_structured_lines():
    text = ExecutorAgent._build_parent_branch_metadata_section(
        {
            "branch_id": "slot_3_alt",
            "candidate_family": "chronos",
            "comparison_target": "slot_1",
        }
    )

    assert "Parent Branch Metadata:" in text
    assert "- branch_id: slot_3_alt" in text
    assert "- candidate_family: chronos" in text
    assert "- comparison_target: slot_1" in text


def test_build_subagent_tools_hint_section_renders_list_hint():
    text = ExecutorAgent._build_subagent_tools_hint_section(
        ["window_stats", "value_at", "llm_reasoning_qa"],
        ["window_stats", "value_at", "llm_reasoning_qa", "sentiment"],
    )

    assert "Assigned Exploration Path:" in text
    assert "window_stats, value_at, llm_reasoning_qa" in text


def test_tool_note_candidates_support_task_specific_llm_reasoning_aliases():
    assert ExecutorAgent._tool_note_candidates(
        "llm_reasoning_qa",
        "qa",
    ) == ["llm_reasoning_qa", "llm_reasoning", "llm_reasoning::qa"]


def test_reasoning_candidate_names_accepts_list_hint_without_self_reference():
    names = ExecutorAgent._reasoning_candidate_names(
        [
            {
                "tool": "spawn_subagent",
                "output": [
                    {
                        "success": True,
                        "result": {
                            "tools_hint": ["window_stats", "value_at", "llm_reasoning_qa"],
                            "execution_log": [],
                        },
                    }
                ],
            }
        ]
    )

    assert names == ["window_stats, value_at, llm_reasoning_qa"]


def test_validate_spawn_subagent_tasks_rejects_non_dict_branch_metadata():
    slots = [
        {"slot_id": 0},
        {"slot_id": 1},
    ]

    error = ExecutorAgent._validate_spawn_subagent_tasks(
        [{"slot_id": 0, "task": "path a", "branch_metadata": "bad"}],
        slots,
    )

    assert "branch_metadata" in error


def test_validate_spawn_subagent_tasks_rejects_reused_slot_ids_across_rounds():
    slots = [
        {"slot_id": 0},
        {"slot_id": 1},
        {"slot_id": 2},
    ]

    error = ExecutorAgent._validate_spawn_subagent_tasks(
        [{"slot_id": 1, "task": "path b"}],
        slots,
        min_tasks=1,
        disallowed_slot_ids={0, 1},
    )

    assert "already used in a previous spawn round" in error


def test_coerce_llm_decision_rejects_multiple_top_level_json_objects():
    raw = (
        '{"done": false, "thought": "Reason: first. Action: spawn.", '
        '"tool": "spawn_subagent", "args": {"tasks": []}}'
        '{"done": false, "thought": "Reason: second. Action: eval.", '
        '"tool": "evaluate_batch_against_gt", "args": {"task_type": "classification"}}'
    )

    decision = ExecutorAgent._coerce_llm_decision(
        parsed=None,
        text=raw,
        fallback_thought="parse failed",
    )

    assert decision["_parse_failed"] is True
    assert decision["thought"] == "multiple_json_objects_in_one_reply"


def test_coerce_llm_decision_prefers_parsed_dict_over_noisy_text():
    parsed = {
        "done": False,
        "thought": "Reason: first. Action: spawn.",
        "tool": "spawn_subagent",
        "args": {"tasks": [{"slot_id": 0, "task": "path a"}]},
    }
    raw = (
        '{"done": false, "thought": "Reason: first. Action: spawn.", '
        '"tool": "spawn_subagent", "args": {"tasks": [{"slot_id": 0, "task": "path a"}]}}'
        "\n选择工具: spawn_subagent"
    )

    decision = ExecutorAgent._coerce_llm_decision(
        parsed=parsed,
        text=raw,
        fallback_thought="parse failed",
    )

    assert decision == parsed


def test_coerce_llm_decision_rejects_multiple_top_level_json_objects_even_with_parsed_dict():
    parsed = {
        "done": False,
        "thought": "Reason: first. Action: segment.",
        "tool": "segment",
        "args": {"window_size": 78, "step_size": 39},
    }
    raw = (
        '{"done": false, "thought": "Reason: first. Action: segment.", '
        '"tool": "segment", "args": {"window_size": 78, "step_size": 39}}'
        '{"macd": [0.0]}'
    )

    decision = ExecutorAgent._coerce_llm_decision(
        parsed=parsed,
        text=raw,
        fallback_thought="parse failed",
    )

    assert decision["_parse_failed"] is True
    assert decision["thought"] == "multiple_json_objects_in_one_reply"


def test_coerce_structured_decision_maps_tool_call_response():
    response = LLMResponse(
        text="",
        parsed=None,
        tool_calls=[{
            "name": "emit_tool_call",
            "arguments": {
                "thought": "Reason: need evidence. Action: run slice.",
                "tool": "slice",
                "args": {"start": 0, "end": 24},
            },
        }],
        model="mock",
    )

    decision = ExecutorAgent._coerce_structured_decision(
        response,
        allow_tool_call=True,
        allow_finish=True,
        fallback_thought="parse failed",
    )

    assert decision == {
        "thought": "Reason: need evidence. Action: run slice.",
        "done": False,
        "tool": "slice",
        "args": {"start": 0, "end": 24},
    }


def test_coerce_structured_decision_maps_finish_response():
    response = LLMResponse(
        text="",
        parsed=None,
        tool_calls=[{
            "name": "emit_final_answer",
            "arguments": {
                "thought": "Reason: enough evidence. Action: finish.",
                "answer_type": "qa",
                "answer_json": "\"B\"",
                "confidence": 0.9,
                "reasoning": "1. Evidence supports B.",
            },
        }],
        model="mock",
    )

    decision = ExecutorAgent._coerce_structured_decision(
        response,
        allow_tool_call=True,
        allow_finish=True,
        fallback_thought="parse failed",
    )

    assert decision == {
        "thought": "Reason: enough evidence. Action: finish.",
        "done": True,
        "answer": "B",
        "answer_type": "qa",
        "confidence": 0.9,
        "reasoning": "1. Evidence supports B.",
    }


def test_coerce_structured_decision_accepts_legacy_json_tool_decision_when_tool_calls_missing():
    response = LLMResponse(
        text='{"thought":"Reason: compare paths. Action: spawn.","done":false,"tool":"spawn_subagent","args":{"tasks":[{"task":"x"}]}}',
        parsed={
            "thought": "Reason: compare paths. Action: spawn.",
            "done": False,
            "tool": "spawn_subagent",
            "args": {"tasks": [{"task": "x"}]},
        },
        tool_calls=[],
        model="mock",
    )

    decision = ExecutorAgent._coerce_structured_decision(
        response,
        allow_tool_call=True,
        allow_finish=True,
        fallback_thought="parse failed",
    )

    assert decision == {
        "thought": "Reason: compare paths. Action: spawn.",
        "done": False,
        "tool": "spawn_subagent",
        "args": {"tasks": [{"task": "x"}]},
    }


def test_coerce_structured_decision_rejects_legacy_json_with_mixed_tool_and_final_fields():
    response = LLMResponse(
        text=(
            '{"thought":"Reason: mixed. Action: invalid.","done":false,'
            '"tool":"spawn_subagent","args":{"tasks":[{"task":"x"}]},'
            '"answer_type":"qa","reasoning":"bad mix"}'
        ),
        parsed={
            "thought": "Reason: mixed. Action: invalid.",
            "done": False,
            "tool": "spawn_subagent",
            "args": {"tasks": [{"task": "x"}]},
            "answer_type": "qa",
            "reasoning": "bad mix",
        },
        tool_calls=[],
        model="mock",
    )

    decision = ExecutorAgent._coerce_structured_decision(
        response,
        allow_tool_call=True,
        allow_finish=True,
        fallback_thought="parse failed",
    )

    assert decision["_parse_failed"] is True
    assert decision["thought"] == "invalid_decision_tool_call_count"


def test_main_output_contract_uses_current_turn_contract_when_finish_not_allowed():
    section = ExecutorAgent._build_main_output_contract_section(
        is_train=True,
        defer_learning_record=False,
        indicator_output_keys=[],
        allow_finish_current_turn=False,
    )

    assert "This is not a completion turn" in section
    assert "Return only the current-turn tool-call JSON" in section
    assert "Required keys inside `answer`: `insight`, `recommendation`." not in section


@pytest.mark.asyncio
async def test_executor_forbids_second_spawn_round_in_training(monkeypatch):
    class _Resp:
        def __init__(self, parsed):
            self.parsed = parsed
            self.text = json.dumps(parsed, ensure_ascii=False)

    class _LLM:
        def __init__(self):
            self.calls = 0

        async def complete(self, *_args, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                return _Resp(
                    {
                        "done": False,
                        "thought": "Reason: I need parallel candidates. Action: spawn the first round.",
                        "tool": "spawn_subagent",
                        "args": {
                            "tasks": [
                                {"slot_id": 0, "task": "path a", "tools_hint": "chronos2_forecast"},
                                {"slot_id": 1, "task": "path b", "tools_hint": "timesfm2_forecast"},
                                {"slot_id": 2, "task": "path c", "tools_hint": "llm_reasoning_forecast"},
                            ]
                        },
                    }
                )
            if self.calls == 2:
                return _Resp(
                    {
                        "done": False,
                        "thought": "Reason: I want more branches. Action: spawn another round.",
                        "tool": "spawn_subagent",
                        "args": {
                            "tasks": [
                                {"slot_id": 3, "task": "path d", "tools_hint": "llm_reasoning_forecast"},
                            ]
                        },
                    }
                )
            return _Resp(
                {
                    "done": True,
                    "thought": "Reason: the first round is enough. Action: finish with the learning summary.",
                    "answer": {
                        "insight": "one spawn round was enough",
                        "recommendation": "reuse existing candidates",
                    },
                    "answer_type": "learning_summary",
                    "confidence": 0.8,
                    "reasoning": (
                        "1. I compared the chronos2_forecast and timesfm2_forecast branches.\n"
                        "2. The first round already covered the useful alternatives, so I summarized instead of reopening exploration."
                    ),
                }
            )

        async def complete_with_tools(self, *args, **kwargs):
            return await _complete_with_tools_via_complete(self, *args, **kwargs)

    spawn_calls = {"count": 0}

    async def _fake_spawn_subagents(*_args, **_kwargs):
        spawn_calls["count"] += 1
        return {
            "n_tasks": 2,
            "n_success": 2,
            "results": [
                {
                    "success": True,
                    "result": {
                        "answer": [1.0, 2.0],
                        "answer_type": "forecast",
                        "execution_log": [{"tool": "chronos2_forecast"}],
                        "tool_chain": ["chronos2_forecast"],
                    },
                },
                {
                    "success": True,
                    "result": {
                        "answer": [1.1, 2.1],
                        "answer_type": "forecast",
                        "execution_log": [{"tool": "timesfm2_forecast"}],
                        "tool_chain": ["timesfm2_forecast"],
                    },
                },
                {
                    "success": True,
                    "result": {
                        "answer": [0.9, 2.2],
                        "answer_type": "forecast",
                        "execution_log": [{"tool": "llm_reasoning_forecast"}],
                        "tool_chain": ["llm_reasoning_forecast"],
                    },
                },
            ],
            "observation": "first round complete",
        }

    executor = ExecutorAgent(llm=_LLM())
    monkeypatch.setattr(executor, "_spawn_subagents", _fake_spawn_subagents)
    monkeypatch.setattr(ExecutorAgent, "_has_training_evaluation", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(ExecutorAgent, "_count_training_candidates", lambda *_args, **_kwargs: 3)
    monkeypatch.setattr(executor, "_has_stepwise_reasoning", lambda *_args, **_kwargs: True)

    plan = {
        "candidates": ["spawn_subagent"],
        "task_type": "forecast",
        "task_description": "test forecast task",
        "tool_names": "spawn_subagent",
        "tool_notes_section": "",
        "skills_section": "",
        "system_memory": "",
        "modality_hint": "",
        "profiling_section": "",
        "subagent_slot_section": "",
        "subagent_slots": [{"slot_id": i} for i in range(4)],
        "fingerprint": {},
        "horizon": 2,
        "domain": "finance",
        "reanalysis_section": "",
        "_mode": "train",
        "_defer_learning_record": False,
        "_ground_truth": [1.0, 2.0],
        "indicator_output_keys": [],
        "modality": "timeseries",
    }
    envelope = TaskEnvelope(
        task=TaskSpec(task_id="test", task_type="forecast", description="test"),
        environment=EnvironmentSnapshot(data=[1.0], modalities=["timeseries"]),
    )

    result = await executor.act(plan, envelope)

    assert result["answer_type"] == "learning_summary"
    assert spawn_calls["count"] == 1


@pytest.mark.asyncio
async def test_spawn_subagents_derives_tools_hint_from_slot_visible_candidates(monkeypatch):
    executor = ExecutorAgent(llm=None)
    seen_hints: list[dict[str, object]] = []

    async def _fake_run_sub_executor(
        idx,
        task_description,
        tools_hint,
        branch_metadata,
        plan,
        envelope,
        *,
        slot_id=None,
        slot_spec=None,
    ):
        seen_hints.append(
            {
                "idx": idx,
                "task_description": task_description,
                "tools_hint": tools_hint,
                "slot_id": slot_id,
                "slot_spec": slot_spec,
            }
        )
        return {
            "answer": [1.0, 2.0],
            "answer_type": "forecast",
            "execution_log": [{"tool": "llm_reasoning_forecast"}],
            "tool_chain": ["llm_reasoning_forecast"],
            "tools_hint": tools_hint,
            "slot_id": slot_id,
        }

    monkeypatch.setattr(executor, "_run_sub_executor", _fake_run_sub_executor)

    plan = {
        "task_type": "forecast",
        "subagent_slots": [
            {
                "slot_id": 0,
                "visible_candidates": ["timesfm2_forecast", "llm_reasoning_forecast"],
            }
        ],
    }
    envelope = SimpleNamespace(environment=SimpleNamespace())
    result = await executor._spawn_subagents(
        {
            "tasks": [
                {
                    "slot_id": 0,
                    "task": "path a",
                    "tools_hint": "chronos2_forecast, timesfm2_forecast",
                }
            ]
        },
        plan,
        envelope,
    )

    assert seen_hints == [
        {
            "idx": 0,
            "task_description": "path a",
            "tools_hint": ["timesfm2_forecast"],
            "slot_id": 0,
            "slot_spec": {
                "slot_id": 0,
                "visible_candidates": ["timesfm2_forecast", "llm_reasoning_forecast"],
            },
        }
    ]
    assert result["results"][0]["result"]["tools_hint"] == ["timesfm2_forecast"]


def test_normalize_spawn_tasks_rejects_incompatible_explicit_hint():
    normalized, error = ExecutorAgent._normalize_spawn_tasks(
        [
            {"slot_id": 0, "task": "path a", "tools_hint": "chronos2_forecast"},
            {"slot_id": 1, "task": "path b", "tools_hint": "chronos2_forecast"},
        ],
        plan={
            "task_type": "forecast",
            "subagent_slots": [
                {
                    "slot_id": 0,
                    "visible_candidates": ["timesfm2_forecast", "llm_reasoning_forecast"],
                },
                {
                    "slot_id": 1,
                    "visible_candidates": ["timesfm2_forecast", "llm_reasoning_forecast"],
                },
            ],
        },
    )

    assert normalized == []
    assert error is not None
    assert "incompatible with the chosen slot_id=0" in error


def test_normalize_spawn_tasks_skips_non_distinct_singleton_paths():
    normalized, error = ExecutorAgent._normalize_spawn_tasks(
        [
            {"slot_id": 0, "task": "path a"},
            {"slot_id": 1, "task": "path b"},
        ],
        plan={
            "task_type": "forecast",
            "subagent_slots": [
                {"slot_id": 0, "visible_candidates": ["llm_reasoning_forecast"]},
                {"slot_id": 1, "visible_candidates": ["llm_reasoning_forecast"]},
            ],
        },
    )

    assert error is None
    assert len(normalized) == 1
    assert normalized[0]["slot_id"] == 0
    assert normalized[0]["tools_hint"] == ["llm_reasoning_forecast"]


@pytest.mark.asyncio
async def test_executor_training_main_agent_rejects_task_typed_final_answer(monkeypatch):
    class _Resp:
        def __init__(self, parsed):
            self.parsed = parsed
            self.text = json.dumps(parsed, ensure_ascii=False)

    class _LLM:
        def __init__(self):
            self.calls = 0

        async def complete(self, *_args, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                return _Resp(
                    {
                        "done": True,
                        "thought": "Reason: I think the forecast itself is enough. Action: return the task-typed answer.",
                        "answer": [1.0, 2.0],
                        "answer_type": "forecast",
                        "confidence": 0.6,
                        "reasoning": "1. I compared the candidates and picked the best forecast.",
                    }
                )
            return _Resp(
                {
                    "done": True,
                    "thought": "Reason: training mode requires a summary. Action: finish with the learning summary.",
                    "answer": {
                        "insight": "Candidate comparison mattered more than any single raw forecast.",
                        "recommendation": "Finish training runs with a learning summary after selecting the winner.",
                    },
                    "answer_type": "learning_summary",
                    "confidence": 0.8,
                    "reasoning": (
                        "1. I compared the candidate approaches and identified the stronger one.\n"
                        "2. I summarized the reusable lesson instead of returning another task-typed answer."
                    ),
                }
            )

        async def complete_with_tools(self, *args, **kwargs):
            return await _complete_with_tools_via_complete(self, *args, **kwargs)

    executor = ExecutorAgent(llm=_LLM())
    monkeypatch.setattr(ExecutorAgent, "_has_training_evaluation", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(ExecutorAgent, "_count_training_candidates", lambda *_args, **_kwargs: 3)
    monkeypatch.setattr(executor, "_has_stepwise_reasoning", lambda *_args, **_kwargs: True)

    plan = {
        "candidates": [],
        "task_type": "forecast",
        "task_description": "test forecast task",
        "tool_names": "",
        "tool_notes_section": "",
        "skills_section": "",
        "system_memory": "",
        "modality_hint": "",
        "profiling_section": "",
        "subagent_slot_section": "",
        "subagent_slots": [],
        "fingerprint": {},
        "horizon": 2,
        "domain": "weather",
        "reanalysis_section": "",
        "_mode": "train",
        "_defer_learning_record": False,
        "_ground_truth": [1.0, 2.0],
        "indicator_output_keys": [],
        "modality": "timeseries",
    }
    envelope = TaskEnvelope(
        task=TaskSpec(task_id="test", task_type="forecast", description="test"),
        environment=EnvironmentSnapshot(data=[1.0], modalities=["timeseries"]),
    )

    result = await executor.act(plan, envelope)

    assert result["answer_type"] == "learning_summary"
    assert executor.llm.calls == 2


def test_react_system_prompt_explicitly_forbids_multi_turn_plans_in_one_reply():
    prompt = executor_prompt.REACT_SYSTEM_PROMPT.format(max_steps=10)

    assert "This reply is ONLY for the current turn." in prompt
    assert "Do not include a multi-step plan" in prompt
    assert "The framework, not you, will ask for the next turn separately." in prompt
    assert "Do not describe what you would do on the next turn." in prompt


def test_build_decision_section_explicitly_forbids_later_turn_plans():
    section = ExecutorAgent._build_decision_section(
        allow_finish=True,
        spawn_available=True,
    )

    assert "This reply is only for the current turn." in section
    assert "Do not include a plan for later turns" in section
    assert "then tool B, then finish" in section
