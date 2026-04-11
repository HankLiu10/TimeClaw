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
        if parsed.get("answer_type") == "learning_summary":
            answer = parsed.get("answer") or {}
            args = {
                "thought": parsed.get("thought", ""),
                "insight": answer.get("insight", ""),
                "recommendation": answer.get("recommendation", ""),
                "confidence": parsed.get("confidence"),
                "reasoning": parsed.get("reasoning", ""),
            }
        else:
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
    tool_name = parsed.get("tool", "")
    if tool_name == "spawn_subagent":
        tasks = []
        for item in (parsed.get("args") or {}).get("tasks", []):
            if not isinstance(item, dict):
                continue
            normalized = dict(item)
            if "slot_local_path_tools" not in normalized and "tools_hint" in normalized:
                tools_hint = normalized.get("tools_hint")
                if isinstance(tools_hint, str):
                    normalized["slot_local_path_tools"] = [
                        token.strip()
                        for token in tools_hint.replace("->", ",").split(",")
                        if token.strip()
                    ]
                elif isinstance(tools_hint, list):
                    normalized["slot_local_path_tools"] = tools_hint
            tasks.append(normalized)
        return LLMResponse(
            text="",
            parsed=None,
            tool_calls=[{
                "name": "emit_spawn_subagent",
                "arguments": {
                    "thought": parsed.get("thought", ""),
                    "tasks": tasks,
                },
            }],
            model="mock",
        )
    if tool_name in {
        "evaluate_against_gt",
        "evaluate_batch_against_gt",
        "batch_evaluate_against_gt",
    }:
        args = parsed.get("args") or {}
        arguments = {
            "thought": parsed.get("thought", ""),
            "tool": tool_name,
            "task_type": args.get("task_type", ""),
        }
        if "prediction" in args:
            arguments["prediction_json"] = json.dumps(args["prediction"], ensure_ascii=False)
        if "predictions" in args:
            arguments["predictions_json"] = json.dumps(args["predictions"], ensure_ascii=False)
        return LLMResponse(
            text="",
            parsed=None,
            tool_calls=[{
                "name": "emit_evaluate_call",
                "arguments": arguments,
            }],
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
                "args_json": json.dumps(parsed.get("args", {}), ensure_ascii=False),
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


def test_should_enforce_stepwise_reasoning_only_in_train_mode():
    assert ExecutorAgent._should_enforce_stepwise_reasoning("train") is True
    assert ExecutorAgent._should_enforce_stepwise_reasoning("test") is False
    assert ExecutorAgent._should_enforce_stepwise_reasoning(None) is False


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


def test_coerce_llm_decision_lifts_misnested_learning_summary_metadata():
    raw = {
        "done": True,
        "thought": "Reason: budget exhausted. Action: return summary.",
        "answer": {
            "insight": "forecast-then-indicator outperformed direct reasoning",
            "recommendation": "compare derived indicators against indicator-space GT",
            "answer_type": "learning_summary",
            "confidence": 0.91,
            "reasoning": "1. Two candidate paths were evaluated.",
        },
    }

    decision = ExecutorAgent._coerce_llm_decision(
        parsed=raw,
        text=json.dumps(raw, ensure_ascii=False),
        fallback_thought="parse failed",
    )

    assert decision["done"] is True
    assert decision["answer_type"] == "learning_summary"
    assert decision["confidence"] == 0.91
    assert decision["reasoning"] == "1. Two candidate paths were evaluated."
    assert decision["answer"] == {
        "insight": "forecast-then-indicator outperformed direct reasoning",
        "recommendation": "compare derived indicators against indicator-space GT",
    }


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
    assert "do not call index-sensitive tools with original-input indices" in msg


def test_build_subagent_tool_backed_result_reuses_exact_tool_answer():
    result = ExecutorAgent._build_subagent_tool_backed_result(
        artifact={"tool": "llm_reasoning_indicator", "answer": {"macd": [-1.2, -1.1]}},
        task_type="indicator",
        execution_log=[{"tool": "llm_reasoning_indicator"}],
        slot_id=3,
        task_description="predict macd",
        tools_hint=["llm_reasoning_indicator"],
        branch_metadata={"path_type": "direct_indicator_only"},
    )

    assert result["answer"] == {"macd": [-1.2, -1.1]}
    assert result["answer_type"] == "indicator"
    assert result["tool_chain"] == ["llm_reasoning_indicator"]
    assert "reuse that exact tool-produced payload unchanged" in result["reasoning"]


def test_thought_schema_requires_reason_then_action_labels():
    schema = ExecutorAgent._thought_schema()

    assert schema["type"] == "string"
    assert "Reason:` then `Action:`" in schema["description"]
    assert schema["pattern"] == r"^Reason:[\s\S]*Action:[\s\S]*$"


def test_filter_agent_available_tool_names_hides_ensemble_forecast():
    filtered = executor_prompt.filter_agent_available_tool_names(
        ["timesfm2_forecast", "ensemble_forecast", "llm_reasoning_forecast"]
    )

    assert filtered == ["timesfm2_forecast", "llm_reasoning_forecast"]


def test_filter_task_specific_candidates_removes_task_mismatched_reasoning_tools():
    filtered = ExecutorAgent._filter_task_specific_candidates(
        [
            "llm_reasoning_qa",
            "llm_reasoning_forecast",
            "llm_reasoning_classification",
            "basic_stats",
        ],
        task_type="forecast",
        fingerprint={},
    )

    assert filtered == ["llm_reasoning_forecast", "basic_stats"]


def test_filter_task_specific_candidates_hides_handle_missing_without_missing_values():
    filtered = ExecutorAgent._filter_task_specific_candidates(
        [
            "handle_missing",
            "llm_reasoning_qa",
            "llm_reasoning_classification",
            "window_stats",
        ],
        task_type="qa",
        fingerprint={"missing_ratio": 0.0},
    )

    assert filtered == ["llm_reasoning_qa", "window_stats"]


def test_task_mismatched_reasoning_tool_message_requires_task_compatible_reasoner():
    msg = ExecutorAgent._task_mismatched_reasoning_tool_message(
        task_type="forecast",
        tool_name="llm_reasoning_qa",
    )

    assert "task-mismatched reasoning tool" in msg
    assert "llm_reasoning_forecast" in msg
    qa_msg = ExecutorAgent._task_mismatched_reasoning_tool_message(
        task_type="qa",
        tool_name="llm_reasoning_classification",
    )
    assert "task-mismatched reasoning tool" in qa_msg
    assert "llm_reasoning_qa" in qa_msg


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


def test_has_stepwise_reasoning_prefers_candidate_paths_over_top_level_rewrite_tools():
    execution_log = [
        {
            "tool": "spawn_subagent",
            "output": [
                {
                    "success": True,
                    "result": {
                        "execution_log": [{"tool": "chronos2_forecast"}],
                    },
                },
                {
                    "success": True,
                    "result": {
                        "execution_log": [{"tool": "timesfm2_forecast"}],
                    },
                },
                {
                    "success": True,
                    "result": {
                        "execution_log": [{"tool": "llm_reasoning_forecast"}],
                    },
                },
            ],
        },
        {"tool": "entropy"},
        {"tool": "evaluate_batch_against_gt"},
        {"tool": "llm_reasoning_qa"},
        {"tool": "llm_reasoning_qa"},
    ]
    reasoning = (
        "1. The chronos2_forecast branch best preserved the short-term cooling and diurnal swing.\n"
        "2. The timesfm2_forecast branch was competitive but stayed slightly too warm on the realized path.\n"
        "3. The direct multimodal llm forecast branch over-smoothed the hourly structure.\n"
        "4. Ground-truth comparison separated the candidates, so I selected the Chronos-based result."
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


def test_stepwise_reasoning_correction_message_prefers_candidates_and_forbids_extra_tools():
    execution_log = [
        {
            "tool": "spawn_subagent",
            "output": [
                {
                    "success": True,
                    "result": {
                        "execution_log": [{"tool": "chronos2_forecast"}],
                    },
                },
                {
                    "success": True,
                    "result": {
                        "execution_log": [{"tool": "timesfm2_forecast"}],
                    },
                },
            ],
        },
        {"tool": "entropy"},
        {"tool": "llm_reasoning_qa"},
    ]

    msg = ExecutorAgent._stepwise_reasoning_correction_message(execution_log)

    assert "chronos2_forecast, timesfm2_forecast" in msg
    assert "llm_reasoning_qa" not in msg
    assert "Do not call another tool" in msg


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


def test_count_training_spawn_candidates_ignores_top_level_only_candidates():
    execution_log = [
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

    assert ExecutorAgent._count_training_spawn_candidates(
        execution_log,
        "qa",
    ) == 0


def test_count_training_spawn_candidates_reads_only_valid_spawned_branches():
    execution_log = [
        {
            "tool": "spawn_subagent",
            "output": {
                "results": [
                    {
                        "success": True,
                        "result": {
                            "slot_id": 0,
                            "answer": "No Correlation",
                            "execution_log": [{"tool": "llm_reasoning_correlation"}],
                        },
                    },
                    {
                        "success": True,
                        "result": {
                            "slot_id": 1,
                            "answer": "Moderate Positive Correlation",
                            "execution_log": [{"tool": "window_stats"}],
                        },
                    },
                    {
                        "success": False,
                        "result": {
                            "slot_id": 2,
                            "execution_log": [{"tool": "basic_stats"}],
                        },
                    },
                ],
            },
        },
        {
            "tool": "llm_reasoning_correlation",
            "success": True,
            "observation": {"answer": "Weak Positive Correlation"},
        },
    ]

    assert ExecutorAgent._count_training_spawn_candidates(
        execution_log,
        "correlation",
    ) == 2


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


def test_collect_batch_eval_predictions_canonicalizes_indicator_alias_keys():
    execution_log = [
        {
            "tool": "spawn_subagent",
            "output": [
                {
                    "success": True,
                    "result": {
                        "slot_id": 1,
                        "answer": {"indicator_name": [0.11, 0.22]},
                        "execution_log": [{"tool": "llm_reasoning_indicator"}],
                    },
                }
            ],
        },
        {
            "tool": "llm_reasoning_indicator",
            "success": True,
            "observation": {
                "upper_bollinger_band": [140.097, 140.161, 140.28],
            },
        },
    ]

    preds = ExecutorAgent._collect_batch_eval_predictions(
        execution_log,
        task_type="indicator",
        horizon=2,
        indicator_output_keys=["macd"],
    )

    assert preds == [
        {"name": "slot_1", "prediction": {"macd": [0.11, 0.22]}},
    ]


def test_normalize_batch_eval_predictions_arg_unwraps_wrapped_branch_envelopes():
    predictions = ExecutorAgent._normalize_batch_eval_predictions_arg(
        [
            {
                "name": "slot_3",
                "prediction": {
                    "thought": "wrapped envelope",
                    "answer_type": "indicator",
                    "answer": {"upper_bb": [1.0, 2.0, 3.0]},
                },
            },
            {
                "name": "top_level::llm_reasoning_indicator",
                "prediction": {"upper_bb": [1.1, 2.1, 3.1]},
            },
        ],
        task_type="indicator",
        horizon=3,
        indicator_output_keys=["upper_bb"],
    )

    assert predictions == [
        {
            "name": "slot_3",
            "prediction": {"upper_bb": [1.0, 2.0, 3.0]},
        },
        {
            "name": "top_level::llm_reasoning_indicator",
            "prediction": {"upper_bb": [1.1, 2.1, 3.1]},
        }
    ]


def test_training_completion_contract_satisfied_requires_eval_and_two_spawned_candidates():
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
                ],
            },
        },
    ]
    assert ExecutorAgent._training_completion_contract_satisfied(
        execution_log,
        task_type="correlation",
    ) is True


def test_training_completion_contract_satisfied_rejects_top_level_only_candidates_without_spawn():
    execution_log = [
        {
            "tool": "llm_reasoning_indicator",
            "success": True,
            "observation": {"upper_bb": [1.0, 2.0, 3.0]},
        },
        {
            "tool": "prophet_forecast",
            "success": True,
            "observation": {"upper_bb": [0.8, 1.8, 2.8]},
        },
        {
            "tool": "evaluate_batch_against_gt",
            "success": True,
            "observation": {
                "results": [
                    {"metrics": {"avg_mae": 0.1}},
                    {"metrics": {"avg_mae": 0.2}},
                ],
            },
        },
    ]

    assert ExecutorAgent._training_completion_contract_satisfied(
        execution_log,
        task_type="indicator",
        horizon=3,
        indicator_output_keys=["upper_bb"],
    ) is False


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


def test_build_main_budget_message_prioritizes_missing_training_actions_before_finish():
    execution_log = [
        {
            "tool": "spawn_subagent",
            "output": {
                "results": [
                    {
                        "success": True,
                        "result": {
                            "answer": "C",
                            "execution_log": [{"tool": "llm_reasoning_qa"}],
                        },
                    },
                ],
            },
        },
    ]
    msg = ExecutorAgent._build_main_budget_message(
        step=9,
        execution_log=execution_log,
        plan={
            "_mode": "train",
            "_defer_learning_record": False,
            "task_type": "qa",
            "horizon": None,
            "indicator_output_keys": [],
        },
    )

    assert "not satisfied yet" in msg
    assert "need at least 2" in msg
    assert "no GT evaluation" in msg
    assert "Do not reserve the next turn for `done=true` yet" in msg


def test_build_main_budget_message_final_turn_uses_missing_eval_not_finish():
    execution_log = [
        {
            "tool": "spawn_subagent",
            "output": {
                "results": [
                    {
                        "success": True,
                        "result": {
                            "answer": "C",
                            "execution_log": [{"tool": "llm_reasoning_qa"}],
                        },
                    },
                    {
                        "success": True,
                        "result": {
                            "answer": "D",
                            "execution_log": [{"tool": "window_stats"}],
                        },
                    },
                ],
            },
        },
    ]
    msg = ExecutorAgent._build_main_budget_message(
        step=10,
        execution_log=execution_log,
        plan={
            "_mode": "train",
            "_defer_learning_record": False,
            "task_type": "qa",
            "horizon": None,
            "indicator_output_keys": [],
        },
    )

    assert "still unmet" in msg
    assert "no GT evaluation" in msg
    assert "run GT evaluation" in msg
    assert "Do not return `done=true` yet" in msg


def test_main_requires_eval_only_turn_as_soon_as_candidates_exist_but_no_eval():
    execution_log = [
        {
            "tool": "spawn_subagent",
            "output": {
                "results": [
                    {
                        "success": True,
                        "result": {
                            "answer": "C",
                            "execution_log": [{"tool": "llm_reasoning_qa"}],
                        },
                    },
                    {
                        "success": True,
                        "result": {
                            "answer": "D",
                            "execution_log": [{"tool": "window_stats"}],
                        },
                    },
                ],
            },
        },
    ]

    assert ExecutorAgent._main_requires_eval_only_turn(
        step=3,
        execution_log=execution_log,
        plan={
            "_mode": "train",
            "_defer_learning_record": False,
            "task_type": "qa",
            "horizon": None,
            "indicator_output_keys": [],
        },
    ) is True


def test_main_requires_eval_only_turn_is_false_without_enough_spawned_candidates():
    execution_log = [
        {
            "tool": "llm_reasoning_qa",
            "success": True,
            "output": {"answer": "C"},
        },
    ]

    assert ExecutorAgent._main_requires_eval_only_turn(
        step=executor_module.MAX_REACT_STEPS,
        execution_log=execution_log,
        plan={
            "_mode": "train",
            "_defer_learning_record": False,
            "task_type": "qa",
            "horizon": None,
            "indicator_output_keys": [],
        },
    ) is False


def test_main_requires_spawn_only_turn_when_two_turns_left_and_spawn_is_only_viable_path():
    execution_log = [
        {
            "tool": "llm_reasoning_classification",
            "success": True,
            "observation": {"answer": "stable"},
        },
    ]

    assert ExecutorAgent._main_requires_spawn_only_turn(
        step=executor_module.MAX_REACT_STEPS - 1,
        execution_log=execution_log,
        plan={
            "_mode": "train",
            "_defer_learning_record": False,
            "task_type": "classification",
            "horizon": None,
            "indicator_output_keys": [],
        },
        spawn_candidate_available=True,
    ) is True


def test_main_requires_spawn_only_turn_when_three_turns_left_and_spawn_is_only_viable_path():
    execution_log = [
        {
            "tool": "llm_reasoning_classification",
            "success": True,
            "observation": {"answer": "stable"},
        },
    ]

    assert ExecutorAgent._main_requires_spawn_only_turn(
        step=executor_module.MAX_REACT_STEPS - 2,
        execution_log=execution_log,
        plan={
            "_mode": "train",
            "_defer_learning_record": False,
            "task_type": "classification",
            "horizon": None,
            "indicator_output_keys": [],
        },
        spawn_candidate_available=True,
    ) is True


def test_main_requires_spawn_only_turn_is_true_as_soon_as_unspawned_candidate_exists():
    execution_log = [
        {
            "tool": "llm_reasoning_classification",
            "success": True,
            "observation": {"answer": "stable"},
        },
    ]

    assert ExecutorAgent._main_requires_spawn_only_turn(
        step=2,
        execution_log=execution_log,
        plan={
            "_mode": "train",
            "_defer_learning_record": False,
            "task_type": "classification",
            "horizon": None,
            "indicator_output_keys": [],
        },
        spawn_candidate_available=True,
    ) is True


def test_main_requires_spawn_only_turn_is_false_without_spawn_candidate():
    execution_log = [
        {
            "tool": "llm_reasoning_classification",
            "success": True,
            "observation": {"answer": "stable"},
        },
    ]

    assert ExecutorAgent._main_requires_spawn_only_turn(
        step=executor_module.MAX_REACT_STEPS - 1,
        execution_log=execution_log,
        plan={
            "_mode": "train",
            "_defer_learning_record": False,
            "task_type": "classification",
            "horizon": None,
            "indicator_output_keys": [],
        },
        spawn_candidate_available=False,
    ) is False


def test_latest_task_valid_tool_answer_returns_latest_valid_branch_payload():
    execution_log = [
        {
            "tool": "chronos2_forecast",
            "success": True,
            "output": {"point_forecast": [1.0, 2.0]},
        },
        {
            "tool": "llm_reasoning_indicator",
            "success": True,
            "output": {"answer": {"indicator_name": [0.1, 0.2]}},
        },
        {
            "tool": "llm_reasoning_indicator",
            "success": True,
            "output": {"answer": {"macd": [0.3, 0.4]}},
        },
    ]

    latest = ExecutorAgent._latest_task_valid_tool_answer(
        task_type="indicator",
        execution_log=execution_log,
        indicator_output_keys=["macd"],
    )

    assert latest == {
        "tool": "llm_reasoning_indicator",
        "answer": {"macd": [0.3, 0.4]},
    }


def test_latest_task_valid_tool_answer_normalizes_requested_indicator_key():
    execution_log = [
        {
            "tool": "llm_reasoning_indicator",
            "success": True,
            "output": {"answer": {"requested_indicator_key": [0.3, 0.4]}},
        },
    ]

    latest = ExecutorAgent._latest_task_valid_tool_answer(
        task_type="indicator",
        execution_log=execution_log,
        indicator_output_keys=["macd"],
    )

    assert latest == {
        "tool": "llm_reasoning_indicator",
        "answer": {"macd": [0.3, 0.4]},
    }


def test_latest_task_valid_tool_answer_ignores_explanatory_text_for_classification():
    execution_log = [
        {
            "tool": "llm_reasoning_qa",
            "success": True,
            "output": {
                "answer": (
                    "Step 1: The series is bearish. Step 2: Mixed news softens the move. "
                    "Final label: -2% ~ -4%"
                )
            },
        },
    ]

    latest = ExecutorAgent._latest_task_valid_tool_answer(
        task_type="classification",
        execution_log=execution_log,
    )

    assert latest is None


def test_latest_task_valid_tool_answer_ignores_raw_non_answer_dict_for_qa():
    execution_log = [
        {
            "tool": "value_at",
            "success": True,
            "output": {"index": -1, "resolved_index": 389, "length": 390, "value": 91.5},
        },
    ]

    latest = ExecutorAgent._latest_task_valid_tool_answer(
        task_type="qa",
        execution_log=execution_log,
    )

    assert latest is None


def test_is_task_answer_backed_by_tool_output_accepts_requested_indicator_key_alias():
    execution_log = [
        {
            "tool": "llm_reasoning_indicator",
            "success": True,
            "output": {"answer": {"requested_indicator_key": [0.3, 0.4]}},
        },
    ]

    assert ExecutorAgent._is_task_answer_backed_by_tool_output(
        "indicator",
        {"macd": [0.3, 0.4]},
        execution_log,
        indicator_output_keys=["macd"],
    ) is True


def test_count_training_candidates_dedupes_repeated_top_level_tool_path():
    execution_log = [
        {
            "tool": "llm_reasoning_qa",
            "success": True,
            "observation": {"answer": "C"},
        },
        {
            "tool": "llm_reasoning_qa",
            "success": True,
            "observation": {"answer": "C"},
        },
        {
            "tool": "llm_reasoning_qa",
            "success": True,
            "observation": {"answer": "D"},
        },
        {
            "tool": "entropy",
            "success": True,
            "observation": {"sample_entropy": 0.4},
        },
    ]

    assert ExecutorAgent._count_training_candidates(
        execution_log,
        task_type="qa",
    ) == 1


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
    monkeypatch.setattr(executor, "_count_training_spawn_candidates", lambda *_args, **_kwargs: 0)
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
async def test_executor_rejects_terminal_learning_summary_when_budget_exhaustion_still_violates_training_contract(monkeypatch):
    class _Resp:
        def __init__(self, parsed):
            self.parsed = parsed
            self.text = json.dumps(parsed, ensure_ascii=False)

    class _LLM:
        def __init__(self):
            self.calls = 0

        async def complete(self, *_args, **_kwargs):
            self.calls += 1
            if self.calls <= 10:
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
                    "thought": "Reason: budget exhausted. Action: summarize the failure pattern.",
                    "answer": {
                        "insight": "The run repeated the same top-level path and never completed GT evaluation.",
                        "recommendation": "Create distinct candidate paths earlier and evaluate before spending the remaining budget on redundant tool calls.",
                    },
                    "answer_type": "learning_summary",
                    "confidence": 0.82,
                    "reasoning": "1. The training contract remained unsatisfied at the step limit.",
                }
            )

        async def complete_with_tools(self, *args, **kwargs):
            return await _complete_with_tools_via_complete(self, *args, **kwargs)

    async def _fake_execute_tool(_name, _args, *_rest, **_kwargs):
        return ToolResult(
            ok=True,
            data={"mean": 1.0},
            error=None,
            metadata={},
        )

    executor = ExecutorAgent(llm=_LLM())
    monkeypatch.setattr(executor, "_execute_tool", _fake_execute_tool)
    monkeypatch.setattr(executor, "_has_stepwise_reasoning", lambda *_args, **_kwargs: True)

    plan = {
        "candidates": ["basic_stats"],
        "task_type": "forecast",
        "task_description": "test forecast task",
        "tool_names": "basic_stats",
        "tool_notes_section": "",
        "skills_section": "",
        "system_memory": "",
        "modality_hint": "",
        "profiling_section": "",
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
async def test_executor_rejects_top_level_task_mismatched_reasoning_tool(monkeypatch):
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
                        "thought": "Reason: try a direct multimodal answer. Action: call llm_reasoning_qa.",
                        "tool": "llm_reasoning_qa",
                        "args": {"question": "bad tool"},
                    }
                )
            if self.calls == 2:
                return _Resp(
                    {
                        "done": False,
                        "thought": "Reason: use the task-compatible forecast reasoner. Action: call llm_reasoning_forecast.",
                        "tool": "llm_reasoning_forecast",
                        "args": {"question": "forecast"},
                    }
                )
            return _Resp(
                {
                    "done": True,
                    "thought": "Reason: a real tool already produced the forecast. Action: finish.",
                    "answer": [1.0, 2.0],
                    "answer_type": "forecast",
                    "confidence": 0.9,
                    "reasoning": "1. llm_reasoning_forecast produced the final forecast candidate.",
                }
            )

        async def complete_with_tools(self, *args, **kwargs):
            return await _complete_with_tools_via_complete(self, *args, **kwargs)

    executed_tools: list[str] = []

    async def _fake_execute_tool(tool_name, *_args, **_kwargs):
        executed_tools.append(tool_name)
        return ToolResult(
            success=True,
            data=[1.0, 2.0],
            error=None,
            metadata={},
        )

    executor = ExecutorAgent(llm=_LLM())
    monkeypatch.setattr(executor, "_execute_tool", _fake_execute_tool)
    monkeypatch.setattr(executor, "_has_stepwise_reasoning", lambda *_args, **_kwargs: True)

    plan = {
        "candidates": ["llm_reasoning_qa", "llm_reasoning_forecast"],
        "task_type": "forecast",
        "task_description": "test forecast task",
        "tool_names": "llm_reasoning_qa, llm_reasoning_forecast",
        "tool_notes_section": "",
        "skills_section": "",
        "system_memory": "",
        "modality_hint": "",
        "profiling_section": "",
        "subagent_slot_section": "",
        "subagent_slots": [],
        "fingerprint": {"missing_ratio": 0.0},
        "horizon": 2,
        "domain": "weather",
        "reanalysis_section": "",
        "_mode": "test",
        "_defer_learning_record": False,
        "_ground_truth": None,
        "indicator_output_keys": [],
        "modality": "multimodal",
    }
    envelope = TaskEnvelope(
        task=TaskSpec(task_id="test", task_type="forecast", description="test"),
        environment=EnvironmentSnapshot(data=[1.0], modalities=["timeseries", "text"]),
    )

    result = await executor.act(plan, envelope)

    assert result["answer_type"] == "forecast"
    assert executed_tools == ["llm_reasoning_forecast"]


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
    monkeypatch.setattr(executor, "_count_training_spawn_candidates", lambda *_args, **_kwargs: 2)
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
                        "thought": "Reason: first exploration round. Action: spawn two branches.",
                        "tool": "spawn_subagent",
                        "args": {
                            "tasks": [
                                {"slot_id": 0, "task": "branch a"},
                                {"slot_id": 1, "task": "branch b"},
                            ]
                        },
                    }
                )
            if self.calls == 2:
                return _Resp(
                    {
                        "done": False,
                        "thought": "Reason: one branch failed, so I still need another exploration round with at least two branches. Action: spawn two more branches.",
                        "tool": "spawn_subagent",
                        "args": {
                            "tasks": [
                                    {"slot_id": 2, "task": "branch d"},
                                    {"slot_id": 3, "task": "branch e"},
                                ]
                            },
                        }
                    )
            if self.calls == 3:
                return _Resp(
                    {
                        "done": False,
                        "thought": "Reason: I now have two valid candidates. Action: batch-evaluate them.",
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
                        "recommendation": "Use a second spawn round with fresh slots, then evaluate immediately once enough candidates exist.",
                    },
                    "answer_type": "learning_summary",
                    "confidence": 0.9,
                    "reasoning": (
                        "1. The first spawn round produced only one valid candidate because one branch failed.\n"
                        "2. A second spawn round used fresh slots to add another valid candidate.\n"
                        "3. After reaching two valid candidates, I evaluated them and summarized the lesson."
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
                "n_tasks": 2,
                "n_success": 1,
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
                        "success": False,
                        "result": {
                            "slot_id": 1,
                            "execution_log": [{"tool": "llm_reasoning_correlation"}],
                        },
                    },
                ],
                "observation": "first spawn complete",
            }
        return {
            "n_tasks": 2,
            "n_success": 1,
            "results": [
                {
                    "success": True,
                    "result": {
                        "slot_id": 2,
                        "answer": "Moderate Positive Correlation",
                        "answer_type": "correlation",
                        "execution_log": [{"tool": "llm_reasoning_correlation"}],
                    },
                },
                {
                    "success": False,
                    "result": {
                        "slot_id": 3,
                        "execution_log": [{"tool": "window_stats"}],
                    },
                },
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
    ] == [[0, 1], [2, 3]]


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
    ]

    error = ExecutorAgent._validate_spawn_subagent_tasks(
        [{"slot_id": 0, "task": "path a"}],
        slots,
        min_tasks=2,
    )

    assert "at least 2 distinct branches" in error


def test_validate_spawn_subagent_tasks_still_requires_two_branches_on_later_rounds():
    slots = [
        {"slot_id": 0},
        {"slot_id": 1},
        {"slot_id": 2},
    ]

    error = ExecutorAgent._validate_spawn_subagent_tasks(
        [{"slot_id": 2, "task": "path c"}],
        slots,
        min_tasks=2,
        disallowed_slot_ids={0, 1},
    )

    assert "at least 2 distinct branches" in error


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
        ],
        task_type="forecast",
    )

    assert "slot_id=0: visible_tools=timesfm2_forecast, llm_reasoning_forecast" in text
    assert "slot_id=1: visible_tools=timesfm2_forecast, llm_reasoning_forecast" in text
    assert "slot_id=2: visible_tools=window_stats" in text
    assert text.count("timesfm2_forecast, llm_reasoning_forecast") == 2
    assert "recommended_slot_local_path_tools=timesfm2_forecast -> llm_reasoning_forecast" in text


def test_build_subagent_slot_section_surfaces_slot_local_classification_paths():
    text = ExecutorAgent._build_subagent_slot_section(
        [
            {
                "slot_id": 7,
                "visible_candidates": ["detect_trend", "llm_reasoning_classification"],
            },
            {
                "slot_id": 1,
                "visible_candidates": ["llm_reasoning_classification"],
            },
        ],
        task_type="classification",
    )

    assert (
        "slot_id=7: visible_tools=detect_trend, llm_reasoning_classification | "
        "recommended_slot_local_path_tools=detect_trend -> llm_reasoning_classification; "
        "llm_reasoning_classification"
    ) in text
    assert (
        "slot_id=1: visible_tools=llm_reasoning_classification | "
        "recommended_slot_local_path_tools=llm_reasoning_classification"
    ) in text


def test_strip_train_path_priors_from_memory_keeps_task_insights_only():
    raw = (
        "## MEMORY — Learned Insights\n\n"
        "### weather_aligned_long_trend_past Specific\n"
        "- Rule: Treat each 24 hourly readings as one day. | When: hourly weather trend classification | Prefer: segment, llm_reasoning_classification | Avoid: detect_trend, resample | Why: day-level aggregation matters. | Evidence: 7 note(s)\n"
        "- Rule: Return exactly one label from increasing, decreasing, or stable. | When: trend classification prompt | Why: output contract is strict. | Evidence: 10 note(s)\n"
    )

    stripped = ExecutorAgent._strip_train_path_priors_from_memory(raw)

    assert "Prefer:" not in stripped
    assert "Avoid:" not in stripped
    assert "Rule: Treat each 24 hourly readings as one day." in stripped
    assert "Why: day-level aggregation matters." in stripped
    assert "Return exactly one label from increasing, decreasing, or stable." in stripped


def test_select_train_spawn_prior_picks_compatible_supported_path():
    selected = ExecutorAgent._select_train_spawn_prior(
        path_priors=[
            {"path": ["segment", "llm_reasoning_classification"], "support": 7},
            {"path": ["detect_trend"], "support": 9},
            {"path": ["resample", "llm_reasoning_classification"], "support": 3},
        ],
        subagent_slots=[
            {"slot_id": 0, "visible_candidates": ["detect_trend", "llm_reasoning_classification"]},
            {"slot_id": 1, "visible_candidates": ["segment", "llm_reasoning_classification"]},
        ],
        task_type="classification",
    )

    assert selected == ["detect_trend"]


def test_validate_first_spawn_prior_mix_requires_guided_and_alternative():
    prior = ["segment", "llm_reasoning_classification"]

    missing_guided = ExecutorAgent._validate_first_spawn_prior_mix(
        [
            {"tools_hint": ["detect_trend"]},
            {"tools_hint": ["ets_forecast"]},
        ],
        prior_path=prior,
    )
    missing_alternative = ExecutorAgent._validate_first_spawn_prior_mix(
        [
            {"tools_hint": ["segment", "llm_reasoning_classification"]},
            {"tools_hint": ["segment", "window_stats"]},
        ],
        prior_path=prior,
    )
    ok = ExecutorAgent._validate_first_spawn_prior_mix(
        [
            {"tools_hint": ["segment", "llm_reasoning_classification"]},
            {"tools_hint": ["detect_trend"]},
        ],
        prior_path=prior,
    )

    assert "guided by the historical preferred branch family" in missing_guided
    assert "distinct alternative branch" in missing_alternative
    assert ok is None


def test_validate_first_spawn_prior_mix_accepts_matching_prior_family_anchor():
    ok = ExecutorAgent._validate_first_spawn_prior_mix(
        [
            {"tools_hint": ["segment", "llm_reasoning_classification"]},
            {"tools_hint": ["prophet_forecast", "llm_reasoning_classification"]},
        ],
        prior_path=["segment", "llm_reasoning_classification"],
    )

    assert ok is None


def test_validate_first_spawn_prior_mix_accepts_anchor_only_hint_for_anchored_workflow():
    ok = ExecutorAgent._validate_first_spawn_prior_mix(
        [
            {"tools_hint": ["chronos2_forecast"]},
            {"tools_hint": ["llm_reasoning_indicator"]},
        ],
        prior_path=["arima_forecast", "chronos2_forecast", "llm_reasoning_indicator"],
    )

    assert ok is None


def test_validate_first_spawn_prior_mix_accepts_preferred_tool_family_anchor():
    ok = ExecutorAgent._validate_first_spawn_prior_mix(
        [
            {"tools_hint": ["chronos2_forecast"]},
            {"tools_hint": ["llm_reasoning_forecast"]},
        ],
        prior_path=["timesfm2_forecast", "chronos2_forecast"],
    )

    assert ok is None


def test_validate_first_spawn_prior_mix_treats_helper_tool_as_non_guided_alternative():
    ok = ExecutorAgent._validate_first_spawn_prior_mix(
        [
            {"tools_hint": ["chronos2_forecast"]},
            {"tools_hint": ["window_stats"]},
        ],
        prior_path=["chronos2_forecast", "ets_forecast", "window_stats"],
    )

    assert ok is None


def test_validate_first_spawn_prior_mix_accepts_validation_family_anywhere_in_branch():
    ok = ExecutorAgent._validate_first_spawn_prior_mix(
        [
            {"tools_hint": ["llm_reasoning_indicator", "bound_check"]},
            {"tools_hint": ["arima_forecast", "llm_reasoning_indicator"]},
        ],
        prior_path=["consistency_check", "bound_check"],
    )

    assert ok is None


def test_validate_first_spawn_prior_mix_accepts_guided_superset_branch():
    prior = ["consistency_check"]

    ok = ExecutorAgent._validate_first_spawn_prior_mix(
        [
            {"tools_hint": ["chronos2_forecast", "consistency_check", "llm_reasoning_classification"]},
            {"tools_hint": ["llm_reasoning_classification"]},
        ],
        prior_path=prior,
    )

    assert ok is None


def test_validate_first_spawn_prior_mix_treats_shared_anchor_as_guided_not_alternative():
    msg = ExecutorAgent._validate_first_spawn_prior_mix(
        [
            {"tools_hint": ["slice", "llm_reasoning_qa"]},
            {"tools_hint": ["slice", "value_at", "window_stats"]},
        ],
        prior_path=["slice", "llm_reasoning_qa"],
    )

    assert "distinct alternative branch" in msg


def test_validate_first_spawn_prior_mix_accepts_string_tools_hint_and_alternative():
    ok = ExecutorAgent._validate_first_spawn_prior_mix(
        [
            {
                "tools_hint": "llm_reasoning_classification",
                "branch_metadata": {
                    "uses_prior": True,
                    "anchor_tool": "llm_reasoning_classification",
                },
            },
            {
                "tools_hint": "ets_forecast, window_stats, llm_reasoning_classification",
                "branch_metadata": {
                    "uses_prior": False,
                    "anchor_tool": "ets_forecast",
                },
            },
        ],
        prior_path=["llm_reasoning_classification"],
    )

    assert ok is None


def test_validate_first_spawn_prior_mix_prefers_branch_metadata_tools_hint_for_direct_prior_family():
    ok = ExecutorAgent._validate_first_spawn_prior_mix(
        [
            {
                "tools_hint": ["slice", "llm_reasoning_correlation"],
                "branch_metadata": {
                    "tools_hint": ["llm_reasoning_correlation"],
                },
            },
            {
                "tools_hint": ["summarize", "llm_reasoning_correlation"],
                "branch_metadata": {
                    "tools_hint": ["summarize", "llm_reasoning_correlation"],
                },
            },
        ],
        prior_path=["llm_reasoning_correlation"],
    )

    assert ok is None


def test_task_uses_prior_path_parses_string_tools_hint():
    assert ExecutorAgent._task_uses_prior_path(
        {"tools_hint": "segment, llm_reasoning_classification"},
        ["segment", "llm_reasoning_classification"],
    ) is True
    assert ExecutorAgent._task_uses_prior_path(
        {"tools_hint": "chronos2_forecast"},
        ["timesfm2_forecast", "chronos2_forecast"],
    ) is True
    assert ExecutorAgent._task_uses_prior_path(
        {"tools_hint": "llm_reasoning_indicator, bound_check"},
        ["consistency_check", "bound_check"],
    ) is True
    assert ExecutorAgent._task_uses_prior_path(
        {"tools_hint": "llm_reasoning_indicator"},
        ["llm_reasoning_indicator"],
    ) is True
    assert ExecutorAgent._task_uses_prior_path(
        {"tools_hint": "arima_forecast, llm_reasoning_indicator"},
        ["llm_reasoning_indicator"],
    ) is False
    assert ExecutorAgent._task_uses_prior_path(
        {"tools_hint": "slice, value_at, window_stats"},
        ["slice", "llm_reasoning_qa"],
    ) is False
    assert ExecutorAgent._task_uses_prior_path(
        {"tools_hint": "basic_stats, llm_reasoning_classification"},
        ["segment", "llm_reasoning_classification"],
    ) is False


def test_prior_anchor_tool_prefers_non_llm_tool():
    assert ExecutorAgent._prior_anchor_tool(
        ["llm_reasoning_indicator", "consistency_check"]
    ) == "consistency_check"
    assert ExecutorAgent._prior_anchor_tool(
        ["segment", "llm_reasoning_classification"]
    ) == "segment"
    assert ExecutorAgent._prior_anchor_tool(
        ["timesfm2_forecast", "chronos2_forecast"]
    ) == "timesfm2_forecast"


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


def test_normalize_spawn_tasks_rejects_task_mismatched_reasoning_hint():
    normalized, error = ExecutorAgent._normalize_spawn_tasks(
        [
            {
                "slot_id": 0,
                "task": "classify with qa",
                "tools_hint": "llm_reasoning_qa",
            }
        ],
        plan={
            "task_type": "classification",
            "subagent_slots": [
                {
                    "slot_id": 0,
                    "visible_candidates": ["llm_reasoning_qa", "llm_reasoning_classification"],
                }
            ],
        },
    )

    assert normalized == []
    assert "task-mismatched reasoning tool" in error
    assert "llm_reasoning_classification" in error


def test_normalize_spawn_tasks_rejects_classification_reasoning_for_qa_branch():
    normalized, error = ExecutorAgent._normalize_spawn_tasks(
        [
            {
                "slot_id": 0,
                "task": "answer the mcqa",
                "tools_hint": "llm_reasoning_classification",
            }
        ],
        plan={
            "task_type": "qa",
            "subagent_slots": [
                {
                    "slot_id": 0,
                    "visible_candidates": ["llm_reasoning_classification", "llm_reasoning_qa"],
                }
            ],
        },
    )

    assert normalized == []
    assert "task-mismatched reasoning tool" in error
    assert "llm_reasoning_qa" in error


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


@pytest.mark.asyncio
async def test_subexecutor_uses_latest_task_valid_answer_as_terminal_fallback(monkeypatch):
    class _DummyLLM:
        async def complete(self, *_args, **_kwargs):
            raise AssertionError("LLM.complete should not be called directly in this test")

    executor = ExecutorAgent(llm=_DummyLLM())
    decision_calls = 0
    terminal_review_calls = 0

    async def _fake_request_structured_decision(*_args, **_kwargs):
        nonlocal decision_calls
        decision_calls += 1
        if decision_calls == 1:
            return (
                _decision_response(
                    {
                        "thought": "Reason: call the classification tool. Action: run it now.",
                        "done": False,
                        "tool": "llm_reasoning_classification",
                        "args": {},
                    }
                ),
                {
                    "thought": "Reason: call the classification tool. Action: run it now.",
                    "done": False,
                    "tool": "llm_reasoning_classification",
                    "args": {},
                },
            )
        return (
            _decision_response(
                {
                    "thought": "Reason: do not finish yet. Action: malformed non-terminal reply.",
                    "done": False,
                }
            ),
            {
                "thought": "Reason: do not finish yet. Action: malformed non-terminal reply.",
                "done": False,
                "_parse_failed": True,
            },
        )

    async def _fake_request_terminal_path_review(*_args, **_kwargs):
        nonlocal terminal_review_calls
        terminal_review_calls += 1
        return {
            "thought": "Reason: terminal review still failed. Action: invalid finish.",
            "done": True,
            "answer": None,
            "answer_type": "classification",
            "confidence": 0.1,
            "reasoning": "1. invalid",
        }

    async def _fake_execute_tool(tool_name, tool_args, data, text_data, images):
        assert tool_name == "llm_reasoning_classification"
        return ToolResult(success=True, data={"answer": "stable"}, error=None)

    monkeypatch.setattr(executor, "_request_structured_decision", _fake_request_structured_decision)
    monkeypatch.setattr(executor, "_request_terminal_path_review", _fake_request_terminal_path_review)
    monkeypatch.setattr(executor, "_execute_tool", _fake_execute_tool)
    monkeypatch.setattr(executor_module, "MAX_SUB_REACT_STEPS", 2)

    result = await executor._run_sub_executor(
        idx=0,
        task_description="Return the trend label.",
        tools_hint="llm_reasoning_classification",
        branch_metadata={"path": "direct_classification"},
        plan={
            "task_type": "classification",
            "task_description": "Classify the next trend.",
            "fingerprint": {},
            "profiling_section": "",
            "system_memory": "",
            "modality_hint": "",
            "skills_section": "",
            "tool_notes": {},
            "domain": "weather",
            "candidates": ["llm_reasoning_classification"],
            "_mode": "train",
        },
        envelope=TaskEnvelope(
            task=TaskSpec(
                task_id="subexecutor-auto-finish",
                task_type="classification",
                description="dummy",
            ),
            environment=EnvironmentSnapshot(
                data=[1.0, 2.0, 3.0],
                text_data=[],
                images=[],
            ),
            trace_id="test-subexecutor-auto-finish",
        ),
        slot_id=0,
        slot_spec={"slot_id": 0, "visible_candidates": ["llm_reasoning_classification"]},
    )

    assert decision_calls == 2
    assert terminal_review_calls == 1
    assert result["answer"] == "stable"
    assert result["answer_type"] == "classification"
    assert result["tool_chain"] == ["llm_reasoning_classification"]


def test_build_main_output_contract_section_includes_learning_summary_contract_for_training():
    text = ExecutorAgent._build_main_output_contract_section(
        task_type="forecast",
        horizon=24,
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
        task_type="forecast",
        horizon=24,
        is_train=False,
        defer_learning_record=False,
        indicator_output_keys=["macd"],
        allow_finish_current_turn=True,
    )

    assert "learning_summary" not in text
    assert "Return ONLY a JSON numeric array" in text
    assert "Do NOT return prose like `Predicted Temperatures: ...`." in text


def test_decision_tools_use_learning_summary_finish_schema_in_training_mode():
    tools = ExecutorAgent._decision_tools(
        allow_tool_call=True,
        allow_finish=True,
        finish_mode="learning_summary",
    )

    finish_tool = next(tool for tool in tools if tool["function"]["name"] == "emit_final_answer")
    props = finish_tool["function"]["parameters"]["properties"]
    required = finish_tool["function"]["parameters"]["required"]

    assert "insight" in props
    assert "recommendation" in props
    assert "answer_json" not in props
    assert "answer_type" not in props
    assert required == ["thought", "insight", "recommendation", "confidence", "reasoning"]


def test_decision_tools_split_spawn_and_evaluate_from_ordinary_tool_calls():
    tools = ExecutorAgent._decision_tools(
        allow_tool_call=True,
        allow_finish=False,
        ordinary_tool_names=["slice", "summarize"],
        evaluate_tool_names=["evaluate_against_gt", "evaluate_batch_against_gt"],
        spawn_available=True,
        finish_mode="task",
    )

    names = {tool["function"]["name"] for tool in tools}
    ordinary_tool = next(tool for tool in tools if tool["function"]["name"] == "emit_tool_call")
    spawn_tool = next(tool for tool in tools if tool["function"]["name"] == "emit_spawn_subagent")
    evaluate_tool = next(tool for tool in tools if tool["function"]["name"] == "emit_evaluate_call")

    assert "emit_tool_call" in names
    assert "emit_spawn_subagent" in names
    assert "emit_evaluate_call" in names
    assert all(tool["function"].get("strict") is True for tool in tools)
    assert ordinary_tool["function"]["parameters"]["required"] == ["thought", "tool", "args_json"]
    assert ordinary_tool["function"]["parameters"]["properties"]["args_json"]["type"] == "string"
    assert (
        spawn_tool["function"]["parameters"]["properties"]["tasks"]["items"]["required"]
        == ["slot_id", "task", "slot_local_path_tools", "branch_metadata_json"]
    )
    assert evaluate_tool["function"]["parameters"]["required"] == [
        "thought",
        "tool",
        "task_type",
        "prediction_json",
        "predictions_json",
    ]


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


def test_build_tool_observation_message_marks_task_valid_answer_availability():
    text = ExecutorAgent._build_tool_observation_message(
        step=2,
        tool_name="llm_reasoning_forecast",
        tool_args={"question": "q"},
        success=True,
        observation_payload={"answer": [1.0, 2.0, 3.0]},
        task_type="forecast",
        horizon=3,
    )

    assert "## Answer Status" in text
    assert "already contains a task-valid answer" in text
    assert "may finish by re-wrapping this tool-produced answer" in text


def test_build_state_update_message_uses_current_working_state_header():
    text = ExecutorAgent._build_state_update_message(
        current_series_state={"artifact_type": "series"},
        latest_artifact_state={"producer_tool": "slice"},
    )

    assert text.startswith("## Current Working State\n")
    assert '"current_series_state"' in text
    assert '"latest_artifact_state"' in text


def test_build_state_update_message_includes_task_answer_status_when_available():
    text = ExecutorAgent._build_state_update_message(
        current_series_state={"artifact_type": "series"},
        latest_artifact_state={"producer_tool": "llm_reasoning_forecast"},
        task_answer_status={
            "available": True,
            "task_type": "forecast",
            "source_tool": "llm_reasoning_forecast",
        },
    )

    assert '"task_answer_status"' in text
    assert '"available": true' in text
    assert '"source_tool": "llm_reasoning_forecast"' in text


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


def test_subagent_must_start_with_hinted_tool_when_hint_not_started():
    assert ExecutorAgent._subagent_must_start_with_hinted_tool(
        execution_log=[],
        tools_hint=["detect_trend"],
        next_tool_name="window_stats",
    ) is True
    assert ExecutorAgent._subagent_must_start_with_hinted_tool(
        execution_log=[],
        tools_hint=["detect_trend"],
        next_tool_name="detect_trend",
    ) is False
    assert ExecutorAgent._subagent_must_start_with_hinted_tool(
        execution_log=[{"tool": "detect_trend"}],
        tools_hint=["detect_trend"],
        next_tool_name="window_stats",
    ) is False


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
                "args_json": "{\"start\": 0, \"end\": 24}",
            },
        }],
        model="mock",
    )

    decision = ExecutorAgent._coerce_structured_decision(
        response,
        allow_tool_call=True,
        allow_finish=True,
        ordinary_tool_names=["slice"],
        fallback_thought="parse failed",
    )

    assert decision == {
        "thought": "Reason: need evidence. Action: run slice.",
        "done": False,
        "tool": "slice",
        "args": {"start": 0, "end": 24},
    }


def test_coerce_structured_decision_maps_spawn_response():
    response = LLMResponse(
        text="",
        parsed=None,
        tool_calls=[{
            "name": "emit_spawn_subagent",
            "arguments": {
                "thought": "Reason: compare candidate paths. Action: spawn two branches.",
                "tasks": [
                    {
                        "slot_id": 0,
                        "task": "timesfm branch",
                        "slot_local_path_tools": ["timesfm2_forecast"],
                        "branch_metadata_json": "",
                    },
                    {
                        "slot_id": 1,
                        "task": "sundial branch",
                        "slot_local_path_tools": ["sundial_base_128m_forecast"],
                        "branch_metadata_json": "{\"candidate_family\":\"seasonal\"}",
                    },
                ],
            },
        }],
        model="mock",
    )

    decision = ExecutorAgent._coerce_structured_decision(
        response,
        allow_tool_call=True,
        allow_finish=True,
        spawn_available=True,
        fallback_thought="parse failed",
    )

    assert decision == {
        "thought": "Reason: compare candidate paths. Action: spawn two branches.",
        "done": False,
        "tool": "spawn_subagent",
        "args": {
            "tasks": [
                {"slot_id": 0, "task": "timesfm branch", "tools_hint": ["timesfm2_forecast"]},
                {
                    "slot_id": 1,
                    "task": "sundial branch",
                    "tools_hint": ["sundial_base_128m_forecast"],
                    "branch_metadata": {"candidate_family": "seasonal"},
                },
            ]
        },
    }


def test_coerce_structured_decision_spawn_legacy_tools_hint_still_works():
    response = LLMResponse(
        text="",
        parsed=None,
        tool_calls=[{
            "name": "emit_spawn_subagent",
            "arguments": {
                "thought": "Reason: compare candidate paths. Action: spawn one branch.",
                "tasks": [
                    {
                        "slot_id": 0,
                        "task": "timesfm branch",
                        "tools_hint": "timesfm2_forecast",
                        "branch_metadata_json": "",
                    },
                ],
            },
        }],
        model="mock",
    )

    decision = ExecutorAgent._coerce_structured_decision(
        response,
        allow_tool_call=True,
        allow_finish=True,
        spawn_available=True,
        fallback_thought="parse failed",
    )

    assert decision["args"]["tasks"][0]["tools_hint"] == "timesfm2_forecast"


def test_coerce_structured_decision_maps_evaluate_response():
    response = LLMResponse(
        text="",
        parsed=None,
        tool_calls=[{
            "name": "emit_evaluate_call",
            "arguments": {
                "thought": "Reason: two candidates are ready. Action: evaluate them.",
                "tool": "evaluate_batch_against_gt",
                "task_type": "forecast",
                "prediction_json": "",
                "predictions_json": "[[1.0, 2.0], [2.0, 3.0]]",
            },
        }],
        model="mock",
    )

    decision = ExecutorAgent._coerce_structured_decision(
        response,
        allow_tool_call=True,
        allow_finish=True,
        evaluate_tool_names=["evaluate_batch_against_gt"],
        fallback_thought="parse failed",
    )

    assert decision == {
        "thought": "Reason: two candidates are ready. Action: evaluate them.",
        "done": False,
        "tool": "evaluate_batch_against_gt",
        "args": {
            "task_type": "forecast",
            "predictions": [[1.0, 2.0], [2.0, 3.0]],
        },
    }


def test_coerce_structured_decision_rejects_wrong_tool_family_name():
    response = LLMResponse(
        text="",
        parsed=None,
        tool_calls=[{
            "name": "emit_tool_call",
            "arguments": {
                "thought": "Reason: compare candidates. Action: spawn.",
                "tool": "spawn_subagent",
                "args_json": "{}",
            },
        }],
        model="mock",
    )

    decision = ExecutorAgent._coerce_structured_decision(
        response,
        allow_tool_call=True,
        allow_finish=True,
        ordinary_tool_names=["slice", "value_at"],
        spawn_available=True,
        fallback_thought="parse failed",
    )

    assert decision["_parse_failed"] is True
    assert decision["thought"] == "invalid_decision_wrong_tool_family"


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


def test_coerce_structured_decision_maps_learning_summary_finish_response():
    response = LLMResponse(
        text="",
        parsed=None,
        tool_calls=[{
            "name": "emit_final_answer",
            "arguments": {
                "thought": "Reason: enough evidence. Action: finish with learning summary.",
                "insight": "Chronos-2 branch won on the batch evaluation.",
                "recommendation": "Prefer regime-aware forecast paths when the recent segment breaks from older structure.",
                "confidence": 0.93,
                "reasoning": "1. Three candidates were compared.\n2. Chronos-2 had the lowest error.",
            },
        }],
        model="mock",
    )

    decision = ExecutorAgent._coerce_structured_decision(
        response,
        allow_tool_call=True,
        allow_finish=True,
        fallback_thought="parse failed",
        finish_mode="learning_summary",
    )

    assert decision == {
        "thought": "Reason: enough evidence. Action: finish with learning summary.",
        "done": True,
        "answer": {
            "insight": "Chronos-2 branch won on the batch evaluation.",
            "recommendation": "Prefer regime-aware forecast paths when the recent segment breaks from older structure.",
        },
        "answer_type": "learning_summary",
        "confidence": 0.93,
        "reasoning": "1. Three candidates were compared.\n2. Chronos-2 had the lowest error.",
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


def test_coerce_structured_decision_preserves_legacy_task_finish_in_learning_summary_mode():
    response = LLMResponse(
        text=(
            '{"thought":"Reason: task answer already exists. Action: finish.","done":true,'
            '"answer":{"macd":[0.1,0.2]},"answer_type":"indicator","confidence":0.9,'
            '"reasoning":"1. A tool already produced this indicator.\\n2. I am returning it unchanged."}'
        ),
        parsed={
            "thought": "Reason: task answer already exists. Action: finish.",
            "done": True,
            "answer": {"macd": [0.1, 0.2]},
            "answer_type": "indicator",
            "confidence": 0.9,
            "reasoning": (
                "1. A tool already produced this indicator.\n"
                "2. I am returning it unchanged."
            ),
        },
        tool_calls=[],
        model="mock",
    )

    decision = ExecutorAgent._coerce_structured_decision(
        response,
        allow_tool_call=True,
        allow_finish=True,
        fallback_thought="parse failed",
        finish_mode="learning_summary",
    )

    assert decision == {
        "thought": "Reason: task answer already exists. Action: finish.",
        "done": True,
        "answer": {"macd": [0.1, 0.2]},
        "answer_type": "indicator",
        "confidence": 0.9,
        "reasoning": (
            "1. A tool already produced this indicator.\n"
            "2. I am returning it unchanged."
        ),
    }


def test_main_output_contract_uses_current_turn_contract_when_finish_not_allowed():
    section = ExecutorAgent._build_main_output_contract_section(
        task_type="forecast",
        horizon=24,
        is_train=True,
        defer_learning_record=False,
        indicator_output_keys=[],
        allow_finish_current_turn=False,
    )

    assert "This is not a completion turn" in section
    assert "Return only the current-turn tool-call JSON" in section
    assert "Required keys inside `answer`: `insight`, `recommendation`." not in section


def test_main_output_contract_for_classification_requires_label_not_forecast_array():
    section = ExecutorAgent._build_main_output_contract_section(
        task_type="classification",
        horizon=None,
        is_train=False,
        defer_learning_record=False,
        indicator_output_keys=[],
        allow_finish_current_turn=True,
    )

    assert "Return a single label string" in section
    assert "Predicted Temperatures" not in section


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
    monkeypatch.setattr(ExecutorAgent, "_count_training_spawn_candidates", lambda *_args, **_kwargs: 2)
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
async def test_spawn_subagents_rejects_partially_supported_explicit_hint(monkeypatch):
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

    assert seen_hints == []
    assert result["n_success"] == 0
    assert result["results"] == []
    assert "not fully supported by the chosen slot_id=0" in result["observation"]


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
    assert "not fully supported by the chosen slot_id=0" in error
    assert "Recommended supported slot-local path(s) for slot_id=0" in error


def test_normalize_spawn_tasks_rejects_partially_supported_explicit_hint():
    normalized, error = ExecutorAgent._normalize_spawn_tasks(
        [
            {
                "slot_id": 0,
                "task": "forecast-first classification",
                "tools_hint": "ets_forecast -> llm_reasoning_classification",
            }
        ],
        plan={
            "task_type": "classification",
            "subagent_slots": [
                {
                    "slot_id": 0,
                    "visible_candidates": ["llm_reasoning_classification", "text_to_event"],
                }
            ],
        },
    )

    assert normalized == []
    assert error is not None
    assert "not fully supported" in error
    assert "ets_forecast" in error


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


def test_select_train_spawn_prior_skips_task_mismatched_reasoning_path():
    prior = ExecutorAgent._select_train_spawn_prior(
        path_priors=[
            {"path": ["slice", "llm_reasoning_qa"], "support": 9},
            {"path": ["detect_trend", "llm_reasoning_classification"], "support": 5},
        ],
        subagent_slots=[
            {
                "slot_id": 0,
                "visible_candidates": [
                    "slice",
                    "detect_trend",
                    "llm_reasoning_qa",
                    "llm_reasoning_classification",
                ],
            }
        ],
        task_type="classification",
    )

    assert prior == ["detect_trend", "llm_reasoning_classification"]


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
    monkeypatch.setattr(ExecutorAgent, "_count_training_spawn_candidates", lambda *_args, **_kwargs: 2)
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


def test_build_decision_section_marks_eval_only_turns():
    section = ExecutorAgent._build_decision_section(
        allow_finish=False,
        spawn_available=False,
        eval_only=True,
    )

    assert "Call exactly one GT evaluation tool" in section
    assert "evaluation-only turn" in section


def test_build_decision_section_marks_spawn_only_turns():
    section = ExecutorAgent._build_decision_section(
        allow_finish=False,
        spawn_available=True,
        spawn_only=True,
    )

    assert "Do not call ordinary tools in this turn" in section
    assert "spawn-only turn" in section


def test_build_decision_section_marks_finish_only_turns():
    section = ExecutorAgent._build_decision_section(
        allow_finish=True,
        spawn_available=False,
        tool_call_allowed=False,
    )

    assert "Do not call any more tools in this turn" in section
    assert "finish-only turn" in section


@pytest.mark.asyncio
async def test_executor_overrides_llm_reasoning_domain_with_plan_domain(monkeypatch):
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
                        "thought": "Reason: produce a candidate. Action: call indicator reasoning.",
                        "tool": "llm_reasoning_indicator",
                        "args": {
                            "question": "Return exactly one JSON object with key 'macd'.",
                            "context": "test",
                            "domain": "finance_aligned_long_indicator_macd",
                        },
                    }
                )
            return _Resp(
                {
                    "done": True,
                    "thought": "Reason: step limit reached. Action: summarize failure.",
                    "answer": {
                        "insight": "No valid training completion in this synthetic test.",
                        "recommendation": "Stop after verifying canonical domain injection.",
                    },
                    "answer_type": "learning_summary",
                    "confidence": 0.6,
                    "reasoning": "1. Verified the outgoing tool args.",
                }
            )

        async def complete_with_tools(self, *args, **kwargs):
            return await _complete_with_tools_via_complete(self, *args, **kwargs)

    captured_args: list[dict[str, Any]] = []

    async def _fake_execute_tool(tool_name, tool_args, *_args, **_kwargs):
        captured_args.append({"tool": tool_name, "args": dict(tool_args)})
        return ToolResult(
            success=True,
            data={"answer": {"macd": [0.1, 0.2, 0.3]}},
            error=None,
            metadata={},
        )

    executor = ExecutorAgent(llm=_LLM())
    monkeypatch.setattr(executor, "_execute_tool", _fake_execute_tool)
    monkeypatch.setattr(
        ExecutorAgent,
        "_training_completion_contract_satisfied",
        classmethod(lambda *_args, **_kwargs: True),
    )
    monkeypatch.setattr(executor, "_has_stepwise_reasoning", lambda *_args, **_kwargs: True)

    plan = {
        "candidates": ["llm_reasoning_indicator"],
        "task_type": "indicator",
        "task_description": "test indicator task",
        "tool_names": "llm_reasoning_indicator",
        "tool_notes_section": "",
        "skills_section": "",
        "system_memory": "",
        "modality_hint": "",
        "profiling_section": "",
        "subagent_slot_section": "",
        "subagent_slots": [],
        "fingerprint": {},
        "horizon": 3,
        "domain": "finance",
        "reanalysis_section": "",
        "_mode": "train",
        "_defer_learning_record": False,
        "_ground_truth": {"macd": [0.1, 0.2, 0.3]},
        "indicator_output_keys": ["macd"],
        "modality": "multimodal",
    }
    envelope = TaskEnvelope(
        task=TaskSpec(task_id="test", task_type="indicator", description="test"),
        environment=EnvironmentSnapshot(data=[1.0], modalities=["timeseries", "text"]),
    )

    result = await executor.act(plan, envelope)

    assert result["answer_type"] == "learning_summary"
    assert captured_args[0]["tool"] == "llm_reasoning_indicator"
    assert captured_args[0]["args"]["domain"] == "finance"
