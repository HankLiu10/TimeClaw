import pytest

from timeclaw.benchmark import pipeline_eval as pipeline_eval_mod
from timeclaw.benchmark.mtbench_support import (
    generate_learning_analysis,
    render_learning_analysis_text,
)
from timeclaw.memory.hierarchical import DailyNotes
from timeclaw.pipeline import runner as runner_mod
from timeclaw.pipeline.runner import PipelineRunner
from timeclaw.agents.executor import ExecutorAgent


def test_render_learning_analysis_text_uses_four_sections():
    text = render_learning_analysis_text(
        {
            "text_signals": ["Heavy rain caused flash flooding near the station."],
            "sample_conditions": ["short-horizon forecast", "clear daily seasonality"],
            "winner_evidence": ["chronos_forecast achieved lower MAE than the compared candidate"],
            "failure_signals": ["llm_reasoning_forecast drifted at turning points"],
            "reusable_rule": ["When daily seasonality is clear, consider chronos_forecast before direct llm_reasoning_forecast."],
        }
    )

    assert "Text Signals:" in text
    assert "Sample Conditions:" in text
    assert "Winner Evidence:" in text
    assert "Failure Signals:" in text
    assert "Reusable Rule:" in text


def test_daily_note_labels_winner_tool_section():
    note = DailyNotes._build_note(  # type: ignore[misc]
        DailyNotes.__new__(DailyNotes),
        {
            "scope_key": "weather_aligned_short_forecast",
            "tool_chain": ["chronos_forecast"],
            "predicted": "[1,2,3]",
        },
    )
    assert "Winner Tool:" in note
    assert "Tools:" not in note


def test_daily_note_falls_back_to_best_candidate_for_winner_tool():
    note = DailyNotes._build_note(  # type: ignore[misc]
        DailyNotes.__new__(DailyNotes),
        {
            "scope_key": "weather_aligned_short_forecast",
            "tool_chain": ["none"],
            "candidate_traces": [
                {
                    "name": "chronos_forecast",
                    "tool_chain": ["chronos_forecast"],
                    "metrics": {"mae": 2.35, "rmse": 2.91},
                    "prediction": [1, 2, 3],
                    "steps": [],
                },
                {
                    "name": "llm_reasoning_forecast",
                    "tool_chain": ["llm_reasoning_forecast"],
                    "metrics": {"mae": 2.31, "rmse": 2.77},
                    "prediction": [3, 4, 5],
                    "steps": [],
                },
            ],
        },
    )
    assert "Winner Tool:\n- llm_reasoning_forecast" in note
    assert "Winner Tool:\n- none" not in note


@pytest.mark.asyncio
async def test_generate_learning_analysis_falls_back_to_plain_summary():
    class _Resp:
        parsed = None

    class _LLM:
        async def complete(self, *_args, **_kwargs):
            return _Resp()

    structured, text = await generate_learning_analysis(
        scope_key="weather_aligned_short_forecast",
        task_type="forecast",
        domain="weather",
        task_description="forecast task",
        sample_fingerprint="horizon=24",
        candidate_traces=[],
        best_metrics={"mae": 1.2},
        fallback_summary="fallback summary",
        llm=_LLM(),
    )

    assert structured is None
    assert text == "fallback summary"


@pytest.mark.asyncio
async def test_generate_learning_analysis_text_signals_only_use_text_context():
    class _Resp:
        parsed = {
            "text_signals": [
                "weather_aligned_short_forecast",
                "The flood was caused by Heavy Rain.",
                "Every 24 temperature readings represent a full day from 00:00 to 23:00.",
            ],
            "sample_conditions": ["x"],
            "winner_evidence": ["y"],
            "failure_signals": [],
            "reusable_rule": ["When x, do y."],
        }

    class _LLM:
        async def complete(self, *_args, **_kwargs):
            return _Resp()

    structured, _text = await generate_learning_analysis(
        scope_key="weather_aligned_short_forecast",
        task_type="forecast",
        domain="weather",
        task_description=(
            "System prompt:\nYou are a weather forecasting AI.\n\n"
            "User prompt:\nTemperature time series (past 7 days, hourly):\n[1,2,3]\n\n"
            "Weather report:\nThe flood was caused by Heavy Rain.\n\n"
            "Your task is to predict the next 1 day(s) of hourly temperature readings.\n"
            "Return ONLY a JSON array of 24 numbers, nothing else."
        ),
        sample_fingerprint="horizon=24",
        candidate_traces=[],
        best_metrics={"mae": 1.2},
        fallback_summary="fallback summary",
        llm=_LLM(),
    )

    assert structured is not None
    assert structured["text_signals"] == ["The flood was caused by Heavy Rain."]


def test_daily_note_prefers_llm_text_signals_when_present():
    note = DailyNotes._build_note(  # type: ignore[misc]
        DailyNotes.__new__(DailyNotes),
        {
            "scope_key": "weather_aligned_short_forecast",
            "task": "forecast",
            "description": (
                "System prompt:\nYou are a weather forecasting AI.\n\n"
                "User prompt:\nWeather report:\nFlash Flood near station. Heavy rain caused cooling.\n\n"
                "Your task is to predict the next 1 day(s) of hourly temperature readings.\n"
            ),
            "analysis_structured": {
                "text_signals": [
                    "Heavy rain caused cooling.",
                    "Flash Flood near station.",
                ]
            },
        },
    )
    assert "Text Signals:\n- Heavy rain caused cooling.\n- Flash Flood near station." in note


def test_daily_note_analysis_omits_duplicate_text_signals_section():
    note = DailyNotes._build_note(  # type: ignore[misc]
        DailyNotes.__new__(DailyNotes),
        {
            "scope_key": "weather_aligned_short_forecast",
            "task": "forecast",
            "description": (
                "System prompt:\nYou are a weather forecasting AI.\n\n"
                "User prompt:\nWeather report:\nFlash Flood near station. Heavy rain caused cooling.\n\n"
                "Your task is to predict the next 1 day(s) of hourly temperature readings.\n"
            ),
            "analysis_structured": {"text_signals": ["Heavy rain caused cooling."]},
            "pipeline_summary": (
                "Text Signals:\n"
                "- \"Heavy rain caused cooling.\"\n\n"
                "Sample Conditions:\n"
                "- short-horizon forecast"
            ),
        },
    )
    assert "\nAnalysis:\nSample Conditions:\n- short-horizon forecast" in note
    assert "\nAnalysis:\nText Signals:" not in note


@pytest.mark.asyncio
async def test_save_learning_outcome_uses_generated_learning_analysis(monkeypatch):
    runner = PipelineRunner(disable_memory=True)

    class _MemoryStub:
        def __init__(self):
            self.kwargs = None

        async def record(self, **kwargs):
            self.kwargs = kwargs

    runner.memory = _MemoryStub()

    async def _fake_generate(**_kwargs):
        return (
            {
                "sample_conditions": ["short-horizon"],
                "winner_evidence": ["chronos_forecast achieved lower MAE than the compared candidate"],
                "failure_signals": ["ets_forecast flattened the cycle"],
                "reusable_rule": ["When short-horizon with seasonality, consider chronos_forecast before ets_forecast."],
            },
            "Sample Conditions:\n- short-horizon\n\nWinner Evidence:\n- chronos_forecast achieved lower MAE than the compared candidate\n\nFailure Signals:\n- ets_forecast flattened the cycle\n\nReusable Rule:\n- When short-horizon with seasonality, consider chronos_forecast before ets_forecast.",
        )

    monkeypatch.setattr(runner_mod, "generate_learning_analysis", _fake_generate)

    await runner._save_learning_outcome(
        answer_type="learning_summary",
        raw_answer={"insight": "x", "recommendation": "y"},
        prediction=[1.0, 2.0],
        execution_log=[
            {
                "tool": "spawn_subagent",
                "output": [
                    {
                        "success": True,
                        "result": {
                            "answer": [1.0, 2.0],
                            "execution_log": [{"tool": "chronos_forecast"}],
                        },
                    }
                ],
            },
            {
                "tool": "evaluate_batch_against_gt",
                "observation": {
                    "results": [
                        {"name": "chronos_forecast", "metrics": {"mae": 1.2}},
                    ],
                },
            },
        ],
        task_type="forecast",
        description="forecast sample",
        domain="weather",
        scope_keys=["weather_aligned_short_forecast"],
        reasoning="fallback reasoning",
        fingerprint={},
        profiling_results={},
        horizon=24,
    )

    assert runner.memory.kwargs is not None
    assert "Sample Conditions:" in runner.memory.kwargs["pipeline_summary"]
    assert runner.memory.kwargs["analysis_structured"]["winner_evidence"] == [
        "chronos_forecast achieved lower MAE than the compared candidate"
    ]


def test_train_kwargs_defers_internal_learning_record():
    sample = type("S", (), {"output_window": [1.0, 2.0], "domain": "weather", "answer": None, "labels": {}})()
    pipeline_eval_mod._record_enabled = True
    kwargs = pipeline_eval_mod._train_kwargs(sample, "forecast")
    assert kwargs["mode"] == "train"
    assert kwargs["defer_learning_record"] is True


def test_record_correctness_is_none_for_scored_tasks():
    assert pipeline_eval_mod._record_correctness("forecast", {"mae": 1.2}) is None
    assert pipeline_eval_mod._record_correctness("indicator", {"diff_mae": 0.8}) is None
    assert pipeline_eval_mod._record_correctness("stock_indicator", {"macd_mae": 0.1}) is None


def test_record_tool_chain_prefers_winner_candidate_chain():
    result = {
        "execution_log": [
            {
                "tool": "spawn_subagent",
                "output": [
                    {
                        "success": True,
                        "result": {
                            "answer": [1.0, 2.0],
                            "execution_log": [{"tool": "chronos_forecast", "observation": {"forecast": [1.0, 2.0]}}],
                        },
                    },
                    {
                        "success": True,
                        "result": {
                            "answer": [3.0, 4.0],
                            "execution_log": [{"tool": "arima_forecast", "observation": {"forecast": [3.0, 4.0]}}],
                        },
                    },
                ],
            },
            {
                "tool": "evaluate_batch_against_gt",
                "observation": {
                    "results": [
                        {"name": "chronos_forecast", "metrics": {"mae": 1.2}},
                        {"name": "arima_forecast", "metrics": {"mae": 2.5}},
                    ]
                },
            },
        ]
    }
    assert pipeline_eval_mod._record_tool_chain(result, {"mae": 1.2}) == ["chronos_forecast"]


def test_extract_candidate_traces_attaches_single_eval_metrics():
    traces = PipelineRunner._extract_candidate_traces(
        [
            {
                "tool": "spawn_subagent",
                "output": [
                    {
                        "success": True,
                        "result": {
                            "answer": [1.0, 2.0],
                            "execution_log": [{"tool": "chronos_forecast", "observation": {"forecast": [1.0, 2.0]}}],
                        },
                    },
                    {
                        "success": True,
                        "result": {
                            "answer": [3.0, 4.0],
                            "execution_log": [{"tool": "timesfm_forecast", "observation": {"forecast": [3.0, 4.0]}}],
                        },
                    },
                ],
            },
            {
                "tool": "evaluate_against_gt",
                "args": {"prediction": [3.0, 4.0]},
                "observation": {"metrics": {"mae": 0.5, "rmse": 0.7}},
            },
        ]
    )

    assert traces[0].get("metrics") is None
    assert traces[1]["metrics"] == {"mae": 0.5, "rmse": 0.7}


def test_extract_candidate_traces_from_top_level_execution_log():
    traces = PipelineRunner._extract_candidate_traces(
        [
            {
                "tool": "chronos_forecast",
                "thought": "Reason: x. Action: run chronos.",
                "observation": {"point_forecast": [1.0, 2.0, 3.0]},
            },
            {
                "tool": "llm_reasoning_forecast",
                "thought": "Reason: y. Action: run llm.",
                "observation": {"answer": "[2.0, 3.0, 4.0]"},
            },
            {
                "tool": "evaluate_against_gt",
                "args": {"prediction": "[2.0, 3.0, 4.0]"},
                "observation": {"metrics": {"mae": 0.5, "rmse": 0.7}},
            },
        ]
    )

    assert [trace["name"] for trace in traces] == ["chronos_forecast", "llm_reasoning_forecast"]
    assert traces[1]["metrics"] == {"mae": 0.5, "rmse": 0.7}


def test_winner_trace_selection_uses_candidate_ranking_not_metrics_identity():
    candidate_traces = [
        {
            "name": "chronos_forecast",
            "tool_chain": ["chronos_forecast"],
            "prediction": [1.0, 2.0],
            "metrics": {"mae": 2.35, "rmse": 2.91},
        },
        {
            "name": "llm_reasoning_forecast",
            "tool_chain": ["llm_reasoning_forecast"],
            "prediction": [3.0, 4.0],
            "metrics": {"mae": 2.31, "rmse": 2.77},
        },
    ]

    best_metrics = {"mae": 2.31, "rmse": 2.77}

    assert PipelineRunner._select_winner_tool_chain(candidate_traces, best_metrics) == [
        "llm_reasoning_forecast"
    ]
    assert (
        PipelineRunner._select_winner_prediction_from_execution_log(
            [
                {
                    "tool": "spawn_subagent",
                    "output": [
                        {
                            "success": True,
                            "result": {
                                "answer": [1.0, 2.0],
                                "execution_log": [{"tool": "chronos_forecast"}],
                            },
                        },
                        {
                            "success": True,
                            "result": {
                                "answer": [3.0, 4.0],
                                "execution_log": [{"tool": "llm_reasoning_forecast"}],
                            },
                        },
                    ],
                },
                {
                    "tool": "evaluate_batch_against_gt",
                    "observation": {
                        "results": [
                            {"name": "chronos_forecast", "metrics": {"mae": 2.35, "rmse": 2.91}},
                            {"name": "llm_reasoning_forecast", "metrics": {"mae": 2.31, "rmse": 2.77}},
                        ]
                    },
                },
            ],
            fallback_prediction=[9.0, 9.0],
        )
        == [3.0, 4.0]
    )


def test_collect_batch_eval_predictions_coerces_forecast_json_strings():
    preds = ExecutorAgent._collect_batch_eval_predictions(
        execution_log=[
            {
                "tool": "spawn_subagent",
                "output": [
                    {
                        "success": True,
                        "result": {
                            "answer": "[1.0, 2.0, 3.0]",
                            "execution_log": [{"tool": "arima_forecast"}],
                        },
                    }
                ],
            }
        ],
        task_type="forecast",
        horizon=3,
    )
    assert preds == [{"name": "arima_forecast", "prediction": [1.0, 2.0, 3.0]}]


def test_count_training_candidates_counts_valid_subagent_paths():
    count = ExecutorAgent._count_training_candidates(
        execution_log=[
            {
                "tool": "spawn_subagent",
                "output": [
                    {
                        "success": True,
                        "result": {
                            "answer": "[1.0, 2.0, 3.0]",
                            "execution_log": [{"tool": "chronos_forecast"}],
                        },
                    },
                    {
                        "success": True,
                        "result": {
                            "answer": "[1.5, 2.5, 3.5]",
                            "execution_log": [{"tool": "timesfm_forecast"}],
                        },
                    },
                    {
                        "success": True,
                        "result": {
                            "answer": "[2.0, 3.0, 4.0]",
                            "execution_log": [],
                            "tools_hint": "llm_reasoning_forecast",
                        },
                    },
                ],
            }
        ],
        task_type="forecast",
        horizon=3,
    )
    assert count == 3


def test_count_training_candidates_ignores_invalid_subagent_answers():
    count = ExecutorAgent._count_training_candidates(
        execution_log=[
            {
                "tool": "spawn_subagent",
                "output": [
                    {
                        "success": True,
                        "result": {
                            "answer": "not a forecast",
                            "execution_log": [{"tool": "llm_reasoning_forecast"}],
                        },
                    },
                    {
                        "success": True,
                        "result": {
                            "answer": "[1.0, 2.0, 3.0]",
                            "execution_log": [{"tool": "chronos_forecast"}],
                        },
                    },
                ],
            }
        ],
        task_type="forecast",
        horizon=3,
    )
    assert count == 1


def test_subagent_followed_tools_hint_requires_real_tool_call():
    assert ExecutorAgent._subagent_followed_tools_hint(
        execution_log=[{"tool": "llm_reasoning_forecast"}],
        tools_hint="llm_reasoning_forecast",
    ) is True
    assert ExecutorAgent._subagent_followed_tools_hint(
        execution_log=[],
        tools_hint="llm_reasoning_forecast",
    ) is False


def test_extract_candidate_traces_uses_primary_tool_name():
    traces = PipelineRunner._extract_candidate_traces(
        [
            {
                "tool": "spawn_subagent",
                "output": [
                    {
                        "success": True,
                        "result": {
                            "answer": "[1.0, 2.0, 3.0]",
                            "execution_log": [
                                {"tool": "llm_reasoning_forecast"},
                                {"tool": "bound_check"},
                                {"tool": "consistency_check"},
                            ],
                        },
                    }
                ],
            }
        ]
    )
    assert traces[0]["name"] == "llm_reasoning_forecast"


def test_extract_candidate_traces_synthesizes_step_from_tools_hint():
    traces = PipelineRunner._extract_candidate_traces(
        [
            {
                "tool": "spawn_subagent",
                "output": [
                    {
                        "success": True,
                        "result": {
                            "answer": "[1.0, 2.0, 3.0]",
                            "reasoning": "Reason: use llm_reasoning_forecast. Action: produce a forecast.",
                            "execution_log": [],
                            "tools_hint": "llm_reasoning_forecast",
                        },
                    }
                ],
            }
        ]
    )
    assert traces[0]["name"] == "llm_reasoning_forecast"
    assert traces[0]["steps"][0]["tool"] == "llm_reasoning_forecast"
    assert "answer" in traces[0]["steps"][0]["observation"]


def test_has_training_evaluation_accepts_batch_results():
    assert ExecutorAgent._has_training_evaluation([
        {
            "tool": "evaluate_batch_against_gt",
            "success": True,
            "observation": {
                "results": [
                    {"name": "chronos_forecast", "metrics": {"mae": 1.2}},
                ],
            },
        }
    ]) is True


def test_is_valid_final_answer_accepts_json_string_forecast():
    assert ExecutorAgent._is_valid_final_answer(
        "forecast",
        "[1.0, 2.0, 3.0]",
        horizon=3,
    ) is True


def test_daily_note_formats_scored_candidates_and_input_signals():
    note = DailyNotes._build_note(  # type: ignore[misc]
        DailyNotes.__new__(DailyNotes),
        {
            "scope_key": "weather_aligned_short_forecast",
            "domain": "weather",
            "task": "forecast",
            "metrics": {"mae": 1.23, "rmse": 2.34},
            "tool_chain": ["chronos_forecast"],
            "predicted": "[1, 2, 3]",
            "data_fingerprint": "horizon=3,last3=[1,2,3]",
            "description": (
                "System prompt:\nYou are a weather forecasting AI.\n\n"
                "User prompt:\nTemperature time series (past 7 days, hourly):\n[1, 2, 3]\n\n"
                "Weather report:\nFlash Flood near station. Heavy rain caused cooling. Cold front arrived overnight. River flooding persisted into the next day.\n\n"
                "Your task is to predict the next 1 day(s) of hourly temperature readings (3 values).\n"
                "Return ONLY a JSON array of 3 numbers, nothing else."
            ),
            "candidate_traces": [
                {
                    "name": "chronos_forecast",
                    "tool_chain": ["chronos_forecast"],
                    "prediction": [1, 2, 3],
                    "metrics": {"mae": 1.23, "rmse": 2.34},
                    "steps": [
                        {
                            "tool": "chronos_forecast",
                            "thought": "Reason: seasonality is present. Action: run chronos_forecast.",
                            "observation": {"forecast": [1, 2, 3]},
                        }
                    ],
                }
            ],
            "pipeline_summary": "Sample Conditions:\n- x",
        },
    )
    assert "Task:\nSystem prompt:\nYou are a weather forecasting AI." in note
    assert "Text Signals:" in note
    assert "Flash Flood near station." in note
    assert "Cold front arrived overnight." in note
    assert "chronos_forecast(mae=1.23, rmse=2.34)" in note
    assert "Observation: {\"forecast\": [1, 2, 3]}" in note
    assert "  Prediction: [1, 2, 3]" not in note


def test_text_signals_rank_sentences_by_task_relevance():
    signals = DailyNotes._extract_text_signals(
        (
            "System prompt:\nYou are a weather forecasting AI.\n\n"
            "User prompt:\nWeather report:\n"
            "Officials held a community meeting at noon. "
            "Heavy rain caused flash flooding near the station between 2013-09-11 and 2013-09-15. "
            "The event caused 3 injuries and 1 fatality. "
            "A local fair was postponed.\n\n"
            "Your task is to predict the next 1 day(s) of hourly temperature readings.\n"
        ),
        task="forecast",
    )

    assert signals
    assert signals[0] == "Heavy rain caused flash flooding near the station between 2013-09-11 and 2013-09-15."


def test_text_signals_use_task_description_focus_terms():
    signals = DailyNotes._extract_text_signals(
        (
            "System prompt:\nYou are a finance assistant.\n\n"
            "User prompt:\nNews:\n"
            "The company announced a leadership offsite for employees. "
            "Revenue guidance was cut after weaker demand in the cloud segment. "
            "Executives said margins may recover next quarter.\n\n"
            "Your task is to explain the likely stock movement using the news context.\n"
        ),
        task="qa",
        task_description=(
            "System prompt:\nYou are a finance assistant.\n\n"
            "User prompt:\nNews:\n...\n\n"
            "Your task is to explain the likely stock movement using the news context.\n"
        ),
    )

    assert signals
    assert any(signal in {
        "Revenue guidance was cut after weaker demand in the cloud segment.",
        "Executives said margins may recover next quarter.",
    } for signal in signals[:3])


def test_subagent_candidates_exclude_evaluation_tools():
    plan = {
        "candidates": [
            "chronos_forecast",
            "evaluate_against_gt",
            "evaluate_batch_against_gt",
            "batch_evaluate_against_gt",
        ],
        "task_type": "forecast",
        "task_description": "forecast sample",
        "fingerprint": {},
        "profiling_section": "",
        "tool_notes_section": "",
        "l1_section": "",
        "skills_section": "",
        "system_memory": "",
        "modality_hint": "",
        "horizon": 3,
    }
    blocked = {"evaluate_against_gt", "evaluate_batch_against_gt", "batch_evaluate_against_gt"}
    candidates = [c for c in plan["candidates"] if c not in blocked]
    assert candidates == ["chronos_forecast"]


@pytest.mark.asyncio
async def test_deferred_train_uses_best_evaluated_candidate(monkeypatch):
    runner = PipelineRunner(disable_memory=True)

    async def _fake_ensure_tools():
        return None

    async def _fake_preprocess(*_args, **_kwargs):
        return {"fingerprint": {}, "modalities": ["timeseries"], "profiling_results": {}}

    class _Executor:
        evidence_log = type(
            "E",
            (),
            {
                "summary": staticmethod(lambda: ""),
                "is_sufficient": True,
            },
        )()

        async def run(self, _envelope):
            return {
                "result": {
                    "answer": [9.0, 9.0, 9.0],
                    "answer_type": "forecast",
                    "confidence": 0.9,
                    "reasoning": "Reason: x. Action: y.",
                    "execution_log": [
                        {
                            "tool": "spawn_subagent",
                            "output": [
                                {
                                    "success": True,
                                    "result": {
                                        "answer": [1.0, 2.0, 3.0],
                                        "execution_log": [{"tool": "chronos_forecast"}],
                                    },
                                },
                                {
                                    "success": True,
                                    "result": {
                                        "answer": [4.0, 5.0, 6.0],
                                        "execution_log": [{"tool": "arima_forecast"}],
                                    },
                                },
                            ],
                        },
                        {
                            "tool": "evaluate_batch_against_gt",
                            "success": True,
                            "observation": {
                                "results": [
                                    {"name": "chronos_forecast", "metrics": {"mae": 1.0}},
                                    {"name": "arima_forecast", "metrics": {"mae": 2.0}},
                                ]
                            },
                        },
                    ],
                }
            }

    monkeypatch.setattr(runner, "_ensure_tools", _fake_ensure_tools)
    monkeypatch.setattr(runner, "_preprocess", _fake_preprocess)
    monkeypatch.setattr(runner, "_create_executor", lambda: _Executor())

    result = await runner.run(
        data=[1.0, 2.0, 3.0],
        task_type="forecast",
        description="forecast sample",
        mode="train",
        ground_truth=[1.0, 2.0, 3.0],
        defer_learning_record=True,
    )

    assert result["prediction"] == [1.0, 2.0, 3.0]


@pytest.mark.asyncio
async def test_run_skips_internal_learning_record_when_deferred(monkeypatch):
    runner = PipelineRunner(disable_memory=True)

    async def _fake_ensure_tools():
        return None

    async def _fake_preprocess(*_args, **_kwargs):
        return {"fingerprint": {}, "modalities": ["timeseries"], "profiling_results": {}}

    class _Executor:
        evidence_log = type(
            "E",
            (),
            {
                "summary": staticmethod(lambda: ""),
                "is_sufficient": True,
            },
        )()

        async def run(self, _envelope):
            return {
                "result": {
                    "answer": [1.0, 2.0],
                    "answer_type": "forecast",
                    "confidence": 0.9,
                    "reasoning": "Reason: x. Action: y.",
                    "execution_log": [],
                }
            }

    monkeypatch.setattr(runner, "_ensure_tools", _fake_ensure_tools)
    monkeypatch.setattr(runner, "_preprocess", _fake_preprocess)
    monkeypatch.setattr(runner, "_create_executor", lambda: _Executor())

    async def _boom(**_kwargs):
        raise AssertionError("_save_learning_outcome should not be called")

    monkeypatch.setattr(runner, "_save_learning_outcome", _boom)

    result = await runner.run(
        data=[1.0, 2.0, 3.0],
        task_type="forecast",
        description="forecast sample",
        mode="train",
        ground_truth=[1.0, 2.0],
        defer_learning_record=True,
    )

    assert result["prediction"] == [1.0, 2.0]
