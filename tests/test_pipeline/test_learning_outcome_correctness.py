import pytest
import numpy as np

from timeclaw.agents.executor import ExecutorAgent
from timeclaw.memory.hierarchical import Notes
from timeclaw.pipeline.runner import PipelineRunner


def test_learning_outcome_correct_uses_explicit_boolean_for_labels():
    assert PipelineRunner._learning_outcome_correctness(
        {"accuracy": 1.0, "correct": True, "predicted_label": "A", "ground_truth_label": "A"}
    ) is True
    assert PipelineRunner._learning_outcome_correctness(
        {"accuracy": 0.0, "correct": False, "predicted_label": "B", "ground_truth_label": "A"}
    ) is False


def test_learning_outcome_correct_returns_none_for_continuous_metrics():
    assert PipelineRunner._learning_outcome_correctness(
        {"mae": 2.3, "rmse": 3.1, "mse": 9.6}
    ) is None
    assert PipelineRunner._learning_outcome_correctness(
        {"avg_mae": 1.2, "max_mae": 1.0, "min_mae": 1.4}
    ) is None


def test_learning_outcome_correct_rejects_parse_and_type_errors():
    assert PipelineRunner._learning_outcome_correctness({"parse_error": 1.0}) is False
    assert PipelineRunner._learning_outcome_correctness({"type_error": 1.0}) is False
    assert PipelineRunner._learning_outcome_correctness(
        {"max_error": 1.0, "avg_mae": 0.1}
    ) is False


def test_daily_note_outcome_label_is_mtbench_aligned():
    assert Notes._outcome_label(True, {"accuracy": 1.0}) == "OK"
    assert Notes._outcome_label(False, {"accuracy": 0.0}) == "FAIL"
    assert Notes._outcome_label(None, {"mae": 2.0, "rmse": 3.0}) == "SCORED"
    assert Notes._outcome_label(None, {"parse_error": 1.0}) == "FAIL"


@pytest.mark.asyncio
async def test_save_learning_outcome_records_task_native_prediction():
    runner = PipelineRunner(disable_memory=True)

    class _MemoryStub:
        def __init__(self):
            self.kwargs = None

        async def record(self, **kwargs):
            self.kwargs = kwargs

    runner.memory = _MemoryStub()

    await runner._save_learning_outcome(
        answer_type="learning_summary",
        raw_answer={
            "insight": "best approach was direct reasoning",
            "recommendation": "prefer numeric output",
        },
        prediction={"max": 24.8, "min": 12.9, "diff": 11.9},
        execution_log=[
            {
                "tool": "evaluate_batch_against_gt",
                "observation": {"metrics": {"avg_mae": 1.2}},
            }
        ],
        task_type="indicator",
        description="Return JSON indicator output",
        domain="weather",
        scope_keys=["weather_aligned_long_indicator"],
        reasoning="",
        fingerprint={},
        profiling_results={},
        horizon=None,
        source_metadata={},
        trace_id="trace-indicator-1",
    )

    assert runner.memory.kwargs is not None
    assert runner.memory.kwargs["predicted"] == "{'max': 24.8, 'min': 12.9, 'diff': 11.9}"


@pytest.mark.asyncio
async def test_save_learning_outcome_prefers_winner_candidate_tool_chain():
    runner = PipelineRunner(disable_memory=True)

    class _MemoryStub:
        def __init__(self):
            self.kwargs = None

        async def record(self, **kwargs):
            self.kwargs = kwargs

    runner.memory = _MemoryStub()

    await runner._save_learning_outcome(
        answer_type="learning_summary",
        raw_answer={"insight": "chronos wins", "recommendation": "prefer chronos"},
        prediction=[1.0, 2.0],
        execution_log=[
            {
                "tool": "spawn_subagent",
                "output": [
                    {
                        "success": True,
                        "result": {
                            "answer": [1.0, 2.0],
                            "execution_log": [{"tool": "chronos2_forecast"}],
                        },
                    },
                    {
                        "success": True,
                        "result": {
                            "answer": [0.0, 0.0],
                            "execution_log": [{"tool": "ets_forecast"}],
                        },
                    },
                ],
            },
            {
                "tool": "evaluate_batch_against_gt",
                "observation": {
                    "results": [
                        {"name": "chronos2_forecast", "metrics": {"mae": 1.2}},
                        {"name": "ets_forecast", "metrics": {"mae": 2.5}},
                    ],
                    "ground_truth": [1.0, 2.0],
                },
            },
        ],
        task_type="forecast",
        description="forecast sample",
        domain="weather",
        scope_keys=["weather_aligned_short_forecast"],
        reasoning="",
        fingerprint={"length": 168, "mean": 12.3, "std": 1.5},
        profiling_results={"detect_seasonality": {"seasonal": True}},
        horizon=24,
        source_metadata={},
        trace_id="trace-forecast-1",
    )

    assert runner.memory.kwargs is not None
    assert runner.memory.kwargs["tool_chain"] == ["chronos2_forecast"]
    assert "horizon=24" in runner.memory.kwargs["data_fingerprint"]


@pytest.mark.asyncio
async def test_save_learning_outcome_records_winner_prediction_not_learning_summary():
    runner = PipelineRunner(disable_memory=True)

    class _MemoryStub:
        def __init__(self):
            self.kwargs = None

        async def record(self, **kwargs):
            self.kwargs = kwargs

    runner.memory = _MemoryStub()

    await runner._save_learning_outcome(
        answer_type="learning_summary",
        raw_answer={
            "insight": "timesfm wins",
            "recommendation": "prefer timesfm",
        },
        prediction={
            "insight": "timesfm wins",
            "recommendation": "prefer timesfm",
        },
        execution_log=[
            {
                "tool": "spawn_subagent",
                "output": [
                    {
                        "success": True,
                        "result": {
                            "answer": [1.0, 2.0],
                            "execution_log": [{"tool": "timesfm2_forecast"}],
                        },
                    },
                    {
                        "success": True,
                        "result": {
                            "answer": [3.0, 4.0],
                            "execution_log": [{"tool": "chronos2_forecast"}],
                        },
                    },
                ],
            },
            {
                "tool": "evaluate_batch_against_gt",
                "observation": {
                    "results": [
                        {"name": "timesfm2_forecast", "metrics": {"mae": 1.2}},
                        {"name": "chronos2_forecast", "metrics": {"mae": 2.5}},
                    ],
                },
            },
        ],
        task_type="forecast",
        description="forecast sample",
        domain="weather",
        scope_keys=["weather_aligned_short_forecast"],
        reasoning="",
        fingerprint={},
        profiling_results={},
        horizon=24,
        source_metadata={},
        trace_id="trace-forecast-2",
    )

    assert runner.memory.kwargs is not None
    assert runner.memory.kwargs["predicted"] == "[1.0, 2.0]"


@pytest.mark.asyncio
async def test_save_learning_outcome_prefers_explicit_reasoned_winner_over_best_metrics():
    runner = PipelineRunner(disable_memory=True)

    class _MemoryStub:
        def __init__(self):
            self.kwargs = None

        async def record(self, **kwargs):
            self.kwargs = kwargs

    runner.memory = _MemoryStub()

    raw_answer = {
        "insight": (
            "Across the explored candidates, the Prophet forecast-led branch was the best "
            "candidate because it matched the positive regime best."
        ),
        "recommendation": (
            "Prefer the Prophet-led conversion over the summarize and Aurora paths when "
            "the winning branch stays positive from the start."
        ),
        "reasoning": (
            "1. The summarize path was evaluated and rejected because it stayed negative.\n"
            "2. Aurora improved shape realism but remained too negative early.\n"
            "3. The Prophet path stayed positive from the start and was selected as the branch winner."
        ),
    }

    await runner._save_learning_outcome(
        answer_type="learning_summary",
        raw_answer=raw_answer,
        prediction=raw_answer,
        execution_log=[
            {
                "tool": "spawn_subagent",
                "output": [
                    {
                        "success": True,
                        "result": {
                            "answer": {"macd": [-0.1, -0.05]},
                            "execution_log": [{"tool": "summarize"}, {"tool": "llm_reasoning_indicator"}],
                        },
                    },
                    {
                        "success": True,
                        "result": {
                            "answer": {"macd": [-0.02, 0.03]},
                            "execution_log": [{"tool": "aurora_forecast"}, {"tool": "llm_reasoning_indicator"}],
                        },
                    },
                    {
                        "success": True,
                        "result": {
                            "answer": {"macd": [0.02, 0.06]},
                            "execution_log": [{"tool": "prophet_forecast"}, {"tool": "llm_reasoning_indicator"}],
                        },
                    },
                ],
            },
            {
                "tool": "evaluate_against_gt",
                "args": {"prediction": {"macd": [-0.1, -0.05]}},
                "observation": {"metrics": {"avg_mae": 0.071}},
            },
        ],
        task_type="indicator",
        description="indicator sample",
        domain="finance",
        scope_keys=["finance_aligned_short_indicator_macd"],
        reasoning=raw_answer["reasoning"],
        fingerprint={},
        profiling_results={},
        horizon=None,
        source_metadata={},
        trace_id="trace-indicator-explicit-winner",
    )

    assert runner.memory.kwargs is not None
    assert runner.memory.kwargs["tool_chain"] == ["prophet_forecast", "llm_reasoning_indicator"]
    assert runner.memory.kwargs["predicted"] == "{'macd': [0.02, 0.06]}"
    assert runner.memory.kwargs["metrics"] == {}
    assert runner.memory.kwargs["correct"] is None
    assert runner.memory.kwargs["pipeline_summary"]
    assert "prophet" in runner.memory.kwargs["pipeline_summary"].lower()


def test_extract_candidate_traces_prefers_real_names_over_approach_ids():
    execution_log = [
        {
            "tool": "spawn_subagent",
            "output": [
                {
                    "success": True,
                    "result": {
                        "answer": [1.0, 2.0],
                        "execution_log": [
                            {
                                "tool": "chronos2_forecast",
                                "thought": "Reason: strong daily cycle. Action: run chronos2_forecast.",
                                "observation": {"forecast": [1.0, 2.0]},
                            }
                        ],
                    },
                },
                {
                    "success": True,
                    "result": {
                        "answer": [0.0, 0.0],
                        "execution_log": [{"tool": "ets_forecast"}],
                    },
                },
            ],
        },
        {
            "tool": "evaluate_batch_against_gt",
            "observation": {
                "results": [
                    {"name": "approach_0", "metrics": {"mae": 1.2}},
                    {"name": "approach_1", "metrics": {"mae": 2.5}},
                ],
            },
        },
    ]

    traces = PipelineRunner._extract_candidate_traces(execution_log)

    assert traces[0]["name"] == "chronos2_forecast"
    assert traces[1]["name"] == "ets_forecast"
    assert traces[0]["metrics"]["mae"] == 1.2
    assert traces[1]["metrics"]["mae"] == 2.5
    assert traces[0]["steps"][0]["tool"] == "chronos2_forecast"
    assert "Reason:" in traces[0]["steps"][0]["thought"]
    assert "forecast" in traces[0]["steps"][0]["observation"]


def test_extract_candidate_traces_normalizes_composite_eval_name_to_real_tool():
    execution_log = [
        {
            "tool": "spawn_subagent",
            "output": [
                {
                    "success": True,
                    "result": {
                        "answer": {"trend": "+2% ~ +4%"},
                        "candidate_name": "sentiment, keyword_extract, llm_reasoning_classification",
                        "execution_log": [{"tool": "llm_reasoning_classification"}],
                    },
                }
            ],
        },
        {
            "tool": "evaluate_batch_against_gt",
            "observation": {
                "results": [
                    {
                        "name": "sentiment, keyword_extract, llm_reasoning_classification",
                        "metrics": {"accuracy": 1.0, "correct": True},
                    }
                ],
            },
        },
    ]

    traces = PipelineRunner._extract_candidate_traces(execution_log)

    assert traces[0]["name"] == "llm_reasoning_classification"
    assert traces[0]["candidate_key"] == "llm_reasoning_classification"
    assert traces[0]["metrics"]["accuracy"] == 1.0


def test_extract_candidate_traces_accepts_list_tools_hint():
    execution_log = [
        {
            "tool": "spawn_subagent",
            "output": [
                {
                    "success": True,
                    "result": {
                        "answer": "C",
                        "tools_hint": ["window_stats", "value_at", "llm_reasoning_qa"],
                        "execution_log": [],
                    },
                }
            ],
        },
        {
            "tool": "evaluate_batch_against_gt",
            "observation": {
                "results": [
                    {
                        "name": "llm_reasoning_qa",
                        "metrics": {"accuracy": 1.0, "correct": True},
                    }
                ],
            },
        },
    ]

    traces = PipelineRunner._extract_candidate_traces(execution_log)

    assert traces[0]["name"] == "llm_reasoning_qa"
    assert traces[0]["candidate_key"] == "llm_reasoning_qa"
    assert traces[0]["steps"][0]["tool"] == "llm_reasoning_qa"


def test_extract_candidate_traces_keeps_top_level_candidates_in_batch_eval_order():
    execution_log = [
        {
            "tool": "llm_reasoning_classification",
            "observation": {"answer": "+2% ~ +4%"},
        },
        {
            "tool": "spawn_subagent",
            "output": [
                {
                    "success": True,
                    "result": {
                        "answer": "-2% ~ +2%",
                        "execution_log": [
                            {
                                "tool": "llm_reasoning_classification",
                                "observation": {"answer": "-2% ~ +2%"},
                            }
                        ],
                    },
                },
                {
                    "success": True,
                    "result": {
                        "answer": "+2% ~ +4%",
                        "execution_log": [
                            {"tool": "segment", "observation": {"segments": [[1.0], [2.0]]}},
                            {
                                "tool": "llm_reasoning_classification",
                                "observation": {"answer": "+2% ~ +4%"},
                            },
                        ],
                    },
                },
            ],
        },
        {
            "tool": "evaluate_batch_against_gt",
            "observation": {
                "results": [
                    {
                        "name": "approach_0",
                        "metrics": {"accuracy": 1.0, "correct": True},
                        "prediction": "+2% ~ +4%",
                    },
                    {
                        "name": "approach_1",
                        "metrics": {"accuracy": 0.0, "correct": False},
                        "prediction": "-2% ~ +2%",
                    },
                    {
                        "name": "approach_2",
                        "metrics": {"accuracy": 1.0, "correct": True},
                        "prediction": "+2% ~ +4%",
                    },
                ],
            },
        },
    ]

    traces = PipelineRunner._extract_candidate_traces(execution_log)

    assert len(traces) == 3
    assert traces[0]["tool_chain"] == ["llm_reasoning_classification"]
    assert traces[0]["prediction"] == "+2% ~ +4%"
    assert traces[0]["metrics"]["accuracy"] == 1.0
    assert traces[1]["tool_chain"] == ["llm_reasoning_classification"]
    assert traces[1]["prediction"] == "-2% ~ +2%"
    assert traces[1]["metrics"]["accuracy"] == 0.0
    assert traces[2]["tool_chain"] == ["segment", "llm_reasoning_classification"]
    assert traces[2]["prediction"] == "+2% ~ +4%"
    assert traces[2]["metrics"]["accuracy"] == 1.0


def test_extract_candidate_traces_trims_top_level_validation_tail_after_answer_step():
    execution_log = [
        {
            "tool": "timesfm2_forecast",
            "observation": {"point_forecast": [10.0, 9.8, 9.7]},
        },
        {
            "tool": "llm_reasoning_forecast",
            "observation": {"answer": [10.0, 9.9, 9.8]},
        },
        {
            "tool": "uncertainty_quantify",
            "observation": {"interval": [9.5, 10.3]},
        },
        {
            "tool": "window_stats",
            "observation": {"mean": 9.9},
        },
        {
            "tool": "fol_check",
            "observation": {"passed": True},
        },
        {
            "tool": "evaluate_against_gt",
            "args": {"prediction": [10.0, 9.9, 9.8]},
            "observation": {"metrics": {"mae": 0.1}},
        },
    ]

    traces = PipelineRunner._extract_candidate_traces(execution_log)

    assert len(traces) == 2
    assert traces[0]["tool_chain"] == ["timesfm2_forecast"]
    assert traces[0]["prediction"] == [10.0, 9.8, 9.7]
    assert traces[1]["tool_chain"] == [
        "llm_reasoning_forecast",
        "uncertainty_quantify",
        "window_stats",
        "fol_check",
    ]
    assert traces[1]["prediction"] == [10.0, 9.9, 9.8]
    assert traces[1]["metrics"] == {"mae": 0.1}


def test_extract_candidate_traces_splits_multiple_answer_steps_in_one_trace():
    execution_log = [
        {
            "tool": "aurora_forecast",
            "observation": {"point_forecast": [11.4, 11.5]},
        },
        {
            "tool": "llm_reasoning_forecast",
            "observation": {"answer": [11.45, 11.55]},
        },
        {
            "tool": "uncertainty_quantify",
            "observation": {"mean_width": 0.4},
        },
        {
            "tool": "sentiment",
            "observation": {"sentiments": [{"polarity": "positive"}]},
        },
        {
            "tool": "ets_forecast",
            "observation": {"forecast": [12.8, 12.9]},
        },
    ]

    traces = PipelineRunner._extract_candidate_traces(execution_log)

    assert len(traces) == 3
    assert traces[0]["tool_chain"] == ["aurora_forecast"]
    assert traces[0]["prediction"] == [11.4, 11.5]
    assert traces[1]["tool_chain"] == [
        "llm_reasoning_forecast",
        "uncertainty_quantify",
        "sentiment",
    ]
    assert traces[1]["prediction"] == [11.45, 11.55]
    assert traces[2]["tool_chain"] == ["ets_forecast"]
    assert traces[2]["prediction"] == [12.8, 12.9]


def test_extract_candidate_traces_matches_same_display_name_batch_results_by_stable_branch_key():
    execution_log = [
        {
            "tool": "spawn_subagent",
            "output": [
                {
                    "success": True,
                    "result": {
                        "slot_id": 0,
                        "answer": "stable",
                        "execution_log": [
                            {
                                "tool": "llm_reasoning_classification",
                                "observation": {"answer": "stable"},
                            }
                        ],
                    },
                },
                {
                    "success": True,
                    "result": {
                        "slot_id": 1,
                        "answer": "increasing",
                        "execution_log": [
                            {
                                "tool": "llm_reasoning_classification",
                                "observation": {"answer": "increasing"},
                            }
                        ],
                    },
                },
            ],
        },
        {
            "tool": "evaluate_batch_against_gt",
            "observation": {
                "results": [
                    {
                        "name": "slot_0",
                        "metrics": {"accuracy": 0.0, "correct": False},
                        "prediction": "stable",
                    },
                    {
                        "name": "slot_1",
                        "metrics": {"accuracy": 1.0, "correct": True},
                        "prediction": "increasing",
                    },
                ],
            },
        },
    ]

    traces = PipelineRunner._extract_candidate_traces(execution_log)

    assert len(traces) == 2
    assert traces[0]["name"] == "llm_reasoning_classification"
    assert traces[0]["prediction"] == "stable"
    assert traces[0]["metrics"]["accuracy"] == 0.0
    assert traces[1]["name"] == "llm_reasoning_classification"
    assert traces[1]["prediction"] == "increasing"
    assert traces[1]["metrics"]["accuracy"] == 1.0


def test_normalize_learning_summary_answer_adds_winner_fields_from_execution_log():
    execution_log = [
        {
            "tool": "spawn_subagent",
            "output": [
                {
                    "success": True,
                    "result": {
                        "slot_id": 0,
                        "answer": [1.0, 2.0],
                        "execution_log": [{"tool": "chronos2_forecast"}],
                    },
                },
                {
                    "success": True,
                    "result": {
                        "slot_id": 1,
                        "answer": [3.0, 4.0],
                        "execution_log": [{"tool": "ets_forecast"}],
                    },
                },
            ],
        },
        {
            "tool": "evaluate_batch_against_gt",
            "observation": {
                "results": [
                    {"name": "chronos2_forecast", "metrics": {"mae": 1.2}},
                    {"name": "ets_forecast", "metrics": {"mae": 2.5}},
                ],
            },
        },
    ]

    normalized = PipelineRunner._normalize_learning_summary_answer(
        {
            "insight": "chronos was best",
            "recommendation": "prefer chronos here",
        },
        execution_log,
    )

    assert "Strongest path on this sample: `chronos2_forecast`" in normalized["insight"]
    assert "mae=1.200" in normalized["insight"]
    assert "Reusable lesson:" in normalized["recommendation"]
    assert "`chronos2_forecast`" in normalized["recommendation"]
    assert "`ets_forecast`" in normalized["recommendation"]
    assert normalized["winner_candidate"] == "chronos2_forecast"
    assert normalized["winner_prediction"] == [1.0, 2.0]
    assert normalized["winner_tool_chain"] == ["chronos2_forecast"]
    assert normalized["winner_metrics"] == {"mae": 1.2}
    assert len(normalized["compared_candidates"]) == 2


def test_normalize_learning_summary_answer_rewrites_conflicting_free_text_to_gt_winner():
    execution_log = [
        {
            "tool": "spawn_subagent",
            "output": [
                {
                    "success": True,
                    "result": {
                        "answer": "Moderate Positive Correlation",
                        "execution_log": [{"tool": "slice"}],
                    },
                },
                {
                    "success": True,
                    "result": {
                        "answer": "No Correlation",
                        "execution_log": [{"tool": "value_at"}],
                    },
                },
                {
                    "success": True,
                    "result": {
                        "answer": "No Correlation",
                        "execution_log": [{"tool": "basic_stats"}],
                    },
                },
            ],
        },
        {
            "tool": "evaluate_against_gt",
            "args": {"prediction": "No Correlation"},
            "observation": {"metrics": {"accuracy": 0.0}},
        },
    ]

    normalized = PipelineRunner._normalize_learning_summary_answer(
        {
            "insight": "Winning candidate: the slice-based path, which concluded Moderate Positive Correlation.",
            "recommendation": "Classify this setup as Moderate Positive Correlation instead of No Correlation.",
        },
        execution_log,
    )

    assert normalized["winner_candidate"] == "value_at"
    assert normalized["winner_prediction"] == "No Correlation"
    assert normalized["winner_metrics"] == {"accuracy": 0.0}
    assert "value_at" in normalized["insight"]
    assert "`value_at`" in normalized["recommendation"]
    assert "No Correlation" in normalized["insight"]
    assert "slice-based path" not in normalized["insight"]
    assert "Moderate Positive Correlation instead of No Correlation" not in normalized["recommendation"]
    assert normalized["recommendation"].startswith("Failure pattern:")
    assert "do not trust" in normalized["recommendation"]


def test_normalize_learning_summary_answer_uses_failure_first_insight_for_incorrect_best_candidate():
    execution_log = [
        {
            "tool": "spawn_subagent",
            "output": [
                {
                    "success": True,
                    "result": {
                        "answer": "stable",
                        "reasoning": "The segmented daily path flattened noisy hourly variation and looked trend-neutral.",
                        "execution_log": [{"tool": "segment"}, {"tool": "llm_reasoning_classification"}],
                    },
                },
                {
                    "success": True,
                    "result": {
                        "answer": "decreasing",
                        "reasoning": "The direct classification path emphasized the visibly cooler tail of the series.",
                        "execution_log": [{"tool": "llm_reasoning_classification"}],
                    },
                },
            ],
        },
        {
            "tool": "evaluate_against_gt",
            "args": {"prediction": "stable"},
            "observation": {"metrics": {"accuracy": 0.0, "correct": False}},
        },
    ]

    normalized = PipelineRunner._normalize_learning_summary_answer(
        {
            "insight": "Both explored paths looked reasonable, but GT showed they were wrong.",
            "recommendation": "Treat this as a failure pattern rather than a reusable win.",
        },
        execution_log,
    )

    assert normalized["winner_candidate"] == "segment"
    assert normalized["winner_prediction"] == "stable"
    assert normalized["winner_metrics"] == {"accuracy": 0.0, "correct": False}
    assert normalized["insight"].startswith("Strongest failed path:")
    assert "Winner for this run:" not in normalized["insight"]
    assert "Why it still looked plausible:" in normalized["insight"]
    assert normalized["recommendation"].startswith("Failure pattern:")


def test_normalize_learning_summary_answer_prefers_explicit_reasoned_winner_over_best_metrics():
    execution_log = [
        {
            "tool": "spawn_subagent",
            "output": [
                {
                    "success": True,
                    "result": {
                        "answer": {"macd": [-0.1, -0.05]},
                        "execution_log": [{"tool": "summarize"}, {"tool": "llm_reasoning_indicator"}],
                    },
                },
                {
                    "success": True,
                    "result": {
                        "answer": {"macd": [-0.02, 0.03]},
                        "execution_log": [{"tool": "aurora_forecast"}, {"tool": "llm_reasoning_indicator"}],
                    },
                },
                {
                    "success": True,
                    "result": {
                        "answer": {"macd": [0.02, 0.06]},
                        "execution_log": [{"tool": "prophet_forecast"}, {"tool": "llm_reasoning_indicator"}],
                    },
                },
            ],
        },
        {
            "tool": "evaluate_against_gt",
            "args": {"prediction": {"macd": [-0.1, -0.05]}},
            "observation": {"metrics": {"avg_mae": 0.071}},
        },
    ]

    normalized = PipelineRunner._normalize_learning_summary_answer(
        {
            "insight": (
                "Across three explored candidates, the Prophet forecast-led path was the best "
                "candidate because it matched the positive regime best."
            ),
            "recommendation": (
                "Prefer the Prophet-led conversion over the sentiment-only and Aurora paths "
                "when the winning branch stays positive from the start."
            ),
            "reasoning": (
                "1. The summarize path was evaluated and rejected because it stayed negative.\n"
                "2. Aurora improved shape realism but remained too negative early.\n"
                "3. The Prophet path stayed positive from the start and was selected as the branch winner."
            ),
        },
        execution_log,
    )

    assert normalized["winner_candidate"] == "prophet_forecast"
    assert normalized["winner_prediction"] == {"macd": [0.02, 0.06]}
    assert normalized["winner_tool_chain"] == ["prophet_forecast", "llm_reasoning_indicator"]
    assert normalized["winner_metrics"] == {}
    assert "Prophet forecast-led path was the best candidate" in normalized["insight"]


def test_normalize_learning_summary_answer_matches_winner_prediction_without_broadcast_error():
    execution_log = [
        {
            "tool": "spawn_subagent",
            "output": [
                {
                    "success": True,
                    "result": {
                        "answer": np.arange(72, dtype=float),
                        "execution_log": [{"tool": "slice"}],
                    },
                },
                {
                    "success": True,
                    "result": {
                        "answer": [13.7, 13.5, 13.3, 13.1],
                        "execution_log": [{"tool": "llm_reasoning_forecast"}],
                    },
                },
            ],
        }
    ]

    normalized = PipelineRunner._normalize_learning_summary_answer(
        {
            "insight": "The llm_reasoning_forecast candidate won.",
            "recommendation": "Prefer the text-aware branch here.",
            "winner_prediction": [13.7, 13.5, 13.3, 13.1],
        },
        execution_log,
    )

    assert normalized["winner_candidate"] == "llm_reasoning_forecast"
    assert normalized["winner_prediction"] == [13.7, 13.5, 13.3, 13.1]


def test_select_learning_summary_winner_trace_disambiguates_same_tool_chain_by_prediction():
    candidate_traces = [
        {
            "name": "llm_reasoning_classification",
            "tool_chain": ["llm_reasoning_classification"],
            "prediction": "stable",
            "reasoning": "The direct classifier judged the daily means as flat overall.",
        },
        {
            "name": "llm_reasoning_classification",
            "tool_chain": ["llm_reasoning_classification"],
            "prediction": "increasing",
            "reasoning": "The same classifier found a positive daily-mean slope.",
        },
    ]

    winner = PipelineRunner._select_learning_summary_winner_trace(
        {
            "winner_tool_chain": ["llm_reasoning_classification"],
            "winner_prediction": "increasing",
            "insight": "The classification branch that returned increasing was the winner.",
        },
        candidate_traces,
    )

    assert winner is not None
    assert winner["prediction"] == "increasing"
    assert winner["reasoning"] == "The same classifier found a positive daily-mean slope."


def test_select_learning_summary_winner_trace_disambiguates_same_name_by_prediction():
    candidate_traces = [
        {
            "name": "llm_reasoning_classification",
            "tool_chain": ["llm_reasoning_classification"],
            "prediction": "stable",
            "reasoning": "The direct classifier judged the daily means as flat overall.",
        },
        {
            "name": "llm_reasoning_classification",
            "tool_chain": ["llm_reasoning_classification"],
            "prediction": "decreasing",
            "reasoning": "The same classifier found a clearly negative daily-mean slope.",
        },
    ]

    winner = PipelineRunner._select_learning_summary_winner_trace(
        {
            "winner_candidate": "llm_reasoning_classification",
            "winner_prediction": "decreasing",
            "insight": "The llm_reasoning_classification branch that returned decreasing was selected.",
        },
        candidate_traces,
    )

    assert winner is not None
    assert winner["prediction"] == "decreasing"
    assert winner["reasoning"] == "The same classifier found a clearly negative daily-mean slope."


def test_extract_candidate_traces_prefers_observed_tool_answer_over_wrapped_result_answer():
    execution_log = [
        {
            "tool": "spawn_subagent",
            "output": [
                {
                    "success": True,
                    "result": {
                        "answer": "C",
                        "execution_log": [
                            {
                                "tool": "llm_reasoning_qa",
                                "thought": "Pick the best answer from the article and price action.",
                                "observation": {"answer": "B"},
                            }
                        ],
                    },
                }
            ],
        }
    ]

    traces = PipelineRunner._extract_candidate_traces(execution_log)

    assert traces[0]["prediction"] == "B"
    assert traces[0]["steps"][0]["tool"] == "llm_reasoning_qa"


def test_extract_candidate_traces_attaches_public_reasoning_snippets():
    execution_log = [
        {
            "tool": "llm_reasoning_classification",
            "thought": (
                "We still need recent-window quantitative evidence to support the classification. "
                "The last 24 points fall by 7.6%, which already matches the <-4% bucket."
            ),
            "observation": {"answer": "<-4%"},
        }
    ]

    traces = PipelineRunner._extract_candidate_traces(execution_log)

    assert traces
    assert traces[0]["public_reasoning_snippets"] == []


def test_extract_candidate_traces_public_reasoning_ignores_final_candidate_summary():
    execution_log = [
        {
            "tool": "spawn_subagent",
            "output": [
                {
                    "success": True,
                    "result": {
                        "answer": "stable",
                        "reasoning": (
                            "1. Additional evidence did not replace that answer with a new valid "
                            "branch answer to use instead.\n"
                            "2. The last three daily means differ by less than 0.1C, so the "
                            "near-term class is stable.\n"
                            "3. The final-turn instructions require no more tool calls."
                        ),
                        "execution_context": {
                            "tool_chain": ["llm_reasoning_classification"],
                            "public_reasoning": [
                                "The last three daily means differ by less than 0.1C, so the near-term class is stable"
                            ],
                            "evidence_summary": "The recent daily means are effectively unchanged, so the near-term class is stable.",
                        },
                        "execution_log": [
                            {
                                "tool": "llm_reasoning_classification",
                                "thought": (
                                    "The last three daily means differ by less than 0.1C, so the "
                                    "near-term class is stable."
                                ),
                                "observation": {"answer": "stable"},
                            }
                        ],
                    },
                }
            ],
        }
    ]

    traces = PipelineRunner._extract_candidate_traces(execution_log)

    assert traces
    assert traces[0]["public_reasoning_snippets"] == [
        "The last three daily means differ by less than 0.1C, so the near-term class is stable"
    ]


def test_extract_candidate_traces_public_reasoning_can_be_empty_when_only_final_summary_exists():
    execution_log = [
        {
            "tool": "spawn_subagent",
            "output": [
                {
                    "success": True,
                    "result": {
                        "answer": "stable",
                        "reasoning": (
                            "1. The task requires the final answer to exactly match a prior "
                            "tool-produced answer in this branch.\n"
                            "2. Return the saved label unchanged."
                        ),
                        "execution_context": {
                            "tool_chain": ["llm_reasoning_classification"],
                            "public_reasoning": [],
                            "evidence_summary": "",
                        },
                        "execution_log": [
                            {
                                "tool": "llm_reasoning_classification",
                                "thought": "",
                                "observation": {"answer": "stable"},
                            }
                        ],
                    },
                }
            ],
        }
    ]

    traces = PipelineRunner._extract_candidate_traces(execution_log)

    assert traces
    assert traces[0]["public_reasoning_snippets"] == []


def test_normalize_learning_summary_answer_uses_execution_context_tool_chain_as_winner_hint():
    execution_log = [
        {
            "tool": "spawn_subagent",
            "output": [
                {
                    "success": True,
                    "result": {
                        "answer": "stable",
                        "execution_log": [
                            {
                                "tool": "llm_reasoning_classification",
                                "thought": "The last three daily means differ by less than 0.1C.",
                                "observation": {"answer": "stable"},
                            }
                        ],
                    },
                }
            ],
        },
        {
            "tool": "evaluate_batch_against_gt",
            "observation": {
                "results": [
                    {
                        "prediction": "stable",
                        "metrics": {"score": 0.0},
                    }
                ]
            },
        },
    ]

    normalized = PipelineRunner._normalize_learning_summary_answer(
        {
            "insight": "stable path won",
            "recommendation": "prefer the stable path",
            "execution_context": {
                "tool_chain": ["llm_reasoning_classification"],
                "public_reasoning": ["The last three daily means differ by less than 0.1C."],
                "evidence_summary": "The near-term daily means are flat.",
            },
        },
        execution_log,
    )

    assert normalized["winner_tool_chain"] == ["llm_reasoning_classification"]


def test_normalize_learning_summary_answer_rewrites_explicit_summary_when_text_conflicts_with_trace():
    execution_log = [
        {
            "tool": "spawn_subagent",
            "output": [
                {
                    "success": True,
                    "result": {
                        "answer": "decreasing",
                        "execution_log": [
                            {
                                "tool": "llm_reasoning_classification",
                                "thought": "Classify the daily means with the task rule.",
                                "observation": {"answer": "stable"},
                            }
                        ],
                    },
                }
            ],
        }
    ]

    normalized = PipelineRunner._normalize_learning_summary_answer(
        {
            "winner_candidate": "llm_reasoning_classification",
            "insight": "The llm_reasoning_classification branch returned the task-valid answer decreasing.",
            "recommendation": "Reuse decreasing in similar samples.",
            "reasoning": "1. The selected branch returned decreasing.",
        },
        execution_log,
    )

    assert normalized["winner_candidate"] == "llm_reasoning_classification"
    assert normalized["winner_prediction"] == "stable"
    assert "stable" in normalized["insight"]
    assert "decreasing" not in normalized["insight"]


def test_normalize_learning_summary_answer_rewrites_meta_finalization_summary():
    execution_log = [
        {
            "tool": "spawn_subagent",
            "output": [
                {
                    "success": True,
                    "result": {
                        "answer": "decreasing",
                        "execution_log": [
                            {
                                "tool": "llm_reasoning_classification",
                                "thought": "Classify the daily means with the task rule.",
                                "observation": {"answer": "decreasing"},
                            }
                        ],
                    },
                }
            ],
        }
    ]

    normalized = PipelineRunner._normalize_learning_summary_answer(
        {
            "winner_candidate": "llm_reasoning_classification",
            "insight": "Winner for this run: llm_reasoning_classification. Why it won: Returning the corrected final JSON with unchanged saved label.",
            "recommendation": "Reuse the exact saved answer unchanged next time.",
            "reasoning": "1. Preserved the existing payload.\n2. Returned the corrected final JSON.",
        },
        execution_log,
    )

    assert "corrected final json" not in normalized["insight"].lower()
    assert "saved label" not in normalized["insight"].lower()
    assert "reuse the exact saved answer" not in normalized["recommendation"].lower()
    assert "decreasing" in normalized["insight"]


def test_normalize_learning_summary_answer_rewrites_meta_execution_summary():
    execution_log = [
        {
            "tool": "spawn_subagent",
            "output": [
                {
                    "success": True,
                    "result": {
                        "answer": "decreasing",
                        "execution_log": [
                            {
                                "tool": "segment",
                                "thought": "Split the hourly readings into daily windows.",
                                "observation": {"segments": [[1, 2], [3, 4]]},
                            },
                            {
                                "tool": "llm_reasoning_classification",
                                "thought": "Classify the daily means with the task rule.",
                                "observation": {"answer": "decreasing"},
                            },
                        ],
                    },
                }
            ],
        }
    ]

    normalized = PipelineRunner._normalize_learning_summary_answer(
        {
            "winner_candidate": "segment",
            "winner_tool_chain": ["segment", "llm_reasoning_classification"],
            "insight": "Winner for this run: llm_reasoning_classification. Why it won: Branch comparison is genuinely useful here because the latest instruction requires an immediate tool call before yielding the task-valid answer.",
            "recommendation": "Reusable lesson: prefer this path because learning mode encourages spawning independent branches before any learning-summary completion.",
            "reasoning": "1. A task-valid label already exists.\n2. Spawning independent branches is justified.",
        },
        execution_log,
    )

    combined = f"{normalized['insight']} {normalized['recommendation']}".lower()
    assert "branch comparison" not in combined
    assert "latest instruction requires" not in combined
    assert "task-valid" not in combined
    assert "learning mode" not in combined
    assert "decreasing" in normalized["insight"].lower()


def test_normalize_learning_summary_answer_keeps_specific_branch_name_when_batch_eval_name_conflicts():
    execution_log = [
        {
            "tool": "spawn_subagent",
            "output": [
                {
                    "success": True,
                    "result": {
                        "answer": [1.0, 2.0],
                        "execution_log": [{"tool": "aurora_forecast"}],
                        "tools_hint": ["aurora_forecast", "llm_reasoning_forecast"],
                    },
                },
                {
                    "success": True,
                    "result": {
                        "answer": [3.0, 4.0],
                        "execution_log": [{"tool": "moirai1_1_r_base_forecast"}],
                        "tools_hint": ["moirai1_1_r_base_forecast", "llm_reasoning_forecast"],
                    },
                },
            ],
        },
        {
            "tool": "evaluate_batch_against_gt",
            "observation": {
                "results": [
                    {
                        "name": "top_level::ets_forecast",
                        "metrics": {"mae": 0.2},
                        "prediction": [1.0, 2.0],
                    },
                    {
                        "name": "top_level::arima_forecast",
                        "metrics": {"mae": 0.8},
                        "prediction": [3.0, 4.0],
                    },
                ],
            },
        },
    ]

    normalized = PipelineRunner._normalize_learning_summary_answer(
        {
            "insight": "free text",
            "recommendation": "free text",
        },
        execution_log,
    )

    assert normalized["winner_candidate"] == "aurora_forecast"
    assert normalized["winner_tool_chain"] == ["aurora_forecast"]
    assert "Strongest path on this sample: `aurora_forecast`" in normalized["insight"]
    assert "using `ets_forecast`" not in normalized["insight"]
    assert "`aurora_forecast`" in normalized["recommendation"]


def test_normalize_learning_summary_answer_uses_candidate_reasoning_without_gt_phrasing():
    execution_log = [
        {
            "tool": "spawn_subagent",
            "output": [
                {
                    "success": True,
                    "result": {
                        "candidate_name": "slot_0",
                        "answer": [1.0, 2.0],
                        "reasoning": (
                            "Reason: The execution context explicitly requires using at least one tool from the "
                            "assigned path before finishing, but this path stayed close to the recent regime and "
                            "avoided overshooting. Action: finish."
                        ),
                        "execution_log": [
                            {"tool": "slice", "thought": "Reason: Focus on the recent regime before forecasting. Action: slice the tail."},
                            {"tool": "arima_forecast", "thought": "Reason: The recent regime is stable enough for a local ARIMA continuation. Action: forecast 2 steps."},
                        ],
                    },
                },
                {
                    "success": True,
                    "result": {
                        "candidate_name": "slot_1",
                        "answer": [3.0, 4.0],
                        "reasoning": "Reason: This path was more aggressive than the recent regime. Action: finish.",
                        "execution_log": [{"tool": "ets_forecast"}],
                    },
                },
            ],
        },
        {
            "tool": "evaluate_batch_against_gt",
            "observation": {
                "results": [
                    {"name": "slot_0", "metrics": {"mae": 0.2}},
                    {"name": "slot_1", "metrics": {"mae": 0.8}},
                ],
            },
        },
    ]

    normalized = PipelineRunner._normalize_learning_summary_answer(
        {"insight": "free text", "recommendation": "free text"},
        execution_log,
    )

    assert "Winner for this run:" not in normalized["insight"]
    assert "Strongest path on this sample: `slice -> arima_forecast`" in normalized["insight"]
    assert "Why it fit:" in normalized["insight"]
    assert "recent regime" in normalized["insight"].lower()
    assert "Alternative signals:" in normalized["insight"]
    assert "GT-selected" not in normalized["insight"]
    assert "assigned path requires" not in normalized["insight"].lower()
    assert "GT evaluation" not in normalized["recommendation"]
    assert "Reusable lesson:" in normalized["recommendation"]
    assert "`slice -> arima_forecast`" in normalized["recommendation"]
    assert "before finishing" not in normalized["recommendation"].lower()


def test_notes_compact_candidate_reasoning_filters_framework_wording():
    trace = {
        "tool_chain": ["slice"],
        "steps": [
            {
                "tool": "slice",
                "thought": (
                    "Reason: The execution context explicitly requires using at least one tool "
                    "from the assigned path before finishing. Action: run slice."
                ),
            },
            {
                "tool": "slice",
                "thought": (
                    "Reason: Focus on the recent regime before forecasting. Action: slice the tail."
                ),
            },
        ],
    }

    from timeclaw.memory.hierarchical import Notes

    text = Notes._compact_candidate_reasoning(trace)
    assert text == ""


def test_notes_compact_candidate_reasoning_drops_control_flow_wording():
    trace = {
        "tool_chain": ["llm_reasoning_correlation"],
        "steps": [
            {
                "tool": "llm_reasoning_correlation",
                "thought": (
                    "We have isolated the post-publication window, but still need a task-valid "
                    "candidate label grounded in both the clearly positive article tone and the "
                    "sliced price reaction. The specialized multimodal correlation reasoning tool "
                    "is the most direct next step."
                ),
            },
            {
                "tool": "llm_reasoning_correlation",
                "thought": (
                    "The post-publication window is positive while the article tone is clearly bullish."
                ),
            },
        ],
    }

    from timeclaw.memory.hierarchical import Notes

    text = Notes._compact_candidate_reasoning(trace)
    assert text == ""


def test_notes_compact_candidate_reasoning_drops_tool_mandate_wording():
    trace = {
        "tool_chain": ["detect_trend"],
        "steps": [
            {
                "tool": "detect_trend",
                "thought": (
                    "A tool must be called this turn, and the most relevant available tool to "
                    "support the existing classification is to reaffirm the strong directional "
                    "evidence directly from the series."
                ),
            },
            {
                "tool": "detect_trend",
                "thought": "The series has a strong directional decline directly in the observed values.",
            },
        ],
    }

    from timeclaw.memory.hierarchical import Notes

    text = Notes._compact_candidate_reasoning(trace)
    assert text == ""


def test_training_evaluation_requires_non_empty_metrics():
    assert ExecutorAgent._has_training_evaluation(
        [
            {"tool": "llm_reasoning_forecast", "success": True, "observation": {"text": "x"}},
            {
                "tool": "evaluate_batch_against_gt",
                "success": True,
                "observation": {"metrics": {"accuracy": 1.0}},
            },
        ]
    ) is True
    assert ExecutorAgent._has_training_evaluation(
        [
            {
                "tool": "evaluate_batch_against_gt",
                "success": True,
                "observation": {"metrics": {}},
            }
        ]
    ) is False
