from timeclaw.pipeline.runner import PipelineRunner
from timeclaw.memory.hierarchical import Notes
from timeclaw.agents.executor import ExecutorAgent
import pytest


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

    assert "Winner for this run: chronos2_forecast" in normalized["insight"]
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

    assert "Winner for this run:" in normalized["insight"]
    assert "`slice -> arima_forecast`" in normalized["insight"]
    assert "Why it won:" in normalized["insight"]
    assert "recent regime" in normalized["insight"].lower()
    assert "Alternatives:" in normalized["insight"]
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
    assert "execution context" not in text.lower()
    assert "assigned path" not in text.lower()
    assert "before finishing" not in text.lower()
    assert "recent regime" in text.lower()


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
