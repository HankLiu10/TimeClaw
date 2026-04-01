from timeclaw.pipeline.runner import PipelineRunner
from timeclaw.memory.hierarchical import DailyNotes
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
    assert DailyNotes._outcome_label(True, {"accuracy": 1.0}) == "OK"
    assert DailyNotes._outcome_label(False, {"accuracy": 0.0}) == "FAIL"
    assert DailyNotes._outcome_label(None, {"mae": 2.0, "rmse": 3.0}) == "SCORED"
    assert DailyNotes._outcome_label(None, {"parse_error": 1.0}) == "FAIL"


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
                            "execution_log": [{"tool": "chronos_forecast"}],
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
                        {"name": "chronos_forecast", "metrics": {"mae": 1.2}},
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
    )

    assert runner.memory.kwargs is not None
    assert runner.memory.kwargs["tool_chain"] == ["chronos_forecast"]
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
                            "execution_log": [{"tool": "timesfm_forecast"}],
                        },
                    },
                    {
                        "success": True,
                        "result": {
                            "answer": [3.0, 4.0],
                            "execution_log": [{"tool": "chronos_forecast"}],
                        },
                    },
                ],
            },
            {
                "tool": "evaluate_batch_against_gt",
                "observation": {
                    "results": [
                        {"name": "timesfm_forecast", "metrics": {"mae": 1.2}},
                        {"name": "chronos_forecast", "metrics": {"mae": 2.5}},
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
                                "tool": "chronos_forecast",
                                "thought": "Reason: strong daily cycle. Action: run chronos_forecast.",
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

    assert traces[0]["name"] == "chronos_forecast"
    assert traces[1]["name"] == "ets_forecast"
    assert traces[0]["metrics"]["mae"] == 1.2
    assert traces[1]["metrics"]["mae"] == 2.5
    assert traces[0]["steps"][0]["tool"] == "chronos_forecast"
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
