import pytest

from timeclaw.benchmark import pipeline_eval as pipeline_eval_mod
from timeclaw.benchmark.mtbench_support import (
    generate_learning_analysis,
    render_learning_analysis_text,
    render_learning_summary_from_analysis,
)
from timeclaw.benchmark.mtbench_support import learning_analysis as learning_analysis_mod
from timeclaw.agents.reasoner import ReasonerAgent
from timeclaw.core.tea_protocol import TaskEnvelope, TaskSpec
from timeclaw.memory.hierarchical import Notes
from timeclaw.pipeline import runner as runner_mod
from timeclaw.pipeline.runner import PipelineRunner
from timeclaw.agents.executor import ExecutorAgent


def test_render_learning_analysis_text_uses_four_sections():
    text = render_learning_analysis_text(
        {
            "text_signals": ["Heavy rain caused flash flooding near the station."],
            "sample_conditions": ["short-horizon forecast", "clear daily seasonality"],
            "winner_evidence": ["chronos2_forecast achieved lower MAE than the compared candidate"],
            "failure_signals": ["llm_reasoning_forecast drifted at turning points"],
            "reusable_rule": ["When daily seasonality is clear, consider chronos2_forecast before direct llm_reasoning_forecast."],
        }
    )

    assert "Text Signals:" in text
    assert "Sample Conditions:" in text
    assert "Winner Evidence:" in text
    assert "Failure Signals:" in text
    assert "Reusable Rule:" in text


def test_render_learning_summary_from_analysis_uses_data_characteristics_not_control_flow():
    insight, recommendation = render_learning_summary_from_analysis(
        {
            "sample_conditions": [
                "recent window is flat with low volatility",
                "the question asks for a categorical direction label",
            ],
            "winner_evidence": [
                "llm_reasoning_classification matched the flat recent window better than detect_trend",
            ],
            "failure_signals": [
                "detect_trend over-read a weak uptick",
            ],
            "reusable_rule": [
                "When the recent window is flat and the task asks for a direction label, consider llm_reasoning_classification before detect_trend.",
            ],
        },
        winner_trace={
            "name": "normalize",
            "tool_chain": ["normalize", "normalize", "llm_reasoning_classification"],
            "prediction": "stable",
            "metrics": {"accuracy": 1.0},
        },
        winner_correct=True,
    )

    combined = f"{insight} {recommendation}".lower()
    assert "recent window is flat" in combined
    assert "task-valid" not in combined
    assert "next step" not in combined
    assert "llm_reasoning_classification matched the flat recent window better than detect_trend" in insight
    assert "detect_trend over-read a weak uptick" in insight
    assert "accepted tool chain was" not in insight.lower()
    assert "winner for this run" not in insight.lower()


def test_daily_note_labels_winner_tool_section():
    note = Notes._build_note(  # type: ignore[misc]
        Notes.__new__(Notes),
        {
            "scope_key": "weather_aligned_short_forecast",
            "tool_chain": ["chronos2_forecast"],
            "predicted": "[1,2,3]",
        },
    )
    assert "Winner Tools:" in note
    assert "\nTools:\n" not in note


def test_daily_note_falls_back_to_best_candidate_for_winner_tool():
    note = Notes._build_note(  # type: ignore[misc]
        Notes.__new__(Notes),
        {
            "scope_key": "weather_aligned_short_forecast",
            "tool_chain": ["none"],
            "candidate_traces": [
                {
                    "name": "chronos2_forecast",
                    "tool_chain": ["chronos2_forecast"],
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
    assert "Winner Tools:\n- llm_reasoning_forecast" in note
    assert "Winner Tools:\n- none" not in note


def test_daily_note_reasoning_strips_framework_preface_but_keeps_task_evidence():
    cleaned = Notes._sanitize_candidate_reasoning(
        "We have the required basic_stats evidence, plus prior context already shows strong "
        "persistence and a statistically significant upward trend."
    )

    assert "required basic_stats evidence" not in cleaned.lower()
    assert cleaned.startswith("prior context already shows strong persistence")
    assert "upward trend" in cleaned


def test_daily_note_drops_boilerplate_reasoning_fallback_when_no_real_reasoning():
    note = Notes._build_note(  # type: ignore[misc]
        Notes.__new__(Notes),
        {
            "scope_key": "finance_aligned_long_forecast",
            "candidate_traces": [
                {
                    "name": "value_at",
                    "tool_chain": ["value_at"],
                    "metrics": {"mae": 0.048, "rmse": 0.061},
                    "prediction": [1, 2, 3],
                    "steps": [],
                }
            ],
        },
    )

    assert "Used the value_at path for this candidate." not in note
    assert "  Reasoning:" not in note


def test_daily_note_reasoning_drops_next_evidence_step_wording():
    cleaned = Notes._sanitize_candidate_reasoning(
        "We already have time-series diagnostics showing a statistically significant downward trend, "
        "but the task explicitly requires combining that with the news text; the most efficient next "
        "evidence step is to classify the multimodal trend directly using the provided series and news context."
    )

    assert cleaned == ""


def test_daily_note_reasoning_drops_malformed_tool_prefix():
    cleaned = Notes._sanitize_candidate_reasoning(
        "Tool while combining the observed strong downward time-series trend with the mixed but slightly supportive news context."
    )

    assert cleaned == ""


def test_daily_note_reasoning_drops_tool_call_mandate_now_wording():
    cleaned = Notes._sanitize_candidate_reasoning(
        "A tool must be called now, and we need concise supporting evidence after the failed consistency "
        "check. The most relevant single step is to quantify recent endpoint weakness directly from the "
        "series to reinforce the existing branch answer. The latest evidence already supports a bearish "
        "classification, but the immediate-response instruction requires another tool call."
    )

    assert cleaned == ""


def test_daily_note_reasoning_drops_immediate_tool_call_finish_wording():
    cleaned = Notes._sanitize_candidate_reasoning(
        "A tool call is required immediately, and with only one later turn reserved for finishing, "
        "the most relevant final validation is a broad bound sanity check on the existing branch-valid forecast."
    )

    assert cleaned == ""


def test_daily_note_reasoning_drops_immediate_instruction_and_final_tool_wording():
    cleaned = Notes._sanitize_candidate_reasoning(
        "The immediate instruction requires calling a tool this turn, and we already have forecast candidates "
        "plus a concise news summary."
    )

    assert cleaned == ""


def test_daily_note_reasoning_drops_task_requires_and_path_priority_wording():
    cleaned = Notes._sanitize_candidate_reasoning(
        "The task requires a forecast-based candidate and explicitly prioritizes the ets_forecast path."
    )

    assert cleaned == ""


def test_daily_note_display_candidate_name_combines_mismatched_name_and_tool():
    name = Notes._display_candidate_name(
        {
            "name": "timesfm2_forecast",
            "tool_chain": ["llm_reasoning_forecast"],
        },
        0,
    )

    assert name == "timesfm2_forecast → llm_reasoning_forecast"


def test_daily_note_omits_weakness_for_perfect_accuracy_candidate():
    note = Notes._build_note(  # type: ignore[misc]
        Notes.__new__(Notes),
        {
            "scope_key": "finance_aligned_long_trend",
            "candidate_traces": [
                {
                    "name": "llm_reasoning_classification",
                    "tool_chain": ["llm_reasoning_classification"],
                    "metrics": {"accuracy": 1.0},
                    "prediction": "<-4%",
                    "steps": [],
                }
            ],
        },
    )

    assert "Weakness: This path recorded acc=1.000" not in note


def test_daily_note_compact_reasoning_splits_and_drops_control_flow_sentences():
    reasoning = Notes._compact_candidate_reasoning(
        {
            "steps": [
                {
                    "thought": (
                        "The immediate instruction requires calling a tool this turn, and we already have "
                        "forecast candidates plus a concise news summary. The most relevant final evidence tool "
                        "is keyword extraction on the article to confirm the dominant bullish themes supporting "
                        "the multimodal interpretation."
                    )
                }
            ]
        }
    )

    assert reasoning == ""


def test_daily_note_compact_reasoning_prefers_public_snippets_over_raw_thought():
    reasoning = Notes._compact_candidate_reasoning(
        {
            "public_reasoning_snippets": [
                "recent window dropped by 7.6%, which already matches the <-4% bucket"
            ],
            "steps": [
                {
                    "thought": (
                        "We still need recent-window quantitative evidence to support the classification."
                    )
                }
            ],
        }
    )

    assert reasoning == "Recent window dropped by 7.6%, which already matches the <-4% bucket."


def test_daily_note_reasoning_drops_additional_step_and_remaining_diagnostic_wording():
    cleaned = Notes._sanitize_candidate_reasoning(
        "The most relevant additional step is a sanity check on the produced 78-step forecast against "
        "observed historical bounds. The most relevant remaining diagnostic is uncertainty quantification "
        "for the selected forecast using the historical series."
    )

    assert cleaned == ""


def test_daily_note_reasoning_drops_independent_branch_comparison_wording():
    cleaned = Notes._sanitize_candidate_reasoning(
        "Independent branches can separately reason from trend diagnostics versus explicit forecasting "
        "with weather-text context."
    )

    assert cleaned == ""


def test_daily_note_reasoning_drops_still_need_evidence_wording():
    cleaned = Notes._sanitize_candidate_reasoning(
        "We already have global trend evidence showing a strong downtrend, but we still need recent-window "
        "quantitative evidence on late-series weakness to support the classification."
    )

    assert cleaned == ""


def test_normalise_candidate_key_prefers_tool_chain_over_slot_name():
    assert PipelineRunner._normalise_candidate_key("slot_8", ["detect_anomaly"], 0) == "detect_anomaly"


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
async def test_generate_learning_analysis_fallback_strips_ground_truth_lines():
    class _Resp:
        parsed = None

    class _LLM:
        async def complete(self, *_args, **_kwargs):
            return _Resp()

    structured, text = await generate_learning_analysis(
        scope_key="finance_aligned_short_trend",
        task_type="classification",
        domain="finance",
        task_description="trend task",
        sample_fingerprint="length=390",
        candidate_traces=[],
        best_metrics={"accuracy": 0.0, "ground_truth_label": "+2% ~ +4%"},
        fallback_summary=(
            "All candidates converged on -2% ~ -4% while ground truth was +2% ~ +4%.\n"
            "Use recent-window evidence."
        ),
        llm=_LLM(),
    )

    assert structured is None
    assert "ground truth" not in text.lower()
    assert text == "Use recent-window evidence."


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


@pytest.mark.asyncio
async def test_generate_learning_analysis_strips_ground_truth_from_prompt_inputs():
    captured: dict[str, str] = {}

    class _Resp:
        parsed = {
            "text_signals": [],
            "sample_conditions": ["x"],
            "winner_evidence": ["y"],
            "failure_signals": [],
            "reusable_rule": ["When x, do y."],
        }

    class _LLM:
        async def complete(self, messages, **_kwargs):
            captured["user"] = messages[1]["content"]
            return _Resp()

    await generate_learning_analysis(
        scope_key="finance_aligned_short_trend",
        task_type="classification",
        domain="finance",
        task_description="trend task",
        sample_fingerprint="length=390",
        candidate_traces=[
            {
                "name": "slot_0",
                "tool_chain": ["window_stats"],
                "metrics": {
                    "accuracy": 0.0,
                    "predicted_label": "-2% ~ -4%",
                    "ground_truth_label": "+2% ~ +4%",
                },
            }
        ],
        best_metrics={
            "accuracy": 0.0,
            "ground_truth_label": "+2% ~ +4%",
        },
        fallback_summary=(
            "All candidates converged on -2% ~ -4% while ground truth was +2% ~ +4%.\n"
            "Use recent-window evidence."
        ),
        llm=_LLM(),
    )

    user_prompt = captured["user"]
    assert "ground truth" not in user_prompt.lower()
    assert "ground_truth" not in user_prompt.lower()
    assert "Use recent-window evidence." in user_prompt


@pytest.mark.asyncio
async def test_generate_learning_analysis_prompt_treats_metrics_as_evidence_not_cause():
    captured: dict[str, str] = {}

    class _Resp:
        parsed = {
            "text_signals": [],
            "sample_conditions": ["recent regime is flat with low volatility"],
            "winner_evidence": [
                "detect_anomaly matched the flat regime and then showed the lower MAE on this sample"
            ],
            "failure_signals": [],
            "reusable_rule": ["When the regime is flat, consider detect_anomaly first."],
        }

    class _LLM:
        async def complete(self, messages, **_kwargs):
            captured["system"] = messages[0]["content"]
            return _Resp()

    await generate_learning_analysis(
        scope_key="finance_aligned_long_forecast",
        task_type="forecast",
        domain="finance",
        task_description="forecast task",
        sample_fingerprint="horizon=33",
        candidate_traces=[
            {"name": "detect_anomaly", "tool_chain": ["detect_anomaly"], "metrics": {"mae": 0.384}},
            {"name": "value_at", "tool_chain": ["value_at"], "metrics": {"mae": 0.691}},
        ],
        best_metrics={"mae": 0.384},
        fallback_summary="fallback summary",
        llm=_LLM(),
    )

    system_prompt = captured["system"]
    assert "Do not treat better metrics as the cause of success." in system_prompt
    assert "Explain method-data fit before metric outcomes" in system_prompt
    assert 'Avoid bullets like "X won because it had lower error"' in system_prompt


@pytest.mark.asyncio
async def test_generate_learning_analysis_prompt_uses_scope_targets_for_multi_scope_runs():
    captured: dict[str, str] = {}

    class _Resp:
        parsed = {
            "text_signals": [],
            "sample_conditions": ["recent regime is flat"],
            "winner_evidence": ["segment preserved the local regime split"],
            "failure_signals": [],
            "reusable_rule": ["When the regime splits locally, segment first."],
        }

    class _LLM:
        async def complete(self, messages, **_kwargs):
            captured["user"] = messages[1]["content"]
            return _Resp()

    await generate_learning_analysis(
        scope_key="weather_aligned_short_indicator",
        scope_keys=[
            "weather_aligned_short_indicator",
            "weather_aligned_short_indicator_macd",
        ],
        task_type="indicator",
        domain="weather",
        task_description="indicator task",
        sample_fingerprint="horizon=3",
        candidate_traces=[],
        best_metrics={"avg_mae": 0.2},
        fallback_summary="fallback summary",
        llm=_LLM(),
    )

    assert "Scope targets:" in captured["user"]
    assert "weather_aligned_short_indicator_macd" in captured["user"]


@pytest.mark.asyncio
async def test_generate_learning_analysis_rewrites_slot_aliases_and_drops_framework_wording():
    class _Resp:
        parsed = {
            "text_signals": [],
            "sample_conditions": [
                "The branch explicitly requires using the hinted path before finishing.",
                "Recent elevated regime remains relevant.",
            ],
            "winner_evidence": [
                "slot_8 using detect_anomaly had the lowest MAE.",
            ],
            "failure_signals": [
                "slot_6 was flatter than slot_8.",
                "evaluate_batch_against_gt confirmed the winner.",
            ],
            "reusable_rule": [
                "When slot_8 is available, prefer it.",
            ],
        }

    class _LLM:
        async def complete(self, *_args, **_kwargs):
            return _Resp()

    structured, _text = await generate_learning_analysis(
        scope_key="finance_aligned_long_forecast",
        task_type="forecast",
        domain="finance",
        task_description="forecast task",
        sample_fingerprint="horizon=33",
        candidate_traces=[
            {
                "name": "slot_8",
                "candidate_key": "slot_8",
                "tool_chain": ["detect_anomaly"],
                "metrics": {"mae": 0.384},
            },
            {
                "name": "slot_6",
                "candidate_key": "slot_6",
                "tool_chain": ["value_at"],
                "metrics": {"mae": 0.691},
            },
        ],
        best_metrics={"mae": 0.384},
        fallback_summary="fallback summary",
        llm=_LLM(),
    )

    assert structured is not None
    assert structured["sample_conditions"] == ["Recent elevated regime remains relevant"]
    assert structured["winner_evidence"] == ["detect_anomaly had the lowest MAE"]
    assert structured["failure_signals"] == ["value_at was flatter than detect_anomaly"]
    assert structured["reusable_rule"] == ["When detect_anomaly is available, prefer it"]


def test_generate_learning_analysis_sanitizes_framework_preface_but_keeps_evidence():
    cleaned = learning_analysis_mod._sanitize_analysis_item(
        "The required hinted tool was used: entropy returned sample_entropy 0.778 and spectral_entropy 3.599.",
        {},
    )

    assert "required hinted tool" not in cleaned.lower()
    assert cleaned.startswith("entropy returned sample_entropy 0.778")
    assert "spectral_entropy 3.599" in cleaned


def test_generate_learning_analysis_drops_control_flow_meta_item():
    cleaned = learning_analysis_mod._sanitize_analysis_item(
        "The current branch state confirms that the task-valid answer is available from a real tool result and may be returned unchanged as the final classification.",
        {},
    )

    assert cleaned == ""


def test_daily_note_prefers_llm_text_signals_when_present():
    note = Notes._build_note(  # type: ignore[misc]
        Notes.__new__(Notes),
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
    note = Notes._build_note(  # type: ignore[misc]
        Notes.__new__(Notes),
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
    assert "\nAnalysis:\n[ANALYSIS_BEGIN]\nSample Conditions:\n- short-horizon forecast" in note
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
                "winner_evidence": ["chronos2_forecast achieved lower MAE than the compared candidate"],
                "failure_signals": ["ets_forecast flattened the cycle"],
                "reusable_rule": ["When short-horizon with seasonality, consider chronos2_forecast before ets_forecast."],
            },
            "Sample Conditions:\n- short-horizon\n\nWinner Evidence:\n- chronos2_forecast achieved lower MAE than the compared candidate\n\nFailure Signals:\n- ets_forecast flattened the cycle\n\nReusable Rule:\n- When short-horizon with seasonality, consider chronos2_forecast before ets_forecast.",
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
                            "execution_log": [{"tool": "chronos2_forecast"}],
                        },
                    }
                ],
            },
            {
                "tool": "evaluate_batch_against_gt",
                "observation": {
                    "results": [
                        {"name": "chronos2_forecast", "metrics": {"mae": 1.2}},
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
        source_metadata={
            "dataset_name": "weather_aligned_short",
            "sample_ref": "weather_aligned_short:file.json:0:forecast",
            "source_file": "file.json",
            "sample_idx": 0,
            "task_name": "forecast",
        },
        trace_id="trace123abc",
    )

    assert runner.memory.kwargs is not None
    assert "Sample Conditions:" in runner.memory.kwargs["pipeline_summary"]
    assert runner.memory.kwargs["trace_id"] == "trace123abc"
    assert runner.memory.kwargs["source_metadata"]["sample_ref"] == "weather_aligned_short:file.json:0:forecast"
    assert runner.memory.kwargs["analysis_structured"]["winner_evidence"] == [
        "chronos2_forecast achieved lower MAE than the compared candidate"
    ]


def test_train_kwargs_keep_pipeline_eval_in_test_mode():
    sample = type("S", (), {"output_window": [1.0, 2.0], "domain": "weather", "answer": None, "labels": {}})()
    pipeline_eval_mod._record_enabled = True
    kwargs = pipeline_eval_mod._train_kwargs(sample, "forecast")
    assert kwargs == {}


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
                            "execution_log": [{"tool": "chronos2_forecast", "observation": {"forecast": [1.0, 2.0]}}],
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
                        {"name": "chronos2_forecast", "metrics": {"mae": 1.2}},
                        {"name": "arima_forecast", "metrics": {"mae": 2.5}},
                    ]
                },
            },
        ]
    }
    assert pipeline_eval_mod._record_tool_chain(result, {"mae": 1.2}) == ["chronos2_forecast"]


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
                            "execution_log": [{"tool": "chronos2_forecast", "observation": {"forecast": [1.0, 2.0]}}],
                        },
                    },
                    {
                        "success": True,
                        "result": {
                            "answer": [3.0, 4.0],
                            "execution_log": [{"tool": "timesfm2_forecast", "observation": {"forecast": [3.0, 4.0]}}],
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
                "tool": "chronos2_forecast",
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

    assert len(traces) == 1
    assert traces[0]["name"] == "chronos2_forecast"
    assert traces[0]["tool_chain"] == ["chronos2_forecast", "llm_reasoning_forecast"]
    assert traces[0]["prediction"] == [2.0, 3.0, 4.0]
    assert traces[0]["metrics"] == {"mae": 0.5, "rmse": 0.7}


def test_extract_candidate_traces_keeps_multi_step_top_level_path_intact_before_spawn():
    traces = PipelineRunner._extract_candidate_traces(
        [
            {
                "tool": "segment",
                "thought": "Reason: group the hourly series into daily windows. Action: run segment.",
                "observation": {"segments": [[1, 2], [3, 4]]},
            },
            {
                "tool": "llm_reasoning_classification",
                "thought": "Reason: apply the day-level rule. Action: classify.",
                "observation": {"answer": "decreasing"},
            },
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
                                    "observation": {"answer": "decreasing"},
                                }
                            ],
                        },
                    }
                ],
            },
            {
                "tool": "evaluate_against_gt",
                "args": {"prediction": "decreasing"},
                "observation": {"metrics": {"accuracy": 1.0, "correct": True}},
            },
        ]
    )

    assert len(traces) == 2
    assert traces[0]["tool_chain"] == ["segment", "llm_reasoning_classification"]
    assert traces[0]["candidate_key"] == "segment -> llm_reasoning_classification"
    assert traces[0]["metrics"] == {"accuracy": 1.0, "correct": True}
    assert traces[1]["tool_chain"] == ["llm_reasoning_classification"]


def test_winner_trace_selection_uses_candidate_ranking_not_metrics_identity():
    candidate_traces = [
        {
            "name": "chronos2_forecast",
            "tool_chain": ["chronos2_forecast"],
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
                                "execution_log": [{"tool": "chronos2_forecast"}],
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
                            {"name": "chronos2_forecast", "metrics": {"mae": 2.35, "rmse": 2.91}},
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
                            "execution_log": [{"tool": "chronos2_forecast"}],
                        },
                    },
                    {
                        "success": True,
                        "result": {
                            "answer": "[1.5, 2.5, 3.5]",
                            "execution_log": [{"tool": "timesfm2_forecast"}],
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
                            "execution_log": [{"tool": "chronos2_forecast"}],
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


def test_subagent_made_any_tool_call_requires_real_execution():
    assert ExecutorAgent._subagent_made_any_tool_call(
        [{"tool": "window_stats"}]
    ) is True
    assert ExecutorAgent._subagent_made_any_tool_call(
        []
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
                    {"name": "chronos2_forecast", "metrics": {"mae": 1.2}},
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


def test_build_subagent_final_answer_contract_section_for_forecast():
    text = ExecutorAgent._build_subagent_final_answer_contract_section(
        "forecast",
        horizon=3,
    )

    assert "JSON numeric array" in text
    assert "at least 3 numbers" in text
    assert "Predicted Temperatures" in text


def test_build_subagent_final_answer_contract_section_for_indicator():
    text = ExecutorAgent._build_subagent_final_answer_contract_section(
        "indicator",
        indicator_output_keys=["upper_bb"],
    )

    assert "exactly these keys: upper_bb" in text
    assert '{"upper_bb": [1.234, 1.234, 1.234]}' in text
    assert "Predicted Prices" in text


def test_daily_note_formats_scored_candidates_and_input_signals():
    note = Notes._build_note(  # type: ignore[misc]
        Notes.__new__(Notes),
        {
            "scope_key": "weather_aligned_short_forecast",
            "domain": "weather",
            "task": "forecast",
            "trace_id": "abc123trace",
            "source_metadata": {
                "dataset_name": "weather_aligned_short",
                "sample_ref": "weather_aligned_short:file.json:5:forecast",
                "source_file": "file.json",
                "sample_idx": 5,
                "task_name": "forecast",
                "train_run_id": "run-123",
            },
            "metrics": {"mae": 1.23, "rmse": 2.34},
            "tool_chain": ["chronos2_forecast"],
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
                    "name": "chronos2_forecast",
                    "tool_chain": ["chronos2_forecast"],
                    "prediction": [1, 2, 3],
                    "metrics": {"mae": 1.23, "rmse": 2.34},
                    "public_reasoning_snippets": ["seasonality is present"],
                    "steps": [
                        {
                            "tool": "chronos2_forecast",
                            "thought": "Reason: seasonality is present. Action: run chronos2_forecast.",
                            "observation": {"forecast": [1, 2, 3]},
                        }
                    ],
                }
            ],
            "pipeline_summary": "Sample Conditions:\n- x",
        },
    )
    assert "Source:\n- Dataset: weather_aligned_short" in note
    assert "- Sample Ref: weather_aligned_short:file.json:5:forecast" in note
    assert "- Train Run ID: run-123" in note
    assert "- Trace ID: abc123trace" in note
    assert "Task Prompt:\n[TASK_PROMPT_BEGIN]" in note
    assert "Your task is to predict the next 1 day(s) of hourly temperature readings (3 values)." in note
    assert "Numeric Time Series:\n[NUMERIC_TIME_SERIES_BEGIN]\nTemperature time series (past 7 days, hourly):\n[1, 2, 3]\n[NUMERIC_TIME_SERIES_END]" in note
    assert "Text Context:" not in note
    assert "Text Signals:" in note
    assert "Flash Flood near station." in note
    assert "Cold front arrived overnight." in note
    assert "chronos2_forecast(mae=1.230, rmse=2.340)" in note
    assert "  Reasoning: Seasonality is present." in note
    assert "Observation:" not in note
    assert "Generation Steps:" not in note
    assert "  Prediction:" not in note


def test_daily_note_sanitizes_storage_breaking_control_bytes():
    note = Notes._build_note(  # type: ignore[misc]
        Notes.__new__(Notes),
        {
            "scope_key": "weather_aligned_short_trend_past",
            "domain": "weather",
            "task": "trend_past",
            "pipeline_summary": (
                "Sample Conditions:\n"
                "- daily mean slope thresholds are \x00b10.25 for stable classification"
            ),
        },
    )

    assert "\x00" not in note
    assert "±0.25" in note


def test_daily_note_splits_generation_and_validation_tools():
    note = Notes._build_note(  # type: ignore[misc]
        Notes.__new__(Notes),
        {
            "scope_key": "finance_aligned_long_indicator_upper_bb",
            "tool_chain": ["ets_forecast", "consistency_check"],
            "candidate_traces": [
                {
                    "name": "ets_forecast",
                    "tool_chain": ["ets_forecast", "consistency_check"],
                    "metrics": {"avg_mae": 0.555},
                    "public_reasoning_snippets": [
                        "ets_forecast produced a stable upper bound candidate",
                    ],
                    "steps": [
                        {"tool": "ets_forecast", "thought": "run ets", "observation": {"forecast": [1.0, 2.0]}},
                        {"tool": "consistency_check", "thought": "validate path", "observation": {"consistent": True}},
                    ],
                }
            ],
        },
    )

    assert "Winner Tools:\n- ets_forecast\n- consistency_check" in note
    assert "  Tools: ets_forecast | Validation: consistency_check" in note
    assert "  Reasoning: Ets_forecast produced a stable upper bound candidate." in note
    assert "Generation Steps:" not in note
    assert "Validation Steps:" not in note


def test_daily_note_uses_public_candidate_reasoning_without_truncation():
    long_reason = (
        "Inspecting the most recent post-event regime matters here because the late-sample drift is "
        "the strongest signal in this case and using only a global summary would miss the regime break"
    )
    note = Notes._build_note(  # type: ignore[misc]
        Notes.__new__(Notes),
        {
            "scope_key": "finance_aligned_long_forecast",
            "candidate_traces": [
                {
                    "name": "slice_path",
                    "tool_chain": ["slice", "arima_forecast"],
                    "metrics": {"mae": 1.0, "rmse": 1.2},
                    "public_reasoning_snippets": [long_reason],
                    "steps": [
                        {
                            "tool": "slice",
                            "thought": "run slice first",
                            "observation": {"window": [1, 2, 3]},
                        }
                    ],
                }
            ],
        },
    )

    assert "because the late-sample drift is the strongest signal in this case" in note
    assert "would miss the regime break." in note


def test_daily_note_does_not_recover_reasoning_from_raw_steps_without_public_snippets():
    note = Notes._build_note(  # type: ignore[misc]
        Notes.__new__(Notes),
        {
            "scope_key": "finance_aligned_long_forecast",
            "candidate_traces": [
                {
                    "name": "slice_path",
                    "tool_chain": ["slice", "arima_forecast"],
                    "metrics": {"mae": 1.0, "rmse": 1.2},
                    "steps": [
                        {
                            "tool": "slice",
                            "thought": (
                                "The recent regime break matters more than the global average, so slice should run first."
                            ),
                            "observation": {"window": [1, 2, 3]},
                        }
                    ],
                }
            ],
        },
    )

    assert "Reasoning:" not in note


def test_notes_compact_candidate_reasoning_respects_explicit_empty_public_snippets():
    trace = {
        "public_reasoning_snippets": [],
        "steps": [
            {
                "tool": "llm_reasoning_classification",
                "thought": "The most relevant next step is to classify with the multimodal tool.",
            }
        ],
    }

    assert Notes._compact_candidate_reasoning(trace) == ""


def test_learning_analysis_candidate_prompt_uses_public_reasoning_not_raw_thought():
    prompt = learning_analysis_mod._candidate_prompt_block(
        [
            {
                "name": "llm_reasoning_qa",
                "tool_chain": ["llm_reasoning_qa"],
                "public_reasoning_snippets": [
                    "llm_reasoning_qa matched the timing mismatch in option C",
                ],
                "steps": [
                    {
                        "tool": "llm_reasoning_qa",
                        "thought": "The most relevant next step is to use QA reasoning before finishing.",
                        "observation": {"answer": "C"},
                    }
                ],
            }
        ]
    )

    assert "Execution evidence: llm_reasoning_qa matched the timing mismatch in option C" in prompt
    assert "thought=" not in prompt
    assert "next step" not in prompt


def test_text_signals_rank_sentences_by_task_relevance():
    signals = Notes._extract_text_signals(
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
    signals = Notes._extract_text_signals(
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
            "chronos2_forecast",
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
    assert candidates == ["chronos2_forecast"]


def test_build_subagent_candidates_applies_train_dropout_rules():
    agent = ExecutorAgent(llm=object())
    envelope = TaskEnvelope(
        task=TaskSpec(task_id="t1", task_type="forecast", metadata={}),
        trace_id="trace-1",
    )
    plan = {
        "candidates": [
            "chronos2_forecast",
            "prophet_forecast",
            "llm_reasoning_forecast",
            "spawn_subagent",
            "evaluate_against_gt",
        ],
        "task_type": "forecast",
        "_mode": "train",
        "_tool_dropout_scope": "weather_aligned_short_forecast",
        "_tool_dropout_counts": {
            "chronos2_forecast": 50,
            "prophet_forecast": 0,
            "llm_reasoning_forecast": 30,
        },
    }

    candidates = agent._build_subagent_candidates(
        idx=0,
        task_description="forecast sample",
        tools_hint="prophet_forecast",
        plan=plan,
        envelope=envelope,
    )

    assert "evaluate_against_gt" not in candidates
    assert "spawn_subagent" not in candidates
    assert "llm_reasoning_forecast" in candidates
    assert "prophet_forecast" in candidates


def test_build_profiling_section_drops_fingerprint_snapshot_and_rounds_numbers():
    text = ExecutorAgent._build_profiling_section(
        {
            "modality_report": {"series_mean": 1.23456},
            "fingerprint_snapshot": {"series_mean": 9.87654},
            "compute_statistics": {"series_std": 2.34567},
        }
    )

    assert "fingerprint_snapshot" not in text
    assert "1.235" in text
    assert "2.346" in text


def test_format_observation_for_prompt_rounds_numbers_to_three_decimals():
    text = ExecutorAgent._format_observation_for_prompt(
        {"forecast": [1.23456, 2.34567], "score": 0.98765}
    )

    assert "1.235" in text
    assert "2.346" in text
    assert "0.988" in text


def test_build_llm_reasoning_context_keeps_task_body_but_excludes_tool_history():
    agent = ExecutorAgent(llm=object())
    text = agent._build_llm_reasoning_context(
        plan={
            "task_description": "Official MTBench prompt body",
            "profiling_section": "## Profiling Results\n- detect_trend: {\"direction\": \"up\"}\n",
            "tool_notes_section": "## Tool Notes\n- llm_reasoning_forecast: note\n",
        },
        execution_log=[
            {
                "tool": "window_stats",
                "thought": "Reason: inspect trend. Action: run window_stats.",
                "observation": {"mean": 1.0},
            }
        ],
        existing_context="caller supplied context",
    )

    assert "Official MTBench prompt body" in text
    assert "## Profiling Results" in text
    assert "caller supplied context" in text
    assert "Tool Notes" not in text
    assert "Prior Tool Observations" not in text


def test_indicator_contract_section_requires_exact_keys():
    text = ExecutorAgent._build_indicator_contract_section(["macd"])

    assert "exactly: macd" in text
    assert "_forecast" in text


def test_indicator_final_answer_rejects_wrong_suffix_key():
    valid, reason = ExecutorAgent._validate_final_answer(
        "indicator",
        {"macd_forecast": [1.0, 2.0]},
        indicator_output_keys=["macd"],
    )

    assert valid is False
    assert "must be exactly ['macd']" in reason


def test_coerce_llm_decision_marks_parse_failure_instead_of_fake_done():
    decision = ExecutorAgent._coerce_llm_decision(
        None,
        "not json",
        fallback_thought="LLM parse failed",
    )

    assert decision["done"] is False
    assert decision["_parse_failed"] is True
    assert decision["thought"] == "LLM parse failed"


def test_runner_indicator_output_keys_follow_scope_and_gt():
    assert PipelineRunner._indicator_output_keys(
        ["finance_aligned_long_indicator_macd"],
        "finance",
        "indicator",
    ) == ["macd"]
    assert PipelineRunner._indicator_output_keys(
        ["finance_aligned_long_indicator_upper_bb"],
        "finance",
        "indicator",
    ) == ["upper_bb"]
    assert PipelineRunner._indicator_output_keys(
        ["weather_aligned_long_indicator"],
        "weather",
        "indicator",
    ) == ["max", "min", "diff"]
    assert PipelineRunner._indicator_output_keys(
        ["whatever"],
        "finance",
        "indicator",
        ground_truth={"macd": [1.0]},
    ) == ["macd"]


def test_numeric_postprocess_payload_accepts_structured_numeric_indicator():
    assert PipelineRunner._has_numeric_postprocess_payload(
        "indicator",
        {"macd": [1.0, 2.0]},
    ) is True


def test_numeric_postprocess_payload_rejects_non_numeric_forecast():
    assert PipelineRunner._has_numeric_postprocess_payload(
        "forecast",
        {"answer": "tomorrow will be higher"},
    ) is False


@pytest.mark.asyncio
async def test_required_profiling_tool_missing_raises(monkeypatch):
    runner = PipelineRunner(disable_memory=True)

    def _missing(_name):
        raise KeyError("missing")

    monkeypatch.setattr(runner_mod, "get_tool_instance", _missing)

    with pytest.raises(RuntimeError, match="Required preprocessing tool 'basic_stats' is not registered"):
        await runner._run_profiling_tool(
            "basic_stats",
            [1.0, 2.0, 3.0],
            {},
            required=True,
        )


def test_deferred_winner_selection_raises_without_evaluated_metrics():
    with pytest.raises(ValueError, match="evaluated candidate metrics"):
        PipelineRunner._select_winner_prediction_from_execution_log(
            [
                {
                    "tool": "spawn_subagent",
                    "output": [
                        {
                            "success": True,
                            "result": {
                                "answer": [1.0, 2.0, 3.0],
                                "execution_log": [{"tool": "chronos2_forecast"}],
                            },
                        }
                    ],
                }
            ],
            fallback_prediction=[9.0, 9.0, 9.0],
        )


def test_learning_summary_winner_selection_accepts_explicit_winner_without_metrics():
    winner = PipelineRunner._select_winner_prediction_from_execution_log(
        [
            {
                "tool": "spawn_subagent",
                "output": [
                    {
                        "success": True,
                        "result": {
                            "answer": {"upper_bb": [55.1, 55.2]},
                            "execution_log": [{"tool": "llm_reasoning_indicator"}],
                        },
                    },
                    {
                        "success": True,
                        "result": {
                            "answer": {"upper_bb": [54.2, 54.2]},
                            "execution_log": [{"tool": "arima_forecast"}, {"tool": "llm_reasoning_indicator"}],
                        },
                    },
                ],
            }
        ],
        fallback_prediction={"insight": "unused"},
        raw_answer={
            "winner_tool_chain": ["llm_reasoning_indicator"],
            "insight": "direct indicator path was best",
            "recommendation": "prefer direct indicator path",
        },
        allow_missing_metrics=True,
    )

    assert winner == {"upper_bb": [55.1, 55.2]}


def test_deferred_winner_selection_prefers_normalized_batch_prediction_over_string_answer():
    winner = PipelineRunner._select_winner_prediction_from_execution_log(
        [
            {
                "tool": "spawn_subagent",
                "output": [
                    {
                        "success": True,
                        "result": {
                            "answer": "[1.0, 2.0, 3.0]",
                            "execution_log": [{"tool": "chronos2_forecast"}],
                        },
                    },
                    {
                        "success": True,
                        "result": {
                            "answer": "[4.0, 5.0, 6.0]",
                            "execution_log": [{"tool": "arima_forecast"}],
                        },
                    },
                ],
            },
            {
                "tool": "evaluate_batch_against_gt",
                "observation": {
                    "results": [
                        {
                            "name": "chronos2_forecast",
                            "metrics": {"mae": 2.0, "rmse": 2.5},
                            "prediction": [1.0, 2.0, 3.0],
                        },
                        {
                            "name": "arima_forecast",
                            "metrics": {"mae": 1.0, "rmse": 1.5},
                            "prediction": [4.0, 5.0, 6.0],
                        },
                    ]
                },
            },
        ],
        fallback_prediction={"insight": "should not be used"},
    )

    assert winner == [4.0, 5.0, 6.0]


def test_deferred_winner_selection_matches_single_eval_to_latest_prior_candidate_when_prediction_is_rounded():
    execution_log = [
        {
            "tool": "handle_missing",
            "observation": {"data": [1.0, 2.0, 3.0], "missing_count": 0},
        },
        {
            "tool": "prophet_forecast",
            "observation": {"forecast": [52.7608, 52.5310, 52.3011], "horizon": 3},
        },
        {
            "tool": "sundial_base_128m_forecast",
            "observation": {"point_forecast": [45.2699, 45.2123, 45.1505], "horizon": 3},
        },
        {
            "tool": "prophet_forecast",
            "observation": {"forecast": [47.9130, 47.8115, 47.7100], "horizon": 3},
        },
        {
            "tool": "evaluate_against_gt",
            "args": {"prediction": [47.913, 47.812, 47.71]},
            "observation": {"metrics": {"mae": 2.078, "rmse": 2.357}},
        },
    ]

    winner = PipelineRunner._select_winner_prediction_from_execution_log(
        execution_log,
        fallback_prediction={"insight": "Prophet won"},
        raw_answer={"insight": "Prophet won", "recommendation": "Use prophet."},
    )

    assert winner == [47.9130, 47.8115, 47.7100]


def test_pipeline_eval_forecast_extractor_rejects_free_text_number_scrape():
    assert pipeline_eval_mod._extract_forecast_array(
        "Forecast: 1.0, 2.0, 3.0",
    ) is None


def test_pipeline_eval_indicator_extractor_rejects_regex_scrape():
    assert pipeline_eval_mod._extract_indicator_dict(
        "max=12.3 min=4.5 diff=7.8",
        ["max", "min", "diff"],
    ) is None


def test_pipeline_eval_indicator_array_rejects_non_json_string():
    assert pipeline_eval_mod._extract_indicator_array(
        "macd: [1.0, 2.0]",
        "macd",
    ) is None


@pytest.mark.asyncio
async def test_reasoner_cross_modal_failure_no_longer_passes_on_exception():
    class _LLM:
        async def complete(self, *_args, **_kwargs):
            raise RuntimeError("boom")

    agent = ReasonerAgent(llm=_LLM())
    envelope = TaskEnvelope(
        task=TaskSpec(task_id="t1", task_type="forecast", description="x", metadata={}),
        trace_id="trace-1",
    )

    result = await agent._check_cross_modal_consistency(
        {"timeseries": {"prediction": [1.0]}, "text": {"prediction": [1.0]}},
        envelope,
    )

    assert result["passed"] is False
    assert "failed" in result["details"]


@pytest.mark.asyncio
async def test_reasoner_llm_verify_invalid_json_is_rejected():
    class _Resp:
        parsed = None

    class _LLM:
        async def complete(self, *_args, **_kwargs):
            return _Resp()

    agent = ReasonerAgent(llm=_LLM())
    envelope = TaskEnvelope(
        task=TaskSpec(task_id="t1", task_type="forecast", description="x", metadata={}),
        trace_id="trace-1",
    )
    envelope.environment.set_data([1.0, 2.0, 3.0])

    result = await agent._llm_verify([1.0, 2.0], envelope, [])

    assert result["accepted"] is False
    assert "invalid JSON" in result["reasoning"]


@pytest.mark.asyncio
async def test_reasoner_reanalysis_invalid_json_is_marked_explicitly():
    class _Resp:
        parsed = None

    class _LLM:
        async def complete(self, *_args, **_kwargs):
            return _Resp()

    agent = ReasonerAgent(llm=_LLM())
    envelope = TaskEnvelope(
        task=TaskSpec(task_id="t1", task_type="forecast", description="x", metadata={}),
        trace_id="trace-1",
    )

    result = await agent._decide_reanalysis(
        {"timeseries": {"prediction": [1.0]}, "text": {"prediction": [2.0]}},
        {"passed": False, "details": "conflict"},
        envelope,
    )

    assert result["should_reanalyze"] is False
    assert "invalid JSON" in result["reasoning"]


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
                                        "execution_log": [{"tool": "chronos2_forecast"}],
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
                                    {"name": "chronos2_forecast", "metrics": {"mae": 1.0}},
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
async def test_deferred_train_normalizes_learning_summary_raw_answer(monkeypatch):
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
                    "answer": {
                        "insight": "chronos was best",
                        "recommendation": "prefer chronos",
                    },
                    "answer_type": "learning_summary",
                    "confidence": 0.9,
                    "reasoning": "1. Compared candidates.\n2. Summarized the lesson.",
                    "execution_log": [
                        {
                            "tool": "spawn_subagent",
                            "output": [
                                {
                                    "success": True,
                                    "result": {
                                        "answer": [1.0, 2.0, 3.0],
                                        "execution_log": [{"tool": "chronos2_forecast"}],
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
                                    {"name": "chronos2_forecast", "metrics": {"mae": 1.0}},
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

    assert result["answer_type"] == "learning_summary"
    assert result["raw_answer"]["winner_candidate"] == "chronos2_forecast"
    assert result["raw_answer"]["winner_prediction"] == [1.0, 2.0, 3.0]
    assert result["raw_answer"]["winner_tool_chain"] == ["chronos2_forecast"]


@pytest.mark.asyncio
async def test_run_requires_evaluated_candidates_when_deferred(monkeypatch):
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

    with pytest.raises(ValueError, match="evaluated candidate metrics"):
        await runner.run(
            data=[1.0, 2.0, 3.0],
            task_type="forecast",
            description="forecast sample",
            mode="train",
            ground_truth=[1.0, 2.0],
            defer_learning_record=True,
        )


@pytest.mark.asyncio
async def test_train_learning_summary_without_metrics_is_kept_as_failure_summary(monkeypatch):
    runner = PipelineRunner(disable_memory=True)

    async def _fake_ensure_tools():
        return None

    async def _fake_preprocess(*_args, **_kwargs):
        return {"fingerprint": {}, "modalities": ["timeseries", "text"], "profiling_results": {}}

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
                    "answer": {
                        "insight": "The direct indicator path looked strongest, but GT evaluation was never completed.",
                        "recommendation": "Create multiple task-valid branches earlier and evaluate before budget runs out.",
                        "winner_tool_chain": ["llm_reasoning_indicator"],
                    },
                    "answer_type": "learning_summary",
                    "confidence": 0.97,
                    "reasoning": "1. Multiple candidates were explored.\n2. The run ended without GT evaluation.",
                    "execution_log": [
                        {
                            "tool": "spawn_subagent",
                            "output": [
                                {
                                    "success": True,
                                    "result": {
                                        "answer": {"upper_bb": [55.1, 55.2]},
                                        "execution_log": [{"tool": "llm_reasoning_indicator"}],
                                    },
                                },
                                {
                                    "success": True,
                                    "result": {
                                        "answer": {"upper_bb": [54.2, 54.2]},
                                        "execution_log": [{"tool": "arima_forecast"}, {"tool": "llm_reasoning_indicator"}],
                                    },
                                },
                            ],
                        },
                    ],
                }
            }

    monkeypatch.setattr(runner, "_ensure_tools", _fake_ensure_tools)
    monkeypatch.setattr(runner, "_preprocess", _fake_preprocess)
    monkeypatch.setattr(runner, "_create_executor", lambda: _Executor())

    result = await runner.run(
        data=[1.0, 2.0, 3.0],
        task_type="indicator",
        description="indicator sample",
        mode="train",
        ground_truth={"upper_bb": [55.0, 55.1]},
        defer_learning_record=False,
    )

    assert result["answer_type"] == "learning_summary"
    assert result["prediction"] == {"upper_bb": [55.1, 55.2]}
    assert result["raw_answer"]["winner_tool_chain"] == ["llm_reasoning_indicator"]


@pytest.mark.asyncio
async def test_run_rejects_non_numeric_forecast_payload(monkeypatch):
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
                    "answer": {"answer": "next day will be warmer"},
                    "answer_type": "forecast",
                    "confidence": 0.9,
                    "reasoning": "Reason: x. Action: y.",
                    "execution_log": [],
                }
            }

    monkeypatch.setattr(runner, "_ensure_tools", _fake_ensure_tools)
    monkeypatch.setattr(runner, "_preprocess", _fake_preprocess)
    monkeypatch.setattr(runner, "_create_executor", lambda: _Executor())

    result = await runner.run(
        data=[1.0, 2.0, 3.0],
        task_type="forecast",
        description="forecast sample",
    )

    assert result["gate_accepted"] is False
