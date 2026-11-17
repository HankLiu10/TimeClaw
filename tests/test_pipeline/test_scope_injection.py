from types import SimpleNamespace

import numpy as np
import pytest

from timeclaw.benchmark.dataset_split import load_split
from timeclaw.benchmark.mtbench_adapter import MTBenchSample
from timeclaw.benchmark.mtbench_support import (
    build_task_description,
    prepare_mtbench_run_kwargs,
    prepare_task_run_kwargs,
)
from timeclaw.agents.prompts import executor_prompt
from timeclaw.clawhub.text.read_raw_text import ReadRawTextTool
from timeclaw.core.config import TimeClawConfig
from timeclaw.core.retry_control import RetryLater, deferred_retry_scope
from timeclaw.pipeline.runner import PipelineRunner
from timeclaw.text import task_pruning as task_pruning_mod


def _pruning_text_response(cleaned_text: str, removed_summary: list[str] | None = None) -> str:
    lines = ["CLEANED_TEXT:", cleaned_text.strip(), "", "REMOVED_SUMMARY:"]
    items = [str(item).strip() for item in (removed_summary or []) if str(item).strip()]
    if items:
        lines.extend(f"- {item}" for item in items)
    else:
        lines.append("- none")
    return "\n".join(lines)


@pytest.mark.asyncio
async def test_text_pruning_does_not_retry_provider_content_rejection(tmp_path, monkeypatch):
    from timeclaw.llm.base import ProviderContentPolicyError

    monkeypatch.setattr(task_pruning_mod, "RAW_TEXT_DIR", tmp_path / "raw")
    monkeypatch.setattr(task_pruning_mod, "CLEANED_TEXT_DIR", tmp_path / "cleaned")
    sample = MTBenchSample(
        file="rejected.json", dataset="finance_qa_short", domain="finance",
        task_types=["correlation"], input_window=np.asarray([100.0, 102.0, 98.0]),
        output_window=None, text="A financial news article.",
        timestamps=[1711929600.0, 1712016000.0, 1712102400.0], start_year=2024,
    )

    class RejectedLLM:
        calls = 0

        async def complete(self, *_args, **_kwargs):
            self.calls += 1
            raise ProviderContentPolicyError("Provider content policy rejection (code 1301): connection closed")

    llm = RejectedLLM()
    with pytest.raises(ProviderContentPolicyError, match="1301"):
        await task_pruning_mod.prepare_task_text(sample, "correlation", llm=llm)
    assert llm.calls == 1
    assert not list((tmp_path / "cleaned").glob("*.json"))


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

    def fake_get_skill_metadata(domain="", task="", scope_keys=None):
        seen["skill_metadata_scope_keys"] = scope_keys
        return {"scopes": [{"scope": "finance_aligned_short_indicator_macd", "skills": []}]}

    monkeypatch.setattr(runner, "_ensure_tools", fake_ensure_tools)
    monkeypatch.setattr(runner, "_preprocess", fake_preprocess)
    monkeypatch.setattr(runner, "_create_executor", fake_create_executor)
    monkeypatch.setattr(runner._quality_gate, "check", lambda prediction, evidence, constraints: SimpleNamespace(accepted=True, rejections=[]))
    monkeypatch.setattr(runner.memory, "get_tool_notes", fake_get_tool_notes)
    monkeypatch.setattr(runner.memory, "get_skill", fake_get_skill)
    monkeypatch.setattr(runner.memory, "get_skill_metadata", fake_get_skill_metadata)

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
    assert seen["skill_metadata_scope_keys"] == [
        "finance_aligned_short_indicator_macd",
        "finance_aligned_short_indicator_upper_bb",
    ]
    assert fake_executor.seen_metadata["memory_l1"] == ""
    assert fake_executor.seen_metadata["tool_notes"] == {"compute_macd": "scope aware"}
    assert fake_executor.seen_metadata["skill_prompt"] == "skill text"
    assert fake_executor.seen_metadata["skill_metadata"] == {
        "scopes": [{"scope": "finance_aligned_short_indicator_macd", "skills": []}]
    }


def test_modality_filtered_tool_names_respect_required_modalities():
    from timeclaw.clawhub.tool_registry import discover_and_register, get_tool_instance

    discover_and_register()
    text_tools = set(executor_prompt.get_all_tool_names(["timeseries", "text"]))
    ts_only_tools = set(executor_prompt.get_all_tool_names(["timeseries"]))

    assert "llm_reasoning_forecast" in text_tools
    assert "llm_reasoning_forecast" in ts_only_tools
    assert "llm_reasoning_indicator" in ts_only_tools
    assert "temporal_align_text" in text_tools
    assert "temporal_align_text" not in ts_only_tools
    if get_tool_instance("prophet_forecast").is_stub:
        assert "prophet_forecast" not in text_tools


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
    assert "source_ref<string>" in catalog
    assert "`original_input`" in catalog
    assert "- spawn_subagent:" in catalog
    assert "tasks<array> required" in catalog
    assert "- evaluate_against_gt:" in catalog
    assert "task_type<string> required" in catalog
    assert "- evaluate_batch_against_gt:" in catalog
    assert "framework will auto-fill them" in catalog


def test_executor_tool_catalog_renders_enum_choices():
    from timeclaw.clawhub.tool_registry import discover_and_register

    discover_and_register()
    catalog = executor_prompt.build_tool_name_text(["temporal_align_text"])

    assert "- temporal_align_text:" in catalog
    assert "Allowed values: `nearest`, `window`, `overlap`" in catalog


def test_executor_tool_catalog_includes_structured_input_contract_for_granger():
    from timeclaw.clawhub.tool_registry import discover_and_register

    discover_and_register()
    catalog = executor_prompt.build_tool_name_text(["granger_causality"])

    assert "Current-state preconditions:" in catalog
    assert "min_series_channels=2" in catalog


def test_build_task_description_uses_full_prompt_template():
    sample = MTBenchSample(file="weather.json", dataset="weather_aligned_short", domain="weather",
        task_types=["forecast"], input_window=np.array([20.,21.,22.]), output_window=None,
        text="Forecast bulletin", timestamps=[0.,3600.,7200.], start_year=2024, forecast_horizon=24)
    description = build_task_description(sample, "forecast")

    assert description.startswith("User prompt:\nYou are a weather forecasting AI.")
    assert "The input time series represents temperature readings" in description
    assert "Predicted Temperatures: value1, value2, ..., valueN." in description


def test_build_task_description_finance_indicator_tasks_are_split():
    sample = MTBenchSample(file="finance.parquet:row0", dataset="finance_aligned_long", domain="finance",
        task_types=["indicator"], input_window=np.array([100.,101.,102.]), output_window=None,
        text="Market bulletin", timestamps=[0.,86400.,172800.], start_year=2024, forecast_horizon=7,
        metadata={"output_timestamps": [259200.,345600.], "sticker": "ACME"})
    macd_description = build_task_description(sample, "indicator_macd")
    bb_description = build_task_description(sample, "indicator_upper_bb")

    assert "Predict the future Moving Average Convergence Divergence (MACD) values" in macd_description
    assert "estimated upper Bollinger Band (BB) values values" not in macd_description
    assert "estimated upper Bollinger Band (BB) values values" not in bb_description
    assert "estimated upper Bollinger Band (BB) values" in bb_description
    assert "Predict the future Moving Average Convergence Divergence (MACD) values" not in bb_description


def test_task_pruning_prompt_does_not_inline_source_text_twice():
    sample = MTBenchSample(
        file="sample.json",
        dataset="finance_qa_long",
        domain="finance",
        task_types=["correlation", "mcqa"],
        input_window=np.asarray([100.0, 101.5, 99.0], dtype=np.float64),
        output_window=np.asarray([98.5, 97.0], dtype=np.float64),
        text=(
            "UNIQUE_LONG_SOURCE_TEXT "
            "This is the source article body that should be supplied separately to pruning."
        ),
        timestamps=[1711929600.0, 1712016000.0, 1712102400.0],
        start_year=2024,
        question="What is the best answer?",
        answer="A",
        labels={},
        metadata={"published_utc": "2024-04-03T09:30:00Z", "sticker": "ACME", "output_timestamps": [1712188800.]}, forecast_horizon=1,
    )

    prompt = task_pruning_mod._task_prompt_text(sample, "correlation")

    assert "UNIQUE_LONG_SOURCE_TEXT" not in prompt
    assert "[SOURCE_TEXT_PROVIDED_SEPARATELY_FOR_TASK_PRUNING]" in prompt


@pytest.mark.asyncio
@pytest.mark.parametrize("configured_timeout", [180, 3600])
async def test_build_task_description_uses_cleaned_text_and_exposes_raw_text_ref(tmp_path, monkeypatch, configured_timeout):
    async def no_sleep(_seconds):
        return None

    monkeypatch.setattr(task_pruning_mod.asyncio, "sleep", no_sleep)
    monkeypatch.setattr(task_pruning_mod, "TEXT_ARTIFACT_ROOT", tmp_path / "text_artifacts")
    monkeypatch.setattr(task_pruning_mod, "RAW_TEXT_DIR", tmp_path / "text_artifacts" / "raw")
    monkeypatch.setattr(task_pruning_mod, "CLEANED_TEXT_DIR", tmp_path / "text_artifacts" / "cleaned")

    sample = MTBenchSample(
        file="sample.json",
        dataset="finance_qa_short",
        domain="finance",
        task_types=["correlation"],
        input_window=np.asarray([100.0, 102.0, 98.0], dtype=np.float64),
        output_window=None,
        text=(
            "We use cookies to improve your experience. "
            "Revenue guidance was cut after weaker demand in the cloud segment. "
            "Executives said margins may recover next quarter."
        ),
        timestamps=[1711929600.0, 1712016000.0, 1712102400.0],
        start_year=2024,
        labels={},
        metadata={"published_utc": "2024-04-03T09:30:00Z", "sticker": "ACME", "output_timestamps": [1712188800.]}, forecast_horizon=1,
    )

    class _Resp:
        text = _pruning_text_response(
            (
                "Revenue guidance was cut after weaker demand in the cloud segment. "
                "Executives said margins may recover next quarter."
            )
        )

    class _LLM:
        async def complete(self, *_args, **_kwargs):
            # Helper calls must honor the configured backend timeout rather
            # than silently overriding it with a fixed 120/300-second limit.
            assert _kwargs.get("timeout", configured_timeout) == configured_timeout
            assert _kwargs.get("first_token_timeout", configured_timeout) == configured_timeout
            return _Resp()

    view = await task_pruning_mod.prepare_task_text(sample, "correlation", llm=_LLM())
    description = build_task_description(sample, "correlation")

    assert "Revenue guidance was cut after weaker demand in the cloud segment." in description
    assert "We use cookies to improve your experience." not in description
    assert "read_raw_text" in description
    assert view["raw_text_ref"] in description


@pytest.mark.asyncio
async def test_prepare_task_run_kwargs_aligns_text_prompt_and_scope_keys(tmp_path, monkeypatch):
    monkeypatch.setattr(task_pruning_mod, "TEXT_ARTIFACT_ROOT", tmp_path / "text_artifacts")
    monkeypatch.setattr(task_pruning_mod, "RAW_TEXT_DIR", tmp_path / "text_artifacts" / "raw")
    monkeypatch.setattr(task_pruning_mod, "CLEANED_TEXT_DIR", tmp_path / "text_artifacts" / "cleaned")

    sample = MTBenchSample(
        file="sample.json",
        dataset="finance_qa_short",
        domain="finance",
        task_types=["correlation"],
        input_window=np.asarray([100.0, 102.0, 98.0], dtype=np.float64),
        output_window=None,
        text=(
            "Subscribe now for more market coverage. "
            "Revenue guidance was cut after weaker demand in the cloud segment."
        ),
        timestamps=[1711929600.0, 1712016000.0, 1712102400.0],
        start_year=2024,
        labels={"correlation": "Strong Negative Correlation"},
        metadata={"published_utc": "2024-04-03T09:30:00Z", "sticker": "ACME", "output_timestamps": [1712188800.]}, forecast_horizon=1,
    )

    class _Resp:
        text = _pruning_text_response(
            "Revenue guidance was cut after weaker demand in the cloud segment.",
            ["subscription banner"],
        )

    class _LLM:
        async def complete(self, *_args, **_kwargs):
            return _Resp()

    class _Router:
        def get_backend(self, name):
            assert name == "text_heavy"
            return _LLM()

    pipeline = SimpleNamespace(router=_Router())
    kwargs = await prepare_task_run_kwargs(
        sample,
        pipeline,
        "correlation",
        source_metadata={"sample_ref": "finance_qa_short:sample.json:0:correlation"},
    )

    assert kwargs["task_type"] == "correlation"
    assert kwargs["scope_keys"] == ["finance_qa_short_correlation"]
    assert kwargs["text_data"][0]["content"] == "Revenue guidance was cut after weaker demand in the cloud segment."
    assert kwargs["text_data"][0]["timestamp"] == 2
    assert kwargs["text_data"][0]["timestamp_raw"] == "2024-04-03T09:30:00Z"
    assert kwargs["source_metadata"]["pipeline_task"] == "correlation"
    assert kwargs["source_metadata"]["dataset_name"] == "finance_qa_short"
    assert "read_raw_text" in kwargs["description"]

    tool = ReadRawTextTool()
    result = await tool.execute(
        None,
        raw_text_ref=kwargs["source_metadata"]["raw_text_ref"],
        max_chars=500,
    )
    assert result.success is True
    assert "Subscribe now for more market coverage." in result.data["text"]


@pytest.mark.asyncio
async def test_prepare_mtbench_run_kwargs_is_backward_compatible_alias(tmp_path, monkeypatch):
    monkeypatch.setattr(task_pruning_mod, "TEXT_ARTIFACT_ROOT", tmp_path / "text_artifacts")
    monkeypatch.setattr(task_pruning_mod, "RAW_TEXT_DIR", tmp_path / "text_artifacts" / "raw")
    monkeypatch.setattr(task_pruning_mod, "CLEANED_TEXT_DIR", tmp_path / "text_artifacts" / "cleaned")

    sample = MTBenchSample(
        file="sample.json",
        dataset="weather_aligned_short",
        domain="weather",
        task_types=["forecast"],
        input_window=np.asarray([10.0, 11.0, 12.0], dtype=np.float64),
        output_window=np.asarray([13.0], dtype=np.float64),
        text="Weather bulletin: temperatures remain mild.",
        timestamps=[1711929600.0, 1712016000.0, 1712102400.0],
        start_year=2024,
        labels={},
        metadata={},
    )

    class _Resp:
        text = _pruning_text_response("Weather bulletin: temperatures remain mild.")

    class _LLM:
        async def complete(self, *_args, **_kwargs):
            return _Resp()

    class _Router:
        def get_backend(self, _name):
            return _LLM()

    pipeline = SimpleNamespace(router=_Router())
    old_kwargs = await prepare_mtbench_run_kwargs(sample, pipeline, "forecast")
    new_kwargs = await prepare_task_run_kwargs(sample, pipeline, "forecast")

    assert old_kwargs == new_kwargs


def test_task_text_data_uses_timestamp_ms_fallback():
    sample = MTBenchSample(
        file="sample.json",
        dataset="finance_qa_long",
        domain="finance",
        task_types=["correlation"],
        input_window=np.asarray([100.0, 102.0, 98.0], dtype=np.float64),
        output_window=None,
        text="Article body.",
        timestamps=[1711929600.0, 1712016000.0, 1712102400.0],
        start_year=2024,
        labels={},
        metadata={"timestamp_ms": 1712102400000, "sticker": "ACME"},
    )

    text_data = task_pruning_mod.task_text_data(sample, "correlation")

    assert text_data is not None
    assert text_data[0]["timestamp"] == 2
    assert text_data[0]["timestamp_raw"] == "1712102400000"


@pytest.mark.asyncio
async def test_prepare_task_text_retries_direct_retryable_exception(tmp_path, monkeypatch):
    monkeypatch.setattr(task_pruning_mod, "TEXT_ARTIFACT_ROOT", tmp_path / "text_artifacts")
    monkeypatch.setattr(task_pruning_mod, "RAW_TEXT_DIR", tmp_path / "text_artifacts" / "raw")
    monkeypatch.setattr(task_pruning_mod, "CLEANED_TEXT_DIR", tmp_path / "text_artifacts" / "cleaned")
    task_pruning_mod._TEXT_PRUNING_DEFERRED_RETRY_STATE.clear()

    sample = MTBenchSample(
        file="sample.json",
        dataset="finance_qa_long",
        domain="finance",
        task_types=["correlation"],
        input_window=np.asarray([100.0, 101.0, 102.0], dtype=np.float64),
        output_window=None,
        text="Revenue declined after weaker enterprise demand. Margin pressure intensified.",
        timestamps=[1711929600.0, 1712016000.0, 1712102400.0],
        start_year=2024,
        labels={},
        metadata={"published_utc": "2024-04-03T09:30:00Z", "sticker": "ACME", "output_timestamps": [1712188800.]}, forecast_horizon=1,
    )

    class _Resp:
        text = _pruning_text_response(
            "Revenue declined after weaker enterprise demand. Margin pressure intensified."
        )

    class _LLM:
        def __init__(self):
            self.calls = 0

        async def complete(self, *_args, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("Server disconnected without sending a response.")
            return _Resp()

    llm = _LLM()
    with deferred_retry_scope("train:finance_qa_long:8:correlation"):
        with pytest.raises(RetryLater):
            await task_pruning_mod.prepare_task_text(sample, "correlation", llm=llm)
    with deferred_retry_scope("train:finance_qa_long:8:correlation"):
        view = await task_pruning_mod.prepare_task_text(sample, "correlation", llm=llm)

    assert llm.calls == 2
    assert view["cleaned_text"].startswith("Revenue declined")
    assert view["raw_text_ref"]
    assert view["cleaned_text_ref"]


@pytest.mark.asyncio
async def test_prepare_task_text_long_text_uses_full_text_pruning(tmp_path, monkeypatch):
    monkeypatch.setattr(task_pruning_mod, "TEXT_ARTIFACT_ROOT", tmp_path / "text_artifacts")
    monkeypatch.setattr(task_pruning_mod, "RAW_TEXT_DIR", tmp_path / "text_artifacts" / "raw")
    monkeypatch.setattr(task_pruning_mod, "CLEANED_TEXT_DIR", tmp_path / "text_artifacts" / "cleaned")

    sample = MTBenchSample(
        file="sample.json",
        dataset="finance_qa_long",
        domain="finance",
        task_types=["correlation", "mcqa"],
        input_window=np.asarray([100.0, 101.0, 102.0], dtype=np.float64),
        output_window=None,
        text=(
            "Title: Example Corp Earnings Call Transcript\n"
            " Content: ["
            "'Publisher promo paragraph.', "
            "'Revenue grew 15% year over year.', "
            "'Margins improved after lower logistics costs.', "
            "'Publisher footer paragraph.'"
            "]"
        ),
        timestamps=[1711929600.0, 1712016000.0, 1712102400.0],
        start_year=2024,
        labels={},
        metadata={"published_utc": "2024-04-03T09:30:00Z", "sticker": "ACME", "output_timestamps": [1712188800.]}, forecast_horizon=1,
    )

    class _Resp:
        text = _pruning_text_response(
            (
                "Example Corp Earnings Call Transcript\n\n"
                "Revenue grew 15% year over year.\n\n"
                "Margins improved after lower logistics costs."
            ),
            ["publisher promo paragraph", "publisher footer paragraph"],
        )

    class _LLM:
        def __init__(self):
            self.calls = []

        async def complete(self, messages, **_kwargs):
            self.calls.append(messages)
            return _Resp()

    llm = _LLM()
    view = await task_pruning_mod.prepare_task_text(sample, "correlation", llm=llm)

    assert len(llm.calls) == 1
    assert "CLEANED_TEXT:" in llm.calls[0][0]["content"]
    assert "Original text:" in llm.calls[0][1]["content"]
    assert "Publisher promo paragraph." not in view["cleaned_text"]
    assert "Revenue grew 15% year over year." in view["cleaned_text"]
    assert "Margins improved after lower logistics costs." in view["cleaned_text"]
    assert view["removed_summary"] == [
        "publisher promo paragraph",
        "publisher footer paragraph",
    ]


@pytest.mark.asyncio
async def test_prepare_task_text_falls_back_to_original_text_after_incomplete(tmp_path, monkeypatch):
    monkeypatch.setattr(task_pruning_mod, "TEXT_ARTIFACT_ROOT", tmp_path / "text_artifacts")
    monkeypatch.setattr(task_pruning_mod, "RAW_TEXT_DIR", tmp_path / "text_artifacts" / "raw")
    monkeypatch.setattr(task_pruning_mod, "CLEANED_TEXT_DIR", tmp_path / "text_artifacts" / "cleaned")

    async def _no_sleep(_delay):
        return None

    monkeypatch.setattr(task_pruning_mod.asyncio, "sleep", _no_sleep)

    sample = MTBenchSample(
        file="sample.json",
        dataset="finance_aligned_long",
        domain="finance",
        task_types=["trend"],
        input_window=np.asarray([100.0, 101.0, 102.0], dtype=np.float64),
        output_window=None,
        text="Headline.\n\nCore evidence remains intact.",
        timestamps=[1711929600.0, 1712016000.0, 1712102400.0],
        start_year=2024,
        labels={},
        metadata={"published_utc": "2024-04-03T09:30:00Z", "sticker": "ACME", "output_timestamps": [1712188800.]}, forecast_horizon=1,
    )

    class _LLM:
        def __init__(self):
            self.calls = 0

        async def complete(self, *_args, **_kwargs):
            self.calls += 1
            raise RuntimeError("incomplete")

    llm = _LLM()
    view = await task_pruning_mod.prepare_task_text(sample, "trend", llm=llm)

    assert llm.calls == task_pruning_mod._TEXT_PRUNING_MAX_RETRY_ATTEMPTS
    assert view["cleaned_text"] == "Headline.\n\nCore evidence remains intact."
    assert view["removed_summary"] == []


@pytest.mark.asyncio
async def test_prepare_task_text_falls_back_to_original_text_after_parser_empty_terminal_response(tmp_path, monkeypatch):
    monkeypatch.setattr(task_pruning_mod, "TEXT_ARTIFACT_ROOT", tmp_path / "text_artifacts")
    monkeypatch.setattr(task_pruning_mod, "RAW_TEXT_DIR", tmp_path / "text_artifacts" / "raw")
    monkeypatch.setattr(task_pruning_mod, "CLEANED_TEXT_DIR", tmp_path / "text_artifacts" / "cleaned")

    async def _no_sleep(_delay):
        return None

    monkeypatch.setattr(task_pruning_mod.asyncio, "sleep", _no_sleep)

    sample = MTBenchSample(
        file="sample.json",
        dataset="finance_aligned_long",
        domain="finance",
        task_types=["forecast"],
        input_window=np.asarray([100.0, 101.0, 102.0], dtype=np.float64),
        output_window=None,
        text="Publisher wrapper.\n\nCore evidence remains intact.",
        timestamps=[1711929600.0, 1712016000.0, 1712102400.0],
        start_year=2024,
        labels={},
        metadata={"published_utc": "2024-04-03T09:30:00Z", "sticker": "ACME", "output_timestamps": [1712188800.]}, forecast_horizon=1,
    )

    class _Resp:
        text = "REMOVED_SUMMARY:\n- none"

    class _LLM:
        def __init__(self):
            self.calls = 0

        async def complete(self, *_args, **_kwargs):
            self.calls += 1
            return _Resp()

    llm = _LLM()
    view = await task_pruning_mod.prepare_task_text(sample, "forecast", llm=llm)

    assert llm.calls == task_pruning_mod._TEXT_PRUNING_MAX_RETRY_ATTEMPTS
    assert view["cleaned_text"] == "Publisher wrapper.\n\nCore evidence remains intact."
    assert view["removed_summary"] == []


def test_prepare_pruning_source_text_unwraps_title_content_wrapper():
    prepared = task_pruning_mod._prepare_pruning_source_text(
        "Title: Example Corp Earnings Call Transcript\n"
        " Content: ['Publisher promo paragraph.', 'Revenue grew 15% year over year.', "
        "'Margins improved after lower logistics costs.', 'Margins improved after lower logistics costs.']"
    )

    assert "Title:" not in prepared
    assert "Content:" not in prepared
    assert "Example Corp Earnings Call Transcript" in prepared
    assert "Revenue grew 15% year over year." in prepared
    assert prepared.count("Margins improved after lower logistics costs.") == 1


def test_prepare_pruning_source_text_removes_article_content_marker_and_adjacent_duplicates():
    prepared = task_pruning_mod._prepare_pruning_source_text(
        "Headline\n\nArticle Content:\n\nSource:\nSource:\n\n"
        "Revenue declined.\n\nRevenue declined.\n\nMargins fell."
    )

    assert "Article Content:" not in prepared
    assert "Source:\nSource:" not in prepared
    assert prepared.count("Revenue declined.") == 1


def test_prepare_pruning_source_text_normalizes_transport_fragile_typography():
    prepared = task_pruning_mod._prepare_pruning_source_text(
        "Roberts’ next role is uncertain — she said, “The Jump” was valuable.\n"
        "The prestigious\xa0award remained in place."
    )

    assert "’" not in prepared
    assert "—" not in prepared
    assert "“" not in prepared
    assert "”" not in prepared
    assert "\xa0" not in prepared
    assert "Roberts'" in prepared
    assert '"The Jump"' in prepared
    assert "prestigious award" in prepared


def test_prepare_pruning_source_text_normalizes_dollar_amounts_for_transport():
    prepared = task_pruning_mod._prepare_pruning_source_text(
        "Legacy plan price rises by up to $6 monthly and up to $12 for family tiers."
    )

    assert "$6" not in prepared
    assert "$12" not in prepared
    assert "USD 6" in prepared
    assert "USD 12" in prepared


def test_prepare_pruning_source_text_unwraps_title_content_before_typography_normalization():
    prepared = task_pruning_mod._prepare_pruning_source_text(
        "Title: ESPN Drops Rachels Nichols From NBA Programming \n"
        " Content: ['ESPN dropped Rachel Nichols.', "
        "\"What Happened: David Roberts, ESPN's senior vice president, issued a statement.\", "
        "'“We mutually agreed that this approach was best for all concerned,” Roberts said.']"
    )

    assert "Title:" not in prepared
    assert "Content:" not in prepared
    assert "[" not in prepared
    assert "]" not in prepared
    assert "Roberts said." in prepared
    assert "ESPN's" in prepared


def test_prepare_pruning_source_text_strips_newswire_transport_wrappers():
    prepared = task_pruning_mod._prepare_pruning_source_text(
        "ACME Deal Update\n"
        "Published on: 2022-03-28T11:30:00Z\n\n"
        "Article Content:\n\n"
        "March 28, 2022 07:30 ET\n\n"
        "| Source:\n"
        "HP Inc.\n\n"
        "PALO ALTO, Calif. (GLOBE NEWSWIRE) -- HP announced a deal to acquire Poly."
    )

    assert "Published on:" not in prepared
    assert "| Source:" not in prepared
    assert "GLOBE NEWSWIRE" not in prepared
    assert "HP announced a deal to acquire Poly." in prepared


def test_prepare_pruning_source_text_strips_transport_fragile_glossary_lines():
    prepared = task_pruning_mod._prepare_pruning_source_text(
        "Market Update\n"
        "Reported positions reflect the total number of shares sold short across venues.\n"
        "A short sale is generally understood to mean the sale of a security that the seller does not own.\n"
        "Core evidence line remains available."
    )

    assert "reflect the total number of shares sold short" not in prepared
    assert "is generally understood to mean" not in prepared
    assert "Core evidence line remains available." in prepared


def test_prepare_pruning_source_text_strips_lines_with_long_direct_quotes():
    long_quote = '"' + ("quoted evidence " * 12).strip() + '"'
    prepared = task_pruning_mod._prepare_pruning_source_text(
        "Headline\n"
        f"Speaker said {long_quote} before the meeting ended.\n"
        '"Short quote stays."\n'
        "Context line remains available."
    )

    assert "quoted evidence" not in prepared
    assert "Short quote stays." in prepared
    assert "Context line remains available." in prepared
