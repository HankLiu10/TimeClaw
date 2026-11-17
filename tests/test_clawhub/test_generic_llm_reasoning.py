import pytest

from timeclaw.clawhub.generic_llm.llm_reasoning import (
    LLMReasoningClassificationTool,
    LLMReasoningForecastTool,
    LLMReasoningIndicatorTool,
)


@pytest.mark.asyncio
async def test_llm_reasoning_tool_raises_on_non_json_text():
    class _Resp:
        parsed = None
        text = "forecast is [1, 2, 3]"

    class _LLM:
        async def complete(self, *_args, **_kwargs):
            return _Resp()

    tool = LLMReasoningForecastTool()

    with pytest.raises(ValueError, match="non-JSON"):
        await tool._run(ts_data=[1.0, 2.0], question="predict", llm=_LLM())


@pytest.mark.asyncio
async def test_llm_reasoning_forecast_normalises_fixed_answer_payload():
    class _Resp:
        parsed = {"forecast": [1.0, 2.0, 3.0]}
        text = ""

    class _LLM:
        async def complete(self, *_args, **_kwargs):
            return _Resp()

    tool = LLMReasoningForecastTool()
    result = await tool._run(ts_data=[1.0, 2.0], question="predict", llm=_LLM())
    assert result == {"answer": [1.0, 2.0, 3.0]}


@pytest.mark.asyncio
async def test_llm_reasoning_indicator_normalises_fixed_answer_payload():
    class _Resp:
        parsed = {"upper_bb": [1.0, 2.0]}
        text = ""

    class _LLM:
        async def complete(self, *_args, **_kwargs):
            return _Resp()

    tool = LLMReasoningIndicatorTool()
    result = await tool._run(ts_data=[1.0, 2.0], question="predict", llm=_LLM())
    assert result == {"answer": {"upper_bb": [1.0, 2.0]}}


@pytest.mark.asyncio
async def test_llm_reasoning_indicator_renames_indicator_name_from_question_context():
    class _Resp:
        parsed = {"indicator_name": [0.1, 0.2]}
        text = ""

    class _LLM:
        async def complete(self, *_args, **_kwargs):
            return _Resp()

    tool = LLMReasoningIndicatorTool()
    result = await tool._run(
        ts_data=[1.0, 2.0],
        question="Predict the next 2 MACD values.",
        llm=_LLM(),
    )
    assert result == {"answer": {"macd": [0.1, 0.2]}}


@pytest.mark.asyncio
async def test_llm_reasoning_indicator_normalises_upper_bollinger_band_alias():
    class _Resp:
        parsed = {"upper_bollinger_band": [1.0, 2.0]}
        text = ""

    class _LLM:
        async def complete(self, *_args, **_kwargs):
            return _Resp()

    tool = LLMReasoningIndicatorTool()
    result = await tool._run(
        ts_data=[1.0, 2.0],
        question="Predict the next 2 upper_bb values.",
        llm=_LLM(),
    )
    assert result == {"answer": {"upper_bb": [1.0, 2.0]}}


@pytest.mark.asyncio
async def test_llm_reasoning_classification_normalises_label_fields():
    class _Resp:
        parsed = {"trend": "decreasing"}
        text = ""

    class _LLM:
        async def complete(self, *_args, **_kwargs):
            return _Resp()

    tool = LLMReasoningClassificationTool()
    result = await tool._run(ts_data=[1.0, 2.0], question="predict", llm=_LLM())
    assert result == {"answer": "decreasing"}
