"""Tests for ClawHub text tools (Cat-8)."""

import pytest
from timeclaw.clawhub.text.sentiment import SentimentTool
from timeclaw.clawhub.text.ner import NERTool
from timeclaw.clawhub.text.summarize import SummarizeTool
from timeclaw.clawhub.text.keyword_extract import KeywordExtractTool
from timeclaw.clawhub.text.text_similarity import TextSimilarityTool
from timeclaw.clawhub.text.temporal_align_text import TemporalAlignTextTool
from timeclaw.clawhub.text.text_to_event import TextToEventTool


@pytest.mark.asyncio
async def test_sentiment_rule():
    tool = SentimentTool()
    result = await tool.execute(text="This is a great and excellent product")
    assert result.success
    assert result.data["sentiments"][0]["polarity"] == "positive"


@pytest.mark.asyncio
async def test_sentiment_negative():
    tool = SentimentTool()
    result = await tool.execute(text="This is a bad and poor product with decline")
    assert result.success
    assert result.data["sentiments"][0]["polarity"] == "negative"


@pytest.mark.asyncio
async def test_ner_regex():
    tool = NERTool()
    result = await tool.execute(text="On 2024-01-15 at 14:30 the value was $1,000")
    assert result.success
    entities = result.data["entities"][0]
    types = {e["type"] for e in entities}
    assert "DATE" in types or "MONEY" in types


@pytest.mark.asyncio
async def test_summarize_extractive():
    tool = SummarizeTool()
    text = (
        "The system detected an anomaly at 3pm. "
        "Engineers were dispatched immediately. "
        "The fault was traced to bearing B3. "
        "Replacement was completed by 5pm."
    )
    result = await tool.execute(text=text, max_sentences=2)
    assert result.success
    assert len(result.data["summary"]) > 0


@pytest.mark.asyncio
async def test_keyword_extract():
    tool = KeywordExtractTool()
    result = await tool.execute(
        text="machine learning deep learning neural network artificial intelligence",
        top_k=3,
    )
    assert result.success
    assert len(result.data["keywords"]) <= 3


@pytest.mark.asyncio
async def test_text_similarity():
    tool = TextSimilarityTool()
    result = await tool.execute(
        text_a="the cat sat on the mat",
        text_b="the cat sat on the mat",
    )
    assert result.success
    assert result.data["similarity"] > 0.9


@pytest.mark.asyncio
async def test_temporal_align_text():
    import numpy as np

    tool = TemporalAlignTextTool()
    ts = np.random.randn(100)
    text_data = [
        {"timestamp": 10, "text": "event A"},
        {"timestamp": 50, "text": "event B"},
    ]
    result = await tool.execute(ts_data=ts, text=text_data)
    assert result.success
    assert result.data["n_matched"] == 2


@pytest.mark.asyncio
async def test_text_to_event():
    tool = TextToEventTool()
    text = (
        "2024-01-15 Fault detected in motor unit. "
        "2024-01-16 Maintenance crew replaced the bearing."
    )
    result = await tool.execute(text=text)
    assert result.success
    assert result.data["n_events"] >= 1
