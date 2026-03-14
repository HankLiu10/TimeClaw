"""Tests for ExperiencePool with vector search."""

import tempfile
from pathlib import Path

import pytest

from timeclaw.memory.experience_pool import ExperienceEntry, ExperiencePool


@pytest.fixture
def pool():
    with tempfile.TemporaryDirectory() as tmpdir:
        p = ExperiencePool(
            storage_dir=Path(tmpdir) / "exp",
            vector_db_path=Path(tmpdir) / "vec.db",
            enable_vector=True,
        )
        yield p


def test_add_and_search(pool):
    entry = ExperienceEntry(
        trace_id="test-001",
        task_type="forecast",
        task_description="forecast temperature",
        tool_chain=["basic_stats", "arima_forecast"],
        quality_gate_passed=True,
    )
    pool.add(entry)
    assert pool.size == 1

    results = pool.search(task_type="forecast")
    assert len(results) == 1
    assert results[0].trace_id == "test-001"


@pytest.mark.asyncio
async def test_semantic_search(pool):
    entries = [
        ExperienceEntry(
            trace_id="t1",
            task_type="forecast",
            task_description="forecast weather temperature",
            tool_chain=["basic_stats", "arima_forecast"],
        ),
        ExperienceEntry(
            trace_id="t2",
            task_type="qa",
            task_description="answer question about stock trading",
            tool_chain=["sentiment", "keyword_extract"],
        ),
        ExperienceEntry(
            trace_id="t3",
            task_type="forecast",
            task_description="forecast temperature humidity",
            tool_chain=["stl_decompose", "sbm_forecast"],
        ),
    ]
    for e in entries:
        pool.add(e)

    results = await pool.semantic_search("forecast weather temperature prediction", limit=2)
    assert len(results) <= 2
    assert any(r.trace_id in ("t1", "t3") for r in results)


def test_to_search_text():
    entry = ExperienceEntry(
        trace_id="t1",
        task_type="forecast",
        task_description="predict temperature",
        tool_chain=["arima", "stats"],
        prediction_summary="value=25.3",
    )
    text = entry.to_search_text()
    assert "forecast" in text
    assert "predict temperature" in text
    assert "arima stats" in text


def test_clear(pool):
    pool.add(ExperienceEntry(trace_id="t1", task_type="test"))
    assert pool.size == 1
    pool.clear()
    assert pool.size == 0
