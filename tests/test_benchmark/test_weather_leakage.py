"""Regression checks for the observed independent-weather target disclosure."""
import importlib.util
import inspect
import json
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from timeclaw.benchmark.mtbench_adapter import _load_finance_json, _load_weather_json, load_independent_dataset
from timeclaw.core.config import TimeClawConfig

ROOT = Path(__file__).resolve().parents[2]






def test_actual_disclosure_rejected_by_weather_adapter(tmp_path):
    record = {
        "text": "Temperature analysis (°F, 24-hour output window beginning at the event anchor): minimum 18.9, maximum 25.0, mean 22.4.",
        "input_window": [20.0], "output_window": [18.9, 25.0],
        "input_timestamps": ["2011-09-13 07:00:00"],
        "output_timestamps": ["2011-09-13 08:00:00", "2011-09-13 09:00:00"],
    }
    (tmp_path / "station_0.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="target-derived"):
        _load_weather_json(tmp_path, "weather_aligned_short", {"tasks": ["forecast"]})


def test_retrospective_record_rejected_even_without_statistics(tmp_path):
    record = {"text": "An event was reported after the forecast origin.",
              "input_window": [20.0], "output_window": [21.0],
              "data_integrity": {"status": "retrospective_only", "reason": "No pre-origin publication time"}}
    (tmp_path / "station_0.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="retrospective_only"):
        _load_weather_json(tmp_path, "weather_aligned_short", {"tasks": ["forecast"]})


def test_removing_statistics_and_metadata_does_not_clean_postevent_report(tmp_path):
    record = {"text": "Official Storm Data Summary (NOAA Storm Events): Measured rainfall and final damage.",
              "input_window": [20.0], "output_window": [21.0]}
    (tmp_path / "station_0.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="retrospective_only"):
        _load_weather_json(tmp_path, "weather_aligned_short", {"tasks": ["forecast"]})


def test_quarantined_training_directory_fails_before_reading_samples(tmp_path):
    split = tmp_path / "independent/splits/weather_aligned_short"
    split.mkdir(parents=True)
    (split / "DATA_QUARANTINE.json").write_text(json.dumps({"reason": "Target-derived input and invalid QA"}))
    cfg = TimeClawConfig()
    cfg.benchmark.repo_root = str(tmp_path)
    cfg.benchmark.training_root = "independent"
    with pytest.raises(ValueError, match="quarantined"):
        load_independent_dataset("weather_aligned_short", config=cfg)


def test_quarantined_memory_is_not_silently_reused(tmp_path, monkeypatch):
    from timeclaw.memory import hierarchical
    root = tmp_path / "memory"
    root.mkdir()
    (root / "DATA_QUARANTINE.json").write_text(json.dumps({"reason": "Weather and shared distilled ancestry requires rebuild"}))
    monkeypatch.setattr(hierarchical, "MEMORY_ROOT", root)
    with pytest.raises(ValueError, match="quarantined"):
        hierarchical.HierarchicalMemory(enabled=True)


def test_direct_finance_qa_loader_cannot_bypass_quarantine(tmp_path):
    (tmp_path / "DATA_QUARANTINE.json").write_text('{"status":"quarantined"}')
    with pytest.raises(ValueError, match="quarantined"):
        _load_finance_json(tmp_path, "finance_qa_short", {"tasks": ["mcqa"]})




def test_pipeline_rejects_copied_leak_before_tools_or_lm():
    from timeclaw.pipeline.runner import PipelineRunner
    runner = object.__new__(PipelineRunner)
    runner._ensure_tools = AsyncMock()
    with pytest.raises(ValueError, match="target-derived"):
        asyncio.run(runner.run(data=[20.0], text_data=[{"content": "24-hour output window beginning at the event anchor"}]))
    runner._ensure_tools.assert_not_awaited()


def test_uncontaminated_record_keeps_its_text(tmp_path):
    text = "A forecast bulletin published before the requested origin."
    (tmp_path / "station_0.json").write_text(json.dumps({"text": text, "input_window": [20.0], "output_window": [21.0]}))
    samples = _load_weather_json(tmp_path, "weather_aligned_short", {"tasks": ["forecast"]})
    assert len(samples) == 1
    assert samples[0].text == text


def test_training_rejects_quarantine_before_service_start(tmp_path, monkeypatch):
    from timeclaw.benchmark import train_loop
    from timeclaw.sbm import service_manager
    split = tmp_path / "independent/splits/weather_aligned_short"
    split.mkdir(parents=True)
    (split / "DATA_QUARANTINE.json").write_text(json.dumps({"reason": "observed leak"}))
    cfg = TimeClawConfig()
    cfg.benchmark.repo_root = str(tmp_path)
    cfg.benchmark.training_root = "independent"
    cfg.workspace_dir = str(tmp_path / "fresh_workspace")
    def service_must_not_start(*args, **kwargs):
        pytest.fail("Started service before rejecting quarantined dataset")
    monkeypatch.setattr(service_manager, "ensure_managed_sbm_server", service_must_not_start)
    with pytest.raises(ValueError, match="quarantined"):
        asyncio.run(train_loop.run_train_loop(dataset_name="weather_aligned_short", config=cfg))


