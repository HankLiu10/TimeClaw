"""Tests for Heartbeat and CronScheduler."""

import asyncio
import tempfile
from pathlib import Path

import pytest

from timeclaw.scheduler.heartbeat import Heartbeat, HeartbeatStatus, HeartbeatTask
from timeclaw.scheduler.cron import CronScheduler


@pytest.mark.asyncio
async def test_heartbeat_run_once():
    call_count = 0

    async def dummy_task():
        nonlocal call_count
        call_count += 1
        return {"checked": True}

    with tempfile.TemporaryDirectory() as tmpdir:
        hb = Heartbeat(
            interval_minutes=1,
            tasks=[HeartbeatTask(name="dummy", callback=dummy_task)],
            status_file=Path(tmpdir) / "HEARTBEAT.md",
        )
        result = await hb.run_once()
        assert result.status == HeartbeatStatus.OK
        assert call_count == 1
        assert "dummy" in result.task_results
        assert (Path(tmpdir) / "HEARTBEAT.md").exists()


@pytest.mark.asyncio
async def test_heartbeat_failing_task():
    async def failing_task():
        raise ValueError("test error")

    with tempfile.TemporaryDirectory() as tmpdir:
        hb = Heartbeat(
            interval_minutes=1,
            tasks=[HeartbeatTask(name="fail", callback=failing_task)],
            status_file=Path(tmpdir) / "HEARTBEAT.md",
        )
        result = await hb.run_once()
        assert result.status == HeartbeatStatus.ERROR
        assert result.task_results["fail"]["status"] == "error"


@pytest.mark.asyncio
async def test_cron_scheduler_status():
    scheduler = CronScheduler()
    call_count = 0

    async def task():
        nonlocal call_count
        call_count += 1

    scheduler.schedule("test_job", task, interval_seconds=3600)
    status = scheduler.get_status()
    assert "test_job" in status
    assert status["test_job"]["enabled"]
    assert status["test_job"]["run_count"] == 0
