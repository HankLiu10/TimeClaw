"""Heartbeat daemon: periodic health checks and maintenance tasks.

Runs configurable tasks at a fixed interval, writing status to HEARTBEAT.md.
Uses pure asyncio — no external scheduler dependency.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Awaitable

from loguru import logger


class HeartbeatStatus(str, Enum):
    OK = "OK"
    WARNING = "WARNING"
    ERROR = "ERROR"


@dataclass
class HeartbeatTask:
    """A single task executed during each heartbeat cycle."""

    name: str
    callback: Callable[[], Awaitable[dict[str, Any]]]
    enabled: bool = True


@dataclass
class HeartbeatResult:
    """Result of a single heartbeat cycle."""

    timestamp: float = field(default_factory=time.time)
    status: HeartbeatStatus = HeartbeatStatus.OK
    task_results: dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0


class Heartbeat:
    """Periodic heartbeat daemon that runs maintenance tasks.

    Tasks include:
    - Drift detection on monitored data sources
    - Experience pool cleanup (expired entries)
    - Vector index maintenance
    """

    def __init__(
        self,
        interval_minutes: int = 30,
        tasks: list[HeartbeatTask] | None = None,
        status_file: str | Path = ".timeclaw/HEARTBEAT.md",
    ):
        self.interval_seconds = interval_minutes * 60
        self.tasks = tasks or []
        self.status_file = Path(status_file)
        self._running = False
        self._task: asyncio.Task | None = None
        self._history: list[HeartbeatResult] = []

    def add_task(self, task: HeartbeatTask) -> None:
        self.tasks.append(task)

    async def start(self) -> None:
        """Start the heartbeat loop in the background."""
        if self._running:
            logger.warning("[Heartbeat] Already running")
            return
        self._running = True
        self._task = asyncio.create_task(self._loop())
        logger.info(f"[Heartbeat] Started with {len(self.tasks)} tasks, "
                     f"interval={self.interval_seconds}s")

    async def stop(self) -> None:
        """Stop the heartbeat loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("[Heartbeat] Stopped")

    async def _loop(self) -> None:
        """Main heartbeat loop."""
        while self._running:
            result = await self._run_cycle()
            self._history.append(result)
            self._write_status(result)
            await asyncio.sleep(self.interval_seconds)

    async def _run_cycle(self) -> HeartbeatResult:
        """Execute one heartbeat cycle: run all enabled tasks."""
        start = time.monotonic()
        result = HeartbeatResult()

        for task in self.tasks:
            if not task.enabled:
                continue
            try:
                task_result = await asyncio.wait_for(task.callback(), timeout=60)
                result.task_results[task.name] = {
                    "status": "ok",
                    "data": task_result,
                }
            except asyncio.TimeoutError:
                result.task_results[task.name] = {"status": "timeout"}
                result.status = HeartbeatStatus.WARNING
            except Exception as e:
                result.task_results[task.name] = {"status": "error", "error": str(e)}
                result.status = HeartbeatStatus.ERROR
                logger.warning(f"[Heartbeat] Task '{task.name}' failed: {e}")

        result.duration_ms = (time.monotonic() - start) * 1000
        logger.info(f"[Heartbeat] Cycle complete: {result.status.value} "
                     f"({result.duration_ms:.0f}ms)")
        return result

    def _write_status(self, result: HeartbeatResult) -> None:
        """Write heartbeat status to HEARTBEAT.md (like OpenClaw's HEARTBEAT_OK)."""
        self.status_file.parent.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(result.timestamp))
        lines = [
            f"# TimeClaw Heartbeat — {result.status.value}",
            f"",
            f"**Last beat**: {ts}",
            f"**Duration**: {result.duration_ms:.0f}ms",
            f"**Tasks**: {len(result.task_results)}",
            f"",
            "## Task Results",
            "",
        ]
        for name, data in result.task_results.items():
            status = data.get("status", "unknown")
            lines.append(f"- **{name}**: {status}")
            if status == "error":
                lines.append(f"  - Error: {data.get('error', 'unknown')}")
        lines.append("")
        self.status_file.write_text("\n".join(lines))

    async def run_once(self) -> HeartbeatResult:
        """Run a single heartbeat cycle (for testing or on-demand use)."""
        result = await self._run_cycle()
        self._history.append(result)
        self._write_status(result)
        return result

    @property
    def last_result(self) -> HeartbeatResult | None:
        return self._history[-1] if self._history else None
