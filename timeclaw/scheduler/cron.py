"""Lightweight cron scheduler using asyncio.

Supports one-shot and recurring tasks with simple interval-based scheduling.
No APScheduler dependency — pure asyncio timers.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable

from loguru import logger


@dataclass
class CronJob:
    """A scheduled job definition."""

    name: str
    callback: Callable[[], Awaitable[Any]]
    interval_seconds: int
    recurring: bool = True
    enabled: bool = True
    last_run: float = 0.0
    run_count: int = 0
    _task: asyncio.Task | None = field(default=None, repr=False)


class CronScheduler:
    """Simple interval-based task scheduler built on asyncio.

    Supports:
    - Recurring tasks (run every N seconds)
    - One-shot tasks (run once after delay)
    - Dynamic task add/remove
    """

    def __init__(self):
        self._jobs: dict[str, CronJob] = {}
        self._running = False

    def schedule(
        self,
        name: str,
        callback: Callable[[], Awaitable[Any]],
        interval_seconds: int = 3600,
        recurring: bool = True,
    ) -> CronJob:
        """Register a new scheduled task."""
        job = CronJob(
            name=name,
            callback=callback,
            interval_seconds=interval_seconds,
            recurring=recurring,
        )
        self._jobs[name] = job
        if self._running:
            job._task = asyncio.create_task(self._run_job(job))
        return job

    def unschedule(self, name: str) -> None:
        """Remove a scheduled task."""
        job = self._jobs.pop(name, None)
        if job and job._task:
            job._task.cancel()

    async def start(self) -> None:
        """Start all scheduled jobs."""
        if self._running:
            return
        self._running = True
        for job in self._jobs.values():
            if job.enabled:
                job._task = asyncio.create_task(self._run_job(job))
        logger.info(f"[CronScheduler] Started with {len(self._jobs)} jobs")

    async def stop(self) -> None:
        """Stop all running jobs."""
        self._running = False
        for job in self._jobs.values():
            if job._task:
                job._task.cancel()
                try:
                    await job._task
                except asyncio.CancelledError:
                    pass
        logger.info("[CronScheduler] Stopped")

    async def _run_job(self, job: CronJob) -> None:
        """Execute a single job with its schedule."""
        try:
            await asyncio.sleep(job.interval_seconds)
            while self._running and job.enabled:
                try:
                    await job.callback()
                    job.last_run = time.time()
                    job.run_count += 1
                    logger.debug(f"[CronScheduler] Job '{job.name}' completed (run #{job.run_count})")
                except Exception as e:
                    logger.warning(f"[CronScheduler] Job '{job.name}' failed: {e}")

                if not job.recurring:
                    break
                await asyncio.sleep(job.interval_seconds)
        except asyncio.CancelledError:
            pass

    @property
    def active_jobs(self) -> list[str]:
        return [name for name, job in self._jobs.items() if job.enabled]

    def get_status(self) -> dict[str, dict[str, Any]]:
        """Get status of all scheduled jobs."""
        return {
            name: {
                "enabled": job.enabled,
                "recurring": job.recurring,
                "interval_seconds": job.interval_seconds,
                "run_count": job.run_count,
                "last_run": job.last_run,
            }
            for name, job in self._jobs.items()
        }
