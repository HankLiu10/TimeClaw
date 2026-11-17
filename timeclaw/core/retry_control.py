from __future__ import annotations

import time
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

_DEFER_RETRY_TO_SCHEDULER: ContextVar[bool] = ContextVar(
    "timeclaw_defer_retry_to_scheduler",
    default=False,
)
_CURRENT_WORK_ITEM_ID: ContextVar[str | None] = ContextVar(
    "timeclaw_current_work_item_id",
    default=None,
)


class RetryLater(RuntimeError):
    """Signal that the caller should park this work item and retry it later.

    New definition:
        This exception is only for transient wait-style retries whose delay
        should be handled by the outer benchmark scheduler instead of by
        sleeping inside the current sample coroutine. The scheduler should park
        the work item until ``retry_at`` and, critically, release the current
        active/per-task slot while it waits.
    """

    def __init__(
        self,
        *,
        delay_s: float,
        reason: str,
        layer: str,
    ) -> None:
        self.delay_s = max(0.0, float(delay_s))
        self.reason = str(reason or "").strip() or "retry later"
        self.layer = str(layer or "").strip() or "unknown"
        self.retry_at = time.monotonic() + self.delay_s
        super().__init__(f"{self.layer}: {self.reason}")


def should_defer_retry_to_scheduler() -> bool:
    return bool(_DEFER_RETRY_TO_SCHEDULER.get())


def current_work_item_id() -> str | None:
    value = _CURRENT_WORK_ITEM_ID.get()
    return str(value).strip() or None


@contextmanager
def deferred_retry_scope(work_item_id: str) -> Iterator[None]:
    """Enable scheduler-owned retry parking for one logical work item."""
    defer_token = _DEFER_RETRY_TO_SCHEDULER.set(True)
    item_token = _CURRENT_WORK_ITEM_ID.set(str(work_item_id))
    try:
        yield
    finally:
        _CURRENT_WORK_ITEM_ID.reset(item_token)
        _DEFER_RETRY_TO_SCHEDULER.reset(defer_token)
