from __future__ import annotations

import asyncio
import heapq
import time
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable
from typing import TypeVar

from timeclaw.core.retry_control import RetryLater

T = TypeVar("T")
K = TypeVar("K")


def resolve_active_work_item_limit(
    *,
    stream_limits: Iterable[int | None],
    llm_max_inflight_requests: int | None,
) -> int | None:
    """Resolve the bounded number of live work-item coroutines.

    New definition:
        ``active_work_item_limit`` is not the same thing as LLM inflight
        concurrency. It caps how many benchmark work-item coroutines may stay
        alive at once so the event loop and memory do not blow up from
        scheduling every pending `(sample, task)` immediately.

        When per-task slots are configured, the tight upper bound is the sum of
        those per-stream slots because additional work items would only sit idle
        behind the task semaphores. Otherwise we fall back to the LLM inflight
        cap, which is the only explicit upstream concurrency budget.
    """
    finite_stream_limits = [int(limit) for limit in stream_limits if limit and int(limit) > 0]
    if finite_stream_limits:
        return max(1, sum(finite_stream_limits))
    if llm_max_inflight_requests is not None and int(llm_max_inflight_requests) > 0:
        return int(llm_max_inflight_requests)
    return None


def interleave_items_round_robin(
    items: Iterable[T],
    *,
    key_fn: Callable[[T], K],
) -> list[T]:
    """Return a stream-balanced item order while preserving per-stream order.

    New definition:
        The benchmark scheduler should not seed its active set with a long
        prefix from only one `(dataset, task)` stream. Otherwise many work-item
        coroutines immediately block on the same per-task semaphore and occupy
        ``active_work_item_limit`` without creating real upstream concurrency.

        This helper groups items by stream key and then emits them in
        round-robin order:
        `stream_a[0], stream_b[0], ..., stream_a[1], stream_b[1], ...`.
        Within each stream, the original order is preserved.
    """
    grouped: dict[K, deque[T]] = {}
    stream_order: deque[K] = deque()
    for item in items:
        key = key_fn(item)
        if key not in grouped:
            grouped[key] = deque()
            stream_order.append(key)
        grouped[key].append(item)

    ordered: list[T] = []
    while stream_order:
        key = stream_order.popleft()
        queue = grouped[key]
        ordered.append(queue.popleft())
        if queue:
            stream_order.append(key)
    return ordered


async def iter_completed_tasks_bounded(
    task_factories: Iterable[Callable[[], Awaitable[T]]],
    *,
    max_active: int | None,
) -> AsyncIterator[asyncio.Task[T]]:
    """Yield completed tasks while keeping only a bounded active set alive."""
    limit = int(max_active) if max_active is not None and int(max_active) > 0 else None
    pending: set[asyncio.Task[T]] = set()
    factory_iter = iter(task_factories)

    def _start_next() -> bool:
        try:
            factory = next(factory_iter)
        except StopIteration:
            return False
        pending.add(asyncio.create_task(factory()))
        return True

    if limit is None:
        while _start_next():
            pass
    else:
        while len(pending) < limit and _start_next():
            pass

    while pending:
        done, still_pending = await asyncio.wait(
            pending,
            return_when=asyncio.FIRST_COMPLETED,
        )
        pending = set(still_pending)
        for task in done:
            yield task
        if limit is not None:
            while len(pending) < limit and _start_next():
                pass


async def iter_completed_tasks_with_retry_queue(
    task_factories: Iterable[Callable[[], Awaitable[T]]],
    *,
    max_active: int | None,
) -> AsyncIterator[asyncio.Task[T]]:
    """Yield completed tasks while parked retries wait outside active slots.

    New definition:
        ``RetryLater`` no longer sleeps inside the current work-item coroutine.
        Instead, this scheduler parks that logical work item in a retry queue
        until its ``retry_at`` time, then re-dispatches it as a fresh coroutine.
        Parked items therefore do not consume ``active_work_item_limit``.
    """
    limit = int(max_active) if max_active is not None and int(max_active) > 0 else None
    ready: deque[Callable[[], Awaitable[T]]] = deque(task_factories)
    parked: list[tuple[float, int, Callable[[], Awaitable[T]]]] = []
    pending: dict[asyncio.Task[T], Callable[[], Awaitable[T]]] = {}
    parked_seq = 0

    def _activate_due_parked() -> None:
        now = time.monotonic()
        while parked and parked[0][0] <= now:
            _retry_at, _seq, factory = heapq.heappop(parked)
            ready.append(factory)

    def _start_ready() -> None:
        _activate_due_parked()
        while ready and (limit is None or len(pending) < limit):
            factory = ready.popleft()
            task = asyncio.create_task(factory())
            pending[task] = factory

    _start_ready()

    while pending or ready or parked:
        if not pending:
            _activate_due_parked()
            if ready:
                _start_ready()
                continue
            if not parked:
                break
            sleep_for = max(0.0, parked[0][0] - time.monotonic())
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)
            _start_ready()
            continue

        timeout: float | None = None
        has_capacity_for_retry = limit is None or len(pending) < limit
        if parked and has_capacity_for_retry:
            timeout = max(0.0, parked[0][0] - time.monotonic())
        done, _still_pending = await asyncio.wait(
            pending.keys(),
            timeout=timeout,
            return_when=asyncio.FIRST_COMPLETED,
        )
        if not done:
            _start_ready()
            continue

        for task in done:
            factory = pending.pop(task)
            try:
                exc = task.exception()
            except asyncio.CancelledError:
                yield task
                continue
            if isinstance(exc, RetryLater):
                heapq.heappush(parked, (float(exc.retry_at), parked_seq, factory))
                parked_seq += 1
                continue
            yield task

        _start_ready()
