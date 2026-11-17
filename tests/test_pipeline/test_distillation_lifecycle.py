"""Offline memory phase boundaries: no live model, provider, or GPU is used."""

import asyncio
from types import SimpleNamespace

import pytest

from timeclaw.memory import hierarchical as hm
from timeclaw.core.accounting import RequestLedger


@pytest.fixture
def agent(monkeypatch, tmp_path):
    # Exercise the real lifecycle methods without initializing persistent stores.
    distiller = hm.DistillationAgent.__new__(hm.DistillationAgent)
    distiller._scope_workers = {}
    distiller._bucket_workers = {}
    distiller._pending_scope_force = set()
    distiller._pending_bucket_force = set()
    distiller.notes = SimpleNamespace(get_recent_notes=lambda count: [])
    monkeypatch.setattr(hm, "MEMORY_DIR", tmp_path / "memory")
    return distiller


@pytest.mark.asyncio
@pytest.mark.parametrize("scope_limited", [False, True])
async def test_drain_joins_bucket_created_after_initial_worker_snapshot(agent, scope_limited):
    scope = "weather_temperature_pilot_v1_forecast"
    bucket = hm._task_bucket_from_scope(scope)
    release_scope = asyncio.Event()
    bucket_started = asyncio.Event()
    release_bucket = asyncio.Event()
    drain_started = asyncio.Event()
    events = []

    async def bucket_work():
        bucket_started.set()
        await release_bucket.wait()
        events.append("bucket_finished")

    async def scope_work():
        await release_scope.wait()
        agent._bucket_workers[bucket] = asyncio.create_task(bucket_work())

    async def synchronous_drain(scopes):
        # Real synchronous drain acquires a file lock; crossing this boundary
        # before the bucket finishes would block the event loop on its own lock.
        assert events == ["bucket_finished"]
        drain_started.set()

    agent._distill_pending_scope_list = synchronous_drain
    agent._scope_workers[scope] = asyncio.create_task(scope_work())
    flush = asyncio.create_task(
        agent.distill_pending_scopes([scope]) if scope_limited
        else agent.distill_pending_all()
    )
    await asyncio.sleep(0)  # Let flush take its initial snapshot (scope only).
    release_scope.set()
    await asyncio.wait_for(bucket_started.wait(), timeout=1)
    await asyncio.sleep(0)
    assert not drain_started.is_set()
    release_bucket.set()
    await asyncio.wait_for(flush, timeout=1)
    assert drain_started.is_set()
    assert agent._active_distillation_workers() == []


@pytest.mark.asyncio
async def test_abort_joins_workers_created_during_cancel_before_ledger_end(agent, tmp_path):
    scope_started = asyncio.Event()
    bucket_started = asyncio.Event()
    block = asyncio.Event()
    ledger = RequestLedger("offline-learning-phase", tmp_path / "ledger")

    async def bucket_work():
        try:
            bucket_started.set()
            await block.wait()
        finally:
            ledger.record("bucket_cleanup_finished")

    async def scope_work():
        try:
            scope_started.set()
            await block.wait()
        finally:
            agent._bucket_workers["new_bucket"] = asyncio.create_task(bucket_work())
            await bucket_started.wait()
            ledger.record("scope_cleanup_finished")

    with ledger.activate():
        agent._pending_scope_force.add("scope")
        agent._pending_bucket_force.add("new_bucket")
        agent._scope_workers["scope"] = asyncio.create_task(scope_work())
        await scope_started.wait()
        await asyncio.wait_for(agent.cancel_pending_distillation(), timeout=1)
        assert agent._active_distillation_workers() == []
    before = ledger.path.read_bytes()
    await asyncio.sleep(0)
    assert ledger.path.read_bytes() == before
    assert [event["kind"] for event in ledger.read_events()] == [
        "request_start", "scope_cleanup_finished", "bucket_cleanup_finished", "request_end",
    ]
    assert agent._pending_scope_force == agent._pending_bucket_force == set()


@pytest.mark.asyncio
async def test_worker_failure_propagates_from_explicit_drain(agent):
    release = asyncio.Event()

    async def failing_work():
        await release.wait()
        raise ValueError("invalid memory output")

    agent._scope_workers["scope"] = asyncio.create_task(failing_work())
    flush = asyncio.create_task(agent.distill_pending_all())
    await asyncio.sleep(0)
    release.set()
    with pytest.raises(ValueError, match="invalid memory output"):
        await flush


@pytest.mark.asyncio
async def test_abort_surfaces_cleanup_failure_after_other_workers_stop(agent):
    started = asyncio.Event()
    stopped = asyncio.Event()

    async def failed_cleanup():
        try:
            started.set()
            await asyncio.Event().wait()
        finally:
            raise ValueError("cleanup failed")

    async def other_work():
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    agent._scope_workers["bad"] = asyncio.create_task(failed_cleanup())
    agent._bucket_workers["other"] = asyncio.create_task(other_work())
    await started.wait()
    with pytest.raises(ValueError, match="cleanup failed"):
        await agent.cancel_pending_distillation()
    assert stopped.is_set()
    assert agent._active_distillation_workers() == []


@pytest.mark.asyncio
async def test_cancellation_of_abort_still_joins_spawned_cleanup_workers(agent):
    started = asyncio.Event()
    cleaning = asyncio.Event()
    child_started = asyncio.Event()
    child_stopped = asyncio.Event()

    async def child_work():
        try:
            child_started.set()
            await asyncio.Event().wait()
        finally:
            child_stopped.set()

    async def scope_work():
        try:
            started.set()
            await asyncio.Event().wait()
        finally:
            try:
                cleaning.set()
                await asyncio.Event().wait()
            finally:
                agent._bucket_workers["late"] = asyncio.create_task(child_work())
                await child_started.wait()

    agent._scope_workers["scope"] = asyncio.create_task(scope_work())
    await started.wait()
    abort = asyncio.create_task(agent.cancel_pending_distillation())
    await cleaning.wait()
    abort.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(abort, timeout=1)
    assert child_stopped.is_set()
    assert agent._active_distillation_workers() == []


@pytest.mark.asyncio
async def test_hierarchical_memory_exposes_abort_boundary(agent):
    memory = hm.HierarchicalMemory.__new__(hm.HierarchicalMemory)
    memory.enabled = True
    memory.distiller = agent
    await memory.cancel_pending_distillation()
    memory.enabled = False
    memory.distiller = None
    await memory.cancel_pending_distillation()
