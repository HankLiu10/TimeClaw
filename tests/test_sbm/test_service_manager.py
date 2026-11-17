from __future__ import annotations

import signal

from timeclaw.sbm import service_manager


def test_ensure_target_port_available_terminates_existing_listener(monkeypatch):
    kill_calls: list[tuple[int, int]] = []
    state = {"remaining_hits": 2}

    def _fake_listener_pids(_port: int) -> list[int]:
        if state["remaining_hits"] > 0:
            state["remaining_hits"] -= 1
            return [12345]
        return []

    monkeypatch.setattr(
        service_manager,
        "_listener_pids_for_port",
        _fake_listener_pids,
    )
    monkeypatch.setattr(service_manager.os, "kill", lambda pid, sig: kill_calls.append((pid, sig)))
    monkeypatch.setattr(service_manager.time, "sleep", lambda _s: None)

    service_manager._ensure_target_port_available(8100, expected_model="chronos2")

    assert kill_calls == [(12345, signal.SIGTERM)]
