"""Shared test fixtures for TimeClaw tests."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import numpy as np
import pytest
import socket
import ipaddress

from timeclaw.llm.base import LLMBackend, LLMResponse


@pytest.fixture(autouse=True)
def no_external_network(monkeypatch):
    """Tests may use loopback fixtures; real paid/provider traffic is forbidden."""
    connect = socket.socket.connect
    def local_connect(sock, address):
        if isinstance(address, tuple):
            host = address[0]
            if host != 'localhost':
                try:
                    local = ipaddress.ip_address(host).is_loopback
                except ValueError:
                    local = False
                if not local:
                    raise AssertionError(f'External network is forbidden in offline tests: {host}')
        return connect(sock, address)
    monkeypatch.setattr(socket.socket, 'connect', local_connect)


@pytest.fixture(autouse=True)
def restore_workspace_bindings(monkeypatch):
    """Independent CLI jobs have separate processes; tests share one process.

    Restore only process-level artifact bindings after each case, so a training
    test cannot redirect a later trace reader or memory test to its workspace.
    Tests still exercise the real configuration functions inside their case.
    """
    from pathlib import Path
    from timeclaw.benchmark import train_loop
    modules = [train_loop, train_loop.hierarchical_memory_module,
               train_loop.trace_logger_module, train_loop.task_pruning_module,
               train_loop.llm_base_module]
    for module in modules:
        for name, value in list(vars(module).items()):
            if isinstance(value, Path):
                monkeypatch.setattr(module, name, value)
            elif name.startswith('_MANAGED_TRAIN_') and isinstance(value, set):
                monkeypatch.setattr(module, name, set(value))


@pytest.fixture
def sample_univariate():
    """A simple univariate time series with trend and noise."""
    np.random.seed(42)
    t = np.arange(100)
    return t * 0.5 + np.random.randn(100) * 2


@pytest.fixture
def sample_multivariate():
    """A 3-channel multivariate time series."""
    np.random.seed(42)
    t = np.arange(100)
    ch1 = t * 0.5 + np.random.randn(100) * 2
    ch2 = np.sin(t * 0.1) * 10 + np.random.randn(100)
    ch3 = np.random.randn(100) * 5
    return np.column_stack([ch1, ch2, ch3])


@pytest.fixture
def sample_stationary():
    """A stationary time series (white noise)."""
    np.random.seed(42)
    return np.random.randn(200)


@pytest.fixture
def mock_llm_backend():
    """A mock LLM backend that returns configurable responses."""
    backend = AsyncMock(spec=LLMBackend)
    backend.complete = AsyncMock(
        return_value=LLMResponse(
            text='{"task_type": "forecast", "sub_goals": ["predict next 10 steps"]}',
            parsed={"task_type": "forecast", "sub_goals": ["predict next 10 steps"]},
            model="mock",
        )
    )
    return backend


def make_mock_llm_response(text: str = "", parsed: dict | None = None) -> LLMResponse:
    """Helper to create mock LLM responses."""
    return LLMResponse(text=text, parsed=parsed, model="mock")
