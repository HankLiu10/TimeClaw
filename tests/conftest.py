"""Shared test fixtures for TimeClaw tests."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import numpy as np
import pytest

from timeclaw.llm.base import LLMBackend, LLMResponse


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
