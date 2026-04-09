"""Tests for ClawHub tool registry and tool discovery."""

from __future__ import annotations

import numpy as np
import pytest

from timeclaw.clawhub.tool_base import ClawTool, ToolCategory, ToolIOType
from timeclaw.clawhub.tool_registry import (
    clear_registry,
    discover_and_register,
    get_tool_instance,
    list_tool_instances,
    register_tool_instance,
)


class DummyTool(ClawTool):
    name = "dummy_test_tool"
    category = ToolCategory.PROCESSING
    description = "A dummy tool for testing"
    io_type = ToolIOType.X_TO_R

    async def _run(self, ts_data, **params):
        return {"result": "dummy"}


class TestToolRegistry:
    def setup_method(self):
        clear_registry()

    def test_register_and_retrieve(self):
        tool = DummyTool()
        register_tool_instance(tool)
        retrieved = get_tool_instance("dummy_test_tool")
        assert retrieved is tool

    def test_list_by_category(self):
        tool = DummyTool()
        register_tool_instance(tool)
        tools = list_tool_instances(ToolCategory.PROCESSING)
        assert any(t.name == "dummy_test_tool" for t in tools)

    def test_not_found(self):
        with pytest.raises(KeyError):
            get_tool_instance("nonexistent_tool")

    def test_discover_and_register(self):
        count = discover_and_register()
        assert count > 0
        all_tools = list_tool_instances()
        assert len(all_tools) > 10
        names = [t.name for t in all_tools]
        assert "basic_stats" in names
        assert "detect_trend" in names
        assert "value_at" in names
        assert "window_stats" in names
        assert "ensemble_forecast" not in names
        assert "checkpoint_forecast" not in names

    def test_function_schema(self):
        tool = DummyTool()
        schema = tool.to_function_schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "dummy_test_tool"


class TestRealTools:
    """Test that real (non-stub) tools produce valid output."""

    def setup_method(self):
        clear_registry()
        discover_and_register()

    @pytest.mark.asyncio
    async def test_basic_stats(self):
        tool = get_tool_instance("basic_stats")
        data = np.random.randn(100)
        result = await tool.execute(data)
        assert result.success
        assert "mean" in result.data
        assert "std" in result.data

    @pytest.mark.asyncio
    async def test_detect_trend(self):
        data = np.arange(100, dtype=float) + np.random.randn(100) * 0.1
        tool = get_tool_instance("detect_trend")
        result = await tool.execute(data)
        assert result.success
        assert result.data["direction"] == "up"

    @pytest.mark.asyncio
    async def test_detect_anomaly(self):
        data = np.random.randn(100)
        data[50] = 100.0  # inject anomaly
        tool = get_tool_instance("detect_anomaly")
        result = await tool.execute(data)
        assert result.success
        assert 50 in result.data["anomaly_indices"]

    @pytest.mark.asyncio
    async def test_normalize(self):
        tool = get_tool_instance("normalize")
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = await tool.execute(data, method="zscore")
        assert result.success
        normalized = np.asarray(result.data)
        assert abs(np.mean(normalized)) < 0.01

    @pytest.mark.asyncio
    async def test_autocorrelation(self):
        tool = get_tool_instance("autocorrelation")
        data = np.random.randn(100)
        result = await tool.execute(data, max_lag=10, mode="acf")
        assert result.success
        assert "acf" in result.data
        assert abs(result.data["acf"][0] - 1.0) < 0.01

    @pytest.mark.asyncio
    async def test_value_at(self):
        tool = get_tool_instance("value_at")
        data = np.array([10.0, 20.0, 30.0, 40.0])
        result = await tool.execute(data, index=-1)
        assert result.success
        assert result.data["index"] == -1
        assert result.data["resolved_index"] == 3
        assert result.data["value"] == 40.0

    @pytest.mark.asyncio
    async def test_window_stats(self):
        tool = get_tool_instance("window_stats")
        data = np.array([1.0, 3.0, 2.0, 5.0, 4.0])
        result = await tool.execute(data, start=1, end=4)
        assert result.success
        assert result.data["resolved_start"] == 1
        assert result.data["resolved_end"] == 4
        assert result.data["length"] == 3
        assert result.data["start_value"] == 3.0
        assert result.data["end_value"] == 5.0
        assert result.data["min"] == 2.0
        assert result.data["max"] == 5.0
        assert result.data["delta"] == 2.0

    @pytest.mark.asyncio
    async def test_fft_spectrum(self):
        tool = get_tool_instance("fft_spectrum")
        t = np.linspace(0, 1, 256)
        data = np.sin(2 * np.pi * 10 * t)
        result = await tool.execute(data)
        assert result.success
        assert "frequencies" in result.data
        assert "magnitudes" in result.data

    @pytest.mark.asyncio
    async def test_bound_check(self):
        tool = get_tool_instance("bound_check")
        data = np.array([5.0, 15.0, 25.0])
        result = await tool.execute(data, lower=0, upper=20)
        assert result.success
        assert not result.data["within_bounds"]
        assert result.data["violations"] == 1
