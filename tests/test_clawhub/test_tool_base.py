import numpy as np
import pytest

from timeclaw.clawhub.processing.slice import SliceTool
from timeclaw.clawhub.tool_base import ClawTool


class UnavailableTool(ClawTool):
    name = "unavailable_test_tool"
    is_stub = True

    async def _run(self, ts_data, **params):
        raise AssertionError("Stub tool must not execute")


@pytest.mark.asyncio
async def test_stub_tool_cannot_report_success():
    result = await UnavailableTool().execute(np.arange(3))

    assert result.success is False
    assert result.metadata["error_kind"] == "unavailable"


@pytest.mark.asyncio
async def test_slice_coerces_integer_like_float_indices():
    tool = SliceTool()

    result = await tool.execute(
        np.arange(500),
        start=318.0,
        end=390.0,
        source_ref="original_input",
        domain="finance",
    )

    assert result.success is True
    assert result.error is None
    assert len(result.data) == 72
    assert result.data[0] == 318
    assert result.data[-1] == 389


@pytest.mark.asyncio
async def test_slice_coerces_integer_like_string_indices_for_ndarray():
    tool = SliceTool()

    result = await tool.execute(
        np.arange(500),
        start="318",
        end="390",
        source_ref="original_input",
        domain="finance",
    )

    assert result.success is True
    assert result.error is None
    assert len(result.data) == 72
    assert result.data[0] == 318
    assert result.data[-1] == 389


@pytest.mark.asyncio
async def test_slice_rejects_non_integer_index_arguments():
    tool = SliceTool()

    result = await tool.execute(
        np.arange(10),
        start=3.5,
        end=8,
        source_ref="original_input",
    )

    assert result.success is False
    assert result.metadata["error_kind"] == "invalid_args"
    assert "invalid tool arguments" in str(result.error)
