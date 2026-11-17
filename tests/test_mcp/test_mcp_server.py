"""Tests for MCP Server."""

import inspect

import pytest

from timeclaw.clawhub.tool_base import ClawTool, ToolCategory, ToolIOType
from timeclaw.mcp.server import ClawHubMCPServer
from timeclaw.clawhub.tool_registry import discover_and_register, clear_registry
from timeclaw.mcp.server_factory import register_tool
from timeclaw.mcp.server_factory import register_category


@pytest.fixture(autouse=True)
def setup_registry():
    clear_registry()
    discover_and_register()
    yield
    clear_registry()


def test_server_get_descriptors():
    server = ClawHubMCPServer()
    descriptors = server.get_tool_descriptors()
    assert len(descriptors) > 0
    for d in descriptors:
        assert "name" in d
        assert "description" in d
        assert "inputSchema" in d


def test_descriptor_format():
    server = ClawHubMCPServer()
    descriptors = server.get_tool_descriptors()
    for d in descriptors:
        schema = d["inputSchema"]
        assert isinstance(schema, dict)
        if "type" in schema:
            assert schema["type"] == "object"


@pytest.mark.asyncio
async def test_handle_tool_call_invalid():
    server = ClawHubMCPServer()
    result = await server.handle_tool_call("nonexistent_tool", {})
    assert result.get("isError") is True


@pytest.mark.asyncio
async def test_handle_jsonrpc_initialize():
    server = ClawHubMCPServer()
    server._collect_tools()
    response = await server._handle_jsonrpc({
        "jsonrpc": "2.0",
        "method": "initialize",
        "params": {},
        "id": 1,
    })
    assert response["result"]["serverInfo"]["name"] == "timeclaw-clawhub"


@pytest.mark.asyncio
async def test_handle_jsonrpc_list_tools():
    server = ClawHubMCPServer()
    server._collect_tools()
    response = await server._handle_jsonrpc({
        "jsonrpc": "2.0",
        "method": "tools/list",
        "params": {},
        "id": 2,
    })
    tools = response["result"]["tools"]
    assert len(tools) > 0


def test_server_factory_prediction_excludes_hidden_tools():
    class DummyMCP:
        def __init__(self):
            self.names = []

        def tool(self, *, name, description):
            def decorator(fn):
                self.names.append(name)
                return fn
            return decorator

    mcp = DummyMCP()
    count = register_category(mcp, "prediction")

    assert count > 0
    assert "checkpoint_forecast" not in mcp.names
    assert "ensemble_forecast" not in mcp.names


def test_register_tool_preserves_nullable_schema_types():
    class DummyTool(ClawTool):
        name = "dummy_nullable_tool"
        category = ToolCategory.PREDICTION
        io_type = ToolIOType.X_TO_X
        description = "dummy"
        parameters_schema = {
            "type": "object",
            "properties": {
                "seasonal_periods": {"type": ["integer", "null"], "default": None},
                "seasonal": {"type": ["string", "null"], "default": None},
            },
            "required": [],
        }

        async def _run(self, ts_data, **params):
            return params

    class DummyMCP:
        def __init__(self):
            self.handlers = {}

        def tool(self, *, name, description):
            def decorator(fn):
                self.handlers[name] = fn
                return fn
            return decorator

    mcp = DummyMCP()
    register_tool(mcp, DummyTool())
    handler = mcp.handlers["dummy_nullable_tool"]
    sig = inspect.signature(handler)

    assert sig.parameters["seasonal_periods"].annotation == (int | None)
    assert sig.parameters["seasonal"].annotation == (str | None)
