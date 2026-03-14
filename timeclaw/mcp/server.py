"""MCP Server: exposes ClawHub tools through the Model Context Protocol.

Converts ClawTool instances to MCP tool descriptors and handles tool invocations.
Supports stdio and streamable-http transports.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from loguru import logger

from timeclaw.clawhub.tool_base import ClawTool
from timeclaw.clawhub.tool_registry import get_tool_instance, list_tool_instances


class ClawHubMCPServer:
    """Exposes all registered ClawHub tools as MCP-compatible tools.

    Uses the `mcp` Python SDK when available; falls back to a lightweight
    JSON-RPC implementation for environments without the SDK.
    """

    def __init__(self, server_name: str = "timeclaw-clawhub"):
        self.server_name = server_name
        self._tools: list[ClawTool] = []

    def _collect_tools(self) -> None:
        """Gather all non-stub ClawHub tools."""
        self._tools = [t for t in list_tool_instances() if not t.is_stub]
        logger.info(f"[MCP Server] Collected {len(self._tools)} tools for exposure")

    def _tool_to_mcp_descriptor(self, tool: ClawTool) -> dict[str, Any]:
        """Convert a ClawTool to an MCP tool descriptor."""
        schema = tool.to_function_schema()["function"]
        return {
            "name": schema["name"],
            "description": schema["description"],
            "inputSchema": schema.get("parameters", {
                "type": "object",
                "properties": {},
                "required": [],
            }),
        }

    def get_tool_descriptors(self) -> list[dict[str, Any]]:
        """Return MCP tool descriptors for all exposed tools."""
        self._collect_tools()
        return [self._tool_to_mcp_descriptor(t) for t in self._tools]

    async def handle_tool_call(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Execute a tool call and return MCP-formatted result."""
        try:
            tool = get_tool_instance(tool_name)
        except KeyError:
            return {
                "isError": True,
                "content": [{"type": "text", "text": f"Tool '{tool_name}' not found"}],
            }

        ts_data = arguments.pop("ts_data", None)
        text = arguments.pop("text", None)
        images = arguments.pop("images", None)

        result = await tool.execute(ts_data, text=text, images=images, **arguments)

        if result.success:
            content_text = json.dumps(result.data, default=str, ensure_ascii=False)
            return {
                "content": [{"type": "text", "text": content_text}],
            }
        return {
            "isError": True,
            "content": [{"type": "text", "text": result.error or "Unknown error"}],
        }

    async def serve_stdio(self) -> None:
        """Serve using the MCP SDK's stdio transport if available, else custom JSON-RPC."""
        self._collect_tools()
        try:
            await self._serve_with_sdk()
        except ImportError:
            logger.info("[MCP Server] mcp SDK not available, using lightweight JSON-RPC")
            await self._serve_jsonrpc_stdio()

    async def _serve_with_sdk(self) -> None:
        """Serve using the official `mcp` Python SDK."""
        from mcp.server import Server
        from mcp.server.stdio import stdio_server
        from mcp import types

        server = Server(self.server_name)

        @server.list_tools()
        async def _list_tools() -> list[types.Tool]:
            tools = []
            for desc in self.get_tool_descriptors():
                tools.append(types.Tool(
                    name=desc["name"],
                    description=desc.get("description", ""),
                    inputSchema=desc.get("inputSchema", {}),
                ))
            return tools

        @server.call_tool()
        async def _call_tool(name: str, arguments: dict) -> list[types.TextContent]:
            result = await self.handle_tool_call(name, arguments)
            contents = []
            for c in result.get("content", []):
                contents.append(types.TextContent(type="text", text=c.get("text", "")))
            return contents

        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, server.create_initialization_options())

    async def _serve_jsonrpc_stdio(self) -> None:
        """Lightweight JSON-RPC 2.0 server over stdin/stdout for MCP compatibility."""
        reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(reader)
        await asyncio.get_event_loop().connect_read_pipe(lambda: protocol, __import__("sys").stdin)

        writer_transport, writer_protocol = await asyncio.get_event_loop().connect_write_pipe(
            asyncio.streams.FlowControlMixin, __import__("sys").stdout
        )
        writer = asyncio.StreamWriter(writer_transport, writer_protocol, None, asyncio.get_event_loop())

        logger.info(f"[MCP Server] JSON-RPC stdio server started with {len(self._tools)} tools")

        while True:
            line = await reader.readline()
            if not line:
                break
            try:
                request = json.loads(line.decode())
                response = await self._handle_jsonrpc(request)
                writer.write((json.dumps(response) + "\n").encode())
                await writer.drain()
            except Exception as e:
                error_resp = {
                    "jsonrpc": "2.0",
                    "error": {"code": -32603, "message": str(e)},
                    "id": None,
                }
                writer.write((json.dumps(error_resp) + "\n").encode())
                await writer.drain()

    async def _handle_jsonrpc(self, request: dict[str, Any]) -> dict[str, Any]:
        """Handle a single JSON-RPC request."""
        method = request.get("method", "")
        req_id = request.get("id")
        params = request.get("params", {})

        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": self.server_name, "version": "1.0.0"},
                },
                "id": req_id,
            }
        elif method == "tools/list":
            descriptors = self.get_tool_descriptors()
            return {
                "jsonrpc": "2.0",
                "result": {"tools": descriptors},
                "id": req_id,
            }
        elif method == "tools/call":
            tool_name = params.get("name", "")
            arguments = params.get("arguments", {})
            result = await self.handle_tool_call(tool_name, arguments)
            return {"jsonrpc": "2.0", "result": result, "id": req_id}
        else:
            return {
                "jsonrpc": "2.0",
                "error": {"code": -32601, "message": f"Method not found: {method}"},
                "id": req_id,
            }

    def serve(self, transport: str = "stdio") -> None:
        """Start the MCP server (blocking)."""
        if transport == "stdio":
            asyncio.run(self.serve_stdio())
        else:
            raise ValueError(f"Unsupported transport: {transport}. Use 'stdio'.")
