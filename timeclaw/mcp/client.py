"""MCP Client: consumes external MCP tools and wraps them as ClawTool instances.

Allows TimeClaw to dynamically discover and invoke tools from external MCP servers,
integrating them seamlessly into the ClawHub tool registry.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
from typing import Any

from loguru import logger

from timeclaw.clawhub.tool_base import ClawTool, ToolCategory, ToolIOType, ToolResult
from timeclaw.clawhub.tool_registry import register_tool_instance


class MCPToolWrapper(ClawTool):
    """Wraps a remote MCP tool as a local ClawTool.

    Translates ClawTool.execute() calls into MCP tool/call requests
    and returns the result as a ToolResult.
    """

    category = ToolCategory.MULTIMODAL
    io_type = ToolIOType.X_TO_R
    is_stub = False

    def __init__(
        self,
        tool_name: str,
        tool_description: str,
        input_schema: dict[str, Any],
        client: MCPClient,
    ):
        self.name = f"mcp_{tool_name}"
        self.description = f"[MCP] {tool_description}"
        self.parameters_schema = input_schema
        self._remote_name = tool_name
        self._client = client

    async def _run(self, ts_data: Any = None, **params: Any) -> Any:
        if ts_data is not None:
            params["ts_data"] = ts_data
        text = params.pop("text", None)
        if text is not None:
            params["text"] = text
        images = params.pop("images", None)
        if images is not None:
            params["images"] = images

        result = await self._client.call_tool(self._remote_name, params)
        return result


class MCPClient:
    """Connects to an MCP server via stdio subprocess and provides tool listing/calling."""

    def __init__(self, command: list[str], env: dict[str, str] | None = None):
        self._command = command
        self._env = env
        self._process: subprocess.Popen | None = None
        self._request_id = 0
        self._initialized = False

    async def connect(self) -> None:
        """Start the MCP server subprocess and send initialize handshake."""
        self._process = subprocess.Popen(
            self._command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=self._env,
        )
        response = await self._send_request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "timeclaw-mcp-client", "version": "1.0.0"},
        })
        if "error" in response:
            raise RuntimeError(f"MCP initialize failed: {response['error']}")
        self._initialized = True
        logger.info(f"[MCP Client] Connected to: {' '.join(self._command)}")

    async def list_tools(self) -> list[dict[str, Any]]:
        """Request the list of available tools from the remote server."""
        if not self._initialized:
            await self.connect()
        response = await self._send_request("tools/list", {})
        return response.get("result", {}).get("tools", [])

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        """Invoke a tool on the remote server."""
        if not self._initialized:
            await self.connect()
        response = await self._send_request("tools/call", {
            "name": name,
            "arguments": arguments,
        })
        result = response.get("result", {})
        if result.get("isError"):
            error_text = ""
            for c in result.get("content", []):
                error_text += c.get("text", "")
            raise RuntimeError(f"MCP tool call failed: {error_text}")

        output_parts = []
        for c in result.get("content", []):
            text = c.get("text", "")
            try:
                output_parts.append(json.loads(text))
            except (json.JSONDecodeError, TypeError):
                output_parts.append(text)

        return output_parts[0] if len(output_parts) == 1 else output_parts

    async def _send_request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        """Send a JSON-RPC request and read the response."""
        if self._process is None or self._process.stdin is None or self._process.stdout is None:
            raise RuntimeError("MCP server process not started")

        self._request_id += 1
        request = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
            "id": self._request_id,
        }
        line = json.dumps(request) + "\n"
        self._process.stdin.write(line.encode())
        self._process.stdin.flush()

        response_line = await asyncio.get_event_loop().run_in_executor(
            None, self._process.stdout.readline
        )
        if not response_line:
            raise RuntimeError("MCP server closed connection")

        return json.loads(response_line.decode())

    async def disconnect(self) -> None:
        if self._process:
            self._process.terminate()
            self._process.wait(timeout=5)
            self._process = None
            self._initialized = False


async def register_mcp_tools(server_config: dict[str, Any]) -> int:
    """Connect to an MCP server, discover its tools, and register them locally.

    Args:
        server_config: Dict with keys:
            - command: list[str] — command to start the MCP server
            - env: dict[str, str] — optional environment variables

    Returns:
        Number of tools registered.
    """
    command = server_config.get("command", [])
    env = server_config.get("env")

    if not command:
        logger.warning("[MCP Client] No command specified, skipping")
        return 0

    client = MCPClient(command=command, env=env)
    try:
        await client.connect()
        remote_tools = await client.list_tools()
    except Exception as e:
        logger.warning(f"[MCP Client] Failed to connect: {e}")
        return 0

    count = 0
    for tool_desc in remote_tools:
        try:
            wrapper = MCPToolWrapper(
                tool_name=tool_desc["name"],
                tool_description=tool_desc.get("description", ""),
                input_schema=tool_desc.get("inputSchema", {}),
                client=client,
            )
            register_tool_instance(wrapper)
            count += 1
            logger.debug(f"[MCP Client] Registered remote tool: {wrapper.name}")
        except Exception as e:
            logger.warning(f"[MCP Client] Failed to wrap tool '{tool_desc.get('name')}': {e}")

    logger.info(f"[MCP Client] Registered {count} remote tools from {' '.join(command)}")
    return count
