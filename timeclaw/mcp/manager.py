"""MCPManager — unified lifecycle and dispatch for multiple MCP servers.

Manages N MCP Server sub-processes (stdio transport), exposes a flat
tool namespace to the rest of TimeClaw:

    mgr = MCPManager(server_configs)
    await mgr.start()
    tools = mgr.list_tools()            # [{name, description, inputSchema}, ...]
    result = await mgr.call_tool("prophet_forecast", {"ts_data": [...], "horizon": 10})
    await mgr.shutdown()
"""

from __future__ import annotations

import json
import os
import sys
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


def _build_subprocess_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    """Build environment dict for MCP sub-processes.

    Ensures the conda lib directory containing a compatible libstdc++ is
    prepended to LD_LIBRARY_PATH so that native extensions (matplotlib,
    torch, etc.) find the correct GLIBCXX symbols even when other
    libraries (gluonts, prophet) are loaded first in the same process.
    """
    env = dict(os.environ)
    conda_prefix = env.get("CONDA_PREFIX") or str(Path(sys.executable).parent.parent)
    conda_lib = str(Path(conda_prefix) / "lib")
    current_ld = env.get("LD_LIBRARY_PATH", "")
    if conda_lib not in current_ld:
        env["LD_LIBRARY_PATH"] = f"{conda_lib}:{current_ld}" if current_ld else conda_lib
    if extra:
        env.update(extra)
    return env


@dataclass
class MCPServerConfig:
    """Configuration for a single MCP server."""

    name: str
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] | None = None
    enabled: bool = True


@dataclass
class ToolDescriptor:
    """Flat descriptor for a tool discovered from an MCP server."""

    name: str
    description: str
    input_schema: dict[str, Any]
    server_name: str


class MCPManager:
    """Manages multiple MCP Server connections and provides a unified tool API.

    Lifecycle:
        1. ``start()`` — spawns all server sub-processes, initialises
           sessions, and discovers tools via ``tools/list``.
        2. ``list_tools()`` — returns the flat union of all tools.
        3. ``call_tool(name, args)`` — routes the call to the correct server
           session and returns the parsed result.
        4. ``shutdown()`` — gracefully closes all sessions and kills sub-processes.
    """

    def __init__(self, configs: list[MCPServerConfig] | None = None):
        self._configs = [c for c in (configs or []) if c.enabled]
        self._exit_stack = AsyncExitStack()
        self._sessions: dict[str, ClientSession] = {}
        self._tools: dict[str, ToolDescriptor] = {}
        self._started = False

    # ── Public API ─────────────────────────────────────────────────────

    async def start(self) -> None:
        """Start all configured MCP servers and discover their tools."""
        if self._started:
            return

        await self._exit_stack.__aenter__()

        for cfg in self._configs:
            try:
                await self._connect_server(cfg)
            except Exception as e:
                logger.error(f"[MCPManager] Failed to start {cfg.name}: {e}")

        self._started = True
        logger.info(
            f"[MCPManager] Started {len(self._sessions)}/{len(self._configs)} "
            f"servers, {len(self._tools)} tools total"
        )

    def list_tools(self) -> list[ToolDescriptor]:
        """Return all discovered tools across all servers."""
        return list(self._tools.values())

    def get_tool(self, name: str) -> ToolDescriptor | None:
        """Look up a single tool by name."""
        return self._tools.get(name)

    def tool_names(self) -> list[str]:
        """Return sorted list of all tool names."""
        return sorted(self._tools.keys())

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Execute a tool via MCP and return the parsed result.

        Returns dict with keys:
            success: bool
            data: Any  (parsed JSON from first text content block)
            error: str | None
        """
        desc = self._tools.get(name)
        if desc is None:
            return {"success": False, "data": None, "error": f"Tool '{name}' not found"}

        session = self._sessions.get(desc.server_name)
        if session is None:
            return {
                "success": False, "data": None,
                "error": f"Server '{desc.server_name}' not connected",
            }

        try:
            result = await session.call_tool(name, arguments)
        except Exception as e:
            return {"success": False, "data": None, "error": str(e)}

        if result.isError:
            error_text = ""
            for c in (result.content or []):
                if hasattr(c, "text"):
                    error_text += c.text
            return {"success": False, "data": None, "error": error_text or "Unknown MCP error"}

        # Parse the first text content block as JSON
        for c in (result.content or []):
            if hasattr(c, "text"):
                try:
                    data = json.loads(c.text)
                except (json.JSONDecodeError, TypeError):
                    data = c.text
                return {"success": True, "data": data, "error": None}

        return {"success": True, "data": None, "error": None}

    async def shutdown(self) -> None:
        """Gracefully stop all MCP servers."""
        if not self._started:
            return
        try:
            await self._exit_stack.aclose()
        except Exception as e:
            logger.warning(f"[MCPManager] Error during shutdown: {e}")
        self._sessions.clear()
        self._tools.clear()
        self._started = False
        logger.info("[MCPManager] All servers shut down")

    # ── Internal ───────────────────────────────────────────────────────

    async def _connect_server(self, cfg: MCPServerConfig) -> None:
        """Spawn one MCP server, initialise a session, and discover its tools."""
        params = StdioServerParameters(
            command=cfg.command,
            args=cfg.args,
            env=_build_subprocess_env(cfg.env),
        )

        # stdio_client is an async context manager that yields (read, write)
        transport = await self._exit_stack.enter_async_context(
            stdio_client(params, errlog=sys.stderr)
        )
        read_stream, write_stream = transport

        session: ClientSession = await self._exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        await session.initialize()

        self._sessions[cfg.name] = session

        # Discover tools
        tools_result = await session.list_tools()
        for t in tools_result.tools:
            desc = ToolDescriptor(
                name=t.name,
                description=t.description or "",
                input_schema=t.inputSchema or {},
                server_name=cfg.name,
            )
            if t.name in self._tools:
                logger.warning(
                    f"[MCPManager] Duplicate tool '{t.name}' from "
                    f"{cfg.name} (already from {self._tools[t.name].server_name}), "
                    f"keeping first"
                )
            else:
                self._tools[t.name] = desc

        logger.info(
            f"[MCPManager] {cfg.name}: connected, "
            f"{len(tools_result.tools)} tools discovered"
        )


# ── Default server configurations ─────────────────────────────────────

DEFAULT_SERVERS: list[MCPServerConfig] = [
    MCPServerConfig(
        name="timeclaw-processing",
        command=sys.executable,
        args=["-m", "timeclaw.mcp.servers.processing"],
    ),
    MCPServerConfig(
        name="timeclaw-profiling",
        command=sys.executable,
        args=["-m", "timeclaw.mcp.servers.profiling"],
    ),
    MCPServerConfig(
        name="timeclaw-detection",
        command=sys.executable,
        args=["-m", "timeclaw.mcp.servers.detection"],
    ),
    MCPServerConfig(
        name="timeclaw-decomposition",
        command=sys.executable,
        args=["-m", "timeclaw.mcp.servers.decomposition"],
    ),
    MCPServerConfig(
        name="timeclaw-relation",
        command=sys.executable,
        args=["-m", "timeclaw.mcp.servers.relation"],
    ),
    MCPServerConfig(
        name="timeclaw-prediction",
        command=sys.executable,
        args=["-m", "timeclaw.mcp.servers.prediction"],
    ),
    MCPServerConfig(
        name="timeclaw-validation",
        command=sys.executable,
        args=["-m", "timeclaw.mcp.servers.validation"],
    ),
    MCPServerConfig(
        name="timeclaw-text",
        command=sys.executable,
        args=["-m", "timeclaw.mcp.servers.text"],
    ),
    MCPServerConfig(
        name="timeclaw-image",
        command=sys.executable,
        args=["-m", "timeclaw.mcp.servers.image"],
    ),
    MCPServerConfig(
        name="timeclaw-generic-llm",
        command=sys.executable,
        args=["-m", "timeclaw.mcp.servers.generic_llm"],
    ),
]
