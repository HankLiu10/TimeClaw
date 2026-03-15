"""MCP (Model Context Protocol) integration for TimeClaw.

Architecture:
    - 9 category-based MCP Servers (stdio transport) expose all 63 ClawHub tools
    - MCPManager manages server lifecycle and provides unified tool API
    - PipelineRunner and Executor route tool calls through MCPManager

Components:
    - server_factory: Wraps ClawTool instances as FastMCP tools
    - servers/: Entry scripts for each category server
    - manager: MCPManager lifecycle + dispatch
    - server/client: Legacy compatibility (retained for external MCP)
"""

from timeclaw.mcp.manager import MCPManager, MCPServerConfig, ToolDescriptor
from timeclaw.mcp.server_factory import create_category_server

__all__ = [
    "MCPManager",
    "MCPServerConfig",
    "ToolDescriptor",
    "create_category_server",
]
