"""MCP (Model Context Protocol) compatibility layer for TimeClaw.

Provides:
- ClawHubMCPServer: exposes ClawHub tools as MCP tools via stdio/HTTP
- MCPToolWrapper: wraps remote MCP tools as local ClawTool instances
- register_mcp_tools: connects to external MCP servers and registers their tools
"""

from timeclaw.mcp.server import ClawHubMCPServer
from timeclaw.mcp.client import MCPToolWrapper, register_mcp_tools

__all__ = ["ClawHubMCPServer", "MCPToolWrapper", "register_mcp_tools"]
