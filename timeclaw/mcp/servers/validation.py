"""MCP Server: timeclaw-validation (bound_check, consistency_check, ...)"""
from timeclaw.mcp.server_factory import create_category_server

mcp = create_category_server("timeclaw-validation", "validation")

if __name__ == "__main__":
    mcp.run(transport="stdio")
