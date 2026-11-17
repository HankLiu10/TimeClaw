"""MCP Server: timeclaw-relation (correlation, granger_causality, ...)"""
from timeclaw.mcp.server_factory import create_category_server

mcp = create_category_server("timeclaw-relation", "relation")

if __name__ == "__main__":
    mcp.run(transport="stdio")
