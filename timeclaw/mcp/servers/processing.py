"""MCP Server: timeclaw-processing (normalize, handle_missing, slice, ...)"""
from timeclaw.mcp.server_factory import create_category_server

mcp = create_category_server("timeclaw-processing", "processing")

if __name__ == "__main__":
    mcp.run(transport="stdio")
