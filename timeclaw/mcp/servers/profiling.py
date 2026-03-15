"""MCP Server: timeclaw-profiling (basic_stats, autocorrelation, ...)"""
from timeclaw.mcp.server_factory import create_category_server

mcp = create_category_server("timeclaw-profiling", "profiling")

if __name__ == "__main__":
    mcp.run(transport="stdio")
