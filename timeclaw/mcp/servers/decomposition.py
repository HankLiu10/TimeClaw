"""MCP Server: timeclaw-decomposition (stl_decompose, fft_spectrum, ...)"""
from timeclaw.mcp.server_factory import create_category_server

mcp = create_category_server("timeclaw-decomposition", "decomposition")

if __name__ == "__main__":
    mcp.run(transport="stdio")
