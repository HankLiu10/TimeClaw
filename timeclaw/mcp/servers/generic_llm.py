"""MCP Server: timeclaw-generic-llm (llm_reasoning, ...)"""
from timeclaw.mcp.server_factory import create_category_server

mcp = create_category_server("timeclaw-generic-llm", "generic_llm")

if __name__ == "__main__":
    mcp.run(transport="stdio")
