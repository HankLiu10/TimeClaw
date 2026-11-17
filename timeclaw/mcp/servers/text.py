"""MCP Server: timeclaw-text (summarize, sentiment, ner, ...)"""
from timeclaw.mcp.server_factory import create_category_server

mcp = create_category_server("timeclaw-text", "text")

if __name__ == "__main__":
    mcp.run(transport="stdio")
