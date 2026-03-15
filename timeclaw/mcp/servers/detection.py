"""MCP Server: timeclaw-detection (detect_trend, detect_anomaly, ...)"""
from timeclaw.mcp.server_factory import create_category_server

mcp = create_category_server("timeclaw-detection", "detection")

if __name__ == "__main__":
    mcp.run(transport="stdio")
