"""MCP Server: timeclaw-image (image_load, image_feature, ...)"""
from timeclaw.mcp.server_factory import create_category_server

mcp = create_category_server("timeclaw-image", "image")

if __name__ == "__main__":
    mcp.run(transport="stdio")
