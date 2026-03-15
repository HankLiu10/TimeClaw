"""MCP Server: timeclaw-prediction (prophet_forecast, arima_forecast, ...)"""
from timeclaw.mcp.server_factory import create_category_server

mcp = create_category_server("timeclaw-prediction", "prediction")

if __name__ == "__main__":
    mcp.run(transport="stdio")
