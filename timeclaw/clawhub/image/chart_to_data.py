"""Chart-to-data tool: parse chart images into numerical data using multimodal LLM."""

from __future__ import annotations

from typing import Any

from timeclaw.clawhub.tool_base import ClawTool, ToolCategory, ToolIOType


class ChartToDataTool(ClawTool):
    """Convert chart/graph images into structured numerical data."""

    name: str = "chart_to_data"
    category: ToolCategory = ToolCategory.IMAGE
    description: str = (
        "Extracts numerical data points from chart or graph images using a multimodal LLM. "
        "Input: image file path of a chart/plot. "
        "Output: dict with extracted 'values' array and 'axis_labels'. "
        "Use when you only have a screenshot or photo of a plot and need to recover the underlying numeric time-series data. "
        "Params: path (file path to chart image), chart_type (hint about chart format, e.g. 'line', 'bar')."
    )
    io_type: ToolIOType = ToolIOType.I_TO_R
    parameters_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to chart image"},
            "chart_type": {
                "type": "string",
                "enum": ["line", "bar", "scatter", "auto"],
                "description": "Expected chart type",
            },
        },
        "required": [],
    }
    is_stub: bool = False

    async def _run(self, ts_data: Any = None, **params: Any) -> Any:
        images = params.get("images")
        path = params.get("path")
        llm = params.get("llm")

        if images and isinstance(images, list) and len(images) > 0:
            first = images[0]
            if isinstance(first, dict):
                path = first.get("path", path)
            elif isinstance(first, str):
                path = first

        if not path:
            raise ValueError("Image path is required")

        chart_type = params.get("chart_type", "auto")

        if llm is None:
            return {"error": "No LLM available for chart parsing", "data_points": []}

        prompt = (
            f"This image contains a {'chart' if chart_type == 'auto' else chart_type + ' chart'}. "
            "Extract the data points from this chart. Return JSON with keys:\n"
            "- chart_type: detected chart type (line/bar/scatter/pie/other)\n"
            "- title: chart title if visible\n"
            "- x_label: x-axis label\n"
            "- y_label: y-axis label\n"
            "- series: array of {name: string, data_points: [{x: value, y: value}]}\n"
            "Extract as many data points as you can read from the chart."
        )

        try:
            resp = await llm.complete_multimodal(
                text=prompt,
                image_paths=[path],
                response_format="json",
            )
            if resp.parsed:
                return resp.parsed
            return {"raw_text": resp.text, "chart_type": chart_type}
        except Exception as e:
            return {"error": str(e), "chart_type": chart_type, "data_points": []}


tool = ChartToDataTool()
