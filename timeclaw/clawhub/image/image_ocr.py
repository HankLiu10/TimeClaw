"""Image OCR tool: extract text and numerical data from images using multimodal LLM."""

from __future__ import annotations

from typing import Any

from timeclaw.clawhub.tool_base import ClawTool, ToolCategory, ToolIOType


class ImageOCRTool(ClawTool):
    """Extract text, numbers, and table data from images (reports, gauges, labels)."""

    name: str = "image_ocr"
    category: ToolCategory = ToolCategory.IMAGE
    description: str = (
        "Reads text, numbers, and labels from images using a multimodal LLM. "
        "Input: image file path. "
        "Output: dict with extracted text and structured numeric values. "
        "Use when you need to digitize charts, labels, instrument readings, or any text visible in an image. "
        "Params: path (file path to image), extract_type (what to extract, e.g. 'text', 'numbers', 'all')."
    )
    io_type: ToolIOType = ToolIOType.I_TO_R
    parameters_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to image file"},
            "extract_type": {
                "type": "string",
                "enum": ["text", "numbers", "table", "all"],
                "description": "What to extract from the image",
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

        extract_type = params.get("extract_type", "all")

        if llm is None:
            return {"error": "No LLM available for OCR", "extracted": []}

        prompts = {
            "text": (
                "Extract all readable text from this image. "
                "Return JSON with key 'text' (string of all text found)."
            ),
            "numbers": (
                "Extract all numerical values from this image. "
                "Return JSON with key 'numbers' as array of objects: "
                "{value: number, unit: string or null, label: string or null}."
            ),
            "table": (
                "Extract any table data from this image. "
                "Return JSON with keys: 'headers' (array of strings), "
                "'rows' (array of arrays), 'n_rows', 'n_cols'."
            ),
            "all": (
                "Extract all text, numbers, and table data from this image. "
                "Return JSON with keys: 'text' (all text), 'numbers' (array of "
                "{value, unit, label}), 'table' ({headers, rows} or null)."
            ),
        }

        prompt = prompts.get(extract_type, prompts["all"])

        try:
            resp = await llm.complete_multimodal(
                text=prompt,
                image_paths=[path],
                response_format="json",
            )
            if resp.parsed:
                return resp.parsed
            return {"raw_text": resp.text, "extract_type": extract_type}
        except Exception as e:
            return {"error": str(e), "extract_type": extract_type}


tool = ImageOCRTool()
