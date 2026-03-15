"""Image classification tool: classify images using multimodal LLM."""

from __future__ import annotations

from typing import Any

from timeclaw.clawhub.tool_base import ClawTool, ToolCategory, ToolIOType


class ImageClassifyTool(ClawTool):
    """Classify images into predefined categories using multimodal LLM."""

    name: str = "image_classify"
    category: ToolCategory = ToolCategory.IMAGE
    description: str = (
        "Assigns an image to one of the provided categories using a vision-language model. "
        "Input: image file path and a list of candidate categories. "
        "Output: dict with predicted 'category' and 'confidence' score. "
        "Use when you need to categorize visual patterns, conditions, or image types from a known set of labels. "
        "Params: path (file path to image), categories (list of category labels to classify into)."
    )
    io_type: ToolIOType = ToolIOType.I_TO_R
    parameters_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to image file"},
            "categories": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of possible categories",
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

        categories = params.get("categories", [
            "normal", "fault", "degraded", "anomaly", "maintenance_needed"
        ])

        if llm is None:
            return {
                "category": "unknown",
                "confidence": 0.0,
                "reason": "No LLM available for classification",
            }

        cats_str = ", ".join(categories)
        prompt = (
            f"Classify this image into exactly one of these categories: {cats_str}. "
            "Return JSON with keys: category, confidence (0-1), reason (brief explanation)."
        )

        try:
            resp = await llm.complete_multimodal(
                text=prompt,
                image_paths=[path],
                response_format="json",
            )
            if resp.parsed:
                result = resp.parsed
                if "category" not in result:
                    result["category"] = "unknown"
                return result
            return {
                "category": "unknown",
                "confidence": 0.0,
                "raw_response": resp.text,
            }
        except Exception as e:
            return {
                "category": "unknown",
                "confidence": 0.0,
                "error": str(e),
            }


tool = ImageClassifyTool()
