"""Image feature extraction tool: describe image content using multimodal LLM."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

from timeclaw.clawhub.tool_base import ClawTool, ToolCategory, ToolIOType


class ImageFeatureTool(ClawTool):
    """Extract visual features / descriptions from images using multimodal LLM."""

    name: str = "image_feature"
    category: ToolCategory = ToolCategory.IMAGE
    description: str = (
        "Uses a vision-language model to describe what is in an image and extract semantic visual features. "
        "Input: image file path. "
        "Output: dict with 'description' text and 'features' list. "
        "Use when you need to understand image content in context with time-series data, e.g. interpreting visual conditions or scenes. "
        "Params: path (file path to image), prompt (optional guidance for the model on what to look for)."
    )
    io_type: ToolIOType = ToolIOType.I_TO_R
    parameters_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to image file"},
            "prompt": {
                "type": "string",
                "description": "Custom prompt for feature extraction",
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

        custom_prompt = params.get("prompt", "Describe this image in detail. "
            "Focus on any equipment, gauges, defects, patterns, or anomalies visible.")

        if llm is not None:
            return await self._llm_describe(path, custom_prompt, llm)
        return self._basic_features(path)

    def _basic_features(self, path: str) -> dict[str, Any]:
        try:
            from PIL import Image
            import numpy as np

            img = Image.open(path)
            arr = np.array(img)
            features = {
                "width": img.width,
                "height": img.height,
                "mode": img.mode,
                "description": "LLM not available; returning pixel statistics only.",
                "brightness": float(arr.mean()),
                "contrast": float(arr.std()),
            }
            if len(arr.shape) == 3 and arr.shape[2] >= 3:
                features["dominant_channel"] = ["R", "G", "B"][
                    int(arr[:, :, :3].mean(axis=(0, 1)).argmax())
                ]
            return features
        except ImportError:
            return {"error": "Pillow not installed", "description": "Cannot analyze image"}

    async def _llm_describe(
        self, path: str, prompt: str, llm: Any
    ) -> dict[str, Any]:
        try:
            resp = await llm.complete_multimodal(
                text=prompt,
                image_paths=[path],
            )
            return {
                "description": resp.text or "No description generated",
                "method": "multimodal_llm",
                "model": resp.model,
            }
        except Exception as e:
            result = self._basic_features(path)
            result["llm_error"] = str(e)
            return result


tool = ImageFeatureTool()
