"""Image loading tool: load and preprocess images from files or base64."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

from timeclaw.clawhub.tool_base import ClawTool, ToolCategory, ToolIOType


class ImageLoadTool(ClawTool):
    """Load images from file paths or base64 strings, returning metadata and pixel stats."""

    name: str = "image_load"
    category: ToolCategory = ToolCategory.IMAGE
    description: str = "Load an image file and return metadata (size, mode, basic pixel stats)."
    io_type: ToolIOType = ToolIOType.I_TO_R
    parameters_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to image file"},
            "b64": {"type": "string", "description": "Base64-encoded image data"},
        },
        "required": [],
    }
    is_stub: bool = False

    async def _run(self, ts_data: Any = None, **params: Any) -> Any:
        images = params.get("images")
        path = params.get("path")
        b64_data = params.get("b64")

        if images and isinstance(images, list) and len(images) > 0:
            first = images[0]
            if isinstance(first, dict):
                path = first.get("path", path)
                b64_data = first.get("data", b64_data)

        try:
            from PIL import Image
            import numpy as np
        except ImportError:
            raise ImportError("Pillow is required: pip install Pillow")

        if path:
            p = Path(path)
            if not p.exists():
                raise FileNotFoundError(f"Image not found: {path}")
            img = Image.open(p)
        elif b64_data:
            import io
            img_bytes = base64.b64decode(b64_data)
            img = Image.open(io.BytesIO(img_bytes))
        else:
            raise ValueError("Either 'path' or 'b64' parameter is required")

        arr = np.array(img)
        return {
            "width": img.width,
            "height": img.height,
            "mode": img.mode,
            "channels": len(img.getbands()),
            "dtype": str(arr.dtype),
            "pixel_mean": float(arr.mean()),
            "pixel_std": float(arr.std()),
            "pixel_min": int(arr.min()),
            "pixel_max": int(arr.max()),
        }


tool = ImageLoadTool()
