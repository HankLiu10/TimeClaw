"""Read original raw text artifacts by reference."""

from __future__ import annotations

from typing import Any

from timeclaw.clawhub.tool_base import ClawTool, ToolCategory, ToolIOType
from timeclaw.text.task_pruning import load_raw_text_artifact


class ReadRawTextTool(ClawTool):
    name: str = "read_raw_text"
    category: ToolCategory = ToolCategory.TEXT
    description: str = (
        "Reads the original raw text source by raw_text_ref when the cleaned task text may have omitted "
        "details you want to verify. Use this only to inspect the original source; it does not analyze or summarize."
    )
    io_type: ToolIOType = ToolIOType.T_TO_R
    parameters_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "raw_text_ref": {
                "type": "string",
                "description": "Reference id for the stored original text artifact.",
            },
            "max_chars": {
                "type": "integer",
                "description": "Optional character cap. Omit this to return the full stored raw text.",
            },
        },
        "required": ["raw_text_ref"],
    }

    async def _run(self, ts_data: Any = None, **params: Any) -> Any:
        raw_text_ref = str(params.get("raw_text_ref") or "").strip()
        if not raw_text_ref:
            raise ValueError("raw_text_ref is required")
        payload = load_raw_text_artifact(raw_text_ref)
        text = str(payload.get("text") or "")
        max_chars_raw = params.get("max_chars")
        max_chars = int(max_chars_raw) if max_chars_raw is not None else None
        if max_chars is not None and max_chars > 0:
            visible_text = text[:max_chars]
            truncated = len(text) > max_chars
        else:
            visible_text = text
            truncated = False
        return {
            "raw_text_ref": raw_text_ref,
            "text": visible_text,
            "truncated": truncated,
            "total_chars": len(text),
            "dataset": payload.get("dataset"),
            "file": payload.get("file"),
            "domain": payload.get("domain"),
        }


tool = ReadRawTextTool()
