"""Named Entity Recognition tool: extract entities from text."""

from __future__ import annotations

import re
from typing import Any

from timeclaw.clawhub.tool_base import ClawTool, ToolCategory, ToolIOType


_PATTERNS = {
    "DATE": r'\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b',
    "TIME": r'\b\d{1,2}:\d{2}(?::\d{2})?\b',
    "NUMBER": r'\b\d+\.?\d*%?\b',
    "MONEY": r'\$\d+(?:,\d{3})*(?:\.\d{2})?',
}


class NERTool(ClawTool):
    """Extract named entities (dates, numbers, organizations, etc.) from text."""

    name: str = "ner"
    category: ToolCategory = ToolCategory.TEXT
    description: str = (
        "Identifies named entities (dates, numbers, organizations, locations, etc.) from text using regex or LLM. "
        "Input: text data (string or list of text dicts). "
        "Output: dict with entities grouped by type. "
        "Use when you need to extract structured facts from unstructured text, e.g. pulling dates and company names from reports. "
        "Params: entity_types (list of entity categories to extract), model ('light'/'heavy')."
    )
    io_type: ToolIOType = ToolIOType.T_TO_R
    parameters_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "entity_types": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Entity types to extract (DATE, TIME, NUMBER, MONEY, PERSON, ORG, LOCATION, EVENT)",
            },
            "model": {
                "type": "string",
                "enum": ["light", "heavy"],
                "description": "LLM tier: 'light' (fast, simple entities) or 'heavy' (complex entity extraction)",
            },
        },
        "required": [],
    }
    is_stub: bool = False

    async def _run(self, ts_data: Any = None, **params: Any) -> Any:
        text = params.get("text")
        if text is None:
            raise ValueError("text parameter is required")

        if isinstance(text, list):
            texts = [self.extract_text_content(t) for t in text]
        else:
            texts = [str(text)]

        entity_types = params.get("entity_types")
        llm = self.resolve_llm(params.get("llm"), params.get("model"))

        all_entities = []
        for t in texts:
            if llm:
                entities = await self._llm_ner(t, llm, entity_types)
            else:
                entities = self._regex_ner(t, entity_types)
            all_entities.append(entities)

        return {"entities": all_entities, "n_texts": len(texts)}

    def _regex_ner(self, text: str, entity_types: list[str] | None = None) -> list[dict[str, str]]:
        patterns = _PATTERNS if entity_types is None else {k: v for k, v in _PATTERNS.items() if k in entity_types}
        entities = []
        for etype, pattern in patterns.items():
            for match in re.finditer(pattern, text):
                entities.append({"type": etype, "value": match.group(), "start": match.start(), "end": match.end()})
        entities.sort(key=lambda e: e["start"])
        return entities

    async def _llm_ner(self, text: str, llm: Any, entity_types: list[str] | None = None) -> list[dict[str, str]]:
        try:
            types_str = ", ".join(entity_types) if entity_types else "DATE, TIME, NUMBER, PERSON, ORG, LOCATION, EVENT"
            prompt = (
                f"Extract named entities from this text. Return JSON array of objects with keys: "
                f"type, value, start, end. Entity types to find: {types_str}.\n"
                f"Text: {text[:1000]}"
            )
            resp = await llm.complete([{"role": "user", "content": prompt}], response_format="json")
            if resp.parsed and isinstance(resp.parsed, list):
                return resp.parsed
            if resp.parsed and isinstance(resp.parsed, dict) and "entities" in resp.parsed:
                return resp.parsed["entities"]
        except Exception:
            pass
        return self._regex_ner(text, entity_types)


tool = NERTool()
