"""Text-to-event tool: extract structured events with timestamps from text."""

from __future__ import annotations

import re
from typing import Any

from timeclaw.clawhub.tool_base import ClawTool, ToolCategory, ToolIOType


class TextToEventTool(ClawTool):
    """Convert unstructured text into structured event sequences with timestamps."""

    name: str = "text_to_event"
    category: ToolCategory = ToolCategory.TEXT
    description: str = (
        "Parses narrative text into structured event records with timestamps. "
        "Input: text data (string or list of text dicts). "
        "Output: list of dicts, each with 'event', 'timestamp', and 'type' fields. "
        "Use when you need to align news, reports, or logs with time-series data by extracting discrete events. "
        "Params: event_types (list of event categories to look for), model ('light'/'heavy')."
    )
    io_type: ToolIOType = ToolIOType.T_TO_R
    parameters_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "event_types": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Event types to look for (e.g., fault, maintenance, alert, weather)",
            },
            "model": {
                "type": "string",
                "enum": ["light", "heavy"],
                "description": "LLM tier: 'light' (fast) or 'heavy' (accurate, complex event extraction)",
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
            combined = "\n".join(texts)
        else:
            combined = str(text)

        event_types = params.get("event_types")
        llm = self.resolve_llm(params.get("llm"), params.get("model"))

        if llm:
            return await self._llm_extract(combined, event_types, llm)
        return self._regex_extract(combined, event_types)

    def _regex_extract(self, text: str, event_types: list[str] | None = None) -> dict[str, Any]:
        events = []

        date_pattern = r'(\d{4}[-/]\d{1,2}[-/]\d{1,2}(?:\s+\d{1,2}:\d{2}(?::\d{2})?)?)'
        sentences = re.split(r'[.\n]+', text)

        event_keywords = {
            "fault": ["fault", "failure", "error", "malfunction", "breakdown", "defect"],
            "maintenance": ["maintenance", "repair", "replace", "fix", "service", "overhaul"],
            "alert": ["alert", "alarm", "warning", "critical", "threshold", "exceed"],
            "change": ["change", "update", "modify", "adjust", "increase", "decrease"],
        }

        if event_types:
            event_keywords = {k: v for k, v in event_keywords.items() if k in event_types}

        for sent in sentences:
            sent = sent.strip()
            if len(sent) < 5:
                continue

            date_match = re.search(date_pattern, sent)
            timestamp = date_match.group(1) if date_match else None

            sent_lower = sent.lower()
            for etype, keywords in event_keywords.items():
                if any(kw in sent_lower for kw in keywords):
                    events.append({
                        "type": etype,
                        "timestamp": timestamp,
                        "description": sent.strip()[:200],
                    })
                    break

        return {"events": events, "n_events": len(events), "method": "regex"}

    async def _llm_extract(self, text: str, event_types: list[str] | None, llm: Any) -> dict[str, Any]:
        try:
            types_str = ", ".join(event_types) if event_types else "fault, maintenance, alert, change"
            prompt = (
                f"Extract structured events from this text. Return JSON with key 'events' "
                f"as array of objects with keys: type ({types_str}), timestamp (ISO format or null), "
                f"description (brief).\n\nText: {text[:3000]}"
            )
            resp = await llm.complete([{"role": "user", "content": prompt}], response_format="json")
            if resp.parsed and "events" in resp.parsed:
                return {"events": resp.parsed["events"], "n_events": len(resp.parsed["events"]), "method": "llm"}
        except Exception:
            pass
        return self._regex_extract(text, event_types)


tool = TextToEventTool()
