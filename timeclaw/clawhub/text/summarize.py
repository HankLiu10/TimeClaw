"""Text summarization tool: generate concise summaries using LLM or extractive fallback."""

from __future__ import annotations

import re
from typing import Any

from timeclaw.clawhub.tool_base import ClawTool, ToolCategory, ToolIOType


class SummarizeTool(ClawTool):
    """Summarize text documents, useful for condensing maintenance logs or clinical notes."""

    name: str = "summarize"
    category: ToolCategory = ToolCategory.TEXT
    description: str = (
        "Condenses text documents into key points using an LLM. "
        "Input: text data (string or list of text dicts). "
        "Output: dict with 'summary' string. "
        "Use when you need to reduce long documents before passing to other tools or to distill key findings. "
        "Params: max_sentences (int, controls summary length), model ('light' for fast/short text, 'heavy' for accurate/long text)."
    )
    io_type: ToolIOType = ToolIOType.T_TO_R
    parameters_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "max_sentences": {"type": "integer", "description": "Max sentences in summary", "default": 3},
            "model": {
                "type": "string",
                "enum": ["light", "heavy"],
                "description": "LLM tier: 'light' (fast, short text) or 'heavy' (accurate, long/complex text)",
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

        max_sentences = params.get("max_sentences", 3)
        llm = self.resolve_llm(params.get("llm"), params.get("model"))

        if llm:
            return await self._llm_summarize(combined, max_sentences, llm)
        return self._extractive_summarize(combined, max_sentences)

    def _extractive_summarize(self, text: str, max_sentences: int) -> dict[str, Any]:
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        if not sentences:
            return {"summary": text[:200], "method": "truncation", "n_sentences": 0}

        word_freq: dict[str, int] = {}
        for s in sentences:
            for w in re.findall(r'\w+', s.lower()):
                word_freq[w] = word_freq.get(w, 0) + 1

        scored = []
        for i, s in enumerate(sentences):
            words = re.findall(r'\w+', s.lower())
            score = sum(word_freq.get(w, 0) for w in words) / max(len(words), 1)
            if i == 0:
                score *= 1.5
            scored.append((score, i, s))

        scored.sort(key=lambda x: x[0], reverse=True)
        selected = sorted(scored[:max_sentences], key=lambda x: x[1])
        summary = " ".join(s for _, _, s in selected)

        return {"summary": summary, "method": "extractive", "n_sentences": len(selected)}

    async def _llm_summarize(self, text: str, max_sentences: int, llm: Any) -> dict[str, Any]:
        try:
            prompt = (
                f"Summarize the following text in at most {max_sentences} sentences. "
                f"Return JSON with key 'summary'.\n\nText: {text[:3000]}"
            )
            resp = await llm.complete(
                [{"role": "user", "content": prompt}],
                response_format="json",
                reasoning_effort="none",
            )
            if resp.parsed and "summary" in resp.parsed:
                return {"summary": resp.parsed["summary"], "method": "llm", "n_sentences": max_sentences}
        except Exception:
            pass
        return self._extractive_summarize(text, max_sentences)


tool = SummarizeTool()
