"""LLM-based indicator / general reasoning tool.

Uses the agent's LLM backend to directly analyze time-series data and
produce structured predictions (e.g. max/min/diff, trend labels, MACD).
Bypasses statistical forecast models, which avoids the over-smoothing
problem that plagues traditional models on indicator-style tasks.

Domain knowledge is NOT hardcoded here — it is injected externally via
the ``domain`` parameter (from pipeline input or memory).
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from timeclaw.clawhub.tool_base import ClawTool, ToolCategory, ToolIOType


def _fmt_array(arr: np.ndarray, max_points: int = 0) -> str:
    """Format array; max_points=0 means no truncation."""
    flat = arr.ravel()
    if max_points > 0 and len(flat) > max_points:
        flat = flat[:max_points]
    return ", ".join(f"{v:.2f}" for v in flat)


class LLMIndicatorTool(ClawTool):
    """General-purpose LLM reasoning tool for time-series analysis.

    Instead of running a statistical model, this tool sends the raw
    time-series values (and optional text context) directly to a large
    language model, asking it to produce a structured JSON answer.

    Best for tasks where human-like pattern recognition outperforms
    statistical extrapolation — indicator extraction, trend
    classification, anomaly explanation, and question answering.
    """

    name: str = "llm_indicator"
    category: ToolCategory = ToolCategory.PREDICTION
    description: str = (
        "Send raw time-series data (and optional text/image context) to a large "
        "language model for direct structured reasoning. The LLM reads the actual "
        "data values and produces a JSON answer. Suitable for any task that "
        "benefits from pattern recognition over raw values: indicator extraction, "
        "trend classification, anomaly explanation, and question answering. "
        "Provide 'question' to specify what to analyze and 'output_schema' to "
        "define the expected JSON output format. The optional 'domain' parameter "
        "accepts a free-text domain description for richer context."
    )
    io_type: ToolIOType = ToolIOType.TX_TO_R
    parameters_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "What to predict or analyze.",
            },
            "output_schema": {
                "type": "string",
                "description": (
                    'Expected JSON output keys. Example: \'{"max": number, "min": number, "diff": number}\''
                ),
            },
            "context": {
                "type": "string",
                "description": "Additional domain context to include in the prompt.",
            },
            "domain": {
                "type": "string",
                "description": "Free-text domain description injected into the prompt.",
            },
        },
        "required": ["question"],
    }
    is_stub: bool = False

    async def _run(self, ts_data: Any = None, **params: Any) -> Any:
        llm = self.resolve_llm(params.get("llm"), "heavy")
        if llm is None:
            try:
                from timeclaw.core.config import TimeClawConfig
                from timeclaw.llm.router import ModelRouter
                cfg = TimeClawConfig.load()
                router = ModelRouter(cfg.llm)
                llm = router.get_backend("executor")
            except Exception:
                raise ValueError("LLM backend is required for llm_indicator")

        question = params.get("question", "Analyze this time-series data.")
        output_schema = params.get("output_schema", '{"answer": "..."}')
        context = params.get("context", "")
        domain = params.get("domain", "")

        ts_section = ""
        if ts_data is not None:
            arr = np.asarray(ts_data, dtype=np.float64).ravel()
            n_points = len(arr)
            ts_section = f"Time-series data ({n_points} data points):\n[{_fmt_array(arr)}]\n\n"

        domain_section = f"Domain context:\n{domain}\n\n" if domain else ""

        text_section = ""
        text = params.get("text")
        if text:
            if isinstance(text, list):
                texts = [self.extract_text_content(t) for t in text[:10]]
                text_section = "Contextual information:\n" + "\n".join(texts[:10000]) + "\n\n"
            else:
                text_section = f"Contextual information:\n{str(text)[:10000]}\n\n"

        system_prompt = (
            "You are a data analysis expert. Analyze the provided data "
            "and any contextual information to answer the question.\n"
            f"Return ONLY a JSON object matching this schema: {output_schema}"
        )
        user_prompt = f"{domain_section}{ts_section}{text_section}{context}\n\nQuestion: {question}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        resp = await llm.complete(messages, response_format="json")

        if resp.parsed:
            return resp.parsed
        if resp.text:
            try:
                return json.loads(resp.text)
            except (json.JSONDecodeError, TypeError):
                pass
            return {"raw_response": resp.text}
        raise ValueError("LLM returned empty response")


tool = LLMIndicatorTool()
