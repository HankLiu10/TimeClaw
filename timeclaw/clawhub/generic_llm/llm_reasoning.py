"""Task-specialized multimodal LLM reasoning tools.

Each wrapper exposes a narrower task boundary while sharing one common
implementation. This avoids mixing unrelated usage experience under a
single generic tool name.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from timeclaw.clawhub.tool_base import ClawTool, ToolCategory, ToolIOType


def _fmt_array(arr: np.ndarray, max_points: int = 0) -> str:
    flat = arr.ravel()
    if max_points > 0 and len(flat) > max_points:
        flat = flat[:max_points]
    return ", ".join(f"{v:.2f}" for v in flat)


class _BaseLLMReasoningTool(ClawTool):
    category: ToolCategory = ToolCategory.GENERIC_LLM
    io_type: ToolIOType = ToolIOType.TX_TO_R
    parameters_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "What to predict or analyze.",
            },
            "output_schema": {
                "oneOf": [{"type": "string"}, {"type": "object"}],
                "description": "Expected JSON output schema.",
            },
            "context": {
                "type": "string",
                "description": "Additional task context to include in the prompt.",
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
                raise ValueError(f"LLM backend is required for {self.name}")

        question = params.get("question", "Analyze this data.")
        output_schema = params.get("output_schema", '{"answer": "..."}')
        if isinstance(output_schema, dict):
            output_schema = json.dumps(output_schema, ensure_ascii=False)
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
            f"You are a specialist for {self.task_boundary}. "
            "Analyze the provided data and contextual information carefully.\n"
            "Return ONLY a JSON object matching this schema: "
            f"{output_schema}"
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
                return {"raw_response": resp.text}
        raise ValueError("LLM returned empty response")


class LLMReasoningForecastTool(_BaseLLMReasoningTool):
    name = "llm_reasoning_forecast"
    task_boundary = "time-series forecasting"
    description = (
        "Multimodal LLM reasoning wrapper specialized for forecast tasks. "
        "Use when the goal is to predict future numeric values from time-series "
        "data, optionally using text or image context."
    )


class LLMReasoningIndicatorTool(_BaseLLMReasoningTool):
    name = "llm_reasoning_indicator"
    task_boundary = "indicator extraction"
    description = (
        "Multimodal LLM reasoning wrapper specialized for indicator tasks. "
        "Use when the goal is to extract numeric indicators such as max, min, "
        "diff, MACD, or Bollinger-related values."
    )


class LLMReasoningClassificationTool(_BaseLLMReasoningTool):
    name = "llm_reasoning_classification"
    task_boundary = "classification and trend labeling"
    description = (
        "Multimodal LLM reasoning wrapper specialized for classification tasks. "
        "Use when the output is a trend label, class label, or structured label object."
    )


class LLMReasoningQATool(_BaseLLMReasoningTool):
    name = "llm_reasoning_qa"
    task_boundary = "question answering"
    description = (
        "Multimodal LLM reasoning wrapper specialized for QA tasks. "
        "Use when the task asks a question over time-series plus optional context."
    )


class LLMReasoningCorrelationTool(_BaseLLMReasoningTool):
    name = "llm_reasoning_correlation"
    task_boundary = "correlation classification"
    description = (
        "Multimodal LLM reasoning wrapper specialized for correlation tasks. "
        "Use when the output is a correlation label or correlation judgment."
    )


tool_forecast = LLMReasoningForecastTool()
tool_indicator = LLMReasoningIndicatorTool()
tool_classification = LLMReasoningClassificationTool()
tool_qa = LLMReasoningQATool()
tool_correlation = LLMReasoningCorrelationTool()
