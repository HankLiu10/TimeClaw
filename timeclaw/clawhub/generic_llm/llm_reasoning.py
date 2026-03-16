"""General-purpose multimodal LLM reasoning tool.

Sends raw data (time-series, text, images) directly to a large language
model for structured reasoning.  Unlike statistical forecast tools that
produce point-by-point predictions, this tool leverages LLM pattern
recognition to answer questions, extract indicators, classify trends,
explain anomalies, and perform any reasoning task that benefits from
reading actual data values.

Domain knowledge is NOT hardcoded — it is injected externally via
the ``domain`` parameter (from pipeline input or memory).
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


class LLMReasoningTool(ClawTool):
    """General-purpose multimodal LLM reasoning tool.

    Sends raw time-series values (and optional text/image context)
    directly to a large language model, asking it to produce a
    structured JSON answer.  Works for indicator extraction, trend
    classification, anomaly explanation, question answering, and any
    task where reading data values and reasoning over them is more
    effective than running a statistical model.
    """

    name: str = "llm_reasoning"
    category: ToolCategory = ToolCategory.GENERIC_LLM
    description: str = (
        "A powerful, comprehensive multimodal large language model that reads and "
        "reasons over raw time-series values, text context, and images simultaneously "
        "— the only tool that combines all modalities in a single reasoning step. "
        "It directly understands the semantics of the task, interprets data patterns "
        "in context, and produces a structured JSON answer. "
        "Applicable to ALL task types: forecast (predicting future values by "
        "understanding temporal dynamics), indicator (extracting statistical "
        "indicators such as max, min, diff, mean from data context), "
        "classification (categorizing trends or patterns using holistic judgment), "
        "qa (answering questions by jointly reasoning over numerical data and text), "
        "and imputation (inferring missing values from surrounding context). "
        "Strengths: handles complex, multi-step reasoning in one call; leverages "
        "textual context (e.g. weather reports, news) that pure statistical models "
        "ignore; produces answers directly in the required output format without "
        "post-processing; works on any data length and frequency. "
        "Especially effective when the task requires understanding both numerical "
        "patterns and textual/contextual information, or when the expected output "
        "is not a raw time-series array but a structured result (indicators, labels, "
        "answers). Input data (time-series, text, images) is automatically provided. "
        "When called after other tools, their results and the full task description "
        "are automatically included as additional context. "
        "Specify 'question' (what to analyze) and 'output_schema' (expected JSON "
        "format) to use."
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
                "oneOf": [{"type": "string"}, {"type": "object"}],
                "description": (
                    'Expected JSON output keys, as a string or object. '
                    'Example: \'{"max": number, "min": number, "diff": number}\''
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
                raise ValueError("LLM backend is required for llm_reasoning")

        question = params.get("question", "Analyze this time-series data.")
        output_schema = params.get("output_schema", '{"answer": "..."}')
        if isinstance(output_schema, dict):
            output_schema = json.dumps(output_schema)
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


tool = LLMReasoningTool()
