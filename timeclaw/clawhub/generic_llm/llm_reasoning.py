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

    output_contract: str = '{"answer": "..."}'

    def _normalise_payload(self, payload: Any) -> dict[str, Any]:
        if isinstance(payload, dict) and "answer" in payload:
            return payload
        return {"answer": payload}

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
                text_section = "Contextual information:\n" + "\n".join(texts) + "\n\n"
            else:
                text_section = f"Contextual information:\n{str(text)}\n\n"

        system_prompt = (
            f"You are a specialist for {self.task_boundary}. "
            "Analyze the provided data and contextual information carefully.\n"
            "Return ONLY a JSON object matching this schema: "
            f"{self.output_contract}"
        )
        user_prompt = f"{domain_section}{ts_section}{text_section}{context}\n\nQuestion: {question}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        resp = await llm.complete(messages, response_format="json")

        if resp.parsed:
            return self._normalise_payload(resp.parsed)
        if resp.text:
            try:
                return self._normalise_payload(json.loads(resp.text))
            except (json.JSONDecodeError, TypeError):
                raise ValueError("LLM returned non-JSON response")
        raise ValueError("LLM returned empty response")


class LLMReasoningForecastTool(_BaseLLMReasoningTool):
    name = "llm_reasoning_forecast"
    task_boundary = "time-series forecasting"
    description = (
        "Multimodal LLM reasoning wrapper specialized for forecast tasks. "
        "Use when the goal is to predict future numeric values from time-series "
        "data, optionally using text or image context. "
        "Returns a fixed payload: {\"answer\": [numbers]}."
    )
    output_contract = '{"answer": [number, number, ...]}'

    def _normalise_payload(self, payload: Any) -> dict[str, Any]:
        if isinstance(payload, dict):
            if "answer" in payload:
                answer = payload["answer"]
            else:
                answer = None
                for key in ("forecast", "point_forecast", "prediction", "values", "temperatures", "predictions", "data"):
                    if key in payload:
                        answer = payload[key]
                        break
                if answer is None:
                    answer = payload
        else:
            answer = payload
        return {"answer": answer}


class LLMReasoningIndicatorTool(_BaseLLMReasoningTool):
    name = "llm_reasoning_indicator"
    task_boundary = "indicator extraction"
    description = (
        "Multimodal LLM reasoning wrapper specialized for indicator tasks. "
        "Use when the goal is to extract numeric indicators such as max, min, "
        "diff, MACD, or Bollinger-related values. "
        "Returns a fixed payload: {\"answer\": {indicator_name: values}}."
    )
    output_contract = '{"answer": {"indicator_name": [number, ...]}}'

    def _normalise_payload(self, payload: Any) -> dict[str, Any]:
        if isinstance(payload, dict) and "answer" in payload:
            answer = payload["answer"]
        else:
            answer = payload
        return {"answer": answer}


class LLMReasoningClassificationTool(_BaseLLMReasoningTool):
    name = "llm_reasoning_classification"
    task_boundary = "classification and trend labeling"
    description = (
        "Multimodal LLM reasoning wrapper specialized for classification tasks. "
        "Use when the output is a trend label, class label, or structured label object. "
        "Returns a fixed payload: {\"answer\": \"label\"}."
    )
    output_contract = '{"answer": "label"}'

    def _normalise_payload(self, payload: Any) -> dict[str, Any]:
        if isinstance(payload, dict):
            if "answer" in payload:
                answer = payload["answer"]
            else:
                answer = None
                for key in ("trend", "label", "prediction"):
                    if key in payload:
                        answer = payload[key]
                        break
                if answer is None:
                    answer = payload
        else:
            answer = payload
        return {"answer": answer}


class LLMReasoningQATool(_BaseLLMReasoningTool):
    name = "llm_reasoning_qa"
    task_boundary = "question answering"
    description = (
        "Multimodal LLM reasoning wrapper specialized for QA tasks. "
        "Use when the task asks a question over time-series plus optional context. "
        "Returns a fixed payload: {\"answer\": \"text\"}."
    )
    output_contract = '{"answer": "text"}'


class LLMReasoningCorrelationTool(_BaseLLMReasoningTool):
    name = "llm_reasoning_correlation"
    task_boundary = "correlation classification"
    description = (
        "Multimodal LLM reasoning wrapper specialized for correlation tasks. "
        "Use when the output is a correlation label or correlation judgment. "
        "Returns a fixed payload: {\"answer\": \"label\"}."
    )
    output_contract = '{"answer": "label"}'

    def _normalise_payload(self, payload: Any) -> dict[str, Any]:
        if isinstance(payload, dict):
            if "answer" in payload:
                answer = payload["answer"]
            else:
                answer = payload.get("correlation", payload)
        else:
            answer = payload
        return {"answer": answer}


tool_forecast = LLMReasoningForecastTool()
tool_indicator = LLMReasoningIndicatorTool()
tool_classification = LLMReasoningClassificationTool()
tool_qa = LLMReasoningQATool()
tool_correlation = LLMReasoningCorrelationTool()
