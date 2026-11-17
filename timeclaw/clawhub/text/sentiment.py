"""Sentiment analysis tool: analyze text sentiment using LLM or rule-based fallback."""

from __future__ import annotations

import re
from typing import Any

from timeclaw.clawhub.tool_base import ClawTool, ToolCategory, ToolIOType


_POSITIVE_WORDS = {
    "good", "great", "excellent", "positive", "increase", "growth", "improve",
    "profit", "gain", "success", "up", "rise", "strong", "better", "high",
    "recovery", "bullish", "optimistic", "stable", "healthy",
}
_NEGATIVE_WORDS = {
    "bad", "poor", "negative", "decrease", "decline", "loss", "fail",
    "down", "drop", "weak", "worse", "low", "crash", "bearish",
    "pessimistic", "unstable", "risk", "danger", "critical", "alarm",
}

_POSITIVE_WORDS_ZH = {
    "稳定", "回升", "增长", "上涨", "利好", "改善", "恢复", "好转",
    "乐观", "健康", "正常", "温和", "平稳", "晴", "暖", "升温",
}
_NEGATIVE_WORDS_ZH = {
    "骤降", "暴跌", "下降", "恶化", "危险", "预警", "警报", "寒潮",
    "冷空气", "暴雨", "暴风", "灾害", "紧急", "严重", "异常", "风险",
    "降温", "低温", "冰冻", "霜冻", "雷暴", "台风", "洪水", "干旱",
}


class SentimentTool(ClawTool):
    """Analyze sentiment of text data, returning polarity and confidence."""

    name: str = "sentiment"
    category: ToolCategory = ToolCategory.TEXT
    description: str = (
        "Determines the emotional tone of text, classifying it as positive, negative, or neutral. "
        "Input: text data (string or list of text dicts). "
        "Output: a dict containing polarity/score/confidence and, when the text supports it, "
        "target-aware stance fields such as 'target_entity', 'target_specificity', "
        "'stance', 'stance_strength', and 'catalyst_strength'. "
        "Use when you need to understand market sentiment, event impact, or opinion polarity from text. "
        "For finance/news text, this tool can help separate target-stock-specific stance from "
        "generic document tone. "
        "Params: model ('light' for fast/short text, 'heavy' for accurate/complex text), "
        "optional target_entity (ticker/company/entity to focus on)."
    )
    io_type: ToolIOType = ToolIOType.T_TO_R
    parameters_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "model": {
                "type": "string",
                "enum": ["light", "heavy"],
                "description": "LLM tier: 'light' (fast, short text) or 'heavy' (accurate, complex text)",
            },
            "target_entity": {
                "type": "string",
                "description": "Optional entity/ticker/company to focus on when extracting target-specific stance",
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

        llm = self.resolve_llm(params.get("llm"), params.get("model"))
        target_entity = str(params.get("target_entity") or "").strip() or None

        results = []
        for t in texts:
            if llm:
                r = await self._llm_sentiment(t, llm, target_entity=target_entity)
            else:
                r = self._rule_sentiment(t)
            results.append(r)

        payload: dict[str, Any] = {"sentiments": results, "n_texts": len(texts)}
        if len(results) == 1:
            payload.update(results[0])
            payload["sentiment"] = results[0].get("polarity")
        return payload

    @staticmethod
    def _strength_from_score(score: float) -> str:
        magnitude = abs(float(score))
        if magnitude >= 0.66:
            return "strong"
        if magnitude >= 0.33:
            return "moderate"
        return "weak"

    @staticmethod
    def _normalize_payload(payload: dict[str, Any], *, fallback_target: str | None) -> dict[str, Any]:
        polarity = str(payload.get("polarity") or payload.get("sentiment") or "").strip().lower()
        if polarity not in {"positive", "negative", "neutral"}:
            polarity = "neutral"

        raw_stance = str(payload.get("stance") or payload.get("target_stance") or payload.get("article_stance") or "").strip().lower()
        if raw_stance not in {"positive", "negative", "mixed", "neutral"}:
            raw_stance = polarity

        try:
            score = float(payload.get("score"))
        except Exception:
            score = 0.0
        score = max(-1.0, min(1.0, score))

        try:
            confidence = float(payload.get("confidence"))
        except Exception:
            confidence = 0.5
        confidence = max(0.0, min(1.0, confidence))

        stance_strength = str(payload.get("stance_strength") or "").strip().lower()
        if stance_strength not in {"strong", "moderate", "weak"}:
            stance_strength = SentimentTool._strength_from_score(score)

        target_specificity = str(payload.get("target_specificity") or payload.get("specificity") or "").strip().lower()
        if target_specificity not in {"target_specific", "mixed", "generic", "unknown"}:
            target_specificity = "unknown"

        catalyst_strength = str(payload.get("catalyst_strength") or "").strip().lower()
        if catalyst_strength not in {"explicit", "weak", "none", "unknown"}:
            catalyst_strength = "unknown"

        target_entity = str(payload.get("target_entity") or fallback_target or "").strip() or None
        evidence = str(payload.get("evidence") or payload.get("why") or "").strip()

        return {
            "polarity": polarity,
            "score": round(score, 3),
            "confidence": round(confidence, 3),
            "target_entity": target_entity,
            "target_specificity": target_specificity,
            "stance": raw_stance,
            "stance_strength": stance_strength,
            "catalyst_strength": catalyst_strength,
            "evidence": evidence,
        }

    def _rule_sentiment(self, text: str) -> dict[str, Any]:
        words = set(re.findall(r"\w+", text.lower()))
        pos = len(words & _POSITIVE_WORDS)
        neg = len(words & _NEGATIVE_WORDS)
        total = pos + neg
        if total == 0:
            return {
                "polarity": "neutral",
                "score": 0.0,
                "confidence": 0.5,
                "target_entity": None,
                "target_specificity": "unknown",
                "stance": "neutral",
                "stance_strength": "weak",
                "catalyst_strength": "unknown",
                "evidence": "",
            }
        score = (pos - neg) / total
        polarity = "positive" if score > 0.1 else ("negative" if score < -0.1 else "neutral")
        return {
            "polarity": polarity,
            "score": round(score, 3),
            "confidence": round(min(total / 5, 1.0), 3),
            "target_entity": None,
            "target_specificity": "unknown",
            "stance": polarity,
            "stance_strength": self._strength_from_score(score),
            "catalyst_strength": "unknown",
            "evidence": "",
        }

    async def _llm_sentiment(self, text: str, llm: Any, *, target_entity: str | None) -> dict[str, Any]:
        try:
            target_line = (
                f"Focus on the entity `{target_entity}` when extracting the stance. "
                if target_entity else
                ""
            )
            prompt = (
                "Analyze the text and return JSON only.\n"
                f"{target_line}"
                "Required keys:\n"
                "- polarity: positive | negative | neutral\n"
                "- score: number between -1 and 1\n"
                "- confidence: number between 0 and 1\n"
                "- target_entity: short string or null\n"
                "- target_specificity: target_specific | mixed | generic | unknown\n"
                "- stance: positive | negative | mixed | neutral\n"
                "- stance_strength: strong | moderate | weak\n"
                "- catalyst_strength: explicit | weak | none | unknown\n"
                "- evidence: one short sentence\n"
                "Distinguish target-specific stance from generic document tone when possible.\n"
                f"Text:\n{str(text or '').strip()}"
            )
            resp = await llm.complete(
                [{"role": "user", "content": prompt}],
                response_format="json",
            )
            if resp.parsed:
                return self._normalize_payload(resp.parsed, fallback_target=target_entity)
        except Exception:
            pass
        return self._rule_sentiment(text)


tool = SentimentTool()
