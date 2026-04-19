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
        "Output: dict with 'sentiment' label and numeric 'score' (-1 to 1). "
        "Use when you need to understand market sentiment, event impact, or opinion polarity from text. "
        "Params: model ('light' for fast/short text, 'heavy' for accurate/complex text)."
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

        results = []
        for t in texts:
            if llm:
                r = await self._llm_sentiment(t, llm)
            else:
                r = self._rule_sentiment(t)
            results.append(r)

        return {"sentiments": results, "n_texts": len(texts)}

    def _rule_sentiment(self, text: str) -> dict[str, Any]:
        words = set(re.findall(r"\w+", text.lower()))
        pos = len(words & _POSITIVE_WORDS)
        neg = len(words & _NEGATIVE_WORDS)
        total = pos + neg
        if total == 0:
            return {"polarity": "neutral", "score": 0.0, "confidence": 0.5}
        score = (pos - neg) / total
        polarity = "positive" if score > 0.1 else ("negative" if score < -0.1 else "neutral")
        return {
            "polarity": polarity,
            "score": round(score, 3),
            "confidence": round(min(total / 5, 1.0), 3),
        }

    async def _llm_sentiment(self, text: str, llm: Any) -> dict[str, Any]:
        try:
            prompt = (
                "Analyze the sentiment of this text. Return JSON with keys: "
                "polarity (positive/negative/neutral), score (-1 to 1), confidence (0 to 1).\n"
                f"Text: {text[:500]}"
            )
            resp = await llm.complete(
                [{"role": "user", "content": prompt}],
                response_format="json",
                reasoning_effort="none",
            )
            if resp.parsed:
                return resp.parsed
        except Exception:
            pass
        return self._rule_sentiment(text)


tool = SentimentTool()
