"""Keyword extraction tool: extract key terms from text using TF-IDF or LLM."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from timeclaw.clawhub.tool_base import ClawTool, ToolCategory, ToolIOType

_STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "during",
    "before", "after", "above", "below", "between", "and", "but", "or",
    "not", "no", "nor", "so", "yet", "both", "either", "neither", "each",
    "every", "all", "any", "few", "more", "most", "other", "some", "such",
    "than", "too", "very", "just", "about", "also", "then", "it", "its",
    "this", "that", "these", "those", "i", "me", "my", "we", "our", "you",
    "your", "he", "him", "his", "she", "her", "they", "them", "their",
}


class KeywordExtractTool(ClawTool):
    """Extract keywords from text using TF-IDF or frequency-based methods."""

    name: str = "keyword_extract"
    category: ToolCategory = ToolCategory.TEXT
    description: str = "Extract top keywords from text using TF-IDF or word frequency."
    io_type: ToolIOType = ToolIOType.T_TO_R
    parameters_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "top_k": {"type": "integer", "description": "Number of keywords to return", "default": 10},
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
            combined = "\n".join(texts)
        else:
            combined = str(text)

        top_k = params.get("top_k", 10)
        llm = self.resolve_llm(params.get("llm"), params.get("model"))

        if llm:
            return await self._llm_keywords(combined, top_k, llm)
        return self._frequency_keywords(combined, top_k)

    def _frequency_keywords(self, text: str, top_k: int) -> dict[str, Any]:
        words = [w for w in re.findall(r'\w+', text.lower()) if w not in _STOP_WORDS and len(w) > 2]
        counter = Counter(words)
        keywords = [{"keyword": w, "count": c} for w, c in counter.most_common(top_k)]
        return {"keywords": keywords, "method": "frequency"}

    def _tfidf_keywords(self, text: str, top_k: int) -> dict[str, Any]:
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer

            sentences = re.split(r'[.!?\n]+', text)
            sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
            if len(sentences) < 2:
                return self._frequency_keywords(text, top_k)

            vectorizer = TfidfVectorizer(max_features=500, stop_words="english")
            tfidf_matrix = vectorizer.fit_transform(sentences)
            feature_names = vectorizer.get_feature_names_out()
            scores = tfidf_matrix.sum(axis=0).A1
            top_indices = scores.argsort()[-top_k:][::-1]
            keywords = [
                {"keyword": feature_names[i], "tfidf_score": round(float(scores[i]), 4)}
                for i in top_indices
            ]
            return {"keywords": keywords, "method": "tfidf"}
        except ImportError:
            return self._frequency_keywords(text, top_k)

    async def _llm_keywords(self, text: str, top_k: int, llm: Any) -> dict[str, Any]:
        try:
            prompt = (
                f"Extract the top {top_k} keywords from this text. "
                f"Return JSON with key 'keywords' as array of strings.\n\nText: {text[:2000]}"
            )
            resp = await llm.complete([{"role": "user", "content": prompt}], response_format="json")
            if resp.parsed and "keywords" in resp.parsed:
                kws = resp.parsed["keywords"]
                return {"keywords": [{"keyword": k} for k in kws[:top_k]], "method": "llm"}
        except Exception:
            pass
        return self._frequency_keywords(text, top_k)


tool = KeywordExtractTool()
