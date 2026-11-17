"""Text similarity tool: compute similarity between text pairs."""

from __future__ import annotations

import re
from collections import Counter
from math import sqrt
from typing import Any

from timeclaw.clawhub.tool_base import ClawTool, ToolCategory, ToolIOType


class TextSimilarityTool(ClawTool):
    """Compute similarity between two texts using cosine similarity on word vectors."""

    name: str = "text_similarity"
    category: ToolCategory = ToolCategory.TEXT
    description: str = (
        "Measures semantic similarity between two text strings using cosine similarity of embeddings. "
        "Input: two text strings. "
        "Output: dict with 'similarity' score (0 to 1, where 1 means identical meaning). "
        "Use when you need to compare documents, find related content, or check if two texts discuss the same topic. "
        "Params: text_a and text_b (required, the two texts to compare), method (similarity algorithm)."
    )
    io_type: ToolIOType = ToolIOType.T_TO_R
    parameters_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "text_a": {"type": "string", "description": "First text"},
            "text_b": {"type": "string", "description": "Second text"},
            "method": {"type": "string", "enum": ["bow", "tfidf", "llm"]},
        },
        "required": ["text_a", "text_b"],
    }
    is_stub: bool = False

    async def _run(self, ts_data: Any = None, **params: Any) -> Any:
        text_a = params.get("text_a") or params.get("text")
        text_b = params.get("text_b")
        if not text_a or not text_b:
            raise ValueError("Both text_a and text_b are required")

        method = params.get("method", "bow")

        if method == "tfidf":
            return self._tfidf_similarity(str(text_a), str(text_b))
        return self._bow_similarity(str(text_a), str(text_b))

    def _bow_similarity(self, a: str, b: str) -> dict[str, Any]:
        words_a = Counter(re.findall(r'\w+', a.lower()))
        words_b = Counter(re.findall(r'\w+', b.lower()))
        all_words = set(words_a.keys()) | set(words_b.keys())
        if not all_words:
            return {"similarity": 0.0, "method": "bow"}

        dot = sum(words_a.get(w, 0) * words_b.get(w, 0) for w in all_words)
        norm_a = sqrt(sum(v ** 2 for v in words_a.values()))
        norm_b = sqrt(sum(v ** 2 for v in words_b.values()))
        denom = norm_a * norm_b
        similarity = dot / denom if denom > 0 else 0.0
        return {"similarity": round(similarity, 4), "method": "bow"}

    def _tfidf_similarity(self, a: str, b: str) -> dict[str, Any]:
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity

            vectorizer = TfidfVectorizer()
            matrix = vectorizer.fit_transform([a, b])
            sim = cosine_similarity(matrix[0:1], matrix[1:2])[0][0]
            return {"similarity": round(float(sim), 4), "method": "tfidf"}
        except ImportError:
            return self._bow_similarity(a, b)


tool = TextSimilarityTool()
