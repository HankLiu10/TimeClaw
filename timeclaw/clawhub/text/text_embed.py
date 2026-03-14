"""Text embedding tool: convert text to dense vector representation."""

from __future__ import annotations

from typing import Any

from timeclaw.clawhub.tool_base import ClawTool, ToolCategory, ToolIOType


class TextEmbedTool(ClawTool):
    """Generate embedding vectors for text using TF-IDF or LLM embedding API."""

    name: str = "text_embed"
    category: ToolCategory = ToolCategory.TEXT
    description: str = (
        "Convert text into a dense embedding vector for downstream similarity/clustering."
    )
    io_type: ToolIOType = ToolIOType.T_TO_R
    parameters_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "model": {
                "type": "string",
                "enum": ["light", "heavy"],
                "description": "LLM tier for embedding: 'light' or 'heavy'",
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

        if llm:
            return await self._llm_embed(texts, llm)
        return self._tfidf_embed(texts)

    def _tfidf_embed(self, texts: list[str]) -> dict[str, Any]:
        from sklearn.feature_extraction.text import TfidfVectorizer

        vectorizer = TfidfVectorizer(max_features=256)
        matrix = vectorizer.fit_transform(texts)
        embeddings = matrix.toarray().tolist()
        return {
            "embeddings": embeddings,
            "dim": matrix.shape[1],
            "n_texts": len(texts),
            "method": "tfidf",
        }

    async def _llm_embed(self, texts: list[str], llm: Any = None) -> dict[str, Any]:
        if llm is None:
            return self._tfidf_embed(texts)
        try:
            import litellm

            response = await litellm.aembedding(
                model=getattr(llm, "default_model", "text-embedding-3-small"),
                input=texts,
            )
            embeddings = [item["embedding"] for item in response.data]
            return {
                "embeddings": embeddings,
                "dim": len(embeddings[0]) if embeddings else 0,
                "n_texts": len(texts),
                "method": "llm",
            }
        except Exception:
            return self._tfidf_embed(texts)


tool = TextEmbedTool()
