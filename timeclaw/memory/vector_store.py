"""Lightweight vector store backed by SQLite + numpy for semantic experience retrieval.

No heavy dependencies — uses TF-IDF for local embedding with optional LLM embedding support.
Inspired by OpenClaw's hybrid BM25 + vector search architecture.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger


class VectorStore:
    """SQLite-backed vector store with cosine similarity search.

    Stores text, metadata, and dense vectors for each entry.
    Supports hybrid search combining vector similarity with keyword matching.
    """

    def __init__(self, db_path: str | Path = ".timeclaw/vector_store.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), timeout=30,
                                     check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=30000")
        self._init_schema()
        self._tfidf_vocab: dict[str, int] | None = None
        self._idf: np.ndarray | None = None

    def _init_schema(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS vectors (
                id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                vector BLOB NOT NULL,
                metadata TEXT DEFAULT '{}',
                created_at REAL DEFAULT (strftime('%s', 'now'))
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_vectors_created
            ON vectors(created_at DESC)
        """)
        self._conn.commit()

    def add(self, id: str, text: str, vector: np.ndarray, metadata: dict[str, Any] | None = None) -> None:
        vec_bytes = vector.astype(np.float32).tobytes()
        meta_json = json.dumps(metadata or {})
        for attempt in range(5):
            try:
                self._conn.execute(
                    "INSERT OR REPLACE INTO vectors (id, text, vector, metadata) VALUES (?, ?, ?, ?)",
                    (id, text, vec_bytes, meta_json),
                )
                self._conn.commit()
                return
            except sqlite3.OperationalError:
                if attempt < 4:
                    import time as _time
                    _time.sleep(0.1 * (attempt + 1))
                else:
                    raise

    def search(
        self,
        query_vector: np.ndarray,
        limit: int = 10,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Cosine similarity search over stored vectors."""
        rows = self._conn.execute("SELECT id, text, vector, metadata FROM vectors").fetchall()
        if not rows:
            return []

        query_vec = query_vector.astype(np.float32)
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            return []

        scored: list[tuple[float, dict[str, Any]]] = []
        for row_id, text, vec_bytes, meta_json in rows:
            stored_vec = np.frombuffer(vec_bytes, dtype=np.float32)
            if stored_vec.shape != query_vec.shape:
                continue
            stored_norm = np.linalg.norm(stored_vec)
            if stored_norm == 0:
                continue
            similarity = float(np.dot(query_vec, stored_vec) / (query_norm * stored_norm))
            meta = json.loads(meta_json)

            if filter_metadata:
                if not all(meta.get(k) == v for k, v in filter_metadata.items()):
                    continue

            scored.append((similarity, {
                "id": row_id,
                "text": text,
                "score": similarity,
                "metadata": meta,
            }))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:limit]]

    def hybrid_search(
        self,
        query_text: str,
        query_vector: np.ndarray,
        limit: int = 10,
        vector_weight: float = 0.7,
        text_weight: float = 0.3,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Hybrid search: weighted combination of vector similarity and BM25-like text score."""
        rows = self._conn.execute("SELECT id, text, vector, metadata FROM vectors").fetchall()
        if not rows:
            return []

        query_vec = query_vector.astype(np.float32)
        query_norm = np.linalg.norm(query_vec)
        query_tokens = set(query_text.lower().split())

        scored: list[tuple[float, dict[str, Any]]] = []
        for row_id, text, vec_bytes, meta_json in rows:
            meta = json.loads(meta_json)
            if filter_metadata and not all(meta.get(k) == v for k, v in filter_metadata.items()):
                continue

            vec_score = 0.0
            if query_norm > 0:
                stored_vec = np.frombuffer(vec_bytes, dtype=np.float32)
                if stored_vec.shape == query_vec.shape:
                    stored_norm = np.linalg.norm(stored_vec)
                    if stored_norm > 0:
                        vec_score = float(np.dot(query_vec, stored_vec) / (query_norm * stored_norm))

            doc_tokens = set(text.lower().split())
            overlap = len(query_tokens & doc_tokens)
            text_score = overlap / max(len(query_tokens), 1)

            combined = vector_weight * vec_score + text_weight * text_score
            scored.append((combined, {
                "id": row_id,
                "text": text,
                "score": combined,
                "vector_score": vec_score,
                "text_score": text_score,
                "metadata": meta,
            }))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:limit]]

    @property
    def size(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM vectors").fetchone()
        return row[0] if row else 0

    def delete(self, ids: list[str]) -> int:
        """Delete entries by ID. Returns number of rows deleted."""
        if not ids:
            return 0
        placeholders = ",".join("?" for _ in ids)
        for attempt in range(5):
            try:
                cursor = self._conn.execute(
                    f"DELETE FROM vectors WHERE id IN ({placeholders})", ids,
                )
                self._conn.commit()
                return cursor.rowcount
            except sqlite3.OperationalError:
                if attempt < 4:
                    import time as _time
                    _time.sleep(0.1 * (attempt + 1))
                else:
                    raise
        return 0

    def clear(self) -> None:
        self._conn.execute("DELETE FROM vectors")
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()


class TFIDFEmbedder:
    """Lightweight TF-IDF embedder that requires no external dependencies.

    Builds vocabulary incrementally; produces fixed-dimension vectors via hashing.
    """

    def __init__(self, dim: int = 256):
        self.dim = dim

    def embed(self, text: str) -> np.ndarray:
        """Produce a fixed-size vector via hashed TF-IDF."""
        tokens = text.lower().split()
        if not tokens:
            return np.zeros(self.dim, dtype=np.float32)

        vec = np.zeros(self.dim, dtype=np.float32)
        tf: dict[str, int] = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1

        for token, count in tf.items():
            idx = hash(token) % self.dim
            vec[idx] += count / len(tokens)

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        return np.array([self.embed(t) for t in texts], dtype=np.float32)
