"""Tests for VectorStore and TFIDFEmbedder."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from timeclaw.memory.vector_store import TFIDFEmbedder, VectorStore


class TestTFIDFEmbedder:
    def test_embed_produces_correct_dim(self):
        embedder = TFIDFEmbedder(dim=128)
        vec = embedder.embed("hello world this is a test")
        assert vec.shape == (128,)
        assert vec.dtype == np.float32

    def test_embed_empty_string(self):
        embedder = TFIDFEmbedder(dim=64)
        vec = embedder.embed("")
        assert np.allclose(vec, 0)

    def test_embed_normalized(self):
        embedder = TFIDFEmbedder(dim=256)
        vec = embedder.embed("some meaningful text about time series")
        norm = np.linalg.norm(vec)
        assert abs(norm - 1.0) < 1e-5

    def test_similar_texts_closer(self):
        embedder = TFIDFEmbedder(dim=256)
        v1 = embedder.embed("forecast temperature weather")
        v2 = embedder.embed("forecast temperature climate")
        v3 = embedder.embed("stock price finance trading")
        sim_12 = np.dot(v1, v2)
        sim_13 = np.dot(v1, v3)
        assert sim_12 > sim_13

    def test_embed_batch(self):
        embedder = TFIDFEmbedder(dim=64)
        vecs = embedder.embed_batch(["hello", "world", "test"])
        assert vecs.shape == (3, 64)


class TestVectorStore:
    def test_add_and_search(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = VectorStore(db_path=Path(tmpdir) / "test.db")
            vec = np.random.randn(64).astype(np.float32)
            vec /= np.linalg.norm(vec)
            store.add("test1", "hello world", vec, {"task_type": "forecast"})
            assert store.size == 1

            results = store.search(vec, limit=5)
            assert len(results) == 1
            assert results[0]["id"] == "test1"
            assert results[0]["score"] > 0.99
            store.close()

    def test_hybrid_search(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = VectorStore(db_path=Path(tmpdir) / "test.db")
            embedder = TFIDFEmbedder(dim=64)

            store.add("e1", "forecast weather temperature", embedder.embed("forecast weather temperature"))
            store.add("e2", "stock price trading", embedder.embed("stock price trading"))
            store.add("e3", "forecast temperature humidity", embedder.embed("forecast temperature humidity"))

            query_vec = embedder.embed("forecast weather prediction")
            results = store.hybrid_search("forecast weather prediction", query_vec, limit=2)
            assert len(results) == 2
            assert results[0]["id"] in ("e1", "e3")
            store.close()

    def test_filter_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = VectorStore(db_path=Path(tmpdir) / "test.db")
            vec1 = np.random.randn(32).astype(np.float32)
            vec2 = np.random.randn(32).astype(np.float32)
            store.add("e1", "text1", vec1, {"task_type": "forecast"})
            store.add("e2", "text2", vec2, {"task_type": "qa"})

            results = store.search(vec1, limit=5, filter_metadata={"task_type": "forecast"})
            assert all(r["metadata"]["task_type"] == "forecast" for r in results)
            store.close()

    def test_clear(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = VectorStore(db_path=Path(tmpdir) / "test.db")
            vec = np.random.randn(32).astype(np.float32)
            store.add("e1", "text", vec)
            assert store.size == 1
            store.clear()
            assert store.size == 0
            store.close()
