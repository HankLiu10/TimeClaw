"""Experience pool: stores and retrieves reasoning traces for learning from past runs.

Enhanced with vector store for semantic search over past experiences,
and LLM-driven tool insight summarization.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import BaseModel, Field

from timeclaw.memory.vector_store import TFIDFEmbedder, VectorStore


class ExperienceEntry(BaseModel):
    """A single reasoning experience record."""

    trace_id: str
    task_type: str
    task_description: str = ""
    fingerprint_summary: str = ""
    tool_chain: list[str] = Field(default_factory=list)
    evidence_steps: int = 0
    quality_gate_passed: bool = False
    quality_score: float = 0.0
    prediction_summary: str = ""
    timestamp: float = Field(default_factory=time.time)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_search_text(self) -> str:
        """Combine key fields into a single searchable text blob."""
        parts = [
            self.task_type,
            self.task_description,
            self.fingerprint_summary,
            " ".join(self.tool_chain),
            self.prediction_summary,
        ]
        return " ".join(p for p in parts if p)


class ExperiencePool:
    """Persistent store for reasoning traces, backed by JSON files + vector index."""

    def __init__(
        self,
        storage_dir: str | Path = ".timeclaw/experiences",
        vector_db_path: str | Path | None = None,
        enable_vector: bool = True,
    ):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._entries: list[ExperienceEntry] = []
        self._enable_vector = enable_vector

        if enable_vector:
            vdb = vector_db_path or (self.storage_dir.parent / "vector_store.db")
            self._vector_store = VectorStore(db_path=vdb)
            self._embedder = TFIDFEmbedder(dim=256)
        else:
            self._vector_store = None
            self._embedder = None

        self._load_existing()

    def _load_existing(self) -> None:
        for f in sorted(self.storage_dir.glob("*.json")):
            try:
                data = json.loads(f.read_text())
                if isinstance(data, list):
                    entries = [ExperienceEntry.model_validate(e) for e in data]
                    self._entries.extend(entries)
                    for e in entries:
                        self._index_entry(e)
                else:
                    entry = ExperienceEntry.model_validate(data)
                    self._entries.append(entry)
                    self._index_entry(entry)
            except Exception as e:
                logger.debug(f"Failed to load experience file {f}: {e}")

    def _index_entry(self, entry: ExperienceEntry) -> None:
        """Add entry to the vector index if vector mode is enabled."""
        if not self._enable_vector or self._vector_store is None:
            return
        try:
            text = entry.to_search_text()
            vector = self._embedder.embed(text)
            self._vector_store.add(
                id=entry.trace_id,
                text=text,
                vector=vector,
                metadata={
                    "task_type": entry.task_type,
                    "quality_gate_passed": entry.quality_gate_passed,
                    "timestamp": entry.timestamp,
                },
            )
        except Exception as e:
            logger.debug(f"Failed to index experience {entry.trace_id}: {e}")

    def add(self, entry: ExperienceEntry) -> None:
        self._entries.append(entry)
        self._persist(entry)
        self._index_entry(entry)

    def _persist(self, entry: ExperienceEntry) -> None:
        filename = f"{entry.trace_id}.json"
        path = self.storage_dir / filename
        path.write_text(entry.model_dump_json(indent=2))

    def search(
        self,
        task_type: str | None = None,
        limit: int = 10,
    ) -> list[ExperienceEntry]:
        """Search experiences by task type (exact match)."""
        results = self._entries
        if task_type:
            results = [e for e in results if e.task_type == task_type]
        return sorted(results, key=lambda e: e.timestamp, reverse=True)[:limit]

    async def semantic_search(
        self,
        query: str,
        limit: int = 5,
        task_type: str | None = None,
    ) -> list[ExperienceEntry]:
        """Semantic search over past experiences using hybrid vector+text matching.

        Returns experiences sorted by relevance (vector similarity 70% + text overlap 30%).
        """
        if not self._enable_vector or self._vector_store is None:
            logger.debug("Vector search disabled, falling back to keyword search")
            return self.search(task_type=task_type, limit=limit)

        query_vector = self._embedder.embed(query)
        filter_meta = {"task_type": task_type} if task_type else None

        results = self._vector_store.hybrid_search(
            query_text=query,
            query_vector=query_vector,
            limit=limit,
            vector_weight=0.7,
            text_weight=0.3,
            filter_metadata=filter_meta,
        )

        matched_entries: list[ExperienceEntry] = []
        entry_map = {e.trace_id: e for e in self._entries}
        for r in results:
            entry = entry_map.get(r["id"])
            if entry:
                matched_entries.append(entry)

        return matched_entries

    def _embed(self, text: str):
        """Generate embedding vector for a text string."""
        if self._embedder is None:
            raise RuntimeError("Vector mode not enabled")
        return self._embedder.embed(text)

    @property
    def size(self) -> int:
        return len(self._entries)

    def clear(self) -> None:
        self._entries.clear()
        for f in self.storage_dir.glob("*.json"):
            f.unlink()
        if self._vector_store:
            self._vector_store.clear()

    # ── Tool insight summarization ────────────────────────────────────

    def get_tool_stats(self) -> dict[str, dict[str, Any]]:
        """Aggregate per-tool usage statistics from all stored experiences.

        Returns a dict keyed by tool name with counts, avg quality score,
        pass rate, and common co-occurring tools.
        """
        tool_data: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for entry in self._entries:
            for tool in entry.tool_chain:
                tool_data[tool].append({
                    "quality_score": entry.quality_score,
                    "passed": entry.quality_gate_passed,
                    "task_type": entry.task_type,
                    "chain": entry.tool_chain,
                    "fingerprint": entry.fingerprint_summary[:200],
                })

        stats: dict[str, dict[str, Any]] = {}
        for tool, records in tool_data.items():
            n = len(records)
            avg_score = sum(r["quality_score"] for r in records) / n if n else 0
            pass_rate = sum(1 for r in records if r["passed"]) / n if n else 0

            co_tools: dict[str, int] = defaultdict(int)
            task_types: dict[str, int] = defaultdict(int)
            for r in records:
                task_types[r["task_type"]] += 1
                for co in r["chain"]:
                    if co != tool:
                        co_tools[co] += 1

            top_co = sorted(co_tools.items(), key=lambda x: -x[1])[:5]
            top_tasks = sorted(task_types.items(), key=lambda x: -x[1])[:3]

            stats[tool] = {
                "usage_count": n,
                "avg_quality_score": round(avg_score, 3),
                "pass_rate": round(pass_rate, 3),
                "top_co_tools": [t for t, _ in top_co],
                "top_task_types": [t for t, _ in top_tasks],
            }
        return stats

    def build_llm_summary_prompt(self) -> str | None:
        """Build a prompt for an LLM to summarize tool effectiveness.

        Returns None if there are fewer than 5 experiences (not enough data).
        """
        if len(self._entries) < 5:
            return None

        stats = self.get_tool_stats()
        lines = ["Here are aggregated tool usage statistics from past pipeline runs:\n"]
        for tool, s in sorted(stats.items(), key=lambda x: -x[1]["usage_count"]):
            lines.append(
                f"- **{tool}**: used {s['usage_count']} times, "
                f"avg_quality={s['avg_quality_score']}, pass_rate={s['pass_rate']}, "
                f"common_co_tools={s['top_co_tools']}, "
                f"task_types={s['top_task_types']}"
            )

        lines.append(
            "\nBased on these statistics, for each tool, provide a concise insight "
            "(1-2 sentences) about:\n"
            "1. In what scenarios does this tool work best?\n"
            "2. What tools does it pair well with?\n"
            "3. When should it be avoided?\n\n"
            "Return JSON: {\"tool_insights\": {\"tool_name\": \"insight text\", ...}}"
        )
        return "\n".join(lines)

    def save_tool_insights(self, insights: dict[str, str]) -> None:
        """Persist LLM-generated tool insights to disk."""
        path = self.storage_dir / "_tool_insights.json"
        path.write_text(json.dumps(insights, indent=2, ensure_ascii=False))
        logger.info(f"[ExperiencePool] Saved tool insights for {len(insights)} tools")

    def load_tool_insights(self) -> dict[str, str]:
        """Load previously saved tool insights."""
        path = self.storage_dir / "_tool_insights.json"
        if path.exists():
            try:
                return json.loads(path.read_text())
            except Exception:
                pass
        return {}
