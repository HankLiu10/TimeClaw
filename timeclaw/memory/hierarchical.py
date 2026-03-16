"""Agent Memory System — OpenClaw-inspired file-first architecture.

Six components:

SOUL.md (fixed, human-written):
    Agent identity and behavioral guidelines. Defines what TimeClaw is,
    how it approaches problems, and core behavioral principles.
    Not learned — maintained by developers.

Daily Notes (auto-recorded):
    Natural-language summaries of each pipeline run + outcome.
    Append-only, stored as daily log files, indexed in a vector store
    for semantic retrieval.

MEMORY.md (auto-distilled from daily notes):
    LLM-distilled causal attributions explaining WHY certain patterns of
    success/failure occur.  Scoped to a domain, task, or cross-cutting concern.

PRINCIPLES.md (auto-distilled from MEMORY):
    Cross-domain, cross-task transferable operating strategies.
    Very few, very stable.

TOOLS.md (auto-distilled from daily notes):
    Per-tool usage notes learned from experience — when each tool works
    well or poorly.

PLAYBOOKS/ (auto-distilled from MEMORY):
    Scene-specific SOPs (Standard Operating Procedures) per (domain, task).
    Step-by-step operational guides.

Distillation chain:
    Daily Notes  ──→  MEMORY.md  ──→  PRINCIPLES.md
                 └──→  TOOLS.md       └──→  PLAYBOOKS/
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from loguru import logger

from timeclaw.memory.vector_store import TFIDFEmbedder, VectorStore

MEMORY_ROOT = Path(".timeclaw/hierarchical_memory")
DAILY_DIR = MEMORY_ROOT / "daily"
MEMORY_DIR = MEMORY_ROOT / "memory"
PRINCIPLES_DIR = MEMORY_ROOT / "principles"
TOOLS_FILE = MEMORY_ROOT / "TOOLS.md"
PLAYBOOKS_DIR = MEMORY_ROOT / "playbooks"
SOUL_FILE = Path("timeclaw/memory/SOUL.md")
VECTOR_DB = MEMORY_ROOT / "notes_vectors.db"

_DISTILL_L1_TO_L2 = 10
_DISTILL_L2_TO_L3 = 20
_L1_RETRIEVAL_K = 5


# ═══════════════════════════════════════════════════════════════════════
# Daily Notes (append-only experience log, semantically indexed)
# ═══════════════════════════════════════════════════════════════════════

class DailyNotes:
    """Append-only natural-language note store with semantic retrieval.

    Each note is a short structured summary of one pipeline run and
    its outcome.  Notes are stored in daily log files and indexed
    in a vector store for semantic search.
    """

    def __init__(self):
        DAILY_DIR.mkdir(parents=True, exist_ok=True)
        self._vector_store = VectorStore(db_path=VECTOR_DB)
        self._embedder = TFIDFEmbedder(dim=256)
        self._total_count = self._vector_store.size
        self._pending: list[dict[str, Any]] = []
        self._lock = asyncio.Lock()
        self._note_llm = None

    @property
    def total(self) -> int:
        return self._total_count

    async def _get_llm(self):
        if self._note_llm is None:
            from timeclaw.core.config import TimeClawConfig
            from timeclaw.llm.router import ModelRouter
            cfg = TimeClawConfig.load()
            router = ModelRouter(cfg.llm)
            self._note_llm = router.get_backend("reasoner")
        return self._note_llm

    async def add(self, run_context: dict[str, Any]) -> int:
        note_text = self._build_note(run_context)

        note_id = f"note_{int(time.time()*1000)}_{self._total_count}"
        metadata = {
            "domain": run_context.get("domain", ""),
            "task": run_context.get("task", ""),
            "correct": run_context.get("correct", False),
            "timestamp": time.time(),
        }

        vector = self._embedder.embed(note_text)
        self._vector_store.add(
            id=note_id, text=note_text,
            vector=vector, metadata=metadata,
        )

        self._append_to_daily_log(note_text, run_context)

        async with self._lock:
            self._total_count += 1
            return self._total_count

    def _build_note(self, ctx: dict[str, Any]) -> str:
        domain = ctx.get("domain", "unknown")
        task = ctx.get("task", "unknown")
        outcome = "OK" if ctx.get("correct") else "FAIL"
        tools = ctx.get("tool_chain", [])
        tools_str = " → ".join(tools[:20]) if tools else "none"
        predicted = str(ctx.get("predicted", ""))[:500]
        ground_truth = str(ctx.get("ground_truth", ""))[:500]
        metrics = ctx.get("metrics") or {}
        metrics_str = ", ".join(
            f"{k}={v:.4f}" for k, v in metrics.items()
            if isinstance(v, (int, float))
        ) if metrics else ""
        fingerprint = ctx.get("data_fingerprint", "")[:500]
        description = ctx.get("description", "")[:200]
        pipeline_summary = ctx.get("pipeline_summary", "")[:500]

        all_evals = ctx.get("all_eval_metrics") or []
        approaches_str = self._format_approaches(tools, all_evals)

        parts = [f"[{domain}/{task}] {outcome}"]
        if description:
            parts.append(f"Task: {description}")
        parts.append(f"Tools: {tools_str}")
        parts.append(f"Pred: {predicted} | GT: {ground_truth}")
        if metrics_str:
            parts.append(f"Metrics: {metrics_str}")
        if fingerprint:
            parts.append(f"Fingerprint: {fingerprint}")
        if approaches_str:
            parts.append(f"Approaches: {approaches_str}")
        if pipeline_summary:
            parts.append(f"Analysis: {pipeline_summary}")
        return " | ".join(parts)

    @staticmethod
    def _format_approaches(
        tool_chain: list[str], all_evals: list[dict[str, Any]],
    ) -> str:
        """Build a compact comparison string from multiple eval results.

        Pairs each evaluate_against_gt call with the tool that preceded it,
        sorted by MAE ascending so the best approach comes first.
        """
        if not all_evals:
            return ""
        pred_tools = [t for t in tool_chain if t != "evaluate_against_gt" and t != "spawn_subagent"]
        pairs: list[tuple[str, float]] = []
        for i, m in enumerate(all_evals):
            mae = m.get("mae") or m.get("avg_mae") or m.get("accuracy")
            if mae is None:
                continue
            tool_name = pred_tools[i] if i < len(pred_tools) else f"approach_{i}"
            pairs.append((tool_name, float(mae)))
        if not pairs:
            return ""
        pairs.sort(key=lambda x: x[1])
        return " > ".join(f"{t}(mae={v:.2f})" for t, v in pairs)

    def _append_to_daily_log(self, note: str, ctx: dict[str, Any]) -> None:
        today = time.strftime("%Y-%m-%d")
        path = DAILY_DIR / f"{today}.md"
        domain = ctx.get("domain", "")
        task = ctx.get("task", "")
        correct = "OK" if ctx.get("correct") else "FAIL"
        timestamp = time.strftime("%H:%M:%S")

        entry = f"\n### [{timestamp}] {domain}/{task} — {correct}\n{note}\n"
        with path.open("a") as f:
            f.write(entry)

    def search(self, query: str, k: int = _L1_RETRIEVAL_K,
               domain: str | None = None, task: str | None = None,
               return_ids: bool = False) -> list[str] | list[dict[str, Any]]:
        query_vec = self._embedder.embed(query)
        filter_meta = {}
        if domain:
            filter_meta["domain"] = domain
        if task:
            filter_meta["task"] = task

        results = self._vector_store.hybrid_search(
            query_text=query,
            query_vector=query_vec,
            limit=k,
            vector_weight=0.7,
            text_weight=0.3,
            filter_metadata=filter_meta if filter_meta else None,
        )
        if return_ids:
            return [{"id": r["id"], "text": r["text"]} for r in results]
        return [r["text"] for r in results]

    def delete_notes(self, note_ids: list[str]) -> int:
        if not note_ids:
            return 0
        n = self._vector_store.delete(note_ids)
        if n > 0:
            async def _dec():
                async with self._lock:
                    self._total_count = max(0, self._total_count - n)
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(_dec())
            except RuntimeError:
                self._total_count = max(0, self._total_count - n)
            logger.info(f"[DailyNotes] Deleted {n} contradicted notes")
        return n

    def get_recent_notes(self, n: int = 10) -> list[str]:
        files = sorted(DAILY_DIR.glob("*.md"), reverse=True)
        notes = []
        for f in files:
            if len(notes) >= n:
                break
            for block in reversed(f.read_text().split("\n### ")):
                if block.strip():
                    notes.append(block.strip())
                    if len(notes) >= n:
                        break
        return notes[:n]

    async def resolve_conflicts(self, query: str,
                                domain: str | None = None,
                                task: str | None = None) -> list[str]:
        candidates = self.search(
            query, k=_L1_RETRIEVAL_K * 3,
            domain=domain, task=task, return_ids=True,
        )
        if len(candidates) < 2:
            return [c["text"] for c in candidates]

        notes_text = "\n".join(
            f"[{i}] {c['text']}" for i, c in enumerate(candidates)
        )
        try:
            llm = await self._get_llm()
            resp = await llm.complete(
                [
                    {"role": "system", "content": (
                        "You review a set of experience notes retrieved for the "
                        "same context. Some older notes may CONTRADICT newer notes "
                        "(e.g., an old note says tool X fails, but a newer note shows "
                        "tool X succeeds on similar data). Identify notes that are "
                        "OUTDATED or CONTRADICTED by other notes in the set.\n\n"
                        "Return JSON: {\"remove_indices\": [list of 0-based indices to remove]}\n"
                        "If no contradictions, return: {\"remove_indices\": []}"
                    )},
                    {"role": "user", "content": notes_text},
                ],
                response_format="json",
            )
            if resp.parsed and "remove_indices" in resp.parsed:
                to_remove = resp.parsed["remove_indices"]
                if to_remove:
                    ids_to_delete = [
                        candidates[i]["id"] for i in to_remove
                        if isinstance(i, int) and 0 <= i < len(candidates)
                    ]
                    self.delete_notes(ids_to_delete)
                    kept_indices = set(range(len(candidates))) - set(to_remove)
                    return [candidates[i]["text"] for i in sorted(kept_indices)]
        except Exception as e:
            logger.debug(f"[DailyNotes] Conflict resolution failed: {e}")

        return [c["text"] for c in candidates]

    def get_prompt_section(self, query: str,
                           domain: str | None = None,
                           task: str | None = None) -> str:
        notes = self.search(query, k=_L1_RETRIEVAL_K, domain=domain, task=task)
        if not notes:
            return ""
        lines = ["## Relevant Past Experiences\n"]
        for i, note in enumerate(notes, 1):
            lines.append(f"{i}. {note}")
        lines.append("")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
# MEMORY.md — Causal insights distilled from daily notes
# ═══════════════════════════════════════════════════════════════════════

class MemoryEntry:
    def __init__(self, scope: str, insight: str, evidence_count: int,
                 timestamp: float = 0.0):
        self.scope = scope
        self.insight = insight
        self.evidence_count = evidence_count
        self.timestamp = timestamp or time.time()

    def to_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope, "insight": self.insight,
            "evidence_count": self.evidence_count,
            "timestamp": self.timestamp,
        }


class MemoryStore:
    """MEMORY.md — causal insights partitioned by scope (domain_task)."""

    def __init__(self):
        MEMORY_DIR.mkdir(parents=True, exist_ok=True)
        self._entries: dict[str, list[dict]] = {}
        self._load_existing()

    def _load_existing(self) -> None:
        for f in MEMORY_DIR.glob("*.json"):
            try:
                self._entries[f.stem] = json.loads(f.read_text())
            except Exception:
                pass

    def get_entries(self, scope: str) -> list[dict]:
        return self._entries.get(scope, [])

    def get_all_entries(self) -> list[dict]:
        all_entries = []
        for entries in self._entries.values():
            all_entries.extend(entries)
        return all_entries

    def total(self) -> int:
        return sum(len(v) for v in self._entries.values())

    def add_entries(self, scope: str, new_entries: list[dict],
                    removed_texts: list[str] | None = None) -> None:
        if removed_texts:
            removed_set = set(removed_texts)
            existing = [
                e for e in self._entries.get(scope, [])
                if e["insight"] not in removed_set
            ]
            n_removed = len(self._entries.get(scope, [])) - len(existing)
            if n_removed:
                logger.info(f"[MEMORY] {scope}: removed {n_removed} contradicted insights")
            self._entries[scope] = existing

        existing = self._entries.get(scope, [])
        existing_texts = {str(e["insight"]) for e in existing}
        for entry in new_entries:
            text = str(entry["insight"]) if not isinstance(entry["insight"], str) else entry["insight"]
            entry["insight"] = text
            if text not in existing_texts:
                existing.append(entry)
                existing_texts.add(text)
        self._entries[scope] = existing
        path = MEMORY_DIR / f"{scope}.json"
        path.write_text(json.dumps(existing, indent=2, ensure_ascii=False))

    def get_prompt_section(self, domain: str, task: str) -> str:
        scope = f"{domain}_{task}"
        entries = self.get_entries(scope)
        domain_entries = self.get_entries(domain)
        all_entries = entries + domain_entries
        if not all_entries:
            return ""
        lines = ["## MEMORY — Causal Insights (learned from analyzing past errors)\n"]
        for e in all_entries[-8:]:
            lines.append(f"- {e['insight']}")
        lines.append("")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
# SOUL.md — Fixed agent identity (human-written, not learned)
# ═══════════════════════════════════════════════════════════════════════

def load_soul() -> str:
    """Load the fixed SOUL.md content."""
    if SOUL_FILE.exists():
        return SOUL_FILE.read_text(encoding="utf-8").strip()
    return ""


# ═══════════════════════════════════════════════════════════════════════
# PRINCIPLES.md — Cross-domain strategies distilled from MEMORY
# ═══════════════════════════════════════════════════════════════════════

class PrinciplesStore:
    """PRINCIPLES.md — very few, very stable cross-domain strategies."""

    def __init__(self):
        PRINCIPLES_DIR.mkdir(parents=True, exist_ok=True)
        self._principles: list[str] = []
        self._load()

    def _load(self) -> None:
        path = PRINCIPLES_DIR / "principles.json"
        if path.exists():
            try:
                self._principles = json.loads(path.read_text())
            except Exception:
                self._principles = []

    def get_principles(self) -> list[str]:
        return list(self._principles)

    def total(self) -> int:
        return len(self._principles)

    def add_principles(self, new_principles: list[str],
                       removed_texts: list[str] | None = None) -> None:
        if removed_texts:
            removed_set = set(removed_texts)
            before = len(self._principles)
            self._principles = [p for p in self._principles if p not in removed_set]
            n_removed = before - len(self._principles)
            if n_removed:
                logger.info(f"[PRINCIPLES] Removed {n_removed} contradicted principles")

        existing = set(self._principles)
        for s in new_principles:
            if s not in existing:
                self._principles.append(s)
                existing.add(s)
        path = PRINCIPLES_DIR / "principles.json"
        path.write_text(json.dumps(self._principles, indent=2, ensure_ascii=False))

    def get_prompt_section(self) -> str:
        if not self._principles:
            return ""
        lines = ["## PRINCIPLES — Learned Operating Strategies\n"]
        for i, s in enumerate(self._principles[-6:], 1):
            lines.append(f"{i}. {s}")
        lines.append("")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
# TOOLS.md — Per-tool usage notes learned from experience
# ═══════════════════════════════════════════════════════════════════════

class ToolsStore:
    """TOOLS.md — per-tool experience notes, distilled from daily notes.

    Global notes describe each tool's general behaviour.  Per-scope
    counters track how many times each tool has been used in a given
    (domain, task) scope so we can flag under-explored tools.
    """

    _MIN_RUNS = 5

    def __init__(self):
        MEMORY_ROOT.mkdir(parents=True, exist_ok=True)
        self._notes: dict[str, str] = {}
        self._scope_counts: dict[str, dict[str, int]] = {}
        self._load()

    def _load(self) -> None:
        if TOOLS_FILE.exists():
            try:
                raw = json.loads(TOOLS_FILE.read_text())
                if isinstance(raw, dict) and "_scope_counts" in raw:
                    self._notes = raw.get("notes", {})
                    self._scope_counts = raw.get("_scope_counts", {})
                else:
                    self._notes = raw if isinstance(raw, dict) else {}
            except Exception:
                self._notes = {}

    def _save(self) -> None:
        payload = {"notes": self._notes, "_scope_counts": self._scope_counts}
        TOOLS_FILE.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    def get_notes(self) -> dict[str, str]:
        return dict(self._notes)

    def update(self, notes: dict[str, str]) -> None:
        self._notes.update(notes)
        self._save()
        logger.info(f"[TOOLS] Updated usage notes for {len(notes)} tools")

    def record_scope_usage(self, scope: str, tool_counts: dict[str, int]) -> None:
        """Replace per-tool usage counts for a scope from the latest scan."""
        self._scope_counts[scope] = dict(tool_counts)
        self._save()

    def total(self) -> int:
        return len(self._notes)

    def get_tool_note(self, tool_name: str) -> str:
        return self._notes.get(tool_name, "")

    def get_prompt_for_tools(
        self, tool_names: list[str], scope: str = "",
    ) -> dict[str, str]:
        """Return notes for the given tools, appending an 'Insufficient data'
        tag for tools with fewer than ``_MIN_RUNS`` uses in this scope."""
        result: dict[str, str] = {}
        scope_data = self._scope_counts.get(scope, {}) if scope else {}
        for name in tool_names:
            note = self._notes.get(name, "")
            if scope and scope_data.get(name, 0) < self._MIN_RUNS:
                tag = (f" [Insufficient data for {scope} — "
                       f"only {scope_data.get(name, 0)} runs, try exploring]")
                note = (note + tag) if note else f"Insufficient data for {scope}"
            if note:
                result[name] = note
        return result


# ═══════════════════════════════════════════════════════════════════════
# PLAYBOOKS/ — Scene-specific SOPs distilled from MEMORY insights
# ═══════════════════════════════════════════════════════════════════════

class PlaybooksStore:
    """PLAYBOOKS/ — per (domain, task) operational guides.

    Each playbook is a natural-language SOP describing how to approach
    a specific task type: steps, tool combinations, and pitfalls.
    """

    def __init__(self):
        PLAYBOOKS_DIR.mkdir(parents=True, exist_ok=True)
        self._playbooks: dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        for f in PLAYBOOKS_DIR.glob("*.md"):
            try:
                self._playbooks[f.stem] = f.read_text()
            except Exception:
                pass

    def get_playbook(self, scope: str) -> str:
        return self._playbooks.get(scope, "")

    def total(self) -> int:
        return len(self._playbooks)

    def update_playbook(self, scope: str, content: str) -> None:
        self._playbooks[scope] = content
        path = PLAYBOOKS_DIR / f"{scope}.md"
        path.write_text(content)
        logger.info(f"[PLAYBOOKS] Updated SOP for {scope}")

    def get_prompt_section(self, domain: str, task: str) -> str:
        scope = f"{domain}_{task}"
        playbook = self._playbooks.get(scope, "")
        if not playbook:
            return ""
        lines = [f"## PLAYBOOK — Operational Guide for {domain}/{task}\n"]
        lines.append(playbook)
        lines.append("")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
# Distillation Prompts
# ═══════════════════════════════════════════════════════════════════════

NOTES_TO_MEMORY_PROMPT = """\
You are a strategy analyst for a multimodal time-series analysis agent.
Analyze the experience data below and produce ACTIONABLE decision rules.

## Context
Scope: {scope}
Total runs analyzed: {total}

## Statistics
{statistics}

## Sample Experience Notes (recent)
{notes}

## Existing Insights (review and update these)
{existing_memory}

## Your Task
Produce an UPDATED set of decision rules. Each rule must be:
1. **Data-condition-aware**: Specify WHEN this rule applies — include data
   characteristics (frequency, seasonality, trend, series length, domain)
2. **Tool-aware**: Name the specific tools or tool categories to use/avoid
3. **Quantitative**: Include concrete metric comparisons from the evidence
4. **Concise**: One rule = one sentence (max 2). No narratives.

FORMAT — follow strictly:
"[WHEN data-condition] → [DO tool/strategy] (BECAUSE reason; EVIDENCE: metric comparison)"

EXAMPLES of GOOD rules:
- "[WHEN hourly data with 24h seasonality, forecast horizon<=24] → \
prefer chronos_forecast over arima_forecast \
(BECAUSE ARIMA flattens diurnal cycle; EVIDENCE: avg MAE chronos=2.3 vs arima=2.9 over 10 runs)"
- "[WHEN indicator task on weather data] → use llm_reasoning directly \
instead of forecast+derive \
(BECAUSE forecast models over-smooth min/max; EVIDENCE: llm_reasoning avg_mae=1.8 vs forecast-derived=3.5)"
- "[WHEN classification/trend task with text context available] → \
weight text signal over numerical detect_trend \
(BECAUSE detect_trend misreads short-term noise; EVIDENCE: acc 35% with text vs 18% without)"

EXAMPLES of BAD rules (DO NOT produce):
- "ARIMA tends to produce flat forecasts" (no data condition, no action)
- "Chronos outperforms Timer on this dataset" (too narrow, not reusable)
- "Foundation models are generally better" (too vague, no specifics)

RULES:
- Review existing rules against new evidence: REMOVE contradicted or obsolete ones
- Max 8 rules per scope — keep only the most actionable
- Every rule MUST specify both a data condition AND a tool recommendation
- Merge overlapping rules rather than keeping duplicates

Return JSON: {{
  "insights": ["rule 1", "rule 2", ...],
  "removed": ["exact text of any removed rule", ...]
}}
"""

MEMORY_TO_PRINCIPLES_PROMPT = """\
You are synthesizing domain-specific decision rules into universal
operating principles for a multimodal time-series analysis agent.

## All Domain-Task Rules
{all_memory_entries}

## Existing Principles (review and update)
{existing_principles}

## Your Task
Produce an UPDATED set of cross-domain operating principles.

Each principle must be:
1. **Actionable**: A concrete rule the agent can follow immediately
2. **Cross-domain**: Applies across weather, finance, energy, etc.
3. **Strategy-level**: References tool *categories* (foundation models,
   statistical models, decomposition tools, LLM reasoning) rather than
   specific tool names

FORMAT: "[RULE]: [action]. [RATIONALE]: [cross-domain evidence]"

EXAMPLES of GOOD principles:
- "RULE: When data shows strong periodicity, apply seasonal decomposition
  before forecasting. RATIONALE: Across weather (24h cycle) and finance
  (weekly cycle), decomposition reduces forecast MAE by 15-30%."
- "RULE: For indicator tasks, prefer LLM reasoning over forecast+derive
  pipelines. RATIONALE: Statistical models over-smooth extremes in both
  temperature and price data, inflating min/max errors."
- "RULE: Always try at least one foundation model and one classical model
  for forecasting, then compare. RATIONALE: No single model category
  dominates across all data patterns."

EXAMPLES of BAD principles:
- "Use good tools" (no action)
- "Chronos is best for weather" (too specific, not cross-domain)

RULES:
- Review existing principles against new evidence: REMOVE outdated ones
- Max 6 principles total
- Every principle MUST give a concrete, testable action

Return JSON: {{
  "principles": ["principle 1", ...],
  "removed": ["exact text of any removed principle", ...]
}}
"""

NOTES_TO_TOOLS_PROMPT = """\
You are analyzing tool performance statistics from a time-series agent's
training runs. For each tool, write a concise usage note (1-2 sentences)
describing when it works well and when it doesn't, based purely on the
observed data.

## Tool Statistics
{tool_stats}

## Existing Tool Notes (review and update)
{existing_tools}

## Rules
- Base notes ONLY on the statistics — no speculation
- Each note should help the agent decide WHEN to pick this tool
- If a tool has too few runs (<5) to draw conclusions, write "Insufficient data"
- Do NOT rank tools against each other or declare "best" tools

Return JSON: {{
  "tool_notes": {{"tool_name": "usage note", ...}}
}}
"""

MEMORY_TO_PLAYBOOKS_PROMPT = """\
You are creating an operational guide (SOP) for a time-series agent.
Based on the causal insights learned for this specific (domain, task),
produce a step-by-step guide the agent can follow.

## Domain/Task: {scope}

## Causal Insights (from MEMORY.md)
{memory_entries}

## Existing SOP (review and update)
{existing_playbook}

## Your Task
Write a concise operational guide in Markdown. Include:
1. **Steps**: Numbered procedure for this task type
2. **Pitfalls**: Common failure modes to watch for
3. **Tool Notes**: Which tools are relevant and when

FORMAT: Plain Markdown (not JSON). Keep it under 300 words.

Example structure:
### Procedure
1. First, ...
2. Then, ...
3. Finally, ...

### Common Pitfalls
- Pitfall 1: ...
- Pitfall 2: ...
"""


# ═══════════════════════════════════════════════════════════════════════
# Distillation Agent
# ═══════════════════════════════════════════════════════════════════════

class DistillationAgent:
    """Orchestrates bottom-up knowledge distillation.

    Distillation chain:
        Daily Notes  ──→  MEMORY.md  ──→  PRINCIPLES.md
                     └──→  TOOLS.md       └──→  PLAYBOOKS/
    """

    def __init__(self, daily: DailyNotes, memory: MemoryStore,
                 principles: PrinciplesStore, tools: ToolsStore,
                 playbooks: PlaybooksStore):
        self.daily = daily
        self.memory = memory
        self.principles = principles
        self.tools = tools
        self.playbooks = playbooks
        self._notes_distilled_at: dict[str, int] = {}
        self._memory_distilled_at: int = 0
        self._llm = None

    async def _get_llm(self):
        if self._llm is None:
            from timeclaw.core.config import TimeClawConfig
            from timeclaw.llm.router import ModelRouter
            cfg = TimeClawConfig.load()
            router = ModelRouter(cfg.llm)
            self._llm = router.get_backend("reasoner")
        return self._llm

    @staticmethod
    def _compute_statistics(notes: list[str]) -> str:
        import re
        from collections import Counter

        tool_ok: Counter = Counter()
        tool_fail: Counter = Counter()
        labels_pred: Counter = Counter()
        labels_gt: Counter = Counter()
        metric_sums: dict[str, list[float]] = {}
        total_ok = 0
        total_fail = 0

        for note in notes:
            is_ok = "] OK" in note
            if is_ok:
                total_ok += 1
            else:
                total_fail += 1

            tools_match = re.search(r"Tools:\s*(.+?)(?:\s*\|)", note)
            if tools_match:
                chain = [t.strip() for t in tools_match.group(1).split("→")]
                _skip = {"evaluate_against_gt", "spawn_subagent", "none"}
                for t in chain:
                    if t and t not in _skip:
                        (tool_ok if is_ok else tool_fail)[t] += 1

            pred_match = re.search(r"Pred:\s*(.+?)\s*\|\s*GT:\s*(.+?)(?:\s*\||$)", note)
            if pred_match:
                p, g = pred_match.group(1).strip(), pred_match.group(2).strip()
                if len(p) < 30:
                    labels_pred[p] += 1
                if len(g) < 30:
                    labels_gt[g] += 1

            metrics_match = re.search(r"Metrics:\s*(.+?)(?:\s*\||$)", note)
            if metrics_match:
                for m in metrics_match.group(1).split(","):
                    m = m.strip()
                    if "=" in m:
                        k, v = m.split("=", 1)
                        try:
                            metric_sums.setdefault(k.strip(), []).append(float(v))
                        except ValueError:
                            pass

        lines: list[str] = []
        lines.append(f"Total runs: {total_ok + total_fail} (OK={total_ok}, FAIL={total_fail})")

        all_tools = set(tool_ok) | set(tool_fail)
        if all_tools:
            lines.append("\nTool success rates:")
            for t in sorted(all_tools, key=lambda x: -(tool_ok[x] + tool_fail[x])):
                ok_n, fail_n = tool_ok[t], tool_fail[t]
                total = ok_n + fail_n
                rate = ok_n / total * 100 if total else 0
                lines.append(f"  {t}: {ok_n}/{total} ({rate:.0f}%)")

        if labels_gt and len(labels_gt) <= 10:
            lines.append("\nGround truth label distribution:")
            for label, cnt in labels_gt.most_common():
                lines.append(f"  '{label}': {cnt}")
            lines.append("Predicted label distribution:")
            for label, cnt in labels_pred.most_common():
                lines.append(f"  '{label}': {cnt}")

        if metric_sums:
            lines.append("\nMetric averages:")
            for k, vals in sorted(metric_sums.items()):
                avg = sum(vals) / len(vals)
                lines.append(f"  {k}: avg={avg:.4f} (n={len(vals)})")

        return "\n".join(lines) if lines else "(no statistics available)"

    @staticmethod
    def _compute_tool_stats(notes: list[str]) -> str:
        """Extract per-tool performance statistics for TOOLS.md distillation."""
        import re
        from collections import Counter

        tool_ok: Counter = Counter()
        tool_fail: Counter = Counter()
        tool_metrics: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

        for note in notes:
            is_ok = "] OK" in note
            tools_match = re.search(r"Tools:\s*(.+?)(?:\s*\|)", note)
            if not tools_match:
                continue
            chain = [t.strip() for t in tools_match.group(1).split("→")]

            metrics_match = re.search(r"Metrics:\s*(.+?)(?:\s*\||$)", note)
            metric_vals: dict[str, float] = {}
            if metrics_match:
                for m in metrics_match.group(1).split(","):
                    m = m.strip()
                    if "=" in m:
                        k, v = m.split("=", 1)
                        try:
                            metric_vals[k.strip()] = float(v)
                        except ValueError:
                            pass

            _FRAMEWORK_TOOLS = {"evaluate_against_gt", "spawn_subagent", "none"}
            for t in chain:
                if t and t not in _FRAMEWORK_TOOLS:
                    (tool_ok if is_ok else tool_fail)[t] += 1
                    for mk, mv in metric_vals.items():
                        tool_metrics[t][mk].append(mv)

        lines = []
        all_tools = sorted(set(tool_ok) | set(tool_fail),
                           key=lambda x: -(tool_ok[x] + tool_fail[x]))
        for t in all_tools:
            ok_n, fail_n = tool_ok[t], tool_fail[t]
            total = ok_n + fail_n
            rate = ok_n / total * 100 if total else 0
            line = f"- **{t}**: {total} runs, {rate:.0f}% success"
            if t in tool_metrics:
                metric_parts = []
                for mk, mvs in sorted(tool_metrics[t].items()):
                    avg = sum(mvs) / len(mvs)
                    metric_parts.append(f"{mk}={avg:.4f}")
                if metric_parts:
                    line += f" | {', '.join(metric_parts)}"
            lines.append(line)

        return "\n".join(lines) if lines else "(no tool data)"

    # ── Notes → MEMORY.md ─────────────────────────────────────────────

    async def maybe_distill_notes_to_memory(self, domain: str, task: str) -> bool:
        key = f"{domain}_{task}"
        current = self.daily.total
        last = self._notes_distilled_at.get(key, 0)
        if current - last < _DISTILL_L1_TO_L2:
            return False

        logger.info(f"[Distill Notes→MEMORY] {key}: {current} notes (+{current - last})")
        self._notes_distilled_at[key] = current

        await self.daily.resolve_conflicts(
            query=f"{domain} {task}",
            domain=domain, task=task,
        )

        notes = self.daily.search(
            query=f"{domain} {task} error failure wrong",
            k=30, domain=domain, task=task,
        )
        success_notes = self.daily.search(
            query=f"{domain} {task} correct success",
            k=15, domain=domain, task=task,
        )
        all_notes = notes + success_notes
        if len(all_notes) < 5:
            return False

        statistics = self._compute_statistics(all_notes)
        notes_text = "\n\n".join(f"- {n}" for n in all_notes[:30])

        existing_memory = self.memory.get_entries(key)
        existing_text = "\n".join(
            f"  - {e['insight']}" for e in existing_memory
        ) if existing_memory else "(none yet)"

        prompt = NOTES_TO_MEMORY_PROMPT.format(
            scope=key, total=len(all_notes),
            statistics=statistics, notes=notes_text,
            existing_memory=existing_text,
        )

        llm = await self._get_llm()
        try:
            resp = await llm.complete(
                [{"role": "system",
                  "content": "You are a strategy analyst. Produce actionable, tool-aware insights grounded in the statistics. Return JSON."},
                 {"role": "user", "content": prompt}],
                response_format="json",
            )
            if resp.parsed and "insights" in resp.parsed:
                removed = resp.parsed.get("removed", [])
                new_entries = [
                    MemoryEntry(scope=key, insight=a, evidence_count=len(all_notes)).to_dict()
                    for a in resp.parsed["insights"]
                ]
                self.memory.add_entries(key, new_entries, removed_texts=removed)
                logger.info(
                    f"[Distill Notes→MEMORY] {key}: updated to {len(self.memory.get_entries(key))} insights "
                    f"(removed {len(removed)})"
                )
                return True
        except Exception as e:
            logger.warning(f"[Distill Notes→MEMORY] {key} failed: {e}")
        return False

    # ── Notes → TOOLS.md ──────────────────────────────────────────────

    async def maybe_distill_notes_to_tools(self) -> bool:
        """Distill per-tool usage notes from daily notes statistics.

        Also updates per-scope tool usage counts used for the
        'Insufficient data' tag at injection time.
        """
        all_notes = self.daily.get_recent_notes(100)
        if len(all_notes) < 10:
            return False

        full_notes = self.daily.get_recent_notes(10000)
        self._update_scope_counts(full_notes)

        tool_stats = self._compute_tool_stats(all_notes)
        existing_text = "\n".join(
            f"- **{k}**: {v}" for k, v in self.tools.get_notes().items()
        ) if self.tools.total() > 0 else "(none yet)"

        prompt = NOTES_TO_TOOLS_PROMPT.format(
            tool_stats=tool_stats,
            existing_tools=existing_text,
        )

        llm = await self._get_llm()
        try:
            resp = await llm.complete(
                [{"role": "system",
                  "content": "You analyze tool performance data and write concise usage notes. Return JSON."},
                 {"role": "user", "content": prompt}],
                response_format="json",
            )
            if resp.parsed and "tool_notes" in resp.parsed:
                self.tools.update(resp.parsed["tool_notes"])
                logger.info(f"[Distill Notes→TOOLS] Updated {len(resp.parsed['tool_notes'])} tool notes")
                return True
        except Exception as e:
            logger.warning(f"[Distill Notes→TOOLS] failed: {e}")
        return False

    def _update_scope_counts(self, notes: list[str]) -> None:
        """Parse notes and rebuild per-scope tool usage counters."""
        import re
        _FRAMEWORK_TOOLS = {"evaluate_against_gt", "spawn_subagent", "none"}
        scope_tool_counts: dict[str, dict[str, int]] = {}

        for note in notes:
            scope_match = re.match(r"\[(\S+?)/(\S+?)\]", note)
            if not scope_match:
                continue
            scope = f"{scope_match.group(1)}_{scope_match.group(2)}"
            tools_match = re.search(r"Tools:\s*(.+?)(?:\s*\|)", note)
            if not tools_match:
                continue
            chain = [t.strip() for t in tools_match.group(1).split("→")]
            bucket = scope_tool_counts.setdefault(scope, {})
            for t in chain:
                if t and t not in _FRAMEWORK_TOOLS:
                    bucket[t] = bucket.get(t, 0) + 1

        for scope, counts in scope_tool_counts.items():
            self.tools.record_scope_usage(scope, counts)

    # ── MEMORY → PRINCIPLES.md ──────────────────────────────────────────

    async def maybe_distill_memory_to_principles(self) -> bool:
        current = self.memory.total()
        if current - self._memory_distilled_at < _DISTILL_L2_TO_L3:
            return False

        logger.info(f"[Distill MEMORY→PRINCIPLES] {current} entries (+{current - self._memory_distilled_at})")
        self._memory_distilled_at = current

        all_memory = self.memory.get_all_entries()
        if len(all_memory) < 4:
            return False

        memory_text = "\n".join(
            f"  [{e.get('scope', '?')}] {e['insight']}"
            for e in all_memory[-60:]
        )

        existing_principles = self.principles.get_principles()
        existing_text = "\n".join(
            f"  - {s}" for s in existing_principles
        ) if existing_principles else "(none yet)"

        prompt = MEMORY_TO_PRINCIPLES_PROMPT.format(
            all_memory_entries=memory_text,
            existing_principles=existing_text,
        )

        llm = await self._get_llm()
        try:
            resp = await llm.complete(
                [{"role": "system",
                  "content": "You review and update universal principles based on new insights. Return JSON."},
                 {"role": "user", "content": prompt}],
                response_format="json",
            )
            if resp.parsed and "principles" in resp.parsed:
                new = resp.parsed["principles"]
                removed = resp.parsed.get("removed", [])
                self.principles.add_principles(new, removed_texts=removed)
                logger.info(
                    f"[Distill MEMORY→PRINCIPLES] Updated to {self.principles.total()} principles "
                    f"(removed {len(removed)})"
                )
                return True
        except Exception as e:
            logger.warning(f"[Distill MEMORY→PRINCIPLES] failed: {e}")
        return False

    # ── MEMORY → PLAYBOOKS/ ────────────────────────────────────────────

    async def maybe_distill_memory_to_playbooks(self, scope: str) -> bool:
        """Generate/update a PLAYBOOKS/ SOP from MEMORY insights for a scope."""
        entries = self.memory.get_entries(scope)
        if len(entries) < 3:
            return False

        entries_text = "\n".join(f"- {e['insight']}" for e in entries)
        existing_playbook = self.playbooks.get_playbook(scope)

        prompt = MEMORY_TO_PLAYBOOKS_PROMPT.format(
            scope=scope,
            memory_entries=entries_text,
            existing_playbook=existing_playbook or "(none yet)",
        )

        llm = await self._get_llm()
        try:
            resp = await llm.complete(
                [{"role": "system",
                  "content": "You write concise operational guides for a time-series agent. Return plain Markdown."},
                 {"role": "user", "content": prompt}],
            )
            if resp.text and len(resp.text.strip()) > 50:
                self.playbooks.update_playbook(scope, resp.text.strip())
                logger.info(f"[Distill MEMORY→PLAYBOOKS] Updated SOP for {scope}")
                return True
        except Exception as e:
            logger.warning(f"[Distill MEMORY→PLAYBOOKS] {scope} failed: {e}")
        return False

    # ── Orchestration ─────────────────────────────────────────────────

    async def on_new_note(self, domain: str, task: str) -> None:
        """Called after each daily note is added. Triggers distillation cascade."""
        did_memory = await self.maybe_distill_notes_to_memory(domain, task)
        if did_memory:
            await self.maybe_distill_memory_to_principles()
            scope = f"{domain}_{task}"
            await self.maybe_distill_memory_to_playbooks(scope)
            await self.maybe_distill_notes_to_tools()

    async def force_distill_all(self) -> None:
        """Force distillation for all scopes."""
        scopes = set()
        for f in MEMORY_DIR.glob("*.json"):
            scopes.add(f.stem)
        notes = self.daily.get_recent_notes(100)
        for note in notes:
            for token in note.split():
                if "/" in token:
                    parts = token.split("/")
                    if len(parts) == 2:
                        scopes.add(f"{parts[0]}_{parts[1]}")

        for scope in scopes:
            parts = scope.split("_", 1)
            if len(parts) == 2:
                self._notes_distilled_at[scope] = 0
                await self.maybe_distill_notes_to_memory(parts[0], parts[1])
                await self.maybe_distill_memory_to_playbooks(scope)

        self._memory_distilled_at = 0
        await self.maybe_distill_memory_to_principles()
        await self.maybe_distill_notes_to_tools()


# ═══════════════════════════════════════════════════════════════════════
# Unified Memory Interface
# ═══════════════════════════════════════════════════════════════════════

class HierarchicalMemory:
    """Unified interface for TimeClaw's six-component memory system.

    Components:
        soul_text   — Fixed SOUL.md (agent identity, human-written)
        daily       — DailyNotes (raw experience log)
        memory      — MemoryStore (MEMORY.md, causal insights)
        principles  — PrinciplesStore (PRINCIPLES.md, cross-domain strategies)
        tools       — ToolsStore (TOOLS.md, per-tool usage notes)
        playbooks   — PlaybooksStore (PLAYBOOKS/, scene-specific SOPs)
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        if enabled:
            MEMORY_ROOT.mkdir(parents=True, exist_ok=True)
            self.soul_text = load_soul()
            self.daily = DailyNotes()
            self.memory = MemoryStore()
            self.principles = PrinciplesStore()
            self.tools = ToolsStore()
            self.playbooks = PlaybooksStore()
            self.distiller = DistillationAgent(
                self.daily, self.memory, self.principles,
                self.tools, self.playbooks,
            )
            n_daily = self.daily.total
            n_memory = self.memory.total()
            n_principles = self.principles.total()
            n_tools = self.tools.total()
            n_playbooks = self.playbooks.total()
            if n_daily + n_memory + n_principles > 0:
                logger.info(
                    f"[AgentMemory] Loaded: "
                    f"daily={n_daily}, MEMORY={n_memory}, PRINCIPLES={n_principles}, "
                    f"TOOLS={n_tools}, PLAYBOOKS={n_playbooks}"
                )
        else:
            self.soul_text = ""
            self.daily = None
            self.memory = None
            self.principles = None
            self.tools = None
            self.playbooks = None
            self.distiller = None

    # Backward compatibility aliases
    @property
    def l1(self):
        return self.daily

    @property
    def l2(self):
        return self.memory

    @property
    def l3(self):
        return self.principles

    @property
    def soul(self):
        return self.principles

    @property
    def skills(self):
        return self.playbooks

    async def record(self, **run_context) -> None:
        if not self.enabled:
            return
        count = await self.daily.add(run_context)
        domain = run_context.get("domain", "")
        task = run_context.get("task", "")
        if count % _DISTILL_L1_TO_L2 == 0 and domain and task:
            asyncio.create_task(self.distiller.on_new_note(domain, task))

    def get_memory(self, query: str, domain: str = "", task: str = "") -> str:
        if not self.enabled:
            return ""
        parts = []
        if self.soul_text:
            parts.append(self.soul_text)
        principles = self.principles.get_prompt_section()
        if principles:
            parts.append(principles)
        mem = self.memory.get_prompt_section(domain, task) if domain and task else ""
        if mem:
            parts.append(mem)
        daily = self.daily.get_prompt_section(query, domain=domain or None, task=task or None)
        if daily:
            parts.append(daily)
        return "\n".join(parts)

    def get_tool_notes(
        self, domain: str = "", task: str = "",
        tool_names: list[str] | None = None,
    ) -> dict[str, str]:
        if not self.enabled or not self.tools:
            return {}
        scope = f"{domain}_{task}" if domain and task else ""
        if tool_names:
            return self.tools.get_prompt_for_tools(tool_names, scope)
        return self.tools.get_prompt_for_tools(
            list(self.tools.get_notes().keys()), scope,
        )

    def get_playbook(self, domain: str, task: str) -> str:
        if not self.enabled or not self.playbooks:
            return ""
        return self.playbooks.get_prompt_section(domain, task)

    def get_skill(self, domain: str, task: str) -> str:
        """Backward compatibility alias for get_playbook."""
        return self.get_playbook(domain, task)

    def flush(self) -> None:
        pass

    async def force_distill(self) -> None:
        if self.enabled and self.distiller:
            await self.distiller.force_distill_all()

    def summary(self) -> dict[str, Any]:
        if not self.enabled:
            return {"enabled": False}
        return {
            "enabled": True,
            "daily_notes": self.daily.total,
            "memory_insights": self.memory.total(),
            "principles": self.principles.total(),
            "tool_notes": self.tools.total(),
            "playbooks": self.playbooks.total(),
        }
