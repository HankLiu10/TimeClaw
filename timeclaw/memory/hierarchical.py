"""Agent Memory System — OpenClaw-inspired file-first architecture.

Five components (aligned with OpenClaw naming):

SOUL.md (fixed, human-written):
    Agent identity and behavioral guidelines.  Corresponds to OpenClaw SOUL.md.
    Not learned — maintained by developers.

Daily Notes (auto-recorded):
    Natural-language summaries of each pipeline run + outcome.
    Append-only, stored as daily log files, indexed in a vector store
    for semantic retrieval.  Corresponds to OpenClaw memory/YYYY-MM-DD.md.

MEMORY (auto-distilled from daily notes):
    LLM-distilled causal attributions explaining WHY certain patterns of
    success/failure occur.  Scoped to (domain, task).  Additionally holds
    a special ``cross_domain`` scope for universal operating strategies
    distilled across all scopes (replaces the former PRINCIPLES layer).
    Corresponds to OpenClaw MEMORY.md, but auto-distilled rather than
    manually curated.

TOOLS.md (auto-distilled from daily notes):
    Per-tool usage notes learned from experience — when each tool works
    well or poorly, helping the agent distinguish between functionally
    similar tools.  Corresponds to OpenClaw TOOLS.md.

SKILLS/ (auto-distilled from MEMORY):
    Scene-specific SOPs (Standard Operating Procedures) per (domain, task).
    Step-by-step operational guides.  Corresponds to OpenClaw SKILLS.md.

Distillation chain:
    Daily Notes  ──→  MEMORY (per-scope + cross_domain)
                 └──→  TOOLS.md
                 └──→  SKILLS/
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
TOOLS_FILE = MEMORY_ROOT / "TOOLS.md"
SKILLS_DIR = MEMORY_ROOT / "skills"
SOUL_FILE = Path("timeclaw/memory/SOUL.md")
VECTOR_DB = MEMORY_ROOT / "notes_vectors.db"

CROSS_DOMAIN_SCOPE = "cross_domain"

_DISTILL_L1_TO_L2 = 10
_DISTILL_L2_TO_CROSS = 20
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
        metrics = ctx.get("metrics") or {}
        _GT_KEYS = {"ground_truth_label", "ground_truth", "correct"}
        metrics_str = ", ".join(
            f"{k}={v:.4f}" for k, v in metrics.items()
            if isinstance(v, (int, float)) and k not in _GT_KEYS
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
        parts.append(f"Pred: {predicted}")
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

    def get_cross_domain_entries(self) -> list[dict]:
        return self.get_entries(CROSS_DOMAIN_SCOPE)

    def get_prompt_section(self, domain: str, task: str) -> str:
        scope = f"{domain}_{task}"
        entries = self.get_entries(scope)
        domain_entries = self.get_entries(domain)
        cross_entries = self.get_cross_domain_entries()
        all_entries = entries + domain_entries
        if not all_entries and not cross_entries:
            return ""
        lines = ["## MEMORY — Learned Insights\n"]
        if cross_entries:
            lines.append("### Cross-Domain Strategies")
            for e in cross_entries[-6:]:
                lines.append(f"- {e['insight']}")
            lines.append("")
        if all_entries:
            lines.append(f"### {domain}/{task} Specific")
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
# SKILLS/ — Scene-specific SOPs distilled from MEMORY insights
# ═══════════════════════════════════════════════════════════════════════

class SkillsStore:
    """SKILLS/ — per (domain, task) operational guides.

    Each skill is a natural-language SOP describing how to approach
    a specific task type: steps, tool combinations, and pitfalls.
    Corresponds to OpenClaw SKILLS.md.
    """

    def __init__(self):
        SKILLS_DIR.mkdir(parents=True, exist_ok=True)
        self._skills: dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        for f in SKILLS_DIR.glob("*.md"):
            try:
                self._skills[f.stem] = f.read_text()
            except Exception:
                pass

    def get_skill(self, scope: str) -> str:
        return self._skills.get(scope, "")

    def total(self) -> int:
        return len(self._skills)

    def update_skill(self, scope: str, content: str) -> None:
        self._skills[scope] = content
        path = SKILLS_DIR / f"{scope}.md"
        path.write_text(content)
        logger.info(f"[SKILLS] Updated SOP for {scope}")

    def get_prompt_section(self, domain: str, task: str) -> str:
        scope = f"{domain}_{task}"
        skill = self._skills.get(scope, "")
        if not skill:
            return ""
        lines = [f"## SKILL — Operational Guide for {domain}/{task}\n"]
        lines.append(skill)
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
- **CRITICAL**: Rules must NEVER reference ground truth values, ground truth
  distributions, or label frequencies (e.g. "60% of GT is decreasing").
  Ground truth is used only during training evaluation — the agent will NOT
  have access to it at test time.  Rules must be based on INPUT data
  characteristics (patterns, seasonality, trend shape, series length) and
  TOOL behavior (which tool produced lower error), not on output label stats.

Return JSON: {{
  "insights": ["rule 1", "rule 2", ...],
  "removed": ["exact text of any removed rule", ...]
}}
"""

MEMORY_TO_CROSS_DOMAIN_PROMPT = """\
You are synthesizing domain-specific decision rules into universal
cross-domain insights for a multimodal time-series analysis agent.

## All Domain-Task Rules
{all_memory_entries}

## Existing Cross-Domain Insights (review and update)
{existing_cross_domain}

## Your Task
Produce an UPDATED set of cross-domain insights using the same format
as per-scope rules:

"[WHEN data-condition across domains] → [DO tool-category/strategy] \
(BECAUSE reason; EVIDENCE: cross-domain metric comparison)"

Each insight must be:
1. **Actionable**: A concrete rule the agent can follow immediately
2. **Cross-domain**: Applies across weather, finance, energy, etc.
3. **Strategy-level**: References tool *categories* (foundation models,
   statistical models, decomposition tools, LLM reasoning) rather than
   specific tool names

RULES:
- Review existing insights against new evidence: REMOVE outdated ones
- Max 6 insights total
- Every insight MUST give a concrete, testable action
- **CRITICAL**: Insights must NEVER reference ground truth values or label
  distributions.  Base all guidance on input data characteristics and tool
  performance metrics, not on output label frequencies.

Return JSON: {{
  "insights": ["insight 1", ...],
  "removed": ["exact text of any removed insight", ...]
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

MEMORY_TO_SKILLS_PROMPT = """\
You are creating an operational guide (skill) for a time-series agent.
Based on the causal insights learned for this specific (domain, task),
produce a step-by-step guide the agent can follow.

## Domain/Task: {scope}

## Causal Insights (from MEMORY)
{memory_entries}

## Existing Skill (review and update)
{existing_skill}

## Your Task
Write a concise operational guide in Markdown. Include:
1. **Steps**: Numbered procedure for this task type
2. **Pitfalls**: Common failure modes to watch for
3. **Tool Notes**: Which tools are relevant and when

**CRITICAL**: The guide must NEVER reference ground truth values, ground
truth distributions, or label base rates.  The agent will NOT have access
to ground truth at test time.  All guidance must be based on input data
characteristics and observed tool behavior, not on output label statistics.

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
        Daily Notes  ──→  MEMORY (per-scope + cross_domain)
                     └──→  TOOLS.md
                     └──→  SKILLS/
    """

    def __init__(self, daily: DailyNotes, memory: MemoryStore,
                 tools: ToolsStore, skills: SkillsStore):
        self.daily = daily
        self.memory = memory
        self.tools = tools
        self.skills = skills
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

    # ── MEMORY → cross_domain scope ──────────────────────────────────────

    async def maybe_distill_memory_to_cross_domain(self) -> bool:
        current = self.memory.total()
        if current - self._memory_distilled_at < _DISTILL_L2_TO_CROSS:
            return False

        logger.info(f"[Distill MEMORY→cross_domain] {current} entries (+{current - self._memory_distilled_at})")
        self._memory_distilled_at = current

        all_memory = self.memory.get_all_entries()
        if len(all_memory) < 4:
            return False

        memory_text = "\n".join(
            f"  [{e.get('scope', '?')}] {e['insight']}"
            for e in all_memory[-60:]
        )

        existing_cross = self.memory.get_cross_domain_entries()
        existing_text = "\n".join(
            f"  - {e['insight']}" for e in existing_cross
        ) if existing_cross else "(none yet)"

        prompt = MEMORY_TO_CROSS_DOMAIN_PROMPT.format(
            all_memory_entries=memory_text,
            existing_cross_domain=existing_text,
        )

        llm = await self._get_llm()
        try:
            resp = await llm.complete(
                [{"role": "system",
                  "content": "You synthesize cross-domain insights from per-scope rules. Return JSON."},
                 {"role": "user", "content": prompt}],
                response_format="json",
            )
            if resp.parsed and "insights" in resp.parsed:
                removed = resp.parsed.get("removed", [])
                new_entries = [
                    MemoryEntry(
                        scope=CROSS_DOMAIN_SCOPE,
                        insight=a,
                        evidence_count=len(all_memory),
                    ).to_dict()
                    for a in resp.parsed["insights"]
                ]
                self.memory.add_entries(
                    CROSS_DOMAIN_SCOPE, new_entries,
                    removed_texts=removed,
                )
                n_cross = len(self.memory.get_cross_domain_entries())
                logger.info(
                    f"[Distill MEMORY→cross_domain] Updated to {n_cross} cross-domain insights "
                    f"(removed {len(removed)})"
                )
                return True
        except Exception as e:
            logger.warning(f"[Distill MEMORY→cross_domain] failed: {e}")
        return False

    # ── MEMORY → PLAYBOOKS/ ────────────────────────────────────────────

    async def maybe_distill_memory_to_skills(self, scope: str) -> bool:
        """Generate/update a SKILLS/ SOP from MEMORY insights for a scope."""
        entries = self.memory.get_entries(scope)
        if len(entries) < 3:
            return False

        entries_text = "\n".join(f"- {e['insight']}" for e in entries)
        existing_skill = self.skills.get_skill(scope)

        prompt = MEMORY_TO_SKILLS_PROMPT.format(
            scope=scope,
            memory_entries=entries_text,
            existing_skill=existing_skill or "(none yet)",
        )

        llm = await self._get_llm()
        try:
            resp = await llm.complete(
                [{"role": "system",
                  "content": "You write concise operational guides for a time-series agent. Return plain Markdown."},
                 {"role": "user", "content": prompt}],
            )
            if resp.text and len(resp.text.strip()) > 50:
                self.skills.update_skill(scope, resp.text.strip())
                logger.info(f"[Distill MEMORY→SKILLS] Updated SOP for {scope}")
                return True
        except Exception as e:
            logger.warning(f"[Distill MEMORY→SKILLS] {scope} failed: {e}")
        return False

    # ── Orchestration ─────────────────────────────────────────────────

    async def on_new_note(self, domain: str, task: str) -> None:
        """Called after each daily note is added. Triggers distillation cascade."""
        did_memory = await self.maybe_distill_notes_to_memory(domain, task)
        if did_memory:
            await self.maybe_distill_memory_to_cross_domain()
            scope = f"{domain}_{task}"
            await self.maybe_distill_memory_to_skills(scope)
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
            if scope == CROSS_DOMAIN_SCOPE:
                continue
            parts = scope.split("_", 1)
            if len(parts) == 2:
                self._notes_distilled_at[scope] = 0
                await self.maybe_distill_notes_to_memory(parts[0], parts[1])
                await self.maybe_distill_memory_to_skills(scope)

        self._memory_distilled_at = 0
        await self.maybe_distill_memory_to_cross_domain()
        await self.maybe_distill_notes_to_tools()


# ═══════════════════════════════════════════════════════════════════════
# Unified Memory Interface
# ═══════════════════════════════════════════════════════════════════════

class HierarchicalMemory:
    """Unified interface for TimeClaw's five-component memory system.

    Components (aligned with OpenClaw naming):
        soul_text   — Fixed SOUL.md (agent identity)
        daily       — DailyNotes (raw experience log)
        memory      — MemoryStore (causal insights, per-scope + cross_domain)
        tools       — ToolsStore (per-tool usage experience notes)
        skills      — SkillsStore (scene-specific SOPs)
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        if enabled:
            MEMORY_ROOT.mkdir(parents=True, exist_ok=True)
            self.soul_text = load_soul()
            self.daily = DailyNotes()
            self.memory = MemoryStore()
            self.tools = ToolsStore()
            self.skills = SkillsStore()
            self.distiller = DistillationAgent(
                self.daily, self.memory,
                self.tools, self.skills,
            )
            n_daily = self.daily.total
            n_memory = self.memory.total()
            n_cross = len(self.memory.get_cross_domain_entries())
            n_tools = self.tools.total()
            n_skills = self.skills.total()
            if n_daily + n_memory > 0:
                logger.info(
                    f"[AgentMemory] Loaded: "
                    f"daily={n_daily}, MEMORY={n_memory} (cross_domain={n_cross}), "
                    f"TOOLS={n_tools}, SKILLS={n_skills}"
                )
        else:
            self.soul_text = ""
            self.daily = None
            self.memory = None
            self.tools = None
            self.skills = None
            self.distiller = None

    # Backward compatibility aliases
    @property
    def l1(self):
        return self.daily

    @property
    def l2(self):
        return self.memory

    @property
    def principles(self):
        """Backward compat: returns memory store (cross_domain is inside)."""
        return self.memory

    @property
    def playbooks(self):
        """Backward compat: renamed to skills."""
        return self.skills

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

    def get_skill(self, domain: str, task: str) -> str:
        if not self.enabled or not self.skills:
            return ""
        return self.skills.get_prompt_section(domain, task)

    def get_playbook(self, domain: str, task: str) -> str:
        """Backward compatibility alias for get_skill."""
        return self.get_skill(domain, task)

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
            "cross_domain_insights": len(self.memory.get_cross_domain_entries()),
            "tool_notes": self.tools.total(),
            "skills": self.skills.total(),
        }
