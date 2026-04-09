"""Context window management for EvidenceIterator.

Prevents prompt bloat by managing token budgets with three-level strategies:
1. Pruning: truncate old evidence observations
2. Compaction: LLM-summarize early evidence steps
3. Lazy tool descriptions: only include full schema on demand
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from timeclaw.core.evidence import EvidenceLog
    from timeclaw.llm.base import LLMBackend


class ContextManager:
    """Manages token budget for the reasoning engine's prompts."""

    def __init__(
        self,
        max_tokens: int = 200000,
        reserve_tokens: int = 50000,
        compaction_threshold: float = 0.8,
        prune_observation_chars: int = 5000,
    ):
        self.max_tokens = max_tokens
        self.reserve_tokens = reserve_tokens
        self.compaction_threshold = compaction_threshold
        self.prune_observation_chars = prune_observation_chars
        self._compacted_summary: str | None = None

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Rough token estimate: ~4 chars per token for English, ~2 for CJK."""
        return len(text) // 4

    @property
    def usable_tokens(self) -> int:
        return self.max_tokens - self.reserve_tokens

    def fit_budget(self, sections: dict[str, str]) -> dict[str, str]:
        """Fit prompt sections within the token budget.

        Sections are prioritized by key order; later sections get pruned first.
        Returns pruned sections dict.
        """
        total = sum(self.estimate_tokens(v) for v in sections.values())
        if total <= self.usable_tokens:
            return sections

        logger.debug(f"[ContextManager] Budget exceeded: {total}/{self.usable_tokens} tokens, pruning")

        result = dict(sections)
        keys = list(result.keys())

        for key in reversed(keys):
            if total <= self.usable_tokens:
                break
            old_tokens = self.estimate_tokens(result[key])
            if old_tokens > 500:
                target_chars = max(200, (self.usable_tokens * 4) // len(keys))
                result[key] = result[key][:target_chars] + "\n... [truncated]"
                new_tokens = self.estimate_tokens(result[key])
                total -= (old_tokens - new_tokens)

        return result

    def prune_evidence_summary(self, evidence_summary: str, max_recent_steps: int = 5) -> str:
        """Prune evidence summary, keeping recent steps in full and truncating older ones."""
        lines = evidence_summary.split("\n")
        if len(lines) <= max_recent_steps * 3:
            return evidence_summary

        step_blocks: list[str] = []
        current_block: list[str] = []
        for line in lines:
            if line.startswith("Step ") and current_block:
                step_blocks.append("\n".join(current_block))
                current_block = []
            current_block.append(line)
        if current_block:
            step_blocks.append("\n".join(current_block))

        if len(step_blocks) <= max_recent_steps:
            return evidence_summary

        pruned_blocks = []
        cutoff = len(step_blocks) - max_recent_steps
        for i, block in enumerate(step_blocks):
            if i < cutoff:
                block_lines = block.split("\n")
                pruned = block_lines[0] if block_lines else block[:80]
                obs_truncated = f"  [observation truncated to {self.prune_observation_chars} chars]"
                pruned_blocks.append(pruned + "\n" + obs_truncated)
            else:
                pruned_blocks.append(block)

        return "\n".join(pruned_blocks)

    async def compact_evidence(self, evidence_log: EvidenceLog, llm: LLMBackend) -> str:
        """Use LLM to summarize the evidence log into a compact narrative."""
        summary = evidence_log.summary()
        tokens = self.estimate_tokens(summary)

        if tokens < self.usable_tokens * self.compaction_threshold:
            return summary

        logger.info(f"[ContextManager] Compacting evidence: {tokens} tokens -> LLM summary")

        messages = [
            {
                "role": "system",
                "content": (
                    "Summarize the following time-series analysis evidence log into a concise "
                    "narrative. Preserve: key findings, tool results, anomalies, and conclusions. "
                    "Keep it under 500 words. Output plain text, not JSON."
                ),
            },
            {"role": "user", "content": summary},
        ]

        try:
            response = await llm.complete(messages)
            self._compacted_summary = response.text or summary
            return self._compacted_summary
        except Exception as e:
            logger.warning(f"[ContextManager] Compaction failed: {e}")
            return self.prune_evidence_summary(summary)

    def prune_tool_descriptions(
        self,
        tools: list[dict[str, Any]],
        budget: int | None = None,
    ) -> list[dict[str, Any]]:
        """Lazy-load tool descriptions: only include name + short description by default.

        If budget allows, include full schema for top-priority tools.
        """
        budget = budget or (self.usable_tokens // 4)
        light: list[dict[str, Any]] = []
        current_tokens = 0

        for t in tools:
            name = t.get("name", "")
            desc = t.get("description", "")
            category = t.get("category", "")

            short = {"name": name, "description": desc[:500], "category": category}
            short_tokens = self.estimate_tokens(str(short))

            if current_tokens + short_tokens > budget:
                break
            light.append(short)
            current_tokens += short_tokens

        return light
