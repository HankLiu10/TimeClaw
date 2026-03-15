"""Task Strategy Memory: learns and injects reasoning strategies for non-forecast tasks.

Unlike tool-selection skills (which guide the Executor on *which tool* to use),
task strategies guide the *final reasoning LLM* on *how to interpret* tool
analysis results for classification/QA tasks.

Strategies are learned online: after each evaluation, the outcome (correct/wrong)
is recorded.  Periodically, an LLM distills accumulated outcomes into concise
strategy rules that are injected into subsequent final-reasoning prompts.

Storage: .timeclaw/task_strategies/
  - records_{domain}_{task}.jsonl   (raw outcome records)
  - strategies.json                 (distilled strategy rules)
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from loguru import logger


STRATEGY_DIR = Path(".timeclaw/task_strategies")
STRATEGY_FILE = STRATEGY_DIR / "strategies.json"

_DISTILL_INTERVAL = 50
_MIN_RECORDS_FOR_DISTILL = 20


class OutcomeRecord:
    __slots__ = ("domain", "task", "predicted", "ground_truth", "correct",
                 "pipeline_summary", "ts_stats", "timestamp")

    def __init__(
        self,
        domain: str,
        task: str,
        predicted: str,
        ground_truth: str,
        correct: bool,
        pipeline_summary: str = "",
        ts_stats: str = "",
    ):
        self.domain = domain
        self.task = task
        self.predicted = predicted
        self.ground_truth = ground_truth
        self.correct = correct
        self.pipeline_summary = pipeline_summary[:2000]
        self.ts_stats = ts_stats[:1000]
        self.timestamp = time.time()

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "task": self.task,
            "predicted": self.predicted,
            "ground_truth": self.ground_truth,
            "correct": self.correct,
            "pipeline_summary": self.pipeline_summary,
            "ts_stats": self.ts_stats,
            "timestamp": self.timestamp,
        }


class TaskStrategyMemory:
    """Online learning memory for task-level reasoning strategies."""

    def __init__(self, enabled: bool = True):
        self._enabled = enabled
        self._strategies: dict[str, list[str]] = {}
        self._record_counts: dict[str, int] = defaultdict(int)
        self._pending_records: dict[str, list[dict]] = defaultdict(list)
        self._lock = asyncio.Lock()

        if enabled:
            STRATEGY_DIR.mkdir(parents=True, exist_ok=True)
            self._load_strategies()
            self._count_existing_records()

    def _load_strategies(self) -> None:
        if STRATEGY_FILE.exists():
            try:
                self._strategies = json.loads(STRATEGY_FILE.read_text())
                logger.info(
                    f"[TaskStrategy] Loaded strategies for "
                    f"{len(self._strategies)} domain-task pairs"
                )
            except Exception:
                self._strategies = {}

    def _count_existing_records(self) -> None:
        for f in STRATEGY_DIR.glob("records_*.jsonl"):
            key = f.stem.replace("records_", "")
            count = sum(1 for _ in f.open())
            self._record_counts[key] = count

    def get_strategy(self, domain: str, task: str) -> str:
        """Get the learned strategy prompt section for a domain-task pair."""
        if not self._enabled:
            return ""
        key = f"{domain}_{task}"
        rules = self._strategies.get(key, [])
        if not rules:
            return ""
        lines = [
            "## Learned Strategy (from past experience)\n"
            "Follow these rules carefully — they are derived from analyzing "
            "hundreds of similar samples:\n"
        ]
        for i, rule in enumerate(rules, 1):
            lines.append(f"{i}. {rule}")
        lines.append("")
        return "\n".join(lines)

    async def record_outcome(
        self,
        domain: str,
        task: str,
        predicted: str,
        ground_truth: str,
        correct: bool,
        pipeline_summary: str = "",
        ts_stats: str = "",
    ) -> None:
        """Record an evaluation outcome for future learning."""
        if not self._enabled:
            return
        record = OutcomeRecord(
            domain=domain, task=task,
            predicted=predicted, ground_truth=ground_truth,
            correct=correct, pipeline_summary=pipeline_summary,
            ts_stats=ts_stats,
        )
        key = f"{domain}_{task}"

        async with self._lock:
            self._pending_records[key].append(record.to_dict())
            self._record_counts[key] += 1

            if len(self._pending_records[key]) >= 20:
                self._flush_records(key)

            if (self._record_counts[key] >= _MIN_RECORDS_FOR_DISTILL
                    and self._record_counts[key] % _DISTILL_INTERVAL == 0):
                asyncio.create_task(self._distill_one(key))

    def _flush_records(self, key: str) -> None:
        records = self._pending_records.pop(key, [])
        if not records:
            return
        path = STRATEGY_DIR / f"records_{key}.jsonl"
        with path.open("a") as f:
            for r in records:
                f.write(json.dumps(r, default=str) + "\n")

    def flush_all(self) -> None:
        for key in list(self._pending_records.keys()):
            self._flush_records(key)

    async def _distill_one(self, key: str) -> None:
        """Distill strategy rules for one domain-task pair."""
        self._flush_records(key)
        path = STRATEGY_DIR / f"records_{key}.jsonl"
        if not path.exists():
            return

        records = []
        for line in path.open():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

        if len(records) < _MIN_RECORDS_FOR_DISTILL:
            return

        total = len(records)
        correct = sum(1 for r in records if r.get("correct"))
        acc = correct / total if total else 0

        gt_dist: dict[str, int] = defaultdict(int)
        pred_dist: dict[str, int] = defaultdict(int)
        confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for r in records:
            gt = r.get("ground_truth", "")
            pr = r.get("predicted", "")
            gt_dist[gt] += 1
            pred_dist[pr] += 1
            if not r.get("correct"):
                confusion[gt][pr] += 1

        summary = (
            f"Domain-Task: {key}\n"
            f"Total samples: {total}, Accuracy: {acc:.1%}\n\n"
            f"Ground truth distribution: {dict(gt_dist)}\n"
            f"Prediction distribution: {dict(pred_dist)}\n\n"
            f"Confusion (GT -> wrong predictions):\n"
        )
        for gt_label, wrong_preds in sorted(confusion.items()):
            for pred_label, count in sorted(wrong_preds.items(), key=lambda x: -x[1]):
                summary += f"  GT={gt_label} -> predicted {pred_label}: {count} times\n"

        sample_wrong = [r for r in records if not r.get("correct")][:10]
        sample_right = [r for r in records if r.get("correct")][:5]

        examples = "Wrong examples:\n"
        for r in sample_wrong:
            examples += (
                f"  GT={r.get('ground_truth')}, Pred={r.get('predicted')}, "
                f"Pipeline hint: {r.get('pipeline_summary', '')[:2000]}\n"
            )
        examples += "\nCorrect examples:\n"
        for r in sample_right:
            examples += (
                f"  GT={r.get('ground_truth')}, Pred={r.get('predicted')}, "
                f"Pipeline hint: {r.get('pipeline_summary', '')[:2000]}\n"
            )

        prompt = (
            "You are analyzing evaluation results for a time-series analysis system.\n"
            "Based on the error patterns below, produce 3-6 concise strategy RULES "
            "that would help a reasoning model make better predictions.\n\n"
            "Focus on:\n"
            "- Class imbalance: if the model over/under-predicts certain labels\n"
            "- When to trust vs ignore tool analysis\n"
            "- Common confusion patterns and how to avoid them\n"
            "- Domain-specific patterns observed in the data\n\n"
            f"{summary}\n{examples}\n\n"
            "Return JSON: {\"rules\": [\"rule 1\", \"rule 2\", ...]}\n"
            "Each rule should be a single actionable sentence."
        )

        try:
            from timeclaw.core.config import TimeClawConfig
            from timeclaw.llm.router import ModelRouter
            cfg = TimeClawConfig.load()
            router = ModelRouter(cfg.llm)
            llm = router.get_backend("reasoner")
            resp = await llm.complete(
                [{"role": "system", "content": "You produce concise strategy rules."},
                 {"role": "user", "content": prompt}],
                response_format="json",
            )
            if resp.parsed and "rules" in resp.parsed:
                self._strategies[key] = resp.parsed["rules"]
                STRATEGY_FILE.write_text(
                    json.dumps(self._strategies, indent=2, ensure_ascii=False)
                )
                logger.info(
                    f"[TaskStrategy] Distilled {len(resp.parsed['rules'])} rules "
                    f"for {key} (from {total} records, acc={acc:.1%})"
                )
        except Exception as e:
            logger.warning(f"[TaskStrategy] Distill failed for {key}: {e}")

    async def distill_all(self) -> None:
        """Force distillation for all domain-task pairs with enough data."""
        self.flush_all()
        tasks = []
        for f in STRATEGY_DIR.glob("records_*.jsonl"):
            key = f.stem.replace("records_", "")
            count = sum(1 for _ in f.open())
            if count >= _MIN_RECORDS_FOR_DISTILL:
                tasks.append(self._distill_one(key))
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
