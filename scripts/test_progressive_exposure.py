"""Quick test: progressive exposure + forced competition + memory feedback.

Runs 40 forecast samples from weather_aligned_short train set in batches,
with tool_insights refreshed between batches (simulating the real train loop).
"""
import asyncio
import sys
from pathlib import Path
from collections import Counter, defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from timeclaw.benchmark.train_loop import (
    _train_forecast, _progressive_expose, _refresh_insights_from_pool,
    FORECAST_TOOLS,
)
from timeclaw.benchmark.dataset_split import load_split
from timeclaw.core.config import TimeClawConfig
from timeclaw.llm.router import ModelRouter
from timeclaw.clawhub.tool_registry import discover_and_register
from timeclaw.memory.experience_pool import ExperienceEntry, ExperiencePool

import tempfile


async def main():
    discover_and_register()
    cfg = TimeClawConfig.load()
    router = ModelRouter(cfg.llm)
    llm = router.get_backend("executor")

    samples = load_split("weather_aligned_short", "train")[:40]
    total = len(samples)
    batch_size = 10

    pool = ExperiencePool(storage_dir=tempfile.mkdtemp())
    tool_insights: dict[str, str] = {}

    print(f"=== Progressive Exposure + Forced Competition + Memory ===")
    print(f"Total: {total}, batch_size: {batch_size}, tools: {FORECAST_TOOLS}\n")

    primary_counter = Counter()
    explore_counter = Counter()
    best_counter = Counter()
    batch_num = 0

    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch_num += 1

        if tool_insights:
            print(f"\n  [Memory] Insights fed to LLM for batch {batch_num}:")
            for t, ins in tool_insights.items():
                print(f"    {t}: {ins}")

        print(f"\n--- Batch {batch_num} (samples {batch_start}-{batch_end-1}) ---")

        for i in range(batch_start, batch_end):
            sample = samples[i]
            result = await _train_forecast(
                sample, llm, tool_insights,
                sample_idx=i, total_samples=total,
            )
            p = result["primary_tool"]
            e = result["explore_tool"]
            b = result["best_tool"]
            vis = result.get("visible_tools", FORECAST_TOOLS)
            primary_counter[p] += 1
            explore_counter[e] += 1
            if b:
                best_counter[b] += 1

            phase = "warmup" if i < total // 2 else "full"
            gt_summary = {t: f"MAE={m.get('mae', '?'):.4f}" if isinstance(m, dict) and 'mae' in m else "err"
                          for t, m in result["gt_metrics"].items()}
            print(f"  [{i:3d}] vis={len(vis)} primary={p} explore={e} best={b} | {gt_summary}")

            entry = ExperienceEntry(
                trace_id=f"test_{i:04d}_forecast",
                task_type="forecast",
                task_description=f"test forecast sample {i}",
                fingerprint_summary=result.get("fingerprint", ""),
                tool_chain=[p, e],
                evidence_steps=1,
                quality_gate_passed=result.get("quality_score", 0) > 0.5,
                quality_score=result.get("quality_score", 0.0),
                prediction_summary="",
                ground_truth_metrics=result.get("gt_metrics", {}),
                metadata={"best_tool": b, "primary_tool": p, "explore_tool": e},
            )
            pool.add(entry)

        tool_insights = _refresh_insights_from_pool(pool)

    print("\n" + "=" * 60)
    print("=== FINAL RESULTS ===")
    print(f"Primary tool distribution: {dict(primary_counter)}")
    print(f"Explore tool distribution: {dict(explore_counter)}")
    print(f"Best tool (ground truth winner): {dict(best_counter)}")
    print(f"\nAll tool appearances (primary+explore):")
    combined = Counter()
    for t in FORECAST_TOOLS:
        combined[t] = primary_counter[t] + explore_counter[t]
    for t, c in combined.most_common():
        print(f"  {t}: {c} ({c / (2 * total) * 100:.1f}%)")

    print(f"\nFinal insights:")
    for t, ins in tool_insights.items():
        print(f"  {t}: {ins}")


if __name__ == "__main__":
    asyncio.run(main())
