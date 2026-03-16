"""Run 12 training samples to trigger L1->L2 distillation and verify results."""
import asyncio
import json
import sys
import os

os.environ.setdefault("TIMECLAW_CONFIG", "configs/claude_proxy.yaml")


async def main():
    from timeclaw.core.config import TimeClawConfig
    from timeclaw.pipeline.runner import PipelineRunner
    from timeclaw.benchmark.mtbench_adapter import load_dataset
    from timeclaw.benchmark.train_loop import _extract_ground_truth

    cfg = TimeClawConfig.load()
    cfg.mcp.enabled = False
    pipeline = PipelineRunner(cfg, disable_memory=False)

    samples = load_dataset("weather_aligned_short")
    n_samples = min(12, len(samples))

    print(f"=== Distillation Test: {n_samples} training samples ===\n")

    for i in range(n_samples):
        sample = samples[i]
        gt = _extract_ground_truth(sample, "forecast")
        horizon = len(sample.output_window) if sample.output_window is not None else 24
        text_data = [{"text": sample.text}] if sample.text else None

        print(f"--- Sample {i+1}/{n_samples} ---")
        try:
            result = await pipeline.run(
                data=sample.input_window,
                task_type="forecast",
                description="Predict the next 24 hourly temperature values based on the historical hourly observations.",
                domain=sample.domain,
                horizon=horizon,
                text_data=text_data,
                mode="train",
                ground_truth=gt,
            )
            atype = result.get("answer_type", "")
            n_tools = len(result.get("execution_log", []))
            print(f"  answer_type={atype}, tools={n_tools}")
        except Exception as e:
            print(f"  ERROR: {e}")

        mem = pipeline.memory_summary()
        print(f"  Memory: daily={mem.get('daily_notes',0)}, L2={mem.get('memory_insights',0)}, "
              f"L3={mem.get('principles',0)}, tools={mem.get('tool_notes',0)}")
        print()

    print("\n=== Final Memory State ===")
    mem = pipeline.memory_summary()
    print(json.dumps(mem, default=str, indent=2))

    # Show L2 contents if any
    from pathlib import Path
    l2_dir = Path(".timeclaw/hierarchical_memory/memory")
    if l2_dir.exists():
        for f in l2_dir.glob("*.json"):
            print(f"\n=== L2 MEMORY: {f.stem} ===")
            data = json.loads(f.read_text())
            for j, entry in enumerate(data):
                insight = entry.get("insight", "")[:300]
                print(f"  [{j+1}] {insight}")
    else:
        print("\n[WARNING] No L2 memory files found — distillation did not trigger")

    # Show L1 daily notes
    daily_dir = Path(".timeclaw/hierarchical_memory/daily")
    if daily_dir.exists():
        for f in sorted(daily_dir.glob("*.md")):
            content = f.read_text()
            n_entries = content.count("### [")
            print(f"\n=== L1 Daily Notes: {f.name} ({n_entries} entries) ===")
            # Show first entry as sample
            lines = content.strip().split("\n")
            for line in lines[:6]:
                print(f"  {line[:200]}")
            if len(lines) > 6:
                print(f"  ... ({len(lines)} total lines)")

    # Show TOOLS.md if any
    tools_path = Path(".timeclaw/hierarchical_memory/TOOLS.md")
    if tools_path.exists():
        print(f"\n=== TOOLS.md ===")
        data = json.loads(tools_path.read_text())
        for tool, note in data.items():
            print(f"  {tool}: {note[:150]}")

    await pipeline.shutdown()
    print("\n=== Done ===")


if __name__ == "__main__":
    asyncio.run(main())
