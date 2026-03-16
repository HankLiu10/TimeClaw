"""TimeClaw CLI: command-line interface for running pipelines, benchmarks, and tools."""

from __future__ import annotations

import argparse
import asyncio
import sys
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="timeclaw",
        description="TimeClaw: Native Multimodal Time-Series Autonomous Agent Platform",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # timeclaw run
    run_parser = subparsers.add_parser("run", help="Run a TimeClaw pipeline on data")
    run_parser.add_argument("--data", type=str, required=True, help="Path to CSV data file")
    run_parser.add_argument("--task", type=str, default="forecast", choices=["forecast", "anomaly", "classification", "imputation", "qa"])
    run_parser.add_argument("--horizon", type=int, default=96)
    run_parser.add_argument("--model", type=str, default="auto", help="Model name or 'auto'")
    run_parser.add_argument("--output", type=str, default="results/run")

    # timeclaw bench
    bench_parser = subparsers.add_parser("bench", help="Run MTBench evaluation")
    bench_parser.add_argument("--tasks", nargs="+", default=None, help="Tasks to evaluate")
    bench_parser.add_argument("--output", type=str, default="results/mtbench")
    bench_parser.add_argument("--format", type=str, default="markdown", choices=["markdown", "latex", "json"])

    # timeclaw tools
    tools_parser = subparsers.add_parser("tools", help="List or run ClawHub tools")
    tools_parser.add_argument("--list", action="store_true", help="List all available tools")
    tools_parser.add_argument("--category", type=str, default=None, help="Filter by category")
    tools_parser.add_argument("--run", type=str, default=None, help="Run a specific tool by name")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    if args.command == "run":
        asyncio.run(_cmd_run(args))
    elif args.command == "bench":
        asyncio.run(_cmd_bench(args))
    elif args.command == "tools":
        _cmd_tools(args)


async def _cmd_run(args: Any) -> None:
    """Run a pipeline on data."""
    import numpy as np
    import pandas as pd
    from pathlib import Path

    from timeclaw.benchmark.datasets import DatasetLoader

    print(f"Loading data from {args.data}...")
    loader = DatasetLoader()
    dataset = loader.load(f"csv:{args.data}")

    train = dataset["train"]
    test = dataset.get("test", train[-100:])

    from timeclaw.clawhub.tool_registry import discover_and_register
    discover_and_register()

    from timeclaw.clawhub.tool_registry import get_tool_instance
    model_name = args.model or "chronos"
    tool_name = f"{model_name}_forecast"
    tool = get_tool_instance(tool_name)
    result = await tool.execute(
        ts_data=train,
        horizon=args.horizon,
    )

    if result.success:
        print(f"\nForecast ({args.task}, horizon={args.horizon}):")
        forecast = result.data.get("point_forecast", [])
        print(f"  Model: {result.data.get('model', 'unknown')}")
        print(f"  First 10 values: {forecast[:10]}")

        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        import json
        (out_dir / "forecast.json").write_text(json.dumps(result.data, indent=2, default=str))
        print(f"  Saved to {out_dir / 'forecast.json'}")
    else:
        print(f"Error: {result.error}")


async def _cmd_bench(args: Any) -> None:
    """Run MTBench evaluation."""
    from timeclaw.benchmark.mtbench import MTBenchRunner
    from experiments.run_mtbench import NaiveLastValue, NaiveTrend

    pipelines = {
        "naive_last": NaiveLastValue(),
        "naive_trend": NaiveTrend(),
    }

    runner = MTBenchRunner(tasks=args.tasks, output_dir=args.output)
    results = await runner.run(pipelines)
    report = runner.generate_report(results, format=args.format)
    print(report)


def _cmd_tools(args: Any) -> None:
    """List or run ClawHub tools."""
    from timeclaw.clawhub.tool_registry import discover_and_register, list_tool_instances

    n = discover_and_register()

    if args.list or args.run is None:
        tools = list_tool_instances(args.category)
        print(f"\nClawHub Tools ({len(tools)} registered, {n} discovered):")
        print(f"{'Name':<25} {'Category':<15} {'IO Type':<12} {'Stub':<6}")
        print("-" * 58)
        for t in sorted(tools, key=lambda x: (x.category.value, x.name)):
            stub = "Yes" if t.is_stub else "No"
            print(f"{t.name:<25} {t.category.value:<15} {t.io_type.value:<12} {stub:<6}")
        return

    if args.run:
        from timeclaw.clawhub.tool_registry import get_tool_instance
        try:
            tool = get_tool_instance(args.run)
            print(f"Tool: {tool.name}")
            print(f"  Category: {tool.category.value}")
            print(f"  IO Type: {tool.io_type.value}")
            print(f"  Description: {tool.description}")
            print(f"  Schema: {tool.parameters_schema}")
        except KeyError as e:
            print(f"Error: {e}")


# Entry point for pyproject.toml [project.scripts]
if __name__ == "__main__":
    main()
