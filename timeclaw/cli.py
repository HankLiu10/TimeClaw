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
    run_parser.add_argument("--description", type=str, default="", help="Task instruction for the ReAct agent")
    run_parser.add_argument("--text", type=str, default=None, help="Optional UTF-8 context file")
    run_parser.add_argument("--output", type=str, default="results/run")

    # timeclaw tools
    tools_parser = subparsers.add_parser("tools", help="List or run ClawHub tools")
    tools_parser.add_argument("--list", action="store_true", help="List all available tools")
    tools_parser.add_argument("--category", type=str, default=None, help="Filter by category")
    tools_parser.add_argument("--run", type=str, default=None, help="Inspect the registered tool schema by name")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    if args.command == "run":
        asyncio.run(_cmd_run(args))
    elif args.command == "tools":
        _cmd_tools(args)


async def _cmd_run(args: Any) -> None:
    """Execute label-free ReAct over every supplied numeric CSV observation."""
    import json
    from pathlib import Path
    import pandas as pd
    from timeclaw.core.config import TimeClawConfig
    from timeclaw.benchmark.train_loop import _configure_workspace_roots
    from timeclaw.pipeline.runner import PipelineRunner
    from timeclaw.sbm.service_manager import ensure_managed_sbm_server, shutdown_managed_sbm_server

    cfg = TimeClawConfig.load()
    from timeclaw.benchmark.source_admission import assert_inference_lineage, workspace_access
    with workspace_access(cfg, write=False):
        assert_inference_lineage(cfg, {}, disable_memory=False)
        _configure_workspace_roots(cfg.workspace_dir)
        frame = pd.read_csv(args.data)
        data = frame.to_numpy(dtype=float)
        text_data = []
        if args.text is not None:
            text_data = [{"text": Path(args.text).read_text(encoding="utf-8")}]
        runner = PipelineRunner(cfg, drain_memory_on_shutdown=False)
        try:
            ensure_managed_sbm_server(cfg)
            result = await runner.run(
                data=data, task_type=args.task, description=args.description,
                horizon=args.horizon if args.task == "forecast" else None,
                column_names=list(frame.columns), text_data=text_data, mode="test",
            )
            out_dir = Path(args.output)
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "result.json").write_text(json.dumps(result, indent=2, default=str))
            print(f"Result saved to {out_dir / 'result.json'}")
            # Native run() returns the quality gate explicitly, without a status
            # field. Writing a rejected payload is evidence preservation, not a
            # successful command; genuine numeric zero remains a valid prediction.
            if result.get("gate_accepted") is not True or result.get("prediction") is None or result.get("error"):
                raise RuntimeError(f"Agent failed or its result was rejected; inspect {out_dir / 'result.json'}")
        finally:
            try:
                await runner.shutdown()
            finally:
                try:
                    shutdown_managed_sbm_server()
                finally:
                    assert_inference_lineage(cfg, {}, disable_memory=False)




def _cmd_tools(args: Any) -> None:
    """List ClawHub tools or inspect a registered tool schema."""
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
