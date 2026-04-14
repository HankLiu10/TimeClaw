#!/usr/bin/env python3
"""Stable learning-summary recheck runner.

New definition:
    This script is only for verification. It runs one sample per task and
    writes a per-run JSON artifact to a unique file, so an old recheck file
    cannot be mistaken for the current run.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from timeclaw.benchmark.dataset_split import load_split
from timeclaw.benchmark.mtbench_adapter import DATASET_REGISTRY
from timeclaw.benchmark.mtbench_support import (
    build_task_description,
    scope_keys_for_dataset_task,
)
from timeclaw.benchmark.train_loop import (
    _build_train_result_row,
    _extract_ground_truth,
    _write_json_atomically,
)
from timeclaw.core.config import TimeClawConfig
from timeclaw.pipeline.runner import PipelineRunner

PIPELINE_TASK = {
    "forecast": "forecast",
    "trend": "classification",
    "trend_past": "classification",
    "indicator_macd": "indicator",
    "indicator_upper_bb": "indicator",
    "mcqa": "qa",
    "correlation": "qa",
}


def _expanded_pairs() -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for ds_name, info in DATASET_REGISTRY.items():
        for task in info.get("tasks", []):
            if ds_name.startswith("finance_aligned") and task == "indicator":
                pairs.append((ds_name, "indicator_macd"))
                pairs.append((ds_name, "indicator_upper_bb"))
            else:
                pairs.append((ds_name, task))
    return pairs


def _default_output_path(memory_enabled: bool) -> Path:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = "with_memory" if memory_enabled else "no_memory"
    return Path(".timeclaw/tmp") / f"recheck_alignment_{suffix}_{run_id}.json"


async def _run_one(
    runner: PipelineRunner,
    cfg: TimeClawConfig,
    *,
    ds_name: str,
    task: str,
    sample_idx: int,
) -> dict[str, Any]:
    sample = load_split(ds_name, "train", config=cfg)[sample_idx]
    pipeline_task = PIPELINE_TASK[task]
    result = await runner.run(
        data=sample.input_window,
        task_type=pipeline_task,
        description=build_task_description(sample, task),
        text_data=[{"content": sample.text, "source": sample.domain, "timestamp": ""}] if sample.text else None,
        domain=sample.domain,
        scope_keys=scope_keys_for_dataset_task(ds_name, task),
        mode="train",
        ground_truth=_extract_ground_truth(sample, task),
        source_metadata={
            "dataset_name": ds_name,
            "sample_ref": f"{ds_name}:{sample.file}:{sample_idx}:{task}",
            "source_file": sample.file,
            "sample_idx": sample_idx,
            "task_name": task,
            "pipeline_task": pipeline_task,
            "domain": sample.domain,
        },
    )
    row = _build_train_result_row(
        ds_name=ds_name,
        sample_idx=sample_idx,
        sample=sample,
        task=task,
        pipeline_task=pipeline_task,
        scope_keys=scope_keys_for_dataset_task(ds_name, task),
        result=result,
    )
    raw_answer = result.get("raw_answer") if isinstance(result.get("raw_answer"), dict) else {}
    return {
        "dataset": ds_name,
        "task": task,
        "trace_id": result.get("trace_id"),
        "answer_type": result.get("answer_type"),
        "result_winner_tool_chain": raw_answer.get("winner_tool_chain"),
        "row_tool_chain": row.get("tool_chain"),
        "reasoning_preview": row.get("reasoning_preview"),
        "evidence_summary_preview": row.get("evidence_summary_preview"),
    }


async def async_main() -> None:
    parser = argparse.ArgumentParser(description="One-sample-per-task recheck with unique output path")
    parser.add_argument("--sample-idx", type=int, default=0, help="train split sample index")
    parser.add_argument("--max-pairs", type=int, default=None, help="only run the first N expanded tasks")
    parser.add_argument("--disable-memory", action="store_true", help="run without memory injection")
    parser.add_argument("--output", type=Path, default=None, help="output JSON path; defaults to a unique path")
    args = parser.parse_args()

    cfg = TimeClawConfig.load()
    runner = PipelineRunner(cfg, disable_memory=args.disable_memory)
    memory_enabled = not args.disable_memory
    output_path = args.output or _default_output_path(memory_enabled)
    pairs = _expanded_pairs()
    if args.max_pairs is not None:
        pairs = pairs[: max(args.max_pairs, 0)]

    payload: dict[str, Any] = {
        "memory_enabled": memory_enabled,
        "sample_idx": args.sample_idx,
        "output_path": str(output_path),
        "rows": [],
    }

    try:
        for ds_name, task in pairs:
            print(f"RUN {ds_name}::{task}", flush=True)
            row = await _run_one(
                runner,
                cfg,
                ds_name=ds_name,
                task=task,
                sample_idx=args.sample_idx,
            )
            payload["rows"].append(row)
            _write_json_atomically(output_path, payload)
            reloaded = json.loads(output_path.read_text())
            last = reloaded["rows"][-1]
            print(
                f"WROTE {row['dataset']}::{row['task']} winner={row['result_winner_tool_chain']}",
                flush=True,
            )
            print(
                f"RELOAD_LAST {last['dataset']}::{last['task']} winner={last['result_winner_tool_chain']}",
                flush=True,
            )
    finally:
        await runner.shutdown()

    print(f"OUTPUT {output_path}", flush=True)


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
