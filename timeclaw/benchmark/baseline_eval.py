"""Main-table direct-LLM baseline using the paper18 prompts and shared scorer.

One task makes one completion call. No agent tools, model substitution, result
selection, no-memory ablation or secondary reasoning-only experiment is run.
"""
from __future__ import annotations

import argparse
import asyncio
from contextlib import contextmanager
import fcntl
import hashlib
import json
import time
from pathlib import Path
from typing import Any


from timeclaw.benchmark import mtbench_prompts as prompts
from timeclaw.benchmark.mtbench_adapter import (
    MTBenchSample, datasets_with_benchmark_tasks, load_dataset,
)
from timeclaw.benchmark.mtbench_support import load_sample_indices, select_samples_with_original_indices
from timeclaw.benchmark.mtbench_support.task_scoring import (
    score_mtbench_prediction, validate_mtbench_sample_for_scoring,
)
from timeclaw.benchmark.source_admission import sample_digest
from timeclaw.benchmark.task_registry import TASK_SET, tasks_for_dataset, task_registry_digest, require_paper_task
from timeclaw.core.accounting import RequestLedger
from timeclaw.core.config import TimeClawConfig
from timeclaw.llm.router import ModelRouter

# Optional Python caller override. CLI outputs default to the configured workspace.
RESULTS_DIR: Path | None = None


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def _prediction(text: str, sample: MTBenchSample, task: str) -> Any:
    if task == "forecast":
        parser = prompts.parse_weather_forecast_output if sample.domain == "weather" else prompts.parse_finance_value_output
        return parser(text)
    if task == "indicator_macd":
        values = prompts.parse_finance_value_output(text)
        return {"macd": values} if values is not None else None
    if task == "indicator":
        return prompts.parse_weather_indicator_output(text)
    return text


async def _eval_one_sample(ds_name: str, idx: int, sample: MTBenchSample,
                           tasks: list[str], llm: Any, sem: asyncio.Semaphore, task_set: str = TASK_SET) -> list[dict[str, Any]]:
    from timeclaw.benchmark.train_loop import _extract_ground_truth
    rows = []
    async with sem:
        for task in tasks:
            require_paper_task(ds_name, task, task_set)
            row = {"dataset": ds_name, "idx": idx, "file": sample.file,
                   "domain": sample.domain, "task": task}
            valid, reason = validate_mtbench_sample_for_scoring(sample, task)
            ground_truth = _extract_ground_truth(sample, task)
            if not valid or ground_truth is None:
                rows.append({**row, "status": "invalid_input", "metrics": {"skipped": reason or "missing_ground_truth"}})
                continue
            started = time.perf_counter()
            try:
                system, user = prompts.build_prompt(sample, task)
                response = await llm.complete(prompts.prompt_messages(system, user))
                prediction = _prediction(response.text or "", sample, task)
                metrics = score_mtbench_prediction(prediction, ground_truth, task)
                failed = bool(metrics.get("parse_error") or metrics.get("type_error") or metrics.get("error") or metrics.get("skipped"))
                row.update(status="failed" if failed else "completed", prediction=prediction,
                           raw_response=response.text, metrics=metrics)
            except Exception as exc:
                row.update(status="failed", prediction=None, metrics={},
                           error=str(exc), error_type=type(exc).__name__)
            row["request_seconds"] = time.perf_counter() - started
            rows.append(row)
    return rows


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    from timeclaw.benchmark.mtbench_support.official_metric_policy import summarize_official_inference_rows
    return {"official_filtered": summarize_official_inference_rows(rows),
            "all_valid": summarize_official_inference_rows(rows, apply_thresholds=False)}


@contextmanager
def _baseline_output_lease(root: Path):
    """One process owns an output directory from admission through cleanup.

    The empty lock file persists to keep one inode across competing launches;
    closing its descriptor releases ownership, including after failure.
    """
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.direct_baseline.lock').open('a') as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Baseline output {root} already has an active writer") from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


async def run_baseline(dataset_name: str | None = None, task_filter: str | None = None,
                       config: TimeClawConfig | None = None, max_samples: int | None = None,
                       sample_indices: dict[str, list[int]] | str | Path | None = None,
                       concurrency: int = 5, full_dataset: bool = True,
                       reuse_existing: bool = True, output_dir: Path | None = None) -> dict[str, Any]:
    """Run direct-LLM main baselines with immutable, verified per-dataset receipts.

    A receipt binds selected input/target hashes, task list, model/configuration,
    source code and saved rows. Existing files without that exact receipt fail
    before model creation; use a new output directory for a different run.
    Evaluation is always official evaluation data, regardless of the historical
    full_dataset parameter. Invalid and failed rows stay in total counts.
    """
    if concurrency < 1 or (max_samples is not None and max_samples < 1):
        raise ValueError("concurrency and any max_samples must be positive")
    cfg = config or TimeClawConfig.load()
    root = Path(output_dir or RESULTS_DIR or Path(cfg.workspace_dir) / "test_results")
    with _baseline_output_lease(root):
        names = [dataset_name] if dataset_name else datasets_with_benchmark_tasks()
        indices = load_sample_indices(sample_indices) if isinstance(sample_indices, (str, Path)) else sample_indices
        code_root = Path(__file__).resolve().parents[1]
        code = {str(p.relative_to(code_root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(code_root.rglob("*.py"))}
        visible_config = cfg.model_dump(mode="json", exclude={"llm": {"api_key", "azure_api_key"}})
        pending = []
        results = {}
        for name in names:
            tasks = tasks_for_dataset(name, cfg.benchmark.task_set)
            if not tasks:
                raise ValueError(f"Not a registered benchmark dataset: {name}")
            if task_filter:
                require_paper_task(name, task_filter, cfg.benchmark.task_set)
                tasks = [task_filter]
            samples = load_dataset(name, config=cfg)
            selected = select_samples_with_original_indices(samples, name, indices)
            if max_samples is not None:
                selected = selected[:max_samples]
            if not selected:
                raise ValueError(f"No evaluation samples selected for {name}")
            identity = _digest({"schema": "timeclaw_direct_baseline_v1", "task_registry": task_registry_digest(cfg.benchmark.task_set),
                                "tasks": tasks, "config": visible_config, "code": code,
                                "samples": [(idx, sample.file, sample_digest(sample, targets=True)) for idx, sample in selected]})
            path = root / f"baseline_{name}.json"
            receipt = root / f"baseline_{name}.receipt.json"
            if path.exists() or receipt.exists():
                if not reuse_existing:
                    raise FileExistsError(f"Preserve existing baseline evidence; select a new output directory: {path}")
                if not (path.is_file() and receipt.is_file()):
                    raise ValueError(f"Unverified baseline without a matching receipt: {path}")
                record = json.loads(receipt.read_text())
                if record.get("identity") != identity or record.get("rows_sha256") != hashlib.sha256(path.read_bytes()).hexdigest():
                    raise ValueError(f"Baseline receipt mismatch: {path}")
                rows = json.loads(path.read_text())
                if any(row["status"] != "completed" for row in rows):
                    raise ValueError(f"Baseline receipt contains unsuccessful rows; retain evidence and use a new output directory: {path}")
                results[name] = {"total": len(rows), "failed": 0, "summary": _summarize(rows), "reused": True}
            else:
                pending.append((name, tasks, selected, identity, path, receipt))
        if not pending:
            return results
        root.mkdir(parents=True, exist_ok=True)
        ledger = RequestLedger("direct_baseline", root / "accounting")
        router = None
        try:
            with ledger.activate():
                router = ModelRouter(cfg.llm)
                llm = router.get_backend("executor")
                sem = asyncio.Semaphore(concurrency)
                for name, tasks, selected, identity, path, receipt in pending:
                    started = time.perf_counter()
                    batches = await asyncio.gather(*[_eval_one_sample(name, idx, sample, tasks, llm, sem, cfg.benchmark.task_set) for idx, sample in selected])
                    rows = [row for batch in batches for row in batch]
                    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2))
                    receipt.write_text(json.dumps({"schema": "timeclaw_direct_baseline_v1", "identity": identity,
                        "rows_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "rows": len(rows)}, indent=2))
                    results[name] = {"total": len(rows), "failed": sum(row["status"] != "completed" for row in rows),
                                      "elapsed_s": time.perf_counter()-started, "summary": _summarize(rows), "reused": False}
        finally:
            try:
                if router is not None:
                    await router.shutdown()
            finally:
                ledger.path.with_suffix(".summary.json").write_text(json.dumps(ledger.summary(), indent=2))
        return results


async def main() -> None:
    parser = argparse.ArgumentParser(description="Paper18 direct-LLM main-table baseline")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--dataset")
    parser.add_argument("--task")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--sample-index-file", type=Path)
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    result = await run_baseline(dataset_name=args.dataset, task_filter=args.task,
        config=TimeClawConfig.load(args.config), max_samples=args.max_samples,
        sample_indices=args.sample_index_file, concurrency=args.concurrency, output_dir=args.output_dir)
    print(json.dumps(result, indent=2))
    if not result or any(info["failed"] for info in result.values()):
        raise RuntimeError("Baseline has failed/invalid rows; all outcomes and accounting were retained")


if __name__ == "__main__":
    asyncio.run(main())
