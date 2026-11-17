"""Regression checks for the two real TimeClaw experiment branches.

These tests are intentionally small and local. They prevent a retired wrapper or
the restricted v2 smoke pilot from silently becoming the source of benchmark
claims again.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from timeclaw.benchmark import run_learning
from timeclaw.pipeline import runner as pipeline_runner
from timeclaw import cli


ROOT = Path(__file__).resolve().parents[2]


def test_learning_entrypoint_stays_on_independent_train_path() -> None:
    source = inspect.getsource(run_learning.cmd_train)
    train_loop_source = inspect.getsource(run_learning)
    assert "run_train_loop" in source
    assert "benchmark.training_root" in train_loop_source
    assert "await cmd_test(args)" not in inspect.getsource(run_learning.cmd_all)


def test_training_core_is_explicitly_train_mode() -> None:
    train_loop = (ROOT / "timeclaw" / "benchmark" / "train_loop.py").read_text()
    runner = inspect.getsource(pipeline_runner.PipelineRunner.run)
    assert 'mode="train"' in train_loop
    assert 'mode="test"' in runner or 'mode: str = "test"' in runner
    assert "ground_truth=gt" in train_loop


def test_inference_script_uses_run_inference_and_current_default_config() -> None:
    source = (ROOT / "scripts" / "run_timeclaw_mtbench_inference.py").read_text()
    assert "runner.run_inference(" in source
    assert "pipeline.run(" not in source

    assert (
        'DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"'
        in source
    )






@pytest.mark.asyncio
@pytest.mark.parametrize("distillation_fails", [False, True])
async def test_training_cli_accounts_for_preparation_and_finalization(tmp_path, monkeypatch, distillation_fails):
    from timeclaw.benchmark import train_loop
    from timeclaw.core.config import TimeClawConfig
    from timeclaw.core.accounting import logical_call, physical_attempt

    class Backend:
        @logical_call
        async def complete(self):
            with physical_attempt("test/glm") as attempt:
                attempt["usage"] = {"prompt_tokens": 10, "completion_tokens": 2}

    async def train(**kwargs):
        assert kwargs['concurrency'] == 5
        await Backend().complete()  # Text preparation before a sample trace.
        await Backend().complete()  # Memory finalization after sample traces.
        if distillation_fails:
            raise RuntimeError("distillation unfinished")
        target = Path(cfg.workspace_dir)/"train_results/train_summary.json"
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text('{"finance_aligned_short":{"total":1}}')
        return {"finance_aligned_short": {"total": 1}}

    cfg = TimeClawConfig(workspace_dir=str(tmp_path))
    monkeypatch.setattr(run_learning.TimeClawConfig, "load", lambda: cfg)
    monkeypatch.setattr(train_loop, "run_train_loop", train)
    args = SimpleNamespace(dataset="finance_aligned_short", task="forecast", max_samples=None,
                           sample_index_file=None, verbose=False, concurrency=5,
                           per_task_concurrency=5, indicator_concurrency=None, workspace_dir=str(tmp_path))
    if distillation_fails:
        with pytest.raises(RuntimeError, match="distillation unfinished"):
            await run_learning.cmd_train(args)
    else:
        await run_learning.cmd_train(args)
    summaries = list((tmp_path / 'accounting').glob('*.summary.json'))
    assert len(summaries) == 1
    summary = json.loads(summaries[0].read_text())
    assert summary['logical_calls'] == summary['physical_attempts'] == 2
    assert summary['observed_prompt_tokens'] == 20
    assert summary['usage_covers_all_attempts']
    assert summary['status'] == ('failed' if distillation_fails else 'completed')
    assert 'online_seconds' not in summary
    assert (tmp_path / 'train_results/train_summary.json').exists() is not distillation_fails


@pytest.mark.asyncio
async def test_all_alias_does_not_repeat_paid_distillation(monkeypatch):
    from unittest.mock import AsyncMock
    train=AsyncMock();distill=AsyncMock()
    monkeypatch.setattr(run_learning,'cmd_train',train)
    monkeypatch.setattr(run_learning,'cmd_distill',distill)
    args=SimpleNamespace()
    await run_learning.cmd_all(args)
    train.assert_awaited_once_with(args)
    distill.assert_not_awaited()


def test_learning_download_honors_configured_data_root_and_dataset(tmp_path,monkeypatch):
    from unittest.mock import Mock
    from timeclaw.core.config import TimeClawConfig
    cfg=TimeClawConfig();cfg.benchmark.repo_root=str(tmp_path)
    monkeypatch.setattr(run_learning.TimeClawConfig,'load',lambda:cfg)
    download=Mock();monkeypatch.setattr('scripts.download_mtbench.download_all',download)
    run_learning.cmd_download(SimpleNamespace(dataset='weather_aligned_short'))
    download.assert_called_once_with(datasets=['weather_aligned_short'],repo_root=tmp_path)
