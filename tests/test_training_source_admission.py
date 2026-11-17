"""Training source failures cannot become successful jobs or trigger model startup."""
from unittest.mock import Mock
import pytest
from timeclaw.core.config import TimeClawConfig
from timeclaw.benchmark import train_loop
from timeclaw.sbm import service_manager


def test_explicitly_disabled_managed_services_do_not_allocate_models(monkeypatch):
    cfg = TimeClawConfig()
    cfg.sbm.managed_server_enabled = False
    spawn = Mock(side_effect=AssertionError('Unexpected managed service startup'))
    monkeypatch.setattr(service_manager, '_start_one_model_server', spawn)
    assert service_manager.ensure_managed_sbm_server(cfg) == {}
    spawn.assert_not_called()


@pytest.mark.asyncio
async def test_missing_independent_data_fails_before_allocating_models(tmp_path, monkeypatch):
    cfg = TimeClawConfig(workspace_dir=str(tmp_path/'artifacts'))
    cfg.benchmark.repo_root = str(tmp_path)
    cfg.benchmark.training_root = 'missing-independent-corpus'
    start = Mock()
    runner = Mock()
    monkeypatch.setattr(service_manager, 'ensure_managed_sbm_server', start)
    monkeypatch.setattr(train_loop, 'PipelineRunner', runner)
    with pytest.raises(FileNotFoundError, match='Independent dataset directory'):
        await train_loop.run_train_loop(dataset_name='weather_aligned_short', config=cfg)
    start.assert_not_called()
    runner.assert_not_called()
    assert not (tmp_path/'artifacts/train_results/train_summary.json').exists()


@pytest.mark.asyncio
async def test_runner_constructor_typeerror_is_not_reinterpreted_as_old_signature(tmp_path, monkeypatch):
    cfg = TimeClawConfig(workspace_dir=str(tmp_path/'artifacts'))
    cfg.sbm.managed_server_enabled = False
    monkeypatch.setattr(train_loop, 'load_independent_dataset', lambda *a, **kw: [])
    monkeypatch.setattr(train_loop, 'begin_training_lineage', lambda *a, **kw: None)
    monkeypatch.setattr(train_loop, 'finish_training_lineage', lambda *a, **kw: None)
    broken = Mock(side_effect=TypeError('constructor implementation failed'))
    monkeypatch.setattr(train_loop, 'PipelineRunner', broken)
    with pytest.raises(TypeError, match='constructor implementation failed'):
        await train_loop.run_train_loop(dataset_name='weather_aligned_short', config=cfg)
    broken.assert_called_once()


@pytest.mark.asyncio
async def test_preparation_error_is_preserved_as_failed_sample(tmp_path, monkeypatch):
    import numpy as np
    from timeclaw.benchmark.mtbench_adapter import MTBenchSample
    from types import SimpleNamespace
    async def reject(*args, **kwargs):
        raise ValueError('provider rejected an invalid request parameter')
    monkeypatch.setattr(train_loop, 'prepare_task_run_kwargs', reject)
    sample = MTBenchSample(file='fixture.json', dataset='weather_aligned_short', domain='weather',
                           task_types=['forecast'], input_window=np.arange(168.),
                           output_window=np.arange(24.), text='Given context', timestamps=list(range(168)),
                           start_year=2024, forecast_horizon=24)
    dataset, rows, error = await train_loop._run_task_work_item(
        ds_name='weather_aligned_short', sample_idx=0, sample=sample, task='forecast',
        pipeline=SimpleNamespace(), sem=None)
    assert error is None and dataset == 'weather_aligned_short'
    assert len(rows) == 1 and rows[0]['status'] == 'failed'
    assert rows[0]['error_type'] == 'ValueError'
    assert rows[0]['has_prediction'] is False


@pytest.mark.asyncio
async def test_cli_reports_failure_after_saving_failed_batch(tmp_path, monkeypatch):
    from timeclaw.benchmark import run_learning
    from types import SimpleNamespace
    cfg = TimeClawConfig(workspace_dir=str(tmp_path))
    monkeypatch.setattr(TimeClawConfig, 'load', lambda: cfg)
    async def failed_job(**kwargs):
        target=tmp_path/'train_results/train_summary.json'
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text('{}')
        return {'weather_aligned_short': {'total': 2, 'summary': {
            'trend_past': {'aligned_counts': {'execution_failed_n': 2}}}}}
    monkeypatch.setattr(train_loop, 'run_train_loop', failed_job)
    args = SimpleNamespace(dataset='weather_aligned_short', task='trend_past', max_samples=2,
        sample_index_file=None, verbose=False, concurrency=1, per_task_concurrency=1,
        indicator_concurrency=None, workspace_dir=str(tmp_path))
    with pytest.raises(RuntimeError, match='execution_failed=2'):
        await run_learning.cmd_train(args)
    assert (tmp_path/'train_results/train_summary.json').is_file()
    assert len(list((tmp_path/'accounting').glob('*.summary.json'))) == 1
