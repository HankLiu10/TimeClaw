"""Main-table baselines must share task boundaries and scoring with TimeClaw."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
import json

import numpy as np
import pytest

from timeclaw.benchmark import baseline_eval as baseline
from timeclaw.benchmark import mtbench_adapter as adapter
from timeclaw.benchmark.mtbench_support.task_scoring import score_mtbench_prediction
from timeclaw.benchmark.task_registry import BENCHMARK_TASKS
from timeclaw.core.config import TimeClawConfig


def sample(dataset):
    domain=adapter.DATASET_REGISTRY[dataset]['domain']
    return adapter.MTBenchSample(file='record.json',dataset=dataset,domain=domain,
        task_types=adapter.DATASET_REGISTRY[dataset]['tasks'],
        input_window=np.arange(336.)*.01+20.,output_window=np.arange(24.)+30.,
        timestamps=[1704067200.+3600*i for i in range(336)],start_year=2024,
        text='Visible context',question='Which option? A: rain B: dry C: snow D: wind',answer='A',
        labels={'out_macd':[1.,2.,3.],'out_upper_bb':[4.,5.,6.],
                'trend_label':'+2% ~ +4%','correlation':'Strong Positive Correlation'},forecast_horizon=24,
        metadata={'output_timestamps':[1704067200.+3600*i for i in range(336,360)]})


@pytest.mark.asyncio
async def test_macd_main_task_does_not_secretly_call_bollinger_prediction():
    import asyncio
    llm=SimpleNamespace(complete=AsyncMock(return_value=SimpleNamespace(text='[1,2,3]')))
    rows=await baseline._eval_one_sample('finance_aligned_short',0,sample('finance_aligned_short'),
        ['indicator_macd'],llm,asyncio.Semaphore(1))
    assert llm.complete.await_count==1
    assert len(rows)==1
    assert not any('upper_bb' in key for key in rows[0]['metrics'])


@pytest.mark.asyncio
async def test_unverified_old_baseline_is_not_silently_reused(tmp_path,monkeypatch):
    cfg=TimeClawConfig(workspace_dir=str(tmp_path))
    old=tmp_path/'test_results';old.mkdir()
    (old/'baseline_weather_aligned_short.json').write_text('[{"task":"mcqa","metrics":{"accuracy":1.0}}]')
    monkeypatch.setattr(baseline,'RESULTS_DIR',old)
    monkeypatch.setattr(baseline,'load_dataset',lambda *a,**kw:[sample('weather_aligned_short')])
    with pytest.raises(ValueError,match='receipt|unverified'):
        await baseline.run_baseline(dataset_name='weather_aligned_short',task_filter='forecast',
            max_samples=1,config=cfg,reuse_existing=True)


@pytest.mark.asyncio
@pytest.mark.parametrize('spec',BENCHMARK_TASKS,ids=lambda t:t.key)
async def test_each_paper_task_uses_shared_scorer_and_one_answer_free_request(spec):
    s=sample(spec.dataset_name)
    llm=SimpleNamespace(complete=AsyncMock(return_value=SimpleNamespace(text='A')))
    import asyncio
    rows=await baseline._eval_one_sample(spec.dataset_name,0,s,[spec.task_name],llm,asyncio.Semaphore(1))
    assert len(rows)==1
    assert llm.complete.await_count==1
    row=rows[0]
    # The scorer sees supervision only after the same visible prompt was sent.
    from timeclaw.benchmark.train_loop import _extract_ground_truth
    assert row['metrics']==score_mtbench_prediction(row['prediction'],_extract_ground_truth(s,spec.task_name),spec.task_name)
    changed=replace(s,output_window=np.full(24,99999.),answer='D',labels={**s.labels,'out_macd':[99999.]*3,
        'trend_label':'<-4%','correlation':'Strong Negative Correlation'})
    await baseline._eval_one_sample(spec.dataset_name,0,changed,[spec.task_name],llm,asyncio.Semaphore(1))
    assert llm.complete.call_args_list[0]==llm.complete.call_args_list[1]

@pytest.mark.asyncio
async def test_baseline_receipt_reuse_tamper_and_model_change(tmp_path,monkeypatch):
    cfg=TimeClawConfig(workspace_dir=str(tmp_path))
    llm=SimpleNamespace(complete=AsyncMock(return_value=SimpleNamespace(text='A')))
    router=SimpleNamespace(get_backend=lambda role:llm,shutdown=AsyncMock())
    monkeypatch.setattr(baseline,'ModelRouter',lambda cfg:router)
    monkeypatch.setattr(baseline,'load_dataset',lambda *a,**kw:[sample('weather_qa_short')])
    first=await baseline.run_baseline(dataset_name='weather_qa_short',task_filter='mcqa',config=cfg)
    assert first['weather_qa_short']['total']==1 and first['weather_qa_short']['failed']==0
    assert llm.complete.await_count==1
    router.shutdown.assert_awaited_once()
    second=await baseline.run_baseline(dataset_name='weather_qa_short',task_filter='mcqa',config=cfg)
    assert second['weather_qa_short']['reused'] and llm.complete.await_count==1
    cfg.llm.reasoning_effort='high'
    with pytest.raises(ValueError,match='receipt mismatch'):
        await baseline.run_baseline(dataset_name='weather_qa_short',task_filter='mcqa',config=cfg)
    cfg.llm.reasoning_effort='medium'
    path=tmp_path/'test_results/baseline_weather_qa_short.json'
    path.write_text(path.read_text()+' ')
    with pytest.raises(ValueError,match='receipt mismatch'):
        await baseline.run_baseline(dataset_name='weather_qa_short',task_filter='mcqa',config=cfg)
    assert llm.complete.await_count==1


@pytest.mark.asyncio
async def test_failed_rows_and_unknown_usage_are_retained_and_resources_close(tmp_path,monkeypatch):
    cfg=TimeClawConfig(workspace_dir=str(tmp_path))
    llm=SimpleNamespace(complete=AsyncMock(side_effect=RuntimeError('provider failed')))
    router=SimpleNamespace(get_backend=lambda role:llm,shutdown=AsyncMock())
    monkeypatch.setattr(baseline,'ModelRouter',lambda cfg:router)
    monkeypatch.setattr(baseline,'load_dataset',lambda *a,**kw:[sample('weather_qa_short')])
    result=await baseline.run_baseline(dataset_name='weather_qa_short',task_filter='mcqa',config=cfg)
    assert result['weather_qa_short']['total']==result['weather_qa_short']['failed']==1
    rows=json.loads((tmp_path/'test_results/baseline_weather_qa_short.json').read_text())
    assert rows[0]['status']=='failed' and rows[0]['error']=='provider failed'
    assert 'accuracy' not in rows[0]['metrics']
    router.shutdown.assert_awaited_once()
    with pytest.raises(ValueError,match='unsuccessful rows'):
        await baseline.run_baseline(dataset_name='weather_qa_short',task_filter='mcqa',config=cfg)


@pytest.mark.asyncio
async def test_overlapping_baselines_cannot_overwrite_one_output_directory(tmp_path, monkeypatch):
    """The second launch must fail before another backend can send a request."""
    import asyncio
    cfg = TimeClawConfig(workspace_dir=str(tmp_path / 'memory'))
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []
    async def complete(messages):
        calls.append(messages)
        if len(calls) == 1:
            entered.set()
            await release.wait()
        return SimpleNamespace(text='A')
    def router_factory(config):
        return SimpleNamespace(get_backend=lambda role: SimpleNamespace(complete=complete),
                               shutdown=AsyncMock())
    monkeypatch.setattr(baseline, 'ModelRouter', router_factory)
    monkeypatch.setattr(baseline, 'load_dataset', lambda *a, **kw: [sample('weather_qa_short')])
    out = tmp_path / 'results'
    kwargs = dict(dataset_name='weather_qa_short', task_filter='mcqa', config=cfg, output_dir=out)
    first = asyncio.create_task(baseline.run_baseline(**kwargs))
    try:
        await asyncio.wait_for(entered.wait(), timeout=2)
        with pytest.raises(RuntimeError, match='active writer'):
            await baseline.run_baseline(**kwargs)
        assert len(calls) == 1
    finally:
        release.set()
        await first
    # Ownership must be released; verified reuse makes no new request.
    reused = await baseline.run_baseline(**kwargs)
    assert reused['weather_qa_short']['reused'] and len(calls) == 1


def test_baseline_output_owner_excludes_another_process_and_releases_on_failure(tmp_path):
    import subprocess
    import sys
    code = (
        'from pathlib import Path; import sys; '
        'from timeclaw.benchmark.baseline_eval import _baseline_output_lease; '
        'owner = _baseline_output_lease(Path(sys.argv[1])); '
        'owner.__enter__(); owner.__exit__(None, None, None)'
    )
    with pytest.raises(ValueError, match='simulated job failure'):
        with baseline._baseline_output_lease(tmp_path):
            child = subprocess.run([sys.executable, '-c', code, str(tmp_path)],
                                   capture_output=True, text=True, timeout=15)
            assert child.returncode != 0 and 'active writer' in child.stderr
            raise ValueError('simulated job failure')
    released = subprocess.run([sys.executable, '-c', code, str(tmp_path)],
                              capture_output=True, text=True, timeout=15)
    assert released.returncode == 0, released.stderr
