"""The release benchmark follows the explicit 18-task software specification."""
import importlib.util
from pathlib import Path
import sys
import pytest
from timeclaw.benchmark.mtbench_adapter import expanded_benchmark_task_specs

EXPECTED = {
    (f'finance_aligned_{length}', task)
    for length in ['short','long'] for task in ['forecast','indicator_macd','trend']
} | {
    (f'finance_qa_{length}', task)
    for length in ['short','long'] for task in ['mcqa','correlation']
} | {
    (f'weather_aligned_{length}', task)
    for length in ['short','long'] for task in ['forecast','trend','trend_past']
} | {(f'weather_qa_{length}','mcqa') for length in ['short','long']}


def inference_module():
    root=Path(__file__).resolve().parents[2]
    sys.path.insert(0,str(root/'scripts'))
    spec=importlib.util.spec_from_file_location('paper18_inference',root/'scripts/run_timeclaw_mtbench_inference.py')
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    return module


def test_training_registry_exactly_matches_paper18():
    assert len(EXPECTED)==18
    assert set(expanded_benchmark_task_specs("timeclaw18"))==EXPECTED


def test_inference_registry_exactly_matches_paper18():
    specs=inference_module()._task_specs('timeclaw18',None)
    assert len(specs)==len({s.key for s in specs})==18
    assert {(s.dataset_name,s.task_name) for s in specs}==EXPECTED


@pytest.mark.parametrize('retired',['official22','paper17'])
def test_other_task_sets_cannot_silently_change_the_experiment(retired):
    with pytest.raises(ValueError,match='timeclaw18'):
        inference_module()._task_specs(retired,None)


@pytest.mark.asyncio
@pytest.mark.parametrize('dataset,task',[
    ('weather_aligned_short','indicator'),('weather_aligned_long','indicator'),
    ('finance_aligned_short','indicator_upper_bb'),('finance_aligned_long','indicator_upper_bb')])
async def test_public_sample_inference_cannot_bypass_the_paper_task_roster(dataset,task):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    import numpy as np
    from timeclaw.benchmark.mtbench_adapter import MTBenchSample
    from timeclaw.pipeline.runner import PipelineRunner
    sample=MTBenchSample(file='example.json',dataset=dataset,domain=dataset.split('_')[0],task_types=[task],
        input_window=np.arange(168.),output_window=np.arange(24.),timestamps=list(range(168)),
        text='Context',start_year=2024,forecast_horizon=24,
        labels={'out_upper_bb':[1.]*24},metadata={'output_timestamps':list(range(168,192))})
    from timeclaw.core.config import TimeClawConfig
    cfg=TimeClawConfig()
    cfg.benchmark.task_set="timeclaw18"
    runner=SimpleNamespace(run=AsyncMock(),config=cfg)
    with pytest.raises(ValueError,match='timeclaw18'):
        await PipelineRunner.run_inference(runner,sample,task)
    runner.run.assert_not_called()


def test_all_26_primary_paper_metrics_are_available_from_the_shared_scorer():
    from timeclaw.benchmark.task_registry import EXTENDED_TASKS as BENCHMARK_TASKS
    from timeclaw.benchmark.train_loop import _extract_ground_truth
    from timeclaw.benchmark.mtbench_support.task_scoring import score_mtbench_prediction
    from tests.test_benchmark.test_direct_baseline import sample
    mapping={('finance','forecast'):['mae','mape'],('finance','indicator_macd'):['macd_mse'],
      ('finance','trend'):['accuracy','accuracy_3way'],('finance','correlation'):['exact_accuracy','brief_accuracy'],
      ('finance','mcqa'):['accuracy'],('weather','forecast'):['mse','mae'],
      ('weather','trend'):['accuracy'],('weather','trend_past'):['accuracy'],('weather','mcqa'):['accuracy']}
    count=0
    for spec in BENCHMARK_TASKS:
        fixture=sample(spec.dataset_name)
        truth=_extract_ground_truth(fixture,spec.task_name)
        metrics=score_mtbench_prediction(truth,truth,spec.task_name)
        expected=mapping[fixture.domain,spec.task_name]
        assert all(key in metrics for key in expected)
        count+=len(expected)
    assert count==26
