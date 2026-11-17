"""The paper table is an independent specification, not a registry count."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock
import pytest
from timeclaw.benchmark.task_registry import benchmark_tasks, primary_metrics, task_registry_digest
from timeclaw.benchmark.mtbench_adapter import expanded_benchmark_task_specs
from timeclaw.benchmark import baseline_eval
from timeclaw.benchmark.train_loop import _extract_ground_truth
from timeclaw.benchmark.mtbench_support.task_scoring import score_mtbench_prediction
from timeclaw.benchmark.mtbench_support.official_metric_policy import summarize_official_inference_rows
from tests.test_benchmark.test_direct_baseline import sample


def test_paper_inventory_and_all_thirty_metric_cells():
    expected = {(f'finance_aligned_{n}', t) for n in ('short', 'long') for t in ('forecast', 'indicator_macd', 'trend')}
    expected |= {(f'finance_qa_{n}', t) for n in ('short', 'long') for t in ('mcqa', 'correlation')}
    expected |= {(f'weather_aligned_{n}', 'forecast') for n in ('short', 'long')}
    expected |= {(f'weather_qa_{n}', 'mcqa') for n in ('short', 'long')}
    expected |= {('weather_aligned_short', t) for t in ('indicator', 'trend', 'trend_past')}
    assert len(expected) == 17
    assert set(expanded_benchmark_task_specs()) == expected
    count = 0
    for spec in benchmark_tasks():
        s = sample(spec.dataset_name)
        truth = _extract_ground_truth(s, spec.task_name)
        metrics = score_mtbench_prediction(truth, truth, spec.task_name)
        assert set(primary_metrics(spec)) <= metrics.keys()
        count += len(primary_metrics(spec))
    assert count == 30
    assert task_registry_digest() != task_registry_digest('timeclaw18')
    assert len(benchmark_tasks('timeclaw18')) == 18


@pytest.mark.asyncio
async def test_weather_indicator_baseline_parsing_and_six_metric_aggregation():
    s = sample('weather_aligned_short')
    truth = _extract_ground_truth(s, 'indicator')
    p = {k: v + delta for (k, v), delta in zip(truth.items(), (2, -3, 5))}
    text = f"Highest temperature: {p['max']}, Lowest temperature: {p['min']}, Temperature difference: {p['diff']}"
    llm = SimpleNamespace(complete=AsyncMock(return_value=SimpleNamespace(text=text)))
    rows = await baseline_eval._eval_one_sample(s.dataset, 0, s, ['indicator'], llm, asyncio.Semaphore(1))
    assert rows[0]['status'] == 'completed'
    assert llm.complete.await_count == 1
    metrics = rows[0]['metrics']
    assert [metrics[k] for k in ('max_mse','min_mse','diff_mse')] == [4,9,25]
    assert [metrics[k] for k in ('max_mae','min_mae','diff_mae')] == [2,3,5]
    summary = summarize_official_inference_rows(rows)
    assert all(f'avg_{k}' in str(summary) for k in ('max_mse','min_mse','diff_mse','max_mae','min_mae','diff_mae'))


@pytest.mark.parametrize('training_set,inference_set',[('paper30_original','timeclaw18'),('timeclaw18','paper30_original')])
def test_sealed_memory_cannot_cross_task_presets(tmp_path,training_set,inference_set):
    from tests.test_benchmark.test_source_lineage import make,load,seal
    from timeclaw.benchmark.source_admission import assert_inference_lineage
    from timeclaw.benchmark.mtbench_adapter import load_dataset
    from timeclaw.data_integrity import DataIntegrityError
    cfg,_,_ = make(tmp_path)
    cfg.benchmark.task_set = training_set
    seal(cfg,load(cfg))
    cfg.benchmark.task_set = inference_set
    with pytest.raises(DataIntegrityError,match='selected task set'):
        assert_inference_lineage(cfg,{'weather_aligned_short':load_dataset('weather_aligned_short',config=cfg)},disable_memory=False)
