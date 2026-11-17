"""Compare actual pinned upstream arithmetic, not a copy of our implementation."""
import importlib.util
import os
from pathlib import Path
import numpy as np
import pytest
from timeclaw.benchmark.mtbench_support.task_scoring import score_mtbench_prediction


def upstream_utils():
    root=Path(os.environ.get('TIMECLAW_MTBENCH_ROOT','mtbench_official'))
    spec=importlib.util.spec_from_file_location('metric_parity_upstream',root/'evaluation/utils.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('truth,prediction',[
    ([10.,20.,30.],[11.,18.,28.]),
    ([0.,2.,4.],[100.,1.,3.]),
    ([-2.,0.,1e-5],[-1.,99.,2e-5]),
    ([1.,2.,3.,4.],[2.]),
    ([1.,2.,3.,4.],[1.,3.]),
])
def test_forecast_metrics_match_official_nonzero_mape_and_interpolation(truth,prediction):
    aligned=np.interp(np.linspace(0,1,len(truth)),np.linspace(0,1,len(prediction)),prediction)
    expected=upstream_utils().calculate_mape(truth,aligned)
    metrics=score_mtbench_prediction(prediction,truth,'forecast')
    assert metrics['mape']==pytest.approx(expected)
    assert metrics['mse']==pytest.approx(float(np.mean((np.asarray(truth)-aligned)**2)))
    assert metrics['mae']==pytest.approx(float(np.mean(np.abs(np.asarray(truth)-aligned))))


def test_all_zero_truth_has_undefined_percentage_error_instead_of_a_fabricated_number():
    metrics=score_mtbench_prediction([1.,2.],[0.,0.],'forecast')
    assert metrics['mape'] is None
    assert metrics['mape_defined'] is False
    assert metrics['mse']==2.5


def test_threshold_decision_uses_unrounded_error():
    from timeclaw.benchmark.mtbench_support.official_metric_policy import apply_official_metric_policy_to_row
    metrics=score_mtbench_prediction([1.+np.sqrt(100.0001)]*2,[1.,1.],'forecast')
    row=apply_official_metric_policy_to_row({'dataset':'finance_aligned_short','task':'forecast','status':'completed','metrics':metrics})
    assert row['summary_included'] is False


def test_weather_long_label_explicitly_follows_first_day_prompt_not_full_output_script():
    from timeclaw.benchmark.ground_truth import _weather_trend_3way_label
    from types import SimpleNamespace
    past=[20.]*168
    future=[21.]*24+[18.]*48
    fixture=SimpleNamespace(input_window=np.asarray(past),output_window=np.asarray(future))
    assert _weather_trend_3way_label(fixture)=='increasing'
    assert upstream_utils().compute_temperature_trend(past,future)=='decreasing'
