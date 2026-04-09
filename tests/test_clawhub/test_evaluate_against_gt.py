import pytest

from timeclaw.clawhub.profiling.autocorrelation import AutocorrelationTool
from timeclaw.clawhub.validation.evaluate_against_gt import EvaluateAgainstGTTool
from timeclaw.clawhub.validation.batch_evaluate_against_gt import BatchEvaluateAgainstGTTool


def test_label_metrics_accepts_trend_dict_prediction():
    metrics = EvaluateAgainstGTTool._label_metrics(
        {"trend": "decreasing"},
        "decreasing",
    )
    assert metrics["accuracy"] == 1.0
    assert metrics["correct"] is True


def test_label_metrics_accepts_json_string_prediction():
    metrics = EvaluateAgainstGTTool._label_metrics(
        '{"trend": "increasing"}',
        "increasing",
    )
    assert metrics["accuracy"] == 1.0
    assert metrics["correct"] is True


def test_forecast_metrics_returns_type_error_for_non_numeric_prediction():
    metrics = EvaluateAgainstGTTool._forecast_metrics(
        "Unable to compute the 24-value ARIMA forecast array",
        [1.0, 2.0, 3.0],
    )
    assert metrics == {"type_error": 1.0}


def test_forecast_metrics_accepts_answer_wrapper():
    metrics = EvaluateAgainstGTTool._forecast_metrics(
        {"answer": [1.0, 2.0, 3.0]},
        [1.0, 2.0, 3.0],
    )
    assert metrics["mae"] == 0.0


def test_indicator_metrics_requires_exact_indicator_keys():
    metrics = EvaluateAgainstGTTool._indicator_metrics(
        {"macd_forecast": [1.0, 2.0, 3.0]},
        {"macd": [1.0, 2.0, 4.0]},
    )

    assert metrics == {}


def test_indicator_metrics_accepts_answer_wrapper():
    metrics = EvaluateAgainstGTTool._indicator_metrics(
        {"answer": {"macd": [1.0, 2.0, 3.0]}},
        {"macd": [1.0, 2.0, 4.0]},
    )

    assert metrics["macd_mae"] == 0.333


@pytest.mark.asyncio
async def test_evaluate_batch_against_gt_returns_per_approach_metrics():
    tool = BatchEvaluateAgainstGTTool()
    result = await tool._run(
        predictions=[
            {"name": "a", "prediction": [1.0, 2.0]},
            {"name": "b", "prediction": [2.0, 3.0]},
        ],
        task_type="forecast",
        _ground_truth=[1.0, 2.0],
    )
    assert len(result["results"]) == 2
    assert result["results"][0]["name"] == "a"
    assert result["results"][0]["metrics"]["mae"] == 0.0


@pytest.mark.asyncio
async def test_evaluate_batch_against_gt_accepts_raw_prediction_payloads():
    tool = BatchEvaluateAgainstGTTool()
    result = await tool._run(
        predictions=[
            [1.0, 2.0],
            [2.0, 3.0],
        ],
        task_type="forecast",
        _ground_truth=[1.0, 2.0],
    )
    assert len(result["results"]) == 2
    assert result["results"][0]["name"] == "approach_0"
    assert result["results"][0]["metrics"]["mae"] == 0.0


@pytest.mark.asyncio
async def test_autocorrelation_caps_pacf_lag_for_short_series():
    tool = AutocorrelationTool()
    result = await tool._run([1.0, 2.0, 3.0, 4.0, 5.0], max_lag=20, mode="both")

    assert result["acf_lags"] == [0, 1]
    assert result["pacf_lags"] == [0, 1]
