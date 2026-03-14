"""Tests for ModalityConfidenceEstimator."""

import numpy as np

from timeclaw.multimodal.confidence import ModalityConfidenceEstimator, estimate_all_confidences


def test_empty_log():
    est = ModalityConfidenceEstimator()
    assert est.estimate([], None) == 0.1


def test_perfect_execution():
    log = [
        {"tool": "basic_stats", "success": True, "result_summary": "mean=25.3, std=2.1"},
        {"tool": "arima_forecast", "success": True, "result_summary": "forecast=[24, 25, 26]"},
    ]
    conf = ModalityConfidenceEstimator().estimate(log, [24.0, 25.0, 26.0])
    assert conf > 0.7


def test_failed_execution():
    log = [
        {"tool": "basic_stats", "success": False, "result_summary": ""},
        {"tool": "arima_forecast", "success": False, "result_summary": ""},
    ]
    conf = ModalityConfidenceEstimator().estimate(log, None)
    assert conf < 0.3


def test_prediction_quality_numeric():
    est = ModalityConfidenceEstimator()
    assert est._prediction_quality(25.0) == 0.8
    assert est._prediction_quality(np.nan) == 0.2
    assert est._prediction_quality(None) == 0.0


def test_prediction_quality_list():
    est = ModalityConfidenceEstimator()
    assert est._prediction_quality([1, 2, 3]) > 0.5
    assert est._prediction_quality([]) == 0.1


def test_estimate_all_confidences():
    results = {
        "timeseries": {
            "prediction": [24, 25],
            "execution_log": [
                {"tool": "stats", "success": True, "result_summary": "ok" * 20},
            ],
        },
        "text": {
            "prediction": "The market will rise",
            "execution_log": [
                {"tool": "sentiment", "success": True, "result_summary": "positive"},
                {"tool": "keyword", "success": False, "result_summary": ""},
            ],
        },
    }
    confs = estimate_all_confidences(results)
    assert "timeseries" in confs
    assert "text" in confs
    assert 0 < confs["timeseries"] <= 1
    assert 0 < confs["text"] <= 1
