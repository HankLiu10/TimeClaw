"""Tests for MTBench metrics."""

import numpy as np
from timeclaw.benchmark.metrics import (
    mse, mae, rmse, mape, smape,
    accuracy, precision_recall_f1, auc_roc,
    anomaly_f1,
    bleu_score, rouge_l,
    compute_forecast_metrics,
    compute_classification_metrics,
    compute_anomaly_metrics,
    compute_qa_metrics,
)


def test_mse_perfect():
    y = np.array([1.0, 2.0, 3.0])
    assert mse(y, y) == 0.0


def test_mae():
    assert mae(np.array([1.0, 2.0]), np.array([2.0, 3.0])) == 1.0


def test_rmse():
    assert rmse(np.array([1.0]), np.array([1.0])) == 0.0


def test_mape_smape():
    y_true = np.array([100.0, 200.0])
    y_pred = np.array([110.0, 180.0])
    assert 0 < mape(y_true, y_pred) < 100
    assert 0 < smape(y_true, y_pred) < 100


def test_accuracy():
    assert accuracy(np.array([0, 1, 1, 0]), np.array([0, 1, 0, 0])) == 0.75


def test_prf():
    result = precision_recall_f1(np.array([0, 1, 1]), np.array([0, 1, 0]))
    assert "f1" in result


def test_auc():
    result = auc_roc(np.array([0, 0, 1, 1]), np.array([0.1, 0.4, 0.35, 0.8]))
    assert 0 <= result <= 1


def test_anomaly_f1_point():
    result = anomaly_f1(np.array([0, 1, 1, 0]), np.array([0, 1, 0, 0]))
    assert 0 <= result["f1"] <= 1


def test_anomaly_f1_range():
    y_true = np.array([0, 0, 1, 1, 1, 0])
    y_pred = np.array([0, 0, 0, 1, 0, 0])
    result = anomaly_f1(y_true, y_pred, mode="range")
    assert result["f1"] >= 0


def test_bleu():
    assert bleu_score("the cat sat", "the cat sat") > 0.5


def test_rouge_l():
    result = rouge_l("the cat sat on the mat", "the cat on the mat")
    assert result["f1"] > 0.5


def test_compute_forecast_metrics():
    y_true = np.random.randn(50)
    y_pred = y_true + np.random.randn(50) * 0.1
    metrics = compute_forecast_metrics(y_true, y_pred, y_train=np.random.randn(200))
    assert "mse" in metrics and "mase" in metrics


def test_compute_classification_metrics():
    metrics = compute_classification_metrics(np.array([0, 1, 2]), np.array([0, 1, 1]))
    assert "accuracy" in metrics


def test_compute_anomaly_metrics():
    metrics = compute_anomaly_metrics(np.array([0, 1, 1, 0]), np.array([0, 1, 0, 0]))
    assert "point_f1" in metrics and "range_f1" in metrics


def test_compute_qa_metrics():
    metrics = compute_qa_metrics("hello world", "hello world")
    assert "bleu" in metrics and "rouge_l_f1" in metrics
