"""Unified evaluation metrics for time-series, classification, anomaly, and QA tasks."""

from __future__ import annotations

from typing import Any

import numpy as np


# ── Forecasting / Imputation Metrics ──────────────────────────────

def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Squared Error."""
    return float(np.mean((y_true - y_pred) ** 2))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(mse(y_true, y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
    """Mean Absolute Percentage Error."""
    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon))) * 100)


def smape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
    """Symmetric Mean Absolute Percentage Error."""
    denom = np.abs(y_true) + np.abs(y_pred) + epsilon
    return float(np.mean(2.0 * np.abs(y_true - y_pred) / denom) * 100)


def mase(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray,
    seasonality: int = 1,
) -> float:
    """Mean Absolute Scaled Error."""
    naive_errors = np.abs(np.diff(y_train, n=seasonality))
    scale = np.mean(naive_errors) if len(naive_errors) > 0 else 1.0
    if scale < 1e-10:
        scale = 1.0
    return float(np.mean(np.abs(y_true - y_pred)) / scale)


def crps_empirical(y_true: np.ndarray, samples: np.ndarray) -> float:
    """Continuous Ranked Probability Score (empirical, from samples).

    Args:
        y_true: Ground truth, shape (T,).
        samples: Forecast samples, shape (num_samples, T).
    """
    n_samples = samples.shape[0]
    abs_diff = np.mean(np.abs(samples - y_true[np.newaxis, :]), axis=0)

    pairwise_sum = 0.0
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            pairwise_sum += np.mean(np.abs(samples[i] - samples[j]))
    pairwise_term = pairwise_sum / (n_samples * (n_samples - 1) / 2) if n_samples > 1 else 0.0

    return float(np.mean(abs_diff) - 0.5 * pairwise_term)


# ── Classification Metrics ────────────────────────────────────────

def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Classification accuracy."""
    return float(np.mean(y_true == y_pred))


def precision_recall_f1(
    y_true: np.ndarray, y_pred: np.ndarray, average: str = "macro"
) -> dict[str, float]:
    """Compute precision, recall, and F1 score."""
    labels = np.unique(np.concatenate([y_true, y_pred]))
    precisions, recalls, f1s = [], [], []

    for label in labels:
        tp = np.sum((y_pred == label) & (y_true == label))
        fp = np.sum((y_pred == label) & (y_true != label))
        fn = np.sum((y_pred != label) & (y_true == label))

        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

        precisions.append(p)
        recalls.append(r)
        f1s.append(f)

    if average == "macro":
        return {
            "precision": float(np.mean(precisions)),
            "recall": float(np.mean(recalls)),
            "f1": float(np.mean(f1s)),
        }
    elif average == "micro":
        total_tp = np.sum(y_true == y_pred)
        total = len(y_true)
        micro_p = micro_r = micro_f = total_tp / total if total > 0 else 0.0
        return {"precision": float(micro_p), "recall": float(micro_r), "f1": float(micro_f)}
    return {"precision": float(np.mean(precisions)), "recall": float(np.mean(recalls)), "f1": float(np.mean(f1s))}


def auc_roc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Area Under ROC Curve (binary classification)."""
    sorted_indices = np.argsort(-y_scores)
    y_sorted = y_true[sorted_indices]

    tpr_list = [0.0]
    fpr_list = [0.0]
    tp = 0
    fp = 0
    total_pos = np.sum(y_true == 1)
    total_neg = np.sum(y_true == 0)

    if total_pos == 0 or total_neg == 0:
        return 0.5

    for y in y_sorted:
        if y == 1:
            tp += 1
        else:
            fp += 1
        tpr_list.append(tp / total_pos)
        fpr_list.append(fp / total_neg)

    auc_val = 0.0
    for i in range(1, len(fpr_list)):
        auc_val += (fpr_list[i] - fpr_list[i - 1]) * (tpr_list[i] + tpr_list[i - 1]) / 2

    return float(auc_val)


# ── Anomaly Detection Metrics ────────────────────────────────────

def anomaly_f1(
    y_true: np.ndarray, y_pred: np.ndarray, mode: str = "point"
) -> dict[str, float]:
    """Anomaly detection F1 (point-wise or range-based).

    Args:
        y_true: Binary ground truth (1=anomaly, 0=normal).
        y_pred: Binary predictions.
        mode: 'point' for point-wise, 'range' for range-adjusted.
    """
    if mode == "range":
        y_pred = _range_adjust(y_true, y_pred)

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1, "mode": mode}


def _range_adjust(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Range-based adjustment: if any point in an anomaly segment is detected, credit the whole segment."""
    adjusted = y_pred.copy()
    in_segment = False
    seg_start = 0

    for i in range(len(y_true)):
        if y_true[i] == 1 and not in_segment:
            seg_start = i
            in_segment = True
        elif y_true[i] == 0 and in_segment:
            if np.any(y_pred[seg_start:i] == 1):
                adjusted[seg_start:i] = 1
            in_segment = False

    if in_segment and np.any(y_pred[seg_start:] == 1):
        adjusted[seg_start:] = 1

    return adjusted


# ── QA / Text Generation Metrics ─────────────────────────────────

def bleu_score(reference: str, hypothesis: str, max_n: int = 4) -> float:
    """Simplified BLEU score computation.

    Automatically adjusts max_n to not exceed the length of the shorter sequence.
    """
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()

    if len(hyp_tokens) == 0:
        return 0.0

    effective_max_n = min(max_n, len(ref_tokens), len(hyp_tokens))
    if effective_max_n == 0:
        return 0.0

    brevity_penalty = min(1.0, np.exp(1 - len(ref_tokens) / max(len(hyp_tokens), 1)))

    precisions = []
    for n in range(1, effective_max_n + 1):
        ref_ngrams = _get_ngrams(ref_tokens, n)
        hyp_ngrams = _get_ngrams(hyp_tokens, n)

        if not hyp_ngrams:
            continue

        matches = sum(min(hyp_ngrams.get(ng, 0), ref_ngrams.get(ng, 0)) for ng in hyp_ngrams)
        total = sum(hyp_ngrams.values())
        precisions.append(matches / total if total > 0 else 0.0)

    if not precisions or any(p == 0 for p in precisions):
        return 0.0

    log_avg = sum(np.log(p + 1e-10) for p in precisions) / len(precisions)
    return float(brevity_penalty * np.exp(log_avg))


def rouge_l(reference: str, hypothesis: str) -> dict[str, float]:
    """ROUGE-L score based on longest common subsequence."""
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()

    lcs_len = _lcs_length(ref_tokens, hyp_tokens)

    precision = lcs_len / len(hyp_tokens) if len(hyp_tokens) > 0 else 0.0
    recall = lcs_len / len(ref_tokens) if len(ref_tokens) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}


def _get_ngrams(tokens: list[str], n: int) -> dict[tuple, int]:
    ngrams: dict[tuple, int] = {}
    for i in range(len(tokens) - n + 1):
        ng = tuple(tokens[i : i + n])
        ngrams[ng] = ngrams.get(ng, 0) + 1
    return ngrams


def _lcs_length(a: list[str], b: list[str]) -> int:
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


# ── Metric Suite ──────────────────────────────────────────────────

def compute_forecast_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray | None = None,
    samples: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute all forecasting metrics at once."""
    results = {
        "mse": mse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "smape": smape(y_true, y_pred),
    }
    if y_train is not None:
        results["mase"] = mase(y_true, y_pred, y_train)
    if samples is not None and samples.ndim == 2:
        results["crps"] = crps_empirical(y_true, samples)
    return results


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_scores: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute all classification metrics at once."""
    results = {"accuracy": accuracy(y_true, y_pred)}
    prf = precision_recall_f1(y_true, y_pred)
    results.update(prf)
    if y_scores is not None and len(np.unique(y_true)) == 2:
        results["auc"] = auc_roc(y_true, y_scores)
    return results


def compute_anomaly_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict[str, Any]:
    """Compute anomaly detection metrics (point-wise and range-based)."""
    point = anomaly_f1(y_true, y_pred, mode="point")
    range_based = anomaly_f1(y_true, y_pred, mode="range")
    return {
        "point_f1": point["f1"],
        "point_precision": point["precision"],
        "point_recall": point["recall"],
        "range_f1": range_based["f1"],
        "range_precision": range_based["precision"],
        "range_recall": range_based["recall"],
    }


def compute_qa_metrics(reference: str, hypothesis: str) -> dict[str, float]:
    """Compute QA/text generation metrics."""
    return {
        "bleu": bleu_score(reference, hypothesis),
        "rouge_l_f1": rouge_l(reference, hypothesis)["f1"],
        "rouge_l_precision": rouge_l(reference, hypothesis)["precision"],
        "rouge_l_recall": rouge_l(reference, hypothesis)["recall"],
    }
