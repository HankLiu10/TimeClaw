"""Tests for ModalityFusion."""

import numpy as np
import pytest
from timeclaw.multimodal.fusion import ModalityFusion


def test_fixed_fusion_numeric():
    """'late' maps to 'fixed' strategy internally."""
    fusion = ModalityFusion(strategy="late")
    evidence = {
        "timeseries": {"prediction": 10.0, "confidence": 0.8},
        "text": {"prediction": 12.0, "confidence": 0.6},
    }
    result = fusion.fuse(evidence)
    assert result["strategy"] == "fixed"
    assert result["confidence"] > 0


def test_early_fusion():
    fusion = ModalityFusion(strategy="early")
    evidence = {
        "timeseries": np.array([1.0, 2.0, 3.0]),
        "text": {"features": [4.0, 5.0]},
    }
    result = fusion.fuse(evidence)
    assert result["strategy"] == "early"
    assert result["dim"] == 5


def test_fixed_fusion_majority_vote():
    fusion = ModalityFusion(strategy="fixed")
    evidence = {
        "timeseries": {"prediction": "up", "confidence": 0.7},
        "text": {"prediction": "up", "confidence": 0.8},
        "image": {"prediction": "down", "confidence": 0.4},
    }
    result = fusion.fuse(evidence)
    assert result["prediction"] == "up"


def test_task_adaptive_fusion():
    fusion = ModalityFusion(strategy="task_adaptive")
    evidence = {
        "timeseries": {"prediction": 10.0, "confidence": 0.8},
        "text": {"prediction": 12.0, "confidence": 0.6},
    }
    result = fusion.fuse(evidence, task_type="qa")
    assert result["strategy"] == "task_adaptive"
    assert "weights_used" in result


def test_confidence_weighted_fusion():
    fusion = ModalityFusion(strategy="confidence_weighted")
    evidence = {
        "timeseries": {"prediction": 10.0, "confidence": 0.9},
        "text": {"prediction": 12.0, "confidence": 0.3},
    }
    confidences = {"timeseries": 0.9, "text": 0.3}
    result = fusion.fuse(evidence, task_type="forecast", modality_confidences=confidences)
    assert result["strategy"] == "confidence_weighted"
    assert "weights_used" in result


def test_invalid_strategy():
    with pytest.raises(ValueError):
        ModalityFusion(strategy="invalid")
