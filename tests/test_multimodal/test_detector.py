"""Tests for ModalityDetector."""

import numpy as np
import pytest
from timeclaw.multimodal.detector import ModalityDetector


@pytest.fixture
def detector():
    return ModalityDetector()


def test_detect_timeseries_only(detector):
    result = detector.detect(ts_data=np.random.randn(100))
    assert "timeseries" in result["modalities"]
    assert result["n_modalities"] == 1
    assert not result["is_multimodal"]


def test_detect_text_only(detector):
    result = detector.detect(text_data="Hello world")
    assert "text" in result["modalities"]


def test_detect_text_list(detector):
    result = detector.detect(text_data=[{"text": "doc1"}, {"text": "doc2"}])
    assert "text" in result["modalities"]
    assert result["details"]["text"]["n_documents"] == 2


def test_detect_images(detector):
    result = detector.detect(images=[{"path": "test.jpg", "timestamp": 0}])
    assert "image" in result["modalities"]


def test_detect_multimodal(detector):
    result = detector.detect(
        ts_data=np.random.randn(100),
        text_data=[{"text": "note", "timestamp": 50}],
        images=[{"path": "img.png", "timestamp": 50}],
    )
    assert result["is_multimodal"]
    assert result["n_modalities"] == 3


def test_detect_none_inputs(detector):
    result = detector.detect()
    assert result["n_modalities"] == 0
