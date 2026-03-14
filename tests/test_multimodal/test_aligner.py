"""Tests for ModalityAligner."""

import numpy as np
from timeclaw.multimodal.aligner import ModalityAligner


def test_align_text_nearest():
    aligner = ModalityAligner()
    ts = np.arange(100, dtype=float)
    text_data = [
        {"timestamp": 10, "text": "doc1"},
        {"timestamp": 50, "text": "doc2"},
    ]
    result = aligner.align(ts, text_data=text_data)
    assert result["n_text_matched"] == 2
    assert result["text_aligned"][0]["ts_idx"] == 10


def test_align_with_tolerance():
    aligner = ModalityAligner(tolerance=5)
    ts = np.arange(100, dtype=float)
    text_data = [
        {"timestamp": 10, "text": "close"},
        {"timestamp": 200, "text": "far_away"},
    ]
    result = aligner.align(ts, text_data=text_data)
    assert result["n_text_matched"] == 1


def test_align_images():
    aligner = ModalityAligner()
    ts = np.arange(50, dtype=float)
    images = [{"timestamp": 25, "path": "img.jpg"}]
    result = aligner.align(ts, images=images)
    assert result["n_image_matched"] == 1


def test_align_empty():
    aligner = ModalityAligner()
    result = aligner.align()
    assert result["n_text_matched"] == 0


def test_timeline_build():
    aligner = ModalityAligner()
    ts = np.arange(10, dtype=float)
    text_data = [{"timestamp": 3, "text": "note"}]
    result = aligner.align(ts, text_data=text_data)
    assert len(result["timeline"]) == 10
    assert "text" in result["timeline"][3]["modalities"]
