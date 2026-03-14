"""Tests for dataset loaders."""

import pytest
from timeclaw.benchmark.datasets import DatasetLoader


@pytest.fixture
def loader():
    return DatasetLoader(cache_dir=".cache/test_datasets")


def test_load_synthetic_sine(loader):
    ds = loader.load("synthetic:sine", length=500)
    assert ds["source"] == "synthetic"
    assert len(ds["train"]) + len(ds["test"]) == 500


def test_load_synthetic_random_walk(loader):
    ds = loader.load("synthetic:random_walk", length=300)
    assert len(ds["data"]) == 300


def test_load_synthetic_anomaly(loader):
    ds = loader.load("synthetic:anomaly", length=200)
    assert "labels" in ds
    assert ds["labels"].sum() > 0


def test_load_classification(loader):
    ds = loader.load_classification(n_classes=3, n_samples=50)
    assert ds["n_classes"] == 3
    assert len(ds["X_test"]) > 0


def test_load_qa(loader):
    ds = loader.load_qa()
    assert len(ds["questions"]) > 0


def test_load_standard_fallback(loader):
    ds = loader.load("nonexistent_dataset")
    assert ds["source"] == "synthetic"
