import numpy as np
import pytest

from timeclaw.benchmark.ground_truth import normalize_numeric_ground_truth
from timeclaw.benchmark.mtbench_adapter import MTBenchSample
from timeclaw.benchmark import train_loop as train_loop_mod


def test_normalize_numeric_ground_truth_repairs_leading_nan_by_interpolation():
    normalized = normalize_numeric_ground_truth([np.nan, 11.0, 13.0])

    assert normalized.valid is True
    assert normalized.repaired is True
    assert normalized.reason == "linear_interpolation"
    assert normalized.value == [11.0, 11.0, 13.0]


def test_normalize_numeric_ground_truth_rejects_insufficient_finite_points():
    normalized = normalize_numeric_ground_truth([np.nan])

    assert normalized.valid is False
    assert normalized.repaired is False
    assert normalized.reason == "insufficient_finite_points"
    assert normalized.value is None


def test_train_extract_ground_truth_repairs_upper_bb_but_skips_unusable_sample():
    repairable = MTBenchSample(
        file="repairable.parquet_row",
        dataset="finance_aligned_short",
        domain="finance",
        task_types=["indicator"],
        input_window=np.asarray([1.0, 2.0, 3.0]),
        output_window=np.asarray([4.0, 5.0, 6.0]),
        text="demo",
        timestamps=[0.0, 1.0, 2.0],
        start_year=2024,
        labels={"out_upper_bb": [np.nan, 7.0, 8.0]},
    )
    unusable = MTBenchSample(
        file="unusable.parquet_row",
        dataset="finance_aligned_short",
        domain="finance",
        task_types=["indicator"],
        input_window=np.asarray([1.0, 2.0]),
        output_window=np.asarray([4.0]),
        text="demo",
        timestamps=[0.0, 1.0],
        start_year=2024,
        labels={"out_upper_bb": [np.nan]},
    )

    assert train_loop_mod._extract_ground_truth(repairable, "indicator_upper_bb") == {
        "upper_bb": [7.0, 7.0, 8.0],
    }
    assert train_loop_mod._extract_ground_truth(unusable, "indicator_upper_bb") is None
