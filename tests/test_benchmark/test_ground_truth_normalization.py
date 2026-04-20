import numpy as np
import pytest

from timeclaw.benchmark.ground_truth import normalize_numeric_ground_truth
from timeclaw.benchmark.mtbench_adapter import MTBenchSample
from timeclaw.benchmark import pipeline_eval as pipeline_eval_mod
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


@pytest.mark.asyncio
async def test_eval_indicator_upper_bb_uses_repaired_ground_truth(monkeypatch):
    sample = MTBenchSample(
        file="repairable.parquet_row",
        dataset="finance_aligned_short",
        domain="finance",
        task_types=["indicator"],
        input_window=np.asarray([1.0, 2.0, 3.0]),
        output_window=np.asarray([4.0, 5.0, 6.0]),
        text="demo article",
        timestamps=[0.0, 1.0, 2.0],
        start_year=2024,
        labels={"out_upper_bb": [np.nan, 7.0, 8.0]},
    )

    class FakePipeline:
        async def run(self, **kwargs):
            return {
                "prediction": {"upper_bb": [7.0, 7.0, 8.0]},
                "execution_log": [],
                "execution_context": {},
                "control_context": {},
                "timeline": {},
            }

    async def fake_final_llm_text_call(system_prompt: str, user_prompt: str) -> str:
        return "[7.0, 7.0, 8.0]"

    monkeypatch.setattr(pipeline_eval_mod, "_final_llm_text_call", fake_final_llm_text_call)

    row = await pipeline_eval_mod._eval_indicator_upper_bb(
        sample,
        FakePipeline(),
        verbose=False,
    )

    assert row["metrics"]["upper_bb_mae"] == 0.0
    assert row["ground_truth"] == {"upper_bb": [7.0, 7.0, 8.0]}
    assert row["official_summary_included"] is True
