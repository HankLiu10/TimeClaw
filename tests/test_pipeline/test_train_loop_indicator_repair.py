import numpy as np

from timeclaw.benchmark import train_loop
from timeclaw.benchmark.mtbench_adapter import MTBenchSample


def test_build_train_result_row_uses_weather_indicator_keys_for_gt_repair(monkeypatch):
    captured = {}

    def _fake_repair(**kwargs):
        captured["indicator_output_keys"] = kwargs.get("indicator_output_keys")
        return kwargs["raw_answer"]

    monkeypatch.setattr(train_loop, "_repair_learning_summary_with_ground_truth", _fake_repair)

    sample = MTBenchSample(
        file="sample.json",
        dataset="weather_aligned_short",
        domain="weather",
        task_types=["indicator"],
        input_window=np.arange(24, dtype=float),
        output_window=np.concatenate(
            [
                np.full(24, 4.0, dtype=float),
                np.full(24, 2.0, dtype=float),
                np.full(24, 5.0, dtype=float),
            ]
        ),
        text="storm ended overnight",
        timestamps=list(range(24)),
        start_year=2024,
    )
    result = {
        "status": "completed",
        "answer_type": "learning_summary",
        "raw_answer": {
            "insight": "test",
            "recommendation": "test",
            "winner_prediction": {"max": [5.0], "min": [2.0], "diff": [3.0]},
        },
        "execution_log": [],
    }

    train_loop._build_train_result_row(
        ds_name="weather_aligned_short",
        sample_idx=0,
        sample=sample,
        task="indicator",
        pipeline_task="indicator",
        scope_keys=["weather_aligned_short_indicator"],
        result=result,
        train_run_id="run-test",
    )

    assert captured["indicator_output_keys"] == ["max", "min", "diff"]
