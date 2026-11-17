"""The independent manifests index the materialized task table, not its source pool."""
import json

import pandas as pd
import pytest

from timeclaw.benchmark.mtbench_adapter import load_independent_dataset
from timeclaw.core.config import TimeClawConfig


@pytest.mark.parametrize("dataset", ["finance_aligned_short", "finance_aligned_long"])
def test_manifest_resolves_against_task_table(tmp_path, dataset):
    root = tmp_path / "independent"
    split = root / "splits" / dataset
    (split / "data").mkdir(parents=True)
    pool = root / "finance/processed"
    pool.mkdir(parents=True)
    def row(text, values):
        return {"text": json.dumps({"content": text}), "input_window": values,
                "input_timestamps": list(range(100, 100 + len(values))),
                "output_window": [4.0], "output_timestamps": [200]}
    pd.DataFrame([row("unselected source pool", [90.0]), row("wrong source row", [99.0])]).to_parquet(
        pool / "train.parquet", index=False)
    pd.DataFrame([row("task sample zero", [1.0, 2.0]), row("selected task sample", [3.0])]).to_parquet(
        split / "data/train-00000-of-00001.parquet", index=False)
    (split / "train_files.json").write_text(json.dumps(["00001.parquet_row"]))
    cfg = TimeClawConfig()
    cfg.benchmark.repo_root = str(tmp_path)
    cfg.benchmark.training_root = "independent"
    cfg.benchmark.evaluation_datasets = [dataset]
    from timeclaw.benchmark.mtbench_adapter import DATASET_REGISTRY
    held_out = tmp_path / DATASET_REGISTRY[dataset]["path"]
    held_out.mkdir(parents=True)
    evaluation_row = row("separate held-out source", [11.0])
    evaluation_row["input_timestamps"] = [10000]
    evaluation_row["output_timestamps"] = [10100]
    pd.DataFrame([evaluation_row]).to_parquet(held_out / "evaluation.parquet", index=False)
    samples = load_independent_dataset(dataset, config=cfg)
    assert len(samples) == 1
    assert samples[0].text == "selected task sample"
    assert samples[0].input_window.tolist() == [3.0]


def test_missing_task_table_cannot_load_source_pool(tmp_path):
    split = tmp_path / "independent/splits/finance_aligned_long"
    split.mkdir(parents=True)
    (split / "train_files.json").write_text('["00000.parquet_row"]')
    cfg = TimeClawConfig()
    cfg.benchmark.repo_root = str(tmp_path)
    cfg.benchmark.training_root = "independent"
    with pytest.raises(FileNotFoundError, match="task parquet"):
        load_independent_dataset("finance_aligned_long", config=cfg)


@pytest.mark.parametrize("manifest,error", [
    (["00000.parquet_row", "00000.parquet_row"], "Duplicate"),
    (["00001.parquet_row"], "missing from task parquet"),
])
def test_invalid_manifest_is_not_silently_shortened(tmp_path, manifest, error):
    split = tmp_path / "independent/splits/finance_aligned_short"
    (split / "data").mkdir(parents=True)
    pd.DataFrame([{"text": '{"content":"news"}', "input_window": [1.0],
                   "input_timestamps": [100], "output_window": [2.0],
                   "output_timestamps": [200]}]).to_parquet(split / "data/train.parquet", index=False)
    (split / "train_files.json").write_text(json.dumps(manifest))
    cfg = TimeClawConfig()
    cfg.benchmark.repo_root = str(tmp_path)
    cfg.benchmark.training_root = "independent"
    with pytest.raises(ValueError, match=error):
        load_independent_dataset("finance_aligned_short", config=cfg)
