import json

from timeclaw.benchmark.mtbench_adapter import _iter_sample_json_files


def test_iter_sample_json_files_excludes_split_manifests(tmp_path):
    (tmp_path / "station_1.json").write_text(json.dumps({"input_window": [1.0]}))
    (tmp_path / "split_meta.json").write_text(json.dumps({"dataset": "demo"}))
    (tmp_path / "train_files.json").write_text(json.dumps(["station_1.json"]))
    (tmp_path / "test_files.json").write_text(json.dumps([]))

    files = _iter_sample_json_files(tmp_path)

    assert [path.name for path in files] == ["station_1.json"]
