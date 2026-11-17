from pathlib import Path
from unittest.mock import Mock
import json
import pytest
from scripts import download_mtbench as download


def test_selected_dataset_uses_exact_revision_and_verified_reuse(tmp_path, monkeypatch):
    def fetch(**kwargs):
        local = Path(kwargs['local_dir'])
        local.mkdir(parents=True)
        (local/'record.json').write_text('{"fixture":true}')
    fetch = Mock(side_effect=fetch)
    monkeypatch.setattr(download, 'snapshot_download', fetch)
    download.download_all(datasets=['weather_aligned_short'], repo_root=tmp_path)
    assert fetch.call_count == 1
    assert fetch.call_args.kwargs['revision'] == download.DATASETS['weather_aligned_short']['revision']
    receipt = json.loads((tmp_path/'data/download_receipts/weather_aligned_short.json').read_text())
    assert len(receipt['files']) == 1
    download.download_all(datasets=['weather_aligned_short'], repo_root=tmp_path)
    assert fetch.call_count == 1
    target = tmp_path/download.DATASETS['weather_aligned_short']['local']/'record.json'
    target.write_text('{"tampered":true}')
    with pytest.raises(ValueError, match='differs'):
        download.download_all(datasets=['weather_aligned_short'], repo_root=tmp_path)
    assert fetch.call_count == 1


def test_unverified_existing_dataset_is_not_overwritten(tmp_path, monkeypatch):
    target = tmp_path/download.DATASETS['weather_aligned_short']['local']
    target.mkdir(parents=True)
    (target/'own_data.json').write_text('{"keep":true}')
    fetch = Mock()
    monkeypatch.setattr(download, 'snapshot_download', fetch)
    with pytest.raises(ValueError, match='no pinned download receipt'):
        download.download_all(datasets=['weather_aligned_short'], repo_root=tmp_path)
    fetch.assert_not_called()
    assert (target/'own_data.json').read_text() == '{"keep":true}'
