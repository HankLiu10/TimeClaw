"""Explicit cohort manifests cannot silently expand, drop or repeat samples."""
import json
from unittest.mock import Mock
import pytest
from timeclaw.benchmark.mtbench_support.sample_selection import (
    load_sample_indices, select_samples_by_index, select_samples_with_original_indices,
)


@pytest.mark.parametrize('payload', [[], {}, {'dataset': 'all'}, {'dataset': []},
    {'dataset': [True]}, {'dataset': ['1']}, {'dataset': [1.5]},
    {'dataset': [-1]}, {'dataset': [0, 0]}, {'dataset': [None]}])
def test_bad_index_file_is_not_coerced_or_silently_ignored(tmp_path, payload):
    path = tmp_path / 'selection.json'
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        load_sample_indices(path)


@pytest.mark.parametrize('selector', [select_samples_by_index, select_samples_with_original_indices])
@pytest.mark.parametrize('mapping', [{}, {'other': [0]}, {'dataset': [0, 0]},
    {'dataset': [-1]}, {'dataset': [2]}, {'dataset': [True]}, {'dataset': [0, 3]}])
def test_bad_explicit_subset_does_not_return_a_changed_cohort(selector, mapping):
    with pytest.raises(ValueError):
        selector(['first', 'second'], 'dataset', mapping)


def test_selection_preserves_caller_order_and_original_indices(tmp_path):
    path = tmp_path / 'selection.json'
    path.write_text('{"dataset": [2, 0]}')
    selected = load_sample_indices(path)
    assert select_samples_with_original_indices(['a', 'b', 'c'], 'dataset', selected) == [(2, 'c'), (0, 'a')]
    assert select_samples_by_index(['a', 'b', 'c'], 'dataset', selected) == ['c', 'a']
    assert select_samples_with_original_indices(['a', 'b'], 'dataset', None) == [(0, 'a'), (1, 'b')]


@pytest.mark.asyncio
async def test_training_rejects_bad_subset_before_services_or_lineage(tmp_path, monkeypatch):
    from timeclaw.benchmark import train_loop
    from tests.test_benchmark.test_source_lineage import make
    cfg, _, _ = make(tmp_path)
    service, pipeline = Mock(), Mock()
    monkeypatch.setattr('timeclaw.sbm.service_manager.ensure_managed_sbm_server', service)
    monkeypatch.setattr(train_loop, 'PipelineRunner', pipeline)
    with pytest.raises(ValueError):
        await train_loop.run_train_loop(dataset_name='weather_aligned_short',
            task_filter='trend_past', sample_indices={'weather_aligned_short': [99]}, config=cfg)
    service.assert_not_called()
    pipeline.assert_not_called()
    from pathlib import Path
    assert not (Path(cfg.workspace_dir) / 'training_lineage.json').exists()
