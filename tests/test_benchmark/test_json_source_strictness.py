"""Invalid source files must not silently reduce the evaluated/admitted pool."""
import json
from pathlib import Path
import pytest
from timeclaw.benchmark.mtbench_adapter import DATASET_REGISTRY, load_dataset, load_independent_dataset
from timeclaw.core.config import TimeClawConfig


@pytest.mark.parametrize('dataset',['weather_aligned_short','finance_qa_short'])
def test_malformed_official_json_cannot_be_dropped_while_other_records_load(tmp_path,dataset):
    root=tmp_path/DATASET_REGISTRY[dataset]['path'];root.mkdir(parents=True)
    raw={'input_window':[1.,2.],'input_timestamps':[100.,101.],
         'output_window':[3.],'output_timestamps':[102.],'text':'Observed context'}
    (root/'valid.json').write_text(json.dumps(raw))
    (root/'broken.json').write_text('{')
    cfg=TimeClawConfig();cfg.benchmark.repo_root=str(tmp_path)
    with pytest.raises(ValueError,match='Invalid sample JSON'):
        load_dataset(dataset,config=cfg)


def test_invalid_training_json_reports_the_source_error_not_an_empty_corpus(tmp_path):
    split=tmp_path/'independent/splits/weather_aligned_short';split.mkdir(parents=True)
    (split/'broken.json').write_text('{')
    cfg=TimeClawConfig();cfg.benchmark.repo_root=str(tmp_path);cfg.benchmark.training_root='independent'
    with pytest.raises(ValueError,match='Invalid sample JSON'):
        load_independent_dataset('weather_aligned_short',config=cfg)
