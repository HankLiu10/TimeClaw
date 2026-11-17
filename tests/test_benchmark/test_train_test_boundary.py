"""Exercise data boundaries through loaders/controllers, with no model traffic."""
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest

from timeclaw.benchmark import dataset_split, mtbench_adapter as adapter, train_loop
from timeclaw.benchmark.mtbench_support.runtime_context import prepare_task_run_kwargs
from timeclaw.core.config import TimeClawConfig
from timeclaw.data_integrity import DataIntegrityError
from timeclaw.pipeline.runner import PipelineRunner


def config(root):
    cfg = TimeClawConfig(workspace_dir=str(root/'artifacts'))
    cfg.benchmark.repo_root = str(root)
    cfg.benchmark.training_root = 'independent'
    cfg.sbm.managed_server_enabled = False
    available = [name for name, info in adapter.DATASET_REGISTRY.items()
                 if (root/info['path']).exists()]
    cfg.benchmark.evaluation_datasets = available or ['weather_aligned_short']
    return cfg


def record(offset=0):
    return {'input_window':[1.+offset,2.+offset,3.+offset],
            'input_timestamps':[1704067200.,1704070800.,1704074400.],
            'output_window':[4.], 'output_timestamps':[1704078000.],
            'text':'Publicly visible context', 'source_group':f'series-{offset}'}


def write(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(raw))


@pytest.mark.parametrize('split', ['train', ' TRAIN ', None])
def test_official_train_alias_cannot_return_evaluation_data(monkeypatch, split):
    read = Mock(return_value=['official-test-sample'])
    monkeypatch.setattr(dataset_split, 'load_dataset', read)
    with pytest.raises(ValueError, match='independent'):
        if split is None:
            dataset_split.load_split('weather_aligned_short')
        else:
            dataset_split.load_split('weather_aligned_short', split)
    read.assert_not_called()


@pytest.mark.parametrize('alias', ['test', 'all', 'full'])
def test_official_evaluation_aliases_still_load_the_evaluation_pool(monkeypatch, alias):
    expected = ['evaluation-only']
    monkeypatch.setattr(dataset_split, 'load_dataset', Mock(return_value=expected))
    assert dataset_split.load_split('weather_aligned_short', alias) is expected




@pytest.mark.parametrize('link_kind', ['directory', 'file'])
def test_independent_alias_into_official_pool_is_rejected(tmp_path, link_kind):
    pool = tmp_path/adapter.DATASET_REGISTRY['weather_aligned_short']['path']
    write(pool/'sample.json', record())
    independent = tmp_path/'independent/splits/weather_aligned_short'
    if link_kind == 'directory':
        independent.parent.mkdir(parents=True)
        independent.symlink_to(pool, target_is_directory=True)
    else:
        independent.mkdir(parents=True)
        (independent/'renamed.json').symlink_to(pool/'sample.json')
    with pytest.raises(DataIntegrityError, match='official evaluation'):
        adapter.load_independent_dataset('weather_aligned_short', config=config(tmp_path))


@pytest.mark.asyncio
@pytest.mark.parametrize('official_dataset', ['weather_aligned_short', 'weather_qa_long', 'finance_qa_short'])
async def test_renamed_reformatted_input_copy_rejects_before_services(tmp_path, monkeypatch, official_dataset):
    pool = tmp_path/adapter.DATASET_REGISTRY[official_dataset]['path']
    write(pool/'official.json', record())
    copied = record()
    copied.update(text='Changed article', output_window=[9999.], answer='CHANGED')
    write(tmp_path/'independent/splits/weather_aligned_short/renamed.json', copied)
    start = Mock()
    runner = Mock()
    monkeypatch.setattr('timeclaw.sbm.service_manager.ensure_managed_sbm_server', start)
    monkeypatch.setattr(train_loop, 'PipelineRunner', runner)
    from timeclaw.memory import hierarchical
    original_note_root = hierarchical.NOTES_DIR
    with pytest.raises(DataIntegrityError, match='official evaluation'):
        await train_loop.run_train_loop(dataset_name='weather_aligned_short', config=config(tmp_path))
    start.assert_not_called()
    runner.assert_not_called()
    assert hierarchical.NOTES_DIR == original_note_root


def test_different_source_input_with_same_filename_is_admitted(tmp_path):
    pool = tmp_path/adapter.DATASET_REGISTRY['weather_aligned_short']['path']
    write(pool/'same.json', record())
    write(tmp_path/'independent/splits/weather_aligned_short/same.json', record(10))
    assert len(adapter.load_independent_dataset('weather_aligned_short', config=config(tmp_path))) == 1


@pytest.mark.asyncio
async def test_test_mode_cannot_accept_training_supervision_before_model_work():
    sample = adapter.MTBenchSample(file='example.json',dataset='weather_aligned_short',
        domain='weather',task_types=['forecast'],input_window=np.arange(168.),
        output_window=np.arange(24.),text='Context',timestamps=list(range(168)),start_year=2024)
    with pytest.raises(DataIntegrityError, match='ground_truth'):
        await prepare_task_run_kwargs(sample, SimpleNamespace(), 'forecast', mode='test', ground_truth=[999.])
    runner = object.__new__(PipelineRunner)
    runner._ensure_tools = AsyncMock()
    with pytest.raises(DataIntegrityError, match='ground_truth'):
        await runner.run(data=np.arange(168.), mode='test', ground_truth=[999.])
    runner._ensure_tools.assert_not_called()


@pytest.mark.parametrize('selected, contaminated', [(0,False),(1,True)])
def test_parquet_manifest_checks_selected_rows_against_official_pool(tmp_path, selected, contaminated):
    import pandas as pd
    cfg=config(tmp_path)
    cfg.benchmark.evaluation_datasets=['finance_aligned_long']
    pool=tmp_path/adapter.DATASET_REGISTRY['finance_aligned_long']['path']
    pool.mkdir(parents=True)
    pd.DataFrame([record()]).to_parquet(pool/'official.parquet', index=False)
    split=tmp_path/'independent/splits/finance_aligned_short'
    (split/'data').mkdir(parents=True)
    pd.DataFrame([record(10),record()]).to_parquet(split/'data/train.parquet', index=False)
    (split/'train_files.json').write_text(json.dumps([f'{selected:05d}.parquet_row']))
    if contaminated:
        with pytest.raises(DataIntegrityError,match='official evaluation'):
            adapter.load_independent_dataset('finance_aligned_short',config=cfg)
    else:
        samples=adapter.load_independent_dataset('finance_aligned_short',config=cfg)
        assert len(samples)==1 and samples[0].input_window[0]==11.
        assert samples[0].metadata['training_source_admission']['evaluation_datasets_checked']==['finance_aligned_long']


@pytest.mark.asyncio
@pytest.mark.parametrize('task_set,dataset,task', [(name,ds,task) for name in ('paper30_original','timeclaw18') for ds,task in adapter.expanded_benchmark_task_specs(name)])
async def test_all_public_task_preparations_are_invariant_to_test_answers(task_set,dataset,task):
    from timeclaw.text.task_pruning import _task_prompt_text
    info=adapter.DATASET_REGISTRY[dataset]
    sample=adapter.MTBenchSample(file='bookkeeping.json',dataset=dataset,domain=info['domain'],
        task_types=[task],input_window=np.arange(336.)*.01+20.,output_window=np.full(24,987654.321),
        timestamps=[1704067200.+3600*i for i in range(336)],start_year=2024,
        text='Visible article about the observed situation.',question='Choose A: warm B: cool C: rain D: snow.',
        answer='SECRET_ANSWER_ONE',forecast_horizon=24,
        labels={'trend_label':'SECRET_TREND_ONE','correlation':'SECRET_CORRELATION_ONE',
                'out_macd':[987654.321]*24,'out_upper_bb':[987654.321]*24,'hasrain':'SECRET_RAIN_ONE'},
        metadata={'sticker':'ASSET','published_utc':'2024-01-01T00:00:00Z',
                  'output_timestamps':[1704067200.+3600*i for i in range(336,360)]})
    changed=replace(sample,output_window=np.full(24,-876543.219),answer='SECRET_ANSWER_TWO',
        labels={'trend_label':'SECRET_TREND_TWO','correlation':'SECRET_CORRELATION_TWO',
                'out_macd':[-876543.219]*24,'out_upper_bb':[-876543.219]*24,'hasrain':'SECRET_RAIN_TWO'},
        metadata=dict(sample.metadata))
    cfg=TimeClawConfig()
    cfg.benchmark.task_set=task_set
    runner=SimpleNamespace(run=AsyncMock(return_value={'prediction':'protocol-only fixture'}),config=cfg)
    await PipelineRunner.run_inference(runner,sample,task)
    first=runner.run.call_args.kwargs
    await PipelineRunner.run_inference(runner,changed,task)
    second=runner.run.call_args.kwargs
    def encoded(obj):
        return json.dumps(obj,sort_keys=True,default=lambda value:value.tolist())
    assert encoded(first)==encoded(second)
    assert first['mode']=='test' and 'ground_truth' not in first
    assert _task_prompt_text(sample,task)==_task_prompt_text(changed,task)
    visible=encoded(first)+_task_prompt_text(sample,task)
    assert '987654.321' not in visible and '876543.219' not in visible and 'SECRET_' not in visible


@pytest.mark.asyncio
@pytest.mark.parametrize('mode',['training','inference'])
async def test_misspelled_mode_is_not_accepted_as_an_untracked_execution(mode):
    runner=object.__new__(PipelineRunner)
    runner._ensure_tools=AsyncMock()
    with pytest.raises(DataIntegrityError,match='execution mode'):
        await runner.run(data=np.arange(10.),mode=mode)
    runner._ensure_tools.assert_not_called()


