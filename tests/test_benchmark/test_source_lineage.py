"""End-to-end source admission and sealed ancestry, with real filesystem data."""
import json
from pathlib import Path
import pytest

from timeclaw.benchmark.mtbench_adapter import DATASET_REGISTRY, load_dataset, load_independent_dataset
from timeclaw.benchmark.source_admission import (begin_training_lineage,finish_training_lineage,
                                                assert_inference_lineage, RECEIPT)
from timeclaw.core.config import TimeClawConfig
from timeclaw.data_integrity import DataIntegrityError


def make(tmp_path):
    cfg=TimeClawConfig(workspace_dir=str(tmp_path/'artifacts'))
    cfg.benchmark.repo_root=str(tmp_path)
    cfg.benchmark.training_root='independent'
    cfg.benchmark.evaluation_datasets=['weather_aligned_short']
    cfg.sbm.managed_server_enabled=False
    def row(start):
        return {'input_window':[1.,2.,3.],'input_timestamps':[start,start+1,start+2],
                'output_window':[4.],'output_timestamps':[start+3], 'text':'Observed context',
                'source_group':'physical-station-A'}
    train=tmp_path/'independent/splits/weather_aligned_short/train.json'
    test=tmp_path/DATASET_REGISTRY['weather_aligned_short']['path']/'test.json'
    write(train,row(100))
    write(test,row(1000))
    return cfg,train,test


def write(p,data):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(data))


def load(cfg):
    return {'weather_aligned_short':load_independent_dataset('weather_aligned_short',config=cfg)}


def seal(cfg,sources):
    begin_training_lineage(cfg,sources)
    note=Path(cfg.workspace_dir)/'hierarchical_memory/notes/source.md'
    note.parent.mkdir(parents=True,exist_ok=True)
    note.write_text('Genuine admitted learning artifact fixture')
    finish_training_lineage(cfg,finalized=True)
    return note


@pytest.mark.parametrize('part',['input','supervised_future'])
def test_partial_window_overlap_is_rejected_even_if_files_and_values_differ(tmp_path,part):
    cfg,train,test=make(tmp_path)
    raw=json.loads(test.read_text())
    raw['input_timestamps']=[101,102,103] if part=='input' else [103,104,105]
    raw['output_timestamps']=[104] if part=='input' else [106]
    raw['input_window']=[77.,88.,99.]
    write(test,raw)
    with pytest.raises(DataIntegrityError,match='source-time'):
        load(cfg)


def test_ambiguous_entity_with_overlapping_times_is_not_assumed_independent(tmp_path):
    cfg,train,test=make(tmp_path)
    raw=json.loads(test.read_text());raw['input_timestamps']=[101,102,103];raw['output_timestamps']=[104]
    raw.pop('source_group');write(test,raw)
    with pytest.raises(DataIntegrityError,match='source_group'):
        load(cfg)


def test_true_distinct_source_groups_allow_coincident_time_ranges(tmp_path):
    cfg,train,test=make(tmp_path)
    raw=json.loads(test.read_text());raw['input_timestamps']=[101,102,103];raw['output_timestamps']=[104]
    raw['source_group']='physical-station-B';write(test,raw)
    assert len(load(cfg)['weather_aligned_short'])==1


def test_shared_article_source_is_rejected_across_separate_time_windows(tmp_path):
    cfg,train,test=make(tmp_path)
    for p in (train,test):
        raw=json.loads(p.read_text());raw['text_source_id']='public-article-source-1';write(p,raw)
    with pytest.raises(DataIntegrityError,match='same text source'):
        load(cfg)


def test_missing_declared_pool_cannot_be_silently_skipped(tmp_path):
    cfg,train,test=make(tmp_path)
    cfg.benchmark.evaluation_datasets.append('finance_aligned_short')
    with pytest.raises(FileNotFoundError,match='Dataset directory'):
        load(cfg)


def test_empty_evaluation_scope_cannot_disable_admission(tmp_path):
    cfg,train,test=make(tmp_path);cfg.benchmark.evaluation_datasets=[]
    with pytest.raises(DataIntegrityError,match='nonempty'):
        load(cfg)


def test_missing_future_time_support_is_rejected(tmp_path):
    cfg,train,test=make(tmp_path)
    raw=json.loads(train.read_text());raw.pop('output_timestamps');write(train,raw)
    with pytest.raises(DataIntegrityError,match='output timestamps'):
        load(cfg)


@pytest.mark.parametrize('folder',['hierarchical_memory/memory','train_results'])
def test_old_unknown_memory_or_checkpoint_cannot_be_adopted(tmp_path,folder):
    cfg,train,test=make(tmp_path)
    unknown=Path(cfg.workspace_dir)/folder/'old.json';write(unknown,{'unknown':'origin'})
    with pytest.raises(DataIntegrityError,match='no training lineage'):
        begin_training_lineage(cfg,load(cfg))
    assert not (Path(cfg.workspace_dir)/RECEIPT).exists()


def test_clean_learning_finalization_and_inference_use_same_sources(tmp_path):
    cfg,train,test=make(tmp_path);sources=load(cfg);seal(cfg,sources)
    before=(Path(cfg.workspace_dir)/RECEIPT).read_bytes()
    receipt=assert_inference_lineage(cfg,{'weather_aligned_short':load_dataset('weather_aligned_short',config=cfg)},disable_memory=False)
    assert receipt['state']=='finalized'
    assert (Path(cfg.workspace_dir)/RECEIPT).read_bytes()==before
    begin_training_lineage(cfg,load(cfg))
    finish_training_lineage(cfg,finalized=True)


@pytest.mark.parametrize('change',['values','text','answer','order'])
def test_resume_cannot_reuse_same_indices_with_changed_training_material(tmp_path,change):
    cfg,train,test=make(tmp_path);seal(cfg,load(cfg))
    raw=json.loads(train.read_text())
    if change=='values':raw['input_window'][0]=99.
    if change=='text':raw['text']='Different source article'
    if change=='answer':raw['output_window']=[99.]
    if change=='order':
        write(train.parent/'new-first.json',{**raw,'input_window':[4.,5.,6.],
            'input_timestamps':[90,91,92],'output_timestamps':[93]})
    else:write(train,raw)
    with pytest.raises(DataIntegrityError,match='Training corpus'):
        begin_training_lineage(cfg,load(cfg))


@pytest.mark.parametrize('artifact',['hierarchical_memory/notes/source.md','train_results/new.json'])
def test_copying_or_editing_learning_artifacts_invalidates_the_seal(tmp_path,artifact):
    cfg,train,test=make(tmp_path);seal(cfg,load(cfg))
    path=Path(cfg.workspace_dir)/artifact;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('Replaced')
    with pytest.raises(DataIntegrityError,match='sealed lineage'):
        begin_training_lineage(cfg,load(cfg))
    with pytest.raises(DataIntegrityError,match='sealed lineage'):
        assert_inference_lineage(cfg,{},disable_memory=False)


def test_inference_cannot_switch_held_out_inputs_after_learning(tmp_path):
    cfg,train,test=make(tmp_path);seal(cfg,load(cfg))
    raw=json.loads(test.read_text());raw['text']='Different evaluation article';write(test,raw)
    with pytest.raises(DataIntegrityError,match='Evaluation inputs changed'):
        assert_inference_lineage(cfg,{'weather_aligned_short':load_dataset('weather_aligned_short',config=cfg)},disable_memory=False)


def test_evaluation_answers_do_not_determine_memory_admission(tmp_path):
    cfg,train,test=make(tmp_path);seal(cfg,load(cfg))
    raw=json.loads(test.read_text());raw['output_window']=[999.];raw['answer']='B';write(test,raw)
    assert assert_inference_lineage(cfg,{'weather_aligned_short':load_dataset('weather_aligned_short',config=cfg)},disable_memory=False)['state']=='finalized'


def test_unfinalized_memory_cannot_be_used_for_inference(tmp_path):
    cfg,train,test=make(tmp_path);begin_training_lineage(cfg,load(cfg))
    with pytest.raises(DataIntegrityError,match='finalized'):
        assert_inference_lineage(cfg,{},disable_memory=False)


def test_finalized_interruption_can_resume_only_with_matching_artifacts(tmp_path):
    cfg,train,test=make(tmp_path);begin_training_lineage(cfg,load(cfg));finish_training_lineage(cfg,finalized=False)
    with pytest.raises(DataIntegrityError,match='finalized'):
        assert_inference_lineage(cfg,{},disable_memory=False)
    begin_training_lineage(cfg,load(cfg));finish_training_lineage(cfg,finalized=True)


def test_empty_memory_is_plain_react_but_unknown_memory_cannot_bypass_lineage(tmp_path):
    cfg,train,test=make(tmp_path)
    assert assert_inference_lineage(cfg,{},disable_memory=False) is None
    write(Path(cfg.workspace_dir)/'hierarchical_memory/unknown.json',{'unknown':True})
    with pytest.raises(DataIntegrityError,match='no training lineage'):
        assert_inference_lineage(cfg,{},disable_memory=False)
    with pytest.raises(DataIntegrityError,match='no-memory bypass'):
        assert_inference_lineage(cfg,{},disable_memory=True)


@pytest.mark.asyncio
async def test_cli_does_not_rewrite_a_summary_after_its_owner_seals_memory(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from timeclaw.benchmark import run_learning,train_loop
    cfg,train,test=make(tmp_path)
    monkeypatch.setattr(TimeClawConfig,'load',lambda:cfg)
    summary=Path(cfg.workspace_dir)/'train_results/train_summary.json'
    expected={}
    async def native_job(**kwargs):
        sources=load(cfg);begin_training_lineage(cfg,sources)
        summary.parent.mkdir(parents=True,exist_ok=True)
        summary.write_text(json.dumps({'weather_aligned_short':{'total':1,'summary':{}}},indent=2)+'\n')
        expected['bytes']=summary.read_bytes()
        finish_training_lineage(cfg,finalized=True)
        return {'weather_aligned_short':{'total':1,'summary':{}}}
    monkeypatch.setattr(train_loop,'run_train_loop',native_job)
    args=SimpleNamespace(dataset='weather_aligned_short',task='trend_past',max_samples=1,
        sample_index_file=None,verbose=False,concurrency=1,per_task_concurrency=1,
        indicator_concurrency=None,workspace_dir=cfg.workspace_dir)
    await run_learning.cmd_train(args)
    assert summary.read_bytes()==expected['bytes']
    assert_inference_lineage(cfg,{},disable_memory=False)


@pytest.mark.parametrize('folder',['hierarchical_memory','train_results'])
def test_symlinked_learning_root_cannot_reuse_other_workspace_state(tmp_path,folder):
    cfg,train,test=make(tmp_path)
    outside=tmp_path/'outside';outside.mkdir()
    root=Path(cfg.workspace_dir);root.mkdir()
    (root/folder).symlink_to(outside,target_is_directory=True)
    with pytest.raises(DataIntegrityError,match='symlink'):
        begin_training_lineage(cfg,load(cfg))


@pytest.mark.asyncio
@pytest.mark.parametrize('has_unknown_notes',[False,True])
async def test_standalone_distillation_cannot_adopt_unknown_or_absent_learning_sources(tmp_path,monkeypatch,has_unknown_notes):
    from types import SimpleNamespace
    from unittest.mock import Mock
    from timeclaw.benchmark import run_learning
    cfg,train,test=make(tmp_path)
    monkeypatch.setattr(TimeClawConfig,'load',lambda:cfg)
    router=Mock()
    monkeypatch.setattr('timeclaw.llm.router.ModelRouter',router)
    if has_unknown_notes:
        write(Path(cfg.workspace_dir)/'hierarchical_memory/notes/unknown.json',{'from':'unverified'})
    with pytest.raises(DataIntegrityError):
        await run_learning.cmd_distill(SimpleNamespace())
    router.assert_not_called()
