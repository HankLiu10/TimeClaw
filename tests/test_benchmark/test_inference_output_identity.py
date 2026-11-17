"""Result files must preserve one admitted row per fixed task/sample identity."""
import hashlib,json,sys
from pathlib import Path
from unittest.mock import Mock
import pytest
from tests.test_benchmark.test_inference_cli_lineage import cli
from tests.test_benchmark.test_source_lineage import make,load,seal
from tests.test_benchmark.test_paper18_scope import inference_module


def snapshot(root):
    return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file()}


def native_boundary(module,monkeypatch):
    calls=[]
    class Boundary:
        def __init__(self,*args,**kwargs):calls.append('constructed')
        async def run_inference(self,*args,**kwargs):
            calls.append('prediction')
            return {'status':'completed','prediction':'stable','answer_type':'classification',
                    'gate_accepted':True,'confidence':0.9,'total_ms':1}
        async def shutdown(self):calls.append('shutdown')
    monkeypatch.setattr(module,'PipelineRunner',Boundary)
    return calls


@pytest.mark.asyncio
async def test_repeated_fresh_launch_refuses_existing_run_before_agent_or_artifact_changes(tmp_path,monkeypatch):
    cfg,train,test=make(tmp_path);seal(cfg,load(cfg))
    module=cli(tmp_path,monkeypatch,cfg);calls=native_boundary(module,monkeypatch)
    assert await module.async_main()==0
    before=snapshot(tmp_path/'outputs')
    with pytest.raises(FileExistsError,match='resume|existing'):
        await module.async_main()
    assert calls==['constructed','prediction','shutdown']
    assert snapshot(tmp_path/'outputs')==before


@pytest.mark.asyncio
async def test_resume_rejects_duplicate_rows_without_picking_a_better_answer(tmp_path,monkeypatch):
    cfg,train,test=make(tmp_path);seal(cfg,load(cfg))
    module=cli(tmp_path,monkeypatch,cfg);calls=native_boundary(module,monkeypatch)
    assert await module.async_main()==0
    path=tmp_path/'outputs/source_guard_test_rows.jsonl'
    raw=path.read_text();path.write_text(raw+raw)
    before=snapshot(tmp_path/'outputs')
    monkeypatch.setattr(sys,'argv',[*sys.argv,'--resume'])
    with pytest.raises(ValueError,match='[Dd]uplicate'):
        await module.async_main()
    assert calls==['constructed','prediction','shutdown']
    assert snapshot(tmp_path/'outputs')==before


@pytest.mark.asyncio
async def test_unchanged_resume_does_not_repeat_prediction(tmp_path,monkeypatch):
    cfg,train,test=make(tmp_path);seal(cfg,load(cfg))
    module=cli(tmp_path,monkeypatch,cfg);calls=native_boundary(module,monkeypatch)
    assert await module.async_main()==0
    monkeypatch.setattr(sys,'argv',[*sys.argv,'--resume'])
    assert await module.async_main()==0
    assert calls.count('prediction')==1
    assert len((tmp_path/'outputs/source_guard_test_rows.jsonl').read_text().splitlines())==1


@pytest.mark.parametrize('row',[[],42,None,{'status':'completed'}])
def test_resume_cannot_silently_drop_or_adopt_non_result_records(tmp_path,row):
    path=tmp_path/'rows.jsonl';path.write_text(json.dumps(row)+'\n')
    with pytest.raises(ValueError,match='object|identity'):
        inference_module()._load_existing_rows(path)


@pytest.mark.asyncio
async def test_resume_cannot_adopt_a_foreign_sample_identity(tmp_path,monkeypatch):
    cfg,train,test=make(tmp_path);seal(cfg,load(cfg))
    module=cli(tmp_path,monkeypatch,cfg);calls=native_boundary(module,monkeypatch)
    assert await module.async_main()==0
    path=tmp_path/'outputs/source_guard_test_rows.jsonl'
    row=json.loads(path.read_text());row['canonical_sample_id']='file:foreign.json'
    path.write_text(json.dumps(row)+'\n')
    before=snapshot(tmp_path/'outputs')
    monkeypatch.setattr(sys,'argv',[*sys.argv,'--resume'])
    with pytest.raises(ValueError,match='frozen sample selection'):
        await module.async_main()
    assert calls==['constructed','prediction','shutdown']
    assert snapshot(tmp_path/'outputs')==before


def test_resume_and_clear_are_not_silently_combined(tmp_path):
    with pytest.raises(ValueError,match='cannot be combined'):
        inference_module()._admit_run_output(output_root=tmp_path,run_id='main',resume=True,clear_output=True)


def test_same_logical_run_has_one_writer_and_releases_after_failure(tmp_path):
    module=inference_module()
    with module._run_output_lease(tmp_path,'main'):
        with pytest.raises(RuntimeError,match='active writer'):
            with module._run_output_lease(tmp_path,'main'):
                pytest.fail('Two owners entered the same result file lifecycle')
        with module._run_output_lease(tmp_path,'different'):
            pass
    with pytest.raises(RuntimeError,match='simulation'):
        with module._run_output_lease(tmp_path,'main'):
            raise RuntimeError('simulation')
    with module._run_output_lease(tmp_path,'main'):
        pass


@pytest.mark.parametrize('run_id',['../other','run*','a/b','[all]','..'])
def test_run_id_cannot_escape_or_glob_other_output_runs(tmp_path,run_id):
    root=tmp_path/'not_created'
    with pytest.raises(ValueError,match='filename component'):
        with inference_module()._run_output_lease(root,run_id):
            pytest.fail('Invalid run ID entered the output owner')
    assert not root.exists()


def test_clear_does_not_delete_another_run_with_a_shared_prefix(tmp_path):
    module=inference_module()
    owned=tmp_path/'main_rows.jsonl';owned.write_text('owned')
    foreign=tmp_path/'main_retry_rows.jsonl';foreign.write_text('must survive')
    foreign_dir=tmp_path/'main_retry_task_manifests';foreign_dir.mkdir()
    (foreign_dir/'selection.json').write_text('must survive')
    own_dir=tmp_path/'main_task_manifests';own_dir.mkdir()
    module._clear_run_output(output_root=tmp_path,run_id='main',clear_output=True)
    assert foreign.read_text()=='must survive'
    assert (foreign_dir/'selection.json').read_text()=='must survive'
    assert not owned.exists() and not own_dir.exists()


def test_fresh_run_does_not_confuse_another_run_prefix_with_its_own_outputs(tmp_path):
    (tmp_path/'main_retry_rows.jsonl').write_text('other run')
    inference_module()._admit_run_output(output_root=tmp_path,run_id='main',resume=False,clear_output=False)


@pytest.mark.asyncio
async def test_initial_fingerprint_is_saved_before_agent_construction(tmp_path,monkeypatch):
    cfg,train,test=make(tmp_path);seal(cfg,load(cfg))
    module=cli(tmp_path,monkeypatch,cfg)
    def interrupted(*args,**kwargs):
        manifest=tmp_path/'outputs/source_guard_test_manifest.json'
        assert manifest.is_file(),'First-batch interruption must leave its admitted resume fingerprint'
        saved=json.loads(manifest.read_text())
        assert saved['completed_rows']==0 and saved['total_work_items']==1
        assert saved['logic_fingerprint']['digest']
        raise RuntimeError('simulated construction interruption')
    monkeypatch.setattr(module,'PipelineRunner',interrupted)
    with pytest.raises(RuntimeError,match='construction interruption'):
        await module.async_main()
    native_boundary(module,monkeypatch)
    monkeypatch.setattr(sys,'argv',[*sys.argv,'--resume'])
    assert await module.async_main()==0


@pytest.mark.asyncio
async def test_resume_rebuilds_failure_sidecar_from_authoritative_rows(tmp_path,monkeypatch):
    cfg,train,test=make(tmp_path);seal(cfg,load(cfg))
    module=cli(tmp_path,monkeypatch,cfg);native_boundary(module,monkeypatch)
    assert await module.async_main()==0
    root=tmp_path/'outputs'
    path=root/'source_guard_test_rows.jsonl'
    row=json.loads(path.read_text());row['status']='failed';row['error']='retained actual failure'
    path.write_text(json.dumps(row)+'\n')
    # Simulate an interrupted snapshot: durable result row, stale empty sidecar.
    (root/'source_guard_test_failed_rows.json').write_text('[]')
    monkeypatch.setattr(sys,'argv',[*sys.argv,'--resume'])
    assert await module.async_main()==1
    failed=json.loads((root/'source_guard_test_failed_rows.json').read_text())
    assert len(failed)==1 and failed[0]['error']=='retained actual failure'
    assert len(path.read_text().splitlines())==1
