"""Exercise the actual CLI control flow, not only the admission helper."""
import sys
from unittest.mock import Mock
import pytest
from tests.test_benchmark.test_paper18_scope import inference_module
from tests.test_benchmark.test_source_lineage import make,load,seal,write
from timeclaw.core.config import TimeClawConfig
from timeclaw.data_integrity import DataIntegrityError


def cli(tmp_path,monkeypatch,cfg):
    module=inference_module()
    config=tmp_path/'config.yaml';config.write_text('workspace_dir: unused\n')
    monkeypatch.setattr(TimeClawConfig,'load',lambda *args:cfg)
    monkeypatch.setattr(sys,'argv',['inference','--config',str(config),'--task-set',cfg.benchmark.task_set,
        '--task','weather_trend_past_short','--samples-per-task','1',
        '--output-root',str(tmp_path/'outputs'),'--run-id','source_guard_test'])
    return module


@pytest.mark.asyncio
async def test_actual_cli_rejects_unknown_memory_before_agent_and_output_changes(tmp_path,monkeypatch):
    cfg,train,test=make(tmp_path)
    write(tmp_path/'artifacts/hierarchical_memory/memory/unknown.json',{'unknown':'origin'})
    module=cli(tmp_path,monkeypatch,cfg)
    runner=Mock(side_effect=AssertionError('Agent constructed before lineage admission'))
    monkeypatch.setattr(module,'PipelineRunner',runner)
    with pytest.raises(DataIntegrityError,match='no training lineage'):
        await module.async_main()
    runner.assert_not_called()
    assert not (tmp_path/'outputs').exists()


@pytest.mark.asyncio
@pytest.mark.parametrize('tamper',[False,True])
async def test_actual_cli_checks_sealed_sources_before_execution_and_after_cleanup(tmp_path,monkeypatch,tamper):
    cfg,train,test=make(tmp_path);note=seal(cfg,load(cfg))
    module=cli(tmp_path,monkeypatch,cfg)
    called=[]
    class NativeBoundary:
        def __init__(self,*args,**kwargs):
            assert kwargs['drain_memory_on_shutdown'] is False
        async def run_inference(self,sample,task,**kwargs):
            called.append(task)
            if tamper:note.write_text('Unexpected inference learning')
            return {'status':'completed','prediction':'increasing','answer_type':'classification',
                    'gate_accepted':True,'confidence':0.9,'total_ms':1}
        async def shutdown(self):called.append('shutdown')
    monkeypatch.setattr(module,'PipelineRunner',NativeBoundary)
    if tamper:
        with pytest.raises(DataIntegrityError,match='sealed lineage'):
            await module.async_main()
    else:
        assert await module.async_main()==0
    assert called==['trend_past','shutdown']


def test_selection_helper_has_no_implicit_cli_globals(tmp_path):
    assert inference_module()._select_from_explicit_manifests(specs=[],datasets={},manifests_dir=tmp_path)=={}


@pytest.mark.asyncio
async def test_actual_resume_refuses_changed_evaluator_targets_without_another_agent_call(tmp_path,monkeypatch):
    import json
    cfg,train,test=make(tmp_path);seal(cfg,load(cfg))
    module=cli(tmp_path,monkeypatch,cfg)
    called=[]
    class Boundary:
        def __init__(self,*args,**kwargs):called.append('constructed')
        async def run_inference(self,sample,task,**kwargs):
            return {'status':'completed','prediction':'stable','answer_type':'classification',
                    'gate_accepted':True,'confidence':0.9,'total_ms':1}
        async def shutdown(self):called.append('shutdown')
    monkeypatch.setattr(module,'PipelineRunner',Boundary)
    assert await module.async_main()==0
    raw=json.loads(test.read_text());raw['output_window']=[9999.];write(test,raw)
    monkeypatch.setattr(sys,'argv',[*sys.argv,'--resume'])
    with pytest.raises(SystemExit,match='fingerprint'):
        await module.async_main()
    assert called==['constructed','shutdown']
