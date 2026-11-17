"""Public jobs keep memory writable by one learner or frozen for inference."""
import fcntl
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from timeclaw.benchmark import train_loop,run_learning
from timeclaw.core.config import TimeClawConfig
from timeclaw.data_integrity import DataIntegrityError
from tests.test_benchmark.test_source_lineage import make,load,seal
from tests.test_benchmark.test_inference_cli_lineage import cli


LOCK_NAME='.workspace.lock'


def can_access(root,*,write):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    with (root/LOCK_NAME).open('a') as handle:
        try:fcntl.flock(handle.fileno(),(fcntl.LOCK_EX if write else fcntl.LOCK_SH)|fcntl.LOCK_NB)
        except BlockingIOError:return False
        fcntl.flock(handle.fileno(),fcntl.LOCK_UN)
    return True


def probe(cfg):
    return {'writer_allowed':can_access(cfg.workspace_dir,write=True),
            'reader_allowed':can_access(cfg.workspace_dir,write=False)}


@pytest.mark.asyncio
@pytest.mark.parametrize('entry',['benchmark','csv','training','distillation'])
async def test_public_job_holds_memory_access_until_cleanup(tmp_path,monkeypatch,entry):
    cfg,_,_=make(tmp_path)
    if entry!='training':seal(cfg,load(cfg))
    monkeypatch.setattr(TimeClawConfig,'load',lambda *args:cfg)
    monkeypatch.setattr('timeclaw.sbm.service_manager.ensure_managed_sbm_server',lambda cfg:None)
    monkeypatch.setattr('timeclaw.sbm.service_manager.shutdown_managed_sbm_server',lambda:None)
    observations=[]
    class Reader:
        def __init__(self,*args,**kwargs):observations.append(probe(cfg))
        async def run_inference(self,*args,**kwargs):
            observations.append(probe(cfg))
            return {'prediction':'stable','answer_type':'classification','gate_accepted':True,'total_ms':1}
        async def run(self,*args,**kwargs):
            observations.append(probe(cfg))
            return {'prediction':[0.0],'gate_accepted':True}
        async def shutdown(self):observations.append(probe(cfg))
    if entry=='benchmark':
        module=cli(tmp_path,monkeypatch,cfg);monkeypatch.setattr(module,'PipelineRunner',Reader)
        assert await module.async_main()==0
    elif entry=='csv':
        from timeclaw import cli as public_cli
        monkeypatch.setattr('timeclaw.pipeline.runner.PipelineRunner',Reader)
        source=tmp_path/'input.csv';source.write_text('value\n1\n2\n3\n')
        await public_cli._cmd_run(SimpleNamespace(data=str(source),task='forecast',horizon=1,text=None,
            description='forecast',output=str(tmp_path/'out')))
    else:
        def stopped(*args,**kwargs):
            observations.append(probe(cfg))
            raise RuntimeError('controlled startup interruption')
        if entry=='training':
            monkeypatch.setattr(train_loop,'PipelineRunner',stopped)
            coroutine=train_loop.run_train_loop(dataset_name='weather_aligned_short',task_filter='trend_past',config=cfg)
        else:
            monkeypatch.setattr('timeclaw.llm.router.ModelRouter',stopped)
            coroutine=run_learning.cmd_distill(SimpleNamespace())
        with pytest.raises(RuntimeError,match='controlled startup'):await coroutine
    expected={'writer_allowed':False,'reader_allowed':entry in {'benchmark','csv'}}
    assert observations and all(item==expected for item in observations)
    assert can_access(cfg.workspace_dir,write=True),'A completed/failed owner must release access'


@pytest.mark.asyncio
@pytest.mark.parametrize('entry',['benchmark','csv','training','distillation'])
async def test_foreign_writer_blocks_public_job_before_backend_or_receipt_change(tmp_path,monkeypatch,entry):
    cfg,_,_=make(tmp_path);seal(cfg,load(cfg));root=Path(cfg.workspace_dir)
    before=(root/'training_lineage.json').read_bytes()
    monkeypatch.setattr(TimeClawConfig,'load',lambda *args:cfg)
    backend=Mock(side_effect=AssertionError('Backend started while another writer owns memory'))
    monkeypatch.setattr('timeclaw.sbm.service_manager.ensure_managed_sbm_server',backend)
    if entry=='benchmark':
        module=cli(tmp_path,monkeypatch,cfg);monkeypatch.setattr(module,'PipelineRunner',backend)
        coroutine=module.async_main()
    elif entry=='csv':
        from timeclaw import cli as public_cli
        monkeypatch.setattr('timeclaw.pipeline.runner.PipelineRunner',backend)
        source=tmp_path/'input.csv';source.write_text('value\n1\n2\n3\n')
        coroutine=public_cli._cmd_run(SimpleNamespace(data=str(source),task='forecast',horizon=1,
            text=None,description='forecast',output=str(tmp_path/'out')))
    elif entry=='training':
        monkeypatch.setattr(train_loop,'PipelineRunner',backend)
        coroutine=train_loop.run_train_loop(dataset_name='weather_aligned_short',task_filter='trend_past',config=cfg)
    else:
        monkeypatch.setattr('timeclaw.llm.router.ModelRouter',backend)
        coroutine=run_learning.cmd_distill(SimpleNamespace())
    with (root/LOCK_NAME).open('a') as external:
        fcntl.flock(external.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(DataIntegrityError,match='workspace.*busy|busy.*workspace'):
            await coroutine
        fcntl.flock(external.fileno(),fcntl.LOCK_UN)
    backend.assert_not_called()
    assert (root/'training_lineage.json').read_bytes()==before
    assert not (tmp_path/'out').exists() and not (tmp_path/'outputs').exists()
