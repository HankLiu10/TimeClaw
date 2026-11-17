"""Public commands must call the real pipeline and honor the chosen workspace."""
from types import SimpleNamespace
from unittest.mock import AsyncMock
import json

import numpy as np
import pytest

from timeclaw import cli
from timeclaw.benchmark import run_learning, train_loop
from timeclaw.core.config import TimeClawConfig
from timeclaw.memory import hierarchical as hm
from timeclaw.pipeline.runner import PipelineRunner


@pytest.mark.asyncio
async def test_csv_command_runs_react_with_every_observation(tmp_path, monkeypatch):
    csv = tmp_path / 'input.csv'
    csv.write_text('temperature,humidity\n20,70\n21,72\n22,74\n')
    cfg = TimeClawConfig(workspace_dir=str(tmp_path/'state'))
    monkeypatch.setattr(TimeClawConfig, 'load', lambda: cfg)
    run = AsyncMock(return_value={'prediction': [23.0], 'gate_accepted': True})
    close = AsyncMock()
    monkeypatch.setattr(PipelineRunner, '__init__', lambda self, *a, **kw: None)
    monkeypatch.setattr(PipelineRunner, 'run', run)
    monkeypatch.setattr(PipelineRunner, 'shutdown', close)
    def reject_direct_tool(*args, **kwargs):
        raise AssertionError('Public run command bypassed the ReAct pipeline')
    monkeypatch.setattr('timeclaw.clawhub.tool_registry.get_tool_instance', reject_direct_tool)
    monkeypatch.setattr('timeclaw.sbm.service_manager.ensure_managed_sbm_server', lambda cfg: None)
    monkeypatch.setattr('timeclaw.sbm.service_manager.shutdown_managed_sbm_server', lambda: None)
    args = SimpleNamespace(data=str(csv), task='forecast', horizon=1, output=str(tmp_path/'out'),
                           text=None, description='Predict the next observation', model='auto')
    await cli._cmd_run(args)
    run.assert_awaited_once()
    kw = run.call_args.kwargs
    np.testing.assert_array_equal(kw['data'], [[20,70],[21,72],[22,74]])
    assert kw['task_type'] == 'forecast' and kw['mode'] == 'test'
    assert 'ground_truth' not in kw
    assert kw['column_names'] == ['temperature','humidity']
    close.assert_awaited_once()
    assert json.loads((tmp_path/'out/result.json').read_text())['prediction'] == [23.0]


@pytest.mark.asyncio
async def test_standalone_distillation_uses_configured_memory_and_reasoner(tmp_path, monkeypatch):
    cfg = TimeClawConfig(workspace_dir=str(tmp_path/'separate-state'))
    monkeypatch.setattr(TimeClawConfig, 'load', lambda: cfg)
    monkeypatch.setattr('timeclaw.benchmark.source_admission.begin_distillation_lineage', lambda *a, **kw: None)
    monkeypatch.setattr('timeclaw.benchmark.source_admission.finish_training_lineage', lambda *a, **kw: None)
    fake_backend = object()
    shutdown = AsyncMock()
    monkeypatch.setattr('timeclaw.llm.router.ModelRouter.get_backend', lambda self, role: fake_backend)
    monkeypatch.setattr('timeclaw.llm.router.ModelRouter.shutdown', shutdown)
    seen = []
    class Memory:
        def __init__(self, *, enabled, reasoner_llm=None):
            seen.append((hm.MEMORY_ROOT, reasoner_llm))
        force_distill = AsyncMock()
        cancel_pending_distillation = AsyncMock()
        def summary(self):
            return {'completed': True}
    monkeypatch.setattr(hm, 'HierarchicalMemory', Memory)
    await run_learning.cmd_distill(SimpleNamespace())
    assert seen == [(tmp_path/'separate-state/hierarchical_memory', fake_backend)]
    shutdown.assert_awaited_once()


def test_paper_task_definition_has_original_scope():
    import importlib.util
    from pathlib import Path
    import sys
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root/'scripts'))
    spec = importlib.util.spec_from_file_location('release_inference', root/'scripts/run_timeclaw_mtbench_inference.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    tasks = module._task_specs('timeclaw18', None)
    assert len(tasks) == len({t.key for t in tasks}) == 18
    assert 'weather_mcqa_short' in {t.key for t in tasks}
    assert 'weather_mcqa_long' in {t.key for t in tasks}
    assert 'finance_pred_bb_short' not in {t.key for t in tasks}
    assert 'weather_indicator_long' not in {t.key for t in tasks}


@pytest.mark.asyncio
@pytest.mark.parametrize('case',['rejected','missing_prediction','error','valid_zero'])
async def test_csv_terminal_outcome_is_not_a_false_success(tmp_path,monkeypatch,case):
    from unittest.mock import Mock
    csv=tmp_path/'input.csv';csv.write_text('value\n1\n2\n3\n')
    cfg=TimeClawConfig(workspace_dir=str(tmp_path/'state'))
    monkeypatch.setattr(TimeClawConfig,'load',lambda:cfg)
    result={'prediction':[0.0],'gate_accepted':True}
    if case=='rejected':result['gate_accepted']=False
    if case=='missing_prediction':result['prediction']=None
    if case=='error':result['error']='retained execution error'
    run=AsyncMock(return_value=result);close=AsyncMock();stop=Mock()
    monkeypatch.setattr(PipelineRunner,'__init__',lambda self,*a,**kw:None)
    monkeypatch.setattr(PipelineRunner,'run',run)
    monkeypatch.setattr(PipelineRunner,'shutdown',close)
    monkeypatch.setattr('timeclaw.sbm.service_manager.ensure_managed_sbm_server',lambda cfg:None)
    monkeypatch.setattr('timeclaw.sbm.service_manager.shutdown_managed_sbm_server',stop)
    args=SimpleNamespace(data=str(csv),task='forecast',horizon=1,output=str(tmp_path/'out'),text=None,description='forecast')
    if case=='valid_zero':
        await cli._cmd_run(args)
    else:
        with pytest.raises(RuntimeError,match='failed|rejected'):
            await cli._cmd_run(args)
    assert json.loads((tmp_path/'out/result.json').read_text())==result
    run.assert_awaited_once();close.assert_awaited_once();stop.assert_called_once()
