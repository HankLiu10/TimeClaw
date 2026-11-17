"""The publication archive exposes only main experiments and their regression tests."""
from pathlib import Path
import subprocess
import sys
import pytest

ROOT=Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('path',['data_preparation','examples','scripts/create_demo_data.py',
    'timeclaw/benchmark/mtbench.py','timeclaw/benchmark/datasets.py','timeclaw/benchmark/tasks',
    'timeclaw/benchmark/pipeline_eval.py','timeclaw/scheduler','timeclaw/tokencast',
    'timeclaw/agents/reasoner.py'])
def test_non_main_implementation_is_not_in_public_tree(path):
    assert not (ROOT/path).exists()


@pytest.mark.parametrize('command,flag',[
    (['-m','timeclaw.benchmark.run_learning'],'--disable-tool-dropout'),
    (['-m','timeclaw.benchmark.run_learning'],'--no-skills'),
    (['-m','timeclaw.benchmark.run_learning'],'--compare'),
    (['-m','timeclaw.benchmark.run_learning'],'--split'),
    (['-m','timeclaw.benchmark.run_learning'],'--test'),
    (['scripts/run_timeclaw_mtbench_inference.py'],'--disable-memory'),
    (['-m','timeclaw.cli'],'bench'),
])
def test_removed_experiment_entrypoints_are_rejected(command,flag):
    result=subprocess.run([sys.executable,*command,flag],cwd=ROOT,text=True,capture_output=True,timeout=15)
    assert result.returncode==2
    assert 'unrecognized arguments' in result.stderr or 'invalid choice' in result.stderr


def test_resume_identity_covers_source_data_targets_model_and_frozen_memory(tmp_path):
    from dataclasses import replace
    from tests.test_benchmark.test_paper18_scope import inference_module
    from tests.test_benchmark.test_direct_baseline import sample
    from timeclaw.core.config import TimeClawConfig
    m=inference_module();cfg=TimeClawConfig(workspace_dir=str(tmp_path))
    specs=m._task_specs('timeclaw18',['weather_mcqa_short'])
    original=sample('weather_qa_short')
    def binding(record=original):return m._request_bindings(cfg,{'weather_qa_short':[record]},specs,1)
    first=binding()
    assert first!=binding(replace(original,text='Same ID, changed context'))
    assert first!=binding(replace(original,answer='D'))
    cfg.llm.reasoning_effort='low'
    assert first!=binding()
    cfg.llm.reasoning_effort='medium'
    (tmp_path/'training_lineage.json').write_text('{"memory":"changed"}')
    assert first!=binding()
    assert any(path.name=='source_admission.py' for path in m.LOGIC_FINGERPRINT_FILES)
    assert any(path.name=='hierarchical.py' for path in m.LOGIC_FINGERPRINT_FILES)
    assert any(path.name=='base.py' and 'llm' in path.parts for path in m.LOGIC_FINGERPRINT_FILES)
