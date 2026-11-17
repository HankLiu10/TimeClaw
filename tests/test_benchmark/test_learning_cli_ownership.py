"""The public CLI must not silently redirect a configured memory workspace."""
import os
import sys
from unittest.mock import AsyncMock
import pytest
from timeclaw.core.config import TimeClawConfig
from timeclaw.benchmark import run_learning


@pytest.mark.parametrize('action',['--train','--distill','--download','--all'])
@pytest.mark.parametrize('source',['default','config','environment','cli'])
def test_actual_cli_workspace_precedence(tmp_path,monkeypatch,action,source):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv('TIMECLAW_WORKSPACE_DIR',raising=False)
    config=tmp_path/'config.yaml'
    config.write_text('workspace_dir: '+str(tmp_path/'from_config')+'\n' if source!='default' else '{}\n')
    monkeypatch.setenv('TIMECLAW_CONFIG',str(config))
    if source in {'environment','cli'}:
        monkeypatch.setenv('TIMECLAW_WORKSPACE_DIR',str(tmp_path/'from_environment'))
    argv=['learning',action]
    if source=='cli':argv+=['--workspace-dir',str(tmp_path/'from_cli')]
    monkeypatch.setattr(sys,'argv',argv)
    seen=[]
    async def perform(args):seen.append(TimeClawConfig.load().workspace_dir)
    def download(args):seen.append(TimeClawConfig.load().workspace_dir)
    monkeypatch.setattr(run_learning,'cmd_train',perform)
    monkeypatch.setattr(run_learning,'cmd_distill',perform)
    monkeypatch.setattr(run_learning,'cmd_download',download)
    # --all deliberately exercises the actual cmd_all -> cmd_train alias.
    run_learning.main()
    expected='.timeclaw' if source=='default' else str(tmp_path/f'from_{source}')
    assert seen==[expected]


@pytest.mark.parametrize('extra',[['--train'],['--distill'],['--train','--distill']])
def test_all_cannot_repeat_its_included_steps(tmp_path,monkeypatch,extra):
    monkeypatch.delenv('TIMECLAW_WORKSPACE_DIR',raising=False)
    monkeypatch.setattr(sys,'argv',['learning','--all',*extra])
    train=AsyncMock();distill=AsyncMock()
    monkeypatch.setattr(run_learning,'cmd_train',train)
    monkeypatch.setattr(run_learning,'cmd_distill',distill)
    with pytest.raises(SystemExit) as caught:run_learning.main()
    assert caught.value.code==2
    train.assert_not_awaited();distill.assert_not_awaited()
    assert 'TIMECLAW_WORKSPACE_DIR' not in os.environ
