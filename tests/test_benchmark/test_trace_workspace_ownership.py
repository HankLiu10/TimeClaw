"""Trace evidence and Note references must belong to the configured workspace."""
import json
from pathlib import Path
import pytest
from timeclaw.benchmark import train_loop
from timeclaw.core import trace_logger
from timeclaw.memory import hierarchical


@pytest.fixture
def workspace(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    current=tmp_path/'custom_workspace'
    train_loop._configure_workspace_roots(current)
    trace_id='shared-trace'
    def record(folder,tag):
        folder.mkdir(parents=True)
        p=folder/f'{trace_id}.jsonl'
        events=[{'event':'agent_step','payload':{'phase':'react_step_1','summary':{
            'tool':'ets_forecast','observation_preview':tag}}},
            {'event':'run_end','payload':{'status':'failed','error':tag}}]
        p.write_text('\n'.join(json.dumps(e) for e in events)+'\n')
        return p
    own=record(current/'traces','CURRENT_WORKSPACE_EVIDENCE')
    record(tmp_path/'.timeclaw/traces','FOREIGN_WORKSPACE_EVIDENCE')
    return current,trace_id,own


def test_trace_lookup_uses_the_same_owner_as_trace_creation(workspace):
    root,trace_id,own=workspace
    assert train_loop._trace_path_for_id(trace_id)==own
    trace_logger.start_run('created-now')
    trace_logger.end_run('failed',{'error':'retained diagnostic'})
    assert train_loop._trace_path_for_id('created-now')==root/'traces/created-now.jsonl'


def test_failed_diagnostics_do_not_read_another_workspaces_trace(workspace):
    _,trace_id,_=workspace
    result=train_loop._recover_failed_result_from_trace(trace_id)
    assert result['trace_error']=='CURRENT_WORKSPACE_EVIDENCE'
    assert result['trace_status']=='failed'


def test_evidence_preview_is_from_the_configured_trace(workspace):
    _,trace_id,_=workspace
    preview=train_loop._build_train_result_evidence_preview_from_trace(
        trace_id=trace_id,substantive_tool_chain=['ets_forecast'])
    assert 'CURRENT_WORKSPACE_EVIDENCE' in preview
    assert 'FOREIGN_WORKSPACE_EVIDENCE' not in preview


def test_note_trace_reference_uses_the_configured_trace_owner(workspace):
    _,trace_id,own=workspace
    lines=hierarchical.Notes._source_lines({'trace_id':trace_id,'source_metadata':{'sample_ref':'input-reference'}})
    assert f'- Trace Path: {own}' in lines


def test_switching_workspace_does_not_adopt_an_old_same_id_trace(workspace):
    root,trace_id,_=workspace
    train_loop._configure_workspace_roots(root.parent/'next_workspace')
    assert train_loop._trace_path_for_id(trace_id) is None
    assert train_loop._recover_failed_result_from_trace(trace_id)=={}
    assert train_loop._build_train_result_evidence_preview_from_trace(
        trace_id=trace_id,substantive_tool_chain=['ets_forecast'])==''
