"""The sole candidate is carried through normalization, analysis, Notes and distillation."""
from types import SimpleNamespace
from unittest.mock import AsyncMock
import pytest
from timeclaw.pipeline.runner import PipelineRunner
from timeclaw.benchmark.mtbench_support.learning_analysis import generate_learning_analysis
from timeclaw.benchmark.train_loop import _normalize_train_result_row
from timeclaw.memory.hierarchical import Notes, DistillationAgent
from tests.test_agents.test_single_execution_learning import single_log


def analysis_payload():
    return {'sample_conditions':['Two-step horizon.'],'winner_evidence':[],'failure_signals':[],
            'reusable_rule':['When this pattern recurs, validate the observed path against more alternatives.'],
            'observed_evidence':['ets_forecast produced two values with measured MAE 1.'],
            'candidate_evidence':[{'path':'ets_forecast','evidence':['Measured MAE 1.']}], 'text_signals':[]}


@pytest.mark.asyncio
async def test_single_note_contains_actual_scores_but_no_winner_and_no_comparative_support(tmp_path,monkeypatch):
    import timeclaw.memory.hierarchical as hm
    monkeypatch.setattr(hm,'NOTES_DIR',tmp_path/'notes')
    notes = Notes()
    record = AsyncMock()
    runner = PipelineRunner(disable_memory=True)
    runner.memory = SimpleNamespace(record=record)
    llm = SimpleNamespace(complete=AsyncMock(return_value=SimpleNamespace(parsed=analysis_payload())))
    runner.router = SimpleNamespace(get_backend=lambda role:llm)
    log = single_log()
    raw = {'evidence_kind':'single_execution','insight':'Observed ets forecast.','recommendation':'Verify further.'}
    normalized = runner._normalize_learning_summary_answer(raw,log,task_type='forecast')
    assert normalized['winner_state'] == 'not_compared'
    assert normalized['winner_prediction'] is None and not normalized['compared_candidates']
    row = _normalize_train_result_row({'status':'completed','answer_type':'learning_summary','task':'forecast',
        'raw_answer':normalized,'execution_log':log})
    assert row['has_prediction'] and row['winner_prediction'] is None
    assert row['observed_candidate']['prediction'] == [1.,2.]
    await runner._save_learning_outcome(answer_type='learning_summary',raw_answer=normalized,prediction=[1.,2.],
        execution_log=log,task_type='forecast',description='Forecast.',domain='finance',scope_keys=['finance_aligned_short_forecast'],
        reasoning='',fingerprint={},profiling_results={},horizon=2,source_metadata={},trace_id='single',ground_truth=[2.,3.])
    ctx = record.call_args.kwargs
    assert ctx['evidence_kind'] == 'single_execution'
    assert ctx['metrics']['mae'] == 1. and ctx['tool_chain'] == ['ets_forecast']
    await notes.add({**ctx,'scope_key':'finance_aligned_short_forecast'})
    item = notes.get_scope_note_items('finance_aligned_short_forecast')[0]
    text = item['text']
    assert 'Observed Tools:' in text and 'Winner Tools:' not in text
    assert 'single_execution' in text and 'no comparative signal' in text.lower()
    assert 'ets_forecast' not in DistillationAgent._compute_tool_stats([text])
    assert 'ets_forecast' not in DistillationAgent._compute_statistics([text])
    distiller = object.__new__(DistillationAgent)
    new_rule = {'kind':'tool_selection','summary':'Prefer ets','support_note_ids':[item['id']]}
    errors = distiller._validate_memory_update_decision({'new_rules':[new_rule]},single_note_ids={item['id']})
    assert any('Single-execution' in e for e in errors)
    errors = distiller._validate_memory_update_decision({'support_updates':[{'rule_id':'r','support_note_ids':[item['id']]}]},
        single_note_ids={item['id']},existing_entries=[{'rule_id':'r','kind':'tool_selection'}])
    assert errors


@pytest.mark.asyncio
async def test_single_analysis_rejects_fabricated_comparative_evidence():
    payload = analysis_payload();payload['winner_evidence'] = ['ets beats other models.']
    llm = SimpleNamespace(complete=AsyncMock(return_value=SimpleNamespace(parsed=payload)))
    analysis,text = await generate_learning_analysis(task_type='forecast',domain='finance',task_description='Forecast.',
        sample_fingerprint='',candidate_traces=[{'tool_chain':['ets_forecast'],'metrics':{'mae':1.}}],
        best_metrics={'mae':1.},evidence_kind='single_execution',llm=llm)
    assert analysis is None


@pytest.mark.asyncio
async def test_single_score_mismatch_cannot_write_a_note():
    runner = PipelineRunner(disable_memory=True);record=AsyncMock();runner.memory=SimpleNamespace(record=record)
    with pytest.raises(Exception,match='differs from training supervision'):
        await runner._save_learning_outcome(answer_type='learning_summary',raw_answer={'evidence_kind':'single_execution'},
            prediction=[1.,2.],execution_log=single_log(),task_type='forecast',description='Forecast.',domain='finance',
            scope_keys=['finance_aligned_short_forecast'],reasoning='',fingerprint={},profiling_results={},horizon=2,
            source_metadata={},trace_id='wrong',ground_truth=[9.,9.])
    record.assert_not_called()


@pytest.mark.asyncio
async def test_strict_single_forecast_cannot_use_interpolation_to_hide_shape_mismatch():
    from timeclaw.benchmark.mtbench_support.task_scoring import score_mtbench_prediction
    log = single_log()
    log[-1]['observation']['metrics'] = score_mtbench_prediction([1.,2.],[1.,2.,3.],'forecast')
    runner = PipelineRunner(disable_memory=True);record=AsyncMock();runner.memory=SimpleNamespace(record=record)
    with pytest.raises(Exception,match='forecast shape'):
        await runner._save_learning_outcome(answer_type='learning_summary',
            raw_answer={'evidence_kind':'single_execution','insight':'Observed path.','recommendation':'Verify further.'},
            prediction=[1.,2.],execution_log=log,task_type='forecast',description='Forecast.',domain='finance',
            scope_keys=['finance_aligned_short_forecast'],reasoning='',fingerprint={},profiling_results={},horizon=2,
            source_metadata={},trace_id='shape',ground_truth=[1.,2.,3.],strict_learning_validation=True)
    record.assert_not_called()


def test_merging_rules_cannot_launder_single_execution_into_relative_support():
    d = object.__new__(DistillationAgent)
    rules = [{'rule_id':'a','kind':'domain_heuristic','evidence':{'sample_refs':['single']}},
             {'rule_id':'b','kind':'tool_selection','evidence':{'sample_refs':['compared']}}]
    errors = d._validate_memory_update_decision({'merge_groups':[['a','b']]},
        single_note_ids={'single'},existing_entries=rules)
    assert any('cannot be merged' in e for e in errors)
