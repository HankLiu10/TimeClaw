"""One scored path is factual evidence, never a comparative winner."""
from copy import deepcopy
import pytest
from timeclaw.agents.executor import ExecutorAgent


def single_log():
    return [
        {'tool':'spawn_subagent','success':True,'args':{'tasks':[{},{}]},'output':[
            {'success':True,'result':{'branch_id':'one','answer':[1.,2.], 'reasoning':'The tool produced two values.',
             'execution_log':[{'tool':'ets_forecast','success':True,'output':{'forecast':[1.,2.]}}]}},
            {'success':False,'error':'other branch failed'}]},
        {'tool':'evaluate_against_gt','success':True,'args':{'prediction':[1.,2.]},
         'observation':{'metrics':{'mae':1.,'rmse':1.,'mse':1.},'ground_truth':[2.,3.]}}]


def test_single_candidate_is_recordable_only_at_exploration_termination():
    log = single_log()
    assert not ExecutorAgent._training_completion_contract_satisfied(log,task_type='forecast',horizon=2)
    c = ExecutorAgent._single_execution_candidate(log,task_type='forecast',horizon=2)
    assert c['prediction'] == [1.,2.]
    assert c['metrics']['mae'] == 1.


@pytest.mark.parametrize('change',['unscored','failed_score','different_prediction','name_only_match','unbacked','zero','no_exploration','score_before_execution'])
def test_single_candidate_rejects_unearned_evidence(change):
    log = deepcopy(single_log())
    if change == 'unscored': log.pop()
    if change == 'failed_score': log[-1]['success'] = False
    if change == 'different_prediction': log[-1]['args']['prediction'] = [9.,9.]
    if change == 'name_only_match':
        log[-1].update(tool='evaluate_batch_against_gt',observation={'results':[{'name':'one','prediction':[9.,9.],'metrics':{'mae':0.}}]})
    if change == 'unbacked': log[0]['output'][0]['result']['execution_log'] = []
    if change == 'zero': log[0]['output'][0]['success'] = False
    if change == 'score_before_execution': log.reverse()
    if change == 'no_exploration': log[0]['output'] = log[0]['output'][:1]; log[0]['args']['tasks'] = [{}]
    assert ExecutorAgent._single_execution_candidate(log,task_type='forecast',horizon=2) is None


@pytest.mark.asyncio
async def test_real_main_loop_evaluates_sole_path_at_budget_then_records_terminal_kind(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    import timeclaw.agents.executor as module
    from timeclaw.clawhub.tool_base import ToolResult
    from timeclaw.core.tea_protocol import TaskEnvelope, TaskSpec, EnvironmentSnapshot
    from tests.test_agents.test_executor_reasoning import _decision_response, _structured_thought
    from timeclaw.clawhub.tool_registry import discover_and_register
    discover_and_register()
    executor = ExecutorAgent(llm=SimpleNamespace())
    decisions = iter([
        {'done':False,'tool':'spawn_subagent','args':{'tasks':[{'slot_id':0,'task':'first'},{'slot_id':1,'task':'second'}]},
         'thought':_structured_thought('Two alternatives are required.','Execute both branches.')},
        {'done':False,'tool':'evaluate_against_gt','args':{'prediction':[1.,2.]},
         'thought':_structured_thought('Only ets_forecast produced a valid output and budget is ending.','Evaluate its observed forecast.')},
    ])
    async def decide(*a,**kw):
        d = next(decisions)
        return _decision_response(d), d
    async def spawn(*a,**kw):
        return {'n_tasks':2,'n_success':1,'results':single_log()[0]['output'],'observation':'One valid output; one failed branch.'}
    async def execute(name,*a,**kw):
        assert name == 'evaluate_against_gt'
        return ToolResult(success=True,data=single_log()[-1]['observation'])
    terminal = {'done':True,'answer_type':'learning_summary',
        'answer':{'insight':'ets_forecast produced two values with MAE 1.','recommendation':'Check this pattern on more cases; no comparison is available.'},
        'reasoning':'1. ets_forecast produced [1,2].\n2. Its measured MAE was 1.',
        'execution_context':{'tool_chain':['ets_forecast'],'public_reasoning':['ets_forecast produced [1,2].'],'evidence_summary':'One scored output.'}}
    review = AsyncMock(return_value=terminal)
    monkeypatch.setattr(executor,'_request_structured_decision',decide)
    monkeypatch.setattr(executor,'_spawn_subagents',spawn)
    monkeypatch.setattr(executor,'_execute_tool',execute)
    monkeypatch.setattr(executor,'_prepare_main_spawn_subagent_tasks',lambda **kw: (kw['tool_args']['tasks'],'',''))
    monkeypatch.setattr(executor,'_request_terminal_path_review',review)
    monkeypatch.setattr(executor,'_stepwise_reasoning_verdict',AsyncMock(return_value={'passed':True}))
    monkeypatch.setattr(executor,'_execution_context_verdict',AsyncMock(return_value={'passed':True}))
    monkeypatch.setattr(module,'MAX_REACT_STEPS',2)
    plan = dict(candidates=['spawn_subagent','evaluate_against_gt'],task_type='forecast',task_description='Forecast two steps.',
        tool_names='spawn_subagent, evaluate_against_gt',tool_notes_section='',skills_section='',system_memory='',
        modality_hint='',profiling_section='',subagent_slot_section='',subagent_slots=[{'slot_id':0},{'slot_id':1}],
        fingerprint={},horizon=2,domain='finance',reanalysis_section='',_mode='train',_defer_learning_record=False,
        _ground_truth=[2.,3.],indicator_output_keys=[],modality='timeseries')
    result = await executor.act(plan,TaskEnvelope(task=TaskSpec(task_id='single',task_type='forecast',description='forecast'),
        environment=EnvironmentSnapshot(data=[1.,2.,3.],modalities=['timeseries'])))
    assert [x['tool'] for x in result['execution_log']] == ['spawn_subagent','evaluate_against_gt']
    assert result['answer']['evidence_kind'] == 'single_execution'
    assert result['answer']['observed_candidate']['metrics']['mae'] == 1.
    assert 'do not claim a winner' in str(review.call_args.kwargs['feedback_messages']).lower()


def test_single_mcqa_rejects_a_news_summary_instead_of_a_letter():
    log = single_log()
    result = log[0]['output'][0]['result']
    result['answer'] = 'A news summary about economic activity.'
    result['execution_log'] = [{'tool':'llm_reasoning_qa','success':True,'output':{'answer':result['answer']}}]
    log[-1]['args']['prediction'] = result['answer']
    log[-1]['observation']['metrics'] = {'accuracy':0.}
    assert ExecutorAgent._single_execution_candidate(log,task_type='qa',scope_keys=['finance_qa_short_mcqa']) is None
