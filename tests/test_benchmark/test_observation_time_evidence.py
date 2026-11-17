"""Read the actual Parquet entrypoint; timestamps may change only with bound evidence."""
import hashlib,json
import numpy as np
import pandas as pd
import pytest
from timeclaw.core.config import TimeClawConfig
from timeclaw.benchmark.mtbench_adapter import DATASET_REGISTRY,load_dataset,observed_input_identity
from timeclaw.benchmark.source_admission import sample_source
from timeclaw.data_integrity import DataIntegrityError


def key(s):
    return hashlib.sha256(json.dumps([observed_input_identity(s),s.metadata['output_timestamps']],sort_keys=True,separators=(',',':')).encode()).hexdigest()


def make(tmp_path,*,dataset='finance_aligned_long',future=3.,reverse=False):
    cfg=TimeClawConfig();cfg.benchmark.repo_root=str(tmp_path)
    folder=tmp_path/DATASET_REGISTRY[dataset]['path'];folder.mkdir(parents=True,exist_ok=True)
    pf=folder/'data/records.parquet';pf.parent.mkdir(exist_ok=True)
    rows=[dict(input_timestamps=[10.,20.],input_window=[1.,1.,2.],output_timestamps=[30.],output_window=[future,future],text='Visible observed news',technical=json.dumps({'out_macd':[future,future]}),trend=json.dumps({'output_bin_label':'>+4%'})),dict(input_timestamps=[50.,60.],input_window=[5.,6.],output_timestamps=[70.],output_window=[7.],text='Other observed news')]
    pd.DataFrame(rows[::-1] if reverse else rows).to_parquet(pf,index=False)
    raw=load_dataset(dataset,config=cfg);s=next(s for s in raw if len(s.input_window)==3)
    evidence={'schema':'observation_time_evidence_v1','datasets':{dataset:{'record_count':2,'source_files':{'data/records.parquet':hashlib.sha256(pf.read_bytes()).hexdigest()},'entries':[{'input_key':key(s),'match_count':1,'input_timestamps':[10.,10.,20.],'output_timestamps':[30.,30.],'source_evidence':[{'repository':'public/raw','revision':'a'*40,'file':'source.parquet','row':0}]}]}}}
    path=tmp_path/'evidence.json';path.write_text(json.dumps(evidence));cfg.benchmark.observation_time_evidence='evidence.json'
    return cfg,path,pf,evidence


def test_actual_loader_restores_duplicate_time_correspondence_without_value_change(tmp_path):
    cfg,path,pf,evidence=make(tmp_path);samples=load_dataset('finance_aligned_long',config=cfg);s=samples[0]
    assert s.timestamps==[10.,10.,20.] and s.metadata['output_timestamps']==[30.,30.]
    assert s.input_window.tolist()==[1.,1.,2.] and s.output_window.tolist()==[3.,3.]
    assert s.text=='Visible observed news' and s.labels['out_macd']==[3.,3.]
    assert samples[1].timestamps==[50.,60.]
    assert sample_source(s)['start']==10. and sample_source(s)['end']==30.
    assert hashlib.sha256(pf.read_bytes()).hexdigest()==evidence['datasets']['finance_aligned_long']['source_files']['data/records.parquet']


@pytest.mark.parametrize('case',['wrong_file','unknown_input','invented_time','wrong_length','duplicate_entry','wrong_count','wrong_schema'])
def test_actual_loader_rejects_unbound_or_invalid_source_evidence(tmp_path,case):
    cfg,path,pf,d=make(tmp_path);scope=d['datasets']['finance_aligned_long'];entry=scope['entries'][0]
    if case=='wrong_file':scope['source_files']['data/records.parquet']='0'*64
    elif case=='unknown_input':entry['input_key']='0'*64
    elif case=='invented_time':entry['input_timestamps']=[10.,11.,20.]
    elif case=='wrong_length':entry['input_timestamps']=[10.,20.]
    elif case=='duplicate_entry':scope['entries'].append(dict(entry))
    elif case=='wrong_count':entry['match_count']=2
    elif case=='wrong_schema':d['schema']='unknown'
    path.write_text(json.dumps(d))
    with pytest.raises(DataIntegrityError,match='evidence|timestamp|source'):
        load_dataset('finance_aligned_long',config=cfg)


def test_raw_profile_keeps_original_data_and_strict_admission(tmp_path):
    cfg,path,pf,d=make(tmp_path);cfg.benchmark.observation_time_evidence=None
    s=load_dataset('finance_aligned_long',config=cfg)[0]
    assert s.timestamps==[10.,20.]
    with pytest.raises(DataIntegrityError,match='timestamps'):sample_source(s)


def test_prediction_prompt_does_not_depend_on_evaluator_values_or_record_order(tmp_path):
    from timeclaw.benchmark.mtbench_prompts import finance_aligned_long_forecast_prompt
    a,_,_,ad=make(tmp_path/'a');b,_,_,bd=make(tmp_path/'b',future=9999.,reverse=True)
    sa=next(s for s in load_dataset('finance_aligned_long',config=a) if len(s.input_window)==3)
    sb=next(s for s in load_dataset('finance_aligned_long',config=b) if len(s.input_window)==3)
    assert ad['datasets']['finance_aligned_long']['entries']==bd['datasets']['finance_aligned_long']['entries']
    assert sa.file!=sb.file and sa.output_window.tolist()!=sb.output_window.tolist()
    assert sa.timestamps==sb.timestamps and finance_aligned_long_forecast_prompt(sa)==finance_aligned_long_forecast_prompt(sb)


def test_time_evidence_is_not_bound_to_one_dataset_name(tmp_path):
    cfg,path,pf,d=make(tmp_path,dataset='finance_aligned_short')
    s=load_dataset('finance_aligned_short',config=cfg)[0]
    assert len(s.timestamps)==len(s.input_window)==3


def test_declared_source_identity_and_time_correspondence_share_the_real_loader(tmp_path):
    cfg,path,pf,d=make(tmp_path)
    d['datasets']['finance_aligned_long']['entries'][0]['source_group']='physical-series-A'
    path.write_text(json.dumps(d))
    sample=load_dataset('finance_aligned_long',config=cfg)[0]
    assert sample.metadata['source_group']=='physical-series-A'
    assert sample_source(sample)['group']=='physical-series-A'
    assert sample.timestamps==[10.,10.,20.]


def test_source_identity_attachment_cannot_overwrite_known_series(tmp_path):
    cfg,path,pf,d=make(tmp_path)
    frame=pd.read_parquet(pf);frame['source_group']=['physical-series-A','physical-series-C'];frame.to_parquet(pf,index=False)
    scope=d['datasets']['finance_aligned_long'];scope['source_files']['data/records.parquet']=hashlib.sha256(pf.read_bytes()).hexdigest()
    scope['entries'][0]['source_group']='physical-series-B';path.write_text(json.dumps(d))
    with pytest.raises(DataIntegrityError,match='source identity'):
        load_dataset('finance_aligned_long',config=cfg)
