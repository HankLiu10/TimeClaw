"""Apply declared source-time metadata without changing values or source support.

An observation-time evidence file is a versioned data attachment, not a learned
policy. It binds complete input files to audited raw-source correspondences
and physical-series identities.
Record lookup uses observed inputs and public future times, never future values.
"""
from collections import Counter
import hashlib
import json
from numbers import Real
from pathlib import Path
import re
import numpy as np
from timeclaw.data_integrity import DataIntegrityError


def input_key(sample):
    from timeclaw.benchmark.mtbench_adapter import observed_input_identity
    return hashlib.sha256(json.dumps(
        [observed_input_identity(sample), sample.metadata.get('output_timestamps', [])],
        sort_keys=True, separators=(',', ':'),
    ).encode()).hexdigest()


def _times(value, original, expected):
    if not isinstance(value, list) or any(not isinstance(t, Real) or isinstance(t, bool) for t in value):
        raise DataIntegrityError('Observation-time evidence timestamps must be numeric lists')
    new=np.asarray(value,dtype=float)
    if not len(new) or len(new)!=expected or not np.isfinite(new).all():
        raise DataIntegrityError('Observation-time evidence timestamp length/finite contract differs')
    if not original or set(new.tolist())!=set(original) or new[0]!=original[0] or new[-1]!=original[-1]:
        raise DataIntegrityError('Observation-time evidence cannot invent, remove or move source-time support')
    return new.tolist()


def apply_observation_time_evidence(samples, *, dataset, directory, source_files, evidence_file):
    """Verify the declared attachment and apply it atomically to normalized rows."""
    path=Path(evidence_file)
    try:
        raw=path.read_bytes();doc=json.loads(raw)
    except (OSError,ValueError) as exc:
        raise DataIntegrityError(f'Cannot read observation-time evidence: {path}') from exc
    if not isinstance(doc,dict) or doc.get('schema')!='observation_time_evidence_v1' or not isinstance(doc.get('datasets'),dict):
        raise DataIntegrityError('Unsupported observation-time evidence schema')
    scope=doc['datasets'].get(dataset)
    if scope is None:
        return  # The declared attachment does not modify this dataset.
    if not isinstance(scope,dict) or scope.get('record_count')!=len(samples):
        raise DataIntegrityError('Observation-time evidence source record count differs')
    root=Path(directory)
    actual={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    expected=scope.get('source_files')
    if not isinstance(expected,dict) or not expected or actual!=expected:
        raise DataIntegrityError('Observation-time evidence source files differ from the bound dataset')
    entries=scope.get('entries')
    if not isinstance(entries,list) or not entries:
        raise DataIntegrityError('Observation-time evidence needs explicit entries')
    records={}
    for entry in entries:
        if not isinstance(entry,dict):raise DataIntegrityError('Invalid observation-time evidence entry')
        key=entry.get('input_key');count=entry.get('match_count');proof=entry.get('source_evidence')
        if not isinstance(key,str) or re.fullmatch('[0-9a-f]{64}',key) is None or key in records:
            raise DataIntegrityError('Duplicate or invalid observation-time evidence input key')
        if type(count) is not int or count<1 or not isinstance(proof,list) or not proof:
            raise DataIntegrityError('Observation-time evidence needs source proof and positive match count')
        for source in proof:
            if (not isinstance(source,dict) or any(not isinstance(source.get(k),str) or not source[k] for k in ['repository','revision','file'])
                    or type(source.get('row')) is not int or source['row']<0):
                raise DataIntegrityError('Invalid observation-time evidence raw source identity')
        allowed={'input_key','match_count','input_timestamps','output_timestamps','source_evidence','source_group'}
        if set(entry)-allowed or not ({'input_timestamps','output_timestamps'}<=set(entry) or 'source_group' in entry):
            raise DataIntegrityError('Unsupported observation-time evidence fields')
        if 'source_group' in entry and (not isinstance(entry['source_group'],str) or not entry['source_group'].strip()):
            raise DataIntegrityError('Invalid observation-time evidence physical source identity')
        records[key]=entry
    changes=[];counts=Counter()
    for sample in samples:
        key=input_key(sample)
        if key not in records:continue
        entry=records[key];counts[key]+=1
        if 'input_timestamps' in entry or 'output_timestamps' in entry:
            observed=_times(entry.get('input_timestamps'),sample.timestamps,len(sample.input_window))
            output=_times(entry.get('output_timestamps'),sample.metadata.get('output_timestamps',[]),
                          0 if sample.output_window is None else len(sample.output_window))
        else:
            observed=list(sample.timestamps);output=list(sample.metadata.get('output_timestamps',[]))
        group=entry.get('source_group')
        if group is not None and sample.metadata.get('source_group') not in (None,group):
            raise DataIntegrityError('Observation-time evidence conflicts with declared physical source identity')
        changes.append((sample,observed,output,key,group))
    if any(counts[key]!=entry['match_count'] for key,entry in records.items()):
        raise DataIntegrityError('Observation-time evidence input matches/counts differ from the bound source')
    digest=hashlib.sha256(raw).hexdigest()
    for sample,observed,output,key,group in changes:
        sample.timestamps=observed
        sample.metadata['output_timestamps']=output
        if group is not None:sample.metadata['source_group']=group
        sample.metadata['observation_time_evidence']={'schema':doc['schema'],'sha256':digest,'input_key':key}
