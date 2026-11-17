"""Data isolation and sealed ancestry for the native benchmark lifecycle.

A source group identifies a physical series/entity, not a sample filename.
A lineage receipt binds the training corpus and declared evaluation inputs to
one workspace, then seals that workspace's learned files after finalization.
"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from timeclaw.data_integrity import DataIntegrityError
from timeclaw.benchmark.task_registry import TASK_SET, task_registry_digest

RECEIPT = 'training_lineage.json'


@contextmanager
def workspace_access(config, *, write: bool):
    """Hold memory access for a whole public job, including its cleanup.

    Writers (learning/distillation) are exclusive; frozen-inference readers can
    coexist. Receipt checks alone cannot close the race between read and write.
    The persistent empty lock is metadata, not a learned artifact or result.
    OS ownership is released on close/process exit; do not unlink its inode.
    """
    root = Path(config.workspace_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.workspace.lock').open('a') as handle:
        mode = fcntl.LOCK_EX if write else fcntl.LOCK_SH
        try:
            fcntl.flock(handle.fileno(), mode | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise DataIntegrityError(
                f'TimeClaw workspace is busy: {root}. Learning/distillation requires '
                'exclusive access; frozen inference may share readers only.'
            ) from error
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _json(value: Any) -> str:
    def scalar(item):
        if isinstance(item, np.ndarray): return item.tolist()
        if isinstance(item, np.generic): return item.item()
        raise TypeError(f'Unsupported source value: {type(item).__name__}')
    return json.dumps(value,sort_keys=True,separators=(',',':'),default=scalar)


def _digest(value: Any) -> str:
    return hashlib.sha256(_json(value).encode()).hexdigest()


def evaluation_datasets(config) -> list[str]:
    from timeclaw.benchmark.mtbench_adapter import DATASET_REGISTRY
    names=config.benchmark.evaluation_datasets
    names=list(DATASET_REGISTRY) if names is None else list(names)
    if not names or len(set(names)) != len(names) or set(names)-set(DATASET_REGISTRY):
        raise DataIntegrityError('evaluation_datasets must be a nonempty, unique list of registered evaluation pools')
    return names


def _source_identifier(value):
    if value is None or (isinstance(value, (float,np.floating)) and np.isnan(value)):
        return None
    if not isinstance(value,str):
        raise DataIntegrityError('source_group/text_source_id must be strings when supplied')
    return value.strip() or None


def sample_source(sample) -> dict[str, Any]:
    """Public source support includes every observed and supervised time step."""
    observed=list(sample.timestamps)
    future=list(sample.metadata.get('output_timestamps') or [])
    if len(observed) != len(sample.input_window) or not observed:
        raise DataIntegrityError(f'Missing/unaligned input timestamps for source isolation: {sample.file}')
    if sample.output_window is not None and len(sample.output_window) != len(future):
        raise DataIntegrityError(f'Missing/unaligned output timestamps for source isolation: {sample.file}')
    times=np.asarray(observed+future,dtype=float)
    if not np.isfinite(times).all():
        raise DataIntegrityError(f'Non-finite source timestamps: {sample.file}')
    return {'domain':sample.domain,'group':_source_identifier(sample.metadata.get('source_group')),
            'text_source_id':_source_identifier(sample.metadata.get('text_source_id')),
            'start':float(times.min()),'end':float(times.max())}


def sample_digest(sample, *, targets: bool) -> str:
    payload={'file':sample.file,'dataset':sample.dataset,'input':sample.input_window,
        'timestamps':sample.timestamps,'text':sample.text,'question':sample.question,
        'source':sample_source(sample),'horizon':sample.forecast_horizon,
        'output_timestamps':sample.metadata.get('output_timestamps') or []}
    if targets:
        payload.update(output=sample.output_window,answer=sample.answer,labels=sample.labels)
    return _digest(payload)


def evaluation_snapshot(config):
    from timeclaw.benchmark.mtbench_adapter import load_dataset
    pools={name:load_dataset(name,config=config) for name in evaluation_datasets(config)}
    if any(not samples for samples in pools.values()):
        raise DataIntegrityError('A declared evaluation pool is empty; download/repair it before learning')
    bindings={name:_digest(sorted(sample_digest(s,targets=False) for s in samples))
              for name,samples in pools.items()}
    return pools,bindings


def admit_independent_samples(samples, config) -> dict[str, str]:
    from timeclaw.benchmark.mtbench_adapter import observed_input_identity
    pools,bindings=evaluation_snapshot(config)
    training=[(s,observed_input_identity(s),sample_source(s)) for s in samples]
    for name,records in pools.items():
        for record in records:
            identity=observed_input_identity(record)
            official=sample_source(record)
            for sample,train_id,source in training:
                if identity == train_id:
                    raise DataIntegrityError(f'Independent training overlaps official evaluation input: {sample.file}, {name}/{record.file}')
                if source['text_source_id'] and source['text_source_id']==official['text_source_id']:
                    raise DataIntegrityError(f'Training and official evaluation reuse the same text source: {sample.file}, {record.file}')
                intersects=source['start']<=official['end'] and official['start']<=source['end']
                if source['domain']==official['domain'] and intersects:
                    distinct=source['group'] and official['group'] and source['group']!=official['group']
                    if not distinct:
                        raise DataIntegrityError(
                            f'Overlapping source-time support with official evaluation: {sample.file}, {name}/{record.file}. '
                            'Rebuild disjoint windows, or supply true source_group identities for distinct physical series.')
    for sample,identity,source in training:
        sample.metadata['training_source_admission']={
            'observed_input_sha256':identity,'evaluation_datasets_checked':list(pools),
            'evaluation_bindings':bindings,'source_support':source,
            'coverage':'all_declared_evaluation_pools'}
    return bindings


def _artifact_hashes(workspace: Path) -> dict[str,str]:
    """Only learning-owned files; request logs and inference outputs stay separate."""
    result={}
    for folder in ('hierarchical_memory','train_results'):
        if (workspace/folder).is_symlink():
            raise DataIntegrityError(f'Learning artifact root must not be symlinked: {workspace/folder}')
        for p in sorted((workspace/folder).rglob('*')):
            if p.is_symlink():
                raise DataIntegrityError(f'Learning artifacts must not be symlinked: {p}')
            if p.is_file(): result[str(p.relative_to(workspace))]=hashlib.sha256(p.read_bytes()).hexdigest()
    return result


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(payload,indent=2)+'\n')
    tmp.replace(path)


def _read_verified(workspace: Path, task_set: str = TASK_SET):
    path=workspace/RECEIPT
    if path.is_symlink():
        raise DataIntegrityError("Training lineage receipt must not be a symlink")
    artifacts=_artifact_hashes(workspace)
    if not path.is_file():
        if artifacts:
            raise DataIntegrityError('Existing memory/checkpoints have no training lineage receipt; rebuild in a fresh workspace, do not relabel old artifacts')
        return None
    receipt=json.loads(path.read_text())
    if receipt.get('schema')!='timeclaw_training_lineage_v1' or receipt.get('task_registry')!=task_registry_digest(task_set) or receipt.get('task_set')!=task_set:
        raise DataIntegrityError('Training lineage schema/task registry differs from the selected task set')
    if receipt.get('artifacts')!=artifacts:
        raise DataIntegrityError('Learning artifacts differ from their sealed lineage; unknown/copied/unfinished memory or checkpoint changes cannot be reused')
    return receipt


def begin_training_lineage(config, source_samples):
    workspace=Path(config.workspace_dir)
    old=_read_verified(workspace, config.benchmark.task_set)
    if old and old.get('state')=='active':
        raise DataIntegrityError('An unfinalized learning process owns this workspace; do not reuse unsealed state')
    # Re-evaluate all declared held-out inputs at the job boundary. No cached
    # positive admission is trusted across dataset/config changes.
    combined=[s for records in source_samples.values() for s in records]
    bindings=admit_independent_samples(combined,config)
    corpora={name:_digest([sample_digest(s,targets=True) for s in records])
             for name,records in source_samples.items()}
    if old:
        if old['evaluation_bindings']!=bindings:
            raise DataIntegrityError('Declared evaluation inputs changed since memory learning; use a new workspace')
        for name,digest in corpora.items():
            if name in old['training_corpora'] and old['training_corpora'][name]!=digest:
                raise DataIntegrityError('Training corpus/order/targets changed under an existing checkpoint; use a new workspace')
        corpora={**old['training_corpora'],**corpora}
    receipt={'schema':'timeclaw_training_lineage_v1','task_set':config.benchmark.task_set,
        'task_registry':task_registry_digest(config.benchmark.task_set),'evaluation_bindings':bindings,
        'training_corpora':corpora,'state':'active','artifacts':_artifact_hashes(workspace)}
    _write(workspace/RECEIPT,receipt)
    return receipt


def begin_distillation_lineage(config):
    workspace=Path(config.workspace_dir)
    receipt=_read_verified(workspace, config.benchmark.task_set)
    if receipt is None:
        raise DataIntegrityError('No admitted training lineage exists for standalone distillation')
    if receipt['state']=='active':
        raise DataIntegrityError('An active learning operation owns this workspace')
    receipt['state']='active'
    _write(workspace/RECEIPT,receipt)
    return receipt


def finish_training_lineage(config, *, finalized: bool):
    workspace=Path(config.workspace_dir)
    path=workspace/RECEIPT
    receipt=json.loads(path.read_text())
    if receipt['state']!='active':
        raise DataIntegrityError('Cannot seal a workspace without an active admitted learning job')
    receipt['state']='finalized' if finalized else 'interrupted'
    receipt['artifacts']=_artifact_hashes(workspace)
    _write(path,receipt)


def assert_inference_lineage(config, datasets, *, disable_memory: bool):
    if disable_memory:
        raise DataIntegrityError("The main experiment requires memory lineage checks; no-memory bypass is not supported")
    receipt=_read_verified(Path(config.workspace_dir), config.benchmark.task_set)
    if receipt is None:
        return None  # Empty workspace: genuine native ReAct without learned files.
    if receipt['state']!='finalized':
        raise DataIntegrityError('Inference requires finalized, sealed training memory')
    for name,samples in datasets.items():
        if name not in receipt['evaluation_bindings']:
            raise DataIntegrityError(f'Evaluation dataset was not declared before learning: {name}')
        actual=_digest(sorted(sample_digest(s,targets=False) for s in samples))
        if actual!=receipt['evaluation_bindings'][name]:
            raise DataIntegrityError(f'Evaluation inputs changed after memory learning: {name}')
    return receipt
