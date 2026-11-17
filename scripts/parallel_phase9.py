"""Schedule the same frozen Phase 9 folds concurrently, then run canonical finalization."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import warnings
import numpy as np
import torch
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import run_phase9 as run


def verify_worker_protocol(settings):
    # The parent alone writes the protocol. Worker verification is read-only.
    frozen=json.loads((run.OUT/'protocol_frozen.json').read_text())
    if frozen['settings']!=settings:raise ValueError('WORKER_SETTINGS_CHANGED')
    run.verify_freeze(ROOT)
    for path,digest in frozen['sources'].items():
        if run.file_digest(ROOT/path)!=digest:raise ValueError('WORKER_SOURCE_CHANGED')
    return json.loads((ROOT/'docs/phase5_results.json').read_text())


def worker(match):
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    with warnings.catch_warnings(),threadpool_limits(limits=2):
        warnings.simplefilter('ignore',RuntimeWarning)
        settings=json.loads((ROOT/'configs/phase9.json').read_text())
        previous=verify_worker_protocol(settings)
        training={}
        for m in run.FROZEN_SPLIT['train']:
            with np.load(run.OUT/'training'/f'{m}.npz',allow_pickle=False) as f:
                training[m]=(f['features'],f['targets'])
        fits,cal=run.training_partition(match)
        model=run.ValidityModel().fit({m:training[m] for m in fits},{cal:training[cal]},fits,cal,settings)
        folder=ROOT/'external_data/idsse/prepared'/match
        # The serial preparation already verified raw publisher checksums. Recheck
        # its exact prepared arrays before parallel access, including labels.
        provenance=json.loads((run.OUT/'provenance'/f'{match}.json').read_text())
        for name,digest in provenance['prepared_hashes'].items():
            if run.file_digest(folder/name)!=digest:raise ValueError('PREPARED_ARRAY_CHANGED')
        config=run.manifest(ROOT/'configs/external_split.json')
        entry=next(e for e in config['files'] if e['match_id']==match and e['kind']=='metadata')
        from mtah.external import metadata
        pitch,_=metadata(ROOT/'external_data/idsse/raw'/entry['name'],match)
        tree,neural=run.fold_predictor(match,previous)
        result=run.evaluate_match(match,folder,pitch,tree,neural,model,settings)
        run.save(run.OUT/'folds'/f'{match}.json',result)
    return match


def main():
    run.OUT.mkdir(parents=True,exist_ok=True)
    with (run.OUT/'run.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        settings=json.loads((ROOT/'configs/phase9.json').read_text())
        run.freeze_protocol(settings)
        execution=dict(scheduler='four independent processes; two CPU threads each',
            scheduler_sha256=run.file_digest(Path(__file__)),settings_changed=False,
            frozen_sources_changed=False,repair='parent-only protocol writes; read-only workers',
            previous_execution_record='parallel_execution.json')
        record=run.OUT/'parallel_execution_v2.json'
        if record.exists() and json.loads(record.read_text())!=execution:raise ValueError('SCHEDULER_CHANGED')
        run.save(record,execution)
        # An interrupted npz write is an execution artifact, never a scoring result.
        for path in (run.OUT/'predictions').rglob('*.npz'):
            with np.load(path,allow_pickle=False) as f:
                for policy in run.POLICIES:_=f[policy]
        with ProcessPoolExecutor(max_workers=4) as pool:
            jobs=[pool.submit(worker,m) for m in run.FROZEN_SPLIT['train']]
            for job in as_completed(jobs):
                match=job.result()
                print(match+'_parallel_fold: PASS',flush=True)
    # Canonical main refits/reloads checkpoints and exactly verifies all cached metrics.
    subprocess.run([sys.executable,str(ROOT/'scripts/run_phase9.py')],cwd=ROOT,check=True)


if __name__=='__main__':main()
