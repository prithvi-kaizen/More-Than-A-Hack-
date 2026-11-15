"""Score frozen models once on the reserved external games; never fit or select."""
import argparse
import fcntl
import json
from pathlib import Path
import pickle
import sys

import numpy as np
import torch
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from mtah.external import file_digest, manifest
from mtah.holdout import verify_holdout
from mtah.inference import FrozenPredictor, feature_bundle, verify_freeze
from mtah.prepared import read_half
from mtah.decoder import decode
from run_phase4 import scenarios, diagnostics


def evaluate(replay=False):
    frozen = verify_freeze(ROOT)
    settings = frozen['settings']
    config = manifest(ROOT/'configs/external_split.json')
    predictor = FrozenPredictor(ROOT)
    with (ROOT/'outputs/phase3/model.pkl').open('rb') as stream:
        phase3 = pickle.load(stream)
    decoder3 = json.loads((ROOT/'configs/phase3_selected.json').read_text())
    out = ROOT/'outputs/phase6/holdouts'
    out.mkdir(parents=True,exist_ok=True)
    report_path = out/'report.json'
    if report_path.exists() and not replay:
        raise ValueError('HOLDOUT_RESULTS_ALREADY_EXIST_USE_REPLAY')
    original = json.loads(report_path.read_text()) if replay else None
    all_results, cache_hashes = {}, {}
    for index,match in enumerate(settings['holdouts']):
        folder,pitch,provenance = verify_holdout(ROOT,config,match)
        truth = {}
        pred = {v:{s:{} for s in settings['scenarios']} for v in settings['comparisons']}
        cache = out/match
        cache.mkdir(exist_ok=True)
        for half in (1,2):
            inputs,truth[half],_ = read_half(folder,half)
            for scene in settings['scenarios']:
                path = cache/f'half_{half}_{scene}_features.npz'
                predicted_path = cache/f'half_{half}_{scene}_predictions.npz'
                if replay:
                    for item in (path,predicted_path):
                        key = str(item.relative_to(out))
                        if file_digest(item) != original['cache_hashes'][key]:
                            raise ValueError('HOLDOUT_CACHE_INTEGRITY_FAILED')
                    with np.load(path,allow_pickle=False) as stream:
                        features = dict(stream)
                else:
                    p,b,t = scenarios(inputs,scene,settings['seed']+settings['holdout_seed_offset']+index*10+half,pitch)
                    features = feature_bundle(p,b,t,int(inputs['fps']),pitch)
                    np.savez_compressed(path,**features)
                a,o,h = features['alive'],features['owner'],features['boundary']
                p3 = phase3.predict_proba(a,o[:,:15])
                p4,h4 = predictor.tree.predict(a,o,h)
                pp,hh = predictor.probabilities(features)
                values = dict(phase3=decode(p3,decoder3['switch_penalty'],decoder3['dead_penalty']),
                    phase4=predictor.decode(p4,h4),motion_fusion=predictor.decode(pp,hh))
                if replay:
                    with np.load(predicted_path,allow_pickle=False) as saved:
                        for variant,labels in values.items():
                            np.testing.assert_array_equal(labels,saved[variant])
                        np.testing.assert_array_equal(pp,saved['probabilities'])
                        np.testing.assert_array_equal(hh,saved['boundary'])
                else:
                    np.savez_compressed(predicted_path,**values,probabilities=pp,boundary=hh)
                for variant,labels in values.items():
                    pred[variant][scene][half] = labels
                for item in (path,predicted_path):
                    cache_hashes[str(item.relative_to(out))] = file_digest(item)
                print(match+f'_half_{half}_'+scene+': PASS',flush=True)
        all_results[match] = {v:{s:diagnostics(truth,halves) for s,halves in cases.items()}
                              for v,cases in pred.items()}
    means = {v:dict(mean_spell_f1=float(np.mean([r[v][s]['spell_f1_mean']
                for r in all_results.values() for s in settings['primary_scenarios']])),
                structured_spell_f1=float(np.mean([r[v]['structured']['spell_f1_mean'] for r in all_results.values()])))
             for v in settings['comparisons']}
    if replay:
        if all_results != original['results'] or means != original['means']:
            raise ValueError('FINAL_HOLDOUT_METRIC_REPLAY_FAILED')
        (out/'replay.json').write_text(json.dumps(dict(probabilities='PASS',labels='PASS',metrics='PASS'),indent=2))
        print('Final holdout checkpoint/label/metric replay: PASS')
    else:
        report = dict(results=all_results,means=means,cache_hashes=cache_hashes,
            selection_freeze_sha256=file_digest(ROOT/'configs/phase6_frozen.json'),
            execution_freeze_sha256=file_digest(ROOT/'configs/phase6_execution_frozen.json'),
            selected_policy_unchanged='motion_fusion',fitting_performed=False,
            credit=config['credit'],license=config['license'])
        report_path.write_text(json.dumps(report,indent=2))
        print('Final external evaluation complete; no model selection or fitting performed.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--replay',action='store_true')
    args = parser.parse_args()
    lock_path = ROOT/'outputs/phase6/holdouts.lock'
    lock_path.parent.mkdir(parents=True,exist_ok=True)
    with lock_path.open('w') as lock:
        try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: raise SystemExit('HOLDOUT_EVALUATION_ALREADY_RUNNING') from None
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        with threadpool_limits(limits=2): evaluate(args.replay)
