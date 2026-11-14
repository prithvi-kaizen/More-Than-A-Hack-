"""Replay external fold checkpoints, probabilities and all policy metrics locally."""
import json
from pathlib import Path
import pickle
import sys

import numpy as np
import torch
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from mtah.external import manifest, file_digest
from mtah.decoder import decode
from mtah.motion import apply_policy
from mtah.temporal import load_motion, predict_motion
from run_phase4 import load_cache, diagnostics
from run_phase5 import verify_base_cache, evaluate


def verify_motion_cache(cache, base_cache, settings, source_digest):
    record = json.loads((cache/'provenance.json').read_text())
    fingerprint = record['fingerprint']
    if (fingerprint['base'] != file_digest(base_cache/'provenance.json') or
            fingerprint['settings'] != settings or fingerprint['source'] != source_digest or
            fingerprint['scenario_source'] != file_digest(ROOT/'scripts/run_phase4.py')):
        raise ValueError('PHASE5_CACHE_PROVENANCE_FAILED')
    for name, digest in record['hashes'].items():
        if Path(name).name != name or file_digest(cache/name) != digest:
            raise ValueError('PHASE5_CACHE_INTEGRITY_FAILED')


def main():
    out = ROOT/'outputs/phase5'
    report = json.loads((out/'report.json').read_text())
    settings = report['settings']
    base_settings = json.loads((ROOT/'configs/phase4.json').read_text())
    config = manifest(ROOT/'configs/external_split.json')
    for name, digest in report['source_hashes'].items():
        if file_digest(ROOT/name) != digest:
            raise ValueError('PHASE5_SOURCE_INTEGRITY_FAILED')
    if file_digest(ROOT/'uv.lock') != report['lock_sha256']:
        raise ValueError('PHASE5_DEPENDENCY_LOCK_CHANGED')
    folds_verified = 0
    for match, fold in report['cv_folds'].items():
        if match not in config['split']['train']:
            raise ValueError('UNAUTHORIZED_REPLAY_MATCH')
        _, _, base_cache = verify_base_cache(match, 'train', config)
        folder = out/'oof'/match
        if any(file_digest(folder/name) != fold[key] for name, key in
               [('tree.pkl', 'tree_sha256'), ('motion.pt', 'motion_sha256')]):
            raise ValueError('PHASE5_CHECKPOINT_INTEGRITY_FAILED')
        state = torch.load(folder/'motion.pt', map_location='cpu', weights_only=True)
        if state['training_matches'] != fold['training_matches'] or match in state['training_matches']:
            raise ValueError('PHASE5_CHECKPOINT_SPLIT_LEAKAGE')
        with (folder/'tree.pkl').open('rb') as stream:
            tree = pickle.load(stream)
        neural = load_motion(folder/'motion.pt')
        motion_cache = out/'cache'/match
        verify_motion_cache(motion_cache, base_cache, settings, report['source_hashes']['src/mtah/motion.py'])
        truth = {}
        predictions = {v: {s: {} for s in settings['scenarios']} for v in settings['variants']}
        for half in (1, 2):
            with np.load(base_cache/f'half_{half}_labels.npz', allow_pickle=False) as labels:
                truth[half] = labels['labels']
            for scenario in settings['scenarios']:
                name = f'half_{half}_{scenario}.npz'
                a, o, b = load_cache(base_cache, half, scenario)
                p, h = tree.predict(a, o, b)
                with np.load(motion_cache/name, allow_pickle=False) as inputs:
                    q = inputs['reliability']
                    motion = predict_motion(neural, inputs['features'], len(p),
                        settings['neural']['sample_stride'], settings['neural']['inference_chunk'])
                with np.load(folder/name, allow_pickle=False) as saved:
                    for key, value in [('tree_probabilities', p), ('boundary_scores', h),
                                       ('reliability', q), ('motion_home', motion)]:
                        np.testing.assert_array_equal(value, saved[key])
                    for variant in settings['variants']:
                        pp, hh = apply_policy(p, h, q, variant, motion,
                            settings['motion_max_weight'], settings['missing_geometry_logit_scale'])
                        labels = decode(pp, base_settings['switch_penalty'], base_settings['dead_penalty'],
                            hh, base_settings['boundary_reduction'])
                        sp, sh = apply_policy(saved['tree_probabilities'], saved['boundary_scores'],
                            saved['reliability'], variant, saved['motion_home'],
                            settings['motion_max_weight'], settings['missing_geometry_logit_scale'])
                        replay = decode(sp, base_settings['switch_penalty'], base_settings['dead_penalty'],
                            sh, base_settings['boundary_reduction'])
                        np.testing.assert_array_equal(labels, replay)
                        predictions[variant][scenario][half] = labels
        for variant, scenarios in predictions.items():
            for scenario, halves in scenarios.items():
                if diagnostics(truth, halves) != fold['results'][variant][scenario]:
                    raise ValueError('PHASE5_FOLD_METRIC_REPLAY_FAILED')
        folds_verified += 1
        print(match+'_checkpoint_replay: PASS', flush=True)

    checkpoint = ROOT/'outputs/phase4/model.pkl'
    if file_digest(checkpoint) != report['frozen_phase4_sha256']:
        raise ValueError('FROZEN_PHASE4_CHECKPOINT_MISMATCH')
    with checkpoint.open('rb') as stream:
        tree = pickle.load(stream)
    neural = None
    if report['selected_policy'] == 'motion_fusion':
        if file_digest(out/'motion.pt') != report['full_temporal_sha256']:
            raise ValueError('FULL_TEMPORAL_CHECKPOINT_MISMATCH')
        state = torch.load(out/'motion.pt', map_location='cpu', weights_only=True)
        if state['training_matches'] != config['split']['train']:
            raise ValueError('FULL_TEMPORAL_CHECKPOINT_SPLIT_LEAKAGE')
        neural = load_motion(out/'motion.pt')
    for match, expected in report['development'].items():
        _, _, base_cache = verify_base_cache(match, 'development', config)
        verify_motion_cache(out/'cache'/match, base_cache, settings, report['source_hashes']['src/mtah/motion.py'])
        actual = evaluate(match, tree, neural, out/'cache'/match, base_cache,
            list(expected), settings, base_settings)
        if actual != expected:
            raise ValueError('PHASE5_DEVELOPMENT_METRIC_REPLAY_FAILED')
    (out/'checkpoint_audit.json').write_text(json.dumps(dict(
        folds_verified=folds_verified, probability_equality='PASS', label_equality='PASS',
        all_policy_fold_metrics='PASS', development_metrics='PASS'), indent=2))
    print('Phase 5 checkpoint, probability, label and metric replay: PASS')


if __name__ == '__main__':
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    with threadpool_limits(limits=2):
        main()
