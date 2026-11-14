"""Replay local development labels from two independently loaded checkpoints."""
import hashlib
import json
from pathlib import Path
import pickle
import sys
import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from mtah.external import manifest, file_digest
from mtah.prepared import verify_prepared
from mtah.decoder import decode
from run_phase4 import load_cache, diagnostics


def main():
    out = ROOT/'outputs/phase4'
    report = json.loads((out/'report.json').read_text())
    checkpoint = out/'model.pkl'
    if file_digest(checkpoint) != report['model_sha256']:
        raise ValueError('PHASE4_CHECKPOINT_INTEGRITY_FAILED')
    models = [pickle.loads(checkpoint.read_bytes()) for _ in range(2)]
    config = manifest(ROOT/'configs/external_split.json')
    hashes = {}
    settings = report['settings']
    for match in config['split']['development']:
        verify_prepared(ROOT, config, match, 'development')
        cache = out/'cache'/match
        provenance = json.loads((cache/'provenance.json').read_text())
        for name, digest in provenance['hashes'].items():
            if Path(name).name != name or file_digest(cache/name) != digest:
                raise ValueError('PHASE4_CACHE_INTEGRITY_FAILED')
        for scenario in settings['scenarios']:
            truth, predicted = {}, {}
            for half in (1, 2):
                a, o, b = load_cache(cache, half, scenario)
                p, h = models[0].predict(a,o,b)
                pp, hh = models[1].predict(a,o,b)
                np.testing.assert_array_equal(p,pp)
                if h is not None: np.testing.assert_array_equal(h,hh)
                labels = decode(p,settings['switch_penalty'],settings['dead_penalty'],h,settings['boundary_reduction'])
                replay = decode(pp,settings['switch_penalty'],settings['dead_penalty'],hh,settings['boundary_reduction'])
                np.testing.assert_array_equal(labels,replay)
                hashes[f'{match}_{scenario}_{half}'] = hashlib.sha256(labels.tobytes()).hexdigest()
                predicted[half] = labels
                with np.load(cache/f'half_{half}_labels.npz',allow_pickle=False) as f: truth[half] = f['labels']
            if diagnostics(truth,predicted) != report['development'][match]['phase4'][scenario]:
                raise ValueError('PHASE4_DEVELOPMENT_METRIC_REPLAY_FAILED')
    (out/'checkpoint_audit.json').write_text(json.dumps(dict(
        independent_loads=2, probability_equality='PASS', label_byte_equality='PASS',
        development_metric_replay='PASS', label_hashes=hashes),indent=2))
    print('Independent checkpoint probabilities, label bytes and metrics: PASS')


if __name__ == '__main__':
    with threadpool_limits(limits=2): main()
