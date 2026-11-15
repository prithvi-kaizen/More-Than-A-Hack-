"""Verify the shared inference wrapper using the existing development caches only."""
import json
from pathlib import Path
import sys

import numpy as np
import torch
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from mtah.external import manifest
from mtah.inference import FrozenPredictor
from mtah.phase4 import boundary_features
from run_phase4 import diagnostics
from run_phase5 import verify_base_cache


def main():
    predictor = FrozenPredictor(ROOT)
    config = manifest(ROOT/'configs/external_split.json')
    report = json.loads((ROOT/'outputs/phase5/report.json').read_text())
    for match in config['split']['development']:
        _,_,base = verify_base_cache(match,'development',config)
        motion = ROOT/'outputs/phase5/cache'/match
        for scene in report['settings']['scenarios']:
            truth,pred = {},{}
            for half in (1,2):
                with np.load(base/f'half_{half}_{scene}_inputs.npz',allow_pickle=False) as data:
                    a,o = data['alive_features'],data['owner_features']
                with np.load(motion/f'half_{half}_{scene}.npz',allow_pickle=False) as data:
                    features = dict(alive=a,owner=o,boundary=boundary_features(a,o),
                                    motion=data['features'],reliability=data['reliability'])
                with np.load(base/f'half_{half}_labels.npz',allow_pickle=False) as data:
                    truth[half] = data['labels']
                pred[half] = predictor.predict(features)
            if diagnostics(truth,pred) != report['development'][match]['motion_fusion'][scene]:
                raise ValueError('FROZEN_WRAPPER_DEVELOPMENT_REPLAY_FAILED')
    path = ROOT/'outputs/phase6/development_replay.json'
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(dict(frozen_wrapper_metrics='PASS',new_model_selection=False),indent=2))
    print('Frozen inference wrapper development replay: PASS')


if __name__ == '__main__':
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    with threadpool_limits(limits=2): main()
