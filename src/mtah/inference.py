"""Frozen, geometry-only inference shared by external testing and the target adapter."""
import json
from pathlib import Path
import pickle
import warnings

import numpy as np

from mtah.decoder import decode
from mtah.external import file_digest, FROZEN_SPLIT
from mtah.motion import player_motion_features, geometry_reliability, apply_policy
from mtah.phase4 import control_features, boundary_features


def verify_hashes(root, hashes):
    root = Path(root).resolve()
    for name, expected in hashes.items():
        path = Path(name)
        if path.is_absolute() or '..' in path.parts or not path.parts:
            raise ValueError('UNSAFE_FROZEN_PATH')
        if file_digest(root/path) != expected:
            raise ValueError('FROZEN_ARTIFACT_INTEGRITY_FAILED')


def verify_freeze(root, execution=True):
    root = Path(root)
    record = json.loads((root/'configs/phase6_frozen.json').read_text())
    verify_hashes(root, record['hashes'])
    if execution:
        code = json.loads((root/'configs/phase6_execution_frozen.json').read_text())
        verify_hashes(root, code['hashes'])
    return record


def feature_bundle(players, ball, teams, fps, pitch):
    if fps != 25:
        raise ValueError('FROZEN_MODEL_REQUIRES_25_FPS')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        a, o = control_features(players, ball, teams, fps, pitch)
        motion = player_motion_features(players, teams, fps, pitch)[::5]
    return dict(alive=a, owner=o, boundary=boundary_features(a,o,fps),
                motion=motion, reliability=geometry_reliability(ball,fps))


class FrozenPredictor:
    def __init__(self, root, execution=True):
        from mtah.temporal import load_motion
        import torch
        root = Path(root)
        self.record = verify_freeze(root, execution)
        self.selected = self.record['selected']
        if self.selected['policy'] != 'motion_fusion':
            raise ValueError('UNSUPPORTED_FROZEN_POLICY')
        with (root/self.selected['base']['checkpoint']).open('rb') as stream:
            self.tree = pickle.load(stream)
        self.neural = load_motion(root/self.selected['temporal']['checkpoint'])
        state = torch.load(root/self.selected['temporal']['checkpoint'],map_location='cpu',weights_only=True)
        if state['training_matches'] != FROZEN_SPLIT['train']:
            raise ValueError('FROZEN_TEMPORAL_TRAINING_PROVENANCE_FAILED')
        if self.neural.input_dim != 108 or self.neural.channels != self.selected['temporal']['channels']:
            raise ValueError('FROZEN_TEMPORAL_ARCHITECTURE_MISMATCH')
        self.decoder = json.loads((root/self.selected['base']['config']).read_text())

    def probabilities(self, features):
        from mtah.temporal import predict_motion
        p, h = self.tree.predict(features['alive'],features['owner'],features['boundary'])
        settings = self.selected['temporal']
        motion = predict_motion(self.neural,features['motion'],len(p),
                                settings['sample_stride'],settings['inference_chunk'])
        fusion = self.selected['fusion']
        return apply_policy(p,h,features['reliability'],self.selected['policy'],motion,
                            fusion['motion_max_weight'],fusion['missing_geometry_logit_scale'])

    def decode(self, probabilities, boundary):
        c = self.decoder
        return decode(probabilities,c['switch_penalty'],c['dead_penalty'],boundary,c['boundary_reduction'])

    def predict(self, features):
        return self.decode(*self.probabilities(features))
