"""External-only validity experiment; deliberately separate from frozen inference."""
import warnings
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from mtah.corruption import corrupt_inputs, SCENARIOS
from mtah.external import FROZEN_SPLIT
from mtah.motion import geometry_reliability
from mtah.phase4 import velocity
from mtah.structured_noise import structured_inputs


def training_partition(held):
    if held not in FROZEN_SPLIT['train']:
        raise ValueError('UNAUTHORIZED_RELIABILITY_FOLD')
    remaining = [m for m in FROZEN_SPLIT['train'] if m != held]
    return remaining[:-1], remaining[-1]


def stress_inputs(players, ball, teams, fps, seed, pitch, scene, settings):
    """No possession/status labels, events or target statistics enter this transform."""
    p, b, t = np.array(players, copy=True), np.array(ball, copy=True), np.array(teams, copy=True)
    rng = np.random.default_rng(seed)
    n = len(b)
    if scene == 'clean': return p, b, t
    if scene in SCENARIOS:
        p, b = corrupt_inputs(p, b, t, fps, seed, SCENARIOS[scene])
    elif scene == 'structured':
        return structured_inputs(p, b, t, fps, seed, pitch)
    elif scene == 'sustained':
        for _ in range(rng.poisson(settings['false_tracks_per_minute']*n/fps/60)):
            start = int(rng.integers(n))
            stop = min(n, start+round(rng.uniform(*settings['false_track_seconds'])*fps))
            angle = rng.uniform(0, 2*np.pi)
            offset = rng.uniform(*settings['false_offset_metres'])*np.array([np.cos(angle),np.sin(angle)])
            b[start:stop] += offset
    elif scene.startswith('outage_'):
        seconds = int(scene.split('_')[1])
        if seconds not in (2, 8, 20): raise ValueError('UNKNOWN_OUTAGE')
        for _ in range(rng.poisson(settings['outages_per_minute']*n/fps/60)):
            start = int(rng.integers(n)); b[start:min(n,start+seconds*fps)] = np.nan
    elif scene in ('camera_independent', 'camera_ball'):
        # Paired camera tests differ only in the crop centre's dependence on the ball.
        width = settings['camera_half_width_metres']
        for start in range(0,n,settings['camera_blocks_seconds']*fps):
            stop = min(n,start+settings['camera_blocks_seconds']*fps)
            centre = rng.uniform(width,pitch[0]-width)
            if scene == 'camera_ball' and np.isfinite(ball[start,0]): centre = float(ball[start,0])
            missing = np.abs(players[start:stop,:,0]-centre) > width
            p[start:stop][missing] = np.nan
        # Each block independently permutes same-team slots, simulating broken identities.
        for side in ('H','A'):
            slots = np.flatnonzero(t == side)
            for start in range(0,n,settings['camera_blocks_seconds']*fps):
                stop = min(n,start+settings['camera_blocks_seconds']*fps)
                p[start:stop,slots] = p[start:stop,rng.permutation(slots)]
        duplicates, dt = [], []
        for side in ('H','A'):
            slots = np.flatnonzero(t == side)
            for _ in range(settings['camera_duplicate_slots_per_team']):
                duplicate = np.full((n,2), np.nan, dtype=p.dtype)
                for start in range(0,n,settings['camera_blocks_seconds']*fps):
                    stop=min(n,start+fps)
                    duplicate[start:stop] = p[start:stop,int(rng.choice(slots))]
                duplicates.append(duplicate); dt.append(side)
        p = np.concatenate((p,np.stack(duplicates,axis=1)),axis=1); t = np.r_[t,dt]
    else: raise ValueError('UNKNOWN_STRESS_SCENARIO')
    return p,b,t


def validity_features(players, ball, fps=25, pitch=(105,68)):
    """Team/slot invariant geometry and compatibility, without clean reference inputs."""
    visible = np.isfinite(ball).all(axis=1)
    q = geometry_reliability(ball,fps)
    pv,bv = velocity(players,fps),velocity(ball,fps)
    distances = np.linalg.norm(players-ball[:,None],axis=-1)
    ordered = np.sort(np.where(np.isfinite(distances),distances,np.inf),axis=1)
    nearest = ordered[:,0]; second=ordered[:,min(1,ordered.shape[1]-1)]
    rel = np.linalg.norm(pv-bv[:,None],axis=-1)
    compatibility = np.exp(-np.minimum(distances,100)/2)*np.exp(-np.minimum(rel,100)/4)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        best = np.nanmax(compatibility,axis=1)
        speed = np.linalg.norm(bv,axis=-1)
        edge = np.minimum.reduce((ball[:,0],pitch[0]-ball[:,0],ball[:,1],pitch[1]-ball[:,1]))
        x=np.column_stack((visible,q,np.minimum(nearest,100),np.minimum(second,100),
            (distances<2).sum(axis=1),(distances<5).sum(axis=1),best,speed,edge,
            np.isfinite(players).all(axis=-1).sum(axis=1)))
    x[~np.isfinite(x)]=np.nan
    return x.astype(np.float32)


def validity_targets(original, observed, tolerance):
    known=np.isfinite(original).all(axis=1)
    good=np.isfinite(observed).all(axis=1) & (np.linalg.norm(original-observed,axis=1)<=tolerance)
    return np.where(known,good.astype(np.int8),-1)


class ValidityModel:
    def fit(self, fitting, calibration, fit_matches, calibration_match, settings):
        if (not fit_matches or len(set(fit_matches))!=len(fit_matches) or
                not set(fit_matches).issubset(FROZEN_SPLIT['train']) or
                calibration_match not in FROZEN_SPLIT['train'] or calibration_match in fit_matches):
            raise ValueError('RELIABILITY_CALIBRATION_SPLIT_LEAKAGE')
        if set(fitting)!=set(fit_matches) or set(calibration)!={calibration_match}:
            raise ValueError('RELIABILITY_ARRAY_SPLIT_MISMATCH')
        def combine(records):
            x=np.concatenate([v[0] for v in records.values()]);y=np.concatenate([v[1] for v in records.values()])
            valid=y>=0
            if set(y[valid])!={0,1}:raise ValueError('RELIABILITY_CLASSES_MISSING')
            return x[valid],y[valid]
        x,y=combine(fitting); cx,cy=combine(calibration)
        self.model=HistGradientBoostingClassifier(**settings['classifier']).fit(x,y)
        self.calibrator=LogisticRegression(C=1,random_state=2026).fit(self.logits(cx)[:,None],cy)
        self.fit_matches=list(fit_matches);self.calibration_match=calibration_match
        return self

    def logits(self,x):
        p=np.clip(self.model.predict_proba(x)[:,1],1e-6,1-1e-6)
        return np.log(p/(1-p))

    def quality(self,x,original_q):
        p=self.calibrator.predict_proba(self.logits(x)[:,None])[:,1]
        return np.minimum(original_q,p)
