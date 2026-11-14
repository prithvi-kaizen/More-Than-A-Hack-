"""Control evidence and capped spell weighting for external-only experiments."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from mtah.features import extract_features, rolling_mean, speed
from mtah.model import PossessionModel

VERSION = 1


def temporal_delta(values, lag):
    values = np.asarray(values)
    result = np.full(values.shape, np.nan)
    if len(values) > 2*lag:
        result[lag:-lag] = values[2*lag:]-values[:-2*lag]
    return result


def velocity(xy, fps):
    lag = max(1, round(0.08*fps))
    v = temporal_delta(rolling_mean(xy, lag), lag)*fps/(2*lag)
    valid = np.isfinite(xy).all(axis=-1)
    # Every observation in the derivative interval must exist.
    count = rolling_mean(valid.astype(float), lag)
    v[count < 1] = np.nan
    return v


def control_features(players, ball, teams, fps=25, pitch=(105, 68)):
    alive, original = extract_features(players, ball, teams, fps, pitch)
    players, ball, teams = np.asarray(players), np.asarray(ball), np.asarray(teams)
    bv, pv = velocity(ball, fps), velocity(players, fps)
    distance = np.linalg.norm(players-ball[:, None], axis=-1)
    sides = []
    for side in ('H', 'A'):
        mask = teams == side
        d = np.where(np.isfinite(distance[:, mask]), distance[:, mask], np.inf)
        index = d.argmin(axis=1)
        nearest = d[np.arange(len(ball)), index]
        relative = np.linalg.norm(bv-pv[:, mask][np.arange(len(ball)), index], axis=-1)
        relative[~np.isfinite(nearest)] = np.nan
        score = np.exp(-nearest/2)*np.exp(-relative/4)
        sides.append((relative, score, nearest))
    with np.errstate(invalid='ignore'):
        sign = np.sign(sides[1][2]-sides[0][2])
    sign[~np.isfinite(sign)] = np.nan
    bs = speed(ball, fps)
    bs[bs > 60] = np.nan
    lag = max(1, round(0.12*fps))
    extra = np.column_stack((
        np.clip(sides[0][0]-sides[1][0], -40, 40),
        sides[0][1]-sides[1][1],
        sign*np.log1p(bs),
        sign*np.abs(temporal_delta(bs, lag)),
        temporal_delta(original[:, 0], lag),
        temporal_delta(original[:, -1], lag)))
    owner = np.column_stack((original, extra)).astype(np.float32)
    owner[~np.isfinite(owner)] = np.nan
    return alive, owner


def boundary_features(alive, owner, fps=25):
    return np.column_stack((alive, np.abs(owner),
        np.abs(temporal_delta(owner, max(1, round(0.2*fps)))))).astype(np.float32)


def spell_weights(labels, fps=25):
    labels = np.asarray(labels)
    result = np.ones(len(labels), dtype=np.float32)
    if not len(labels): return result
    edges = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1])+1, len(labels)]
    for a, b in zip(edges[:-1], edges[1:]):
        if labels[a] in ('H', 'A'):
            result[a:b] = np.clip(np.sqrt(5/max((b-a)/fps, 0.2)), 0.25, 4)
    valid = np.isin(labels, ['H', 'A'])
    if valid.any():
        raw = result[valid].copy()
        lo, hi = 0., 16.
        for _ in range(32):
            mid = (lo+hi)/2
            if np.clip(mid*raw, 0.25, 4).mean() < 1: lo = mid
            else: hi = mid
        result[valid] = np.clip((lo+hi)/2*raw, 0.25, 4)
    return result


def boundary_targets(labels, radius=5):
    labels = np.asarray(labels)
    y = np.zeros(len(labels), dtype=np.int8)
    both = np.isin(labels[:-1], ['H', 'A']) & np.isin(labels[1:], ['H', 'A'])
    for t in np.flatnonzero(both & (labels[1:] != labels[:-1]))+1:
        y[max(0, t-radius):min(len(y), t+radius+1)] = 1
    return y


class Phase4Model:
    def __init__(self, variant='frame', max_iter=120, boundary_positive_weight=20):
        if variant not in ('frame', 'balanced', 'control_adaptive'):
            raise ValueError('UNKNOWN_PHASE4_VARIANT')
        self.variant = variant
        self.boundary_positive_weight = boundary_positive_weight
        self.base = PossessionModel(max_iter)
        self.boundary_model = None

    def fit(self, a, owner, boundary, labels, alive, weights, targets, shared_alive=None):
        live = (alive == 1) & np.isin(labels, ['H', 'A'])
        valid = np.isin(alive, [0, 1])
        if set(labels[live]) != {'H', 'A'} or set(alive[valid]) != {0, 1}:
            raise ValueError('TRAINING_CLASSES_MISSING')
        if shared_alive is None:
            self.base.alive_model.fit(a[valid], alive[valid])
        else:
            self.base.alive_model = shared_alive
        x = owner[live] if self.variant == 'control_adaptive' else owner[live, :15]
        y = (labels[live] == 'H').astype(int)
        w = None if self.variant == 'frame' else np.tile(weights[live], 2)
        self.base.owner_model.fit(np.concatenate((x, -x)), np.r_[y, 1-y], sample_weight=w)
        if self.variant == 'control_adaptive':
            self.boundary_model = HistGradientBoostingClassifier(max_iter=80, max_leaf_nodes=7,
                min_samples_leaf=80, l2_regularization=1, early_stopping=False, random_state=2026)
            if set(targets[live]) != {0, 1}:
                raise ValueError('BOUNDARY_CLASSES_MISSING')
            self.boundary_model.fit(boundary[live], targets[live],
                                    sample_weight=np.where(targets[live] == 1, self.boundary_positive_weight, 1))
        return self

    def predict(self, a, owner, boundary):
        x = owner if self.variant == 'control_adaptive' else owner[:, :15]
        probabilities = self.base.predict_proba(a, x)
        hazard = None if self.boundary_model is None else self.boundary_model.predict_proba(boundary)[:, 1]
        return probabilities, hazard
