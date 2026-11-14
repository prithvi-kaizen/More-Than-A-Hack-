"""Exact three-state offline Viterbi decoding with no cross-half memory."""
import numpy as np


def decode(probabilities, switch_penalty=4.0, dead_penalty=None,
           boundary_scores=None, boundary_reduction=0.75):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] != 3 or not np.isfinite(probabilities).all():
        raise ValueError('INVALID_STATE_PROBABILITIES')
    if (probabilities < 0).any() or (probabilities.sum(axis=1) <= 0).any():
        raise ValueError('INVALID_STATE_PROBABILITIES')
    if not np.isfinite(switch_penalty) or switch_penalty < 0 or (dead_penalty is not None and
            (not np.isfinite(dead_penalty) or dead_penalty < 0)):
        raise ValueError('INVALID_TRANSITION_PENALTY')
    if not len(probabilities):
        return np.array([], dtype='<U1')
    if not 0 <= boundary_reduction <= 1:
        raise ValueError('INVALID_BOUNDARY_REDUCTION')
    if boundary_scores is not None:
        boundary_scores = np.asarray(boundary_scores)
        if boundary_scores.shape != (len(probabilities),) or not np.isfinite(boundary_scores).all() or (
                (boundary_scores < 0).any() or (boundary_scores > 1).any()):
            raise ValueError('INVALID_BOUNDARY_SCORES')
    p = probabilities/probabilities.sum(axis=1, keepdims=True)
    emission = np.log(np.clip(p, 1e-6, 1))
    transition = np.full((3, 3), -float(switch_penalty))
    if dead_penalty is not None:
        transition[2, :2] = transition[:2, 2] = -float(dead_penalty)
    np.fill_diagonal(transition, 0)
    back = np.zeros((len(p), 3), dtype=np.int8)
    scores = emission[0].copy()
    for t in range(1, len(p)):
        candidates = scores[:, None]+transition
        if boundary_scores is not None:
            turnover = switch_penalty*(1-boundary_reduction*boundary_scores[t])
            candidates[0, 1], candidates[1, 0] = scores[0]-turnover, scores[1]-turnover
        back[t] = candidates.argmax(axis=0)
        scores = candidates[back[t], np.arange(3)]+emission[t]
        scores -= scores.max()
    states = np.empty(len(p), dtype=np.int8)
    states[-1] = scores.argmax()
    for t in range(len(p)-1, 0, -1):
        states[t-1] = back[t, states[t]]
    return np.array(['H', 'A', 'D'])[states]
