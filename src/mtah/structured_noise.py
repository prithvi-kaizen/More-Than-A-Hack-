"""Input-only crowd/camera occlusion, duplicate tracks and false ball detections."""
import numpy as np
from mtah.corruption import SCENARIOS, corrupt_inputs


def structured_inputs(players, ball, teams, fps, seed, pitch=(105, 68)):
    players, ball = np.asarray(players), np.asarray(ball)
    p, b = corrupt_inputs(players, ball, teams, fps, seed, SCENARIOS['severe'])
    rng = np.random.default_rng(seed+100000)
    teams = np.array(teams, copy=True)
    n = len(ball)
    if not n:
        return p, b, teams
    distance = np.linalg.norm(np.asarray(players)-np.asarray(ball)[:, None], axis=-1)
    crowd = (distance < 3).sum(axis=1) >= 3
    edge = np.minimum.reduce((ball[:, 0], pitch[0]-ball[:, 0], ball[:, 1], pitch[1]-ball[:, 1])) < 3
    triggers = np.flatnonzero(crowd | edge)
    for _ in range(rng.poisson(2*n/fps/60)):
        if len(triggers):
            a = int(rng.choice(triggers))
            b[a:min(n, a+round(rng.uniform(0.3, 2)*fps))] = np.nan
    for _ in range(rng.poisson(3*n/fps/60)):
        a = int(rng.integers(n)); end = min(n, a+round(rng.uniform(2, 5)*fps))
        invisible = np.abs(players[a:end, :, 0]-ball[a:end, None, 0]) > 35
        p[a:end][invisible] = np.nan
    for _ in range(rng.poisson(3*n/fps/60)):
        a = int(rng.integers(n)); end = min(n, a+int(rng.integers(1, 4)))
        b[a:end] = rng.uniform([0, 0], pitch, (end-a, 2))
    duplicates = []
    duplicate_teams = []
    for team in ('H', 'A'):
        slots = np.flatnonzero(teams == team)
        for _ in range(2):
            duplicate = np.full((n, 2), np.nan, dtype=np.float32)
            if len(slots):
                for _ in range(rng.poisson(n/fps/60)):
                    a = int(rng.integers(n)); end = min(n, a+round(rng.uniform(0.5, 2)*fps))
                    duplicate[a:end] = p[a:end, int(rng.choice(slots))]
            duplicates.append(duplicate)
            duplicate_teams.append(team)
    return np.concatenate((p, np.stack(duplicates, axis=1)), axis=1), b, np.r_[teams, duplicate_teams]
