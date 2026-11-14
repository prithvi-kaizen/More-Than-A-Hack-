"""Geometry-only control and stoppage features; call once per continuous half."""
import warnings
import numpy as np

FEATURE_VERSION = 1


def rolling_mean(values, radius):
    values = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(values)
    totals = np.concatenate((np.zeros((1, *values.shape[1:])),
                             np.cumsum(np.where(valid, values, 0), axis=0)))
    counts = np.concatenate((np.zeros((1, *values.shape[1:])), np.cumsum(valid, axis=0)))
    indices = np.arange(len(values))
    lo, hi = np.maximum(0, indices-radius), np.minimum(len(values), indices+radius+1)
    count = counts[hi] - counts[lo]
    return np.divide(totals[hi]-totals[lo], count,
                     out=np.full_like(totals[hi], np.nan), where=count>0)


def speed(xy, fps):
    lag = max(1, round(0.2*fps))
    smooth = rolling_mean(xy, max(1, round(0.08*fps)))
    out = np.full(xy.shape[:-1], np.nan)
    if len(xy) > lag:
        out[lag:] = np.linalg.norm(smooth[lag:]-smooth[:-lag], axis=-1)*fps/lag
        valid = np.isfinite(xy).all(axis=-1)
        cumulative = np.concatenate((np.zeros((1, *valid.shape[1:])), np.cumsum(valid, axis=0)))
        count = cumulative[lag+1:]-cumulative[:-(lag+1)]
        out[lag:][count != lag+1] = np.nan
    return out


def carry(values):
    indices = np.maximum.accumulate(np.where(np.isfinite(values), np.arange(len(values)), -1))
    result = np.full(len(values), np.nan)
    valid = indices >= 0
    result[valid] = values[indices[valid]]
    return result, indices


def extract_features(players, ball, teams, fps=25, pitch=(105, 68)):
    players, ball, teams = np.asarray(players), np.asarray(ball), np.asarray(teams)
    n = len(ball)
    if n == 0 or players.shape != (n, len(teams), 2) or ball.shape != (n, 2) or fps <= 0:
        raise ValueError('INVALID_FEATURE_INPUT')
    if not set(teams).issubset({'H', 'A'}) or not all(np.any(teams == t) for t in ('H', 'A')):
        raise ValueError('INVALID_FEATURE_TEAMS')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        distance = np.linalg.norm(players-ball[:, None, :], axis=-1)
        player_speed = speed(players, fps)
        player_speed[player_speed > 12] = np.nan
        ball_speed = speed(ball, fps)
        ball_speed[ball_speed > 60] = np.nan
        groups = []
        for side in ('H', 'A'):
            mask = teams == side
            d = np.where(np.isfinite(distance[:, mask]), distance[:, mask], np.inf)
            ordered = np.sort(d, axis=1)
            nearest = ordered[:, 0].copy()
            second = ordered[:, min(1, d.shape[1]-1)].copy()
            nearest[~np.isfinite(nearest)] = np.nan
            second[~np.isfinite(second)] = np.nan
            ps = player_speed[:, mask]
            xy = players[:, mask]
            groups.append(np.column_stack((
                np.minimum(nearest, 40), np.minimum(second, 40),
                (d < 1.5).sum(axis=1), (d < 4).sum(axis=1),
                np.isfinite(xy).all(axis=-1).sum(axis=1),
                np.nanmedian(ps, axis=1), np.nanquantile(ps, 0.75, axis=1),
                np.nanstd(xy[:, :, 0], axis=1), np.nanstd(xy[:, :, 1], axis=1))))
        home, away = groups
        # Ownership inputs are antisymmetric; flipping teams negates this matrix.
        ownership = [home-away]
        nearest_any = np.fmin(home[:, 0], away[:, 0])
        winner = np.sign(away[:, 0]-home[:, 0])
        winner[(nearest_any > 5) | ~np.isfinite(nearest_any)] = np.nan
        previous, last_index = carry(winner)
        age = np.where(last_index >= 0, np.minimum((np.arange(n)-last_index)/fps, 10), np.nan)
        ownership += [previous[:, None], (previous*age)[:, None]]
        for seconds in (0.4, 1, 2, 4):
            ownership.append(rolling_mean(winner, max(1, round(seconds*fps/2)))[:, None])
        owner_features = np.column_stack(ownership)
        seen = np.isfinite(ball).all(axis=1).astype(float)
        _, last_ball = carry(np.where(seen, 1, np.nan))
        since_ball = np.where(last_ball >= 0, np.minimum((np.arange(n)-last_ball)/fps, 10), np.nan)
        length, width = pitch
        x, y = ball.T
        goal_line = np.minimum(x, length-x)
        touchline = np.minimum(y, width-y)
        dead_features = [seen, since_ball, np.minimum(goal_line, 20),
                         np.minimum(touchline, 20), ball_speed, nearest_any,
                         np.nanmedian(player_speed, axis=1),
                         np.nanquantile(player_speed, 0.75, axis=1),
                         np.isfinite(players).all(axis=-1).sum(axis=1)]
        symmetric_groups = np.column_stack((home+away, np.abs(home-away)))
        base = np.column_stack(dead_features + [symmetric_groups])
        alive_features = [base]
        for seconds in (0.4, 1, 2):
            radius = max(1, round(seconds*fps/2))
            alive_features.append(rolling_mean(base[:, [0, 4, 5, 6, 7]], radius))
        alive_features = np.column_stack(alive_features)
    alive_features[~np.isfinite(alive_features)] = np.nan
    owner_features[~np.isfinite(owner_features)] = np.nan
    return alive_features.astype(np.float32), owner_features.astype(np.float32)
