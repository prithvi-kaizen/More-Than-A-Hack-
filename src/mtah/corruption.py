"""Seeded external-input corruption; labels never enter the transform."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Corruption:
    ball_gap_starts_per_minute: float = 3
    ball_gap_seconds: tuple[float, float] = (0.4, 6.0)
    player_gap_starts_per_minute: float = 0.2
    player_gap_seconds: tuple[float, float] = (0.4, 3.0)
    position_noise_metres: float = 0.15
    track_swaps_per_minute: float = 0.2

    def __post_init__(self):
        rates = (self.ball_gap_starts_per_minute, self.player_gap_starts_per_minute,
                 self.position_noise_metres, self.track_swaps_per_minute)
        durations = (self.ball_gap_seconds, self.player_gap_seconds)
        if any(not np.isfinite(x) or x < 0 for x in rates) or any(
                len(d) != 2 or not 0 < d[0] <= d[1] or not np.isfinite(d).all()
                for d in durations):
            raise ValueError('INVALID_CORRUPTION_CONFIG')


SCENARIOS = {
    'moderate': Corruption(),
    'severe': Corruption(ball_gap_starts_per_minute=12, ball_gap_seconds=(1, 8),
                         player_gap_starts_per_minute=1, position_noise_metres=0.5,
                         track_swaps_per_minute=1),
}


def corrupt_inputs(players, ball, teams, fps, seed, config=Corruption()):
    """Call separately per period; swaps are team-preserving local fragments."""
    rng = np.random.default_rng(seed)
    players, ball = np.array(players, dtype=np.float32, copy=True), np.array(ball, dtype=np.float32, copy=True)
    n = len(ball)
    if fps <= 0 or players.shape != (n, len(teams), 2) or ball.shape != (n, 2):
        raise ValueError('INVALID_CORRUPTION_INPUT')
    if n == 0:
        return players, ball
    def gaps(rate, duration):
        for _ in range(rng.poisson(rate*n/fps/60)):
            start = int(rng.integers(n))
            length = max(1, round(rng.uniform(*duration)*fps))
            yield start, min(n, start+length)
    for a, b in gaps(config.ball_gap_starts_per_minute, config.ball_gap_seconds):
        ball[a:b] = np.nan
    for slot in range(len(teams)):
        for a, b in gaps(config.player_gap_starts_per_minute, config.player_gap_seconds):
            players[a:b, slot] = np.nan
    for side in ('H', 'A'):
        slots = np.flatnonzero(np.asarray(teams) == side)
        if len(slots) < 2:
            continue
        for a, b in gaps(config.track_swaps_per_minute, (0.4, 4)):
            i, j = rng.choice(slots, 2, replace=False)
            players[a:b, [i, j]] = players[a:b, [j, i]]
    players += rng.normal(0, config.position_noise_metres, players.shape)
    ball += rng.normal(0, config.position_noise_metres, ball.shape)
    return players, ball
