"""External manifest guards and an IDSSE adapter; no target-data dependencies."""
from dataclasses import dataclass
import hashlib
import gzip
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

MATCHES = {'J03WMX', 'J03WN1', 'J03WOH', 'J03WOY', 'J03WPY', 'J03WQQ', 'J03WR9'}
SECTIONS = {'firstHalf': 1, 'secondHalf': 2}
FROZEN_SPLIT = {'train': ['J03WMX', 'J03WN1', 'J03WOH', 'J03WOY'],
                'development': ['J03WPY'], 'holdout': ['J03WQQ', 'J03WR9']}


def manifest(path):
    data = json.loads(Path(path).read_text())
    splits = data['split']
    ids = [m for group in splits.values() for m in group]
    if splits != FROZEN_SPLIT or len(ids) != 7 or set(ids) != MATCHES:
        raise ValueError('INVALID_EXTERNAL_SPLIT')
    if data['license'] != 'CC-BY-4.0' or data['fps'] != 25:
        raise ValueError('INVALID_EXTERNAL_PROVENANCE')
    entries = data['files']
    keys = [(e['match_id'], e['kind']) for e in entries]
    expected = {(m, k) for m in MATCHES for k in ('metadata', 'events', 'positions')}
    if len(keys) != 21 or set(keys) != expected:
        raise ValueError('INVALID_EXTERNAL_CATALOGUE')
    for e in entries:
        if Path(e['name']).name != e['name'] or e['match_id'] not in e['name']:
            raise ValueError('INVALID_EXTERNAL_PATH')
        if not e['url'].startswith('https://ndownloader.figshare.com/files/'):
            raise ValueError('UNAPPROVED_EXTERNAL_SOURCE')
    return data


def label_from_status(status, owner):
    # Unknown ownership is not the away team, and unknown status is not dead ball.
    if status == '0':
        return 'D', 0
    if status == '1':
        return {'1': 'H', '2': 'A'}.get(owner, 'X'), 1
    return 'X', -1


@dataclass
class ExternalPeriod:
    frame_ids: np.ndarray
    players: np.ndarray
    ball: np.ndarray
    teams: np.ndarray
    labels: np.ndarray
    alive: np.ndarray
    fps: int = 25


def metadata(path, match_id):
    root = ET.parse(path).getroot()
    general = root.find('.//General')
    if general is None or general.get('MatchId') != 'DFL-MAT-' + match_id:
        raise ValueError('EXTERNAL_MATCH_ID_MISMATCH')
    environment = root.find('.//Environment')
    pitch = float(environment.get('PitchX')), float(environment.get('PitchY'))
    if not all(0 < x < 200 for x in pitch):
        raise ValueError('INVALID_EXTERNAL_PITCH')
    player_map = {}
    roles = []
    for team in root.findall('.//Teams/Team'):
        role = team.get('Role')
        if role not in ('home', 'guest'):
            raise ValueError('UNKNOWN_EXTERNAL_TEAM_ROLE')
        roles.append(role)
        for player in team.findall('./Players/Player'):
            key = player.get('PersonId')
            if not key or key in player_map:
                raise ValueError('DUPLICATE_EXTERNAL_PLAYER')
            player_map[key] = 'H' if role == 'home' else 'A'
    if sorted(roles) != ['guest', 'home']:
        raise ValueError('INVALID_EXTERNAL_TEAMS')
    return pitch, player_map


def framesets(path):
    """Bound memory to one object/period rather than a whole match XML tree."""
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rb') as stream:
        for _, element in ET.iterparse(stream, events=('end',)):
            if element.tag == 'FrameSet':
                yield element
                element.clear()


def load_idsse(meta_path, positions_path, match_id):
    if match_id not in MATCHES:
        raise ValueError('EXTERNAL_MATCH_ID_MISMATCH')
    pitch, people = metadata(meta_path, match_id)
    player_ids = sorted(people)
    columns = {p: i for i, p in enumerate(player_ids)}
    bounds, balls = {}, {}
    # Pass 1 retains every observed time range, even frames lacking a ball record.
    for fs in framesets(positions_path):
        if fs.get('MatchId') != 'DFL-MAT-' + match_id:
            raise ValueError('EXTERNAL_MATCH_ID_MISMATCH')
        period = SECTIONS.get(fs.get('GameSection'))
        if period is None:
            raise ValueError('UNKNOWN_EXTERNAL_PERIOD')
        is_ball = fs.get('TeamId') == 'BALL'
        if not is_ball and fs.get('PersonId') not in people:
            continue  # Officials are not players.
        previous = None
        for frame in fs.findall('Frame'):
            n = int(frame.get('N'))
            if previous is not None and n <= previous:
                raise ValueError('UNORDERED_EXTERNAL_FRAMESET')
            previous = n
            a, b = bounds.get(period, (n, n))
            bounds[period] = min(a, n), max(b, n)
            if is_ball:
                if n in balls.setdefault(period, {}):
                    raise ValueError('DUPLICATE_BALL_FRAME')
                balls[period][n] = (frame.get('X'), frame.get('Y'),
                                    frame.get('BallStatus'), frame.get('BallPossession'))
    result = {}
    shift = np.array(pitch, dtype=np.float32) / 2
    def xy(x, y):
        return [float(x) if x is not None else np.nan, float(y) if y is not None else np.nan]
    for period, (first, last) in sorted(bounds.items()):
        if last - first > 250_000:
            raise ValueError('EXTERNAL_PERIOD_TOO_LONG')
        n = last - first + 1
        ball = np.full((n, 2), np.nan, dtype=np.float32)
        labels = np.full(n, 'X', dtype='<U1')
        alive = np.full(n, -1, dtype=np.int8)
        for frame, (x, y, status, owner) in balls.get(period, {}).items():
            ball[frame-first] = xy(x, y)
            labels[frame-first], alive[frame-first] = label_from_status(status, owner)
        result[period] = ExternalPeriod(np.arange(first, last+1),
            np.full((n, len(player_ids), 2), np.nan, dtype=np.float32),
            ball + shift, np.array([people[p] for p in player_ids]), labels, alive)
    seen = {p: np.zeros(d.players.shape[:2], dtype=bool) for p, d in result.items()}
    for fs in framesets(positions_path):
        player = fs.get('PersonId')
        if fs.get('TeamId') == 'BALL' or player not in columns:
            continue
        period = SECTIONS[fs.get('GameSection')]
        data = result[period]
        for frame in fs.findall('Frame'):
            n = int(frame.get('N')) - int(data.frame_ids[0])
            if seen[period][n, columns[player]]:
                raise ValueError('DUPLICATE_PLAYER_FRAME')
            seen[period][n, columns[player]] = True
            data.players[n, columns[player]] = xy(frame.get('X'), frame.get('Y'))
    for data in result.values():
        data.players += shift
        if np.isinf(data.players).any() or np.isinf(data.ball).any():
            raise ValueError('INFINITE_EXTERNAL_COORDINATE')
    return result


def file_digest(path, algorithm='sha256'):
    h = hashlib.new(algorithm)
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def verify_raw(path, entry):
    h, size = hashlib.md5(), 0
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
            size += len(chunk)
    return size == entry['size'] and h.hexdigest() == entry['md5']
