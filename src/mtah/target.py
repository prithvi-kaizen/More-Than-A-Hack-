"""Tactical ATD adapter and exact CSV composition; no labels/video are read."""
import csv
import importlib.util
from pathlib import Path

import numpy as np
from mtah.inputs import iter_tracking, team_membership


def organizer_contract(root):
    path = Path(root)/'vendor/mtah/games.py'
    spec = importlib.util.spec_from_file_location('mtah_pinned_games',path)
    games = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(games)
    game = games.GAMES[games.GAME]
    return dict(n_frames=game['n_video_frames'],fps=game['fps'],pitch=game['pitch'],
                blocks=games.half_video_blocks(games.GAME))


def load_tactical(metadata, tracking, contract):
    teams = np.array(team_membership(Path(metadata)))
    if set(teams) != {'H','A'}:
        raise ValueError('TARGET_TEAMS_MISSING')
    n = contract['n_frames']
    players = np.full((n,len(teams),2),np.nan,dtype=np.float32)
    ball = np.full((n,2),np.nan,dtype=np.float32)
    pitch = np.asarray(contract['pitch'],dtype=np.float64)
    for frame in iter_tracking(Path(tracking),len(teams),n):
        players[frame.video_frame] = np.asarray(frame.players)[:,:2]*pitch
        ball[frame.video_frame] = np.asarray(frame.ball)[:2]*pitch
    return players,ball,teams


def validate_blocks(n_frames, blocks):
    previous = -1
    halves = set()
    for half, start, stop in blocks:
        if (half in halves or any(not isinstance(v,int) or isinstance(v,bool) for v in (half,start,stop))
                or not 0 <= start <= stop < n_frames or start <= previous):
            raise ValueError('INVALID_HALF_WINDOWS')
        previous = stop
        halves.add(half)
    if not halves:
        raise ValueError('MISSING_HALF_WINDOWS')


def compose_submission(predictions, n_frames, blocks):
    validate_blocks(n_frames,blocks)
    if set(predictions) != {half for half,_,_ in blocks}:
        raise ValueError('PREDICTION_HALF_MISMATCH')
    labels = np.full(n_frames,'X',dtype='<U1')
    for half,start,stop in blocks:
        values = np.asarray(predictions[half])
        if values.shape != (stop-start+1,) or not np.isin(values,['H','A','D']).all():
            raise ValueError('INVALID_HALF_PREDICTIONS')
        labels[start:stop+1] = values
    return labels


def predict_halves(predictor, players, ball, teams, contract):
    from mtah.inference import feature_bundle
    validate_blocks(len(ball),contract['blocks'])
    predictions = {}
    for half,start,stop in contract['blocks']:
        features = feature_bundle(players[start:stop+1],ball[start:stop+1],teams,
                                  contract['fps'],contract['pitch'])
        predictions[half] = predictor.predict(features)
    return compose_submission(predictions,len(ball),contract['blocks'])


def write_submission(path, labels):
    labels = np.asarray(labels)
    if labels.ndim != 1 or not np.isin(labels,['H','A','D','X']).all():
        raise ValueError('INVALID_SUBMISSION_LABELS')
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.part')
    path.parent.mkdir(parents=True,exist_ok=True)
    with temporary.open('w',encoding='utf-8',newline='') as stream:
        writer = csv.writer(stream,lineterminator='\n')
        writer.writerow(['frame','label'])
        writer.writerows(enumerate(labels))
    temporary.replace(path)
