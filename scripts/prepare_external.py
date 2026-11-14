"""Verify and prepare licensed external data; keep match arrays and reports local."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from mtah.external import file_digest, load_idsse, manifest, verify_raw


def prepare(match_id, config):
    entries = {e['kind']: e for e in config['files'] if e['match_id'] == match_id}
    raw = ROOT / 'external_data/idsse/raw'
    paths = {kind: raw / entry['name'] for kind, entry in entries.items()}
    paths = {k: p if p.exists() else p.with_suffix('.xml.gz') for k, p in paths.items()}
    for kind, path in paths.items():
        e = entries[kind]
        if not path.is_file() or not verify_raw(path, e):
            raise ValueError('UNVERIFIED_EXTERNAL_INPUT')
    periods = load_idsse(paths['metadata'], paths['positions'], match_id)
    folder = ROOT / 'external_data/idsse/prepared' / match_id
    folder.mkdir(parents=True, exist_ok=True)
    summary = {}
    if set(periods) != {1, 2}:
        raise ValueError('MISSING_EXTERNAL_HALF')
    for number, period in periods.items():
        if not np.isin(period.alive, [0, 1]).any() or not np.isfinite(period.players).any():
            raise ValueError('UNUSABLE_EXTERNAL_PERIOD')
        # Labels are separate from inputs; callers must not feed alive/labels to a model.
        np.savez_compressed(folder / f'half_{number}_inputs.npz',
            frame_ids=period.frame_ids, players=period.players, ball=period.ball,
            teams=period.teams, fps=period.fps)
        np.savez_compressed(folder / f'half_{number}_labels.npz',
            labels=period.labels, alive=period.alive)
        summary[number] = dict(frames=len(period.labels),
            counts={str(k): int(v) for k, v in zip(*np.unique(period.labels, return_counts=True))},
            unknown_status=int((period.alive == -1).sum()),
            missing_ball=int((~np.isfinite(period.ball).all(axis=1)).sum()))
    provenance = dict(match_id=match_id, source_hashes={k: file_digest(p) for k, p in paths.items()},
                      license=config['license'], credit=config['credit'],
                      adapter_version=1, fps=config['fps'])
    provenance['prepared_hashes'] = {p.name: file_digest(p) for p in folder.glob('*.npz')}
    (folder / 'provenance.json').write_text(json.dumps(provenance, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--match')
    args = parser.parse_args()
    config = manifest(ROOT / 'configs/external_split.json')
    ids = [m for group in config['split'].values() for m in group]
    if args.match and args.match not in ids:
        raise ValueError('UNREGISTERED_EXTERNAL_MATCH')
    ids = [args.match] if args.match else ids
    results = {}
    for match_id in ids:
        results[match_id] = prepare(match_id, config)
        print(match_id + '_preparation: PASS', flush=True)
    report = ROOT / 'local_reports' / ('external_preparation_' + (args.match or 'all') + '.json')
    report.parent.mkdir(exist_ok=True)
    report.write_text(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
