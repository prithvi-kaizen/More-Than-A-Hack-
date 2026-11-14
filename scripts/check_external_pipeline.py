"""Diagnostic untrained comparator on training matches only, never model selection."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from mtah.external import file_digest, manifest
from mtah.corruption import SCENARIOS, corrupt_inputs
from mtah.validation import evaluate_periods


def nearest(players, ball, teams):
    distances = np.linalg.norm(players - ball[:, None, :], axis=2)
    distances[~np.isfinite(distances)] = np.inf
    indices = np.argmin(distances, axis=1)
    labels = np.asarray(teams)[indices].copy()
    labels[~np.isfinite(distances.min(axis=1))] = 'D'
    return labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--match', default='J03WMX')
    args = parser.parse_args()
    config = manifest(ROOT / 'configs/external_split.json')
    if args.match not in config['split']['train']:
        raise ValueError('DIAGNOSTIC_REQUIRES_TRAIN_MATCH')
    folder = ROOT / 'external_data/idsse/prepared' / args.match
    provenance = json.loads((folder / 'provenance.json').read_text())
    for name, expected in provenance['prepared_hashes'].items():
        if Path(name).name != name or file_digest(folder / name) != expected:
            raise ValueError('PREPARED_DATA_INTEGRITY_FAILED')
    gt = {}
    predictions_by_scenario = {name: {} for name in ('clean', *SCENARIOS)}
    for half in (1, 2):
        with np.load(folder / f'half_{half}_inputs.npz', allow_pickle=False) as inputs:
            p, b, teams = inputs['players'], inputs['ball'], inputs['teams']
            predictions_by_scenario['clean'][half] = nearest(p, b, teams)
            for name, corruption in SCENARIOS.items():
                cp, cb = corrupt_inputs(p, b, teams, int(inputs['fps']), seed=100+half, config=corruption)
                predictions_by_scenario[name][half] = nearest(cp, cb, teams)
        with np.load(folder / f'half_{half}_labels.npz', allow_pickle=False) as labels:
            gt[half] = labels['labels']
    report = {}
    for name, predictions in predictions_by_scenario.items():
        result = evaluate_periods(gt, predictions)
        report[name] = {key: result[key] for key in ('spell_f1_mean', 'macro_f1', 'n_gt_spells', 'n_pred_spells')}
    (ROOT / 'local_reports' / (args.match + '_diagnostic.json')).write_text(json.dumps(report, indent=2))
    print('External scoring/corruption integration: PASS')


if __name__ == '__main__':
    main()
