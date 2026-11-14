"""Replay selected development predictions and audit classes and short spells."""
import json
from pathlib import Path
import pickle
import sys
import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from mtah.external import file_digest, manifest
from mtah.prepared import read_half, verify_prepared
from mtah.features import extract_features
from mtah.decoder import decode
from mtah.validation import evaluate_periods
from mtah.official import metrics as pm
from train_phase3 import scenario_inputs


def main():
    folder = ROOT / 'outputs/phase3'
    report = json.loads((folder / 'report.json').read_text())
    if file_digest(folder / 'model.pkl') != report['model_sha256']:
        raise ValueError('MODEL_CHECKPOINT_INTEGRITY_FAILED')
    with (folder / 'model.pkl').open('rb') as stream:
        model = pickle.load(stream)
    config = manifest(ROOT / 'configs/external_split.json')
    audit = {}
    for match in config['split']['development']:
        data_folder, pitch, _ = verify_prepared(ROOT, config, match, 'development')
        truth, predictions = {}, {name: {} for name in report['settings']['scenarios']}
        for half in (1, 2):
            inputs, labels, _ = read_half(data_folder, half)
            truth[half] = labels
            for name in predictions:
                p, b = scenario_inputs(inputs, name, report['settings']['seed']+1000+half)
                a, o = extract_features(p, b, inputs['teams'], int(inputs['fps']), pitch)
                predictions[name][half] = decode(model.predict_proba(a, o),
                    report['selected']['switch_penalty'], report['selected']['dead_penalty'])
        for name, halves in predictions.items():
            result = evaluate_periods(truth, halves)
            expected = report['selected']['results'][match+'_'+name]
            for metric in ('spell_f1_mean', 'macro_f1', 'n_gt_spells', 'n_pred_spells'):
                if not np.isclose(result[metric], expected[metric], atol=1e-12, rtol=0):
                    raise ValueError('DEVELOPMENT_REPLAY_MISMATCH')
            gt = pm.label_codes(np.concatenate([np.append(truth[h], 'X') for h in (1, 2)]))
            pred = pm.label_codes(np.concatenate([np.append(halves[h], 'X') for h in (1, 2)]))
            runs = pm.valid_runs(gt)
            spells, predicted = pm.possession_spells(gt, runs), pm.possession_spells(pred, runs)
            flags = pm.matched_gt_flags(pm.spell_overlaps(spells, predicted), len(spells), 0.5)
            durations = np.array([(b-a)/25 for a, b, _ in spells])
            recall = {}
            for lo, hi in [(0, 2), (2, 5), (5, 15), (15, np.inf)]:
                mask = (durations >= lo) & (durations < hi)
                recall[f'{lo}_{hi}_seconds'] = dict(support=int(mask.sum()),
                    recall=float(flags[mask].mean()) if mask.any() else None)
            audit[match+'_'+name] = dict(
                alive_dead_f1=result['alive_dead_f1'],
                team_accuracy_on_alive=result['team_acc_gt_alive'],
                per_class=result['per_class'],
                transition_f1={str(t): result['transitions'][t]['f1'] for t in (12, 25)},
                possession_recall_iou_05_by_duration=recall)
    (folder / 'audit.json').write_text(json.dumps(audit, indent=2))
    print('Full development checkpoint replay and diagnostic audit: PASS')


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
