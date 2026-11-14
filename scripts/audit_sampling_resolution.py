"""Descriptive training-only audit; does not select or modify a model."""
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from mtah.external import manifest
from mtah.prepared import verify_prepared, read_half
from mtah.official import metrics as pm


def resolution_counts(labels, stride=5):
    codes = pm.label_codes(labels)
    counts, missed = Counter(), 0
    for a, b, _ in pm.possession_spells(codes, pm.valid_runs(codes)):
        length = b-a
        if length < 50:
            counts['under_0.2s' if length < 5 else '0.2_to_1s' if length < 25 else '1_to_2s'] += 1
            missed += ((a+stride-1)//stride)*stride >= b
    return counts, missed


def main():
    config = manifest(ROOT/'configs/external_split.json')
    counts, missed = Counter(), 0
    for match in config['split']['train']:
        folder, _, _ = verify_prepared(ROOT, config, match, 'train')
        for half in (1, 2):
            _, labels, _ = read_half(folder, half)
            c, n = resolution_counts(labels)
            counts.update(c); missed += n
    out = ROOT/'outputs/phase4'
    out.mkdir(parents=True, exist_ok=True)
    (out/'resolution_audit.json').write_text(json.dumps(dict(training_short_spell_counts=dict(counts),
        with_no_frame_selected_by_stride5=missed, used_for_model_selection=False), indent=2))
    print('Training-only sampling-resolution audit: PASS')


if __name__ == '__main__': main()
