"""Record the selected artifacts before final-holdout acquisition or scoring."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from mtah.external import file_digest, manifest


def main():
    selected = json.loads((ROOT/'configs/phase5_selected.json').read_text())
    config = manifest(ROOT/'configs/external_split.json')
    hashes = dict(selected['source_hashes'])
    for name in ['configs/phase5_selected.json', 'configs/phase4_selected.json',
                 'configs/phase3_selected.json', 'configs/phase5.json',
                 'configs/external_split.json', 'uv.lock']:
        hashes[name] = file_digest(ROOT/name)
    if hashes['uv.lock'] != selected['dependency_lock_sha256']:
        raise ValueError('SELECTED_DEPENDENCY_MISMATCH')
    if hashes['configs/phase4_selected.json'] != selected['base']['config_sha256']:
        raise ValueError('SELECTED_DECODER_CONFIG_MISMATCH')
    for model in [selected['base'], selected['temporal']]:
        hashes[model['checkpoint']] = model['sha256']
    previous = json.loads((ROOT/'docs/phase3_results.json').read_text())
    hashes['outputs/phase3/model.pkl'] = previous['model_sha256']
    for name, digest in hashes.items():
        if file_digest(ROOT/name) != digest:
            raise ValueError('SELECTED_ARTIFACT_MISMATCH')
    record = dict(schema=1, selected=selected, hashes=hashes,
        settings=dict(seed=2026, holdout_seed_offset=2000,
            scenarios=['clean','moderate','severe','structured'],
            primary_scenarios=['clean','moderate','severe'],
            comparisons=['phase3','phase4','motion_fusion'],
            target_source='tactical', outside_half_label='X',
            holdouts=config['split']['holdout'], tuning_after_holdouts=False))
    path = ROOT/'configs/phase6_frozen.json'
    if path.exists():
        old = json.loads(path.read_text())
        if any(old[k] != v for k, v in record.items()):
            raise ValueError('PHASE6_FREEZE_ALREADY_EXISTS_WITH_DIFFERENT_ARTIFACTS')
    else:
        record['created_utc'] = datetime.now(timezone.utc).isoformat()
        record['selection_commit'] = subprocess.check_output(
            ['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip()
        path.write_text(json.dumps(record, indent=2)+'\n')
    print('Pre-holdout selection freeze: PASS')


if __name__ == '__main__':
    main()
