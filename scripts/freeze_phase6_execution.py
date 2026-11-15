"""Freeze tested execution code before any final-holdout performance is opened."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from mtah.external import file_digest
from mtah.inference import verify_freeze


def main():
    verify_freeze(ROOT,execution=False)
    files = [*sorted((ROOT/'src/mtah').glob('*.py')),
             *sorted((ROOT/'scripts').glob('*.py')),
             *sorted((ROOT/'vendor/mtah').rglob('*.py')),
             ROOT/'configs/phase6_frozen.json']
    record = dict(hashes={str(p.relative_to(ROOT)):file_digest(p) for p in files})
    path = ROOT/'configs/phase6_execution_frozen.json'
    if path.exists():
        old = json.loads(path.read_text())
        if old['hashes'] != record['hashes']:
            raise ValueError('PHASE6_EXECUTION_FREEZE_CHANGED')
    else:
        record['created_utc'] = datetime.now(timezone.utc).isoformat()
        path.write_text(json.dumps(record,indent=2)+'\n')
    print('Pre-evaluation execution freeze: PASS')


if __name__ == '__main__':
    main()
