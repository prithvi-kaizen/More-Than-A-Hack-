"""Acquire the two reserved external matches after selection has been frozen."""
import fcntl
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from mtah.external import manifest
from acquire_phase3 import compress_raw
from download_external import download
from prepare_external import prepare
from freeze_phase6 import main as verify_selection_freeze


def main():
    verify_selection_freeze()
    config = manifest(ROOT/'configs/external_split.json')
    for match in config['split']['holdout']:
        entries = [e for e in config['files'] if e['match_id'] == match]
        for entry in entries:
            for attempt in range(4):
                try:
                    download(entry, mirror=True, workers=24)
                    break
                except (subprocess.CalledProcessError, ValueError):
                    if attempt == 3:
                        raise
                    print(match+'_transfer_retry', flush=True)
                    time.sleep(2)
        prepare(match, config)
        entry = next(e for e in entries if e['kind'] == 'positions')
        raw = ROOT/'external_data/idsse/raw'/entry['name']
        if raw.exists():
            compress_raw(raw, entry)
        print(match+'_verified_prepared_compressed: PASS', flush=True)


if __name__ == '__main__':
    path = ROOT/'external_data/idsse/acquisition.lock'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('ACQUISITION_ALREADY_RUNNING') from None
        main()
