"""Acquire training/development only and compress verified raw XML to save disk."""
import gzip
import fcntl
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from mtah.external import manifest, verify_raw
from download_external import download
from prepare_external import prepare


def compress_raw(path, entry):
    compressed = path.with_suffix('.xml.gz')
    temporary = compressed.with_suffix('.gz.part')
    with path.open('rb') as source, temporary.open('wb') as output:
        with gzip.GzipFile(filename='', mode='wb', fileobj=output, mtime=0) as stream:
            shutil.copyfileobj(source, stream)
    temporary.replace(compressed)
    if not verify_raw(compressed, entry):
        raise ValueError('COMPRESSED_INTEGRITY_FAILED')
    path.unlink()


def main():
    config = manifest(ROOT / 'configs/external_split.json')
    for match in config['split']['train'] + config['split']['development']:
        entries = [e for e in config['files'] if e['match_id'] == match]
        for entry in entries:
            for attempt in range(4):
                try:
                    download(entry, mirror=True, workers=24)
                    break
                except (subprocess.CalledProcessError, ValueError) as error:
                    if isinstance(error, ValueError) and str(error) not in (
                            'EXTERNAL_RANGE_TRANSFER_INCOMPLETE', 'EXTERNAL_CHECKSUM_MISMATCH'):
                        raise
                    if attempt == 3:
                        raise
                    print(match + '_transfer_retry: ' + str(attempt+1), flush=True)
                    time.sleep(2)
        prepare(match, config)
        entry = next(e for e in entries if e['kind'] == 'positions')
        raw = ROOT / 'external_data/idsse/raw' / entry['name']
        if raw.exists():
            compress_raw(raw, entry)
        print(match + '_verified_prepared_compressed: PASS', flush=True)


if __name__ == '__main__':
    lock_path = ROOT / 'external_data/idsse/acquisition.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('ACQUISITION_ALREADY_RUNNING') from None
        main()
