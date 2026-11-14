"""Download the frozen licensed external catalogue; never accesses target inputs."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import shutil
import sys
import re
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from mtah.external import manifest as read_manifest, verify_raw


def checksum(path):
    h = hashlib.md5()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def download(entry, mirror=False, workers=8):
    folder = ROOT / 'external_data/idsse/raw'
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / entry['name']
    compressed = target.with_suffix('.xml.gz')
    if compressed.exists() and verify_raw(compressed, entry):
        return
    if target.exists() and target.stat().st_size == entry['size'] and checksum(target) == entry['md5']:
        return
    temporary = target.with_suffix('.xml.part')
    if mirror:
        parts = folder / (entry['name'] + '.parts')
        parts.mkdir(exist_ok=True)
        chunk_size = 4 * 1024 * 1024
        def part(start):
            stop = min(entry['size'], start + chunk_size) - 1
            path = parts / str(start)
            for attempt in range(8):
                existing = path.stat().st_size if path.exists() else 0
                if existing == stop-start+1:
                    return path
                if existing > stop-start+1:
                    path.unlink()
                    existing = 0
                offset = start + existing
                url = ('https://huggingface.co/datasets/pysport/idsse-data/resolve/'
                       'a715a38/' + entry['name'] + '?download=true&part=' + str(start)
                       + '&offset=' + str(offset))
                tail, headers = parts / (str(start)+'.tail'), parts / (str(start)+'.headers')
                tail.write_bytes(b'')
                subprocess.run(['curl', '--http1.1', '--fail', '--location', '--silent', '--show-error',
                                '--speed-limit', '4096', '--speed-time', '20',
                                '--range', f'{offset}-{stop}', '--max-time', '90',
                                url, '-D', str(headers), '-o', str(tail)], check=False)
                ranges = re.findall(r'content-range:\s*bytes\s+(\d+)-(\d+)/(\d+)',
                                    headers.read_text(errors='replace') if headers.exists() else '', re.I)
                # Even a truncated response can supply a valid prefix; retain it only
                # when the server explicitly confirms the requested byte interval.
                if ranges and tuple(map(int, ranges[-1])) == (offset, stop, entry['size']):
                    size = tail.stat().st_size
                    if 0 < size <= stop-offset+1:
                        with path.open('ab') as output, tail.open('rb') as source:
                            shutil.copyfileobj(source, output)
                tail.unlink(missing_ok=True)
                headers.unlink(missing_ok=True)
                if path.exists() and path.stat().st_size == stop-start+1:
                    return path
                time.sleep(1)
            raise ValueError('EXTERNAL_RANGE_TRANSFER_INCOMPLETE')
        with ThreadPoolExecutor(max_workers=workers) as pool:
            paths = list(pool.map(part, range(0, entry['size'], chunk_size)))
        with temporary.open('wb') as stream:
            for path in paths:
                with path.open('rb') as source:
                    shutil.copyfileobj(source, stream)
        if temporary.stat().st_size != entry['size'] or checksum(temporary) != entry['md5']:
            temporary.unlink(missing_ok=True)
            shutil.rmtree(parts)
            raise ValueError('EXTERNAL_CHECKSUM_MISMATCH')
        temporary.replace(target)
        shutil.rmtree(parts)
        return
    # curl uses the host certificate store; TLS verification stays enabled.
    subprocess.run(['curl', '--fail', '--location', '--silent', '--show-error',
                    '--compressed', '--retry', '2', '--max-time', '1200',
                    entry['url'], '-o', str(temporary)], check=True)
    if temporary.stat().st_size != entry['size'] or checksum(temporary) != entry['md5']:
        raise ValueError('EXTERNAL_CHECKSUM_MISMATCH')
    temporary.replace(target)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--small-only', action='store_true')
    parser.add_argument('--mirror', action='store_true')
    parser.add_argument('--match', choices=['J03WMX', 'J03WN1', 'J03WOH', 'J03WOY', 'J03WPY', 'J03WQQ', 'J03WR9'])
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    manifest = read_manifest(ROOT / 'configs/external_split.json')
    entries = [e for e in manifest['files'] if not args.small_only or e['kind'] != 'positions']
    entries = [e for e in entries if args.match is None or e['match_id'] == args.match]
    if args.workers < 1 or args.workers > 24:
        raise ValueError('INVALID_DOWNLOAD_WORKERS')
    for entry in entries:
        download(entry, mirror=args.mirror, workers=args.workers)
        print(entry['match_id'] + '_' + entry['kind'] + ': verified', flush=True)
    print('External download checksums: PASS')


if __name__ == '__main__':
    main()
