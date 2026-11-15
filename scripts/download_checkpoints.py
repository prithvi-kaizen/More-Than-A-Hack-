"""Fetch the public external-trained checkpoint release; no training data is needed."""
import json
from pathlib import Path
import sys
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from mtah.checkpoints import installed,unpack_verified


def main():
    manifest = json.loads((ROOT/'configs/checkpoint_release.json').read_text())
    if installed(manifest,ROOT):
        print('Pinned external checkpoints already installed: PASS')
        return
    if not manifest['url'].startswith('https://github.com/prithvi-kaizen/More-Than-A-Hack-/releases/download/'):
        raise ValueError('UNEXPECTED_CHECKPOINT_RELEASE_URL')
    cache = ROOT/'outputs/checkpoint_download'
    cache.mkdir(parents=True,exist_ok=True)
    # Use the host certificate store, as the external-data downloader does.
    with tempfile.TemporaryDirectory(dir=cache) as temporary:
        path = Path(temporary)/'checkpoints.tar.gz'
        subprocess.run(['curl','--fail','--location','--silent','--show-error',
            '--retry','2','--max-time','120',manifest['url'],'-o',str(path)],check=True)
        if path.stat().st_size != manifest['archive_bytes']:
            raise ValueError('CHECKPOINT_DOWNLOAD_SIZE_MISMATCH')
        content = path.read_bytes()
    unpack_verified(content,manifest,ROOT)
    if not installed(manifest,ROOT):
        raise ValueError('CHECKPOINT_INSTALLATION_FAILED')
    print('Public checkpoint download, all hashes and installation: PASS')


if __name__ == '__main__':
    main()
