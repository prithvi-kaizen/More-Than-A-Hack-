"""Build a deterministic, allowlisted release containing external-trained models only."""
import gzip
import io
import json
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from mtah.external import file_digest
from mtah.inference import verify_freeze

TAG = 'models-v1'
ASSET = 'external-checkpoints-v1.tar.gz'


def main():
    frozen = verify_freeze(ROOT)
    checkpoints = {n:h for n,h in frozen['hashes'].items() if n.startswith('outputs/')}
    expected = {'outputs/phase3/model.pkl','outputs/phase4/model.pkl','outputs/phase5/motion.pt'}
    if set(checkpoints) != expected:
        raise ValueError('UNEXPECTED_CHECKPOINT_RELEASE_SET')
    members = dict(checkpoints)
    members['MODEL_CARD.md'] = file_digest(ROOT/'docs/MODEL_CARD.md')
    members['MODEL_LICENSE.md'] = file_digest(ROOT/'docs/MODEL_LICENSE.md')
    out = ROOT/'outputs/phase7/release'
    out.mkdir(parents=True,exist_ok=True)
    path = out/ASSET
    with path.open('wb') as raw:
        with gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as zipped:
            with tarfile.open(fileobj=zipped,mode='w') as archive:
                for name in sorted(members):
                    source = ROOT/name if name in checkpoints else ROOT/'docs'/name
                    content = source.read_bytes()
                    info = tarfile.TarInfo(name)
                    info.size,info.mode,info.mtime = len(content),0o644,0
                    archive.addfile(info,io.BytesIO(content))
    # Hash the compressed archive, not the transparent-gzip digest used for raw XML.
    import hashlib
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    record = dict(tag=TAG,asset=ASSET,license='CC-BY-4.0',
        url=f'https://github.com/prithvi-kaizen/More-Than-A-Hack-/releases/download/{TAG}/{ASSET}',
        archive_sha256=sha,archive_bytes=path.stat().st_size,
        members=members,checkpoints=checkpoints,compatible_runtime_commit='59991894aed6595476f15a64602fa447fa89a324')
    manifest = ROOT/'configs/checkpoint_release.json'
    if manifest.exists() and json.loads(manifest.read_text()) != record:
        raise ValueError('EXISTING_MODEL_RELEASE_MANIFEST_DIFFERS')
    manifest.write_text(json.dumps(record,indent=2)+'\n')
    print('External-only checkpoint archive and manifest: PASS')


if __name__ == '__main__':
    main()
