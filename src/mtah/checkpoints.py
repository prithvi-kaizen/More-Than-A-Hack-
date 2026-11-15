"""Verify every archive member before installing pinned external checkpoints."""
import hashlib
import io
from pathlib import Path
import tarfile

CHECKPOINTS = {'outputs/phase3/model.pkl','outputs/phase4/model.pkl','outputs/phase5/motion.pt'}


def validate_manifest(manifest):
    if (set(manifest['checkpoints']) != CHECKPOINTS or
            set(manifest['members']) != CHECKPOINTS|{'MODEL_CARD.md','MODEL_LICENSE.md'} or
            any(manifest['members'][name] != digest for name,digest in manifest['checkpoints'].items())):
        raise ValueError('INVALID_CHECKPOINT_MANIFEST')


def unpack_verified(content, manifest, root):
    validate_manifest(manifest)
    if len(content) != manifest['archive_bytes'] or hashlib.sha256(content).hexdigest() != manifest['archive_sha256']:
        raise ValueError('CHECKPOINT_ARCHIVE_INTEGRITY_FAILED')
    expected = manifest['members']
    payloads = {}
    with tarfile.open(fileobj=io.BytesIO(content),mode='r:gz') as archive:
        for member in archive.getmembers():
            if not member.isfile() or member.name not in expected or member.name in payloads:
                raise ValueError('UNEXPECTED_CHECKPOINT_ARCHIVE_MEMBER')
            stream = archive.extractfile(member)
            value = stream.read()
            if hashlib.sha256(value).hexdigest() != expected[member.name]:
                raise ValueError('CHECKPOINT_MEMBER_INTEGRITY_FAILED')
            payloads[member.name] = value
    if set(payloads) != set(expected):
        raise ValueError('INCOMPLETE_CHECKPOINT_ARCHIVE')
    root = Path(root).resolve()
    targets = {}
    for name,digest in manifest['checkpoints'].items():
        path = Path(name)
        if (path.is_absolute() or '..' in path.parts or not name.startswith('outputs/') or
                name not in payloads or expected[name] != digest):
            raise ValueError('INVALID_CHECKPOINT_DESTINATION')
        target = root/path
        if not target.resolve().is_relative_to(root/'outputs'):
            raise ValueError('INVALID_CHECKPOINT_DESTINATION')
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            raise ValueError('CHECKPOINT_DESTINATION_CONFLICT')
        targets[name] = target
    for name,target in targets.items():
        if not target.exists():
            target.parent.mkdir(parents=True,exist_ok=True)
            temporary = target.with_suffix(target.suffix+'.part')
            temporary.write_bytes(payloads[name])
            temporary.replace(target)


def installed(manifest,root):
    validate_manifest(manifest)
    for name,digest in manifest['checkpoints'].items():
        path = Path(root)/name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            return False
    return True
