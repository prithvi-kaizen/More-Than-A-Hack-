"""Consolidate supplied assets; delete copies only after SHA-256 verification."""
from pathlib import Path
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    report = {'retained': [], 'removed': [], 'moved': []}
    archives = sorted(ROOT.glob('*.zip'))
    verified_archives = []
    for archive in archives:
        with zipfile.ZipFile(archive) as z:
            verified = []
            for entry in z.infolist():
                if entry.is_dir():
                    continue
                rel = Path(entry.filename)
                if len(rel.parts) != 2 or rel.parts[0] not in ('tactical', 'broadcast') or not rel.name.startswith('BAR-ATH_'):
                    raise ValueError('Unexpected archive member; archive retained')
                destination = ROOT / 'data' / rel
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_suffix(destination.suffix + '.extracting')
                h = hashlib.sha256()
                with z.open(entry) as source, temporary.open('wb') as target:
                    for block in iter(lambda: source.read(1024 * 1024), b''):
                        target.write(block)
                        h.update(block)
                expected = h.hexdigest()
                if digest(temporary) != expected:
                    raise ValueError('Extraction verification failed')
                if destination.exists():
                    if digest(destination) != expected:
                        raise ValueError('Distinct same-name files; all copies retained')
                    temporary.unlink()
                else:
                    temporary.rename(destination)
                verified.append((destination, expected))
                report['retained'].append({'path': str(destination.relative_to(ROOT)), 'sha256': expected})
            verified_archives.append((archive, verified))

    candidates = list(ROOT.glob('BAR-ATH_*'))
    for folder in ('tactical', 'tactical 2', 'broadcast'):
        path = ROOT / folder
        if path.exists():
            candidates.extend(p for p in path.rglob('*') if p.is_file())
    for source in candidates:
        if not source.name.startswith('BAR-ATH_'):
            continue
        kind = 'broadcast' if source.name.startswith('BAR-ATH_broadcast_') else 'tactical'
        target = ROOT / 'data' / kind / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if digest(source) != digest(target):
                raise ValueError('Distinct same-name file; source retained')
            source.unlink()
            report['removed'].append(str(source.relative_to(ROOT)))
        else:
            source.rename(target)
            report['moved'].append(str(source.relative_to(ROOT)))

    for archive, members in verified_archives:
        if members and all(p.exists() and digest(p) == h for p, h in members):
            archive.unlink()
            report['removed'].append(archive.name)
    original, duplicate = ROOT / 'README.md', ROOT / 'README (1).md'
    if original.exists() and duplicate.exists() and digest(original) == digest(duplicate):
        duplicate.unlink()
        report['removed'].append(duplicate.name)
    for folder in ('tactical', 'tactical 2', 'broadcast'):
        path = ROOT / folder
        if path.exists() and not any(path.iterdir()):
            path.rmdir()
    report_dir = ROOT / 'local_reports'
    report_dir.mkdir(exist_ok=True)
    (report_dir / 'asset_cleanup.json').write_text(json.dumps(report, indent=2))
    print('Consolidation complete. Removed verified redundant files:', len(report['removed']))
    print('Detailed manifest saved locally; no asset contents emitted.')


if __name__ == '__main__':
    main()
