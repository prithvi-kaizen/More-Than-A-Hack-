"""Reject staged artifacts outside the public source/documentation allowlist."""
from pathlib import PurePosixPath
import subprocess


def allowed(path, mode, size):
    p = PurePosixPath(path)
    roots = {'src', 'tests', 'scripts', 'docs', 'research', 'vendor', '.github', 'configs'}
    root_files = {'README.md', '.gitignore', '.python-version', 'pyproject.toml', 'uv.lock', 'AGENTS.md'}
    extensions = {'.py', '.md', '.json', '.yml', '.yaml', '.ipynb'}
    return (mode in {'100644', '100755'} and size <= 500_000
            and (path in root_files or (p.parts[0] in roots and
                 (p.suffix in extensions or p.name == 'LICENSE')))
            and not any(part in {'data', 'outputs', 'local_reports', 'external_data'} for part in p.parts))


def main():
    entries = subprocess.check_output(['git', 'ls-files', '--stage', '-z']).split(b'\0')
    rejected = 0
    for entry in filter(None, entries):
        info, raw_path = entry.split(b'\t', 1)
        mode, blob, stage = info.decode().split()
        size = int(subprocess.check_output(['git', 'cat-file', '-s', blob]))
        if stage != '0' or not allowed(raw_path.decode(), mode, size):
            rejected += 1
    print('Staged public-file gate: ' + ('PASS' if rejected == 0 else 'FAIL'))
    return bool(rejected)


if __name__ == '__main__':
    raise SystemExit(main())
