"""Verify registered external raw/prepared provenance before model access."""
import json
from pathlib import Path
import numpy as np
from mtah.external import file_digest, metadata, verify_raw


def verify_prepared(root, config, match_id, split):
    if split not in ('train', 'development') or match_id not in config['split'][split]:
        raise ValueError('UNAUTHORIZED_MODEL_DATA_SPLIT')
    root = Path(root)
    folder = root / 'external_data/idsse/prepared' / match_id
    provenance = json.loads((folder / 'provenance.json').read_text())
    if provenance['match_id'] != match_id or provenance['adapter_version'] != 1:
        raise ValueError('PREPARED_PROVENANCE_MISMATCH')
    expected = {f'half_{half}_{kind}.npz' for half in (1, 2) for kind in ('inputs', 'labels')}
    if set(provenance['prepared_hashes']) != expected:
        raise ValueError('INVALID_PREPARED_FILE_SET')
    for name, digest in provenance['prepared_hashes'].items():
        if file_digest(folder / name) != digest:
            raise ValueError('PREPARED_INTEGRITY_FAILED')
    for entry in config['files']:
        if entry['match_id'] != match_id:
            continue
        path = root / 'external_data/idsse/raw' / entry['name']
        if not path.exists():
            path = path.with_suffix('.xml.gz')
        if not verify_raw(path, entry) or file_digest(path) != provenance['source_hashes'][entry['kind']]:
            raise ValueError('RAW_PROVENANCE_FAILED')
        if entry['kind'] == 'metadata':
            pitch, _ = metadata(path, match_id)
    return folder, pitch, provenance


def read_half(folder, half):
    with np.load(Path(folder) / f'half_{half}_inputs.npz', allow_pickle=False) as f:
        required = {'frame_ids', 'players', 'ball', 'teams', 'fps'}
        if set(f.files) != required:
            raise ValueError('UNEXPECTED_MODEL_INPUT_FIELDS')
        inputs = {key: f[key] for key in required}
    with np.load(Path(folder) / f'half_{half}_labels.npz', allow_pickle=False) as f:
        labels, alive = f['labels'], f['alive']
    if len(inputs['ball']) != len(labels) or len(alive) != len(labels):
        raise ValueError('PREPARED_FRAME_MISMATCH')
    return inputs, labels, alive
