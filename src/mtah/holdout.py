"""Final-test access is separate from the training/development loaders."""
import json
from pathlib import Path

from mtah.external import FROZEN_SPLIT, file_digest, metadata, verify_raw
from mtah.inference import verify_freeze


def verify_holdout(root, config, match):
    if match not in FROZEN_SPLIT['holdout'] or config['split'] != FROZEN_SPLIT:
        raise ValueError('UNAUTHORIZED_FINAL_HOLDOUT')
    verify_freeze(root)
    root = Path(root)
    folder = root/'external_data/idsse/prepared'/match
    record = json.loads((folder/'provenance.json').read_text())
    expected = {f'half_{h}_{kind}.npz' for h in (1,2) for kind in ('inputs','labels')}
    if (record['match_id'] != match or record['adapter_version'] != 1 or
            set(record['prepared_hashes']) != expected):
        raise ValueError('HOLDOUT_PREPARED_PROVENANCE_FAILED')
    for name,digest in record['prepared_hashes'].items():
        if file_digest(folder/name) != digest:
            raise ValueError('HOLDOUT_PREPARED_INTEGRITY_FAILED')
    for entry in config['files']:
        if entry['match_id'] != match:
            continue
        path = root/'external_data/idsse/raw'/entry['name']
        if not path.exists():
            path = path.with_suffix('.xml.gz')
        if not verify_raw(path,entry) or file_digest(path) != record['source_hashes'][entry['kind']]:
            raise ValueError('HOLDOUT_RAW_PROVENANCE_FAILED')
        if entry['kind'] == 'metadata':
            pitch,_ = metadata(path,match)
    return folder,pitch,record
