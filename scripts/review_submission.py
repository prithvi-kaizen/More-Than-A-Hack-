"""Read-only local delivery audit; disclose only fixed statuses for restricted files."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from mtah.inference import verify_freeze
from mtah.official import metrics
from mtah.target import organizer_contract


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def audit_package():
    verify_freeze(ROOT)
    print('Frozen source, configuration and checkpoints: PASS', flush=True)
    folder = ROOT / 'outputs/phase7/upload'
    manifest = json.loads((folder / 'manifest.json').read_text())
    require(manifest['team_name'] == 'Kaizen', 'TEAM_NAME_MISMATCH')
    for name in ('Kaizen.csv', 'Kaizen.md'):
        path = folder / name
        record = manifest['files'][name]
        require(digest(path) == record['sha256'] and path.stat().st_size == record['bytes'],
                'UPLOAD_MANIFEST_MISMATCH')
    expected_document = 'Team: Kaizen\n\n' + (ROOT / 'docs/TECHNICAL_REPORT.md').read_text()
    require((folder / 'Kaizen.md').read_text() == expected_document, 'UPLOAD_DOCUMENT_STALE')
    candidate = ROOT / 'outputs/phase6/submission.csv'
    replay = ROOT / 'outputs/phase6/submission_replay.csv'
    clean = ROOT / 'outputs/phase7/clean-checkout/outputs/phase6/submission.csv'
    require(len({digest(p) for p in (folder / 'Kaizen.csv', candidate, replay, clean)}) == 1,
            'INDEPENDENT_CSV_BYTES_DIFFER')
    original = json.loads(candidate.with_suffix('.report.json').read_text())
    require(original['csv_sha256'] == digest(candidate), 'CANDIDATE_PROVENANCE_MISMATCH')
    for other in (replay, clean):
        require(json.loads(other.with_suffix('.report.json').read_text()) == original,
                'INDEPENDENT_PROVENANCE_MISMATCH')
    for name in ('BAR-ATH_tactical_atd.xml', 'BAR-ATH_tactical_atd.txt'):
        require(digest(ROOT / 'data/tactical' / name) == original['input_hashes'][name],
                'ORIGINAL_INPUT_CHANGED')
    print('Upload manifest, document, original inputs and replay bytes: PASS', flush=True)

    contract = organizer_contract(ROOT)
    with (folder / 'Kaizen.csv').open(newline='', encoding='utf-8') as stream:
        rows = csv.reader(stream)
        require(next(rows) == ['frame', 'label'], 'INDEPENDENT_HEADER_CHECK_FAILED')
        count = 0
        for index, row in enumerate(rows):
            require(len(row) == 2 and row[0] == str(index), 'INDEPENDENT_FRAME_CHECK_FAILED')
            inside = any(start <= index <= stop for _, start, stop in contract['blocks'])
            require(row[1] in ('H', 'A', 'D') if inside else row[1] == 'X',
                    'INDEPENDENT_HALF_LABEL_CHECK_FAILED')
            count += 1
        require(count == contract['n_frames'], 'INDEPENDENT_ROW_COUNT_FAILED')
    require(metrics.validate_submission(metrics.read_label_csv(folder / 'Kaizen.csv'),
                                        contract['n_frames']) is None,
            'OFFICIAL_FUNCTION_FAILED')
    logs = ROOT / 'local_reports/submission_review'
    logs.mkdir(parents=True, exist_ok=True)
    with (logs / 'validator.log').open('w') as stream:
        subprocess.run([sys.executable, str(ROOT / 'vendor/mtah/evaluation/validate_submission.py'),
                        str(folder / 'Kaizen.csv')], stdout=stream, stderr=subprocess.STDOUT,
                       check=True)
    print('Independent CSV contract and both official validators: PASS', flush=True)


def audit_aggregates():
    record = json.loads((ROOT / 'docs/phase6_results.json').read_text())
    results = record['results']
    require(set(results) == {'J03WQQ', 'J03WR9'}, 'EXTERNAL_TEST_MATCHES_CHANGED')
    for model in ('phase3', 'phase4', 'motion_fusion'):
        primary = sum(r[model][scene]['spell_f1_mean'] for r in results.values()
                      for scene in ('clean', 'moderate', 'severe')) / 6
        structured = sum(r[model]['structured']['spell_f1_mean'] for r in results.values()) / 2
        require(abs(primary - record['means'][model]['mean_spell_f1']) < 1e-12 and
                abs(structured - record['means'][model]['structured_spell_f1']) < 1e-12,
                'EXTERNAL_AGGREGATE_MISMATCH')
    require(all(r['motion_fusion'][scene]['spell_f1_mean'] > r['phase4'][scene]['spell_f1_mean']
                for r in results.values() for scene in ('clean', 'moderate', 'severe', 'structured')),
            'EXTERNAL_GAIN_CLAIM_MISMATCH')
    for scene, duration, expected in [('clean', '0_2', .308), ('severe', '0_2', .133),
                                      ('structured', '0_2', .190), ('structured', '15_inf', .945)]:
        bins = [r['motion_fusion'][scene]['recall_iou_05_by_duration'][duration]
                for r in results.values()]
        pooled = sum(b['support'] * b['recall'] for b in bins) / sum(b['support'] for b in bins)
        require(abs(pooled - expected) < .0005, 'EXTERNAL_RECALL_CLAIM_MISMATCH')
    print('Published external means, gains and pooled recall arithmetic: PASS', flush=True)


def audit_history_paths():
    names = subprocess.check_output(['git', 'log', '--all', '--format=', '--name-only', '-z'],
                                    cwd=ROOT).decode().split('\0')
    restricted = {'data', 'outputs', 'external_data', 'local_reports'}
    extensions = {'.csv', '.npz', '.npy', '.pkl', '.pt', '.mp4', '.zip'}
    for name in names:
        name = name.strip()
        if not name:
            continue
        path = Path(name)
        require(not restricted.intersection(path.parts) and path.suffix not in extensions,
                'RESTRICTED_HISTORY_PATH')
    print('Reachable Git history restricted-path and artifact-extension check: PASS', flush=True)


def audit_synthetic_limitations():
    import numpy as np
    from mtah.motion import apply_policy, geometry_reliability

    # These verify the review's counterexamples, not successful error correction.
    ball = np.column_stack((np.arange(100) * .1, np.full(100, 20.)))
    quality = geometry_reliability(ball)
    shifted = geometry_reliability(ball + np.array([25., 15.]))
    require(np.array_equal(quality, shifted) and np.all(quality[2:-2] == 1),
            'SYNTHETIC_OFFSET_COUNTEREXAMPLE_CHANGED')
    p = np.tile([.6, .3, .1], (100, 1))
    _, boundary = apply_policy(p, np.ones(100), np.zeros(100), 'motion_fusion',
                               np.full(100, .5))
    require(np.all(boundary == 0), 'SYNTHETIC_BOUNDARY_COUNTEREXAMPLE_CHANGED')
    print('Synthetic evidence for the two documented limitations: PASS', flush=True)


if __name__ == '__main__':
    try:
        audit_package()
        audit_aggregates()
        audit_history_paths()
        audit_synthetic_limitations()
    except Exception:
        logs = ROOT / 'local_reports/submission_review'
        logs.mkdir(parents=True, exist_ok=True)
        (logs / 'failure.log').write_text(traceback.format_exc())
        print('Submission review: FAIL; diagnostics retained locally.')
        raise SystemExit(1) from None
