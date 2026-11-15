"""Copy the validated candidate into a private team-named upload package."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from mtah.inference import verify_freeze
from mtah.official import metrics
from mtah.target import organizer_contract


def team_filename(name):
    if (not name or name != name.strip() or name in ('.','..') or len(name)>100 or
            any(c in '/\\' or ord(c)<32 or ord(c)==127 for c in name)):
        raise ValueError('INVALID_TEAM_FILENAME')
    return name


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--team-name',required=True)
    args = parser.parse_args()
    name = team_filename(args.team_name)
    verify_freeze(ROOT)
    candidate = ROOT/'outputs/phase6/submission.csv'
    original = json.loads(candidate.with_suffix('.report.json').read_text())
    if digest(candidate) != original['csv_sha256']:
        raise ValueError('VALIDATED_CANDIDATE_CHANGED')
    contract = organizer_contract(ROOT)
    if metrics.validate_submission(metrics.read_label_csv(candidate),contract['n_frames']):
        raise ValueError('CANDIDATE_VALIDATION_FAILED')
    out = ROOT/'outputs/phase7/upload'
    out.mkdir(parents=True,exist_ok=True)
    csv = out/(name+'.csv')
    document = out/(name+'.md')
    report = (ROOT/'docs/TECHNICAL_REPORT.md').read_text()
    document.write_text('Team: '+name+'\n\n'+report,encoding='utf-8')
    shutil.copyfile(candidate,csv)
    if digest(candidate) != digest(csv):
        raise ValueError('PACKAGED_CSV_BYTES_CHANGED')
    if metrics.validate_submission(metrics.read_label_csv(csv),contract['n_frames']):
        raise ValueError('PACKAGED_CSV_VALIDATION_FAILED')
    log = ROOT/'local_reports/phase7_packaged_validator.log'
    with log.open('w') as stream:
        result = subprocess.run([sys.executable,str(ROOT/'vendor/mtah/evaluation/validate_submission.py'),
                                 str(csv)],stdout=stream,stderr=subprocess.STDOUT)
    if result.returncode:
        raise ValueError('PACKAGED_CSV_VALIDATOR_CLI_FAILED')
    instructions = ('# Upload package\n\nRegistered team: '+name+'\n\n'
        'Upload `'+csv.name+'` as the possession file and `'+document.name+'` as the technical document.\n\n'
        'Public repository: https://github.com/prithvi-kaizen/More-Than-A-Hack-\n\n'
        'Use the submission form emailed to registered participants. The organizer states '
        '17 November 2026; check the exact cutoff time in that email. '
        'This package has not been submitted. Read the technical document and confirm its claims before uploading.\n')
    (out/'UPLOAD_INSTRUCTIONS.md').write_text(instructions,encoding='utf-8')
    manifest = dict(team_name=name,files={p.name:dict(sha256=digest(p),bytes=p.stat().st_size)
                    for p in [csv,document]},csv_byte_equality='PASS',official_validator='PASS',
                    repository='https://github.com/prithvi-kaizen/More-Than-A-Hack-',submitted=False)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print('Team-named CSV/document package and exact-file validation: PASS')


if __name__ == '__main__':
    try:
        main()
    except Exception:
        path = ROOT/'local_reports/phase7_package_failure.log'
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(traceback.format_exc())
        print('Submission packaging: FAIL; diagnostic details retained locally.')
        raise SystemExit(1) from None
