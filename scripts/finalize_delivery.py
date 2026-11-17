"""Seal the validated v1 upload pair locally; this never submits the form."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from mtah.inference import verify_freeze
from mtah.official import metrics
from mtah.target import organizer_contract


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    verify_freeze(ROOT)
    source=ROOT/'outputs/phase7/upload'
    manifest=json.loads((source/'manifest.json').read_text())
    if manifest['team_name']!='Kaizen' or manifest['submitted'] is not False:
        raise ValueError('FINAL_PACKAGE_STATUS_REQUIRES_REVIEW')
    out=ROOT/'outputs/phase10';upload=out/'upload';upload.mkdir(parents=True,exist_ok=True)
    paths=[source/'Kaizen.csv',source/'Kaizen.md']
    expected_doc='Team: Kaizen\n\n'+(ROOT/'docs/TECHNICAL_REPORT.md').read_text()
    if paths[1].read_text()!=expected_doc:raise ValueError('FINAL_TECHNICAL_DOCUMENT_STALE')
    candidate=ROOT/'outputs/phase6/submission.csv'
    if digest(paths[0])!=digest(candidate):raise ValueError('FINAL_CSV_DIFFERS_FROM_V1')
    # Validate every source and destination before copying either file.
    for path in paths:
        expected=manifest['files'][path.name]
        if digest(path)!=expected['sha256'] or path.stat().st_size!=expected['bytes']:
            raise ValueError('SOURCE_PACKAGE_MANIFEST_MISMATCH')
        destination=upload/path.name
        if destination.exists() and digest(destination)!=digest(path):
            raise ValueError('SEALED_UPLOAD_CONFLICT')
    if {p.name for p in upload.iterdir()}-{'Kaizen.csv','Kaizen.md'}:
        raise ValueError('UNEXPECTED_FINAL_UPLOAD_FILE')
    for path in paths:
        destination=upload/path.name
        if not destination.exists():destination.write_bytes(path.read_bytes())
        if digest(destination)!=digest(path):raise ValueError('FINAL_UPLOAD_COPY_MISMATCH')
    csv=upload/'Kaizen.csv'
    if metrics.validate_submission(metrics.read_label_csv(csv),organizer_contract(ROOT)['n_frames']):
        raise ValueError('FINAL_EXACT_FILE_VALIDATION_FAILED')
    logs=ROOT/'local_reports/phase10';logs.mkdir(parents=True,exist_ok=True)
    with (logs/'final_csv_validator.log').open('w') as stream:
        subprocess.run([sys.executable,str(ROOT/'vendor/mtah/evaluation/validate_submission.py'),str(csv)],
                       stdout=stream,stderr=subprocess.STDOUT,check=True)
    record=dict(team_name='Kaizen',selected_release='models-v1',source_tag='inference-v1',
        files={p.name:dict(sha256=digest(upload/p.name),bytes=(upload/p.name).stat().st_size) for p in paths},
        csv_matches_validated_v1='PASS',technical_document_current='PASS',official_validator='PASS',
        eligibility='UNRESOLVED',form_submitted=False)
    record_path=out/'final_manifest.json'
    if record_path.exists() and json.loads(record_path.read_text())!=record:
        raise ValueError('SEALED_MANIFEST_CONFLICT')
    record_path.write_text(json.dumps(record,indent=2)+'\n')
    print('Final two-file upload pair, v1 equality and both validators: PASS')
    print('Final local file hashes and document currency: PASS')


if __name__=='__main__':
    try:main()
    except Exception:
        logs=ROOT/'local_reports/phase10';logs.mkdir(parents=True,exist_ok=True)
        (logs/'finalize_failure.log').write_text(traceback.format_exc())
        print('Final delivery sealing: FAIL; details retained locally.')
        raise SystemExit(1) from None
