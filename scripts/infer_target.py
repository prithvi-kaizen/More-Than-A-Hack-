"""Generate a local tactical submission with frozen models and status-only output."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

import torch
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from mtah.external import file_digest
from mtah.inference import FrozenPredictor, verify_freeze
from mtah.official import metrics
from mtah.target import organizer_contract, load_tactical, predict_halves, write_submission


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir',type=Path,default=Path(os.environ.get('MTAH_DATA_DIR') or ROOT/'data'))
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/phase6/submission.csv')
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to((ROOT/'outputs').resolve()):
        raise ValueError('TARGET_OUTPUT_MUST_REMAIN_IN_IGNORED_OUTPUTS')
    verify_freeze(ROOT)
    predictor = FrozenPredictor(ROOT)
    contract = organizer_contract(ROOT)
    metadata = args.data_dir/'tactical/BAR-ATH_tactical_atd.xml'
    tracking = args.data_dir/'tactical/BAR-ATH_tactical_atd.txt'
    input_hashes = {p.name:file_digest(p) for p in (metadata,tracking)}
    players,ball,teams = load_tactical(metadata,tracking,contract)
    print('Tactical inputs and frame alignment: PASS',flush=True)
    labels = predict_halves(predictor,players,ball,teams,contract)
    write_submission(output,labels)
    reason = metrics.validate_submission(metrics.read_label_csv(output),contract['n_frames'])
    if reason:
        raise ValueError('OFFICIAL_SUBMISSION_VALIDATION_FAILED')
    log = ROOT/'local_reports/phase6_validator'
    log.mkdir(parents=True,exist_ok=True)
    with (log/(output.stem+'.log')).open('w') as stream:
        result = subprocess.run([sys.executable,str(ROOT/'vendor/mtah/evaluation/validate_submission.py'),
            str(output),'--rows',str(contract['n_frames'])],stdout=stream,stderr=subprocess.STDOUT)
    if result.returncode:
        raise ValueError('OFFICIAL_VALIDATOR_CLI_FAILED')
    verify_freeze(ROOT)
    if input_hashes != {p.name:file_digest(p) for p in (metadata,tracking)}:
        raise ValueError('TARGET_INPUT_CHANGED_DURING_INFERENCE')
    report = dict(input_hashes=input_hashes,csv_sha256=file_digest(output),
        selection_freeze_sha256=file_digest(ROOT/'configs/phase6_frozen.json'),
        execution_freeze_sha256=file_digest(ROOT/'configs/phase6_execution_frozen.json'),
        official_validator='PASS',frame_alignment='PASS',half_isolation='PASS',
        target_label_data_read=False,video_read=False,fitting_performed=False)
    output.with_suffix('.report.json').write_text(json.dumps(report,indent=2))
    print('Local prediction and pinned official validator: PASS',flush=True)


if __name__ == '__main__':
    try:
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        with threadpool_limits(limits=2): main()
    except Exception:
        path = ROOT/'local_reports/phase6_target_failure.log'
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(traceback.format_exc())
        print('Target inference: FAIL; diagnostic details retained locally.')
        raise SystemExit(1) from None
