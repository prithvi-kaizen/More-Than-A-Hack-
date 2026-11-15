"""Run the full target CLI twice in independent processes; compare bytes locally."""
import fcntl
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from mtah.inference import verify_freeze


def main():
    verify_freeze(ROOT)
    out = ROOT/'outputs/phase6'
    out.mkdir(parents=True,exist_ok=True)
    files = [out/'submission.csv',out/'submission_replay.csv']
    for number,path in enumerate(files,1):
        log = ROOT/f'local_reports/phase6_target_run_{number}.log'
        log.parent.mkdir(parents=True,exist_ok=True)
        with log.open('w') as stream:
            result = subprocess.run([sys.executable,str(ROOT/'scripts/infer_target.py'),
                '--output',str(path)],stdout=stream,stderr=subprocess.STDOUT,cwd=ROOT)
        if result.returncode:
            print('Independent target inference: FAIL; details retained locally.')
            return 1
        print(f'Independent target inference {number}: PASS',flush=True)
    reports = [json.loads(path.with_suffix('.report.json').read_text()) for path in files]
    if files[0].read_bytes() != files[1].read_bytes() or reports[0] != reports[1]:
        print('Target CSV byte reproduction: FAIL')
        return 1
    (out/'target_replay.json').write_text(json.dumps(dict(independent_processes=2,
        csv_byte_equality='PASS',input_and_freeze_equality='PASS',official_validator='PASS'),indent=2))
    print('Target CSV byte reproduction and both official validators: PASS')
    return 0


if __name__ == '__main__':
    path = ROOT/'outputs/phase6/target.lock'
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w') as lock:
        try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: raise SystemExit('TARGET_INFERENCE_ALREADY_RUNNING') from None
        raise SystemExit(main())
