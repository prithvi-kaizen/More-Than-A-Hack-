"""Clone the public release into an empty folder and reproduce the private CSV locally."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

ROOT=Path(__file__).resolve().parents[1]
REPOSITORY='https://github.com/prithvi-kaizen/More-Than-A-Hack-.git'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-dir',type=Path,default=ROOT/'data')
    parser.add_argument('--reference',type=Path,default=ROOT/'outputs/phase6/submission.csv')
    args=parser.parse_args()
    checkout=ROOT/'outputs/phase7/clean-checkout'
    if checkout.exists():raise ValueError('CLEAN_CHECKOUT_ALREADY_EXISTS')
    checkout.parent.mkdir(parents=True,exist_ok=True)
    logs=ROOT/'local_reports/phase7_clean_checkout.log'
    logs.parent.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ)
    env.pop('PYTHONPATH',None)
    env.pop('UV_NO_SYNC',None)
    env['UV_PROJECT_ENVIRONMENT']=str(checkout/'.venv')
    env['UV_NATIVE_TLS']='true'
    env['LOKY_MAX_CPU_COUNT']='2'
    with logs.open('w') as log:
        def run(command,cwd=checkout):
            subprocess.run(command,cwd=cwd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        run(['git','-c','credential.helper=','clone','--depth','1','--single-branch',
             '--branch','inference-v1',REPOSITORY,str(checkout)],cwd=ROOT)
        if any((checkout/n).exists() for n in ['outputs','data','external_data','local_reports','.venv']):
            raise ValueError('CHECKOUT_NOT_CLEAN')
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=checkout,text=True).strip()
        print('Anonymous public source checkout, no data/models/environment: PASS',flush=True)
        run(['uv','sync','--frozen','--extra','temporal'])
        print('Separate locked environment installation: PASS',flush=True)
        run(['uv','run','--frozen','--extra','temporal','python','scripts/download_checkpoints.py'])
        print('Anonymous public checkpoint download and installation: PASS',flush=True)
        run(['uv','run','--frozen','--extra','temporal','python','scripts/infer_target.py',
             '--data-dir',str(args.data_dir.resolve())])
    regenerated=checkout/'outputs/phase6/submission.csv'
    if regenerated.read_bytes()!=args.reference.read_bytes():
        raise ValueError('CLEAN_CHECKOUT_CSV_BYTES_DIFFER')
    original=json.loads(args.reference.with_suffix('.report.json').read_text())
    replay=json.loads(regenerated.with_suffix('.report.json').read_text())
    if original!=replay:
        raise ValueError('CLEAN_CHECKOUT_PROVENANCE_DIFFERS')
    version_code=("import json,platform,torch,numpy,sklearn;print(json.dumps(dict("
        "python=platform.python_version(),system=platform.system(),architecture=platform.machine(),"
        "torch=torch.__version__,numpy=numpy.__version__,sklearn=sklearn.__version__)))")
    versions=json.loads(subprocess.check_output(['uv','run','--frozen','--extra','temporal',
        'python','-c',version_code],cwd=checkout,env=env,text=True))
    report=dict(source_commit=commit,source_tag='inference-v1',initial_data_models_environment_absent=True,
        anonymous_source_clone='PASS',separate_locked_environment='PASS',public_checkpoint_download='PASS',
        csv_byte_equality='PASS',input_and_freeze_equality='PASS',official_validator=replay['official_validator'],
        environment=versions)
    (ROOT/'outputs/phase7/clean_checkout_report.json').write_text(json.dumps(report,indent=2))
    print('Clean public checkout CSV bytes, provenance and validator: PASS',flush=True)


if __name__=='__main__':
    try:main()
    except Exception:
        path=ROOT/'local_reports/phase7_clean_checkout_failure.log'
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(traceback.format_exc())
        print('Clean-checkout reproduction: FAIL; diagnostic details retained locally.')
        raise SystemExit(1) from None
