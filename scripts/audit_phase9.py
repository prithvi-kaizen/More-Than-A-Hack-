"""Verify Phase 9 source/data artifact hashes and independently rescore saved folds."""
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from mtah.external import file_digest,FROZEN_SPLIT
from mtah.inference import verify_freeze
from mtah.official import metrics as pm


def main():
    out=ROOT/'outputs/phase9'
    report=json.loads((out/'report.json').read_text())
    frozen=json.loads((out/'protocol_frozen.json').read_text())
    verify_freeze(ROOT)
    execution_path=out/'parallel_execution_v2.json'
    if execution_path.exists():
        execution=json.loads(execution_path.read_text())
        if file_digest(ROOT/'scripts/parallel_phase9.py')!=execution['scheduler_sha256']:
            raise ValueError('SCHEDULER_SOURCE_CHANGED')
        old=json.loads((out/'parallel_execution.json').read_text())
        if file_digest(out/'scheduler_before_repair.py')!=old['scheduler_sha256']:
            raise ValueError('ORIGINAL_SCHEDULER_RECORD_CHANGED')
    elif (out/'parallel_execution.json').exists():
        raise ValueError('INCOMPLETE_PARALLEL_EXECUTION_RECORD')
    for path,digest in frozen['sources'].items():
        if file_digest(ROOT/path)!=digest:raise ValueError('EXPERIMENT_SOURCE_CHANGED')
    for path,digest in json.loads((out/'artifact_hashes.json').read_text()).items():
        p=Path(path)
        if p.is_absolute() or '..' in p.parts or file_digest(out/p)!=digest:
            raise ValueError('EXPERIMENT_ARTIFACT_CHANGED')
    settings=report['settings'];previous=json.loads((ROOT/'docs/phase5_results.json').read_text())
    if set(report['cv_folds'])!=set(FROZEN_SPLIT['train']):raise ValueError('FOLD_SET_MISMATCH')
    values={};per_scene={}
    for match,scenes in report['cv_folds'].items():
        truth=[]
        for half in (1,2):
            path=ROOT/'external_data/idsse/prepared'/match/f'half_{half}_labels.npz'
            provenance=json.loads((out/'provenance'/f'{match}.json').read_text())
            if file_digest(path)!=provenance['prepared_hashes'][path.name]:
                raise ValueError('EXTERNAL_LABEL_PROVENANCE_CHANGED')
            with np.load(path,allow_pickle=False) as f:truth.extend([f['labels'],np.array(['X'])])
        gt=pm.label_codes(np.concatenate(truth));runs=pm.valid_runs(gt)
        values[match]={v:[] for v in ('v1','challenger','v1_no_motion','challenger_no_motion')}
        for scene,seeds in scenes.items():
            for seed,policies in seeds.items():
                for policy,expected in policies.items():
                    pred=[]
                    actual_seed=settings['evaluation_seeds'][0] if scene=='clean' else seed
                    for half in (1,2):
                        path=out/'predictions'/match/scene/f'{actual_seed}_{half}.npz'
                        with np.load(path,allow_pickle=False) as f:pred.extend([f[policy],np.array(['X'])])
                    codes=pm.label_codes(np.concatenate(pred))
                    score=pm.spell_scores(gt,codes,runs)
                    np.testing.assert_allclose(score['mean_f1'],expected['spell_f1_mean'],atol=1e-12,rtol=0)
                    for iou,record in score['per_iou'].items():
                        np.testing.assert_allclose(record['f1'],expected['per_iou'][str(iou)],atol=1e-12,rtol=0)
                    values[match][policy].append(score['mean_f1'])
                    per_scene.setdefault((scene,policy),[]).append(score['mean_f1'])
            # Clean baseline must replay the historical whole-match fold exactly.
            if scene=='clean':
                score=seeds[str(settings['evaluation_seeds'][0])]['v1']['spell_f1_mean']
                np.testing.assert_allclose(score,previous['cv_folds'][match]['results']['motion_fusion']['clean']['spell_f1_mean'],atol=1e-12,rtol=0)
        print(match+'_independent_saved_spell_scores_and_clean_replay: PASS',flush=True)
    for policy in values[FROZEN_SPLIT['train'][0]]:
        mean=float(np.mean([np.mean(v[policy]) for v in values.values()]))
        np.testing.assert_allclose(mean,report['selection']['means'][policy],atol=1e-12,rtol=0)
    for scene in settings['evaluation_scenarios']:
        delta=np.mean(per_scene[(scene,'challenger')])-np.mean(per_scene[(scene,'v1')])
        np.testing.assert_allclose(delta,report['selection']['condition_differences'][scene],atol=1e-12,rtol=0)
    result=dict(source_and_artifact_hashes='PASS',official_spell_scores='PASS',
                per_iou_scores='PASS',original_clean_fold_replay='PASS',aggregation='PASS',
                target_read=False,final_holdouts_read=False)
    (out/'audit.json').write_text(json.dumps(result,indent=2))
    print('Phase 9 independent spell rescore and aggregate arithmetic: PASS')


if __name__=='__main__':
    main()
