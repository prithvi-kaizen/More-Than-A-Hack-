"""Publish external aggregate evidence only; predictions and checkpoints stay local."""
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from mtah.external import file_digest


def summarize(records):
    result={k:float(np.mean([r[k] for r in records])) for k in
            ['spell_f1_mean','macro_f1','alive_dead_f1','n_pred_spells','fragmentation',
             'false_d_runs_inside_true_possessions']}
    result['spell_f1_per_iou']={k:float(np.mean([r['per_iou'][k] for r in records]))
                              for k in records[0]['per_iou']}
    result['duration_recall']={}
    for name in ['short','long']:
        support=sum(r['duration_recall'][name]['support'] for r in records)
        hits=sum(r['duration_recall'][name]['hits'] for r in records)
        result['duration_recall'][name]=dict(recall=hits/support if support else None,
                                            repeated_support=support,repeated_hits=hits)
    result['transition_timing']={}
    for kind in ['turnover','restart']:
        stats=[r['transition_timing'][kind] for r in records]
        counts={k:sum(r[k] for r in stats) for k in
                ['true','predicted','matched','absolute_latency_sum_frames','signed_latency_sum_frames']}
        counts['mean_absolute_latency_seconds']=(counts['absolute_latency_sum_frames']/counts['matched']/25
                                                 if counts['matched'] else None)
        result['transition_timing'][kind]=counts
    return result


def main():
    out=ROOT/'outputs/phase9'
    audit=json.loads((out/'audit.json').read_text())
    if any(v!='PASS' for k,v in audit.items() if k not in ['target_read','final_holdouts_read']):
        raise ValueError('PHASE9_AUDIT_REQUIRED')
    r=json.loads((out/'report.json').read_text());settings=r['settings']
    cells=[x for match in r['cv_folds'].values() for seeds in match.values() for x in seeds.values()]
    policies=list(cells[0])
    overall={p:summarize([c[p] for c in cells]) for p in policies}
    by_condition={scene:{p:summarize([c[p] for match in r['cv_folds'].values()
                    for c in match[scene].values()]) for p in policies}
                    for scene in settings['evaluation_scenarios']}
    by_match={m:{p:summarize([c[p] for seeds in scenes.values() for c in seeds.values()])
                 for p in policies} for m,scenes in r['cv_folds'].items()}
    scheduler_record=out/'parallel_execution_v2.json'
    scheduler=(dict(previous_sha256=file_digest(out/'scheduler_before_repair.py'),
                    repaired_sha256=file_digest(ROOT/'scripts/parallel_phase9.py'),
                    model_or_selection_settings_changed=False) if scheduler_record.exists()
               else dict(execution='serial canonical runner',model_or_selection_settings_changed=False))
    public=dict(settings=settings,selection=r['selection'],overall=overall,by_condition=by_condition,
        by_match=by_match,protocol_sha256=r['protocol_sha256'],
        source_hashes=json.loads((out/'protocol_frozen.json').read_text())['sources'],
        scheduler_repair=scheduler,
        validity_checkpoint_hashes=r['validity_checkpoints'],audit=audit,
        development_skipped_reason=r['development_skipped_reason'],
        development=r['selection'].get('development'),
        independent_matches=4,condition_seed_cells=len(cells),
        repeated_support_note='Same reference spells recur across 10 conditions and 5 seeds; clean seeds are identical. Repeated counts are not independent sample sizes.',
        final_holdouts_read=False,target_read=False,v1_changed=False,data_credit=r['data_credit'])
    (ROOT/'docs/phase9_results.json').write_text(json.dumps(public,indent=2,allow_nan=False)+'\n')
    print('External aggregate-only publication summary: PASS')


if __name__=='__main__':main()
