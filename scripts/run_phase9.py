"""Frozen external-only validity challenger; caches store no competition data."""
import fcntl
import json
from pathlib import Path
import pickle
import sys
import traceback
import warnings
import numpy as np
import torch
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from mtah.external import FROZEN_SPLIT, file_digest, manifest
from mtah.prepared import verify_prepared,read_half
from mtah.inference import verify_freeze,feature_bundle
from mtah.temporal import load_motion,predict_motion
from mtah.motion import apply_policy
from mtah.decoder import decode
from mtah.reliability import stress_inputs,validity_features,validity_targets,ValidityModel,training_partition
from mtah.official import metrics as pm

OUT=ROOT/'outputs/phase9'
POLICIES=('v1','challenger','v1_no_motion','challenger_no_motion')


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.part')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False));temporary.replace(path)


def freeze_protocol(settings):
    verify_freeze(ROOT)
    previous=json.loads((ROOT/'docs/phase5_results.json').read_text())
    paths=['configs/phase9.json','src/mtah/reliability.py','scripts/run_phase9.py',
           'docs/phase5_results.json','configs/phase4.json','configs/phase5.json','uv.lock']
    source={p:file_digest(ROOT/p) for p in paths}
    for match,fold in previous['cv_folds'].items():
        if match not in FROZEN_SPLIT['train'] or fold['training_matches']!=[m for m in FROZEN_SPLIT['train'] if m!=match]:
            raise ValueError('OLD_FOLD_SPLIT_MISMATCH')
        for name,key in [('tree.pkl','tree_sha256'),('motion.pt','motion_sha256')]:
            path=f'outputs/phase5/oof/{match}/{name}'
            if file_digest(ROOT/path)!=fold[key]:raise ValueError('OLD_FOLD_CHECKPOINT_CHANGED')
            source[path]=fold[key]
    record=dict(settings=settings,sources=source,prior_fold_provenance='verified',
                final_holdouts_read=False,target_read=False,training_restricted_to=FROZEN_SPLIT['train'])
    frozen=OUT/'protocol_frozen.json'
    if frozen.exists() and json.loads(frozen.read_text())!=record:
        raise ValueError('PHASE9_FROZEN_PROTOCOL_CHANGED')
    save(frozen,record)
    return previous


def seed_for(base,match,half):
    index=FROZEN_SPLIT['train'].index(match) if match in FROZEN_SPLIT['train'] else 100
    return base+100000*index+1000*half


def training_data(match,folder,pitch,settings):
    if match not in FROZEN_SPLIT['train']:raise ValueError('UNAUTHORIZED_VALIDITY_TRAINING')
    cache=OUT/'training'/f'{match}.npz'
    if cache.exists():
        with np.load(cache,allow_pickle=False) as f:return f['features'],f['targets']
    features,targets=[],[]
    for half in (1,2):
        inputs,_,_=read_half(folder,half)
        for scene in settings['training_scenarios']:
            seeds=settings['train_seeds'][:1] if scene=='clean' else settings['train_seeds']
            for base in seeds:
                p,b,t=stress_inputs(inputs['players'],inputs['ball'],inputs['teams'],25,
                    seed_for(base,match,half),pitch,scene,settings)
                features.append(validity_features(p,b,25,pitch)[::settings['training_stride']])
                targets.append(validity_targets(inputs['ball'],b,settings['validity_error_metres'])[::settings['training_stride']])
    x,y=np.concatenate(features),np.concatenate(targets)
    cache.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(cache,features=x,targets=y)
    return x,y


def metric_summary(truth,predictions):
    gt=pm.label_codes(np.concatenate([np.append(truth[h],'X') for h in (1,2)]))
    pred=pm.label_codes(np.concatenate([np.append(predictions[h],'X') for h in (1,2)]))
    r=pm.evaluate(gt,pred,bootstrap_n=0)
    runs=pm.valid_runs(gt);spells=pm.possession_spells(gt,runs)
    ps=pm.possession_spells(pred,runs)
    flags=pm.matched_gt_flags(pm.spell_overlaps(spells,ps),len(spells),.5)
    durations=np.array([(b-a)/25 for a,b,_ in spells])
    result={k:r[k] for k in ['spell_f1_mean','macro_f1','alive_dead_f1','n_gt_spells','n_pred_spells','fragmentation']}
    result['per_iou']={str(k):v['f1'] for k,v in r['spell_f1'].items()}
    result['duration_recall']={}
    for name,mask in [('short',durations<2),('long',durations>=15)]:
        result['duration_recall'][name]=dict(support=int(mask.sum()),hits=int(flags[mask].sum()))
    # Count false predicted D runs intersecting a true H/A spell (one per split).
    splits=0
    for start,stop,_ in spells:
        local=pred[start:stop]==pm.D
        splits+=int(np.count_nonzero(local & ~np.r_[False,local[:-1]]))
    result['false_d_runs_inside_true_possessions']=splits
    result['transition_timing']=transition_timing(gt,pred,runs)
    return result


def transition_timing(gt,pred,runs):
    # Same type, one-to-one nearest-first matches within 1 second; report misses too.
    actual=pm.transitions(gt,runs); estimated=pm.transitions(pred,runs)
    result={}
    for name in ('turnover','restart'):
        def eligible(t):
            return (t[1] in (pm.H,pm.A) and t[2] in (pm.H,pm.A)) if name=='turnover' else (t[1]==pm.D and t[2] in (pm.H,pm.A))
        a=[t for t in actual if eligible(t)];b=[t for t in estimated if eligible(t)]
        pairs=sorted((abs(x[0]-y[0]),i,j,y[0]-x[0]) for i,x in enumerate(a) for j,y in enumerate(b)
                     if x[1:]==y[1:] and abs(x[0]-y[0])<=25)
        used_a,used_b,latencies=set(),set(),[]
        for _,i,j,lag in pairs:
            if i not in used_a and j not in used_b:
                used_a.add(i);used_b.add(j);latencies.append(lag)
        result[name]=dict(true=len(a),predicted=len(b),matched=len(latencies),
            absolute_latency_sum_frames=int(sum(abs(x) for x in latencies)),
            signed_latency_sum_frames=int(sum(latencies)),tolerance_frames=25)
    return result


def fold_predictor(match,previous):
    path=ROOT/'outputs/phase5/oof'/match
    record=previous['cv_folds'][match]
    state=torch.load(path/'motion.pt',map_location='cpu',weights_only=True)
    if state['training_matches']!=record['training_matches'] or match in state['training_matches']:
        raise ValueError('TEMPORAL_FOLD_LEAKAGE')
    with (path/'tree.pkl').open('rb') as f:tree=pickle.load(f)
    return tree,load_motion(path/'motion.pt')


def evaluate_match(match,folder,pitch,tree,neural,validity,settings):
    result={};base_settings=json.loads((ROOT/'configs/phase4.json').read_text())
    for scene in settings['evaluation_scenarios']:
        result[scene]={}
        for base in settings['evaluation_seeds']:
            record=OUT/'evaluations'/match/scene/f'{base}.json'
            predictions={v:{} for v in POLICIES};truth={}
            for half in (1,2):
                inputs,labels,_=read_half(folder,half);truth[half]=labels
                path=OUT/'predictions'/match/scene/f'{base}_{half}.npz'
                if path.exists():
                    with np.load(path,allow_pickle=False) as f:
                        for v in POLICIES:predictions[v][half]=f[v]
                    continue
                # Clean inputs are seed-independent: compute them once, preserving five reported repeats.
                first=OUT/'predictions'/match/scene/f"{settings['evaluation_seeds'][0]}_{half}.npz"
                if scene=='clean' and first.exists():
                    with np.load(first,allow_pickle=False) as f:
                        for v in POLICIES:predictions[v][half]=f[v]
                    continue
                p,b,t=stress_inputs(inputs['players'],inputs['ball'],inputs['teams'],25,
                    seed_for(base,match,half),pitch,scene,settings)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore',RuntimeWarning)
                    features=feature_bundle(p,b,t,25,pitch)
                probs,hazard=tree.predict(features['alive'],features['owner'],features['boundary'])
                motion=predict_motion(neural,features['motion'],len(b),5,512)
                q=features['reliability'];new_q=validity.quality(validity_features(p,b,25,pitch),q)
                for v in POLICIES:
                    quality=new_q if v.startswith('challenger') else q
                    pp,hh=apply_policy(probs,hazard,quality,'motion_fusion',motion,
                                      0 if v.endswith('no_motion') else .35,.5)
                    predictions[v][half]=decode(pp,base_settings['switch_penalty'],base_settings['dead_penalty'],
                                               hh,base_settings['boundary_reduction'])
                path.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(path,**{v:predictions[v][half] for v in POLICIES})
            current={v:metric_summary(truth,predictions[v]) for v in POLICIES}
            if record.exists() and json.loads(record.read_text())!=current:
                raise ValueError('PHASE9_METRIC_REPLAY_MISMATCH')
            save(record,current);result[scene][str(base)]=current
        print(f'{match}_{scene}: PASS',flush=True)
    return result


def select(folds,settings):
    scenes=settings['evaluation_scenarios'];seeds=[str(x) for x in settings['evaluation_seeds']]
    def mean(records,policy):return float(np.mean([r[policy]['spell_f1_mean'] for r in records]))
    def cells(match=None,scene=None):
        return [folds[m][s][seed] for m in folds if match is None or m==match
                for s in scenes if scene is None or s==scene for seed in seeds]
    means={v:mean(cells(),v) for v in POLICIES}
    differences={m:mean(cells(match=m),'challenger')-mean(cells(match=m),'v1') for m in folds}
    conditions={s:mean(cells(scene=s),'challenger')-mean(cells(scene=s),'v1') for s in scenes}
    def recall(policy):
        bins=[r[policy]['duration_recall']['long'] for r in cells()]
        return sum(b['hits'] for b in bins)/sum(b['support'] for b in bins)
    long_difference=recall('challenger')-recall('v1');g=settings['gates']
    gates=dict(mean_gain=means['challenger']>=means['v1']+g['mean_gain'],
        winning_matches=sum(d>0 for d in differences.values())>=g['winning_matches'],
        match_guard=min(differences.values())>=-g['max_match_loss'],
        condition_guard=min(conditions.values())>=-g['max_condition_loss'],
        long_recall_guard=long_difference>=-g['max_long_recall_loss'],
        sustained=conditions['sustained']>0,independent_camera=conditions['camera_independent']>0)
    return dict(means=means,match_differences=differences,condition_differences=conditions,
                long_recall_difference=long_difference,gates=gates,accepted=all(gates.values()),
                recommendation='challenger_pending_independent_confirmation' if all(gates.values()) else 'retain_v1')


def main():
    settings=json.loads((ROOT/'configs/phase9.json').read_text())
    if set(settings['train_seeds'])&set(settings['evaluation_seeds']):raise ValueError('CORRUPTION_SEED_LEAKAGE')
    previous=freeze_protocol(settings)
    config=manifest(ROOT/'configs/external_split.json')
    prepared={};training={}
    for match in FROZEN_SPLIT['train']:
        folder,pitch,provenance=verify_prepared(ROOT,config,match,'train')
        save(OUT/'provenance'/f'{match}.json',provenance)
        prepared[match]=(folder,pitch);training[match]=training_data(match,folder,pitch,settings)
        print(match+'_external_training_ready: PASS',flush=True)
    folds={}
    for match in FROZEN_SPLIT['train']:
        fitting,calibration=training_partition(match)
        quality=ValidityModel().fit({m:training[m] for m in fitting},{calibration:training[calibration]},
                                    fitting,calibration,settings)
        checkpoint=OUT/'models'/f'{match}.pkl';checkpoint.parent.mkdir(parents=True,exist_ok=True)
        if checkpoint.exists():
            with checkpoint.open('rb') as f:reloaded=pickle.load(f)
            np.testing.assert_array_equal(quality.quality(training[match][0],np.ones(len(training[match][0]))),
                                          reloaded.quality(training[match][0],np.ones(len(training[match][0]))))
        else:
            with checkpoint.open('wb') as f:pickle.dump(quality,f,protocol=5)
        with checkpoint.open('rb') as f:reloaded=pickle.load(f)
        probe=training[match][0];probe_q=np.ones(len(probe))
        np.testing.assert_array_equal(quality.quality(probe,probe_q),reloaded.quality(probe,probe_q))
        tree,neural=fold_predictor(match,previous)
        folds[match]=evaluate_match(match,*prepared[match],tree,neural,quality,settings)
        save(OUT/'folds'/f'{match}.json',folds[match])
    selection=select(folds,settings);save(OUT/'cv_selection.json',selection)
    development=None
    if selection['accepted']:
        # One fixed dev comparison; its outcome cannot trigger another tuning pass.
        fit_matches=FROZEN_SPLIT['train'][:-1];cal=FROZEN_SPLIT['train'][-1]
        quality=ValidityModel().fit({m:training[m] for m in fit_matches},{cal:training[cal]},fit_matches,cal,settings)
        with (ROOT/'outputs/phase4/model.pkl').open('rb') as f:tree=pickle.load(f)
        neural=load_motion(ROOT/'outputs/phase5/motion.pt')
        match=FROZEN_SPLIT['development'][0]
        folder,pitch,_=verify_prepared(ROOT,config,match,'development')
        development=evaluate_match(match,folder,pitch,tree,neural,quality,settings)
        dev_settings=dict(settings,gates=dict(settings['gates'],winning_matches=1))
        selection['development']=select({match:development},dev_settings)
        if not selection['development']['accepted']:selection['recommendation']='retain_v1'
    freeze_protocol(settings)
    artifacts={str(p.relative_to(OUT)):file_digest(p) for directory in ['training','predictions','evaluations','models','provenance','folds']
               for p in (OUT/directory).rglob('*') if p.is_file()}
    save(OUT/'artifact_hashes.json',artifacts)
    report=dict(settings=settings,cv_folds=folds,selection=selection,development=development,
        development_skipped_reason=None if development else 'CV adoption gates failed',
        protocol_sha256=file_digest(OUT/'protocol_frozen.json'),
        validity_checkpoints={m:file_digest(OUT/'models'/f'{m}.pkl') for m in folds},
        final_holdouts_read=False,target_read=False,v1_changed=False,
        data_credit='DFL; Bassek et al. (2025), doi:10.1038/s41597-025-04505-y; CC BY 4.0')
    save(OUT/'report.json',report)
    print('Phase 9 complete; '+selection['recommendation'],flush=True)


if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'run.lock').open('w') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit('PHASE9_ALREADY_RUNNING') from None
        try:
            torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
            with warnings.catch_warnings(),threadpool_limits(limits=2):
                warnings.simplefilter('ignore',RuntimeWarning)
                main()
        except Exception:
            log=ROOT/'local_reports/phase9_failure.log';log.parent.mkdir(parents=True,exist_ok=True)
            log.write_text(traceback.format_exc())
            print('Phase 9: FAIL; details retained locally.')
            raise SystemExit(1) from None
