"""Match-separated reliability and ball-free temporal-expert experiments."""
import fcntl
import json
from pathlib import Path
import pickle
import sys
import warnings
import numpy as np
import torch
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from mtah.external import manifest,file_digest,FROZEN_SPLIT
from mtah.prepared import verify_prepared,read_half
from mtah.phase4 import Phase4Model
from mtah.motion import player_motion_features,geometry_reliability,apply_policy
from mtah.temporal import train_motion,predict_motion,save_motion,load_motion
from mtah.decoder import decode
from run_phase4 import scenarios,training_arrays,load_cache,diagnostics


def verify_base_cache(match,split,config):
    folder,pitch,provenance=verify_prepared(ROOT,config,match,split)
    cache=ROOT/'outputs/phase4/cache'/match
    record=json.loads((cache/'provenance.json').read_text())
    if record['fingerprint']['prepared']!=provenance['prepared_hashes']:
        raise ValueError('BASE_CACHE_PREPARED_MISMATCH')
    for name,h in record['fingerprint']['sources'].items():
        if file_digest(ROOT/name)!=h:raise ValueError('FROZEN_PHASE4_FEATURE_SOURCE_CHANGED')
    for name,h in record['hashes'].items():
        if Path(name).name!=name or file_digest(cache/name)!=h:raise ValueError('BASE_CACHE_INTEGRITY_FAILED')
    return folder,pitch,cache


def make_motion_cache(match,split,config,settings,base_settings,out):
    folder,pitch,base=verify_base_cache(match,split,config)
    cache=out/'cache'/match
    cache.mkdir(parents=True,exist_ok=True)
    fingerprint=dict(base=file_digest(base/'provenance.json'),settings=settings,
        source=file_digest(ROOT/'src/mtah/motion.py'),scenario_source=file_digest(ROOT/'scripts/run_phase4.py'))
    record=cache/'provenance.json'
    if record.exists():
        r=json.loads(record.read_text())
        if r['fingerprint']==fingerprint and all(file_digest(cache/n)==h for n,h in r['hashes'].items()):return cache,base
    index=config['split']['train'].index(match) if split=='train' else 100
    for half in (1,2):
        inputs,_,_=read_half(folder,half)
        for name in settings['scenarios']:
            p,b,t=scenarios(inputs,name,base_settings['seed']+index*10+half,pitch)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore',RuntimeWarning)
                features=player_motion_features(p,t,int(inputs['fps']),pitch)
            stride=settings['neural']['sample_stride']
            np.savez_compressed(cache/f'half_{half}_{name}.npz',
                features=features[::stride],reliability=geometry_reliability(b,int(inputs['fps'])))
    record.write_text(json.dumps(dict(fingerprint=fingerprint,
        hashes={p.name:file_digest(p) for p in cache.glob('*.npz')}),indent=2))
    print(match+'_motion_and_quality_cache: PASS',flush=True)
    return cache,base


def sequences_for_training(matches,motion_caches,base_caches,settings):
    if not matches or len(set(matches))!=len(matches) or not set(matches).issubset(FROZEN_SPLIT['train']):
        raise ValueError('UNAUTHORIZED_TEMPORAL_TRAINING_MATCHES')
    sequences=[];stride=settings['neural']['sample_stride']
    for match in matches:
        for half in (1,2):
            with np.load(base_caches[match]/f'half_{half}_labels.npz',allow_pickle=False) as f:
                labels,weights=f['labels'][::stride],f['weights'][::stride]
            for name in settings['scenarios']:
                with np.load(motion_caches[match]/f'half_{half}_{name}.npz',allow_pickle=False) as f:
                    features=f['features']
                sequences.append(dict(match_id=match,half=half,scenario=name,features=features,
                                      labels=labels,weights=weights))
    return sequences


def evaluate(match,tree,neural,motion_cache,base_cache,variants,settings,base_settings,out=None):
    truth={};pred={v:{n:{} for n in settings['scenarios']} for v in variants}
    for half in (1,2):
        with np.load(base_cache/f'half_{half}_labels.npz',allow_pickle=False) as f:truth[half]=f['labels']
        for name in settings['scenarios']:
            a,o,b=load_cache(base_cache,half,name)
            p,h=tree.predict(a,o,b)
            with np.load(motion_cache/f'half_{half}_{name}.npz',allow_pickle=False) as f:
                features,q=f['features'],f['reliability']
            motion=None if neural is None else predict_motion(neural,features,len(p),
                settings['neural']['sample_stride'],settings['neural']['inference_chunk'])
            if out is not None:
                out.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(out/f'half_{half}_{name}.npz',tree_probabilities=p,
                    boundary_scores=h,reliability=q,motion_home=motion)
            for variant in variants:
                pp,hh=apply_policy(p,h,q,variant,motion,settings['motion_max_weight'],settings['missing_geometry_logit_scale'])
                pred[variant][name][half]=decode(pp,base_settings['switch_penalty'],
                    base_settings['dead_penalty'],hh,base_settings['boundary_reduction'])
    return {v:{n:diagnostics(truth,halves) for n,halves in names.items()} for v,names in pred.items()}


def primary_mean(results,names,key='spell_f1_mean'):
    return float(np.mean([r[n][key] for r in results for n in names]))


def long_recall(results):
    bins=[r['structured']['recall_iou_05_by_duration']['15_inf'] for r in results]
    support=sum(b['support'] for b in bins)
    return sum(b['support']*(b['recall'] or 0) for b in bins)/max(support,1)


def select_policy(folds,settings):
    means={}
    for variant in settings['variants']:
        rs=[f['results'][variant] for f in folds.values()]
        means[variant]=dict(mean_spell_f1=primary_mean(rs,settings['primary_scenarios']),
            mean_macro_f1=primary_mean(rs,settings['primary_scenarios'],'macro_f1'),
            structured_spell_f1=primary_mean(rs,['structured']),structured_long_recall=long_recall(rs))
    base=means['phase4'];eligible=['phase4']
    for v,m in means.items():
        if v=='phase4':continue
        wins=sum(primary_mean([f['results'][v]],settings['primary_scenarios'])>
                 primary_mean([f['results']['phase4']],settings['primary_scenarios']) for f in folds.values())
        m['winning_folds']=wins
        m['eligible']=(m['mean_spell_f1']>=base['mean_spell_f1']+settings['minimum_cv_gain'] and
            wins>=settings['minimum_winning_folds'] and
            m['structured_spell_f1']>=base['structured_spell_f1']-settings['structured_regression_tolerance'] and
            m['structured_long_recall']>=base['structured_long_recall']-settings['structured_long_recall_tolerance'])
        if m['eligible']:eligible.append(v)
    selected=max(eligible,key=lambda v:(means[v]['mean_spell_f1'],means[v]['mean_macro_f1']))
    return selected,means


def main():
    settings=json.loads((ROOT/'configs/phase5.json').read_text())
    config=manifest(ROOT/'configs/external_split.json')
    previous=json.loads((ROOT/'outputs/phase4/report.json').read_text());base_settings=previous['settings']
    if file_digest(ROOT/'outputs/phase4/model.pkl')!=previous['model_sha256']:
        raise ValueError('FROZEN_PHASE4_CHECKPOINT_MISMATCH')
    out=ROOT/'outputs/phase5';out.mkdir(parents=True,exist_ok=True)
    motion_caches,base_caches={},{}
    for m in config['split']['train']:
        motion_caches[m],base_caches[m]=make_motion_cache(m,'train',config,settings,base_settings,out)
    folds={}
    for index,held in enumerate(config['split']['train']):
        training=[m for m in config['split']['train'] if m!=held]
        arrays=training_arrays(training,base_caches,base_settings)
        tree=Phase4Model('control_adaptive',base_settings['max_iter'],base_settings['boundary_positive_weight']).fit(*arrays)
        del arrays
        sequences=sequences_for_training(training,motion_caches,base_caches,settings)
        neural,losses=train_motion(sequences,training,settings['neural'],settings['seed']+index)
        del sequences
        fold_path=out/'oof'/held;fold_path.mkdir(parents=True,exist_ok=True)
        save_motion(neural,fold_path/'motion.pt',training)
        with (fold_path/'tree.pkl').open('wb') as f:pickle.dump(tree,f,protocol=5)
        results=evaluate(held,tree,neural,motion_caches[held],base_caches[held],settings['variants'],
            settings,base_settings,fold_path)
        for name in settings['scenarios']:
            expected=previous['cv_folds'][held]['results']['control_adaptive'][name]
            for k in ('spell_f1_mean','macro_f1','n_gt_spells','n_pred_spells'):
                if not np.isclose(results['phase4'][name][k],expected[k],atol=1e-12,rtol=0):
                    raise ValueError('FROZEN_PHASE4_FOLD_REPLAY_FAILED')
        folds[held]=dict(training_matches=training,results=results,temporal_epoch_losses=losses,
            tree_sha256=file_digest(fold_path/'tree.pkl'),motion_sha256=file_digest(fold_path/'motion.pt'))
        (out/'cv_progress.json').write_text(json.dumps(folds,indent=2))
        print(held+'_phase5_fold: PASS',flush=True)
    selected,means=select_policy(folds,settings)
    (out/'cv_selection.json').write_text(json.dumps(dict(selected=selected,means=means),indent=2))
    print('Cross-validation selection: '+selected,flush=True)
    with (ROOT/'outputs/phase4/model.pkl').open('rb') as f:tree=pickle.load(f)
    neural=None
    if selected=='motion_fusion':
        seq=sequences_for_training(config['split']['train'],motion_caches,base_caches,settings)
        neural,full_losses=train_motion(seq,config['split']['train'],settings['neural'],settings['seed']+100)
        del seq
        save_motion(neural,out/'motion.pt',config['split']['train'])
    development={}
    for m in config['split']['development']:
        mc,bc=make_motion_cache(m,'development',config,settings,base_settings,out)
        variants=list(dict.fromkeys(['phase4',selected]))
        development[m]=evaluate(m,tree,neural,mc,bc,variants,settings,base_settings)
        for name in settings['scenarios']:
            for k in ('spell_f1_mean','macro_f1','n_gt_spells','n_pred_spells'):
                if not np.isclose(development[m]['phase4'][name][k],previous['development'][m]['phase4'][name][k],atol=1e-12,rtol=0):
                    raise ValueError('FROZEN_PHASE4_DEVELOPMENT_REPLAY_FAILED')
    base=[d['phase4'] for d in development.values()];candidate=[d[selected] for d in development.values()]
    accepted=(selected!='phase4' and primary_mean(candidate,settings['primary_scenarios'])>
        primary_mean(base,settings['primary_scenarios']) and
        primary_mean(candidate,['structured'])>=primary_mean(base,['structured'])-settings['structured_regression_tolerance'] and
        long_recall(candidate)>=long_recall(base)-settings['structured_long_recall_tolerance'])
    recommended=selected if accepted else 'phase4'
    report=dict(settings=settings,cv_folds=folds,cv_means=means,selected_policy=selected,
        development=development,recommended_policy=recommended,holdouts_evaluated=False,competition_inference=False,
        frozen_phase4_sha256=previous['model_sha256'],torch_version=torch.__version__,
        source_hashes={str(p.relative_to(ROOT)):file_digest(p) for p in [Path(__file__),ROOT/'src/mtah/motion.py',ROOT/'src/mtah/temporal.py']},
        lock_sha256=file_digest(ROOT/'uv.lock'),neural_training='from_scratch_external_training_matches_only')
    if neural is not None:
        report['full_temporal_epoch_losses']=full_losses
        report['full_temporal_sha256']=file_digest(out/'motion.pt')
    (out/'report.json').write_text(json.dumps(report,indent=2))
    print('Development comparison and Phase 4 replay: PASS; recommended '+recommended,flush=True)


if __name__=='__main__':
    lock_path=ROOT/'outputs/phase5/run.lock';lock_path.parent.mkdir(parents=True,exist_ok=True)
    with lock_path.open('w') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit('PHASE5_ALREADY_RUNNING') from None
        torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
        with threadpool_limits(limits=2):main()
