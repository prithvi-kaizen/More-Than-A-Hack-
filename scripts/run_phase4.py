"""Predeclared match-held-out comparisons, then one development comparison."""
import fcntl
import json
from pathlib import Path
import pickle
import sys
import warnings
import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from mtah.external import FROZEN_SPLIT, file_digest, manifest
from mtah.prepared import verify_prepared, read_half
from mtah.corruption import SCENARIOS, corrupt_inputs
from mtah.structured_noise import structured_inputs
from mtah.phase4 import control_features, boundary_features, spell_weights, boundary_targets, Phase4Model
from mtah.decoder import decode
from mtah.validation import evaluate_periods
from mtah.official import metrics as pm


def scenarios(inputs, name, seed, pitch):
    p, b, t, fps = inputs['players'], inputs['ball'], inputs['teams'], int(inputs['fps'])
    if name == 'clean': return p, b, t
    if name == 'structured': return structured_inputs(p, b, t, fps, seed+5000, pitch)
    p, b = corrupt_inputs(p, b, t, fps, seed, SCENARIOS[name])
    return p, b, t


def cache_match(match, split, config, settings, out):
    folder, pitch, provenance = verify_prepared(ROOT, config, match, split)
    cache = out / 'cache' / match
    cache.mkdir(parents=True, exist_ok=True)
    fingerprint = dict(prepared=provenance['prepared_hashes'], settings=settings,
        sources={p: file_digest(ROOT/p) for p in ('src/mtah/phase4.py', 'src/mtah/features.py',
                    'src/mtah/structured_noise.py', 'src/mtah/corruption.py', 'scripts/run_phase4.py')})
    record = cache / 'provenance.json'
    if record.exists():
        old = json.loads(record.read_text())
        if old['fingerprint'] == fingerprint and all(file_digest(cache/n) == h for n, h in old['hashes'].items()):
            return cache
    index = config['split']['train'].index(match) if split == 'train' else 100
    for half in (1, 2):
        inputs, labels, alive = read_half(folder, half)
        np.savez_compressed(cache/f'half_{half}_labels.npz', labels=labels, alive=alive,
            weights=spell_weights(labels), targets=boundary_targets(labels, settings['boundary_window_frames']))
        for name in settings['scenarios']:
            p, b, t = scenarios(inputs, name, settings['seed']+index*10+half, pitch)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                a, o = control_features(p, b, t, int(inputs['fps']), pitch)
            np.savez_compressed(cache/f'half_{half}_{name}_inputs.npz', alive_features=a, owner_features=o)
    record.write_text(json.dumps(dict(fingerprint=fingerprint,
        hashes={p.name: file_digest(p) for p in cache.glob('*.npz')}), indent=2))
    print(match+'_feature_cache: PASS', flush=True)
    return cache


def load_cache(cache, half, name):
    with np.load(cache/f'half_{half}_{name}_inputs.npz', allow_pickle=False) as f:
        if set(f.files) != {'alive_features', 'owner_features'}:
            raise ValueError('INVALID_PHASE4_INPUT_FIELDS')
        a, o = f['alive_features'], f['owner_features']
    return a, o, boundary_features(a, o)


def training_arrays(matches, caches, settings):
    if not matches or len(set(matches)) != len(matches) or not set(matches).issubset(FROZEN_SPLIT['train']):
        raise ValueError('UNAUTHORIZED_PHASE4_TRAINING_MATCHES')
    arrays = [[] for _ in range(7)]
    stride = settings['training_stride']
    for match in matches:
        for half in (1, 2):
            with np.load(caches[match]/f'half_{half}_labels.npz', allow_pickle=False) as f:
                labels = [f[k] for k in ('labels', 'alive', 'weights', 'targets')]
            for name in settings['scenarios']:
                a, o, b = load_cache(caches[match], half, name)
                for destination, source in zip(arrays, [a, o, b, *labels]):
                    destination.append(source[::stride].copy())
    return [np.concatenate(a) for a in arrays]


def diagnostics(truth, predictions):
    r = evaluate_periods(truth, predictions)
    result = {k: r[k] for k in ('spell_f1_mean', 'macro_f1', 'n_gt_spells', 'n_pred_spells', 'alive_dead_f1')}
    gt = pm.label_codes(np.concatenate([np.append(truth[h], 'X') for h in sorted(truth)]))
    pred = pm.label_codes(np.concatenate([np.append(predictions[h], 'X') for h in sorted(truth)]))
    runs = pm.valid_runs(gt)
    spells, predicted = pm.possession_spells(gt, runs), pm.possession_spells(pred, runs)
    flags = pm.matched_gt_flags(pm.spell_overlaps(spells, predicted), len(spells), 0.5)
    durations = np.array([(b-a)/25 for a, b, _ in spells])
    result['recall_iou_05_by_duration'] = {}
    for lo, hi in [(0, 2), (2, 5), (5, 15), (15, np.inf)]:
        mask = (durations >= lo) & (durations < hi)
        result['recall_iou_05_by_duration'][f'{lo}_{hi}'] = dict(support=int(mask.sum()),
            recall=float(flags[mask].mean()) if mask.any() else None)
    return result


def evaluate_models(match, cache, models, settings):
    truth, predictions = {}, {v: {n: {} for n in settings['scenarios']} for v in models}
    for half in (1, 2):
        with np.load(cache/f'half_{half}_labels.npz', allow_pickle=False) as f: truth[half] = f['labels']
        for name in settings['scenarios']:
            a, o, b = load_cache(cache, half, name)
            for variant, model in models.items():
                if isinstance(model, Phase4Model): probabilities, hazard = model.predict(a, o, b)
                else: probabilities, hazard = model.predict_proba(a, o[:, :15]), None
                predictions[variant][name][half] = decode(probabilities,
                    settings['switch_penalty'], settings['dead_penalty'], hazard, settings['boundary_reduction'])
    return {variant: {name: diagnostics(truth, halves) for name, halves in scenarios.items()}
            for variant, scenarios in predictions.items()}


def main():
    config = manifest(ROOT/'configs/external_split.json')
    settings = json.loads((ROOT/'configs/phase4.json').read_text())
    out = ROOT/'outputs/phase4'
    out.mkdir(parents=True, exist_ok=True)
    caches = {m: cache_match(m, 'train', config, settings, out) for m in config['split']['train']}
    folds = {}
    for held in config['split']['train']:
        training = [m for m in config['split']['train'] if m != held]
        if held in training: raise ValueError('CV_MATCH_LEAKAGE')
        arrays = training_arrays(training, caches, settings)
        models, shared = {}, None
        for variant in settings['variants']:
            model = Phase4Model(variant, settings['max_iter'], settings['boundary_positive_weight']).fit(*arrays, shared_alive=shared)
            shared = model.base.alive_model
            models[variant] = model
        del arrays
        folds[held] = dict(training_matches=training, results=evaluate_models(held, caches[held], models, settings))
        (out/'cv_progress.json').write_text(json.dumps(folds, indent=2))
        print(held+'_match_held_out_fold: PASS', flush=True)
    means = {}
    for variant in settings['variants']:
        results = [f['results'][variant] for f in folds.values()]
        means[variant] = dict(
            mean_spell_f1=float(np.mean([r[n]['spell_f1_mean'] for r in results for n in settings['primary_scenarios']])),
            mean_macro_f1=float(np.mean([r[n]['macro_f1'] for r in results for n in settings['primary_scenarios']])),
            structured_spell_f1=float(np.mean([r['structured']['spell_f1_mean'] for r in results])))
    floor = means['frame']['structured_spell_f1']-settings['structured_regression_tolerance']
    eligible = [v for v in settings['variants'] if means[v]['structured_spell_f1'] >= floor]
    selected = max(eligible, key=lambda v: (means[v]['mean_spell_f1'], means[v]['mean_macro_f1']))
    (out/'cv_selection.json').write_text(json.dumps(dict(settings=settings, folds=folds, means=means, selected=selected), indent=2))
    print('Training-match cross-validation selection: '+selected, flush=True)
    arrays = training_arrays(config['split']['train'], caches, settings)
    model = Phase4Model(selected, settings['max_iter'], settings['boundary_positive_weight']).fit(*arrays)
    del arrays
    with (out/'model.pkl').open('wb') as stream: pickle.dump(model, stream, protocol=5)
    phase3 = json.loads((ROOT/'outputs/phase3/report.json').read_text())
    if file_digest(ROOT/'outputs/phase3/model.pkl') != phase3['model_sha256']:
        raise ValueError('FROZEN_PHASE3_CHECKPOINT_MISMATCH')
    with (ROOT/'outputs/phase3/model.pkl').open('rb') as stream: frozen = pickle.load(stream)
    with (out/'model.pkl').open('rb') as stream: restored = pickle.load(stream)
    development = {}
    for match in config['split']['development']:
        cache = cache_match(match, 'development', config, settings, out)
        development[match] = evaluate_models(match, cache,
            {'phase3': frozen, 'phase4': model, 'replayed_phase4': restored}, settings)
        if development[match]['phase4'] != development[match]['replayed_phase4']:
            raise ValueError('PHASE4_CHECKPOINT_REPLAY_FAILED')
        for name in settings['primary_scenarios']:
            expected = phase3['selected']['results'][match+'_'+name]
            for k in ('spell_f1_mean', 'macro_f1', 'n_gt_spells', 'n_pred_spells'):
                if not np.isclose(development[match]['phase3'][name][k], expected[k], atol=1e-12, rtol=0):
                    raise ValueError('FROZEN_PHASE3_DEVELOPMENT_REPLAY_FAILED')
    dev_means = {v: float(np.mean([r[v][n]['spell_f1_mean'] for r in development.values()
                     for n in settings['primary_scenarios']])) for v in ('phase3', 'phase4')}
    recommendation = 'phase4' if dev_means['phase4'] > dev_means['phase3'] else 'phase3'
    report = dict(settings=settings, cv_folds=folds, cv_means=means, selected_variant=selected,
        development=development, development_means=dev_means, recommended_pipeline=recommendation,
        checkpoint_replay='PASS', frozen_phase3_replay='PASS', model_sha256=file_digest(out/'model.pkl'),
        holdouts_evaluated=False, competition_inference=False,
        cache_provenance={m: json.loads((p/'provenance.json').read_text()) for m,p in caches.items()},
        source_hashes={str(p.relative_to(ROOT)): file_digest(p) for p in
            [Path(__file__), ROOT/'src/mtah/decoder.py', ROOT/'src/mtah/phase4.py', ROOT/'src/mtah/structured_noise.py']})
    (out/'report.json').write_text(json.dumps(report, indent=2))
    print('Development comparison and frozen-checkpoint replay: PASS', flush=True)


if __name__ == '__main__':
    path = ROOT/'outputs/phase4/run.lock'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as lock:
        try: fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: raise SystemExit('PHASE4_ALREADY_RUNNING') from None
        with threadpool_limits(limits=2): main()
