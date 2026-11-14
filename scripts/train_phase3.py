"""Train only on frozen external train matches; select decoding on development."""
import json
import fcntl
from pathlib import Path
import pickle
import platform
import sys
import numpy as np
import scipy
import sklearn
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from mtah.external import file_digest, manifest
from mtah.prepared import read_half, verify_prepared
from mtah.features import FEATURE_VERSION, extract_features
from mtah.corruption import SCENARIOS, corrupt_inputs
from mtah.model import PossessionModel
from mtah.decoder import decode
from mtah.validation import evaluate_periods
from mtah.official import load_baseline
from check_external_pipeline import nearest


def scenario_inputs(inputs, name, seed):
    p, b = inputs['players'], inputs['ball']
    if name == 'clean':
        return p, b
    return corrupt_inputs(p, b, inputs['teams'], int(inputs['fps']), seed, SCENARIOS[name])


def summary(result):
    return {k: result[k] for k in ('spell_f1_mean', 'macro_f1', 'n_gt_spells', 'n_pred_spells')}


def main():
    config = manifest(ROOT / 'configs/external_split.json')
    settings = json.loads((ROOT / 'configs/phase3.json').read_text())
    target = ROOT / 'outputs/phase3'
    target.mkdir(parents=True, exist_ok=True)
    alive_x, owner_x, labels_all, alive_all = [], [], [], []
    provenance = {}
    # Verify all train/development prerequisites before starting any fit.
    verified = {}
    for split in ('train', 'development'):
        for match in config['split'][split]:
            verified[match] = verify_prepared(ROOT, config, match, split)
            provenance[match] = verified[match][2]
    for index, match in enumerate(config['split']['train']):
        folder, pitch, _ = verified[match]
        for half in (1, 2):
            inputs, labels, alive = read_half(folder, half)
            for name in settings['scenarios']:
                p, b = scenario_inputs(inputs, name, settings['seed']+index*10+half)
                a, o = extract_features(p, b, inputs['teams'], int(inputs['fps']), pitch)
                stride = settings['training_stride']
                alive_x.append(a[::stride].copy()); owner_x.append(o[::stride].copy())
                labels_all.append(labels[::stride].copy()); alive_all.append(alive[::stride].copy())
        print(match + '_training_features: PASS', flush=True)
    model = PossessionModel(max_iter=settings['max_iter']).fit(
        np.concatenate(alive_x), np.concatenate(owner_x),
        np.concatenate(labels_all), np.concatenate(alive_all))
    training_rows = sum(len(x) for x in labels_all)
    del alive_x, owner_x, labels_all, alive_all
    with (target / 'model.pkl').open('wb') as stream:
        pickle.dump(model, stream, protocol=5)
    print('External training: PASS', flush=True)
    probabilities, truths, baselines, smooth_baselines = {}, {}, {}, {}
    reference = load_baseline()
    for match in config['split']['development']:
        folder, pitch, _ = verified[match]
        truths[match] = {}
        for name in settings['scenarios']:
            probabilities[match, name], baselines[match, name], smooth_baselines[match, name] = {}, {}, {}
        for half in (1, 2):
            inputs, labels, _ = read_half(folder, half)
            truths[match][half] = labels
            for name in settings['scenarios']:
                p, b = scenario_inputs(inputs, name, settings['seed']+1000+half)
                a, o = extract_features(p, b, inputs['teams'], int(inputs['fps']), pitch)
                probabilities[match, name][half] = model.predict_proba(a, o)
                baselines[match, name][half] = nearest(p, b, inputs['teams'])
                raw = baselines[match, name][half]
                codes = np.where(raw == 'H', 0, np.where(raw == 'A', 1, -1))
                smooth_baselines[match, name][half] = reference.smooth(codes, 50)
    candidates = []
    for switch, dead in settings['transition_grid']:
        results = {}
        for (match, name), halves in probabilities.items():
            pred = {half: decode(p, switch, dead) for half, p in halves.items()}
            results[match+'_'+name] = summary(evaluate_periods(truths[match], pred))
        mean = float(np.mean([r['spell_f1_mean'] for r in results.values()]))
        macro = float(np.mean([r['macro_f1'] for r in results.values()]))
        candidates.append(dict(switch_penalty=switch, dead_penalty=dead,
                               mean_spell_f1=mean, mean_macro_f1=macro, results=results))
        print(f'Development decoder candidate {switch}/{dead}: scored', flush=True)
    best = max(candidates, key=lambda x: (x['mean_spell_f1'], x['mean_macro_f1']))
    baseline_results = {m+'_'+name: summary(evaluate_periods(truths[m], p))
                        for (m, name), p in baselines.items()}
    smooth_results = {m+'_'+name: summary(evaluate_periods(truths[m], p))
                      for (m, name), p in smooth_baselines.items()}
    pipeline_scores = {}
    for name, results in [('nearest', baseline_results), ('official_smooth50', smooth_results),
                          ('boosted_viterbi', best['results'])]:
        pipeline_scores[name] = dict(
            mean_spell_f1=float(np.mean([r['spell_f1_mean'] for r in results.values()])),
            mean_macro_f1=float(np.mean([r['macro_f1'] for r in results.values()])))
    recommended = max(pipeline_scores, key=lambda name: (
        pipeline_scores[name]['mean_spell_f1'], pipeline_scores[name]['mean_macro_f1']))
    # Verify checkpoint roundtrip on the exact already-computed development features.
    with (target / 'model.pkl').open('rb') as stream:
        restored = pickle.load(stream)
    np.testing.assert_array_equal(model.predict_proba(a, o), restored.predict_proba(a, o))
    report = dict(settings=settings, feature_version=FEATURE_VERSION, selected=best,
                  training_rows=training_rows,
                  versions=dict(python=platform.python_version(), numpy=np.__version__,
                                scipy=scipy.__version__, sklearn=sklearn.__version__),
                  lock_sha256=file_digest(ROOT / 'uv.lock'),
                  candidates=candidates, nearest_baseline=baseline_results,
                  official_smooth50_baseline=smooth_results,
                  pipeline_scores=pipeline_scores, recommended_pipeline=recommended,
                  source_provenance=provenance, model_sha256=file_digest(target / 'model.pkl'),
                  manifest_sha256=file_digest(ROOT / 'configs/external_split.json'),
                  source_hashes={str(p.relative_to(ROOT)): file_digest(p) for p in
                      [Path(__file__), *sorted((ROOT / 'src/mtah').glob('*.py'))]},
                  checkpoint_roundtrip='PASS', holdouts_evaluated=False)
    (target / 'report.json').write_text(json.dumps(report, indent=2))
    selected = {k: best[k] for k in ('switch_penalty', 'dead_penalty')}
    selected['pipeline'] = recommended
    (target / 'selected_config.json').write_text(json.dumps(selected, indent=2))
    print('Development selection and checkpoint replay: PASS', flush=True)


if __name__ == '__main__':
    lock_path = ROOT / 'outputs/phase3/training.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('TRAINING_ALREADY_RUNNING') from None
        with threadpool_limits(limits=2):
            main()
