import itertools
import pickle
from pathlib import Path
import sys
import unittest
import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'scripts'))
from mtah.phase4 import control_features, boundary_features, spell_weights, boundary_targets, Phase4Model
from mtah.structured_noise import structured_inputs
from mtah.decoder import decode
from run_phase4 import training_arrays
from audit_external_spells import event_evidence
from audit_sampling_resolution import resolution_counts


class WeightTests(unittest.TestCase):
    def test_sampling_audit_checks_actual_stride_phase(self):
        counts, missed = resolution_counts(list('DH HAAAAAD'.replace(' ','')))
        self.assertEqual(sum(counts.values()),2)
        self.assertEqual(missed,1)

    def test_capped_unit_mean_weights_favor_brief_spells(self):
        labels = np.array(list('H'*500+'A'*10+'D'*40+'H'*2000))
        w = spell_weights(labels)
        self.assertGreater(w[505], w[100])
        self.assertTrue(((w >= 0.25) & (w <= 4)).all())
        self.assertAlmostEqual(float(w[labels != 'D'].mean()), 1, places=6)
        self.assertTrue((w[labels == 'D'] == 1).all())

    def test_empty_and_unknown_labels(self):
        self.assertEqual(len(spell_weights([])), 0)
        np.testing.assert_array_equal(spell_weights(list('XXX')), 1)

    def test_boundaries_exclude_stoppages_and_unknowns(self):
        np.testing.assert_array_equal(boundary_targets(list('HHADDHXA'), 0), [0, 0, 1, 0, 0, 0, 0, 0])
        y = boundary_targets(list('HHHAAA'), 1)
        np.testing.assert_array_equal(y, [0, 0, 1, 1, 1, 0])


class FeatureTests(unittest.TestCase):
    def test_extended_feature_symmetry_and_missingness(self):
        rng = np.random.default_rng(2)
        p = rng.normal(size=(100, 4, 2))+[50, 34]
        b = rng.normal(size=(100, 2))+[50, 34]
        teams = np.array(['H', 'A', 'H', 'A'])
        a, o = control_features(p, b, teams)
        aa, oo = control_features(p, b, np.where(teams == 'H', 'A', 'H'))
        np.testing.assert_allclose(a, aa, equal_nan=True)
        np.testing.assert_allclose(o, -oo, equal_nan=True)
        np.testing.assert_allclose(boundary_features(a, o), boundary_features(aa, oo), equal_nan=True)
        a, o = control_features(p, np.full_like(b, np.nan), teams)
        self.assertFalse(np.isinf(o).any())

    def test_structured_noise_is_input_only_deterministic_and_adds_slots(self):
        p = np.ones((3000, 4, 2), dtype=float)*10
        b = np.ones((3000, 2), dtype=float)*10
        teams = np.array(['H', 'A', 'H', 'A'])
        x, y, t = structured_inputs(p, b, teams, 25, 7)
        xx, yy, tt = structured_inputs(p, b, teams, 25, 7)
        np.testing.assert_equal(x, xx); np.testing.assert_equal(y, yy); np.testing.assert_equal(t, tt)
        self.assertEqual(x.shape, (3000, 8, 2))
        self.assertTrue(np.isfinite(p).all() and np.isfinite(b).all())
        self.assertTrue(np.isnan(y).any())


class AdaptiveDecoderTests(unittest.TestCase):
    def test_exhaustive_dynamic_optimality(self):
        rng = np.random.default_rng(3)
        p, hazard = rng.uniform(.1, 1, (5, 3)), rng.uniform(size=5)
        p /= p.sum(axis=1, keepdims=True)
        def score(s):
            result = sum(np.log(p[i, k]) for i,k in enumerate(s))
            for t in range(1, len(s)):
                if s[t] != s[t-1]: result -= 1.2 if 2 in (s[t],s[t-1]) else .7*(1-.75*hazard[t])
            return result
        chosen = [dict(H=0,A=1,D=2)[x] for x in decode(p,.7,1.2,hazard)]
        self.assertAlmostEqual(score(chosen), max(map(score,itertools.product(range(3),repeat=5))))

    def test_zero_hazard_preserves_frozen_decoder(self):
        p = np.random.default_rng(3).uniform(size=(100, 3))
        np.testing.assert_equal(decode(p,5,40), decode(p,5,40,np.zeros(100)))
        for bad in (np.ones(99), np.ones(100)*2, np.ones(100)*np.nan):
            with self.assertRaises(ValueError): decode(p,5,40,bad)


class TrainingTests(unittest.TestCase):
    def test_training_loader_rejects_development_target_and_holdout(self):
        for matches in (['J03WPY'], ['BAR-ATH'], ['J03WQQ'], ['J03WMX','J03WMX']):
            with self.assertRaisesRegex(ValueError,'UNAUTHORIZED_PHASE4_TRAINING_MATCHES'):
                training_arrays(matches, {}, {})

    def test_models_accept_weights_and_preserve_probability_symmetry(self):
        alive = np.tile([1,1,0],300)
        labels = np.tile(['H','A','D'],300)
        a = alive[:,None].astype(float)
        o = np.tile(np.array([[1],[-1],[0]]),(300,21)).astype(float)
        b = np.column_stack((np.arange(900)%2, np.ones(900)))
        targets = (np.arange(900)%2).astype(np.int8)
        with threadpool_limits(limits=2):
            for variant in ('frame','balanced','control_adaptive'):
                model = Phase4Model(variant,15).fit(a,o,b,labels,alive,np.ones(900),targets)
                prob,h = model.predict(a,o,b)
                flip,hh = model.predict(a,-o,b)
                np.testing.assert_allclose(prob.sum(axis=1),1)
                np.testing.assert_allclose(prob[:,[1,0,2]],flip)
                if h is not None: np.testing.assert_equal(h,hh)
                replay, replay_h = pickle.loads(pickle.dumps(model)).predict(a,o,b)
                np.testing.assert_array_equal(prob,replay)
                if h is not None: np.testing.assert_array_equal(h,replay_h)

    def test_event_evidence_does_not_equate_aerial_duel_with_ground_control(self):
        import xml.etree.ElementTree as ET
        e = ET.fromstring('<Event><TacklingGame Type="air" WinnerTeam="x" PossessionChange="true"/></Event>')
        self.assertEqual(event_evidence(e,{'x':'H'}), [('aerial_duel','H')])
