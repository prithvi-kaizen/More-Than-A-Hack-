import json
from pathlib import Path
import sys
import unittest
import tempfile
from unittest.mock import patch
import numpy as np
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'scripts'))
from mtah.reliability import stress_inputs,validity_features,validity_targets,training_partition,ValidityModel
try:
    import torch
except ImportError:
    torch=None
else:
    from run_phase9 import metric_summary,select
    import parallel_phase9

SETTINGS=json.loads((ROOT/'configs/phase9.json').read_text())


class StressTests(unittest.TestCase):
    def fixture(self):
        rng=np.random.default_rng(8)
        p=rng.uniform([0,0],[105,68],(1000,6,2)).astype(np.float32)
        b=np.column_stack((np.linspace(20,80,1000),np.full(1000,30))).astype(np.float32)
        return p,b,np.array(['H','A','H','A','H','A'])

    def test_deterministic_input_only_and_nonmutating(self):
        p,b,t=self.fixture();old=p.copy(),b.copy(),t.copy()
        for scene in SETTINGS['evaluation_scenarios']:
            a=stress_inputs(p,b,t,25,9,(105,68),scene,SETTINGS)
            c=stress_inputs(p,b,t,25,9,(105,68),scene,SETTINGS)
            for x,y in zip(a,c):np.testing.assert_array_equal(x,y)
        for x,y in zip((p,b,t),old):np.testing.assert_array_equal(x,y)

    def test_independent_camera_masks_do_not_follow_ball(self):
        p,b,t=self.fixture()
        a=stress_inputs(p,b,t,25,9,(105,68),'camera_independent',SETTINGS)
        c=stress_inputs(p,b+[40,0],t,25,9,(105,68),'camera_independent',SETTINGS)
        np.testing.assert_array_equal(a[0],c[0])
        np.testing.assert_array_equal(a[2],c[2])
        self.assertGreater(a[0].shape[1],p.shape[1])
        d=stress_inputs(p,b+[40,0],t,25,9,(105,68),'camera_ball',SETTINGS)
        self.assertFalse(np.array_equal(np.isnan(a[0]),np.isnan(d[0])))

    def test_sustained_false_tracks_and_long_outages(self):
        p,b,t=self.fixture()
        settings=dict(SETTINGS,false_tracks_per_minute=60,outages_per_minute=60)
        _,shifted,_=stress_inputs(p,b,t,25,9,(105,68),'sustained',settings)
        bad=np.linalg.norm(shifted-b,axis=1)>2
        self.assertGreater(bad.sum(),50)
        _,missing,_=stress_inputs(p,b,t,25,9,(105,68),'outage_20',settings)
        self.assertGreater(np.isnan(missing).all(axis=1).sum(),100)

    def test_geometry_features_are_slot_and_team_invariant(self):
        p,b,t=self.fixture();order=[3,1,4,0,5,2]
        x=validity_features(p,b)
        np.testing.assert_allclose(x,validity_features(p[:,order],b),equal_nan=True,atol=1e-6)
        self.assertEqual(x.shape,(1000,10))

    def test_validity_labels_distinguish_original_unknowns(self):
        original=np.array([[0.,0],[0,0],[np.nan,np.nan],[0,0]])
        observed=np.array([[1.,0],[3,0],[1,1],[np.nan,np.nan]])
        np.testing.assert_array_equal(validity_targets(original,observed,2),[1,0,-1,0])


class CalibrationTests(unittest.TestCase):
    def test_fold_and_calibration_guards(self):
        for held in ['J03WMX','J03WN1','J03WOH','J03WOY']:
            fits,cal=training_partition(held)
            self.assertNotIn(held,fits+[cal]);self.assertNotIn(cal,fits)
        for held in ['BAR-ATH','J03WPY','J03WQQ','J03WR9']:
            with self.assertRaises(ValueError):training_partition(held)
        with self.assertRaisesRegex(ValueError,'SPLIT_LEAKAGE'):
            ValidityModel().fit({}, {}, ['J03WMX'],'J03WMX',SETTINGS)
        with self.assertRaisesRegex(ValueError,'ARRAY_SPLIT'):
            ValidityModel().fit({'J03WQQ':None},{'J03WN1':None},['J03WMX'],'J03WN1',SETTINGS)

    def test_fit_quality_and_checkpoint_roundtrip(self):
        import pickle
        rng=np.random.default_rng(7);x=rng.normal(size=(500,10));y=(x[:,0]>0).astype(int)
        settings=dict(SETTINGS,classifier=dict(SETTINGS['classifier'],max_iter=5,min_samples_leaf=10))
        with threadpool_limits(limits=2):
            m=ValidityModel().fit({'J03WMX':(x,y)},{'J03WN1':(x+.1,y)},['J03WMX'],'J03WN1',settings)
            q=np.linspace(0,1,len(x));a=m.quality(x,q)
            self.assertTrue(((a>=0)&(a<=q)).all())
            np.testing.assert_array_equal(a,pickle.loads(pickle.dumps(m)).quality(x,q))


@unittest.skipIf(torch is None,'optional temporal dependency is not installed')
class EvaluationTests(unittest.TestCase):
    def test_parallel_worker_verifies_protocol_without_writing(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);out=root/'outputs';out.mkdir();(root/'docs').mkdir()
            protocol=out/'protocol_frozen.json'
            protocol.write_text(json.dumps(dict(settings=SETTINGS,sources={})))
            (root/'docs/phase5_results.json').write_text('{"synthetic": true}')
            before=protocol.read_bytes()
            with patch.object(parallel_phase9,'ROOT',root), \
                    patch.object(parallel_phase9.run,'OUT',out), \
                    patch.object(parallel_phase9.run,'verify_freeze'), \
                    patch.object(parallel_phase9.run,'save',side_effect=AssertionError('worker wrote protocol')):
                self.assertEqual(parallel_phase9.verify_worker_protocol(SETTINGS),{'synthetic':True})
                with self.assertRaisesRegex(ValueError,'SETTINGS_CHANGED'):
                    parallel_phase9.verify_worker_protocol(dict(SETTINGS,schema=999))
            self.assertEqual(protocol.read_bytes(),before)

    def test_official_metric_period_isolation_and_timing(self):
        truth={1:np.array(list('HHDAAHH')),2:np.array(list('HHDAA'))}
        m=metric_summary(truth,truth)
        self.assertEqual(m['spell_f1_mean'],1)
        self.assertEqual(m['n_gt_spells'],5)
        self.assertEqual(m['false_d_runs_inside_true_possessions'],0)
        self.assertEqual(m['transition_timing']['turnover']['matched'],1)
        self.assertEqual(m['transition_timing']['restart']['matched'],2)
        self.assertTrue(all(v==1 for v in m['per_iou'].values()))

    def test_adoption_requires_gain_and_independent_camera_guard(self):
        def folds(gain,camera_gain):
            r={}
            for match in ['a','b','c','d']:
                r[match]={}
                for scene in SETTINGS['evaluation_scenarios']:
                    r[match][scene]={}
                    for seed in SETTINGS['evaluation_seeds']:
                        r[match][scene][str(seed)]={}
                        for policy in ('v1','challenger','v1_no_motion','challenger_no_motion'):
                            delta=(camera_gain if scene=='camera_independent' else gain) if policy.startswith('challenger') else 0
                            r[match][scene][str(seed)][policy]=dict(spell_f1_mean=.5+delta,
                                duration_recall={'long':dict(support=10,hits=8)})
            return r
        self.assertTrue(select(folds(.02,.02),SETTINGS)['accepted'])
        self.assertFalse(select(folds(.02,-.001),SETTINGS)['accepted'])
        self.assertFalse(select(folds(.001,.001),SETTINGS)['accepted'])
