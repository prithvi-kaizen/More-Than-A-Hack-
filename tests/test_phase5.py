from pathlib import Path
import json
import sys
import tempfile
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'scripts'))
from mtah.motion import player_motion_features,geometry_reliability,apply_policy
try:
    import torch
except ImportError:
    torch=None
else:
    from mtah.temporal import MotionTCN,Windows,train_motion,predict_motion,save_motion,load_motion
    from run_phase5 import select_policy,sequences_for_training


class ReliabilityTests(unittest.TestCase):
    def test_constant_motion_visible_and_spike_missing_quality(self):
        b=np.column_stack((np.arange(100)*.1,np.zeros(100)))
        q=geometry_reliability(b)
        np.testing.assert_allclose(q[5:95],1)
        b[50,0]+=30;b[20:30]=np.nan
        q=geometry_reliability(b)
        self.assertLess(q[50],.001)
        self.assertTrue((q[20:30]==0).all())

    def test_policy_preserves_dead_mass_and_team_symmetry(self):
        p=np.tile([.5,.3,.2],(30,1));h=np.ones(30)*.8;q=np.linspace(0,1,30);m=np.ones(30)*.7
        for variant in ['phase4','reliable_boundary','reliable_tempered','motion_fusion']:
            a,b=apply_policy(p,h,q,variant,m)
            aa,bb=apply_policy(p[:,[1,0,2]],h,q,variant,1-m)
            np.testing.assert_allclose(a.sum(axis=1),1)
            np.testing.assert_array_equal(a[:,2],p[:,2])
            np.testing.assert_allclose(a[:,[1,0,2]],aa)
            np.testing.assert_array_equal(b,bb)
        a,b=apply_policy(p,h,q,'phase4')
        np.testing.assert_array_equal(a,p);np.testing.assert_array_equal(b,h)

    def test_player_only_pooling_is_permutation_invariant_and_equivariant(self):
        rng=np.random.default_rng(8);p=rng.normal(size=(100,4,2))+[50,34]
        t=np.array(['H','A','H','A']);order=[3,0,2,1]
        x=player_motion_features(p,t)
        self.assertEqual(x.shape,(100,108))
        np.testing.assert_allclose(x,player_motion_features(p[:,order],t[order]),atol=1e-6)
        flip=player_motion_features(p,np.where(t=='H','A','H'))
        np.testing.assert_allclose(x[:,:54],-flip[:,:54])
        np.testing.assert_allclose(x[:,54:],flip[:,54:])
        self.assertTrue(np.isfinite(player_motion_features(np.full_like(p,np.nan),t)).all())

    def test_invalid_fusion_data(self):
        p=np.tile([.4,.4,.2],(5,1))
        with self.assertRaises(ValueError):apply_policy(p,np.zeros(4),np.zeros(5),'phase4')
        with self.assertRaises(ValueError):apply_policy(p,np.zeros(5),np.zeros(5),'motion_fusion',np.ones(4))


@unittest.skipIf(torch is None,'optional temporal dependency is not installed')
class TemporalTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(7);torch.use_deterministic_algorithms(True)

    def test_equivariance_and_zero_evidence(self):
        model=MotionTCN(108,8).eval();x=torch.randn(2,100,108)
        flip=x.clone();flip[:,:,:54]*=-1
        with torch.inference_mode():
            np.testing.assert_array_equal(model(x).numpy(),-model(flip).numpy())
            x[:,:,:54]=0
            np.testing.assert_array_equal(model(x).numpy(),0)

    def test_chunk_halo_matches_full_sequence_and_checkpoint(self):
        model=MotionTCN(108,8).eval();x=np.random.default_rng(4).normal(size=(151,108)).astype('float32')
        full=predict_motion(model,x,751,5,1000)
        chunked=predict_motion(model,x,751,5,31)
        np.testing.assert_allclose(full,chunked,atol=2e-6)
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'motion.pt';save_motion(model,path,['J03WMX'])
            replay=load_motion(path)
            np.testing.assert_array_equal(chunked,predict_motion(replay,x,751,5,31))

    def test_windows_never_cross_periods(self):
        sequences=[dict(features=np.ones((150,108),dtype='float32')*i,
            labels=np.array(['H']*150),weights=np.ones(150,dtype='float32')) for i in [1,2]]
        data=Windows(sequences)
        for x,y,w in data:
            self.assertEqual(len(torch.unique(x)),1)
        x,y,w=data[1]
        self.assertTrue((w[:14]==0).all() and (w[-14:]==0).all())

    def test_training_rejects_external_split_leakage(self):
        with self.assertRaises(ValueError):train_motion([],['J03WPY'],{},0)
        with self.assertRaises(ValueError):train_motion([{'match_id':'J03WN1'}],['J03WMX'],{},0)
        with self.assertRaises(ValueError):sequences_for_training(['J03WQQ'],{},{},{})

    def test_loss_excludes_dead_and_unknown_ownership(self):
        labels=np.tile(['H','A','D','X'],16)
        sequence=dict(features=np.ones((64,108),dtype='float32'),labels=labels,
                      weights=np.full(64,2.5,dtype='float32'))
        _,target,weight=Windows([sequence])[0]
        np.testing.assert_array_equal(weight.numpy(),np.tile([2.5,2.5,0,0],16))
        np.testing.assert_array_equal(target.numpy()[weight.numpy()>0],np.tile([1,0],16))

    def test_real_optimizer_is_deterministic(self):
        rng=np.random.default_rng(1);x=rng.normal(size=(160,108)).astype('float32')
        seq=[dict(match_id='J03WMX',features=x,labels=np.where(x[:,0]>0,'H','A'),weights=np.ones(160,dtype='float32'))]
        settings=dict(channels=8,window=64,window_stride=32,batch_size=4,epochs=2,learning_rate=.001,weight_decay=.001)
        a,loss=train_motion(seq,['J03WMX'],settings,44)
        b,again=train_motion(seq,['J03WMX'],settings,44)
        self.assertEqual(loss,again)
        self.assertTrue(all(np.isfinite(loss)))
        for k in a.state_dict():np.testing.assert_array_equal(a.state_dict()[k].numpy(),b.state_dict()[k].numpy())

    def test_selection_guard_rejects_long_spell_regression(self):
        settings=json.loads((ROOT/'configs/phase5.json').read_text())
        def scores(value,long):
            return {name:dict(spell_f1_mean=value,macro_f1=.9,
                recall_iou_05_by_duration={'15_inf':dict(support=10,recall=long)}) for name in settings['scenarios']}
        folds={str(i):dict(results={v:scores(.5,.9) for v in settings['variants']}) for i in range(4)}
        for f in folds.values():f['results']['motion_fusion']=scores(.55,.7)
        chosen,_=select_policy(folds,settings);self.assertEqual(chosen,'phase4')
        for f in folds.values():f['results']['reliable_boundary']=scores(.52,.92)
        chosen,_=select_policy(folds,settings);self.assertEqual(chosen,'reliable_boundary')
