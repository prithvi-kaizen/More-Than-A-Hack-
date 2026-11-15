from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from mtah.external import file_digest, FROZEN_SPLIT
from mtah.holdout import verify_holdout
from mtah.inference import feature_bundle, verify_hashes
from mtah.official import metrics
from mtah.target import (organizer_contract,load_tactical,compose_submission,
                         predict_halves,write_submission)


class FreezeTests(unittest.TestCase):
    def test_mutation_and_path_escape_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'weights').write_bytes(b'frozen synthetic checkpoint')
            hashes = {'weights':file_digest(root/'weights')}
            verify_hashes(root,hashes)
            (root/'weights').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'FROZEN_ARTIFACT_INTEGRITY_FAILED'):
                verify_hashes(root,hashes)
            for name in ['../weights','/weights']:
                with self.assertRaisesRegex(ValueError,'UNSAFE_FROZEN_PATH'):
                    verify_hashes(root,{name:'unused'})

    def test_final_test_loader_is_not_a_training_loader(self):
        config = {'split':FROZEN_SPLIT}
        for match in ['BAR-ATH','J03WMX','J03WPY']:
            with self.assertRaisesRegex(ValueError,'UNAUTHORIZED_FINAL_HOLDOUT'):
                verify_holdout('/nonexistent',config,match)
        with patch('mtah.holdout.verify_freeze',side_effect=ValueError('NO_FREEZE')) as guard:
            with self.assertRaisesRegex(ValueError,'NO_FREEZE'):
                verify_holdout('/nonexistent',config,'J03WQQ')
            guard.assert_called_once()

    def test_feature_bundle_rejects_wrong_time_base(self):
        with self.assertRaisesRegex(ValueError,'FROZEN_MODEL_REQUIRES_25_FPS'):
            feature_bundle(None,None,None,30,(105,68))


class TargetTests(unittest.TestCase):
    def test_pinned_contract_uses_video_windows_without_atd_offsets(self):
        contract = organizer_contract(ROOT)
        self.assertEqual(contract['n_frames'],199319)
        self.assertEqual(contract['fps'],25)
        self.assertEqual(contract['blocks'],[(1,11185,90651),(2,110808,190197)])

    def test_normalized_units_missingness_and_raw_frame_mapping(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            meta = root/'synthetic.xml'
            meta.write_text('<Root><Team id="alpha"/><Team id="beta"/>'
                            '<Player teamId="alpha"/><Player teamId="beta"/></Root>')
            raw = root/'synthetic.txt'
            raw.write_text('1:0.5,0.25,999;-0.1,1.1,0:NaN,NaN,0\n'
                           '2:0,0,0;1,1,0:0.5,0.5,0\n')
            contract = dict(n_frames=2,pitch=(100,60))
            p,b,t = load_tactical(meta,raw,contract)
            np.testing.assert_allclose(p[0],[[50,15],[-10,66]],atol=1e-5)
            self.assertTrue(np.isnan(b[0]).all())
            np.testing.assert_array_equal(b[1],[50,30])
            self.assertEqual(t.tolist(),['H','A'])
            raw.write_text(raw.read_text().replace('1:','2:',1))
            with self.assertRaises(ValueError):load_tactical(meta,raw,contract)

    def test_half_bounds_are_inclusive_and_outside_is_abstention(self):
        labels = compose_submission({1:list('HAD'),2:list('AHAD')},13,[(1,2,4),(2,8,11)])
        self.assertEqual(''.join(labels),'XXHADXXXAHADX')
        with self.assertRaises(ValueError):
            compose_submission({1:list('HAD')},13,[(1,2,4),(2,8,11)])
        for blocks in [[(1,2,4),(2,4,6)],[(1,2,14)],[(1,5,4)],[]]:
            with self.assertRaises(ValueError):compose_submission({},13,blocks)
        for bad in ['HA','HAX','HAT']:
            with self.assertRaises(ValueError):compose_submission({1:list(bad)},13,[(1,2,4)])

    def test_features_and_decoder_never_span_halftime(self):
        p = np.zeros((13,2,2));p[2:5]=1;p[8:12]=2
        ball = np.zeros((13,2));teams = np.array(['H','A'])
        contract = dict(fps=25,pitch=(100,60),blocks=[(1,2,4),(2,8,11)])
        seen = []
        def bundle(players,*args):
            seen.append(players.copy())
            return players
        class Predictor:
            def predict(self,features):
                return np.full(len(features),'H' if features[0,0,0]==1 else 'A')
        with patch('mtah.inference.feature_bundle',side_effect=bundle):
            labels = predict_halves(Predictor(),p,ball,teams,contract)
        self.assertEqual([len(x) for x in seen],[3,4])
        self.assertEqual(''.join(labels),'XXHHHXXXAAAAX')

    def test_csv_is_canonical_and_officially_valid(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'synthetic.csv'
            write_submission(path,np.array(list('HADX')))
            self.assertEqual(path.read_bytes(),b'frame,label\n0,H\n1,A\n2,D\n3,X\n')
            self.assertIsNone(metrics.validate_submission(metrics.read_label_csv(path),4))
            with self.assertRaises(ValueError):write_submission(path,['H',''])


if __name__ == '__main__':
    unittest.main()
