import gzip
import hashlib
import itertools
from pathlib import Path
import pickle
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from mtah.features import extract_features, speed
from mtah.decoder import decode
from mtah.model import PossessionModel
from mtah.prepared import read_half, verify_prepared
from mtah.external import manifest, verify_raw, file_digest
from mtah.external import load_idsse
from mtah.official import load_baseline


class FeatureTests(unittest.TestCase):
    def fixture(self):
        rng = np.random.default_rng(9)
        p = rng.normal(size=(100, 4, 2))+[50, 34]
        b = rng.normal(size=(100, 2))+[50, 34]
        return p, b, np.array(['H', 'A', 'H', 'A'])

    def test_slot_order_invariance(self):
        p, b, t = self.fixture()
        a, o = extract_features(p, b, t)
        order = [3, 1, 0, 2]
        aa, oo = extract_features(p[:, order], b, t[order])
        np.testing.assert_allclose(a, aa, equal_nan=True)
        np.testing.assert_allclose(o, oo, equal_nan=True)

    def test_team_symmetry(self):
        p, b, t = self.fixture()
        a, o = extract_features(p, b, t)
        aa, oo = extract_features(p, b, np.where(t == 'H', 'A', 'H'))
        np.testing.assert_allclose(a, aa, equal_nan=True)
        np.testing.assert_allclose(o, -oo, equal_nan=True)

    def test_missing_ball_does_not_become_dead_label(self):
        p, b, t = self.fixture()
        b[:] = np.nan
        a, o = extract_features(p, b, t)
        self.assertTrue((a[:, 0] == 0).all())
        self.assertTrue(np.isnan(o[:, -1]).all())
        self.assertFalse(np.isinf(a).any() or np.isinf(o).any())

    def test_speed_units_and_gap_rejection(self):
        xy = np.column_stack((np.arange(100)*0.2, np.zeros(100)))
        v = speed(xy, 25)
        np.testing.assert_allclose(v[20:80], 5, atol=1e-5)
        xy[45:50] = np.nan
        self.assertTrue(np.isnan(speed(xy, 25)[45:55]).all())


class DecoderTests(unittest.TestCase):
    def test_official_reference_missingness_semantics(self):
        labels = load_baseline().smooth(np.array([0, -1, -1]), 1)
        self.assertEqual(labels.tolist(), ['H', 'H', 'D'])

    def test_exhaustive_optimality(self):
        rng = np.random.default_rng(3)
        for n in range(1, 6):
            p = rng.uniform(0.1, 1, (n, 3))
            p /= p.sum(axis=1, keepdims=True)
            transition = np.full((3, 3), -0.7)
            transition[2, :2] = transition[:2, 2] = -1.2
            np.fill_diagonal(transition, 0)
            def score(s):
                return sum(np.log(p[i, k]) for i, k in enumerate(s)) + sum(
                    transition[s[i-1], s[i]] for i in range(1, n))
            labels = decode(p, 0.7, 1.2)
            chosen = [dict(H=0, A=1, D=2)[x] for x in labels]
            self.assertAlmostEqual(score(chosen), max(map(score, itertools.product(range(3), repeat=n))))

    def test_zero_penalty_is_argmax(self):
        p = np.random.default_rng(8).uniform(size=(20, 3))
        np.testing.assert_array_equal(decode(p, 0), np.array(['H', 'A', 'D'])[p.argmax(axis=1)])

    def test_flicker_suppression_and_short_spell(self):
        p = np.tile([0.99, 0.005, 0.005], (20, 1))
        p[10] = [0.39, 0.01, 0.6]
        self.assertTrue((decode(p, 4) == 'H').all())
        p[8:11] = [0.005, 0.99, 0.005]
        self.assertTrue((decode(p, 4)[8:11] == 'A').all())

    def test_invalid_probabilities(self):
        for p in (np.ones((2, 2)), np.full((2, 3), np.nan), np.zeros((2, 3))):
            with self.assertRaises(ValueError):
                decode(p)
        for penalty in (-1, np.nan, np.inf):
            with self.assertRaises(ValueError):
                decode(np.ones((2, 3)), penalty)


class ModelTests(unittest.TestCase):
    def test_probabilities_symmetry_and_checkpoint(self):
        alive = np.tile([1, 1, 0], 300)
        labels = np.tile(['H', 'A', 'D'], 300)
        a = alive[:, None].astype(float)
        o = np.tile([[1.], [-1.], [0.]], (300, 1))
        with threadpool_limits(limits=2):
            model = PossessionModel(max_iter=15).fit(a, o, labels, alive)
            p, flip = model.predict_proba(a, o), model.predict_proba(a, -o)
            np.testing.assert_allclose(p.sum(axis=1), 1)
            np.testing.assert_allclose(p[:, [1, 0, 2]], flip)
            self.assertGreater((np.array(['H', 'A', 'D'])[p.argmax(axis=1)] == labels).mean(), 0.95)
            restored = pickle.loads(pickle.dumps(model))
            np.testing.assert_array_equal(p, restored.predict_proba(a, o))


class DataGuardTests(unittest.TestCase):
    def test_wrong_server_range_never_appended(self):
        import download_external
        import subprocess
        entry = dict(name='external.xml', size=8, md5=hashlib.md5(b'abcdefgh').hexdigest())
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            parts = root / 'external_data/idsse/raw/external.xml.parts'
            parts.mkdir(parents=True)
            (parts/'0').write_bytes(b'abcd')
            def transfer(command, **kwargs):
                Path(command[command.index('-o')+1]).write_bytes(b'BAD!')
                Path(command[command.index('-D')+1]).write_text('Content-Range: bytes 0-3/8\n')
                return subprocess.CompletedProcess(command, 0)
            with patch.object(download_external, 'ROOT', root), patch.object(
                    download_external.subprocess, 'run', side_effect=transfer), patch.object(
                    download_external.time, 'sleep'):
                with self.assertRaisesRegex(ValueError, 'EXTERNAL_RANGE_TRANSFER_INCOMPLETE'):
                    download_external.download(entry, mirror=True, workers=1)
            self.assertEqual((parts/'0').read_bytes(), b'abcd')

    def test_resumed_transfer_retains_valid_partial_prefix(self):
        import download_external
        import subprocess
        raw = b'abcdefgh'
        entry = dict(name='external.xml', size=8, md5=hashlib.md5(raw).hexdigest())
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            parts = root / 'external_data/idsse/raw/external.xml.parts'
            parts.mkdir(parents=True)
            (parts/'0').write_bytes(raw[:4])
            requests = []
            def transfer(command, **kwargs):
                interval = command[command.index('--range')+1]
                requests.append(interval)
                offset, stop = map(int, interval.split('-'))
                data = raw[offset:offset+2]
                Path(command[command.index('-o')+1]).write_bytes(data)
                Path(command[command.index('-D')+1]).write_text(
                    f'HTTP/1.1 206 Partial Content\nContent-Range: bytes {offset}-{stop}/8\n')
                return subprocess.CompletedProcess(command, 18)
            with patch.object(download_external, 'ROOT', root), patch.object(
                    download_external.subprocess, 'run', side_effect=transfer), patch.object(
                    download_external.time, 'sleep'):
                download_external.download(entry, mirror=True, workers=1)
            self.assertEqual(requests, ['4-7', '6-7'])
            self.assertEqual((parts.parent/'external.xml').read_bytes(), raw)

    def test_holdout_and_target_rejected_before_file_access(self):
        config = manifest(ROOT / 'configs/external_split.json')
        for match, split in [('BAR-ATH', 'train'), ('J03WQQ', 'holdout'), ('J03WPY', 'train')]:
            with self.assertRaisesRegex(ValueError, 'UNAUTHORIZED_MODEL_DATA_SPLIT'):
                verify_prepared('/nonexistent', config, match, split)

    def test_label_field_cannot_enter_input_archive(self):
        with tempfile.TemporaryDirectory() as temp:
            np.savez(Path(temp)/'half_1_inputs.npz', players=np.zeros((1, 2, 2)), labels=['H'])
            with self.assertRaisesRegex(ValueError, 'UNEXPECTED_MODEL_INPUT_FIELDS'):
                read_half(temp, 1)

    def test_compressed_raw_integrity(self):
        raw = b'<Root>synthetic external XML</Root>'
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'external.xml.gz'
            with gzip.open(path, 'wb') as stream:
                stream.write(raw)
            self.assertTrue(verify_raw(path, dict(size=len(raw), md5=hashlib.md5(raw).hexdigest())))
            self.assertEqual(file_digest(path), hashlib.sha256(raw).hexdigest())

    def test_gzip_adapter_matches_uncompressed(self):
        from test_external import LabelTests
        with tempfile.TemporaryDirectory() as temp:
            meta, positions = LabelTests().fixture(Path(temp))
            compressed = positions.with_suffix('.xml.gz')
            with gzip.open(compressed, 'wb') as stream:
                stream.write(positions.read_bytes())
            plain, zipped = load_idsse(meta, positions, 'J03WMX'), load_idsse(meta, compressed, 'J03WMX')
            for half in plain:
                np.testing.assert_equal(plain[half].players, zipped[half].players)
                np.testing.assert_equal(plain[half].labels, zipped[half].labels)
