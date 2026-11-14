import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from mtah.external import label_from_status, load_idsse, manifest
from mtah.corruption import Corruption, corrupt_inputs
from mtah.validation import evaluate_periods


class ManifestTests(unittest.TestCase):
    def test_frozen_split_and_catalogue(self):
        config = manifest(ROOT / 'configs/external_split.json')
        self.assertEqual([len(config['split'][s]) for s in ('train', 'development', 'holdout')], [4, 1, 2])

    def test_duplicate_or_target_match_rejected(self):
        config = manifest(ROOT / 'configs/external_split.json')
        for match in ('J03WMX', 'BAR-ATH'):
            with tempfile.TemporaryDirectory() as temp:
                config['split']['holdout'][0] = match
                path = Path(temp) / 'manifest.json'
                path.write_text(json.dumps(config))
                with self.assertRaises(ValueError):
                    manifest(path)

    def test_repartitioning_rejected(self):
        config = manifest(ROOT / 'configs/external_split.json')
        config['split']['train'][0], config['split']['development'][0] = (
            config['split']['development'][0], config['split']['train'][0])
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'manifest.json'
            path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, 'INVALID_EXTERNAL_SPLIT'):
                manifest(path)


class LabelTests(unittest.TestCase):
    def test_status_precedes_ownership(self):
        self.assertEqual(label_from_status('0', '1'), ('D', 0))
        self.assertEqual(label_from_status('1', '1'), ('H', 1))
        self.assertEqual(label_from_status('1', '2'), ('A', 1))

    def test_unknown_is_neither_dead_nor_away(self):
        self.assertEqual(label_from_status('1', '0'), ('X', 1))
        self.assertEqual(label_from_status(None, '2'), ('X', -1))

    def fixture(self, folder):
        meta = folder / 'meta.xml'
        meta.write_text('<Root><General MatchId="DFL-MAT-J03WMX"/>'
            '<Environment PitchX="100" PitchY="68"/><Teams>'
            '<Team Role="guest"><Players><Player PersonId="a"/></Players></Team>'
            '<Team Role="home"><Players><Player PersonId="h"/></Players></Team>'
            '</Teams></Root>')
        tracking = folder / 'positions.xml'
        tracking.write_text('<Root>'
            '<FrameSet MatchId="DFL-MAT-J03WMX" GameSection="firstHalf" TeamId="BALL">'
            '<Frame N="10" X="0" Y="0" BallStatus="1" BallPossession="1"/>'
            '<Frame N="12" X="1" Y="0" BallStatus="0" BallPossession="2"/>'
            '</FrameSet>'
            '<FrameSet MatchId="DFL-MAT-J03WMX" GameSection="firstHalf" PersonId="h">'
            '<Frame N="10" X="-2" Y="3"/><Frame N="11" X="-1" Y="3"/>'
            '</FrameSet>'
            '<FrameSet MatchId="DFL-MAT-J03WMX" GameSection="secondHalf" TeamId="BALL">'
            '<Frame N="100" X="2" Y="0" BallStatus="1" BallPossession="0"/>'
            '</FrameSet></Root>')
        return meta, tracking

    def test_adapter_gaps_coordinates_roles_and_halves(self):
        with tempfile.TemporaryDirectory() as temp:
            meta, tracking = self.fixture(Path(temp))
            data = load_idsse(meta, tracking, 'J03WMX')
            self.assertEqual(set(data), {1, 2})
            self.assertEqual(data[1].labels.tolist(), ['H', 'X', 'D'])
            self.assertEqual(data[1].alive.tolist(), [1, -1, 0])
            self.assertTrue(np.isnan(data[1].ball[1]).all())
            self.assertEqual(data[1].ball[0].tolist(), [50, 34])
            self.assertEqual(data[1].teams.tolist(), ['A', 'H'])
            self.assertEqual(data[1].players[0, 1].tolist(), [48, 37])
            self.assertEqual(data[2].labels.tolist(), ['X'])

    def test_wrong_game_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            meta, tracking = self.fixture(Path(temp))
            with self.assertRaisesRegex(ValueError, 'EXTERNAL_MATCH_ID_MISMATCH'):
                load_idsse(meta, tracking, 'BAR-ATH')

    def test_duplicate_ball_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            meta, tracking = self.fixture(Path(temp))
            text = tracking.read_text().replace('N="12"', 'N="10"')
            tracking.write_text(text)
            with self.assertRaises(ValueError):
                load_idsse(meta, tracking, 'J03WMX')


class CorruptionTests(unittest.TestCase):
    def test_invalid_corruption_parameters_rejected(self):
        for kwargs in ({'position_noise_metres': -1}, {'ball_gap_seconds': (3, 1)}):
            with self.assertRaises(ValueError):
                Corruption(**kwargs)

    def test_seed_determinism_and_no_input_mutation(self):
        players, ball = np.ones((1500, 4, 2)), np.ones((1500, 2))
        teams = np.array(['H', 'H', 'A', 'A'])
        a, b = corrupt_inputs(players, ball, teams, 25, 7)
        c, d = corrupt_inputs(players, ball, teams, 25, 7)
        np.testing.assert_equal(a, c)
        np.testing.assert_equal(b, d)
        self.assertTrue(np.isfinite(players).all() and np.isfinite(ball).all())
        self.assertTrue(np.isnan(b).any())

    def test_track_swaps_preserve_team_sets(self):
        players = np.tile(np.arange(4).reshape(1, 4, 1), (1500, 1, 2)).astype(float)
        teams = np.array(['H', 'H', 'A', 'A'])
        config = Corruption(ball_gap_starts_per_minute=0, player_gap_starts_per_minute=0,
                            position_noise_metres=0, track_swaps_per_minute=100)
        changed, _ = corrupt_inputs(players, np.zeros((1500, 2)), teams, 25, 9, config)
        for slots in ([0, 1], [2, 3]):
            np.testing.assert_equal(np.sort(changed[:, slots, 0], axis=1),
                                    np.sort(players[:, slots, 0], axis=1))


class ValidationTests(unittest.TestCase):
    def test_halves_never_merge(self):
        truth = {1: np.array(list('HH')), 2: np.array(list('HH'))}
        result = evaluate_periods(truth, truth)
        self.assertEqual(result['n_gt_spells'], 2)
        self.assertEqual(result['spell_f1_mean'], 1)

    def test_missing_period_or_length_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_periods({1: np.array(['H'])}, {})
        with self.assertRaises(ValueError):
            evaluate_periods({1: np.array(['H'])}, {1: np.array(['H', 'H'])})
