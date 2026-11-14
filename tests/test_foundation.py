import hashlib
import json
import math
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from mtah.inputs import InputError, iter_tracking, position_metres, team_membership, video_frame_of
from mtah.official import metrics as pm
from check_staged_files import allowed


def codes(text):
    return pm.label_codes(list(text))


class MetricsTests(unittest.TestCase):
    def test_perfect_with_halftime(self):
        result = pm.evaluate(codes('HHDAAXXDDHH'), codes('HHDAAXXDDHH'))
        self.assertEqual(result['spell_f1_mean'], 1)
        self.assertEqual(result['macro_f1'], 1)

    def test_same_team_restart_is_two_spells(self):
        gt = codes('HHDDHH')
        self.assertEqual(len(pm.possession_spells(gt, pm.valid_runs(gt))), 2)

    def test_abstention_carries_only_inside_run(self):
        gt, pred = codes('HHXXAA'), codes('HXXXXA')
        filled = pm.fill_abstentions(pred, pm.valid_runs(gt))
        self.assertEqual(filled.tolist(), codes('HHXXAA').tolist())

    def test_abstention_preserves_temporal_but_costs_frame_score(self):
        gt = codes('HHHHAAAADD')
        result = pm.evaluate(gt, codes('HXXHAAAADD'))
        self.assertEqual(result['spell_f1_mean'], 1)
        self.assertLess(result['macro_f1'], 1)

    def test_false_dead_ball_fragmentation_cost(self):
        gt = codes('H' * 500)
        pred = codes('H' * 237 + 'D' * 25 + 'H' * 238)
        self.assertAlmostEqual(pm.evaluate(gt, pred)['spell_f1_mean'], 4 / 15)

    def test_wrong_team_cannot_match(self):
        self.assertEqual(pm.evaluate(codes('HHHH'), codes('AAAA'))['spell_f1_mean'], 0)

    def test_submission_roundtrip_and_rejections(self):
        import pandas as pd
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'synthetic.csv'
            pd.DataFrame({'frame': range(4), 'label': list('HADX')}).to_csv(path, index=False)
            table = pm.read_label_csv(path)
            self.assertIsNone(pm.validate_submission(table, 4))
            self.assertIsNotNone(pm.validate_submission(table, 5))
            table.loc[1, 'frame'] = 0
            self.assertIsNotNone(pm.validate_submission(table, 4))
            table.loc[1, 'frame'] = 1
            table.loc[1, 'label'] = None
            self.assertIsNotNone(pm.validate_submission(table, 4))

    def test_official_validator_cli(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'synthetic.csv'
            path.write_text('frame,label\n0,H\n1,A\n2,D\n3,X\n')
            cmd = [sys.executable, str(ROOT / 'vendor/mtah/evaluation/validate_submission.py'),
                   str(path), '--rows', '4']
            self.assertEqual(subprocess.run(cmd, capture_output=True).returncode, 0)
            path.write_text('frame,label\n0,H\n0,A\n2,D\n3,X\n')
            self.assertNotEqual(subprocess.run(cmd, capture_output=True).returncode, 0)


class InputTests(unittest.TestCase):
    def parse(self, text, expected=1):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'synthetic.txt'
            path.write_text(text)
            return list(iter_tracking(path, 2, expected))

    def test_alignment_missingness_and_outside_pitch(self):
        frame = self.parse('1:0.1,0.2,1;NaN,NaN,NaN:1.01,0.3,2\n')[0]
        self.assertEqual(frame.video_frame, 0)
        self.assertTrue(math.isnan(frame.players[1][0]))
        self.assertGreater(position_metres(frame.ball)[0], 105.3)
        self.assertAlmostEqual(position_metres(frame.players[0])[1], 13.6)

    def test_missing_ball_remains_nan(self):
        frame = self.parse('1:0,0,0;1,1,0:NaN,NaN,NaN\n')[0]
        self.assertTrue(math.isnan(frame.ball[0]))

    def test_invalid_index(self):
        for value in (0, -1, True, 1.1):
            with self.assertRaises(InputError):
                video_frame_of(value)

    def test_duplicate_and_truncated_frames(self):
        with self.assertRaisesRegex(InputError, '^NONCONTIGUOUS_FRAME_IDS$'):
            self.parse('2:0,0,0;1,1,0:0,0,0\n')
        with self.assertRaisesRegex(InputError, '^FRAME_COUNT_MISMATCH$'):
            self.parse('1:0,0,0;1,1,0:0,0,0\n', expected=2)

    def test_sensitive_parser_content_not_in_exception(self):
        marker = 'synthetic-sensitive-marker'
        with self.assertRaisesRegex(InputError, '^INVALID_OBSERVATION$') as context:
            self.parse('1:0,0,0;1,1,0:' + marker + '\n')
        self.assertNotIn(marker, str(context.exception))

    def test_metadata_membership(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'synthetic.xml'
            path.write_text('<Root><Team id="home"/><Team id="away"/>'
                            '<Player teamId="away"/><Player teamId="home"/></Root>')
            self.assertEqual(team_membership(path), ('A', 'H'))
            path.write_text('<Root><Team id="home"/><Team id="away"/><Player teamId="other"/></Root>')
            with self.assertRaisesRegex(InputError, '^UNKNOWN_TEAM$'):
                team_membership(path)


class ReproducibilityTests(unittest.TestCase):
    def test_upstream_blobs_unchanged(self):
        base = ROOT / 'vendor/mtah'
        manifest = json.loads((base / 'UPSTREAM.json').read_text())
        for entry in manifest['files']:
            content = (base / entry['path']).read_bytes()
            blob = b'blob ' + str(len(content)).encode() + b'\0' + content
            self.assertEqual(hashlib.sha1(blob).hexdigest(), entry['blob'], entry['path'])

    def test_artifact_gate(self):
        self.assertTrue(allowed('src/mtah/inputs.py', '100644', 100))
        for path, mode, size in [('data/raw.txt', '100644', 100),
                                 ('docs/link.md', '120000', 10),
                                 ('outputs/report.json', '100644', 10),
                                 ('src/model.pkl', '100644', 100),
                                 ('docs/huge.json', '100644', 900000)]:
            self.assertFalse(allowed(path, mode, size))


if __name__ == '__main__':
    unittest.main()
