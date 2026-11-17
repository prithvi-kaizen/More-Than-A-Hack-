import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
sys.path.insert(0,str(ROOT/'src'))
import finalize_delivery


class FinalDeliveryTests(unittest.TestCase):
    def fixture(self,root):
        source=root/'outputs/phase7/upload';source.mkdir(parents=True)
        candidate=root/'outputs/phase6/submission.csv';candidate.parent.mkdir(parents=True)
        candidate.write_bytes(b'frame,label\n0,X\n1,H\n2,D\n3,A\n')
        (source/'Kaizen.csv').write_bytes(candidate.read_bytes())
        report=root/'docs/TECHNICAL_REPORT.md';report.parent.mkdir()
        report.write_text('Synthetic technical report.\n')
        (source/'Kaizen.md').write_text('Team: Kaizen\n\n'+report.read_text())
        manifest=dict(team_name='Kaizen',submitted=False,files={
            name:dict(sha256=hashlib.sha256((source/name).read_bytes()).hexdigest(),
                      bytes=(source/name).stat().st_size) for name in ['Kaizen.csv','Kaizen.md']})
        (source/'manifest.json').write_text(json.dumps(manifest))
        return source

    def seal(self,root):
        with patch.object(finalize_delivery,'ROOT',root), \
                patch.object(finalize_delivery,'verify_freeze'), \
                patch.object(finalize_delivery,'organizer_contract',return_value={'n_frames':4}), \
                patch.object(finalize_delivery.subprocess,'run'):
            finalize_delivery.main()

    def test_sealed_pair_is_exact_and_repeatable(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=self.fixture(root)
            self.seal(root);self.seal(root)
            final=root/'outputs/phase10/upload'
            self.assertEqual({p.name for p in final.iterdir()},{'Kaizen.csv','Kaizen.md'})
            for name in ['Kaizen.csv','Kaizen.md']:
                self.assertEqual((source/name).read_bytes(),(final/name).read_bytes())
            manifest=json.loads((root/'outputs/phase10/final_manifest.json').read_text())
            self.assertFalse(manifest['form_submitted'])

    def test_changed_source_and_conflicting_destination_are_rejected(self):
        for mode in ['source','destination']:
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as temp:
                root=Path(temp);source=self.fixture(root)
                if mode=='source':(source/'Kaizen.md').write_text('Stale document')
                else:
                    destination=root/'outputs/phase10/upload';destination.mkdir(parents=True)
                    (destination/'Kaizen.md').write_bytes(b'unrelated existing document')
                with self.assertRaises(ValueError):self.seal(root)
                self.assertFalse((root/'outputs/phase10/upload/Kaizen.csv').exists())

    def test_submitted_package_cannot_be_resealed(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=self.fixture(root)
            path=source/'manifest.json';manifest=json.loads(path.read_text());manifest['submitted']=True
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,'STATUS_REQUIRES_REVIEW'):self.seal(root)
            self.assertFalse((root/'outputs/phase10').exists())
