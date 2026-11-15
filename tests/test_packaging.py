import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'scripts'))
from mtah.checkpoints import CHECKPOINTS,installed,unpack_verified
from package_submission import team_filename
import package_submission


class CheckpointTests(unittest.TestCase):
    def fixture(self,extra=None,missing=None,corrupt=None,link=None):
        values={n:('synthetic '+n).encode() for n in CHECKPOINTS|{'MODEL_CARD.md','MODEL_LICENSE.md'}}
        manifest=dict(members={n:hashlib.sha256(v).hexdigest() for n,v in values.items()},
                      checkpoints={n:hashlib.sha256(values[n]).hexdigest() for n in CHECKPOINTS})
        content=io.BytesIO()
        with tarfile.open(fileobj=content,mode='w:gz') as archive:
            for name,value in values.items():
                if name==missing:continue
                if name==corrupt:value=b'changed'
                info=tarfile.TarInfo(name);info.size=len(value)
                archive.addfile(info,io.BytesIO(value))
            if extra:
                info=tarfile.TarInfo(extra);info.size=3;archive.addfile(info,io.BytesIO(b'bad'))
            if link:
                info=tarfile.TarInfo(link);info.type=tarfile.SYMTYPE;info.linkname='/tmp/outside'
                archive.addfile(info)
        content=content.getvalue();manifest.update(archive_sha256=hashlib.sha256(content).hexdigest(),archive_bytes=len(content))
        return content,manifest,values

    def test_complete_install_and_repeat_are_identical(self):
        content,manifest,values=self.fixture()
        with tempfile.TemporaryDirectory() as temp:
            self.assertFalse(installed(manifest,temp))
            unpack_verified(content,manifest,temp)
            self.assertTrue(installed(manifest,temp))
            unpack_verified(content,manifest,temp)
            for name in CHECKPOINTS:self.assertEqual((Path(temp)/name).read_bytes(),values[name])
            self.assertFalse((Path(temp)/'MODEL_CARD.md').exists())

    def test_bad_archives_do_not_install_any_checkpoint(self):
        cases=[dict(extra='../outside'),dict(extra='data/private.csv'),dict(missing='MODEL_CARD.md'),
               dict(corrupt='outputs/phase3/model.pkl'),dict(link='outputs/phase4/model.pkl'),
               dict(extra='outputs/phase3/model.pkl')]
        for case in cases:
            content,manifest,_=self.fixture(**case)
            with tempfile.TemporaryDirectory() as temp:
                with self.assertRaises(ValueError):unpack_verified(content,manifest,temp)
                self.assertFalse((Path(temp)/'outputs').exists())

    def test_outer_hash_and_destination_conflicts_fail(self):
        content,manifest,_=self.fixture()
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError,'ARCHIVE_INTEGRITY'):
                unpack_verified(content+b'bad',manifest,temp)
            path=Path(temp)/'outputs/phase3/model.pkl';path.parent.mkdir(parents=True);path.write_bytes(b'unrelated')
            with self.assertRaisesRegex(ValueError,'DESTINATION_CONFLICT'):
                unpack_verified(content,manifest,temp)
            self.assertEqual(path.read_bytes(),b'unrelated')
            self.assertFalse((Path(temp)/'outputs/phase4/model.pkl').exists())

    def test_unexpected_manifest_is_rejected_before_file_access(self):
        _,manifest,_=self.fixture();manifest['checkpoints']['../outside']='unused'
        with self.assertRaisesRegex(ValueError,'INVALID_CHECKPOINT_MANIFEST'):installed(manifest,'/nonexistent')


class TeamNameTests(unittest.TestCase):
    def test_registered_name_is_preserved(self):
        for name in ['Kaizen FC','Équipe 7','A.B-C_2026']:
            self.assertEqual(team_filename(name),name)
        for name in ['', ' team','team ','.','..','a/b','a\\b','a\nb','x'*101]:
            with self.assertRaises(ValueError):team_filename(name)


class PackageRerunTests(unittest.TestCase):
    def fixture(self, root):
        candidate = root/'outputs/phase6/submission.csv'
        candidate.parent.mkdir(parents=True)
        candidate.write_bytes(b'frame,label\n0,X\n1,H\n2,D\n3,A\n')
        candidate.with_suffix('.report.json').write_text(json.dumps(
            dict(csv_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest())))
        report = root/'docs/TECHNICAL_REPORT.md'
        report.parent.mkdir()
        report.write_text('Synthetic technical document.\n')
        return candidate, report

    def package(self, root, team='Kaizen'):
        with patch.object(package_submission, 'ROOT', root), \
                patch.object(package_submission, 'verify_freeze'), \
                patch.object(package_submission, 'organizer_contract', return_value={'n_frames':4}), \
                patch.object(package_submission.subprocess, 'run', return_value=SimpleNamespace(returncode=0)), \
                patch.object(sys, 'argv', ['package_submission.py', '--team-name', team]):
            package_submission.main()

    def test_repackage_preserves_handoff_and_csv_but_refreshes_document(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            candidate,report=self.fixture(root)
            self.package(root)
            out=root/'outputs/phase7/upload'
            handoff=out/'UPLOAD_INSTRUCTIONS.md'
            private=b'Private form: https://example.invalid/form\r\nCutoff: confirmed locally\r\n'
            handoff.write_bytes(private)
            report.write_text('Revised synthetic technical document.\n')
            self.package(root)
            self.assertEqual(handoff.read_bytes(),private)
            self.assertEqual((out/'Kaizen.csv').read_bytes(),candidate.read_bytes())
            self.assertEqual((out/'Kaizen.md').read_text(),'Team: Kaizen\n\n'+report.read_text())
            manifest=json.loads((out/'manifest.json').read_text())
            for name in ['Kaizen.csv','Kaizen.md']:
                self.assertEqual(manifest['files'][name]['sha256'],
                                 hashlib.sha256((out/name).read_bytes()).hexdigest())

    def test_team_change_and_submitted_status_fail_before_package_writes(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            self.fixture(root)
            self.package(root)
            out=root/'outputs/phase7/upload'
            before={p.name:p.read_bytes() for p in out.iterdir()}
            with self.assertRaisesRegex(ValueError,'TEAM_MISMATCH'):
                self.package(root,'Different Team')
            self.assertEqual(before,{p.name:p.read_bytes() for p in out.iterdir()})
            manifest=json.loads((out/'manifest.json').read_text())
            manifest['submitted']=True
            (out/'manifest.json').write_text(json.dumps(manifest))
            before={p.name:p.read_bytes() for p in out.iterdir()}
            with self.assertRaisesRegex(ValueError,'SUBMISSION_STATUS'):
                self.package(root)
            self.assertEqual(before,{p.name:p.read_bytes() for p in out.iterdir()})
