import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'scripts'))
from mtah.checkpoints import CHECKPOINTS,installed,unpack_verified
from package_submission import team_filename


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
