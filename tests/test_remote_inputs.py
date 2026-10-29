"""Protect annotations and recordings during remote input synchronization."""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory
import hashlib
import io
import sys
import unittest

from bivosia.config import PACKAGE_DIRECTORY
from bivosia.dataset.layout import Dataset
from bivosia.dataset.locations import location_for, remote_path, save_locations
from bivosia.dataset.transfer import InputUpload, vos_inputs
from bivosia.processing.remote import BOOTSTRAP, Remote


class LocalTransport(Remote):
    """Exercise the actual streaming protocol through a local Python subprocess."""

    def command(self):
        return [sys.executable, '-c', BOOTSTRAP]


class RemoteInputTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.dataset = Dataset(self.root / 'local')
        video = self.root / 'source.mp4'
        video.write_bytes(b'video contents')
        self.dataset.recording.mkdir(parents=True)
        (self.dataset.recording / video.name).symlink_to(video)
        self.dataset.settings = {'video': video.name}
        self.dataset.save()
        for key, data in (('key_frames', b'{"frames": [0]}'), ('key_frame_masks', b'mask contents')):
            path = self.dataset.file(key, write=True)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        self.destination = self.root / 'remote'
        self.remote = LocalTransport('workstation', sys.executable,
                                     str(PACKAGE_DIRECTORY), str(self.destination))

    def test_protocol_transfers_symlink_contents_and_reuses_identical_files(self):
        save_locations(self.dataset, [], [{'host': 'workstation', 'path': str(self.destination)}])
        plan = self.remote.sync(self.dataset, dry_run=True)
        self.assertEqual(len(plan['missing']), 4)
        self.assertFalse(self.destination.exists())
        result = self.remote.sync(self.dataset)
        self.assertEqual(len(result['uploaded']), 4)
        remote = Dataset(self.destination)
        self.assertEqual(remote.video.read_bytes(), b'video contents')
        self.assertFalse(remote.video.is_symlink())
        self.assertNotIn('remote_locations', remote.settings)
        times = {path: path.stat().st_mtime_ns for path in self.destination.rglob('*') if path.is_file()}
        result = self.remote.sync(self.dataset)
        self.assertEqual(result['uploaded'], [])
        self.assertEqual(times, {path: path.stat().st_mtime_ns for path in times})

    def test_conflict_preserves_remote_annotations_and_does_not_upload_video(self):
        path = Dataset(self.destination).file('key_frames', write=True)
        path.parent.mkdir(parents=True)
        path.write_bytes(b'remote annotations')
        with self.assertRaisesRegex(RuntimeError, 'Remote files differ'):
            self.remote.sync(self.dataset)
        self.assertEqual(path.read_bytes(), b'remote annotations')
        self.assertFalse((self.destination / 'recording').exists())

    def upload(self):
        with vos_inputs(self.dataset) as sources:
            contents = {name: path.read_bytes() for name, path in sources.items()}
        manifest = {name: {'size': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
                    for name, data in contents.items()}
        upload = InputUpload(Dataset(self.destination), manifest)
        upload.plan()
        return upload, b''.join(contents[name] for name in upload.missing)

    def test_truncated_and_corrupt_uploads_publish_nothing(self):
        for corrupt in (False, True):
            upload, content = self.upload()
            with closing(upload):
                content = b'x' * len(content) if corrupt else content[:-1]
                with self.assertRaises((OSError, IOError)):
                    upload.receive(io.BytesIO(content))
            self.assertFalse(any(path.is_file() for path in self.destination.rglob('*')))

    def test_publication_race_keeps_existing_file_and_rolls_back_new_files(self):
        upload, content = self.upload()
        with closing(upload):
            upload.receive(io.BytesIO(content))
            target = upload.targets[upload.missing[-1]]
            target.write_bytes(b'concurrent writer')
            with self.assertRaises(FileExistsError):
                upload.publish()
        self.assertEqual(target.read_bytes(), b'concurrent writer')
        self.assertEqual([path for path in self.destination.rglob('*') if path.is_file()], [target])

    def test_cancellation_leaves_destination_untouched(self):
        with self.assertRaises(InterruptedError):
            self.remote.sync(self.dataset, cancelled=lambda: True)
        self.assertFalse(self.destination.exists())

    def test_locations_preserve_other_settings_and_override_default(self):
        other = Dataset(self.dataset.directory)
        other.settings['animals'] = 6
        other.save()
        profile = {'host': 'workstation', 'data_folder': '/data'}
        self.assertEqual(location_for(self.dataset, profile), '/data/local')
        save_locations(self.dataset, [], [{'host': 'workstation', 'path': '/experiments/trial'}])
        self.assertEqual(location_for(other, profile), '/experiments/trial')
        self.assertEqual(Dataset(self.dataset.directory).settings['animals'], 6)
        other.settings['video'] = 'another.mp4'
        other.save()
        self.assertEqual(location_for(other, profile), '/experiments/trial')
        for value in ('sftp://workstation/data/trial', 'data/trial', '/data/../trial'):
            with self.assertRaises(ValueError):
                remote_path(value)


if __name__ == '__main__':
    unittest.main()
