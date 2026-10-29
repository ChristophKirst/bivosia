"""Protect selected files and existing destinations during file-menu operations."""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import hashlib
import sys
import unittest

from bivosia.config import PACKAGE_DIRECTORY
from bivosia.dataset.file_operations import FileDestination, delete_file, file_state
from bivosia.dataset.inventory import scan_dataset
from bivosia.dataset.jobs import Job
from bivosia.dataset.layout import Dataset
from bivosia.dataset.locations import save_locations
from bivosia.processing.file_operations import delete_selected_file, move_selected_file
from bivosia.processing.remote import BOOTSTRAP, Remote


class LocalTransport(Remote):
    """Run the SSH file protocol in a separate local Python process."""

    def command(self):
        return [sys.executable, '-c', BOOTSTRAP]


class FileOperationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.dataset = Dataset(self.root / 'local')
        self.dataset.directory.mkdir()
        self.path = self.dataset.directory / 'experiment' / 'notes.bin'
        self.path.parent.mkdir()
        self.content = bytes(range(256)) * 8193
        self.path.write_bytes(self.content)
        self.remote = self.root / 'remote'
        self.profiles = [{'host': 'server', 'python': sys.executable, 'code': str(PACKAGE_DIRECTORY)}]
        save_locations(self.dataset, [], [{'host': 'server', 'path': str(self.remote)}])

    def entry(self, dataset=None, host=''):
        dataset = dataset or self.dataset
        return replace(next(entry for entry in scan_dataset(dataset)
                            if entry.relative == Path('experiment/notes.bin')), host=host)

    def test_local_move_registers_folder_and_never_replaces_existing_file(self):
        destination = self.root / 'mounted'
        destination.mkdir()
        entry = self.entry()
        move_selected_file(self.dataset, entry, {'host': '', 'path': str(destination)}, [])
        target = destination / entry.relative
        self.assertEqual(target.read_bytes(), self.content)
        self.assertFalse(self.path.exists())
        self.assertEqual(Dataset(self.dataset.directory).path(entry.relative), target)
        self.path.write_bytes(b'another copy')
        with self.assertRaises(FileExistsError):
            move_selected_file(self.dataset, self.entry(), {'host': '', 'path': str(destination)}, [])
        self.assertEqual(self.path.read_bytes(), b'another copy')
        self.assertEqual(target.read_bytes(), self.content)

    def test_delete_rejects_stale_entries_and_only_removes_selected_copy(self):
        entry = self.entry()
        self.path.write_bytes(b'new annotations')
        with self.assertRaisesRegex(ValueError, 'File changed'):
            delete_selected_file(self.dataset, entry, [])
        elsewhere = self.root / 'another-copy.bin'
        elsewhere.write_bytes(b'preserve')
        delete_selected_file(self.dataset, self.entry(), [])
        self.assertFalse(self.path.exists())
        self.assertEqual(elsewhere.read_bytes(), b'preserve')
        with self.assertRaisesRegex(ValueError, 'outside'):
            delete_file(self.dataset, elsewhere, file_state(elsewhere))

    def test_failed_or_conflicting_publication_preserves_files(self):
        target = self.root / 'copy.bin'
        state = file_state(self.path)
        with FileDestination(target, state) as destination:
            destination.stream.write(self.content)
            with self.assertRaises(IOError):
                destination.publish('0' * 64)
        self.assertFalse(target.exists())
        with FileDestination(target, state) as destination:
            destination.stream.write(self.content)
            target.write_bytes(b'concurrent writer')
            with self.assertRaises(FileExistsError):
                destination.publish(hashlib.sha256(self.content).hexdigest())
        self.assertEqual(target.read_bytes(), b'concurrent writer')
        self.assertEqual(self.path.read_bytes(), self.content)
        self.assertFalse(list(self.root.glob('.bivosia-move-*')))

    def test_symlink_deletion_preserves_its_target_and_active_jobs_block_changes(self):
        link = self.dataset.directory / 'reference.bin'
        link.symlink_to(self.path)
        delete_file(self.dataset, link, file_state(link))
        self.assertFalse(link.exists())
        self.assertEqual(self.path.read_bytes(), self.content)
        Job.create(self.dataset, 'quantify', {})
        with self.assertRaisesRegex(ValueError, 'active dataset job'):
            delete_selected_file(self.dataset, self.entry(), [])
        self.assertEqual(self.path.read_bytes(), self.content)

    def test_move_updates_an_external_recording_reference(self):
        video = self.root / 'original.mp4'
        video.write_bytes(b'recording')
        self.dataset.settings['video'] = str(video)
        self.dataset.save()
        entry = next(entry for entry in scan_dataset(self.dataset) if entry.path == video)
        destination = self.root / 'recordings'
        destination.mkdir()
        move_selected_file(self.dataset, entry, {'host': '', 'path': str(destination)}, [])
        current = Dataset(self.dataset.directory)
        self.assertEqual(current.video, destination / 'recording' / video.name)
        self.assertEqual(current.video.read_bytes(), b'recording')
        self.assertFalse(video.exists())

    @patch('bivosia.processing.file_operations.Remote', LocalTransport)
    def test_ssh_upload_download_and_delete_use_selected_paths(self):
        entry = self.entry()
        move_selected_file(self.dataset, entry, {'host': 'server', 'path': str(self.remote)}, self.profiles)
        target = self.remote / entry.relative
        self.assertEqual(target.read_bytes(), self.content)
        self.assertFalse(self.path.exists())
        remote_entry = self.entry(Dataset(self.remote), 'server')
        move_selected_file(self.dataset, remote_entry,
                           {'host': '', 'path': str(self.dataset.directory)}, self.profiles)
        self.assertEqual(self.path.read_bytes(), self.content)
        self.assertFalse(target.exists())
        target.write_bytes(b'remote-only')
        delete_selected_file(self.dataset, self.entry(Dataset(self.remote), 'server'), self.profiles)
        self.assertFalse(target.exists())
        self.assertEqual(self.path.read_bytes(), self.content)

    @patch('bivosia.processing.file_operations.Remote', LocalTransport)
    def test_remote_conflict_keeps_both_original_files(self):
        entry = self.entry()
        target = self.remote / entry.relative
        target.parent.mkdir(parents=True)
        target.write_bytes(b'remote annotations')
        with self.assertRaisesRegex(RuntimeError, 'Destination already exists'):
            move_selected_file(self.dataset, entry, {'host': 'server', 'path': str(self.remote)}, self.profiles)
        self.assertEqual(self.path.read_bytes(), self.content)
        self.assertEqual(target.read_bytes(), b'remote annotations')

    @patch('bivosia.processing.file_operations.Remote', LocalTransport)
    def test_remote_to_remote_and_same_host_storage_moves(self):
        target = self.remote / 'experiment' / 'notes.bin'
        target.parent.mkdir(parents=True)
        target.write_bytes(self.content)
        second = self.root / 'second'
        profiles = [*self.profiles, {**self.profiles[0], 'host': 'second'}]
        save_locations(self.dataset, [], [{'host': 'server', 'path': str(self.remote)},
                                          {'host': 'second', 'path': str(second)}])
        entry = self.entry(Dataset(self.remote), 'server')
        move_selected_file(self.dataset, entry, {'host': 'second', 'path': str(second)}, profiles)
        self.assertFalse(target.exists())
        destination = second / entry.relative
        self.assertEqual(destination.read_bytes(), self.content)
        mounted = self.root / 'remote-mounted'
        mounted.mkdir()
        save_locations(Dataset(second), [str(mounted)], [])
        remote = LocalTransport('second', sys.executable, str(PACKAGE_DIRECTORY), str(second))
        remote.call('move_file', path=str(destination), expected=file_state(destination),
                    root=str(mounted), relative=str(entry.relative))
        self.assertFalse(destination.exists())
        self.assertEqual((mounted / entry.relative).read_bytes(), self.content)


if __name__ == '__main__':
    unittest.main()
