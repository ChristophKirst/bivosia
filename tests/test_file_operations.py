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
from bivosia.dataset.paths import folder_path
from bivosia.processing.file_operations import delete_selected_file, move_selected_file
from bivosia.processing.folder_operations import (completed_job_entries, delete_folder_entries, folder_entries,
                                                 plan_folder_transfer, transfer_folder)
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

    def test_consolidation_resolves_remote_conflict_without_changing_source(self):
        from bivosia.processing.consolidation import plan_consolidation, consolidate
        job = Job.create(self.dataset, 'quantify', {})
        job.update(state='complete')
        self.remote.mkdir()
        target = self.remote / self.path.relative_to(self.dataset.directory)
        target.parent.mkdir()
        target.write_bytes(b'conflicting destination')
        location = {'host': 'server', 'path': str(self.remote)}
        with patch('bivosia.processing.file_operations.Remote', LocalTransport), patch(
                'bivosia.processing.remote_inventory.Remote', LocalTransport):
            plan = plan_consolidation(self.dataset, location, self.profiles)
            self.assertFalse(any(Path(name).is_relative_to(folder_path('jobs')) for name in plan.files))
            name = self.path.relative_to(self.dataset.directory).as_posix()
            self.assertEqual(plan.conflicts, (name,))
            self.assertEqual(target.read_bytes(), b'conflicting destination')
            choice = next(i for i, entry in enumerate(plan.files[name]) if not entry.host)
            consolidate(self.dataset, plan, {name: choice}, self.profiles)
        self.assertEqual(target.read_bytes(), self.content)
        self.assertEqual(self.path.read_bytes(), self.content)
        self.assertEqual(Dataset(self.remote).roots, (self.remote,))

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

    def test_folder_delete_checks_all_files_first_and_preserves_link_targets(self):
        child = self.path.parent / 'nested'
        child.mkdir()
        other = child / 'notes.txt'
        other.write_text('original')
        linked = self.path.parent / 'linked'
        outside = self.root / 'outside'
        outside.mkdir()
        (outside / 'keep.txt').write_text('keep')
        linked.symlink_to(outside, target_is_directory=True)
        entries = folder_entries(scan_dataset(self.dataset), 'experiment')
        other.write_text('changed since confirmation')
        with self.assertRaisesRegex(ValueError, 'File changed'):
            delete_folder_entries(self.dataset, entries, [])
        self.assertEqual(self.path.read_bytes(), self.content)
        entries = folder_entries(scan_dataset(self.dataset), 'experiment')
        delete_folder_entries(self.dataset, entries, [])
        self.assertFalse(self.path.parent.exists())
        self.assertEqual((outside / 'keep.txt').read_text(), 'keep')

    @patch('bivosia.processing.file_operations.Remote', LocalTransport)
    def test_job_folders_stay_in_place_and_deletion_is_batched(self):
        job = Job.create(self.dataset, 'quantify', {})
        entries = folder_entries(scan_dataset(self.dataset), folder_path('jobs'))
        with self.assertRaisesRegex(ValueError, 'active dataset job'):
            delete_folder_entries(self.dataset, entries, [])
        job.update(state='complete')
        (job.directory / 'empty').mkdir()
        entries = folder_entries(scan_dataset(self.dataset), folder_path('jobs'))
        location = {'host': 'server', 'path': str(self.remote)}
        with self.assertRaisesRegex(ValueError, 'execution location'):
            plan_folder_transfer(self.dataset, entries, location, self.profiles)
        job_file = next(entry for entry in entries if not entry.is_directory)
        with self.assertRaisesRegex(ValueError, 'execution location'):
            move_selected_file(self.dataset, job_file, location, self.profiles)
        delete_folder_entries(self.dataset, entries, [])
        self.assertFalse(job.directory.exists())
        remote_job = Job.create(Dataset(self.remote), 'quantify', {})
        remote_job.update(state='complete')
        remote_entries = tuple(replace(entry, host='server') for entry in
                               folder_entries(scan_dataset(Dataset(self.remote)), folder_path('jobs')))
        delete_folder_entries(self.dataset, remote_entries, self.profiles)
        self.assertFalse(remote_job.directory.exists())
        self.assertTrue(self.path.exists())

    @patch('bivosia.processing.file_operations.Remote', LocalTransport)
    def test_folder_move_preserves_empty_subfolders_over_ssh(self):
        (self.path.parent / 'empty').mkdir()
        entries = folder_entries(scan_dataset(self.dataset), 'experiment')
        location = {'host': 'server', 'path': str(self.remote)}
        plan = plan_folder_transfer(self.dataset, entries, location, self.profiles)
        transfer_folder(self.dataset, entries, plan, {}, self.profiles, move=True)
        self.assertFalse(self.path.parent.exists())
        self.assertEqual((self.remote / 'experiment' / 'notes.bin').read_bytes(), self.content)
        self.assertTrue((self.remote / 'experiment' / 'empty').is_dir())

    @patch('bivosia.processing.file_operations.Remote', LocalTransport)
    def test_cleanup_selects_only_completed_jobs_across_locations(self):
        done = Job.create(self.dataset, 'quantify', {})
        done.update(state='complete')
        failed = Job.create(self.dataset, 'quantify', {})
        failed.update(state='failed')
        remote_done = Job.create(Dataset(self.remote), 'quantify', {})
        remote_done.update(state='complete')
        entries = (*folder_entries(scan_dataset(self.dataset), folder_path('jobs')),
                   *(replace(entry, host='server') for entry in
                     folder_entries(scan_dataset(Dataset(self.remote)), folder_path('jobs'))))
        selected = completed_job_entries(self.dataset, entries, self.profiles)
        self.assertTrue(any(entry.path == done.directory for entry in selected))
        self.assertTrue(any(entry.path == remote_done.directory for entry in selected))
        self.assertFalse(any(entry.path.is_relative_to(failed.directory) for entry in selected))
        self.assertTrue(done.directory.exists())
        delete_folder_entries(self.dataset, selected, self.profiles)
        self.assertFalse(done.directory.exists())
        self.assertFalse(remote_done.directory.exists())
        self.assertTrue(failed.directory.exists())
        self.assertEqual(self.path.read_bytes(), self.content)

    def test_folder_move_resolves_conflicts_and_keeps_unchosen_versions(self):
        target = self.root / 'mounted'
        target.mkdir()
        third = self.root / 'third'
        third.mkdir()
        save_locations(self.dataset, [str(target), str(third)], [])
        relative = self.path.relative_to(self.dataset.directory)
        for root, data in ((target, b'destination'), (third, b'other version')):
            (root / relative).parent.mkdir()
            (root / relative).write_bytes(data)
        entries = folder_entries(scan_dataset(self.dataset), 'experiment')
        plan = plan_folder_transfer(self.dataset, entries, {'host': '', 'path': str(target)}, [])
        choice = next(i for i, entry in enumerate(plan.files[str(relative)]) if entry.path == self.path)
        transfer_folder(self.dataset, entries, plan, {str(relative): choice}, [], move=True)
        self.assertFalse(self.path.exists())
        self.assertEqual((target / relative).read_bytes(), self.content)
        self.assertEqual((third / relative).read_bytes(), b'other version')

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
