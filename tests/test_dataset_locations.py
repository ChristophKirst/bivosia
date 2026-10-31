"""Remote dataset location selection and path validation."""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from bivosia.dataset.layout import Dataset
from bivosia.dataset.storage.locations import location_for, remote_path, save_locations


class DatasetLocationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.dataset = Dataset(Path(self.temporary.name) / 'local')
        self.dataset.save()

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
