"""Small cache provenance checks without model files or experiment resources."""

import tempfile
import unittest

import numpy as np

from ..multiview_experiment import save_surface
from .io import target_surface
from .test_geometry import _observation, _surface


class SurfaceCacheTests(unittest.TestCase):
    def setUp(self):
        depth = np.ones((13, 13))
        observation = _observation(depth)
        self.loaded = {'views': [dict(id=name, depth_m=depth,
            K=observation['K'], T_world_camera_cv=observation['T_world_camera_cv'],
            observation=observation) for name in ('A', 'B')]}

    def test_complete_binding_preserves_actual_representative(self):
        original = _surface()
        with tempfile.TemporaryDirectory() as directory:
            save_surface(directory, original)
            loaded = target_surface(self.loaded, directory)
        np.testing.assert_array_equal(loaded['xyz'], original['xyz'])
        np.testing.assert_array_equal(loaded['observation_to_element'],
                                      original['observation_to_element'])

    def test_raw_matching_observations_do_not_hide_corrupt_representative(self):
        corrupt = _surface()
        corrupt['xyz'] = corrupt['xyz'].copy()
        corrupt['xyz'][0, 0] += .001
        with tempfile.TemporaryDirectory() as directory:
            save_surface(directory, corrupt)
            with self.assertRaises(AssertionError):
                target_surface(self.loaded, directory)

    def test_wrong_observation_to_element_binding_is_rejected(self):
        corrupt = _surface()
        corrupt['observation_to_element'][corrupt['representative_observation'][0]] = 1
        with tempfile.TemporaryDirectory() as directory:
            save_surface(directory, corrupt)
            with self.assertRaises(AssertionError):
                target_surface(self.loaded, directory)

    def test_missing_same_view_entity_labels_cannot_use_other_cache_labels(self):
        corrupt = _surface()
        corrupt['observation_entity_member'][0] = False
        with tempfile.TemporaryDirectory() as directory:
            save_surface(directory, corrupt)
            with self.assertRaises(AssertionError):
                target_surface(self.loaded, directory)


if __name__ == '__main__':
    unittest.main()
