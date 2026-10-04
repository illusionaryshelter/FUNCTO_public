"""Scene integration uses one pre-cost choice for each whole physical body."""
import unittest

import numpy as np
import torch

from manipulation.planning_scene import PlanningScene, ObservedSpace
from manipulation.voxel_layers import LayeredVoxelQuery
from manipulation.collision_records import decode_array
from test_voxel_layers import GROUPS, NativeStub, region


class TestLayeredPlanningScene(unittest.TestCase):
    def make_scene(self):
        return PlanningScene(background=LayeredVoxelQuery(NativeStub(10., 2.), NativeStub(0., 3.), region()),
            owners=(), coverage=ObservedSpace(np.ones((41, 41, 41), bool), origin=[-2.]*3,
                voxel_size=.1, observation_ids=["snapshot"], reference_frame="world"),
            reference_frame="world", robot_sphere_materials=(None,)*3,
            robot_geometry_source="complete native test covers", robot_collision_bodies=GROUPS, has_gripper=False)

    def test_output_is_the_actual_shared_choice_and_binding_preserves_both_fields(self):
        scene = self.make_scene()
        spheres = torch.tensor([[[[0., 0., 0., .01], [.08, 0., 0., .01],
                                  [0., .02, 0., .01]]]], requires_grad=True)
        capture = {}
        result = scene.evaluate(spheres, torch.zeros(1, 1, 3), torch.tensor([[[1., 0., 0., 0.]]]),
                                torch.zeros(1, 1), layer_selection_out=capture)
        self.assertAlmostEqual(result.item(), 20.16, places=5)
        self.assertEqual(decode_array(capture["robot/background"]["body_fine"]).tolist(), [[[False, True]]])
        result.sum().backward()
        self.assertEqual(spheres.grad[..., 0].tolist(), [[[2., 2., 3.]]])
        bound = scene.to_execution_binding()
        self.assertEqual(bound["robot_collision_bodies"], GROUPS)
        self.assertEqual(set(bound["background"]["geometry"]["layers"]), {"coarse", "fine"})
        self.assertEqual(bound["background"]["geometry"]["fine_region"]["measurement_allowance_m"], .001)

    def test_clones_share_only_immutable_layer_geometry(self):
        scene = self.make_scene()
        cloned = scene.clone_for_rollout()
        self.assertIsNot(scene.background, cloned.background)
        self.assertIsNot(scene.background.coarse, cloned.background.coarse)
        self.assertIs(scene.background.region, cloned.background.region)
        estimate = scene.estimated_workspace_allocation(1, 2, robot_coverage_horizon=0)
        self.assertGreaterEqual(estimate["shared_cpu"], scene.background.region.storage_bytes)
        self.assertGreaterEqual(estimate["retained"], 2*3*2*20)

    def test_missing_whole_body_binding_is_rejected(self):
        scene = self.make_scene()
        with self.assertRaises(ValueError):
            PlanningScene(background=scene.background, owners=(), coverage=scene.coverage,
                reference_frame="world", robot_sphere_materials=(None,)*3,
                robot_geometry_source="missing full body identities", has_gripper=False)


if __name__ == "__main__":
    unittest.main()
