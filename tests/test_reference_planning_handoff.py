"""CPU checks through the real PlanningProblem reference consumer.

Analytic observed tracks verify the representation-to-planner boundary. They
do not stand in for robot IK, scene collision evidence or executed motion.
"""

# BEGIN fixed-frame reference handoff and continuous attachment regressions.
from copy import deepcopy
from types import SimpleNamespace
import unittest

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from manipulation.motion import MotionModel, _inverse
from manipulation.planning_problem import (PlanningProblem, PoseEventRequirement,
    PosePath, PoseProcessRequirement, reference_phase)
from manipulation.planning_scene import CollisionOwner
from manipulation.representation import build_representation, generate_reference


def pose(x=0., y=0., z=0., angle=0.):
    result = np.eye(4)
    result[:3, :3] = Rotation.from_euler("z", angle).as_matrix()
    result[:3, 3] = [x, y, z]
    return result


def problem(phase, **requirements):
    """Use explicit analytic state/calibration, independent of source motion."""
    return PlanningProblem(joint_names=("analytic_joint",), start_q=np.array([.37]),
        start_dq=np.array([-.02]), tool_link="analytic_tool", reference_frame="world",
        base_pose=np.eye(4), tool_transform=np.eye(4), phases=(phase,),
        source_timestamps=phase.metadata["reference_timestamps"],
        source_progress=phase.reference.progress, **requirements)


def sample(problem, progress):
    """Convert the actual solver-facing tensors using independent SciPy math."""
    p, q = problem.sample_reference(torch.tensor(progress, dtype=torch.float64))
    result = np.broadcast_to(np.eye(4), (*p.shape[:-1], 4, 4)).copy()
    result[..., :3, 3] = p.numpy()
    result[..., :3, :3] = Rotation.from_quat(q.numpy().reshape(-1, 4)[:, [1, 2, 3, 0]]
        ).as_matrix().reshape(*p.shape[:-1], 3, 3)
    return result


class ReferencePlanningHandoffTest(unittest.TestCase):
    def test_grasp_clock_and_typed_goal_remain_independent(self):
        times, frames = [5., 5.2, 6.4], [4, 10, 45]
        event = {"event_id": "observed_contact", "timestamp_s": 5.7, "source": "analytic observation"}
        rep = build_representation(kind="grasp", frame_indices=frames, timestamps=times,
            T_world_carrier=np.repeat(np.eye(4)[None], 3, axis=0),
            T_world_eef=np.array([pose(z=.3), pose(z=.2), pose(z=.05)]),
            L_G=pose(.1), interaction_time=5.7, events=[event])
        # BEGIN an actual tool entrance is independent of source-reference samples.
        reference = generate_reference(rep, T_world_function=pose(.7, .2), T_world_eef=pose(.4, -.1, .6, .2))
        # END planner handoff preserves the adapted path and independent task goals.
        phase = reference_phase(reference, skill_id="analytic_grasp", source="observed transformed reference")
        empty = problem(phase)
        self.assertEqual(empty.pose_events, ())
        self.assertEqual((empty.events, empty.pose_events, empty.pose_relations, empty.mechanisms), ((), (), (), ()))
        np.testing.assert_array_equal(phase.metadata["source_frame_indices"], frames)
        np.testing.assert_array_equal(phase.metadata["source_timestamps"], times)
        self.assertEqual(phase.metadata["source_events"], [event])
        self.assertNotIn("mode", phase.metadata)
        self.assertNotIn("carrier_id", phase.metadata)
        # A task goal is separately authored; neither the path end nor the
        # inserted interaction knot may replace its target or tolerances.
        target = pose(.9, -.1, .4)
        goal = PoseEventRequirement(1., np.eye(4), target, .004, .03,
                                   source="independent analytic task specification")
        compiled = problem(phase, pose_events=(goal,))
        self.assertEqual(compiled.pose_events, (goal,))
        np.testing.assert_array_equal(goal.target, target)
        self.assertEqual((goal.position_tolerance_m, goal.rotation_tolerance_rad), (.004, .03))
        np.testing.assert_allclose(sample(compiled, phase.reference.progress), reference["eef_poses"], atol=1e-12)
        np.testing.assert_allclose(compiled.progress_to_source_time(phase.reference.progress), reference["eef_timestamps"])
        np.testing.assert_array_equal(compiled.start_q, [.37])
        np.testing.assert_array_equal(compiled.start_dq, [-.02])

    def held_reference(self):
        rep = build_representation(kind="manipulation", frame_indices=[5, 40], timestamps=[5., 7.],
            T_world_carrier=np.array([pose(), pose(angle=np.pi/2)]),
            T_world_reference=np.repeat(np.eye(4)[None], 2, axis=0), L_R=np.eye(4),
            interaction_time=6., release_time=6.5)
        A = pose(-.2, .05, .1, -.2)
        # BEGIN explicit matching entry keeps this fixture focused on the arm arc.
        return generate_reference(rep, T_world_function=np.eye(4), J=np.eye(4), A=A, T_world_body=np.eye(4)), A
        # END a missing current entrance can no longer create a production phase.

    def test_held_reference_rotates_tool_arm_and_ends_at_release(self):
        reference, A = self.held_reference()
        phase = reference_phase(reference, skill_id="analytic_holding", source="measured attachment")
        self.assertIs(phase.reference, reference["eef_path"])
        compiled = problem(phase)
        query = np.array([0., .5, 1.])
        expected = np.array([pose(angle=t*np.pi/2) @ _inverse(A) for t in [0., .375, .75]])
        np.testing.assert_allclose(sample(compiled, query), expected, atol=1e-12)
        np.testing.assert_allclose(compiled.progress_to_source_time(query), [5., 5.75, 6.5])
        # This interior point detects the erroneous interpolation of already
        # composed tool positions, which cuts the rotating attachment's arc.
        chord_midpoint = .5*(reference["eef_poses"][0, :3, 3]+reference["eef_poses"][-1, :3, 3])
        self.assertGreater(np.linalg.norm(expected[1, :3, 3]-chord_midpoint), .02)
        self.assertEqual(phase.metadata["reference_timestamps"][-1], 6.5)
        self.assertEqual(phase.metadata["source_timestamps"][-1], 7.)
        self.assertEqual(len(compiled.phases), 1)
        self.assertEqual(compiled.conditions, ())
        self.assertEqual(compiled.pose_events, ())

    def test_mechanism_interpolates_scalar_before_fk_and_attachment(self):
        model = MotionModel("revolute", np.array([[0.], [np.pi]]), pose(.4),
                            np.eye(3), np.zeros(3), 1e-4, 1e-4, {})
        rep = build_representation(kind="manipulation", frame_indices=[8, 20], timestamps=[1., 3.],
                                   interaction_time=2., release_time=2.5)
        A = pose(-.15, .03, .1)
        reference = generate_reference(rep, motion_model=model, source_coordinates=[.1, .1+np.pi],
                                       current_coordinates=.2, A=A)
        phase = reference_phase(reference, skill_id="analytic_hinge", source="observed scalar mechanism")
        compiled = problem(phase)
        # BEGIN verify actual scalar FK through the composed production tool path.
        # The tool's fixed attachment is now a separate parent composition;
        # the physical midpoint, rather than object layout, is the contract.
        angle = .2+.375*np.pi
        expected = model.poses([[angle]])[0] @ _inverse(A)
        np.testing.assert_allclose(sample(compiled, .5), expected, atol=1e-12)
        np.testing.assert_allclose(compiled.progress_to_source_time(.5), 1.75)
        # END a Cartesian chord cannot satisfy this independent hinge midpoint.

    def test_missing_or_inconsistent_inputs_do_not_create_a_tool_reference(self):
        reference, A = self.held_reference()
        missing = deepcopy(reference)
        del missing["eef_poses"]
        with self.assertRaisesRegex(ValueError, "object motion alone is insufficient"):
            reference_phase(missing, skill_id="missing", source="test")
        missing = deepcopy(reference)
        del missing["eef_path"]
        with self.assertRaisesRegex(ValueError, "explicit composed eef_path"):
            reference_phase(missing, skill_id="missing_path", source="test")
        wrong_pose = deepcopy(reference)
        wrong_pose["eef_poses"][0, 0, 3] += .05
        with self.assertRaisesRegex(ValueError, "same timeline and poses"):
            reference_phase(wrong_pose, skill_id="wrong_pose", source="test")
        invalid = deepcopy(reference)
        invalid["eef_timestamps"][-1] = 7.1
        with self.assertRaisesRegex(ValueError, "original observed timeline"):
            reference_phase(invalid, skill_id="wrong_time", source="test")


class PosePathCompositionTest(unittest.TestCase):
    def test_body_correspondence_offset_rotates_before_the_tool_attachment(self):
        H = np.array([pose(), pose(angle=np.pi/2)])
        J, A = pose(.2), pose()
        path = PosePath([0., 1.], H, source="analytic relative motion", attachment=J @ _inverse(A))
        np.testing.assert_allclose(path.sample_numpy(.5)[:3, 3], [np.sqrt(.02), np.sqrt(.02), 0.], atol=1e-12)
        # Rebuilding from H@J would create (.1,.1), which is inside the arc.
        bad_chord = .5*((H[0] @ J)[:3, 3]+(H[1] @ J)[:3, 3])
        self.assertGreater(np.linalg.norm(path.sample_numpy(.5)[:3, 3]-bad_chord), .05)

    def test_dynamic_parent_values_and_torch_derivatives_match_composed_formula(self):
        def F(s):
            return pose(.1+.2*s, -.3+.1*s, .2, .6*s)

        def H(s):
            return pose(.05*s, .1+.04*s, .2-.03*s, .9*s)

        keys = np.array([0., .4, 1.])
        J, A = pose(.17, -.04, .03, .25), pose(-.09, .06, -.02, -.15)
        parent = PosePath(keys, np.array([F(s) for s in keys]), source="known current reference-body motion")
        path = PosePath(keys, np.array([H(s) for s in keys]), source="demonstrated relative motion",
                        attachment=J @ _inverse(A), parent=parent)
        values = np.array([.19, .63])
        expected = np.array([F(s) @ H(s) @ J @ _inverse(A) for s in values])
        np.testing.assert_allclose(path.sample_numpy(values), expected, atol=1e-12)
        progress = torch.tensor(values, dtype=torch.float64, requires_grad=True)
        position, quaternion = path.sample(progress)
        # These are the same sampled tensors used by pose/process residuals
        # and their automatic derivatives. Check all translation/rotation
        # components against independent central differences of the formula.
        actual = torch.cat((position, quaternion), dim=-1)
        derivative = torch.stack([torch.autograd.grad(actual[:, i].sum(), progress, retain_graph=True)[0]
                                  for i in range(7)], dim=-1).detach().numpy()

        def components(s):
            transform = F(s) @ H(s) @ J @ _inverse(A)
            q = Rotation.from_matrix(transform[:3, :3]).as_quat()[[3, 0, 1, 2]]
            return np.r_[transform[:3, 3], q]

        step = 1e-6
        expected_derivative = np.array([(components(s+step)-components(s-step))/(2*step) for s in values])
        np.testing.assert_allclose(derivative, expected_derivative, atol=1e-8, rtol=1e-7)
        requirement = PoseProcessRequirement(0., 1., path, np.zeros((6, 2)), source="independently required analytic relation")
        residual = requirement.error(position, quaternion, progress)
        np.testing.assert_allclose(residual.detach().numpy(), 0., atol=1e-12)

    def test_parent_breakpoints_cannot_disappear_from_process_anchors(self):
        parent = PosePath([0., .3, 1.], np.array([pose(), pose(.2), pose(.3)]), source="known parent")
        with self.assertRaisesRegex(ValueError, "every parent progress knot"):
            PosePath([0., 1.], np.array([pose(), pose(.1)]), source="child", parent=parent)
        path = PosePath([0., .3, 1.], np.array([pose(), pose(.03), pose(.1)]), source="child", parent=parent)
        np.testing.assert_allclose(path.sample_numpy(.3)[:3, 3], [.23, 0., 0.])

    def test_composed_reference_does_not_enter_world_frame_owner_serialization(self):
        ordinary = PosePath([0., 1.], np.array([pose(), pose()]), source="current static owner")
        query = SimpleNamespace(owner_ids=frozenset(("observed_part",)))
        owner = CollisionOwner("observed_part", query, ordinary, "observed geometry", True)
        self.assertIs(owner.pose_path, ordinary)
        composed = PosePath([0., 1.], np.array([pose(), pose()]), source="relative reference", parent=ordinary)
        with self.assertRaisesRegex(ValueError, "single world-frame path"):
            CollisionOwner("observed_part", query, composed, "observed geometry", True)
# END no analytic fixture claims robot planning or physical execution success.
