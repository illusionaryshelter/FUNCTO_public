"""CPU contracts for the shared, differentiable interaction planning problem.

Analytic poses and actuator states test requirements, frame conventions and
observation barriers. They are not robot IK, contact-physics or GPU benchmarks.
The fixtures deliberately keep source evidence separate from future attachment
assumptions and do not obtain geometry from a simulator or another project.
"""

# BEGIN independent process-constraint and existing-grounding integration tests.
from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from manipulation.goals import SupportSurface
from manipulation.motion import MotionModel, transfer_motion
from manipulation.planning_problem import (
    JointCoupling,
    MotionPhase,
    NumericalTolerance,
    PlanningEvent,
    PlanningProblem,
    PosePath,
    SupportRequirement,
    anchor_pose,
    build_planning_problem,
)
from manipulation.program import ground_program


def pose(position=(0., 0., 0.), angles=(0., 0., 0.)):
    """Construct independent reference transforms in metres and radians."""
    result = np.eye(4)
    result[:3, :3] = Rotation.from_euler("xyz", angles).as_matrix()
    result[:3, 3] = position
    return result


def path(poses, *, attachment=None):
    """Give each analytic observation a distinct, explicitly declared phase."""
    poses = np.asarray(poses)
    return PosePath(np.linspace(0., 1., len(poses)), poses,
                    source="analytic supplied reference",
                    attachment=np.eye(4) if attachment is None else attachment)


def problem(phases, **changes):
    """A small named joint domain, not an implied physical robot model."""
    options = {
        "joint_names": ("joint_a", "joint_b", "joint_c"),
        "start_q": np.zeros(3), "start_dq": np.zeros(3),
        "tool_link": "declared_tool_link", "reference_frame": "world",
        "base_pose": np.eye(4), "tool_transform": np.eye(4),
        "phases": tuple(phases),
    }
    options.update(changes)
    return PlanningProblem(**options)


def components(poses):
    """Convert independent matrices to the public wxyz tensor convention."""
    poses = np.asarray(poses)
    quaternion = Rotation.from_matrix(poses[..., :3, :3].reshape(-1, 3, 3)).as_quat()
    quaternion = quaternion[:, [3, 0, 1, 2]].reshape(*poses.shape[:-2], 4)
    return (torch.as_tensor(poses[..., :3, 3], dtype=torch.float64),
            torch.as_tensor(quaternion, dtype=torch.float64))


def matrices(position, quaternion):
    """Use SciPy, independently of production Torch quaternion operations."""
    xyz = position.detach().numpy()
    wxyz = quaternion.detach().numpy()
    result = np.broadcast_to(np.eye(4), (*xyz.shape[:-1], 4, 4)).copy()
    result[..., :3, 3] = xyz
    result[..., :3, :3] = Rotation.from_quat(
        wxyz.reshape(-1, 4)[:, [1, 2, 3, 0]]).as_matrix().reshape(*xyz.shape[:-1], 3, 3)
    return result


def grasp_input():
    """A grasp with irregular frame IDs and a protected four-node suffix."""
    relative = np.array([pose((0., 0., height))
                         for height in (.15, .12, .09, .06, .03, .02)])
    demo = {
        "T_carrier_eef": relative, "frame_indices": np.arange(11, 27, 3),
        "timestamps": 2. + np.arange(6) * .1, "protected_start_frame": 17,
        "gripper_events": [{"event_id": "open", "frame_index": 11, "command": "open"},
                           {"event_id": "close", "frame_index": 20, "command": "close"}],
        "tool_id": "parallel_gripper", "tool_frame": "calibrated_tcp",
    }
    scene = {
        "reference_frame": "world", "carrier_id": "carrier",
        "tool_id": "parallel_gripper", "tool_frame": "calibrated_tcp",
        "carrier_pose": pose((.1, .05, 0.)), "region_transform": pose(),
        "eef_pose": pose((-.1, .02, .2)),
    }
    return {"skill_id": "grasp", "program": {"skill_kind": "grasp", "demo": demo},
            "scene": scene, "barrier_can_stop": True}


def manipulation_input():
    """A transfer whose function, open intention and release are distinct."""
    objects = np.array([pose((x, 0., 0.)) for x in (0., .01, .03, .04, .04, .04, .04)])
    hand = np.array([pose((0., 0., z)) for z in (.02, .02, .02, .02, .02, .02, .05)])
    event = {"event_id": "function", "frame_index": 106,
             "carrier_frame": pose(), "reference_frame": pose(),
             "source": "independently supplied original functional event",
             "position_tolerance_m": 1e-7, "rotation_tolerance_rad": 1e-7}
    program = {
        "skill_kind": "manipulation", "task": {"relation": "transfer", "motion": "free"},
        "demo": {"T_reference_carrier": objects, "T_carrier_eef": hand,
                 "frame_indices": np.arange(100, 113, 2), "timestamps": 5. + np.arange(7) * .1,
                 "tool_id": "parallel_gripper", "tool_frame": "calibrated_tcp",
                 "functional_events": [event],
                 "gripper_events": [{"event_id": "open", "frame_index": 108, "command": "open"}],
                 "release_frame": 110},
    }
    scene = {
        "reference_frame": "world", "reference_pose": pose(),
        "carrier_id": "carrier", "reference_id": "target",
        "tool_id": "parallel_gripper", "tool_frame": "calibrated_tcp",
        "initial_pose": pose(), "eef_pose": pose((0., 0., .03)),
        "holding": {"verified": True, "source": "actual synchronized holding fixture",
                    "T_carrier_eef": pose((0., 0., .03))},
        "functional_bindings": {"function": {
            "carrier_id": "carrier", "reference_id": "target",
            "carrier_frame": pose(), "reference_frame": pose(),
            "source": "independently supplied part correspondence"}},
    }
    return {"skill_id": "transfer", "program": program, "scene": scene, "barrier_can_stop": True}


def compile_inputs(*skills):
    """Compile source programs with explicit current robot calibration."""
    return build_planning_problem(
        skills, joint_names=("joint_a", "joint_b", "joint_c"),
        start_q=np.zeros(3), start_dq=np.zeros(3), tool_link="declared_tool_link",
        reference_frame="world", base_pose=pose(), tool_transform=pose())


def on_input():
    """A finite support relation with a translated, rotated material frame."""
    skill = manipulation_input()
    skill["skill_id"] = "place"
    program, scene = skill["program"], skill["scene"]
    program["task"]["relation"] = "on"
    event = program["demo"]["functional_events"][0]
    event.pop("reference_frame")
    event["carrier_frame"] = pose((.02, -.03, .01), (.25, -.3, .4))
    scene.pop("reference_pose")
    body = np.array([[x, y, z] for x in (-.01, .01)
                     for y in (-.015, .015) for z in (0., .025)])
    scene["body_points"] = body
    scene["support"] = {
        "origin": [0., 0., 0.], "normal": [0., 0., 1.],
        "basis": [[1., 0.], [0., 1.], [0., 0.]],
        "boundary": [[-.3, -.3], [.3, -.3], [.3, .3], [-.3, .3]],
        "source": "independently observed finite support patch",
    }
    scene["functional_bindings"]["function"] = {
        "carrier_id": "carrier", "reference_id": "target", "carrier_mapping": pose(),
        "source": "evidenced body correspondence and material lower face",
        "on_geometry": {"carrier_region_points": body[body[:, 2] == 0].copy(),
                        "carrier_outward_normal": np.array([0., 0., -1.]),
                        "gap_tolerance": 1e-6, "normal_tolerance": 1e-6},
    }
    attachment = pose((.01, -.02, .06), (.3, -.4, .1))
    scene["holding"]["T_carrier_eef"] = attachment
    scene["eef_pose"] = scene["initial_pose"] @ attachment
    return skill


def prismatic_input():
    """Explicit slider coordinates with distinct source and current origins."""
    skill = manipulation_input()
    skill["skill_id"] = "slide"
    program, scene = skill["program"], skill["scene"]
    program["task"] = {"relation": "articulation", "motion": "prismatic"}
    coordinates = np.array([0., .01, .03, .04, .04, .04, .04])
    program["demo"]["motion_coordinates"] = coordinates
    program["demo"]["motion_coordinates_source"] = "observed signed slider coordinates"
    program["demo"]["functional_events"] = [{
        "event_id": "function", "frame_index": 106, "constraint_kind": "joint_coordinate",
        "coordinate_value": [.04], "coordinate_tolerance": 1e-7,
        "source": "explicit observed source opening extent",
    }]
    scene.pop("reference_pose")
    scene["initial_pose"] = pose((.1, 0., 0.))
    scene["eef_pose"] = scene["initial_pose"] @ scene["holding"]["T_carrier_eef"]
    scene["carrier_mapping"] = pose()
    scene["carrier_mapping_source"] = "explicit same carrier coordinates"
    scene["motion_model"] = {
        "kind": "prismatic", "reference_pose": pose(),
        "basis": Rotation.from_euler("y", np.pi/2).as_matrix(),
        "source": "independent current signed slider axis", "coordinate_limits": [[0., .3]],
    }
    scene["functional_bindings"]["function"] = {
        "carrier_id": "carrier", "reference_id": "target", "current_coordinates": [.14],
        "source": "requested opening in current slider coordinates",
    }
    return skill


def revolute_input():
    """A full recorded revolution whose endpoint angle is explicitly lifted."""
    skill = prismatic_input()
    skill["skill_id"] = "turn"
    program, scene = skill["program"], skill["scene"]
    program["task"]["motion"] = "revolute"
    coordinates = np.array([0., np.pi/2, np.pi, 2*np.pi, 2*np.pi, 2*np.pi, 2*np.pi])
    zero, center = pose((.2, 0., 0.)), np.zeros(3)
    rotations = Rotation.from_rotvec(coordinates[:, None]*np.array([0., 0., 1.])).as_matrix()
    source = np.repeat(pose()[None], len(coordinates), axis=0)
    source[:, :3, :3] = rotations
    source[:, :3, 3] = rotations @ np.array([.2, 0., 0.])
    program["demo"]["T_reference_carrier"] = source
    program["demo"]["motion_coordinates"] = coordinates
    program["demo"]["functional_events"][0]["coordinate_value"] = [2*np.pi]
    scene["initial_pose"] = zero
    scene["eef_pose"] = zero @ scene["holding"]["T_carrier_eef"]
    scene["motion_model"] = {
        "kind": "revolute", "reference_pose": zero, "basis": np.eye(3), "axis_point": center,
        "source": "independent current hinge axis and signed angle convention", "coordinate_limits": [[0., 7.]],
    }
    scene["functional_bindings"]["function"]["current_coordinates"] = [2*np.pi]
    return skill


def alternative_placement(skill, goal=None):
    """Build a permitted alternate trajectory using independent NumPy kernels."""
    grounded = ground_program(skill["program"], skill["scene"])
    objects = grounded["object_poses"]
    frame = (skill["scene"]["functional_bindings"]["function"]["carrier_mapping"]
             @ skill["program"]["demo"]["functional_events"][0]["carrier_frame"])
    attachment = skill["scene"]["holding"]["T_carrier_eef"]
    goal = pose((.12, .08, 0.), (0., 0., .7)) if goal is None else goal
    first = transfer_motion(objects[:4] @ frame, objects[0] @ frame,
                            goal_pose=goal @ frame)["poses"] @ np.linalg.inv(frame)
    second = transfer_motion(objects[3:] @ frame, goal @ frame)["poses"] @ np.linalg.inv(frame)
    held = np.concatenate((first, second[1:])) @ attachment
    release_delta = held[-1] @ np.linalg.inv(grounded["eef_poses"][5])
    retreat = release_delta @ grounded["eef_poses"][5:]
    result = np.concatenate((held, retreat[1:]))
    return result, release_delta


class PlanningProcessTest(unittest.TestCase):
    # BEGIN first-arrival evidence-anchor regression for a progress plateau.
    def test_event_anchor_uses_first_actual_fk_and_backpropagates_only_to_that_arrival(self):
        positions = torch.tensor([[[0., 0., 0.], [1., 2., 3.], [7., 8., 9.], [10., 11., 12.]],
                                  [[0., 0., 0.], [.1, .2, .3], [-1., -2., -3.], [-7., -8., -9.]]],
                                 dtype=torch.float64, requires_grad=True)
        rotations = Rotation.from_euler("xyz", [[.2, .4, -.7], [-.8, .3, 1.2],
                                               [.9, -.6, .2], [-1.3, .1, -.5]]).as_quat()[:, [3, 0, 1, 2]]
        quaternion = torch.tensor(np.stack((rotations, rotations[::-1].copy())),
                                  dtype=torch.float64, requires_grad=True)
        progress = torch.tensor([[0., 1., 1., 2.], [0., .5, 1., 1.]], dtype=torch.float64)
        position_at_event, quaternion_at_event = anchor_pose(positions, quaternion, progress, 1.)
        np.testing.assert_array_equal(position_at_event.detach().numpy()[:, 0], [[1., 2., 3.], [-1., -2., -3.]])
        np.testing.assert_array_equal(quaternion_at_event.detach().numpy()[:, 0], np.stack((rotations[1], rotations[1])))
        (position_at_event.sum()+quaternion_at_event.sum()).backward()
        expected_position, expected_quaternion = np.zeros((2, 4, 3)), np.zeros((2, 4, 4))
        expected_position[0, 1], expected_position[1, 2] = 1., 1.
        expected_quaternion[0, 1], expected_quaternion[1, 2] = 1., 1.
        np.testing.assert_array_equal(positions.grad.numpy(), expected_position)
        np.testing.assert_array_equal(quaternion.grad.numpy(), expected_quaternion)
        missing = anchor_pose(positions, quaternion, progress, 1.5)
        self.assertTrue(all(not torch.isfinite(value).any() for value in missing))
    # END first-arrival evidence-anchor regression for a progress plateau.

    def test_required_short_loop_knots_are_checked_without_locking_free_prefix_samples(self):
        free = PosePath(np.array([0., .17, .49, 1.]),
                        np.array([pose(), pose((.1, 0., 0.)), pose((.2, .1, 0.)), pose()]),
                        "only the approach endpoint is required")
        reference = PosePath(np.array([0., .501, .502, .503, 1.]),
                             np.array([pose(), pose(), pose((.02, 0., 0.)), pose(), pose()]),
                             "observed necessary short loop")
        request = problem([MotionPhase("approach", free, kind="free"),
                           MotionPhase("required_loop", reference)])
        np.testing.assert_allclose(request.anchors, [0., 1., 1.501, 1.502, 1.503, 2.], atol=1e-12)
        # A stopped candidate contributes no extra speed-based subdivision;
        # the required loop itself must still make its peak unavoidable.
        progress = torch.tensor(request.anchors[None], dtype=torch.float64)
        position = torch.zeros((*progress.shape, 3), dtype=torch.float64)
        quaternion = torch.zeros((*progress.shape, 4), dtype=torch.float64)
        quaternion[..., 0] = 1.
        self.assertGreater(request.evaluate(position, quaternion, progress).max().item(), .019)

    def test_mechanism_pause_and_reversal_knots_are_mandatory_queries(self):
        coordinates = np.array([[.1], [.13], [.13], [.08], [.15]])
        knots = np.array([0., .31, .32, .33, 1.])
        model = MotionModel("prismatic", coordinates, pose(), np.eye(3), None,
                            1e-6, 1e-6, {"source": "independent signed mechanism"})
        reference = PosePath(knots, model.poses(coordinates), "required pause and reversal",
                             motion_model=model, coordinates=coordinates)
        request = problem([MotionPhase("slide", reference)])
        np.testing.assert_array_equal(request.anchors, knots)
        measured = reference.sample_numpy(request.anchors)[:, 2, 3]
        np.testing.assert_allclose(measured, [.1, .13, .13, .08, .15], atol=1e-12)

    def test_on_endpoint_moves_within_support_without_freeing_local_motion(self):
        body = np.array([[x, y, z] for x in (-.01, .01)
                         for y in (-.015, .015) for z in (0., .025)])
        surface = SupportSurface(
            origin=np.zeros(3), basis=np.array([[1., 0.], [0., 1.], [0., 0.]]),
            normal=np.array([0., 0., 1.]),
            boundary=np.array([[-.3, -.3], [.3, -.3], [.3, .3], [-.3, .3]]),
            source="independent finite support fixture")
        attachment = pose((.02, -.01, .06), (.2, -.3, .1))
        reference = np.array([pose((0., 0., .2)), pose((.02, .01, .1)), pose((.04, 0., 0.))])
        phase = MotionPhase("place", path(reference, attachment=attachment),
                            endpoint_family={"end": "variable"})
        support = SupportRequirement(1., np.linalg.inv(attachment), body, body[body[:, 2] == 0],
                                     np.array([0., 0., -1.]), surface,
                                     1e-6, 1e-6, 0., "specified bottom face on observed support")
        request = problem([phase], supports=(support,))
        goal = pose((.12, .08, 0.), (0., 0., .4))
        # The pre-existing numerical transfer is an independent oracle for the
        # permitted endpoint-dependent family, not an arbitrary free path.
        alternative = transfer_motion(reference, reference[0], goal_pose=goal)["poses"] @ attachment
        position, quaternion = components(alternative[None])
        progress = torch.tensor([[0., .5, 1.]], dtype=torch.float64)
        self.assertTrue(torch.all(request.evaluate(position, quaternion, progress) == 0))
        position = position.clone()
        position[0, 1, 1] += .025
        terms = request.residuals(position, quaternion, progress)
        self.assertGreater(terms["phase:0:place"][0, 1].item(), .015)
        self.assertTrue(torch.all(terms["support:0"] == 0))

    def test_on_checks_face_normal_height_and_whole_body_footprint(self):
        body = np.array([[x, y, z] for x in (-.01, .01)
                         for y in (-.02, .02) for z in (0., .04)])
        surface = SupportSurface(
            origin=np.zeros(3), basis=np.array([[1., 0.], [0., 1.], [0., 0.]]),
            normal=np.array([0., 0., 1.]),
            boundary=np.array([[-.15, -.1], [.15, -.1], [.15, .1], [-.15, .1]]),
            source="finite support fixture")
        attachment = pose((.04, -.02, .07), (.2, -.3, .1))
        support = SupportRequirement(1., np.linalg.inv(attachment), body, body[body[:, 2] == 0],
                                     np.array([0., 0., -1.]), surface,
                                     1e-6, 1e-6, 0., "material lower face")
        actual = np.array([pose((.03, .04, 0.), (0., 0., .5)),
                           pose((.03, .04, .015)),
                           pose((.03, .04, -.01)),
                           pose((.03, .04, 0.), (.2, 0., 0.)),
                           pose((.149, 0., 0.))]) @ attachment
        position, quaternion = components(actual[None])
        value = support.evaluate(position, quaternion, NumericalTolerance())
        self.assertEqual(value[0, 0].item(), 0.)
        self.assertTrue(torch.all(value[0, 1:] > .005))

    def test_projection_also_enforces_declared_actuator_coupling(self):
        coupling = JointCoupling(("left", "right"), np.array([[1., -1.]]),
                                 np.array([.005]), "actual differential actuator relation")
        request = problem([MotionPhase("free", path([pose(), pose()]), kind="free")],
                          couplings=(coupling,))
        progress = torch.tensor([[.5]], dtype=torch.float64)
        position, quaternion = request.sample_reference(progress)
        joints = torch.tensor([[[.02, .016]]], dtype=torch.float64)
        equality = request.project_residual(position, quaternion, progress, 0,
                                            joint_position=joints, joint_names=("left", "right"))
        np.testing.assert_allclose(equality.numpy(), [[[-.001]]], atol=1e-12)

    def test_free_connection_can_change_but_required_middle_motion_cannot(self):
        entry = pose((.3, 0., 0.))
        request = problem([
            MotionPhase("reach", path([pose(), entry]), kind="free"),
            MotionPhase("required_loop", path([entry, pose((.3, .1, 0.)), entry])),
        ])
        progress = torch.tensor([[0., .5, 1., 1.5, 2.]], dtype=torch.float64)
        position, quaternion = request.sample_reference(progress)
        position = position.clone()
        position[0, 1] += torch.tensor([.1, -.2, .07])
        self.assertTrue(torch.equal(request.evaluate(position, quaternion, progress),
                                    torch.zeros_like(progress)))
        # Matching both endpoints is insufficient: the required local loop has
        # a meaningful middle position that an endpoint-only planner omits.
        position[0, 3, 1] = 0.
        violation = request.evaluate(position, quaternion, progress)
        self.assertGreater(violation[0, 3].item(), .09)
        self.assertEqual(violation[0, -1].item(), 0.)
        self.assertEqual(violation[0, 2].item(), 0.)

    def test_mode_boundary_is_one_pose_that_must_satisfy_both_sides(self):
        entry = pose((.2, -.1, .05))
        request = problem([
            MotionPhase("reach", path([pose(), entry]), kind="free"),
            MotionPhase("contact", path([entry, pose((.2, -.1, .08))])),
        ])
        progress = torch.tensor([[1.]], dtype=torch.float64)
        position, quaternion = request.sample_reference(progress)
        position = position + torch.tensor([[[.02, 0., 0.]]])
        terms = request.residuals(position, quaternion, progress)
        self.assertGreater(terms["phase:0:reach"].item(), .01)
        self.assertGreater(terms["phase:1:contact"].item(), .01)

    def test_required_process_drives_finite_position_and_orientation_gradients(self):
        reference = pose((.1, -.2, .3), (.2, -.3, .4))
        request = problem([MotionPhase("required", path([reference, reference]))])
        progress = torch.tensor([[.2, .8]], dtype=torch.float64)
        actual = np.repeat(reference[None, None], 2, axis=1)
        actual[0, 0] = reference @ pose((.015, -.02, .03), (.04, -.02, .01))
        position, quaternion = components(actual)
        position, quaternion = position.requires_grad_(), quaternion.requires_grad_()
        loss = request.evaluate(position, quaternion, progress)
        self.assertGreater(loss[0, 0].item(), 0.)
        self.assertEqual(loss[0, 1].item(), 0.)
        loss.sum().backward()
        self.assertTrue(torch.isfinite(position.grad).all())
        self.assertTrue(torch.isfinite(quaternion.grad).all())
        self.assertGreater(torch.linalg.vector_norm(position.grad[0, 0]).item(), 0.)
        self.assertGreater(torch.linalg.vector_norm(quaternion.grad[0, 0]).item(), 0.)
        np.testing.assert_allclose(position.grad[0, 1].numpy(), 0., atol=0.)
        np.testing.assert_allclose(quaternion.grad[0, 1].numpy(), 0., atol=0.)

    # BEGIN optimizer equality preferences preserve source freedoms and raw feasibility.
    def test_pose_equality_preference_keeps_gradients_inside_numerical_acceptance(self):
        request = problem([MotionPhase("required", path([pose(), pose()]))])
        progress = torch.tensor([[.5]], dtype=torch.float64)
        dx, rz = .5*request.numerical.position_m, .5*request.numerical.rotation_rad
        p, q = components(pose((dx, 0., 0.), (0., 0., rz))[None, None])
        p, q = p.requires_grad_(), q.requires_grad_()
        self.assertEqual(request.evaluate(p, q, progress).item(), 0.)
        preferred = request.optimizer_preferences(p, q, progress)
        self.assertAlmostEqual(preferred.item(), dx*dx+rz*rz, places=17)
        preferred.sum().backward()
        np.testing.assert_allclose(p.grad.numpy(), [[[2*dx, 0., 0.]]], atol=1e-15)
        self.assertTrue(bool(torch.isfinite(q.grad).all()))
        self.assertGreater(float(q.grad.abs().max()), 0.)
        torch.testing.assert_close(preferred, request.optimizer_preferences(p, -q, progress), rtol=0, atol=1e-17)

    def test_nonzero_finite_equality_uses_its_actual_target_local_value(self):
        reference = pose((.3, -.2, .1), (.2, -.4, .7))
        bounds = np.array([[.2, .2], [-1., 1.], [-np.inf, np.inf],
                           [-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf]])
        request = problem([MotionPhase("offset equality", path([reference, reference]), pose_bounds=bounds)])
        error = .5*request.numerical.position_m
        p, q = components((reference @ pose((.2+error, .7, -.8), (.2, -.3, .4)))[None, None])
        p, q = p.requires_grad_(), q.requires_grad_()
        progress = torch.tensor([[.5]], dtype=torch.float64)
        self.assertEqual(request.evaluate(p, q, progress).item(), 0.)
        preferred = request.optimizer_preferences(p, q, progress)
        self.assertAlmostEqual(preferred.item(), error*error, places=17)
        preferred.sum().backward()
        np.testing.assert_allclose(p.grad.numpy()[0, 0], 2*error*reference[:3, 0], atol=2e-15)
        np.testing.assert_allclose(q.grad.numpy(), 0., atol=1e-15)
        np.testing.assert_array_equal(request.phases[0].pose_bounds, bounds)

    def test_finite_width_and_infinite_axes_have_no_invented_center_preference(self):
        bounds = np.array([[0., 0.], [-.8, .8], [-np.inf, np.inf],
                           [-.5, .5], [-np.inf, np.inf], [-np.inf, np.inf]])
        request = problem([MotionPhase("allowed axes", path([pose(), pose()]), pose_bounds=bounds)])
        p, q = components(pose((0., .7, 10.), (.2, 0., 0.))[None, None])
        p, q = p.requires_grad_(), q.requires_grad_()
        progress = torch.tensor([[.4]], dtype=torch.float64)
        self.assertEqual(request.evaluate(p, q, progress).item(), 0.)
        preferred = request.optimizer_preferences(p, q, progress)
        self.assertEqual(preferred.item(), 0.)
        preferred.sum().backward()
        self.assertTrue(bool(torch.isfinite(p.grad).all() and torch.isfinite(q.grad).all()))
        np.testing.assert_array_equal(p.grad.numpy(), np.zeros((1, 1, 3)))
        np.testing.assert_array_equal(q.grad.numpy(), np.zeros((1, 1, 4)))
        unconstrained = problem([MotionPhase("fully free axes", path([pose(), pose()]),
                                             pose_bounds=np.tile([-np.inf, np.inf], (6, 1)))])
        self.assertEqual(unconstrained.optimizer_preferences(p, q, progress).item(), 0.)

    def test_free_phase_preference_only_applies_to_its_required_endpoint(self):
        request = problem([MotionPhase("approach", path([pose(), pose()]), kind="free"),
                           MotionPhase("unconstrained end", path([pose(), pose()]),
                                       kind="free", require_end=False)])
        progress = torch.tensor([[0., 1.-.5e-7, 1., 1.+2e-7, 1.5, 2.]], dtype=torch.float64)
        p, q = components(np.broadcast_to(pose((.01, 0., 0.)), (1, 6, 4, 4)).copy())
        preferred = request.optimizer_preferences(p, q, progress)
        np.testing.assert_allclose(preferred.numpy(), [[0., 1e-4, 1e-4, 0., 0., 0.]], atol=1e-18)
        raw = request.residuals(p, q, progress)
        np.testing.assert_array_equal(preferred.numpy() > 0, raw["phase:0:approach"].numpy() > 0)
        np.testing.assert_array_equal(raw["phase:1:unconstrained end"].numpy(), np.zeros((1, 6)))

    def test_endpoint_family_translation_yaw_and_shared_endpoint_gradient_remain_available(self):
        attachment = pose((.02, -.01, .06), (.2, -.3, .1))
        reference = np.array([pose((0., 0., .2)), pose((.02, .01, .1)), pose((.04, 0., 0.))])
        request = problem([MotionPhase("allowed on family", path(reference, attachment=attachment),
                                       endpoint_family={"end": "variable"})])
        goals = (pose((.12, .08, 0.), (0., 0., .4)), pose((-.06, .11, 0.), (0., 0., -.7)))
        # Existing NumPy grounding supplies each complete allowed family.
        # Neither substantial tangential relocation nor yaw may gain a cost
        # toward the saved nominal placement when its actual motion matches.
        alternatives = np.stack([transfer_motion(reference, reference[0], goal_pose=goal)["poses"] @ attachment
                                 for goal in goals])
        p, q = components(alternatives)
        progress = torch.tensor([[0., .5, 1.], [0., .5, 1.]], dtype=torch.float64)
        preferred = request.optimizer_preferences(p, q, progress)
        self.assertLess(float(preferred.max()), 1e-28)
        np.testing.assert_array_equal(request.evaluate(p, q, progress).numpy(), np.zeros((2, 3)))
        # A small interior defect must also differentiate through the actual
        # selected endpoint, instead of silently freezing that family member.
        p = p.clone()
        p[:, 1, 0] += .5*request.numerical.position_m
        p, q = p.requires_grad_(), q.requires_grad_()
        preferred = request.optimizer_preferences(p, q, progress)
        preferred.sum().backward()
        self.assertTrue(bool(torch.isfinite(p.grad).all() and torch.isfinite(q.grad).all()))
        self.assertGreater(float(torch.linalg.vector_norm(p.grad[:, -1])), 0.)

    def test_optimizer_pose_preference_does_not_mutate_any_raw_residual(self):
        request = problem([MotionPhase("required", path([pose(), pose()]))])
        progress = torch.tensor([[.2, .8]], dtype=torch.float64)
        dx = request.numerical.position_m
        p, q = components(np.array([[pose((.5*dx, 0., 0.)), pose((2*dx, 0., 0.))]]))
        p, q = p.requires_grad_(), q.requires_grad_()
        before = request.residuals(p, q, progress)
        preferred = request.optimizer_preferences(p, q, progress)
        preferred.sum().backward()
        after = request.residuals(p, q, progress)
        self.assertEqual(before.keys(), after.keys())
        for name in before:
            torch.testing.assert_close(before[name], after[name], rtol=0, atol=0)
        np.testing.assert_allclose(after["phase:0:required"].detach().numpy(), [[0., dx]], atol=1e-18)
        self.assertTrue(bool(torch.all(preferred > 0)))
    # END optimizer equality preferences preserve source freedoms and raw feasibility.

    def test_quaternion_double_cover_does_not_change_physical_requirement(self):
        reference = pose(angles=(.4, -.7, .2))
        request = problem([MotionPhase("required", path([reference, reference]))])
        progress = torch.tensor([[.35]], dtype=torch.float64)
        position, quaternion = components((reference @ pose(angles=(.02, .01, -.04)))[None, None])
        torch.testing.assert_close(request.evaluate(position, quaternion, progress),
                                   request.evaluate(position, -quaternion, progress), rtol=0., atol=1e-12)

    def test_allowed_band_is_target_local_and_not_an_equality_centerline(self):
        reference = pose(angles=(0., 0., np.pi/2))
        bounds = np.zeros((6, 2))
        bounds[0] = [-.10, .10]
        bounds[3:] = [-np.inf, np.inf]
        phase = MotionPhase("partial", path([reference, reference]), pose_bounds=bounds)
        request = problem([phase])
        progress = torch.tensor([[.5]], dtype=torch.float64)
        position, quaternion = components((reference @ pose((.08, 0., 0.), (0., 0., .4)))[None, None])
        self.assertEqual(request.evaluate(position, quaternion, progress).item(), 0.)
        equality = request.project_residual(position, quaternion, progress, 0)
        # Only Y and Z are equalities. The finite X interval and free rotations
        # must not become extra projection equations deleting valid solutions.
        self.assertEqual(equality.shape, (1, 1, 2))
        np.testing.assert_allclose(equality.numpy(), 0., atol=1e-12)
        position[0, 0, 0] += .015
        self.assertGreater(request.evaluate(position, quaternion, progress).item(), .014)
        np.testing.assert_array_equal(phase.pose_bounds, bounds)

    def test_mechanism_interpolation_keeps_hinge_arc_and_unwrapped_winding(self):
        center = np.array([.1, -.2, .3])
        zero = pose(center + [.2, 0., 0.], (.1, .2, -.3))
        coordinates = np.array([[0.], [np.pi], [2*np.pi]])
        model = MotionModel("revolute", coordinates, zero, np.eye(3), center,
                            1e-6, 1e-6, {"source": "analytic fixed hinge"})
        attachment = pose((.03, -.01, .04), (.2, -.1, .3))
        reference = PosePath(np.array([0., .5, 1.]), model.poses(coordinates),
                             "observed unwrapped mechanism coordinate fixture",
                             motion_model=model, coordinates=coordinates, attachment=attachment)
        query = torch.tensor([[.25, .75]], dtype=torch.float64, requires_grad=True)
        position, quaternion = reference.sample(query)
        expected = model.poses(np.array([[np.pi/2], [3*np.pi/2]])) @ attachment
        np.testing.assert_allclose(matrices(position, quaternion)[0], expected, atol=1e-12, rtol=0.)
        # A chord between the endpoints would have put the body at its axis.
        self.assertGreater(np.linalg.norm(expected[0, :3, 3]-center), .2)
        (position[..., 0].sum() + quaternion[..., 0].sum()).backward()
        self.assertTrue(torch.isfinite(query.grad).all())
        self.assertGreater(torch.linalg.vector_norm(query.grad).item(), 0.)

    def test_prismatic_pause_and_reversal_remain_in_phase_reference(self):
        coordinates = np.array([[.4], [.45], [.45], [.42], [.5]])
        zero = pose((.2, -.1, .4), (.3, -.2, .1))
        basis = Rotation.from_euler("xyz", [.2, .4, -.3]).as_matrix()
        model = MotionModel("prismatic", coordinates, zero, basis, None,
                            1e-6, 1e-6, {"source": "analytic slider"})
        reference = PosePath(np.linspace(0., 1., 5), model.poses(coordinates),
                             "signed source coordinates", motion_model=model, coordinates=coordinates)
        query = np.array([.25, .375, .5, .625, .75])
        actual = reference.sample_numpy(query)
        measured = (actual[:, :3, 3]-zero[:3, 3]) @ basis[:, 2]
        np.testing.assert_allclose(measured, [.45, .45, .45, .435, .42], atol=1e-12)

    def test_base_and_tcp_mapping_match_independent_noncommuting_composition(self):
        base = pose((.3, -.2, .5), (.2, .3, -.4))
        tcp = pose((.03, -.02, .07), (-.2, .4, .1))
        request = problem([MotionPhase("required", path([pose(), pose()]))],
                          base_pose=base, tool_transform=tcp)
        links = np.array([[pose((.1, .2, .3), (.1, -.5, .7)),
                           pose((-.2, .1, .4), (-.3, .2, .6))]])
        position, quaternion = components(links)
        tool_position, tool_quaternion = request.transform_tool_pose(position, quaternion)
        np.testing.assert_allclose(matrices(tool_position, tool_quaternion), base @ links @ tcp,
                                   atol=1e-12, rtol=0.)
        recovered = request.reference_to_link_pose(tool_position, tool_quaternion)
        np.testing.assert_allclose(matrices(*recovered), links, atol=1e-12, rtol=0.)
        spheres = torch.tensor([[[[.1, .2, .3, .04], [-.2, .3, .1, .02]]]], dtype=torch.float64)
        transformed = request.transform_robot_spheres(spheres).numpy()
        np.testing.assert_allclose(transformed[..., :3], spheres.numpy()[..., :3] @ base[:3, :3].T + base[:3, 3],
                                   atol=1e-12)
        np.testing.assert_array_equal(transformed[..., 3], spheres.numpy()[..., 3])

    def test_named_actuator_coupling_does_not_assume_joint_array_order(self):
        coupling = JointCoupling(("left_slide", "right_slide"), np.array([[1., -1.]]),
                                 np.array([.005]), "declared actuator coupling")
        request = problem([MotionPhase("free", path([pose(), pose()]), kind="free", require_end=False)],
                          couplings=(coupling,))
        progress = torch.tensor([[.25, .75]], dtype=torch.float64)
        position, quaternion = request.sample_reference(progress)
        names = ("right_slide", "arm", "left_slide")
        joints = torch.tensor([[[.015, .7, .02], [.015, .7, .025]]], dtype=torch.float64,
                              requires_grad=True)
        value = request.evaluate(position, quaternion, progress, joints, names)
        self.assertEqual(value[0, 0].item(), 0.)
        self.assertGreater(value[0, 1].item(), .004)
        value.sum().backward()
        self.assertLess(joints.grad[0, 1, 0].item(), 0.)
        self.assertGreater(joints.grad[0, 1, 2].item(), 0.)
        self.assertEqual(joints.grad[0, 1, 1].item(), 0.)

    def test_source_event_ids_are_scoped_by_skill_and_barriers_require_stops(self):
        phases = [MotionPhase("free", path([pose(), pose()]), kind="free", require_end=False)]
        events = (PlanningEvent("open", "first", .2, 42, "gripper", "first source intention"),
                  PlanningEvent("open", "second", .7, 42, "gripper", "second source intention"),
                  PlanningEvent("holding", "first", 1., 43, "verification", "observed source boundary",
                                requires_evidence="synchronized actual Holding", stop=True))
        request = problem(phases, events=events)
        self.assertEqual([event.event_id for event in request.events], ["open", "open", "holding"])
        np.testing.assert_allclose(request.anchors, [0., .2, .7, 1.])
        progress = torch.tensor([[.7, 1.]], dtype=torch.float64)
        position, quaternion = request.sample_reference(progress)
        with self.assertRaisesRegex(ValueError, "velocity"):
            request.evaluate(position, quaternion, progress)
        velocity = torch.tensor([[[.1, .2, .3], [.02, 0., 0.]]], dtype=torch.float64)
        value = request.evaluate(position, quaternion, progress, joint_velocity=velocity)
        self.assertEqual(value[0, 0].item(), 0.)
        self.assertGreater(value[0, 1].item(), .019)
        with self.assertRaisesRegex(ValueError, "planned stop"):
            problem(phases, events=(PlanningEvent("holding", "first", 1., 43, "verification", "source",
                                                 requires_evidence="actual Holding"),))

    def test_numerical_acceptance_does_not_mutate_declared_skill_bounds(self):
        declared = np.zeros((6, 2))
        declared[0] = [-.02, .02]
        phase = MotionPhase("bounded", path([pose(), pose()]), pose_bounds=declared)
        numerical = NumericalTolerance(position_m=2e-5, rotation_rad=3e-4)
        request = problem([phase], numerical=numerical)
        progress = torch.tensor([[.5]], dtype=torch.float64)
        position, quaternion = request.sample_reference(progress)
        position[0, 0, 0] = .025
        self.assertGreater(request.evaluate(position, quaternion, progress).item(), .004)
        np.testing.assert_array_equal(phase.pose_bounds, declared)
        self.assertEqual(request.numerical.position_m, 2e-5)


class PlanningBuilderTest(unittest.TestCase):
    def test_declared_support_tolerances_are_not_widened_by_solver_numerics(self):
        skill = on_input()
        request = compile_inputs(skill)
        attachment = skill["scene"]["holding"]["T_carrier_eef"]
        actual = np.array([pose((.02, .03, 2e-5)),
                           pose((.02, .03, 0.), (2e-4, 0., 0.))]) @ attachment
        position, quaternion = components(actual[None])
        value = request.supports[0].evaluate(position, quaternion, request.numerical)
        self.assertTrue(torch.all(value > 1e-5))

    def test_revolute_event_measures_radians_and_retains_the_full_recorded_turn(self):
        request = compile_inputs(revolute_input())
        self.assertAlmostEqual(request.phases[0].reference.coordinates[-1, 0], 2*np.pi)
        progress = torch.tensor([[0., 2./3, 1., 1.5, 2., 3.]], dtype=torch.float64)
        actual = matrices(*request.sample_reference(progress))
        self.assertLess(actual[0, 1, 0, 3], -.19)
        actual[0, 2] = pose(angles=(0., 0., 2e-4)) @ actual[0, 2]
        position, quaternion = components(actual)
        value = request.evaluate(position, quaternion, progress,
                                 joint_velocity=torch.zeros((1, 6, 3)))
        self.assertGreater(value[0, 2].item(), 1e-4)
        measurements = request.measurements(position, quaternion, progress)
        np.testing.assert_allclose(measurements["coordinate:0:error_rad"][0, 2].item(), 2e-4, atol=1e-12)

    def test_declared_coordinate_event_tolerance_keeps_its_own_metric(self):
        skill = prismatic_input()
        request = compile_inputs(skill)
        progress = torch.tensor([[0., 1., 1.5, 2., 3.]], dtype=torch.float64)
        position, quaternion = request.sample_reference(progress)
        position = position.clone()
        position[0, 1, 0] += 2e-5
        value = request.evaluate(position, quaternion, progress,
                                 joint_velocity=torch.zeros((1, 5, 3)))
        self.assertGreater(value[0, 1].item(), 1e-5)

    def test_successor_mechanism_coordinates_start_at_the_predecessor_state(self):
        first, second = prismatic_input(), prismatic_input()
        first["program"]["demo"]["release_frame"] = None
        first["program"]["demo"]["gripper_events"] = []
        second["skill_id"] = "next_slide"
        second["scene"]["functional_bindings"]["function"]["current_coordinates"] = [.18]
        second["scene"]["holding"]["verified"] = False
        request = compile_inputs(first, second)
        previous, successor = request.phases[1], request.phases[2]
        np.testing.assert_allclose(previous.reference.coordinates[-1], [.14], atol=1e-12)
        np.testing.assert_allclose(successor.reference.coordinates[0], [.14], atol=1e-12)
        np.testing.assert_allclose(successor.reference.coordinates[-1], [.18], atol=1e-12)
        np.testing.assert_allclose(previous.reference.sample_numpy(np.array([1.]))[0],
                                   successor.reference.sample_numpy(np.array([0.]))[0], atol=1e-12)
        self.assertFalse(second["scene"]["holding"]["verified"])
        self.assertEqual(second["scene"]["initial_pose"][0, 3], .1)

    def test_declared_function_event_tolerance_is_not_replaced_by_global_numerics(self):
        skill = manipulation_input()
        request = compile_inputs(skill)
        event = skill["program"]["demo"]["functional_events"][0]
        self.assertEqual(event["position_tolerance_m"], 1e-7)
        self.assertGreater(request.numerical.position_m, event["position_tolerance_m"])
        progress = torch.tensor([[0., 1., 1.5, 2., 3.]], dtype=torch.float64)
        nominal = matrices(*request.sample_reference(progress))
        for defect in ("position", "rotation"):
            actual = nominal.copy()
            if defect == "position":
                actual[0, 1, 0, 3] += 2e-5
            else:
                actual[0, 1] = actual[0, 1] @ pose(angles=(2e-4, 0., 0.))
            position, quaternion = components(actual)
            value = request.evaluate(position, quaternion, progress,
                                     joint_velocity=torch.zeros((1, 5, 3)))
            with self.subTest(defect=defect):
                self.assertGreater(value[0, 1].item(), 1e-5)

    def test_still_held_on_successor_uses_the_actual_boundary_for_its_fixed_goal_family(self):
        place, transfer = on_input(), manipulation_input()
        place["program"]["demo"]["release_frame"] = None
        place["program"]["demo"]["gripper_events"] = []
        request = compile_inputs(place, transfer)
        nominal = ground_program(place["program"], place["scene"])
        frame = place["program"]["demo"]["functional_events"][0]["carrier_frame"]
        goal = pose((.12, .08, 0.), (0., 0., .7))
        objects = nominal["object_poses"]
        first = transfer_motion(objects[:4] @ frame, objects[0] @ frame,
                                goal_pose=goal @ frame)["poses"] @ np.linalg.inv(frame)
        second = transfer_motion(objects[3:] @ frame, goal @ frame)["poses"] @ np.linalg.inv(frame)
        attachment = place["scene"]["holding"]["T_carrier_eef"]
        held = np.concatenate((first, second[1:])) @ attachment
        successor = request.phases[2]
        body = successor.reference.poses
        next_body = transfer_motion(body, second[-1], goal_pose=body[-1])["poses"]
        final_progress = torch.tensor([[3.5, 4., 5.]], dtype=torch.float64)
        suffix = matrices(*request.sample_reference(final_progress))[0]
        actual = np.concatenate((held, (next_body @ attachment)[1:], suffix))
        progress = torch.tensor([[0., 1./3, 2./3, 1., 1.+1./3, 1.+2./3, 2.,
                                  2.+1./3, 2.+2./3, 3., 3.5, 4., 5.]], dtype=torch.float64)
        position, quaternion = components(actual[None])
        residuals = request.residuals(position, quaternion, progress,
                                      joint_velocity=torch.zeros((1, progress.shape[1], 3)))
        for name, value in residuals.items():
            with self.subTest(term=name):
                np.testing.assert_allclose(value.numpy(), 0., atol=1e-12)

    def test_fixed_search_proposal_checks_single_states_without_freezing_the_full_problem(self):
        skill = on_input()
        request = compile_inputs(skill)
        progress = torch.tensor([[0., 1./3, 2./3, 1., 1.5, 2., 3.]], dtype=torch.float64)
        first, _ = alternative_placement(skill)
        position, quaternion = components(first[None])
        with self.assertRaisesRegex(ValueError, "explicitly frozen whole-window proposal"):
            request.project_residual(position[:, 1:2], quaternion[:, 1:2], progress[:, 1:2], 0)
        frozen = request.with_fixed_endpoints(position, quaternion, progress)
        for row, phase_index in ((1, 0), (4, 1), (6, 2)):
            # An OMPL callback sees one state; its endpoint family comes from
            # the one complete proposal, never from a missing nearest sample.
            p, q, s = position[:, row:row+1], quaternion[:, row:row+1], progress[:, row:row+1]
            with self.subTest(phase=phase_index):
                residual = frozen.project_residual(p, q, s, phase_index)
                np.testing.assert_allclose(residual.numpy(), 0., atol=1e-12)
                np.testing.assert_allclose(frozen.evaluate(p, q, s,
                                                          joint_velocity=torch.zeros((1, 1, 3))).numpy(), 0., atol=1e-12)
        other, _ = alternative_placement(skill, pose((.08, -.06, 0.), (0., 0., -.3)))
        other_p, other_q = components(other[None])
        velocity = torch.zeros((1, len(other), 3), dtype=torch.float64)
        np.testing.assert_allclose(request.evaluate(other_p, other_q, progress,
                                                    joint_velocity=velocity).numpy(), 0., atol=1e-12)
        self.assertGreater(frozen.evaluate(other_p, other_q, progress,
                                           joint_velocity=velocity).max().item(), .05)
        np.testing.assert_allclose(request.evaluate(position, quaternion, progress,
                                                    joint_velocity=velocity).numpy(), 0., atol=1e-12)
        # Event 108 at progress 1.5 remains mandatory even though it is not a
        # pose-phase boundary. A proposal must not silently round that event.
        kept = [0, 1, 2, 3, 5, 6]
        with self.assertRaisesRegex(ValueError, "every exact problem anchor"):
            request.with_fixed_endpoints(position[:, kept], quaternion[:, kept], progress[:, kept])

    def test_on_material_frame_family_and_retreat_follow_the_actual_release(self):
        skill = on_input()
        request = compile_inputs(skill)
        actual, _ = alternative_placement(skill)
        progress = torch.tensor([[0., 1./3, 2./3, 1., 1.5, 2., 3.]], dtype=torch.float64)
        position, quaternion = components(actual[None])
        velocity = torch.zeros((1, progress.shape[1], 3), dtype=torch.float64)
        residuals = request.residuals(position, quaternion, progress, joint_velocity=velocity)
        for name, value in residuals.items():
            with self.subTest(term=name):
                np.testing.assert_allclose(value.numpy(), 0., atol=1e-12)
        # Keeping the nominal retreat after moving the placed object fails the
        # same shared process checker; release is a physical anchor, not an ID.
        wrong_position = position.clone()
        wrong_position[0, -1] += torch.tensor([.03, 0., 0.])
        self.assertGreater(request.evaluate(wrong_position, quaternion, progress,
                                            joint_velocity=velocity)[0, -1].item(), .02)

    def test_on_carrier_coordinate_gauge_does_not_move_physical_joint_requirements(self):
        skill = on_input()
        changed = deepcopy(skill)
        gauge = pose((.03, -.04, .05), (.2, -.1, .3))
        inverse = np.linalg.inv(gauge)
        old, scene = skill["scene"], changed["scene"]
        scene["initial_pose"] = old["initial_pose"] @ gauge
        scene["holding"]["T_carrier_eef"] = inverse @ old["holding"]["T_carrier_eef"]
        scene["body_points"] = old["body_points"] @ inverse[:3, :3].T + inverse[:3, 3]
        binding = scene["functional_bindings"]["function"]
        binding["carrier_mapping"] = inverse
        face = binding["on_geometry"]
        face["carrier_region_points"] = face["carrier_region_points"] @ inverse[:3, :3].T + inverse[:3, 3]
        face["carrier_outward_normal"] = inverse[:3, :3] @ face["carrier_outward_normal"]
        requests = (compile_inputs(skill), compile_inputs(changed))
        progress = torch.tensor([[0., 1./3, 2./3, 1., 1.5, 2., 3.]], dtype=torch.float64)
        np.testing.assert_allclose(matrices(*requests[0].sample_reference(progress)),
                                   matrices(*requests[1].sample_reference(progress)), atol=1e-12)
        actual, _ = alternative_placement(skill)
        position, quaternion = components(actual[None])
        velocity = torch.zeros((1, progress.shape[1], 3), dtype=torch.float64)
        for request in requests:
            np.testing.assert_allclose(request.evaluate(position, quaternion, progress,
                                                        joint_velocity=velocity).numpy(), 0., atol=1e-12)

    def test_regrasp_tracks_the_actual_variable_placement_instead_of_its_nominal_pose(self):
        place, grasp = on_input(), grasp_input()
        request = compile_inputs(place, grasp)
        placement, release_delta = alternative_placement(place)
        grasp_phases = [phase for phase in request.phases if phase.metadata.get("skill_id") == "grasp"]
        nominal_grasp = np.concatenate((grasp_phases[0].reference.sample_numpy(np.linspace(0., 1., 3)),
                                        grasp_phases[1].reference.sample_numpy(np.linspace(0., 1., 4))[1:]))
        actual = np.concatenate((placement, (release_delta @ nominal_grasp)[1:]))
        progress = torch.tensor([[0., 1./3, 2./3, 1., 1.5, 2., 3., 3.5,
                                  4., 4.+1./3, 4.+2./3, 5.]], dtype=torch.float64)
        position, quaternion = components(actual[None])
        velocity = torch.zeros((1, progress.shape[1], 3), dtype=torch.float64)
        residuals = request.residuals(position, quaternion, progress, joint_velocity=velocity)
        for name, value in residuals.items():
            with self.subTest(term=name):
                np.testing.assert_allclose(value.numpy(), 0., atol=1e-12)

    def test_future_manipulation_uses_candidate_without_fabricating_actual_holding(self):
        grasp, transfer = grasp_input(), manipulation_input()
        transfer["scene"]["holding"]["verified"] = False
        transfer["scene"]["holding"]["source"] = "no actual future evidence"
        transfer["scene"]["initial_pose"] = pose((-.4, -.3, -.2))
        before = deepcopy(transfer)
        with patch("manipulation.program.ground_program", wraps=ground_program) as actual_consumer:
            request = compile_inputs(grasp, transfer)
        self.assertEqual(actual_consumer.call_count, 1)
        self.assertEqual(actual_consumer.call_args.args[0]["skill_kind"], "grasp")
        with self.assertRaisesRegex(ValueError, "actual holding"):
            ground_program(transfer["program"], transfer["scene"])
        grasp_end = next(event.progress for event in request.events
                         if event.skill_id == "grasp" and event.kind == "verification")
        previous = request.phases[int(grasp_end)-1].reference.sample_numpy(np.array([1.]))[0]
        successor = request.phases[int(grasp_end)].reference.sample_numpy(np.array([0.]))[0]
        np.testing.assert_allclose(successor, previous, atol=1e-12)
        prediction = next(item for item in request.conditions if item["kind"] == "predicted_skill_boundary")
        np.testing.assert_allclose(prediction["initial_pose"], grasp["scene"]["carrier_pose"], atol=1e-12)
        np.testing.assert_allclose(prediction["T_carrier_eef"],
                                   grasp["program"]["demo"]["T_carrier_eef"][-1], atol=1e-12)
        self.assertFalse(transfer["scene"]["holding"]["verified"])
        self.assertEqual(transfer["scene"]["holding"]["source"], before["scene"]["holding"]["source"])
        np.testing.assert_array_equal(transfer["scene"]["initial_pose"], before["scene"]["initial_pose"])

    def test_protected_first_observation_gets_an_explicit_connection(self):
        skill = grasp_input()
        demo, scene = skill["program"]["demo"], skill["scene"]
        demo["protected_start_frame"] = int(demo["frame_indices"][0])
        request = compile_inputs(skill)
        self.assertEqual(len(request.phases), 2)
        self.assertEqual(request.phases[0].kind, "free")
        self.assertTrue(request.phases[0].metadata["no_source_frames"])
        expected = scene["carrier_pose"] @ scene["region_transform"] @ demo["T_carrier_eef"]
        np.testing.assert_allclose(request.phases[0].reference.poses,
                                   np.stack((scene["eef_pose"], expected[0])), atol=1e-12)
        np.testing.assert_allclose(request.phases[1].reference.poses, expected, atol=1e-12)
        self.assertEqual(request.phases[1].metadata["source_frames"], demo["frame_indices"].tolist())

    def test_function_open_release_and_retreat_keep_distinct_source_boundaries(self):
        request = compile_inputs(manipulation_input())
        events = {event.event_id: event for event in request.events}
        self.assertEqual(events["function"].source_frame, 106)
        self.assertEqual(events["open"].source_frame, 108)
        self.assertEqual(events["__release_verified__"].source_frame, 110)
        self.assertLess(events["function"].progress, events["open"].progress)
        self.assertLess(events["open"].progress, events["__release_verified__"].progress)
        retreat = request.phases[int(events["__release_verified__"].progress)]
        self.assertEqual(retreat.metadata["mode"], "released_retreat")
        self.assertTrue(events["__release_verified__"].stop)
        self.assertEqual(retreat.metadata["source_frames"], [110, 112])
        self.assertGreater(retreat.reference.poses[-1, 2, 3], retreat.reference.poses[0, 2, 3])

    def test_current_manipulation_still_requires_real_synchronized_holding(self):
        for defect in ("candidate", "unsynchronized"):
            skill = manipulation_input()
            if defect == "candidate":
                skill["scene"]["holding"]["verified"] = False
            else:
                skill["scene"]["eef_pose"][0, 3] += .01
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                compile_inputs(skill)

    def test_future_manipulation_keeps_current_tool_and_release_validation(self):
        # Conditional attachment removes an evidence prerequisite only; it
        # cannot authorize a different TCP or redefine opening as release.
        for defect in ("tool_id", "tool_frame", "missing_release"):
            grasp, transfer = grasp_input(), manipulation_input()
            transfer["scene"]["holding"]["verified"] = False
            if defect == "missing_release":
                transfer["program"]["demo"]["release_frame"] = None
            else:
                transfer["scene"][defect] = "another_tool_or_frame"
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                compile_inputs(grasp, transfer)

    def test_barriers_need_declared_safe_holds_and_cannot_reuse_a_released_attachment(self):
        for skill in (grasp_input(), manipulation_input()):
            skill["barrier_can_stop"] = False
            with self.subTest(kind=skill["program"]["skill_kind"]), self.assertRaisesRegex(ValueError, "boundar"):
                compile_inputs(skill)
        first, second = manipulation_input(), manipulation_input()
        second["skill_id"] = "next_transfer"
        with self.assertRaisesRegex(ValueError, "still-held carrier"):
            compile_inputs(first, second)

    def test_regrasp_inherits_predicted_placed_object_and_previous_retreat_end(self):
        transfer, grasp = manipulation_input(), grasp_input()
        actual = ground_program(transfer["program"], transfer["scene"])
        request = compile_inputs(transfer, grasp)
        phases = [phase for phase in request.phases if phase.metadata.get("skill_id") == "grasp"]
        np.testing.assert_allclose(phases[0].reference.poses[0], actual["eef_poses"][-1], atol=1e-12)
        expected = (actual["object_poses"][-1] @ grasp["scene"]["region_transform"]
                    @ grasp["program"]["demo"]["T_carrier_eef"][-1])
        np.testing.assert_allclose(phases[-1].reference.poses[-1], expected, atol=1e-12)
        # The stale caller scene is retained intact for future observation.
        np.testing.assert_allclose(grasp["scene"]["carrier_pose"], pose((.1, .05, 0.)), atol=1e-12)

    # BEGIN attachment provenance and exact collision-span contract regressions.
    def test_actual_holding_spans_merge_across_skills_until_the_real_release(self):
        first, second = manipulation_input(), manipulation_input()
        first["skill_id"], second["skill_id"] = "carry", "place"
        first["program"]["demo"]["release_frame"] = None
        first["program"]["demo"]["gripper_events"] = [
            {"event_id": "hold", "frame_index": 100, "command": "hold"}]
        second["scene"]["holding"]["verified"] = False
        second["scene"]["holding"]["source"] = "future sensor reading has not happened"
        first["scene"]["holding"]["observation_ids"] = ["current_wrist_31", "current_fingers_31"]
        original_holding = deepcopy(first["scene"]["holding"])
        request = compile_inputs(first, second)
        held = [phase for phase in request.phases if phase.metadata.get("mode") == "held"]
        self.assertEqual(len(held), 4)
        for phase in held:
            with self.subTest(phase=phase.name):
                self.assertEqual(phase.metadata["attachment_interval"], [0., 4.])
                self.assertEqual(phase.metadata["attachment_condition_id"], "carry:actual_holding")
                self.assertEqual(phase.metadata["release_condition_id"], "place:actual_release")
                np.testing.assert_array_equal(phase.metadata["attachment"], original_holding["T_carrier_eef"])
                np.testing.assert_array_equal(phase.metadata["owner_initial_pose"], first["scene"]["initial_pose"])
        observed = [item for item in request.conditions if item["kind"] == "observed_attachment"]
        self.assertEqual(len(observed), 1)
        self.assertEqual(observed[0]["condition_id"], "carry:actual_holding")
        self.assertEqual(observed[0]["source"]["observation_ids"], original_holding["observation_ids"])
        self.assertTrue(observed[0]["source"]["verified"])
        first["scene"]["holding"]["observation_ids"].append("later_external_edit")
        self.assertEqual(observed[0]["source"]["observation_ids"], original_holding["observation_ids"])
        self.assertFalse(second["scene"]["holding"]["verified"])

    def test_still_held_window_end_does_not_manufacture_release_or_evidence(self):
        skill = manipulation_input()
        skill["program"]["demo"]["release_frame"] = None
        skill["program"]["demo"]["gripper_events"] = [
            {"event_id": "hold", "frame_index": 100, "command": "hold"}]
        request = compile_inputs(skill)
        self.assertEqual(request.phase_count, 2)
        for phase in request.phases:
            self.assertEqual(phase.metadata["attachment_interval"], [0., 2.])
            self.assertIsNone(phase.metadata["release_condition_id"])
        self.assertFalse(any(item["kind"] == "predicted_release" for item in request.conditions))
        self.assertFalse(any(event.requires_evidence is not None for event in request.events))
        self.assertEqual(len([item for item in request.conditions if item["kind"] == "observed_attachment"]), 1)

        # A terminal grasp has a future verification barrier but no later
        # carrying interval. It must not create a zero-length attachment or
        # silently label its candidate as an observed Holding.
        terminal = compile_inputs(grasp_input())
        self.assertFalse(any("attachment_interval" in phase.metadata for phase in terminal.phases))
        self.assertFalse(any(item["kind"] == "observed_attachment" for item in terminal.conditions))
        barrier = next(event for event in terminal.events if event.requires_evidence is not None)
        self.assertEqual(barrier.progress, terminal.phase_count)

    def test_release_then_regrasp_creates_distinct_conditional_spans_from_one_observation(self):
        grasp, place, regrasp, carry = grasp_input(), manipulation_input(), grasp_input(), manipulation_input()
        regrasp["skill_id"], carry["skill_id"] = "regrasp", "carry"
        carry["program"]["demo"]["release_frame"] = None
        carry["program"]["demo"]["gripper_events"] = [
            {"event_id": "hold", "frame_index": 100, "command": "hold"}]
        for future in (place, carry):
            future["scene"]["holding"]["verified"] = False
            future["scene"]["holding"]["source"] = "no future Holding observation exists"
        request = compile_inputs(grasp, place, regrasp, carry)
        self.assertEqual(request.phase_count, 9)
        first_held = [phase for phase in request.phases if phase.metadata.get("skill_id") == "transfer"
                      and phase.metadata.get("mode") == "held"]
        later_held = [phase for phase in request.phases if phase.metadata.get("skill_id") == "carry"]
        self.assertEqual(len(first_held), 2)
        self.assertEqual(len(later_held), 2)
        for phase in first_held:
            self.assertEqual(phase.metadata["attachment_interval"], [2., 4.])
            self.assertEqual(phase.metadata["attachment_condition_id"], "grasp:stable_holding")
            self.assertEqual(phase.metadata["release_condition_id"], "transfer:actual_release")
        for phase in later_held:
            self.assertEqual(phase.metadata["attachment_interval"], [7., 9.])
            self.assertEqual(phase.metadata["attachment_condition_id"], "regrasp:stable_holding")
            self.assertIsNone(phase.metadata["release_condition_id"])
        for phase in request.phases:
            np.testing.assert_array_equal(phase.metadata["owner_initial_pose"], grasp["scene"]["carrier_pose"])
        self.assertFalse(any(item["kind"] == "observed_attachment" for item in request.conditions))
        candidates = [item for item in request.conditions if item["kind"] == "candidate_attachment"]
        self.assertEqual([item["condition_id"] for item in candidates],
                         ["grasp:stable_holding", "regrasp:stable_holding"])
        self.assertTrue(all(item["requires_evidence"] for item in candidates))
        self.assertFalse(any("verified" in item for item in request.conditions))
        self.assertFalse(place["scene"]["holding"]["verified"])
        self.assertFalse(carry["scene"]["holding"]["verified"])
    # END attachment provenance and exact collision-span contract regressions.


if __name__ == "__main__":
    unittest.main()
# END independent process-constraint and existing-grounding integration tests.
