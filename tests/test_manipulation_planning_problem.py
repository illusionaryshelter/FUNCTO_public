"""CPU contracts for the shared, differentiable interaction planning problem.

Analytic poses and actuator states test requirements, frame conventions and
observation barriers. They are not robot IK, contact-physics or GPU benchmarks.
The fixtures deliberately keep source evidence separate from future attachment
assumptions and do not obtain geometry from a simulator or another project.
"""

# BEGIN independent process-constraint and existing-grounding integration tests.
from copy import deepcopy
from dataclasses import replace
import json
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from manipulation.goals import SupportSurface
from manipulation.motion import MotionModel, transfer_motion
from manipulation.planning_problem import (
    CoordinateEventRequirement,
    MechanismRequirement,
    PoseEventRequirement,
    PoseProcessRequirement,
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
    return {"skill_id": "grasp", "program": {"skill_kind": "grasp", "demo": demo}, "scene": scene}


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
    return {"skill_id": "transfer", "program": program, "scene": scene}


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


# BEGIN current-skill reference/goal/process separation regression coverage.
class PlanningProcessTest(unittest.TestCase):
    def test_ordinary_reference_never_creates_a_path_or_terminal_obligation(self):
        request = problem([MotionPhase("reference", path([pose(), pose((.2, 0., 0.)), pose()]))])
        progress = torch.tensor([[0., .3, .5, 1.]], dtype=torch.float64)
        position, quaternion = components(np.array([[pose((0., .6, .3))]*4]))
        np.testing.assert_array_equal(request.evaluate(position, quaternion, progress).numpy(), 0.)
        np.testing.assert_array_equal(request.optimizer_preferences(position, quaternion, progress).numpy(), 0.)
        np.testing.assert_array_equal(request.anchors, [0., 1.])
        self.assertIsNone(request.terminal_pose_target)
        self.assertGreater(request.measurements(position, quaternion, progress)[
            "phase:0:original_reference_position_error_m"].min().item(), .5)
        with self.assertRaises(TypeError):
            MotionPhase("old implicit lock", path([pose(), pose()]), kind="pose")

    def test_required_pose_relation_has_scope_and_finite_gradients(self):
        reference = path([pose(), pose((.1, 0., 0.)), pose()])
        relation = PoseProcessRequirement(.2, .8, reference, np.zeros((6, 2)), "required tool axis/position relation")
        request = problem([MotionPhase("source", reference)], pose_relations=(relation,))
        progress = torch.tensor([[0., .2, .5, .8, 1.]], dtype=torch.float64)
        position = torch.zeros((1, 5, 3), dtype=torch.float64, requires_grad=True)
        quaternion = torch.tensor([[[1., 0., 0., 0.]]*5], dtype=torch.float64, requires_grad=True)
        values = request.evaluate(position, quaternion, progress)
        self.assertGreater(values[0, 2].item(), .09)
        self.assertEqual(values[0, 0].item(), 0.)
        loss = values.sum()+request.optimizer_preferences(position, quaternion, progress).sum()
        loss.backward()
        self.assertLess(position.grad[0, 2, 0].item(), 0.)
        self.assertTrue(torch.isfinite(quaternion.grad).all())
        np.testing.assert_allclose(request.anchors, [0., .2, .5, .8, 1.])

    def test_allowed_pose_band_has_no_invented_center_preference(self):
        bounds = np.tile([-np.inf, np.inf], (6, 1))
        bounds[0] = [-.03, .04]
        reference = path([pose(angles=(0., 0., np.pi/2))]*2)
        relation = PoseProcessRequirement(0., 1., reference, bounds, "explicit process allowance")
        request = problem([MotionPhase("source", reference)], pose_relations=(relation,))
        progress = torch.tensor([[.5]], dtype=torch.float64)
        position, quaternion = components(np.array([[pose((0., .035, .2), (0., 0., np.pi/2))]]))
        self.assertEqual(request.evaluate(position, quaternion, progress).item(), 0.)
        self.assertEqual(request.optimizer_preferences(position, quaternion, progress).item(), 0.)
        self.assertEqual(request.project_residual(position, quaternion, progress).shape[-1], 0)
        position[0, 0, 1] = .06
        self.assertGreater(request.evaluate(position, quaternion, progress).item(), .019)
        for malformed in (np.full((6, 2), np.inf), np.zeros((5, 2))):
            with self.assertRaises(ValueError):
                PoseProcessRequirement(0., 1., reference, malformed, "declared relation")

    def test_real_pose_event_generates_gradient_and_uses_its_own_tolerance(self):
        goal = PoseEventRequirement(.6, pose(), pose((.1, 0., 0.)), 1e-7, 1e-7, "true functional target")
        request = problem([MotionPhase("unrelated source", path([pose(), pose()]))], pose_events=(goal,))
        progress = torch.tensor([[.3, .6, 1.]], dtype=torch.float64)
        position = torch.tensor([[[.4, 0., 0.], [.100002, 0., 0.], [.4, 0., 0.]]], dtype=torch.float64, requires_grad=True)
        quaternion = torch.tensor([[[1., 0., 0., 0.]]*3], dtype=torch.float64)
        residual = request.evaluate(position, quaternion, progress)
        self.assertGreater(residual[0, 1].item(), 1e-6)
        self.assertEqual(residual[0, 0].item()+residual[0, 2].item(), 0.)
        request.optimizer_preferences(position, quaternion, progress).sum().backward()
        self.assertGreater(position.grad[0, 1, 0].item(), 0.)
        self.assertEqual(position.grad[0, 0].abs().sum().item(), 0.)
        self.assertIsNone(request.terminal_pose_target)

    def test_mechanism_is_independent_of_reference_and_bounds_the_actual_coordinate(self):
        request = compile_inputs(prismatic_input())
        progress = torch.tensor([[0., .2, .4, 1.2, 2.]], dtype=torch.float64)
        position, quaternion = request.sample_reference(progress)
        position = position.clone()
        position[0, 1, 0] = .24  # valid motion different from the source profile
        self.assertEqual(request.evaluate(position, quaternion, progress)[0, 1].item(), 0.)
        position[0, 1, 0] = .34  # beyond the bound current slider limit
        self.assertGreater(request.residuals(position, quaternion, progress)["mechanism_limits:0"][0, 1].item(), .039)
        position[0, 2, 1] += .03
        self.assertGreater(request.residuals(position, quaternion, progress)["mechanism:0"][0, 2].item(), .02)
        position[0, 3, 1] += 5.  # independent retreat is outside the holding scope
        self.assertEqual(request.residuals(position, quaternion, progress)["mechanism:0"][0, 3].item(), 0.)

    def test_full_candidate_winding_is_checked_without_locking_reference_velocity(self):
        request = compile_inputs(revolute_input())
        progress = torch.as_tensor(np.unique(np.r_[np.linspace(0., 2., 161), request.anchors])[None], dtype=torch.float64)
        position, quaternion = request.sample_reference(progress)
        self.assertLess(request.evaluate(position, quaternion, progress).max().item(), 1e-10)
        stationary_p = position[..., :1, :].expand_as(position).clone()
        stationary_q = quaternion[..., :1, :].expand_as(quaternion).clone()
        residual = request.residuals(stationary_p, stationary_q, progress)["function_coordinate:0"]
        self.assertGreater(residual.max().item(), 6.2)
        # The same turn with a different speed profile reaches the explicit
        # event and respects current limits; the reference samples are free.
        event = request.coordinate_events[0]
        coordinate = 2*np.pi*np.minimum(progress.numpy()[0]/event.progress, 1.)**2
        body = event.mechanism.model.poses(coordinate[:, None])
        actual = body @ np.linalg.inv(event.mechanism.tool_from_carrier)
        p, q = components(actual[None])
        self.assertLess(request.evaluate(p, q, progress).max().item(), 1e-10)
        self.assertEqual(request.verification_issues(progress), [])
        self.assertTrue(request.verification_issues(torch.tensor([[event.progress]])))

    def test_unwrapped_event_sequence_rejects_same_pose_with_wrong_turn_history(self):
        model = MotionModel("revolute", np.zeros((1, 1)), pose((.2, 0., 0.)), np.eye(3), np.zeros(3),
                            1e-6, 1e-6, {"source": "current hinge"})
        mechanism = MechanismRequirement(0., 1., model, pose(), [[0., 7.]], [0.], "current mechanism")
        goals = (CoordinateEventRequirement(.5, mechanism, 2*np.pi, 1e-5, "first turn"),
                 CoordinateEventRequirement(1., mechanism, 0., 1e-5, "return through reverse turn"))
        request = problem([MotionPhase("uninformative reference", path([pose(), pose()]))], mechanisms=(mechanism,), coordinate_events=goals)
        progress = torch.as_tensor(np.linspace(0., 1., 101)[None], dtype=torch.float64)
        angles = 2*np.pi*(1-np.abs(2*progress.numpy()[0]-1))
        p, q = components(model.poses(angles[:, None])[None])
        self.assertLess(request.evaluate(p, q, progress).max().item(), 1e-10)
        order = torch.randperm(progress.shape[-1], generator=torch.Generator().manual_seed(7))
        self.assertLess(request.evaluate(p[:, order], q[:, order], progress[:, order]).max().item(), 1e-10)
        p, q = components(model.poses(np.zeros((101, 1)))[None])
        self.assertGreater(request.evaluate(p, q, progress).max().item(), 6.2)

    def test_ambiguous_axis_projection_has_finite_gradients_and_is_rejected(self):
        model = MotionModel("revolute", np.zeros((1, 1)), pose(), np.eye(3), np.array([.2, 0., 0.]),
                            1e-6, 1e-6, {"source": "independent hinge"})
        relation = MechanismRequirement(0., 1., model, pose(), None, [0.], "axis evidence")
        request = problem([MotionPhase("source", path([pose(), pose()]))], mechanisms=(relation,))
        p = torch.zeros((1, 1, 3), dtype=torch.float64, requires_grad=True)
        q = torch.tensor([[[0., 1., 0., 0.]]], dtype=torch.float64, requires_grad=True)
        value = request.evaluate(p, q, torch.tensor([[.5]], dtype=torch.float64))
        self.assertGreater(value.item(), 3.)
        value.sum().backward()
        self.assertTrue(torch.isfinite(p.grad).all() and torch.isfinite(q.grad).all())

    def test_actual_frame_calibration_and_joint_coupling_remain_independent(self):
        base, tool = pose((.4, -.1, .2), (.2, -.3, .4)), pose((.1, .02, -.03), (.1, .2, -.1))
        coupling = JointCoupling(("joint_b", "joint_a"), np.array([[1., 2.]]), np.array([.1]), "actual actuator relation")
        request = problem([MotionPhase("source", path([pose(), pose()]))], base_pose=base, tool_transform=tool, couplings=(coupling,))
        link = pose((.2, .1, .3), (.4, .2, .5))
        p, q = components(np.array([[link]]))
        tp, tq = request.transform_tool_pose(p, q)
        np.testing.assert_allclose(matrices(tp, tq)[0, 0], base @ link @ tool, atol=1e-12)
        np.testing.assert_allclose(matrices(*request.reference_to_link_pose(tp, tq))[0, 0], link, atol=1e-12)
        actual = torch.tensor([[[.02, .06, 0.]]], dtype=torch.float64)
        value = request.evaluate(tp, tq, torch.tensor([[.5]]), actual, request.joint_names)
        self.assertEqual(value.item(), 0.)
        actual[..., 1] += .01
        self.assertGreater(request.evaluate(tp, tq, torch.tensor([[.5]]), actual, request.joint_names).item(), .009)

    def test_mechanism_metrics_match_source_projection_with_attachment_and_frame_changes(self):
        coordinates = {
            "prismatic": np.array([[0.], [.05], [.09]]),
            "revolute": np.array([[0.], [.3], [.6]]),
            "planar": np.array([[0., 0., 0.], [.03, -.02, .3], [.06, -.01, .6]]),
            "translation_plane": np.array([[0., 0.], [.03, -.02], [.06, -.01]]),
        }
        attachment = pose((.17, -.12, .09), (.4, .2, -.3))
        basis = Rotation.from_euler("xyz", (.3, .4, -.2)).as_matrix()
        world = pose((.6, -.4, .8), (.3, -.2, .5))
        carrier = pose((.03, .02, -.04), (-.1, .15, .2))
        progress = torch.tensor([[0., .5, 1.]], dtype=torch.float64)
        for kind, values in coordinates.items():
            source = MotionModel(kind, values, pose((.23, -.1, .08), (.2, -.15, .4)), basis,
                                 np.array([.05, .04, -.03]) if kind == "revolute" else None,
                                 1e-6, 1e-6, {"source": "independent physical mechanism"})
            original = source.poses(values)
            perturbed = original.copy()
            transverse = basis[:, 2] if kind in {"planar", "translation_plane"} else basis[:, 0]
            perturbed[1, :3, 3] += 5e-5*transverse
            perturbed[2, :3, :3] = Rotation.from_rotvec(2e-5*basis[:, 0]).as_matrix() @ perturbed[2, :3, :3]
            for world_frame, carrier_frame in ((pose(), pose()), (world, carrier)):
                with self.subTest(kind=kind, changed_frames=not np.array_equal(world_frame, pose())):
                    reference_body = world_frame @ original @ carrier_frame
                    actual_body = world_frame @ perturbed @ carrier_frame
                    reference_pose = world_frame @ source.reference_pose @ carrier_frame
                    current_basis = world_frame[:3, :3] @ basis
                    center = (None if source.axis_point is None else
                              world_frame[:3, :3] @ source.axis_point+world_frame[:3, 3])
                    model = MotionModel(kind, values.copy(), reference_pose, current_basis, center,
                                        1e-6, 1e-6, source.diagnostics)
                    # A translated planar body origin changes its x/y coordinate
                    # gauge while representing the same physical set of poses.
                    current_coordinates = model.coordinates_of(reference_body)
                    model.coordinates = current_coordinates
                    current_attachment = np.linalg.inv(carrier_frame) @ attachment
                    reference = PosePath(np.array([0., .5, 1.]), reference_body, "complete source path",
                                         motion_model=model, coordinates=current_coordinates,
                                         attachment=current_attachment)
                    requirement = MechanismRequirement(0., 1., model, np.linalg.inv(current_attachment), None,
                                                       current_coordinates[0], "current physical model")
                    request = problem([MotionPhase("mechanism", reference)], mechanisms=(requirement,))
                    on_position, on_quaternion = components((reference_body @ current_attachment)[None])
                    on_position.requires_grad_(), on_quaternion.requires_grad_()
                    accepted = request.evaluate(on_position, on_quaternion, progress)
                    np.testing.assert_allclose(accepted.detach().numpy(), 0., atol=1e-12)
                    accepted.sum().backward()
                    self.assertTrue(torch.isfinite(on_position.grad).all())
                    self.assertTrue(torch.isfinite(on_quaternion.grad).all())
                    position, quaternion = components((actual_body @ current_attachment)[None])
                    position.requires_grad_(), quaternion.requires_grad_()
                    measured = request.measurements(position, quaternion, progress)
                    projected = model.poses(model._coordinates_unchecked(actual_body))
                    oracle_position = np.linalg.norm(actual_body[:, :3, 3]-projected[:, :3, 3], axis=1)
                    oracle_rotation = Rotation.from_matrix(
                        actual_body[:, :3, :3] @ projected[:, :3, :3].transpose(0, 2, 1)).magnitude()
                    np.testing.assert_allclose(measured["mechanism:0:position_error_m"].detach().numpy()[0],
                                               oracle_position, atol=2e-15)
                    np.testing.assert_allclose(measured["mechanism:0:rotation_error_rad"].detach().numpy()[0],
                                               oracle_rotation, atol=2e-15)
                    residuals = request.residuals(position, quaternion, progress)
                    self.assertGreater(residuals["mechanism:0"][0, 1].item(), 4e-5)
                    self.assertGreater(residuals["mechanism:0"][0, 2].item(), 1e-5)
                    with self.assertRaisesRegex(ValueError, "incompatible"):
                        model.coordinates_of(actual_body)
                    sum(residuals.values()).sum().backward()
                    self.assertTrue(torch.isfinite(position.grad).all())
                    self.assertTrue(torch.isfinite(quaternion.grad).all())


class PlanningBuilderTest(unittest.TestCase):
    def test_no_pregrasp_or_automatic_stop_and_source_map_keeps_irregular_seconds(self):
        skill = grasp_input()
        skill["program"]["demo"]["timestamps"] = np.array([2., 2.02, 2.03, 2.5, 2.7, 3.])
        request = compile_inputs(skill)
        self.assertEqual(len(request.phases), 1)
        self.assertEqual(len(request.pose_events), 1)
        self.assertEqual(request.stop_progress.tolist(), [])
        self.assertFalse(any(event.requires_evidence for event in request.events))
        # BEGIN independent effect assessment keeps its exact event key.
        effect = next(row for row in request.conditions if row.get("effect_requirement"))
        self.assertEqual((effect["skill_id"], effect["event_id"]),
                         (skill["skill_id"], "__stable_holding__"))
        # END effect timing alone supplies no outcome evidence.
        self.assertEqual(request.pose_relations, ())
        np.testing.assert_allclose(request.source_time_map["progress"], [0., .02, .03, .5, .7, 1.])
        np.testing.assert_allclose(request.source_time_to_progress([2.025, 2.6]), [.025, .6])
        np.testing.assert_allclose(request.progress_to_source_time([.025, .6]), [2.025, 2.6])
        np.testing.assert_allclose(request.anchors, [0., .5, 1.])
        self.assertIs(request.terminal_pose_target, request.pose_events[0])
        p, q = request.sample_reference(torch.tensor([[.02, .03]], dtype=torch.float64))
        p = p+3.
        self.assertEqual(request.evaluate(p, q, torch.tensor([[.02, .03]])).sum().item(), 0.)

    def test_only_one_current_skill_and_actual_holding_are_accepted(self):
        with self.assertRaisesRegex(ValueError, "exactly one"):
            compile_inputs(grasp_input(), manipulation_input())
        skill = manipulation_input()
        skill["scene"]["holding"]["verified"] = False
        with self.assertRaisesRegex(ValueError, "actual holding"):
            compile_inputs(skill)
        a = MotionPhase("a", path([pose(), pose()]), metadata={"skill_id": "a"})
        b = MotionPhase("b", path([pose(), pose()]), metadata={"skill_id": "b"})
        with self.assertRaisesRegex(ValueError, "one current"):
            problem([a, b])

    def test_function_release_and_retreat_preserve_identity_without_stage_targets(self):
        request = compile_inputs(manipulation_input())
        self.assertEqual(len(request.phases), 2)
        by_id = {event.event_id: event for event in request.events}
        self.assertAlmostEqual(by_id["function"].progress, .6)
        self.assertAlmostEqual(by_id["open"].progress, .8)
        self.assertEqual(by_id["__release_verified__"].progress, 1.)
        # BEGIN release observations bind to identity rather than a clock guess.
        effect = next(row for row in request.conditions if row.get("effect_requirement"))
        self.assertEqual((effect["skill_id"], effect["event_id"]),
                         (by_id["__release_verified__"].skill_id, "__release_verified__"))
        # END the predicted geometry is still distinct from the observed effect.
        self.assertIsNone(request.terminal_pose_target)
        self.assertEqual(request.stop_progress.tolist(), [])
        self.assertEqual(request.phases[0].metadata["attachment_interval"], [0., 1.])
        self.assertEqual(request.phases[1].metadata["mode"], "released_retreat")
        p, q = request.sample_reference(torch.tensor([[2.]], dtype=torch.float64))
        self.assertEqual(request.evaluate(p+7., q, torch.tensor([[2.]])).item(), 0.)

    def test_reference_kernel_failure_keeps_real_goal_and_malformed_goal_still_fails(self):
        skill = manipulation_input()
        skill["program"]["task"]["relation"] = "tool_use"
        skill["scene"]["functional_bindings"]["function"]["reference_frame"] = pose((.4, 0., 0.))
        grounded = ground_program(skill["program"], skill["scene"])
        self.assertEqual(grounded["motion_checks"]["reference_generation"]["status"], "failed")
        request = compile_inputs(skill)
        self.assertEqual(len(request.pose_events), 1)
        goal = request.pose_events[0]
        p, q = request.sample_reference(torch.tensor([[goal.progress]], dtype=torch.float64))
        p = p.clone()
        p[..., 0] -= .2
        self.assertGreater(request.evaluate(p, q, torch.tensor([[goal.progress]])).item(), .19)
        bad = deepcopy(skill)
        del bad["scene"]["functional_bindings"]["function"]["source"]
        with self.assertRaisesRegex(ValueError, "evidenced"):
            compile_inputs(bad)

    def test_current_coordinate_goal_conflict_is_not_a_reference_failure(self):
        skill = prismatic_input()
        skill["scene"]["functional_bindings"]["function"]["current_coordinates"] = [.4]
        with self.assertRaisesRegex(ValueError, "required start or functional goal"):
            compile_inputs(skill)

    def test_actual_entry_tool_goal_does_not_require_a_multinode_reference_fit(self):
        skill = manipulation_input()
        skill["program"]["task"]["relation"] = "tool_use"
        demo = skill["program"]["demo"]
        demo["functional_events"][0]["frame_index"] = 100
        demo["release_frame"] = 100
        demo["gripper_events"][0]["frame_index"] = 100
        with patch("manipulation.function_motion.transfer_function_motion") as fit:
            request = compile_inputs(skill)
        fit.assert_not_called()
        self.assertEqual(request.pose_events[0].progress, 0.)
        self.assertEqual(request.phase_count, 1)
        self.assertEqual(request.phases[0].metadata["mode"], "released_retreat")

    def test_support_is_a_finite_allowed_set_with_free_tangent_and_yaw(self):
        skill = on_input()
        request = compile_inputs(skill)
        actual, _ = alternative_placement(skill)
        progress = torch.as_tensor(request.source_time_map["progress"][None], dtype=torch.float64)
        p, q = components(actual[None])
        self.assertLess(request.evaluate(p, q, progress).max().item(), 1e-10)
        self.assertEqual(request.pose_events, ())
        self.assertEqual(request.pose_relations, ())
        event = request.supports[0]
        bad = pose((.7, .08, 0.)) @ skill["scene"]["holding"]["T_carrier_eef"]
        p, q = components(np.array([[bad]]))
        self.assertGreater(request.evaluate(p, q, torch.tensor([[event.progress]])).item(), .3)
        lifted = pose((.02, .03, 2e-5)) @ skill["scene"]["holding"]["T_carrier_eef"]
        p, q = components(np.array([[lifted]]))
        self.assertGreater(event.evaluate(p, q, request.numerical).item(), 1e-5)

    def test_explicit_evidence_prerequisite_does_not_imply_a_stop(self):
        skill = grasp_input()
        skill["execution_events"] = [{"event_id": "check", "frame_index": 23,
            "kind": "prerequisite", "source": "explicit operator dependency", "requires_evidence": "tool_ready"}]
        request = compile_inputs(skill)
        self.assertEqual(request.stop_progress.tolist(), [])
        self.assertEqual(next(e for e in request.events if e.event_id == "check").requires_evidence, "tool_ready")
        skill["execution_events"][0]["stop"] = True
        request = compile_inputs(skill)
        self.assertEqual(len(request.stop_progress), 1)

    def test_json_bindings_are_equivalent_and_unmodified(self):
        for skill in (grasp_input(), manipulation_input(), on_input(), prismatic_input(), revolute_input()):
            encoded = json.dumps(skill, default=lambda value: value.tolist(), sort_keys=True)
            serialized = json.loads(encoded)
            expected, actual = compile_inputs(skill), compile_inputs(serialized)
            np.testing.assert_allclose(expected.source_time_map["progress"], actual.source_time_map["progress"])
            progress = torch.as_tensor(expected.anchors[None], dtype=torch.float64)
            np.testing.assert_allclose(matrices(*expected.sample_reference(progress)), matrices(*actual.sample_reference(progress)))
            self.assertEqual(json.dumps(serialized, sort_keys=True), encoded)

    def test_remaining_obligations_preserve_clock_and_explicit_actual_input_gaps(self):
        request = compile_inputs(revolute_input())
        remaining = request.remaining_obligations(.7, (("turn", "open"),))
        self.assertEqual(remaining["process"]["mechanisms"][0]["remaining_scope"], (.7, 1.))
        self.assertEqual(remaining["process"]["mechanisms"][0]["requirement"].start, 0.)
        self.assertNotIn("open", [event.event_id for event in remaining["events"]])
        self.assertTrue(any("lifted coordinate" in item for item in remaining["required_actual_inputs"]))
        np.testing.assert_array_equal(remaining["source_time_map"]["source_times"], request.source_timestamps)
        self.assertEqual(request.start_progress, 0.)
        self.assertEqual(len(request.events), 3)

    def test_suffix_keeps_original_progress_and_requires_rebound_coordinate_origin(self):
        model = MotionModel("revolute", np.zeros((1, 1)), pose((.2, 0., 0.)), np.eye(3), np.zeros(3),
                            1e-6, 1e-6, {"source": "current hinge"})
        original = MechanismRequirement(0., 1., model, pose(), [[0., 7.]], [0.], "original current mechanism")
        goal = CoordinateEventRequirement(1., original, 2*np.pi, 1e-5, "remaining full-turn goal")
        rebound = replace(original, start=.5, initial_coordinates=np.array([3.5]), source="observed accumulated current angle")
        options = {"start_progress": .5, "source_timestamps": np.array([5., 6.]),
                   "source_progress": np.array([.5, 1.]), "mechanisms": (rebound,)}
        phases = [MotionPhase("original source", path([pose(), pose()]))]
        with self.assertRaisesRegex(ValueError, "bound mechanism instance"):
            problem(phases, coordinate_events=(goal,), **options)
        request = problem(phases, coordinate_events=(replace(goal, mechanism=rebound),), **options)
        progress = torch.as_tensor(np.linspace(.5, 1., 33)[None], dtype=torch.float64)
        p, q = components(model.poses(np.linspace(3.5, 2*np.pi, 33)[:, None])[None])
        self.assertLess(request.evaluate(p, q, progress).max().item(), 1e-10)
        self.assertEqual(request.verification_issues(progress), [])
        np.testing.assert_array_equal(request.anchors, [.5, 1.])
        np.testing.assert_allclose(request.source_time_to_progress([5., 5.5, 6.]), [.5, .75, 1.])
# END current-skill reference/goal/process separation regression coverage.

if __name__ == "__main__":
    unittest.main()
