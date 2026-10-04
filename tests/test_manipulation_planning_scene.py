"""Small CPU geometry/gradient counterexamples for the planning-scene contract.

Analytic queries below stand in for a library kernel only in these tests. They
do not add a second production collision backend or certify any real scene.
"""

# BEGIN CPU scene ownership, contact, release and coverage regression tests.
from dataclasses import replace
from types import SimpleNamespace
import unittest

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from manipulation.planning_problem import (
    JointCoupling, MotionPhase, PlanningEvent, PlanningProblem, PosePath,
)
from manipulation.planning_scene import (
    Attachment, CollisionOwner, ContactRule, CuroboSphereQuery, ObservedSpace,
    PlanningScene, PrismaticGripper, _self_pair_sum,
    _coverage_halfspace_max, _coverage_plane_chunk_size, _COVERAGE_POINT_PLANE_ELEMENTS,
)


# BEGIN multiple-collision and cross-chunk gradient regression.
# BEGIN preserved halfspace values/gradients and bounded backward storage.
class CoverageGuidanceScratchTest(unittest.TestCase):
    def test_chunk_rule_uses_all_planes_under_one_fixed_scratch_target(self):
        for states in (2, 8, 170, 680, 4096, 65536):
            for points in (20, 32):
                chunk = _coverage_plane_chunk_size(states, points, 3174)
                self.assertGreaterEqual(chunk, 1)
                self.assertLessEqual(chunk, 3174)
                self.assertLessEqual(states*points*chunk, max(_COVERAGE_POINT_PLANE_ELEMENTS, states*points))
        self.assertEqual(_coverage_plane_chunk_size(8, 32, 3174), 3174)
        self.assertEqual(_coverage_plane_chunk_size(680, 32, 3174), 48)

    def test_values_unique_gradients_and_saved_tensor_shapes_match_halfspace_math(self):
        rng = np.random.default_rng(123)
        points = torch.tensor(rng.normal(size=(1, 8, 32, 3)), dtype=torch.float64, requires_grad=True)
        planes = torch.tensor(rng.normal(size=(3174, 4)), dtype=torch.float64)
        saved = []

        def retain(value):
            saved.append((tuple(value.shape), value.dtype))
            return value

        with torch.autograd.graph.saved_tensors_hooks(retain, lambda value: value):
            optimized = _coverage_halfspace_max(points, planes)
        self.assertNotIn(((1, 8, 32, 3174), torch.float64), saved)
        self.assertIn(((1, 8, 32), torch.int64), saved)
        signed = (points@planes[:, :3].T+planes[:, 3])/torch.linalg.vector_norm(planes[:, :3], dim=-1)
        reference = signed.amax(-1)
        self.assertTrue(bool((signed.topk(2, dim=-1).values.diff(dim=-1).abs() > 1e-10).all()))
        torch.testing.assert_close(optimized, reference, rtol=0, atol=0)
        first = torch.autograd.grad(optimized.sum(), points, retain_graph=True)[0]
        second = torch.autograd.grad(reference.sum(), points)[0]
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        old_chunks = torch.full_like(optimized, -float("inf"))
        for begin in range(0, len(planes), 32):
            old_chunks = torch.maximum(old_chunks, _coverage_halfspace_max(points, planes[begin:begin+32]))
        torch.testing.assert_close(old_chunks, optimized, rtol=1e-13, atol=1e-13)
        chunk_gradient = torch.autograd.grad(old_chunks.sum(), points)[0]
        torch.testing.assert_close(chunk_gradient, first, rtol=1e-13, atol=1e-13)

    def test_equal_planes_choose_a_valid_face_subgradient(self):
        point = torch.zeros((1, 1, 1, 3), dtype=torch.float64, requires_grad=True)
        planes = point.new_tensor([[1., 0., 0., 0.], [0., 1., 0., 0.]])
        value = _coverage_halfspace_max(point, planes)
        self.assertEqual(value.item(), 0.)
        gradient = torch.autograd.grad(value.sum(), point)[0]
        torch.testing.assert_close(gradient, point.new_tensor([[[[1., 0., 0.]]]]), rtol=0, atol=0)
# END only a tie's valid subgradient selection changes, never its constraint value.


class BatchedHandCollisionTest(unittest.TestCase):
    def test_all_pair_penetrations_and_gradients_survive_chunk_boundaries(self):
        generator = torch.Generator().manual_seed(29)
        values = torch.randn((2, 3, 34, 4), generator=generator, dtype=torch.float64)
        values[..., 3] = .9
        # Coincident centres test Torch's original zero-distance derivative;
        # a reversed declared pair remains a separate term in the old sum.
        values[..., 1, :3] = values[..., 0, :3]
        pairs = np.asarray([(a, b) for a in range(34) for b in range(a+1, 34)] + [(1, 0)])
        actual_input = values.clone().requires_grad_()
        expected_input = values.clone().requires_grad_()
        actual = _self_pair_sum(actual_input, pairs)
        expected = torch.stack([
            torch.relu(expected_input[..., a, 3]+expected_input[..., b, 3]
                       - torch.linalg.vector_norm(expected_input[..., a, :3]
                                                  - expected_input[..., b, :3], dim=-1))
            for a, b in pairs]).sum(0)
        weights = torch.randn((2, 3), generator=generator, dtype=torch.float64)
        actual_grad = torch.autograd.grad((actual*weights).sum(), actual_input)[0]
        expected_grad = torch.autograd.grad((expected*weights).sum(), expected_input)[0]
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(actual_grad, expected_grad, atol=1e-12, rtol=1e-12)
        with torch.no_grad():
            torch.testing.assert_close(_self_pair_sum(values, pairs), expected)

    def test_distinct_pair_sum_is_not_replaced_by_a_maximum(self):
        spheres = torch.tensor([[[[0., 0., 0., .2], [.1, 0., 0., .2],
                                   [.2, 0., 0., .2]]]], dtype=torch.float64)
        value = _self_pair_sum(spheres, np.asarray([[0, 1], [1, 2]]))
        self.assertAlmostEqual(value.item(), .6)
        self.assertEqual(_self_pair_sum(spheres, np.empty((0, 2), dtype=int)).item(), 0.)
# END multiple-collision and cross-chunk gradient regression.


class SphereQuery:
    def __init__(self, owner_ids=(), center=(8., 0., 0.), radius=.1):
        self.owner_ids = frozenset(owner_ids)
        # BEGIN formal source bindings need explicit test geometry provenance.
        self.source = "explicit analytic test sphere field"
        # END this fixture remains a test-only query implementation.
        self.center, self.radius = center, radius
        self.calls = {}

    def clone_for_rollout(self):
        return SphereQuery(self.owner_ids, self.center, self.radius)

    def __call__(self, spheres, *, key):
        self.calls[key] = spheres
        center = spheres.new_tensor(self.center)
        distance = torch.linalg.vector_norm(spheres[..., :3] - center, dim=-1)
        return torch.relu(spheres[..., 3] + self.radius - distance)


class BoxQuery(SphereQuery):
    def __init__(self, owner):
        super().__init__((owner,))
        # BEGIN the contact query consumes a complete independent solid model.
        self.independent_geometry = {"solids": [{"name": "test_box", "kind": "cuboid",
            "dimensions": [.8, .8, .2], "pose": [0., 0., -.1, 1., 0., 0., 0.],
            "source": "explicit analytic test box"}], "query_gaps": []}
        # END no sphere-distance witness invents finite target occupancy.

    def clone_for_rollout(self):
        return BoxQuery(next(iter(self.owner_ids)))

    def __call__(self, spheres, *, key):
        self.calls[key] = spheres
        relative = (spheres[..., :3] - spheres.new_tensor([0., 0., -.1])).abs() - spheres.new_tensor([.4, .4, .1])
        sdf = torch.linalg.vector_norm(torch.relu(relative), dim=-1) + relative.amax(-1).clamp_max(0)
        return torch.relu(spheres[..., 3] - sdf)


def pose_path(position=(0., 0., 0.), end=None, turn=0.):
    poses = np.repeat(np.eye(4)[None], 2, axis=0)
    poses[:, :3, 3] = position
    if end is not None:
        poses[1, :3, 3] = end
    poses[1, :3, :3] = Rotation.from_rotvec([0., 0., turn]).as_matrix()
    return PosePath(np.array([0., 1.]), poses, source={"kind": "synthetic_test"})


def coverage(mask=None, **kwargs):
    return ObservedSpace(np.ones((41, 41, 41), bool) if mask is None else mask,
                         origin=[-2., -2., -2.], voxel_size=.1,
                         observation_ids=["synthetic_current_scene"], reference_frame="world", **kwargs)


def scene(owners=(), materials=(None,), **kwargs):
    options = dict(background=SphereQuery(("background",)), owners=owners,
                   coverage=coverage(), reference_frame="world",
                   robot_sphere_materials=materials, robot_geometry_source="synthetic_robot",
                   has_gripper=False)
    options.update(kwargs)
    return PlanningScene(**options)


def inputs(progress=(.5,), positions=None, sphere_rows=None, dtype=torch.float64):
    s = torch.tensor([progress], dtype=dtype)
    p = torch.zeros((*s.shape, 3), dtype=dtype) if positions is None else torch.tensor([positions], dtype=dtype)
    q = torch.tensor([1., 0., 0., 0.], dtype=dtype).expand(*s.shape, 4).clone()
    rows = [[[-1., 0., 0., .05]]] * s.shape[1] if sphere_rows is None else sphere_rows
    balls = torch.tensor([rows], dtype=dtype)
    return balls, p, q, s


# BEGIN installation exclusions are local native robot-pair policy, not scene clearing.
class FixedInstallationPolicyTest(unittest.TestCase):
    def mount(self, **changes):
        options = dict(owner_id="mount", query=SphereQuery(("mount",), center=(0., 0., 0.), radius=.5),
                       pose_path=pose_path(), source="native fixed installation", geometry_complete=True,
                       robot_excluded_spheres=(0,), robot_exclusion_source="native welded body policy",
                       robot_exclusion_geometry_source="synthetic_robot")
        options.update(changes)
        return CollisionOwner(**options)

    def test_exclusion_preserves_other_robot_links_payload_background_and_unknown(self):
        mount = self.mount()
        payload = CollisionOwner("payload", SphereQuery(("payload",)), pose_path(), "current payload",
                                 geometry_complete=True, moving=True, local_spheres=[[0., 0., 0., .1]])
        current = scene((mount, payload), materials=(None, None),
                        background=SphereQuery(("background",), center=(0., 0., 0.), radius=.25))
        arguments = inputs(sphere_rows=[[[0., 0., 0., .1], [0., 0., 0., .1]]])
        terms = current.components(*arguments)
        self.assertAlmostEqual(terms["collision/robot/mount"].item(), .6)
        self.assertAlmostEqual(terms["collision/payload/mount"].item(), .6)
        self.assertAlmostEqual(terms["collision/robot/background"].item(), .7)
        self.assertIn("unknown/space/robot", terms)
        self.assertIn("unknown/space/payload", terms)
        self.assertNotIn("unknown/space/mount", terms)
        torch.testing.assert_close(mount.query.calls[("robot", "mount")][..., 3], arguments[0][..., 3])

    def test_partial_moving_or_stale_installation_cannot_inherit_exclusion(self):
        for change in ({"geometry_complete": False}, {"moving": True},
                       {"pose_path": pose_path(end=(0., 0., .1))}, {"robot_excluded_spheres": (0, 0)}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.mount(**change)
        with self.assertRaisesRegex(ValueError, "complete robot sphere model"):
            scene((self.mount(robot_exclusion_geometry_source="another robot"),))
        with self.assertRaisesRegex(ValueError, "complete robot sphere model"):
            scene((self.mount(robot_excluded_spheres=(1,)),))
# END mounting geometry remains an obstacle rather than becoming observed free volume.


# BEGIN source installation pairs replace only their duplicate robot proxy gate.
def source_installation_scene(tmp_path):
    """One movable box and one fixed box with intentionally larger sphere covers."""
    from manipulation.robot_geometry_check import NativePrimitive, PairQuery, RobotGeometryChecker
    from manipulation.robot_validation import RobotStateValidator
    path = tmp_path / "slide.urdf"
    path.write_text('''<robot name="source_boxes"><link name="base"/><link name="slider"/>
      <joint name="slide" type="prismatic"><parent link="base"/><child link="slider"/>
      <axis xyz="1 0 0"/><limit lower="-1" upper="1" effort="10" velocity="1"/></joint></robot>''')
    checker = RobotGeometryChecker(path, base_frame="base", geometries=[
        NativePrimitive("mount", "base", "box", (.04,)*3, np.eye(4), "native fixed installation"),
        NativePrimitive("moving", "slider", "box", (.04,)*3, np.eye(4), "native moving box")])
    validator = RobotStateValidator(checker, joint_names=("slide",), arm_joint_names=("slide",), finger_joint_names=(),
        pairs=(PairQuery("mount", "moving", "required", "explicit independent bodies"),),
        named_velocity_limits={"slide": 1.}, reference_frame="world", T_reference_base=np.eye(4),
        tool_frame="slider", T_frame_tcp=np.eye(4), fk_translation_tolerance_m=1e-8,
        fk_rotation_tolerance_rad=1e-8, source="explicit source boxes")
    query = SphereQuery(("mount",), center=(0., 0., 0.), radius=.1)
    query.independent_geometry = {"solids": [{"name": "mount_proxy", "kind": "cuboid",
        "dimensions": [.04]*3, "pose": [0., 0., 0., 1., 0., 0., 0.],
        "source": "native fixed installation"}], "query_gaps": []}
    mount = CollisionOwner("mount", query, pose_path(), "native fixed installation", True,
                           native_geometry_names=("mount",))
    payload = CollisionOwner("payload", SphereQuery(("payload",)), pose_path(), "current payload",
        geometry_complete=True, moving=True, local_spheres=[[0., 0., 0., .1]])
    current = scene((mount, payload))
    problem = SimpleNamespace(reference_frame="world", tool_link="slider", base_pose=np.eye(4),
        tool_transform=np.eye(4), start_q=np.array([.15]), joint_names=("slide",))
    current.bind_robot_validator(validator, problem)
    return current, validator, problem


def test_source_acceptance_survives_proxy_overlap_and_keeps_payload_collision(tmp_path):
    current, validator, _ = source_installation_scene(tmp_path)
    args = inputs((0.,), sphere_rows=[[[.15, 0., 0., .1]]], dtype=torch.float32)
    q = torch.tensor([[[.15]]], dtype=torch.float32, requires_grad=True)
    times = torch.zeros((1, 1))
    terms = current.components(*args, q, ("slide",), trajectory_time=times)
    assert "collision/robot/mount" not in terms
    assert terms["collision/payload/mount"].item() > 0
    args[0].requires_grad_()
    guidance = current.robot_self_collision_guidance(*args, q, ("slide",), trajectory_time=times)
    assert guidance.item() > 0
    assert torch.autograd.grad(guidance.sum(), args[0])[0][..., 0].item() == -1
    evidence = {}
    assert current.check_robot_self_collision(q, ("slide",), times, diagnostics=evidence).item() == 0
    assert evidence["checked_states"] == 1
    assert evidence["worst_state"]["planning_joint_positions"] == q.detach()[0, 0].tolist()
    assert evidence["worst_state"]["minimum_pair"]["relation"] == "separated"
    assert current.clone_for_rollout().robot_validator is validator
    binding = current.to_execution_binding()["robot_self_collision"]
    assert binding["pair_order"][0]["policy"] == "required"
    assert len(binding["model"]["geometries"]) == 2


def test_source_queries_reject_solid_collision_and_ignore_only_invalid_padding(tmp_path):
    current, _, _ = source_installation_scene(tmp_path)
    q = torch.tensor([[[.15], [.02], [float("nan")]]])
    times = torch.tensor([[0., .5, float("nan")]])
    valid = torch.tensor([[True, True, False]])
    report = {}
    result = current.check_robot_self_collision(q, ("slide",), times, valid=valid, diagnostics=report)
    assert result.tolist() == [[0., 1., 0.]]
    assert report["checked_states"] == 2
    assert report["worst_state"]["violations"][0]["relation"] == "penetrating"
    # Even a misleading zero sphere overlap never bypasses the source query.
    assert report["worst_state"]["planning_joint_positions"] == q[0, 1].tolist()


def test_source_installation_alias_rejects_different_pose_or_movable_solid(tmp_path):
    import pytest
    current, validator, problem = source_installation_scene(tmp_path)
    mount = current.owners[0]
    mount.pose_path = pose_path(position=(.1, 0., 0.))
    with pytest.raises(ValueError, match="source identity and fixed pose"):
        current.bind_robot_validator(validator, problem)
    mount.pose_path = pose_path()
    mount.native_geometry_names = ("moving",)
    with pytest.raises(ValueError, match="fixed source solid"):
        current.bind_robot_validator(validator, problem)
# END no owner name pattern or sphere overlap can infer a source-solid exemption.


def contact_rule(**changes):
    value = ContactRule(
        rule_id="pad_to_surface", actor="robot", target_owner="surface",
        interval=(0., 1.), sphere_indices=(0,), material_id="pad",
        feature_points=[[-.012, -.012, -.01], [.012, -.012, -.01], [.012, .012, -.01], [-.012, .012, -.01]],
        feature_normal=[0., 0., -1.], patch_origin=[0., 0., 0.], patch_axes=np.eye(3),
        patch_half_size=[.03, .03], gap_bounds=(-.002, .002), normal_tolerance_rad=.01,
        modeling_allowance_m=.002, geometry_error_m=.003, minimum_thickness_m=.02,
        source="synthetic_material_and_patch")
    return replace(value, **changes)


class PlanningSceneTest(unittest.TestCase):
    # BEGIN timed fingers and native-volume guidance have separate contracts.
    @staticmethod
    def timed_gripper():
        initial = np.array([.03, -.02])

        def factory(duration, mapping):
            def sample(times):
                t = np.asarray(times)[..., None]/duration
                return initial*(1-t), np.broadcast_to(-initial/duration, (*t.shape[:-1], 2))

            def sample_torch(times):
                q0 = times.new_tensor(initial)
                return q0*(1-times[..., None]/duration), (-q0/duration).expand(*times.shape, 2)

            return SimpleNamespace(joint_names=("left", "right"), duration=duration,
                                   sample=sample, sample_torch=sample_torch)

        return PrismaticGripper(controller="timed_reference", control_timestep_s=.002, joint_names=("left", "right"),
            current_positions=initial, joint_limits=[[0, .04], [-.04, 0]], joint_bounds=None,
            finger_trajectory_factory=factory, nominal_finger_trajectory_factory=factory, coupling_matrix=np.empty((0, 2)), coupling_offset=[],
            sphere_indices=(0, 1), local_spheres=[[0, 0, 0, .001], [0, 0, 0, .001]],
            sphere_joint_translation=[[[1, 0], [0, 0], [0, 0]], [[0, 1], [0, 0], [0, 0]]],
            self_collision_pairs=((0, 1),), commands=[{"skill_id": "timed", "event_id": "close", "occurrence": 0,
                "command": "close", "progress": 0.}], source="analytic independent finger reference")

    def test_timed_fingers_use_time_during_progress_pause_without_full_range_inflation(self):
        hand = self.timed_gripper()
        hand.bind_clock(1., None)
        progress = torch.zeros((1, 2), dtype=torch.float64)
        times = torch.tensor([[0., .5]], dtype=torch.float64)
        q, half, violations = hand.state(progress, None, None, trajectory_time=times)
        np.testing.assert_array_equal(q[0, 0], [.03, -.02])
        np.testing.assert_array_equal(q[0, 1], [.015, -.01])
        self.assertEqual(half.abs().max().item(), 0.)
        self.assertEqual(violations.max().item(), 0.)
        world = scene(materials=(None, None), gripper=hand, has_gripper=True,
                      background=SphereQuery(("background",), center=(.015, 0, 0), radius=.001))
        args = inputs((0., 0.), sphere_rows=[[[.03, 0, 0, .001], [-.02, 0, 0, .001]]]*2)
        costs = world.components(*args, trajectory_time=times)
        self.assertEqual(costs["collision/robot/background"][0, 0].item(), 0.)
        self.assertGreater(costs["collision/robot/background"][0, 1].item(), 0.)
        with self.assertRaisesRegex(ValueError, "actual query time"):
            world.components(*args)

    def test_raw_free_evidence_is_owned_readonly_and_not_filled_from_occupied_cells(self):
        mask = np.ones((4, 4, 4), bool)
        free = mask.copy()
        free[1, 1, 1] = False
        observed = coverage(mask, known_free=free)
        free[:] = True
        self.assertFalse(observed.known_free[1, 1, 1])
        with self.assertRaises(ValueError):
            observed.known_free[1, 1, 1] = True

    def test_native_coverage_estimate_removes_only_disabled_robot_sphere_scratch(self):
        # The solid body/FK layout is held constant while the robot's collision
        # proxy resolution changes. With the native-volume oracle, additional
        # robot balls add collision-query storage but no whole-ball unknown
        # query. A moving payload still needs both kinds of storage.
        world = scene()
        world.robot_coverage = SimpleNamespace(required_link_names=("body",), baseline_joint_names=("q",))
        world._coverage_geometry = {"solids": [{"baseline_halfspaces_world": np.zeros((32, 4))}]}
        world._free_distance = world.coverage._distance
        # BEGIN explicit immutable free-grid storage used by workspace admission.
        world._free_mask = world.coverage._distance
        # END the fixture needs both actual production grid fields.
        # BEGIN storage admission now reads the real per-owner field count.
        world.owners = (SimpleNamespace(owner_id="payload", moving=True, local_spheres=np.zeros((1, 4)),
                                       query=SphereQuery(("payload",))),)
        # END a complete owner fixture includes its collision query.
        batch, horizon = 2, 10
        before = world.estimated_workspace_bytes(batch, horizon)
        world.robot_sphere_materials = (None,)*17
        refined_robot = world.estimated_workspace_bytes(batch, horizon)
        self.assertEqual(refined_robot-before, batch*horizon*16*2*60)
        world.owners[0].local_spheres = np.zeros((3, 4))
        refined_payload = world.estimated_workspace_bytes(batch, horizon)
        self.assertEqual(refined_payload-refined_robot, batch*horizon*2*(60+128))
        # The two native free-space grids remain cached even at stages without
        # guidance. Removing the native oracle also removes those two caches.
        without_guidance = world.estimated_workspace_bytes(batch, horizon, robot_coverage_guidance=False)
        self.assertGreater(refined_payload, without_guidance)
        shared = world.estimated_workspace_allocation(batch, horizon, robot_coverage_horizon=0)["shared_grids"]
        self.assertEqual(shared, 24*world.coverage._distance.numel())
        self.assertEqual(shared, world.estimated_workspace_allocation(
            batch, horizon, robot_coverage_horizon=horizon)["shared_grids"])
        native_oracle, world.robot_coverage = world.robot_coverage, None
        ordinary = world.estimated_workspace_bytes(batch, horizon)
        world.robot_coverage = native_oracle
        self.assertEqual(ordinary-without_guidance,
                         batch*horizon*17*128-16*world.coverage._distance.numel())

    def test_full_volume_entrance_accepts_only_exact_named_representations(self):
        baseline = np.array([.123456789, -.234567891])
        received = []
        oracle = SimpleNamespace(reference_frame="world", required_link_names=("body",),
            baseline_joint_names=("q", "finger"), baseline_joint_positions=baseline,
            guidance_geometry=lambda: {"solids": []},
            validate_observation_grid=lambda **kwargs: None,
            check_state=lambda **kwargs: received.append(kwargs) or {"passed": kwargs["entrance"]})
        world = scene(robot_coverage=oracle)
        # BEGIN each named coordinate, including the independent finger, must
        # match one exact representation before the oracle gets entrance=True.
        q = torch.tensor([[[float(np.float32(baseline[0])), baseline[1]]]], dtype=torch.float64)
        world.check_robot_coverage(q, ("q", "finger"), torch.zeros((1, 1)), entrance=True)
        self.assertTrue(received[0]["entrance"])
        for column, step in ((0, np.spacing(q[0, 0, 0].item())),
                             (0, float(np.spacing(np.float32(baseline[0])))),
                             (1, np.spacing(baseline[1]))):
            changed = q.clone()
            changed[0, 0, column] += step
            report = world.check_robot_coverage(changed, ("q", "finger"), torch.zeros((1, 1)), entrance=True)
            self.assertFalse(report[0]["passed"])
            self.assertFalse(received[-1]["entrance"])
            np.testing.assert_array_equal(received[-1]["joint_positions"], changed.numpy()[0, 0])
        # Even an exact float32 start representation at a future sample must
        # pass the ordinary volume predicate without entrance canonicalization.
        world.check_robot_coverage(q, ("q", "finger"), torch.ones((1, 1)), entrance=False)
        np.testing.assert_array_equal(received[-1]["joint_positions"], q.numpy()[0, 0])
        self.assertFalse(received[-1]["entrance"])
        # END nearby first samples produce a verdict without changing the curve.
    # END command references never substitute for physical finger feedback.

    def test_partial_scene_cannot_be_declared_without_coverage_and_provenance(self):
        with self.assertRaisesRegex(ValueError, "coverage"):
            scene(coverage=None)
        with self.assertRaisesRegex(ValueError, "provenance"):
            ObservedSpace(np.ones((4, 4, 4), bool), origin=[0, 0, 0], voxel_size=.1,
                          observation_ids=[], reference_frame="world")
        with self.assertRaisesRegex(ValueError, "gripper"):
            scene(has_gripper=True)

    def test_unknown_grid_checks_whole_sphere_and_outside_grid(self):
        mask = np.ones((41, 41, 41), bool)
        mask[20, 20, 20] = False
        observed = coverage(mask)
        balls = torch.tensor([[[[.16, 0, 0, .2], [2.2, 0, 0, .02], [-1., 0, 0, .03]]]], dtype=torch.float64)
        violation = observed.violation(balls)
        self.assertGreater(violation[0, 0, 0].item(), 0)
        self.assertGreater(violation[0, 0, 1].item(), 0)
        self.assertEqual(violation[0, 0, 2].item(), 0)

    # BEGIN occupied-interior evidence must preserve collision and state contracts.
    def test_static_partial_owner_used_for_occupied_coverage_remains_unknown(self):
        evidence = {"solids": [{"body_kind": "owner", "body_id": "part", "occupancy_kind": "verified_inner_solid"}],
                    "owners": {"part": {"T_output_owner": np.eye(4).tolist()}}, "robot": None}
        owner = CollisionOwner("part", SphereQuery(("part",)), pose_path(), "synthetic_owner", False)
        world = scene((owner,), coverage=coverage(occupancy_evidence=evidence))
        values = world.components(*inputs())
        self.assertEqual(values["unknown/owner/part"].item(), 1.)
        self.assertEqual(values["collision/robot/part"].item(), 0.)
        self.assertEqual(world.describe()["occupied_owner_ids"], ("part",))
        owner.geometry_complete = True
        self.assertEqual(world.evaluate(*inputs()).item(), 0.)
        # Completing geometry never exempts that same owner from collision.
        owner.query = SphereQuery(("part",), center=(-1., 0., 0.), radius=.1)
        self.assertGreater(world.components(*inputs())["collision/robot/part"].item(), 0.)

    def test_occupied_owner_identity_and_current_pose_cannot_be_removed_or_edited(self):
        evidence = {"solids": [{"body_kind": "owner", "body_id": "part", "occupancy_kind": "verified_inner_solid"}],
                    "owners": {"part": {"T_output_owner": np.eye(4).tolist()}}, "robot": None}
        observed = coverage(occupancy_evidence=evidence)
        with self.assertRaisesRegex(ValueError, "explicit collision owner"):
            scene(coverage=observed)
        owner = CollisionOwner("part", SphereQuery(("part",)), pose_path(), "synthetic_owner", True)
        world = scene((owner,), coverage=observed)
        problem = PlanningProblem(("joint",), np.zeros(1), np.zeros(1), "tool", "world", np.eye(4), np.eye(4),
                                  (MotionPhase("move", pose_path()),))
        world.validate_problem(problem)
        owner.pose_path = pose_path(position=(.1, 0., 0.))
        with self.assertRaisesRegex(ValueError, "occupied-evidence current pose"):
            world.validate_problem(problem)

    def test_occupied_robot_evidence_requires_same_model_measured_feedback_and_base(self):
        evidence = {"solids": [{"body_kind": "robot_link", "body_id": "tool", "occupancy_kind": "verified_inner_solid"}],
                    "owners": {}, "robot": {"joint_names": ["joint"], "joint_position": [0.],
                                                "locked_joint_positions": {},
                                                "T_output_robot_base": np.eye(4).tolist(),
                                                "robot_geometry_source": "synthetic_robot", "robot_spheres": 1,
                                                "current_spheres_output": [[-1., 0., 0., .05]],
                                                "link_poses_output": {"tool": np.eye(4).tolist()}}}
        observed = coverage(occupancy_evidence=evidence)
        with self.assertRaisesRegex(ValueError, "complete native collision model"):
            scene(coverage=observed, robot_geometry_source="different_model")
        world = scene(coverage=observed)
        self.assertEqual(world.required_robot_links, ("tool",))
        world.validate_robot_geometry([[-1., 0., 0., .05]], {"tool": np.eye(4)})
        with self.assertRaisesRegex(ValueError, "collision sphere model disagree"):
            world.validate_robot_geometry([[-1., 0., 0., .06]], {"tool": np.eye(4)})
        moved = np.eye(4)
        moved[0, 3] = .01
        with self.assertRaisesRegex(ValueError, "link FK disagree"):
            world.validate_robot_geometry([[-1., 0., 0., .05]], {"tool": moved})
        problem = PlanningProblem(("joint",), np.zeros(1), np.zeros(1), "tool", "world", np.eye(4), np.eye(4),
                                  (MotionPhase("move", pose_path()),))
        world.validate_problem(problem)
        problem.start_q[0] = .000001
        with self.assertRaisesRegex(ValueError, "exact current joint feedback"):
            world.validate_problem(problem)
        problem.start_q[0] = 0.
        problem.base_pose[0, 3] = .000001
        with self.assertRaisesRegex(ValueError, "same current base calibration"):
            world.validate_problem(problem)
        # A binary drive still needs true current finger feedback for this
        # optional solid evidence; a YAML lock at another width is not it.
        problem.base_pose[0, 3] = 0.
        problem.events = (PlanningEvent("cmd", "skill", 0., 0, "close", "synthetic command"),)
        evidence["robot"]["locked_joint_positions"] = {"finger": .03}
        binary = scene(coverage=coverage(occupancy_evidence=evidence), gripper=self.gripper(), has_gripper=True)
        with self.assertRaisesRegex(ValueError, "actual current fingers"):
            binary.validate_problem(problem)
        evidence["robot"]["locked_joint_positions"] = {"finger": .04}
        scene(coverage=coverage(occupancy_evidence=evidence), gripper=self.gripper(), has_gripper=True).validate_problem(problem)
    # END occupied-interior evidence must preserve collision and state contracts.

    def test_owned_geometry_cannot_remain_in_background_under_another_partition(self):
        owner = CollisionOwner("part", SphereQuery(("part",)), pose_path(), "measured_part", True)
        with self.assertRaisesRegex(ValueError, "partitioned"):
            scene((owner,), background=SphereQuery(("part", "background")))

    def test_moving_owner_inverse_transform_and_progress_gradients(self):
        query = SphereQuery(("part",), center=(.3, 0., 0.), radius=.1)
        owner = CollisionOwner("part", query, pose_path(turn=np.pi / 2), "measured_part", True,
                               local_spheres=[[.3, 0, 0, .1]], moving=True)
        world = scene((owner,))
        balls, p, q, s = inputs((1.,), sphere_rows=[[[.02, .4, 0., .08]]])
        balls.requires_grad_()
        s.requires_grad_()
        value = world.components(balls, p, q, s)["collision/robot/part"]
        value.sum().backward()
        torch.testing.assert_close(query.calls[("robot", "part")][0, 0, 0, :3],
                                   torch.tensor([.4, -.02, 0.], dtype=torch.float64), atol=1e-8, rtol=0)
        self.assertGreater(value.item(), 0)
        self.assertTrue(torch.isfinite(balls.grad).all())
        self.assertGreater(balls.grad[..., :3].abs().sum().item(), 0)
        self.assertTrue(torch.isfinite(s.grad).all())
        self.assertGreater(s.grad.abs().sum().item(), 0)

    def test_owner_local_path_is_placed_in_declared_global_phase(self):
        owner = CollisionOwner("part", SphereQuery(("part",)), pose_path(end=(1., 0, 0)),
                               "measured_part", True, pose_interval=(2., 3.))
        s = torch.tensor([[1., 2.5, 4.]], dtype=torch.float64)
        p, _ = owner.predicted_pose(s)
        torch.testing.assert_close(p[..., 0], torch.tensor([[0., .5, 1.]], dtype=torch.float64))

    def test_local_contact_permission_does_not_remove_palm_or_other_owner(self):
        target = CollisionOwner("surface", BoxQuery("surface"), pose_path(), "measured_surface", True)
        allowed = scene((target,), materials=("pad",), contacts=(contact_rule(),))
        args = inputs(positions=[[0., 0., .009]], sphere_rows=[[[0., 0., .009, .01]]])
        self.assertLess(allowed.evaluate(*args).item(), 1e-12)
        outside = list(args)
        outside[-1] = torch.tensor([[1.2]], dtype=torch.float64)
        self.assertGreater(allowed.evaluate(*outside).item(), .0009)
        palm = scene((target,), materials=("pad", None), contacts=(contact_rule(),))
        palm_args = inputs(positions=[[0., 0., .009]], sphere_rows=[[[0., 0., .009, .01], [.02, 0., .009, .01]]])
        self.assertGreater(palm.evaluate(*palm_args).item(), .0009)
        other = CollisionOwner("other", BoxQuery("other"), pose_path(), "other_measurement", True)
        both = scene((target, other), materials=("pad",), contacts=(contact_rule(),))
        self.assertGreater(both.evaluate(*args).item(), .0009)

    def test_partial_feature_overlap_and_normal_guard_contact_margin(self):
        target = CollisionOwner("surface", BoxQuery("surface"), pose_path(), "measured_surface", True)
        args = inputs(positions=[[0., 0., .009]], sphere_rows=[[[0., 0., .009, .01]]])
        large = contact_rule(feature_points=[[-.05, -.012, -.01], [.05, -.012, -.01], [.05, .012, -.01], [-.05, .012, -.01]])
        self.assertEqual(scene((target,), materials=("pad",), contacts=(large,)).evaluate(*args).item(), 0.)
        backwards = contact_rule(feature_normal=[0, 0, 1])
        self.assertGreater(scene((target,), materials=("pad",), contacts=(backwards,)).evaluate(*args).item(), .0009)
        with self.assertRaisesRegex(ValueError, "whole links"):
            scene((target,), materials=(None,), contacts=(contact_rule(),))
        with self.assertRaisesRegex(ValueError, "thin-part"):
            contact_rule(modeling_allowance_m=.006, geometry_error_m=.01, minimum_thickness_m=.01)

    # BEGIN adjacent permissions preserve earlier contact without summing allowances.
    def test_adjacent_material_permissions_are_order_independent_and_never_stack(self):
        target = CollisionOwner("surface", BoxQuery("surface"), pose_path(), "measured_surface", True)
        other = CollisionOwner("other", SphereQuery(("other",)), pose_path(), "other_measurement", True)
        rules = (contact_rule(rule_id="first_contact", interval=(0., .5), gap_bounds=(-.004, .002)),
                 contact_rule(rule_id="second_contact", interval=(.5, 1.), gap_bounds=(-.004, .002)))
        args = inputs(progress=(.25, .5, .75, 1.2), positions=[[0., 0., .009]]*4,
                      sphere_rows=[[[0., 0., .009, .01]]]*4)
        for ordered in (rules, rules[::-1]):
            world = scene((target, other), materials=("pad",), contacts=ordered)
            values = world.evaluate(*args)
            torch.testing.assert_close(values[0, :3], torch.zeros(3, dtype=torch.float64))
            self.assertGreater(values[0, 3].item(), .0009)
            ordinary = target.query.calls[("robot", "surface")][0, :, 0, 3]
            torch.testing.assert_close(ordinary, torch.tensor([.01]*4, dtype=torch.float64))
            self.assertGreater(world._contact_cells["surface"].last_report["permitted_pairs"], 0)
            # Material permission is scoped to its owner. The same physical
            # robot retains its complete original radius for every other one.
            torch.testing.assert_close(other.query.calls[("robot", "other")][..., 3], args[0][..., 3])
            deep = inputs(progress=(.5,), positions=[[0., 0., .007]],
                          sphere_rows=[[[0., 0., .007, .01]]])
            deep[0].requires_grad_()
            cost = world.evaluate(*deep).sum()
            self.assertGreater(cost.item(), .0009)
            cost.backward()
            self.assertAlmostEqual(deep[0].grad[0, 0, 0, 2].item(), -1.)
    # END permission-order and non-additive-allowance counterexample.

    def payload_scene(self, *, complete=True, background=None):
        relation = Attachment((0., 1.), [.3, 0, 0], [1., 0, 0, 0], "observed_grasp_needed", "release_feedback_needed")
        owner = CollisionOwner("payload", SphereQuery(("payload",), center=(0, 0, 0), radius=.05),
                               pose_path((.3, 0, 0)), "observed_payload", complete,
                               local_spheres=[[0, 0, 0, .05]], attachments=(relation,))
        return scene((owner,), **({} if background is None else {"background": background}))

    def test_release_uses_actual_anchor_and_keeps_gradient_to_release_q(self):
        background = SphereQuery(("background",), center=(.85, 0, 0), radius=.07)
        world = self.payload_scene(background=background)
        balls, p, q, s = inputs((.5, 1., 1.5), positions=[[0., 0, 0], [.5, 0, 0], [1.2, 0, 0]])
        p.requires_grad_()
        components = world.components(balls, p, q, s)
        actual = background.calls[("payload", "background")][..., 0, 0]
        torch.testing.assert_close(actual, torch.tensor([[.3, .8, .8]], dtype=torch.float64))
        components["collision/payload/background"][0, -1].backward()
        self.assertGreater(abs(p.grad[0, 1, 0].item()), .9)
        self.assertEqual(p.grad[0, 2].abs().sum().item(), 0.)
        self.assertFalse(any(key.startswith("attachment/boundary") for key in components))
        self.assertIn("payload", world.describe()["release_requires_evidence"])

    def test_missing_release_anchor_fails_and_selected_prefix_supplies_history(self):
        world = self.payload_scene()
        balls, p, q, s = inputs((.5, 1.5), positions=[[0., 0, 0], [1., 0, 0]])
        missing = world.components(balls, p, q, s)["unknown/release_anchor/payload/0"]
        torch.testing.assert_close(missing, torch.tensor([[0., 1.]], dtype=torch.float64))
        with self.assertRaisesRegex(ValueError, "exact release"):
            world.with_release_history(p, q, s)
        _, ep, eq, es = inputs((0., 1.), positions=[[0., 0, 0], [.5, 0, 0]])
        fixed = world.with_release_history(ep, eq, es)
        args = inputs((1.5,), positions=[[1.2, 0, 0]])
        fixed.evaluate(*args)
        torch.testing.assert_close(fixed.background.calls[("payload", "background")][..., 0, 0],
                                   torch.tensor([[.8]], dtype=torch.float64))
        self.assertIsNot(fixed.background, world.background)
        self.assertFalse(world._fixed_release_poses)

    # BEGIN only reached releases are inherited, with private sibling poses.
    def test_search_prefix_does_not_freeze_future_release_or_overwrite_first_pose(self):
        world = self.payload_scene()
        _, p, q, s = inputs((.5,), positions=[[.1, 0., 0.]])
        before = world.with_release_history(p, q, s)
        self.assertFalse(before._fixed_release_poses)
        _, p, q, s = inputs((1.,), positions=[[.5, 0., 0.]])
        first = before.with_release_history(p, q, s)
        _, p, q, s = inputs((1.,), positions=[[.8, 0., 0.]])
        sibling = before.with_release_history(p, q, s)
        _, p, q, s = inputs((1., 1.5), positions=[[1.1, 0., 0.], [1.3, 0., 0.]])
        extended = first.with_release_history(p, q, s)
        key = ("payload", 0)
        self.assertAlmostEqual(first._fixed_release_poses[key][0][0, 0, 0].item(), .8)
        self.assertAlmostEqual(extended._fixed_release_poses[key][0][0, 0, 0].item(), .8)
        self.assertAlmostEqual(sibling._fixed_release_poses[key][0][0, 0, 0].item(), 1.1)
        self.assertFalse(world._fixed_release_poses)
        self.assertFalse(before._fixed_release_poses)
    # END the selected search history is never written into actual scene facts.

    def test_payload_keeps_noncontact_robot_collision_and_partial_cover_unknown(self):
        world = self.payload_scene()
        args = inputs((.5, 1.), positions=[[0., 0, 0], [0., 0, 0]],
                      sphere_rows=[[[.32, 0, 0, .04]], [[.32, 0, 0, .04]]])
        self.assertGreater(world.components(*args)["collision/robot/payload"].min().item(), .06)
        partial = self.payload_scene(complete=False)
        self.assertEqual(partial.components(*args)["unknown/owner/payload"].min().item(), 1.)

    def test_place_regrasp_carry_shares_each_previous_release_state(self):
        first = Attachment((0., 1.), [.3, 0, 0], [1., 0, 0, 0], "grasp1", "release1")
        second = Attachment((2., 3.), [.2, 0, 0], [1., 0, 0, 0], "grasp2", "release2")
        owner = CollisionOwner("payload", SphereQuery(("payload",)), pose_path((.3, 0, 0)),
                               "observed_payload", True, local_spheres=[[0, 0, 0, .05]],
                               attachments=(first, second))
        world = scene((owner,))
        args = inputs((0., .5, 1., 1.5, 2., 2.5, 3., 3.5),
                      positions=[[0., 0, 0], [.2, 0, 0], [.5, 0, 0], [-.1, 0, 0],
                                 [.6, 0, 0], [1., 0, 0], [1.2, 0, 0], [1.8, 0, 0]])
        terms = world.components(*args)
        actual = world.background.calls[("payload", "background")][..., 0, 0]
        torch.testing.assert_close(actual, torch.tensor([[.3, .5, .8, .8, .8, 1.2, 1.4, 1.4]], dtype=torch.float64))
        self.assertEqual(terms["attachment/start/payload/1"].sum().item(), 0.)
        # A second grasp cannot reset the same owner to its initial .3 m pose.
        bad = list(args)
        bad[1] = args[1].clone()
        bad[1][0, 4, 0] = .1
        self.assertGreater(world.components(*bad)["attachment/start/payload/1"][0, 4].item(), .49)
        frozen = world.with_release_history(args[1], args[2], args[3])
        later = inputs((2.5,), positions=[[1., 0, 0]])
        frozen.components(*later)
        self.assertEqual(len(frozen._fixed_release_poses), 2)
        self.assertEqual(world.required_progress_anchors, (0., 1., 2., 3.))

    def test_attachment_start_has_separate_numerical_pose_tolerances(self):
        world = self.payload_scene()
        args = inputs((0., 1.), positions=[[.00001, 0, 0], [0, 0, 0]])
        terms = world.components(*args)
        self.assertEqual(terms["attachment/start/payload/0"].sum().item(), 0.)
        bad = list(args)
        bad[1] = args[1].clone()
        bad[1][0, 0, 0] = .002
        self.assertGreater(world.components(*bad)["attachment/start/payload/0"][0, 0].item(), .0018)

    def test_fp32_release_anchor_is_matched_before_any_precision_conversion(self):
        anchor = 17 + 1 / 3
        relation = Attachment((17., anchor), [.3, 0, 0], [1., 0, 0, 0], "grasp", "release")
        owner = CollisionOwner("payload", SphereQuery(("payload",)), pose_path((.3, 0, 0)),
                               "observed_payload", True, local_spheres=[[0, 0, 0, .05]],
                               attachments=(relation,))
        world = scene((owner,))
        args = inputs((17., anchor), positions=[[0., 0, 0], [.5, 0, 0]], dtype=torch.float32)
        fixed = world.with_release_history(args[1], args[2], args[3])
        self.assertEqual(len(fixed._fixed_release_poses), 1)
        result = fixed.components(*inputs((18.,), positions=[[1., 0, 0]], dtype=torch.float32))
        self.assertFalse(any(key.startswith("unknown/release_anchor") for key in result))

    def test_overlapping_attachment_cycles_are_rejected(self):
        a = Attachment((0., 2.), [.3, 0, 0], [1., 0, 0, 0], "first", "release1")
        b = Attachment((1., 3.), [.2, 0, 0], [1., 0, 0, 0], "second", "release2")
        with self.assertRaisesRegex(ValueError, "cannot overlap"):
            CollisionOwner("payload", SphereQuery(("payload",)), pose_path(), "observation", True,
                           local_spheres=[[0, 0, 0, .05]], attachments=(a, b))

    def gripper(self, controller="binary", tolerance=1e-5, control_timestep_s=.002):
        def bounds(progress):
            value = progress.new_full((*progress.shape, 1), .04)
            return value, value
        return PrismaticGripper(
            controller=controller, control_timestep_s=control_timestep_s,
            joint_names=("finger",), current_positions=[.04], joint_limits=[[0, .04]],
            joint_bounds=bounds if controller == "position_tracking" else None,
            coupling_matrix=np.zeros((0, 1)), coupling_offset=[], sphere_indices=(0,),
            local_spheres=[[0, 0, 0, .005]], sphere_joint_translation=[[[1.], [0.], [0.]]],
            self_collision_pairs=(), commands=[{"skill_id": "skill", "event_id": "cmd", "occurrence": 0, "command": "close", "progress": 0.}],
            source="synthetic_drive_and_tool", numerical_joint_tolerance=tolerance)

    # BEGIN binary zero-progress queries must retain the physical sweep envelope.
    def test_binary_continuous_envelope_catches_collision_between_open_and_closed(self):
        background = SphereQuery(("background",), center=(.012, 0., 0.), radius=.001)
        world = scene(gripper=self.gripper(), has_gripper=True, background=background)
        args = inputs((0., .5), sphere_rows=[[[.04, 0, 0, .005]], [[.04, 0, 0, .005]]])
        terms = world.components(*args)
        self.assertEqual(terms["collision/robot/background"].max().item(), 0.)
        self.assertGreater(terms["unknown/collision/robot/background"].min().item(), .01)
        self.assertIn("time-independent", world.describe()["gripper_timing_contract"])
        # Neither endpoint sphere intersects this obstacle; the swept interval does.
        endpoints = torch.tensor([[[[0, 0, 0, .005], [.04, 0, 0, .005]]]], dtype=torch.float64)
        self.assertEqual(background(endpoints, key="endpoint_only").sum().item(), 0.)

    def test_binary_paused_and_isolated_zero_progress_do_not_receive_initial_exemption(self):
        background = SphereQuery(("background",), center=(.012, 0., 0.), radius=.001)
        world = scene(gripper=self.gripper(), has_gripper=True, background=background)
        # The arm can move 1 mm while progress remains zero. A binary close
        # command can put the finger at .012 m then, although the earlier
        # feedback was .04 m. This physically reachable sphere intersects.
        actual = torch.tensor([[[[.013, 0, 0, .005]]]], dtype=torch.float64)
        self.assertGreater(background(actual, key="reachable_actual_finger").item(), .004)
        for progress, positions in (
                ((0., 0., .5), [[0., 0., 0.], [.001, 0., 0.], [.001, 0., 0.]]),
                ((0.,), [[.001, 0., 0.]])):
            with self.subTest(progress=progress):
                rows = [[[position[0]+.04, 0., 0., .005]] for position in positions]
                terms = world.components(*inputs(progress, positions=positions, sphere_rows=rows))
                self.assertEqual(terms["collision/robot/background"].max().item(), 0.)
                self.assertGreater(terms["unknown/collision/robot/background"].min().item(), .01)
    # END paused and isolated zero-progress binary regression.

    def test_tracking_commands_and_coupling_use_separate_numerical_tolerance(self):
        hand = self.gripper("position_tracking")
        s = torch.tensor([[.5]], dtype=torch.float32)
        _, _, small = hand.state(s, torch.tensor([[[.0400001]]]), ("finger",))
        self.assertEqual(small.item(), 0.)
        _, _, changed = hand.state(s, torch.tensor([[[.02]]]), ("finger",))
        self.assertGreater(changed.item(), .019)
        with self.assertRaisesRegex(ValueError, "named solver joints"):
            hand.state(s, None, None)

    def test_tracking_preference_keeps_driving_to_command_without_changing_acceptance(self):
        hand = self.gripper("position_tracking")
        world = scene(gripper=hand, has_gripper=True)
        s = torch.tensor([[.5]], dtype=torch.float64)
        q = torch.tensor([[[.039998]]], dtype=torch.float64, requires_grad=True)
        self.assertEqual(hand.state(s, q, ("finger",))[2].item(), 0.)
        objective = world.optimizer_preferences(s, q, ("finger",))
        self.assertGreater(objective.item(), 0.)
        objective.sum().backward()
        self.assertLess(q.grad.item(), 0.)
        self.assertEqual(world.optimizer_preferences(s, q.detach().new_full(q.shape, .04), ("finger",)).item(), 0.)
        self.assertEqual(scene().optimizer_preferences(s, q, ("finger",)).item(), 0.)
        binary = scene(gripper=self.gripper("binary"), has_gripper=True)
        self.assertEqual(binary.optimizer_preferences(s, q, ("finger",)).item(), 0.)

    def test_tracking_projection_preserves_gradients_without_repeating_coupling(self):
        hand = PrismaticGripper(
            controller="position_tracking", control_timestep_s=.002, joint_names=("left", "right"),
            current_positions=[.02, -.02], joint_limits=[[0, .04], [-.04, 0]],
            joint_bounds=lambda s: (s.new_tensor([.02, -.02]).expand(*s.shape, 2),
                                    s.new_tensor([.02, -.02]).expand(*s.shape, 2)),
            coupling_matrix=[[1., 1.]], coupling_offset=[0.], sphere_indices=(0,),
            local_spheres=[[0, 0, 0, .005]], sphere_joint_translation=[[[1., 0], [0., 0], [0., 0]]],
            self_collision_pairs=(), commands=[{"skill_id": "skill", "event_id": "drive", "occurrence": 0, "command": "hold", "progress": 0.}],
            source="synthetic_coupled_native_drive")
        world = scene(gripper=hand, has_gripper=True)
        q = torch.tensor([[[.03, -.03]]], dtype=torch.float64, requires_grad=True)
        residual = world.project_residual(torch.tensor([[.5]], dtype=torch.float64), q, ("left", "right"))
        self.assertEqual(residual.shape, (1, 1, 1))
        self.assertGreater(residual.abs().item(), .014)
        residual.sum().backward()
        self.assertAlmostEqual(q.grad.sum().item(), 0.)
        self.assertEqual(world.describe()["gripper_projection_dimension"], 1)

    def test_required_contact_with_uncertain_fingers_is_explicitly_unknown(self):
        target = CollisionOwner("surface", BoxQuery("surface"), pose_path(), "measured_surface", True)
        rule = contact_rule(required=True,
                            feature_joint_translation=np.tile(np.array([[[1.], [0.], [0.]]]), (4, 1, 1)))
        world = scene((target,), materials=("pad",), contacts=(rule,), gripper=self.gripper(), has_gripper=True)
        args = inputs(positions=[[0., 0., .1]], sphere_rows=[[[.02, 0., .1, .005]]])
        result = world.components(*args)
        self.assertEqual(result["unknown/contact/pad_to_surface"].item(), 1.)

    # BEGIN paired geometric successes for moving closure, carried support and new observations.
    def test_tracking_closes_while_tool_moves_and_keeps_material_permission_local(self):
        def command(progress):
            width = (.04-.03*progress)[..., None]
            return width, width
        hand = PrismaticGripper(
            controller="position_tracking", control_timestep_s=.002, joint_names=("finger",), current_positions=[.04],
            joint_limits=[[0., .04]], joint_bounds=command, coupling_matrix=np.zeros((0, 1)), coupling_offset=[],
            sphere_indices=(0,), local_spheres=[[0., 0., 0., .005]],
            sphere_joint_translation=[[[0.], [0.], [1.]]], self_collision_pairs=(),
            commands=[{"skill_id": "moving_close", "event_id": "close", "occurrence": 0, "command": "close", "progress": 0.}],
            source="explicit affine finger model and position-tracking command")
        target = CollisionOwner("surface", BoxQuery("surface"), pose_path(), "known test surface", True)
        rule = contact_rule(feature_points=[[-.005, -.005, -.005], [.005, -.005, -.005],
                                             [.005, .005, -.005], [-.005, .005, -.005]],
                            feature_joint_translation=np.tile(np.array([[[0.], [0.], [1.]]]), (4, 1, 1)))
        world = scene((target,), materials=("pad",), contacts=(rule,), gripper=hand, has_gripper=True)
        args = inputs((0., .25, .5, .75, 1.), positions=[[-.02+.02*s, 0., -.006] for s in (0., .25, .5, .75, 1.)])
        joints = command(args[-1])[0]
        self.assertLess(world.evaluate(*args, joint_position=joints, joint_names=("finger",)).max().item(), 1e-12)
        actual = world.background.calls[("robot", "background")][0, :, 0, :3]
        self.assertGreater(float(torch.diff(actual[:, 0]).min()), 0.)
        self.assertLess(float(torch.diff(actual[:, 2]).max()), 0.)
        # The same moving closure fails without the finite material patch;
        # a separate obstacle on the intermediate sweep also remains active.
        unpermitted = scene((target,), materials=("pad",), gripper=hand, has_gripper=True)
        self.assertGreater(unpermitted.evaluate(*args, joint_position=joints, joint_names=("finger",))[0, -1].item(), .0009)
        blocked = scene((target,), materials=("pad",), contacts=(rule,), gripper=hand, has_gripper=True,
                        background=SphereQuery(("background",), center=(-.01, 0., .019), radius=.002))
        self.assertGreater(blocked.evaluate(*args, joint_position=joints, joint_names=("finger",))[0, 2].item(), .006)

    def test_carried_payload_can_reach_its_support_patch_but_environment_is_still_checked(self):
        relation = Attachment((0., 1.), [.3, 0., 0.], [1., 0., 0., 0.], "conditional_grasp", None)
        payload = CollisionOwner("payload", SphereQuery(("payload",), center=(0., 0., 0.), radius=.01),
                                 pose_path((.3, 0., .1)), "complete synthetic payload sphere", True,
                                 local_spheres=[[0., 0., 0., .01]], sphere_materials=("bottom",),
                                 attachments=(relation,))
        support = CollisionOwner("surface", BoxQuery("surface"), pose_path(), "known finite support", True)
        permission = contact_rule(actor="payload", material_id="bottom", patch_origin=[.3, 0., 0.])
        world = scene((payload, support), contacts=(permission,))
        args = inputs((0., .5, 1.), positions=[[0., 0., .1], [0., 0., .06], [0., 0., .009]])
        self.assertLess(world.evaluate(*args).max().item(), 1e-12)
        unpermitted = scene((payload, support))
        self.assertGreater(unpermitted.evaluate(*args)[0, -1].item(), .0009)
        blocked = scene((payload, support), contacts=(permission,),
                        background=SphereQuery(("background",), center=(.3, 0., .06), radius=.006))
        self.assertGreater(blocked.components(*args)["collision/payload/background"][0, 1].item(), .015)

    def test_new_calibrated_view_makes_the_same_free_trajectory_observed_and_feasible(self):
        from manipulation.planning_observation import observed_space_from_rgbd
        first_depth, second_depth = np.full((96, 96), 2.), np.full((96, 96), 2.)
        first_depth[35:61, 35:61] = 0.
        matrix = np.array([[60., 0., 47.5], [0., 60., 47.5], [0., 0., 1.]])
        first = {"depth": first_depth, "K": matrix, "T_output_camera": np.eye(4), "reference_frame": "world"}
        second_pose = np.eye(4)
        second_pose[0, 3] = .3
        second = {"depth": second_depth, "K": matrix, "T_output_camera": second_pose, "reference_frame": "world"}
        def observed(views):
            return observed_space_from_rgbd(views, origin=[-.5, -.5, .2], shape=(45, 21, 21),
                                              voxel_size=.05, depth_error_m=.01, reference_frame="world", depth_range_m=(.1, 3.))
        args = inputs((0., .25, .5, .75, 1.),
                      sphere_rows=[[[x, 0., 1., .02]] for x in (-.12, -.06, 0., .06, .12)])
        missing = scene(coverage=observed({"first_current_camera": first}))
        self.assertGreater(missing.components(*args)["unknown/space/robot"].max().item(), .019)
        complete = scene(coverage=observed({"first_current_camera": first, "second_current_camera": second}))
        self.assertLess(complete.evaluate(*args).max().item(), 1e-12)
        self.assertEqual(complete.coverage.observation_ids, ("first_current_camera", "second_current_camera"))
        np.testing.assert_array_equal(first_depth[35:61, 35:61], np.zeros((26, 26)))
    # END paired geometric successes; these CPU proxies are not robot planning certificates.

    def test_cache_work_estimate_grows_with_horizon_and_payload_partitions(self):
        plain, payload = scene(), self.payload_scene()
        self.assertGreater(plain.estimated_workspace_bytes(2, 32), plain.estimated_workspace_bytes(1, 8))
        self.assertGreater(payload.estimated_workspace_bytes(2, 32), plain.estimated_workspace_bytes(2, 32))

    # BEGIN scene/problem binding counterexamples, independent of GPU kernels.
    def bound_payload(self):
        initial, attachment = np.eye(4), np.eye(4)
        initial[0, 3], attachment[0, 3] = .3, -.3
        relation = Attachment((0., 1.), [.3, 0, 0], [1., 0, 0, 0], "observed_holding", None)
        owner = CollisionOwner("payload", SphereQuery(("payload",)), pose_path((.3, 0, 0)),
                               "current_observation", True, local_spheres=[[0, 0, 0, .05]], attachments=(relation,))
        phase = MotionPhase("held", pose_path(), metadata={
            "mode": "held", "carrier_id": "payload", "owner_initial_pose": initial,
            "attachment": attachment, "attachment_interval": [0., 1.],
            "attachment_condition_id": "observed_holding", "release_condition_id": None})
        problem = PlanningProblem(("arm",), np.zeros(1), np.zeros(1), "tool", "world", np.eye(4), np.eye(4),
                                  (phase,), conditions=({"kind": "observed_attachment", "carrier_id": "payload",
                                                        "condition_id": "observed_holding", "progress": 0.},))
        return scene((owner,)), problem

    def test_bound_payload_cannot_be_omitted_or_changed_to_evade_collision(self):
        world, problem = self.bound_payload()
        world.validate_problem(problem)
        self.assertNotIn("payload", world.describe()["release_requires_evidence"])
        with self.assertRaisesRegex(ValueError, "explicit owner"):
            scene().validate_problem(problem)
        owner = world.owners[0]
        owner.attachments = (replace(owner.attachments[0], position_tool_owner=[.2, 0, 0]),)
        with self.assertRaisesRegex(ValueError, "attachment transform"):
            world.validate_problem(problem)
        world, problem = self.bound_payload()
        world.owners[0].pose_path = pose_path((.4, 0, 0))
        with self.assertRaisesRegex(ValueError, "initial observed pose"):
            world.validate_problem(problem)

    def test_window_end_holding_follows_each_fk_while_real_release_freezes_first(self):
        world, _ = self.bound_payload()
        args = inputs((1., 1.), positions=[[0., 0., 0.], [.2, 0., 0.]])
        world.components(*args)
        actual = world.background.calls[("payload", "background")][..., 0, 0]
        torch.testing.assert_close(actual, actual.new_tensor([[.3, .5]]))
        # No real release means search can evaluate a subpath without asking
        # for the artificial window end as a frozen physical event.
        subset = inputs((.5,), positions=[[.1, 0., 0.]])
        fixed = world.with_release_history(subset[1], subset[2], subset[3])
        self.assertFalse(fixed._fixed_release_poses)
        self.assertFalse(any(key.startswith("unknown/release_anchor") for key in fixed.components(*subset)))
        owner = world.owners[0]
        owner.attachments = (replace(owner.attachments[0], release_condition_id="actual_release"),)
        world.components(*args)
        actual = world.background.calls[("payload", "background")][..., 0, 0]
        torch.testing.assert_close(actual, actual.new_tensor([[.3, .3]]))

    def test_attachment_requires_exact_span_condition_and_real_release_boundary(self):
        world, problem = self.bound_payload()
        world.owners[0].attachments = (replace(world.owners[0].attachments[0], interval=(0., .8)),)
        with self.assertRaisesRegex(ValueError, "attachment span"):
            world.validate_problem(problem)
        world, problem = self.bound_payload()
        world.owners[0].attachments = (replace(world.owners[0].attachments[0], condition_id="invented"),)
        with self.assertRaisesRegex(ValueError, "condition IDs"):
            world.validate_problem(problem)
        world, problem = self.bound_payload()
        problem.phases += (MotionPhase("later", pose_path(), "free"),)
        with self.assertRaisesRegex(ValueError, "window boundary"):
            world.validate_problem(problem)
        world, problem = self.bound_payload()
        problem.conditions += ({"kind": "predicted_release", "carrier_id": "payload", "progress": 1.,
                                "condition_id": "required_release"},)
        with self.assertRaisesRegex(ValueError, "real release condition"):
            world.validate_problem(problem)

    def test_unattached_task_carrier_cannot_move_away_and_return(self):
        world, problem = self.bound_payload()
        owner = world.owners[0]
        owner.attachments = ()
        initial = problem.phases[0].metadata["owner_initial_pose"]
        problem.phases = (MotionPhase("grasp", pose_path(), metadata={
            "mode": "grasp", "carrier_id": "payload", "owner_initial_pose": initial}),)
        poses = np.repeat(initial[None], 3, axis=0)
        poses[1, 1, 3] = .4
        owner.pose_path = PosePath(np.array([0., .5, 1.]), poses, "independent altered obstacle path")
        with self.assertRaisesRegex(ValueError, "unattached task-carrier pose"):
            world.validate_problem(problem)

    def test_gripper_events_require_same_skill_and_native_joint_feedback(self):
        event = PlanningEvent("cmd", "skill", 0., 0, "close", "original_command")
        problem = PlanningProblem(("finger",), np.array([.04]), np.zeros(1), "tool", "world", np.eye(4), np.eye(4),
                                  (MotionPhase("free", pose_path()),), events=(event,))
        with self.assertRaisesRegex(ValueError, "skill/event/occurrence/command/progress"):
            scene().validate_problem(problem)
        hand = self.gripper("position_tracking")
        world = scene(gripper=hand, has_gripper=True)
        world.validate_problem(problem)
        hand.commands = ({**hand.commands[0], "skill_id": "another_instance"},)
        with self.assertRaisesRegex(ValueError, "skill/event/occurrence/command/progress"):
            world.validate_problem(problem)
        hand.commands = ({**hand.commands[0], "skill_id": "skill"},)
        # BEGIN retry geometry must bind the new occurrence of the same command.
        hand.commands = ({**hand.commands[0], "occurrence": 1},)
        with self.assertRaisesRegex(ValueError, "occurrence"):
            world.validate_problem(problem)
        problem.events = (PlanningEvent("cmd", "skill", 0., 0, "close", "original_command", occurrence=1),)
        world.validate_problem(problem)
        # END preserving the original source ID alone cannot authorize a retry.
        problem.start_q[0] = .02
        with self.assertRaisesRegex(ValueError, "current native feedback"):
            world.validate_problem(problem)

    # BEGIN an arm-only request must not invent a hold action to bind the hand.
    def test_arm_only_request_binds_measured_hand_with_explicit_empty_commands(self):
        hand = PrismaticGripper(controller="binary", control_timestep_s=.002,
            joint_names=("finger",), current_positions=[.04], joint_limits=[[0., .04]],
            joint_bounds=None, coupling_matrix=np.empty((0, 1)), coupling_offset=[],
            sphere_indices=(0,), local_spheres=[[0., 0., 0., .005]],
            sphere_joint_translation=[[[1.], [0.], [0.]]], self_collision_pairs=(),
            commands=(), source="explicit measured hand; no requested hand action")
        world = scene(gripper=hand, has_gripper=True)
        problem = PlanningProblem(("arm",), np.zeros(1), np.zeros(1), "tool", "world",
            np.eye(4), np.eye(4), (MotionPhase("arm reach", pose_path()),))
        world.validate_problem(problem)
        self.assertEqual(hand.commands, ())
        problem.events = (PlanningEvent("close", "skill", 0., 0, "close", "explicit new action"),)
        with self.assertRaisesRegex(ValueError, "skill/event/occurrence/command/progress"):
            world.validate_problem(problem)
    # END real hand actions still require their exact scene command identities.

    def test_tracking_drive_projection_cannot_omit_problem_coupling(self):
        hand = PrismaticGripper(
            controller="position_tracking", control_timestep_s=.002, joint_names=("left", "right"),
            current_positions=[.02, -.02], joint_limits=[[0, .04], [-.04, 0]],
            joint_bounds=lambda s: (s.new_tensor([.02, -.02]).expand(*s.shape, 2),
                                    s.new_tensor([.02, -.02]).expand(*s.shape, 2)),
            coupling_matrix=[[1., 1.]], coupling_offset=[0.], sphere_indices=(0,),
            local_spheres=[[0, 0, 0, .005]], sphere_joint_translation=[[[1., 0], [0., 0], [0., 0]]],
            self_collision_pairs=(), commands=[{"skill_id": "skill", "event_id": "drive", "occurrence": 0, "command": "hold", "progress": 0.}],
            source="actual_native_coupling")
        event = PlanningEvent("drive", "skill", 0., 0, "hold", "original_command")
        problem = PlanningProblem(("left", "right"), np.array([.02, -.02]), np.zeros(2), "tool", "world",
                                  np.eye(4), np.eye(4), (MotionPhase("free", pose_path()),), events=(event,))
        world = scene(gripper=hand, has_gripper=True)
        with self.assertRaisesRegex(ValueError, "A q=b"):
            world.validate_problem(problem)
        problem.couplings = (JointCoupling(("right", "left"), np.array([[1., 1.]]), np.array([0.]), "actual_native_coupling"),)
        world.validate_problem(problem)
        problem.couplings[0].matrix[0, 1] = 2.
        with self.assertRaisesRegex(ValueError, "A q=b"):
            world.validate_problem(problem)
    # END scene/problem binding counterexamples, independent of GPU kernels.


class MutableKernel(torch.autograd.Function):
    """Match V2's mutable precomputed-gradient buffer contract on CPU."""
    @staticmethod
    def forward(ctx, spheres, buffer):
        buffer.distance.copy_(spheres[..., 0].square())
        buffer.gradient.zero_()
        buffer.gradient[..., 0] = 2 * spheres[..., 0]
        ctx.save_for_backward(buffer.gradient)
        return buffer.distance

    @staticmethod
    def backward(ctx, output_gradient):
        (gradient,) = ctx.saved_tensors
        return gradient * output_gradient[..., None], None


class RawQuery:
    device_cfg = SimpleNamespace(device=torch.device("cpu"), collision_distance_dtype=torch.float32,
                                 collision_gradient_dtype=torch.float32,
                                 is_same_torch_device=lambda device: device == torch.device("cpu"))

    def get_sphere_distance_raw(self, spheres, buffer, weight, activation, return_loss):
        assert return_loss
        return MutableKernel.apply(spheres, buffer)


class CuroboQueryWorkspaceTest(unittest.TestCase):
    # BEGIN shape replacement preserves outstanding values and gradients.
    def test_replaced_shape_frees_cache_but_preserves_both_backward_snapshots(self):
        import weakref
        query = CuroboSphereQuery(RawQuery(), owner_ids=("fixture",), source="test_kernel_contract",
                                 max_buffers=1, max_cache_bytes=48)
        first = torch.tensor([[[[.2, 0, 0, .1]]]], requires_grad=True)
        second = torch.tensor([[[[-.6, 0, 0, .1]], [[.7, 0, 0, .1]]]], requires_grad=True)
        a = query(first, key="same")
        old_buffer = weakref.ref(next(iter(query._buffers.values()))[1])
        b = query(second, key="same")
        self.assertIsNone(old_buffer())
        self.assertEqual(len(query._buffers), 1)
        self.assertEqual(query._cache_bytes, 48)
        current = next(iter(query._buffers.values()))[1]
        query(second.detach(), key="same")
        self.assertIs(next(iter(query._buffers.values()))[1], current)
        (3*a.sum()+2*b.sum()).backward()
        torch.testing.assert_close(first.grad[..., 0], torch.full((1, 1, 1), 1.2))
        torch.testing.assert_close(second.grad[..., 0], torch.tensor([[[-2.4], [2.8]]]))
        self.assertAlmostEqual(a.item(), .04, places=6)
        torch.testing.assert_close(b, torch.tensor([[[.36], [.49]]]))
        oversized = torch.zeros((1, 3, 1, 4))
        with self.assertRaisesRegex(ValueError, "total query cache"):
            query(oversized, key="same")
        self.assertIs(next(iter(query._buffers.values()))[1], current)
        self.assertEqual(query._cache_bytes, 48)
        with self.assertRaisesRegex(ValueError, "bounded workspace cache"):
            query.max_cache_bytes = 128
            query(first.detach(), key="another")
    # END cache limits still reject a genuinely inadmissible replacement.

    def test_reused_kernel_buffers_cannot_overwrite_unbackpropagated_queries(self):
        query = CuroboSphereQuery(RawQuery(), owner_ids=("fixture",), source="test_kernel_contract")
        first = torch.tensor([[[[.2, 0, 0, .1]]]], requires_grad=True)
        second = torch.tensor([[[[-.6, 0, 0, .1]]]], requires_grad=True)
        a, b = query(first, key="same"), query(second, key="same")
        (3*a + 2*b).sum().backward()
        self.assertAlmostEqual(a.item(), .04, places=6)
        self.assertAlmostEqual(first.grad[0, 0, 0, 0].item(), 1.2, places=6)
        self.assertAlmostEqual(second.grad[0, 0, 0, 0].item(), -2.4, places=6)
        cloned = query.clone_for_rollout()
        self.assertFalse(cloned._buffers)
        self.assertIs(cloned.scene_collision, query.scene_collision)

    def test_total_cache_byte_cap_prevents_accumulating_many_query_shapes(self):
        query = CuroboSphereQuery(RawQuery(), owner_ids=("fixture",), source="test_kernel_contract", max_cache_bytes=28)
        value = torch.tensor([[[[.2, 0, 0, .1]]]])
        query(value, key="first")
        with self.assertRaisesRegex(ValueError, "total query cache"):
            query(value, key="second")
        # BEGIN reject the complete query before making it contiguous or allocating.
        default = CuroboSphereQuery(RawQuery(), owner_ids=("fixture",), source="bounded native query")
        oversized = value.expand(1, 1, default.max_query_spheres+1, 4)
        with self.assertRaisesRegex(ValueError, "sphere.*budget|sphere.*limit|sphere.*bound"):
            default(oversized, key="oversized")
        self.assertFalse(default._buffers)
        # END a broadcast view exercises admission without a large allocation.


if __name__ == "__main__":
    unittest.main()
# END CPU scene ownership, contact, release and coverage regression tests.

# BEGIN formal scene snapshots carry model values rather than labels or counts.
def test_execution_scene_binding_owns_geometry_masks_permissions_and_nominal_identity():
    import json
    native = SimpleNamespace(get_sphere_distance_raw=lambda *args, **kwargs: None,
        scene_model=SimpleNamespace(cuboid=[], cylinder=[], sphere=[], capsule=[], mesh=[], voxel=[]))
    background = CuroboSphereQuery(native, owner_ids=("background",), source="empty observed native partition")
    target_query = CuroboSphereQuery(native, owner_ids=("target",), source="observed target body")
    target_query.independent_geometry = {"solids": [{"name": "target_box", "kind": "cuboid", "dimensions": [.2, .3, .4],
        "pose": [0., 0., 0., 1., 0., 0., 0.], "source": "native target dimensions"}], "query_gaps": []}
    owner = CollisionOwner("target", target_query, pose_path(), "current target", True)
    current = scene((owner,), background=background, native_geometry_materials={"pad_solid": "pad"})
    snapshot = current.to_execution_binding()
    json.dumps(snapshot, allow_nan=False)
    assert snapshot["owners"][0]["geometry"]["solids"][0]["dimensions"] == [.2, .3, .4]
    assert snapshot["coverage"]["known_packbits"]
    assert snapshot["native_geometry_materials"] == {"pad_solid": "pad"}
    target_query.independent_geometry["solids"][0]["dimensions"][0] = 9.
    owner.pose_path.poses[:, 0, 3] = 7.
    assert snapshot["owners"][0]["geometry"]["solids"][0]["dimensions"][0] == .2
    assert snapshot["owners"][0]["pose_path"]["poses"][0][0][3] == 0.


def test_binary_measured_entrance_collision_is_not_waived_by_future_unknown_width():
    hand = PlanningSceneTest().gripper()
    current = scene(gripper=hand, has_gripper=True,
                    background=SphereQuery(("background",), center=(.04, 0., 0.), radius=.002))
    args = inputs((0., 0.), sphere_rows=[[[.04, 0., 0., .005]]]*2)
    time = torch.tensor([[0., .5]], dtype=torch.float64)
    terms = current.components(*args, trajectory_time=time)
    assert terms["collision/robot/background"][0, 0] > 0
    assert terms["collision/robot/background"][0, 1] == 0
    assert terms["unknown/collision/robot/background"][0, 1] > 0
# END partial observations remain explicit while actual known conflicts still fail.

# BEGIN independent known/unknown reductions preserve exact hard evidence.
def test_future_unknown_spheres_cannot_create_roundoff_or_erase_known_penetration():
    """Exercise both scene query routes and evaluate, not just a sum helper.

    The synthetic values have only uncertain finger collisions. Different
    float32 reduction widths leave one ULP after total-minus-unknown; a much
    smaller real fixed-body collision must nevertheless survive independently.
    """
    generator = torch.Generator().manual_seed(1)
    values = torch.zeros((1, 1, 416), dtype=torch.float32)
    for _ in range(25):
        values[..., 272:] = torch.rand((1, 1, 144), generator=generator)*.001
    assert (values.sum(-1)-values[..., 272:].sum(-1)).item() == 2**-27

    class PrescribedQuery(SphereQuery):
        def __call__(self, spheres, *, key):
            return self.values.expand(*spheres.shape[:2], spheres.shape[-2])

    hand = PrismaticGripper(
        controller="binary", control_timestep_s=.002, joint_names=("finger",),
        current_positions=[.04], joint_limits=[[0., .04]], joint_bounds=None,
        coupling_matrix=np.empty((0, 1)), coupling_offset=[],
        sphere_indices=tuple(range(272, 416)),
        local_spheres=np.tile([0., 0., 0., .005], (144, 1)),
        sphere_joint_translation=np.tile([[[1.], [0.], [0.]]], (144, 1, 1)),
        self_collision_pairs=(), commands=[{"skill_id": "test", "event_id": "close",
            "occurrence": 0, "command": "close", "progress": 0.}], source="synthetic uncertain fingers")
    args = inputs((0., .5), sphere_rows=[[[-1., 0., 0., .005]]*416]*2, dtype=torch.float32)
    times = torch.tensor([[0., .5]], dtype=torch.float32)
    for target in ("background", "owner"):
        query = PrescribedQuery((target,))
        options = {"background": query} if target == "background" else {
            "owners": (CollisionOwner("owner", query, pose_path(), "synthetic owner", True),)}
        current = scene(materials=(None,)*416, gripper=hand, has_gripper=True, **options)
        key = "collision/robot/"+target
        for known_value in (0., 2**-40):
            query.values = values.clone()
            query.values[..., 0] = known_value
            query.values.requires_grad_()
            terms = current.components(*args, trajectory_time=times)
            assert terms[key][0, 1].item() == known_value
            assert terms[key][0, 0].item() == query.values.sum().item()
            assert terms["unknown/"+key][0, 0].item() == 0.
            assert terms["unknown/"+key][0, 1].item() == query.values[..., 272:].sum().item()
            assert current.evaluate(*args, trajectory_time=times)[0, 1].item() == known_value
            gradient = torch.autograd.grad(terms[key][0, 1], query.values)[0]
            assert torch.equal(gradient[..., :272], torch.ones_like(gradient[..., :272]))
            assert torch.equal(gradient[..., 272:], torch.zeros_like(gradient[..., 272:]))


def test_future_unknown_self_pairs_do_not_erase_small_known_pair_or_its_gradient():
    """Known palm guidance survives unknown fingers without becoming a veto."""
    hand = PrismaticGripper(
        controller="binary", control_timestep_s=.002, joint_names=("finger",),
        current_positions=[.04], joint_limits=[[0., .04]], joint_bounds=None,
        coupling_matrix=np.empty((0, 1)), coupling_offset=[], sphere_indices=(0, 1),
        local_spheres=[[0., 0., 0., 2**-15], [.5, 0., 0., .1]],
        sphere_joint_translation=[[[0.], [0.], [0.]], [[0.], [1.], [0.]]],
        self_collision_pairs=((0, 2), (1, 3)), commands=[{"skill_id": "test", "event_id": "close",
            "occurrence": 0, "command": "close", "progress": 0.}], source="synthetic fixed palm and moving finger")
    current = scene(materials=(None,)*4, gripper=hand, has_gripper=True)
    rows = [[0., 0., 0., 2**-15], [.5, .04, 0., .1],
            [2**-14-2**-30, 0., 0., 2**-15], [.5, .02, 0., .1]]
    args = inputs((0., .5), sphere_rows=[rows, rows], dtype=torch.float32)
    args[0].requires_grad_()
    times = torch.tensor([[0., .5]], dtype=torch.float32)
    # BEGIN proxy gradients belong solely to optimizer guidance.
    terms = current.components(*args, trajectory_time=times)
    assert "gripper/self" not in terms and "unknown/gripper/self" not in terms
    guidance = current.robot_self_collision_guidance(*args, None, None, trajectory_time=times)
    assert guidance[0, 0].item() > .1
    assert guidance[0, 1].item() == 2**-30
    assert current.evaluate(*args, trajectory_time=times)[0, 1].item() == 0.
    gradient = torch.autograd.grad(guidance[0, 1], args[0])[0]
    # END native source geometry supplies the independent hard self verdict.
    assert gradient[0, 1, 2, 0].item() == -1.
    assert gradient[0, 1, 3].count_nonzero().item() == 0
# END no tolerance, clipping, or unknown reduction can alter known components.


# BEGIN device command cadence is explicit across every controller family.
def test_gripper_device_period_is_validated_and_preserved_in_formal_scene():
    fixture = PlanningSceneTest()
    for invalid in (0., -.002, float("nan"), float("inf"), True):
        with fixture.assertRaisesRegex(ValueError, "positive native control timestep"):
            fixture.gripper(control_timestep_s=invalid)
    for hand in (fixture.gripper("binary"), fixture.gripper("position_tracking"), fixture.timed_gripper()):
        background = SphereQuery(("background",))
        background.source = "synthetic sphere query for clock metadata; no native geometry proof"
        current = scene(gripper=hand, has_gripper=True, background=background,
                        materials=(None,)*len(hand.sphere_indices))
        assert current.to_execution_binding()["gripper"]["control_timestep_s"] == .002
        assert current.describe()["gripper_control_timestep_s"] == .002
# END a tracking hand does not need a fabricated timed-reference factory.
