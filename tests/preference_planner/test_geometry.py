"""Signed rows, material provenance and original scene-domain semantics."""

from pathlib import Path
import sys
from types import SimpleNamespace

import coal
import numpy as np
import pytest
import torch

PROJECT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(PROJECT), str(PROJECT / "tests"), str(PROJECT / "tests/planning")]


from manipulation.preference_planner.geometry import GeometryConstraints, WitnessRow, field_point_value, native_point_value, occupied_material
from manipulation.cubic_constraints import CubicAffineMap
from manipulation.implicit_distance import ObservedField
from manipulation.observed_mapping import FrozenObservedField
from manipulation.observed_robot import DifferentiableRobotKinematics
from manipulation.pin_kinematics import PinKinematics
from manipulation.robot_geometry_check import NativePrimitive, RobotGeometryChecker
from manipulation.unified_solver import robot_tool_states
from test_native_scene_contract import make_scene


def test_native_point_derivative_uses_real_signed_shape_query():
    native = coal.Box(2., 1.6, 1.2)
    point = torch.tensor([.8, .13, .07], dtype=torch.float64, requires_grad=True)
    assert native_point_value(native, point).item() == pytest.approx(-.2)
    assert torch.autograd.gradcheck(lambda p: native_point_value(native, p), (point,),
                                    eps=1e-6, atol=1e-8, rtol=1e-6)
    exterior = point.detach().clone(); exterior[0] = 1.2
    assert native_point_value(native, exterior).item() == pytest.approx(.2)


def test_field_value_keeps_actual_interpolant_and_unknown_support():
    axis = np.arange(3, dtype=float)
    values = axis[:, None, None] + .3 * axis[None, :, None] + .2 * axis[None, None, :] - 1.
    field = ObservedField(values, np.ones_like(values, bool), origin=[0.] * 3, voxel_size=1.)
    point = torch.tensor([.4, .5, .6], dtype=torch.float64, requires_grad=True)
    value, supported, direction = field_point_value(field, point)
    assert supported and direction and value.item() == pytest.approx(-.33)
    assert torch.autograd.gradcheck(lambda p: field_point_value(field, p)[0], (point,),
                                    eps=1e-6, atol=1e-8, rtol=1e-6)
    outside = torch.tensor([-1., .2, .3], dtype=torch.float64, requires_grad=True)
    value, supported, direction = field_point_value(field, outside)
    assert not supported and not direction and value.item() == 0
    assert torch.count_nonzero(torch.autograd.grad(value, outside)[0]) == 0


def test_volume_material_distinguishes_constant_zero_from_zero_only_boundary():
    plane = np.broadcast_to(np.array([0., 1.])[:, None, None], (2, 2, 2)).copy()
    boundary = ObservedField(plane, np.ones_like(plane, bool), origin=[0.] * 3, voxel_size=1.)
    assert not occupied_material(boundary, [0., .4, .6])
    zero = ObservedField(np.zeros((2, 2, 2)), np.ones((2, 2, 2), bool),
                         origin=[0.] * 3, voxel_size=1.)
    assert occupied_material(zero, [.3, .4, .6])
    assert not occupied_material(zero, [.3, .4, .6], excluded_cells=(0,))


def computation(tmp_path, *, value=-1.):
    """Existing small robot fixture with one translated original native box."""
    scene = make_scene(tmp_path)
    first = np.eye(4); first[1, 3] = -.2
    second = np.eye(4); second[0, 3] = .09
    checker = RobotGeometryChecker(tmp_path / "robot.urdf", base_frame="base", geometries=(
        NativePrimitive("first", "left", "box", (.02, .02, .02), first, "synthetic robot unit fixture"),
        NativePrimitive("second", "base", "box", (.02, .02, .02), second, "synthetic robot unit fixture")))
    scene.robot_model = scene.robot_validator.checker = checker
    scene.kinematics = DifferentiableRobotKinematics(checker.model, frames=("base", "left"),
                                                     T_reference_base=np.eye(4))
    scene.background = FrozenObservedField(np.full((2, 2, 2), value), np.ones((2, 2, 2), bool),
        weights=np.ones((2, 2, 2)), origin=[.041, -.003, -.002], voxel_size=.006,
        name="background", source="analytic observed-field unit fixture")
    curve = CubicAffineMap([.04, -.02], [0., 0.], 4, .4)
    parameters = np.array([0., .5, 1.])
    result = SimpleNamespace(scene=scene, parameters=parameters, duration=1.2,
        max_host_rss_bytes=12 * 1024**3,
        domain=SimpleNamespace(_curve_map=curve), joint_names=("left_slide", "right_slide"),
        times=torch.tensor(parameters[None] * 1.2, dtype=torch.float64),
        progress=torch.tensor(parameters[None], dtype=torch.float64),
        problem=SimpleNamespace(tool_link="left", tool_transform=np.eye(4)),
        kinematics=PinKinematics(checker.model, frames=("base", "left"), T_reference_base=np.eye(4)))
    return result, np.tile([.04, -.02], (4, 1))


def original_evidence(base, controls, *, time=1.2):
    """One real complete scene query at a supplied physical curve time."""
    query_times = np.atleast_1d(time).astype(float)
    parameters = query_times / base.duration
    q, dq = [torch.as_tensor(base.domain._curve_map.evaluate(controls, parameters, order))[None]
             for order in (0, 1)]
    progress = torch.tensor(parameters[None], dtype=torch.float64)
    times = torch.tensor(query_times[None], dtype=torch.float64)
    _, _, _, _, tool, _ = robot_tool_states(problem=base.problem, scene=base.scene,
        kinematics=base.kinematics, q=q, dq=dq, joint_names=base.joint_names,
        progress=progress, times=times)
    evidence = []
    base.scene.components(None, *tool, progress, q, base.joint_names,
                          trajectory_time=times, geometry_evidence_out=evidence)
    return [{"time_s": float(moment), "side": 0, "progress": float(parameters[index]),
             "pairs": [row for row in evidence if row["state_index"][1] == index]}
            for index, moment in enumerate(query_times)]


@pytest.mark.parametrize("field_value", [-1., 0.])
def test_complete_intersection_generates_signed_material_rows_and_real_gradients(tmp_path, field_value):
    base, controls = computation(tmp_path, value=field_value)
    states = original_evidence(base, controls)
    assert any(row["status"] == "intersection" for row in states[0]["pairs"])
    geometry = GeometryConstraints(base)
    added = geometry.add_evidence(controls, states)
    assert added["added_rows"] > 0
    tensor = torch.tensor(controls, dtype=torch.float64, requires_grad=True)
    rows = geometry.rows(tensor)
    assert rows.max() > 0
    assert torch.autograd.gradcheck(geometry.rows, (tensor,), eps=1e-6, atol=1e-7, rtol=1e-5)
    before = len(geometry.witnesses)
    geometry.rows(tensor + .01)
    assert len(geometry.witnesses) == before
    assert geometry.add_evidence(controls, states)["added_rows"] == 0


def test_rows_keep_unknown_and_declared_internal_pairs_inactive(tmp_path):
    base, controls = computation(tmp_path)
    states = original_evidence(base, controls)
    geometry = GeometryConstraints(base)
    internal = [{**states[0], "pairs": [{**row, "status": "internal_assembly"}
                                       for row in states[0]["pairs"]]}]
    assert geometry.add_evidence(controls, internal)["added_rows"] == 0
    unknown = [{**states[0], "pairs": [{**row, "status": "unmodelled_motion"}
                                      for row in states[0]["pairs"]]}]
    assert geometry.add_evidence(controls, unknown)["added_rows"] == 0
    tensor = torch.tensor(controls, dtype=torch.float64, requires_grad=True)
    assert geometry.rows(tensor).shape == (0,)


def test_native_boundary_witness_is_refined_inside_actual_observed_volume(tmp_path):
    base, controls = computation(tmp_path, value=0.)
    geometry = GeometryConstraints(base)
    native = coal.Box(.02, .02, .02)
    native_pose, field_pose = np.eye(4), np.eye(4)
    native_pose[0, 3] = .04
    world = np.array([.05, 0., 0.])
    field = ObservedField(np.zeros((2, 2, 2)), np.ones((2, 2, 2), bool),
                         origin=[.045, -.005, -.005], voxel_size=.01)
    point = geometry._native_interior_witness(native, native_pose, field, field_pose,
                                               world, None, ())
    assert point is not None and occupied_material(field, point)
    assert native_point_value(native, torch.tensor(point - native_pose[:3, 3])).item() < 0


def test_released_material_row_differentiates_the_release_anchor(tmp_path):
    from manipulation.planning_problem import PosePath
    from manipulation.planning_scene import Attachment, CollisionOwner

    base, controls = computation(tmp_path)
    field = FrozenObservedField(-np.ones((2, 2, 2)), np.ones((2, 2, 2), bool),
        weights=np.ones((2, 2, 2)), origin=[-.001] * 3, voxel_size=.002,
        name="material", source="analytic observed-field unit fixture")
    base.scene.owners = (CollisionOwner("material", field,
        PosePath(np.array([0., 1.]), np.stack((np.eye(4), np.eye(4))), "declared path"),
        "observed material", moving=True,
        attachments=(Attachment((0., .5), [0., -.2, 0.], [1., 0., 0., 0.], "declared", "release"),)),)
    geometry = GeometryConstraints(base)
    geometry.witnesses.append(WitnessRow(time_s=1.2, progress=1., permission_progress=1.,
        first="robot/second", second="owner/material", component="collision/robot/material",
        source_status="clear", kind="material", source="owner/material", target="robot/second",
        point_local=(0., 0., 0.)))
    tensor = torch.tensor(controls, dtype=torch.float64, requires_grad=True)
    gradient, = torch.autograd.grad(geometry.rows(tensor).sum(), tensor)
    release_row = base.domain._curve_map.query(np.array([.5]))[0][0]
    np.testing.assert_allclose(gradient[:, 0].numpy(), release_row, rtol=0, atol=1e-12)
    assert torch.count_nonzero(gradient[:, 1]) == 0
    assert torch.autograd.gradcheck(geometry.rows, (tensor,), eps=1e-6, atol=1e-7, rtol=1e-5)


def test_permission_region_follows_current_feature_pose_instead_of_seed_world_pose(tmp_path):
    from manipulation.planning_problem import PosePath
    from manipulation.planning_scene import CollisionOwner, ContactRule

    base, controls = computation(tmp_path)
    observed = base.scene.background
    field = FrozenObservedField(observed.values, observed.observed, weights=observed.weights,
        origin=observed.origin, voxel_size=observed.voxel_size, name="material", source=observed.source)
    base.scene.owners = (CollisionOwner("material", field,
        PosePath(np.array([0., 1.]), np.stack((np.eye(4), np.eye(4))), "declared static frame"),
        "observed material", moving=False),)
    base.scene.contacts = (ContactRule(rule_id="surface", actor="robot", target_owner="material",
        robot_body="first", interval=(0., 1.), material_id="declared_face", required=False,
        feature_points=[[.004, -.204, -.004], [.004, -.196, -.004],
                        [.004, -.196, .004], [.004, -.204, .004]],
        feature_normal=[-1., 0., 0.], patch_origin=[.044, 0., 0.],
        patch_axes=np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]]),
        patch_half_size=[.004, .004], gap_bounds=(-.001, .001),
        normal_tolerance_rad=.1, modeling_allowance_m=.0005, geometry_error_m=.001,
        source="declared finite surface"),)
    geometry = GeometryConstraints(base)
    geometry.witnesses.append(WitnessRow(time_s=1.2, progress=1., permission_progress=1.,
        first="robot/first", second="owner/material", component="collision/robot/material",
        source_status="intersection", kind="material", source="owner/material", target="robot/first",
        point_local=(.044, 0., 0.)))
    unaligned = torch.tensor(controls, dtype=torch.float64, requires_grad=True)
    unaligned = unaligned + unaligned.new_tensor([.003, 0.])
    assert geometry.rows(unaligned).item() > 0
    assert geometry.last_evaluation["rows"][0]["mode"] == "active"
    aligned = torch.tensor(controls, dtype=torch.float64, requires_grad=True)
    assert geometry.rows(aligned).item() == 0
    assert geometry.last_evaluation["rows"][0]["mode"] == "permitted"
