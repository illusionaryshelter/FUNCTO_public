"""CPU checks for shared official-robot geometry in both planning scopes."""

# BEGIN current native entities and observed fields replace obsolete sphere fixtures.
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from manipulation.planning_scene import GripperState
from manipulation.robot_geometry_check import NativePrimitive, RobotGeometryChecker
from test_native_scene_contract import make_scene


def arguments(q=None):
    """One named joint batch and physical clock serve original scene conditions."""
    if q is None:
        q = torch.tensor([[[.04, -.02]]], dtype=torch.float64, requires_grad=True)
    progress = q[..., 0] * 0
    position = q.new_zeros((*q.shape[:-1], 3))
    rotation = q.new_zeros((*q.shape[:-1], 4))
    rotation[..., 0] = 1.
    return (None, position, rotation, progress, q, ("left_slide", "right_slide"))


def query(scene, scope, q=None):
    """Return original component evidence for the declared geometry scope."""
    args = arguments(q)
    evidence = []
    terms = scene.components(*args, trajectory_time=args[3], geometry_scope=scope,
                             geometry_evidence_out=evidence)
    return terms, evidence


@pytest.mark.parametrize("gap,status", [(.001, "clear"), (-.001, "intersection"), (0., "unresolved")])
def test_same_native_pair_evidence_survives_both_geometry_scopes(tmp_path, gap, status):
    """Robot-only and full-scene queries consume identical native pair evidence."""
    scene = make_scene(tmp_path, gap=gap)
    original = None
    for scope in ("full", "robot"):
        terms, evidence = query(scene, scope)
        own = [row for row in evidence if row["component"] == "collision/robot/self"]
        assert len(own) == 1 and own[0]["status"] == status
        assert own[0]["geometry_entity_key"] == {"first": "robot/first", "second": "robot/second"}
        assert own[0]["distance_scope"] == "full_entities"
        current = (terms["collision/robot/self"].detach().item(),
                   terms["unresolved/robot/self"].detach().item(), own)
        if original is None:
            original = deepcopy(current)
        else:
            assert current == original
        assert scene.last_query_report["geometry_scope"] == scope
        assert scene.last_query_report["robot_geometry_source"] == "registered_native_model"
        if scope == "robot":
            assert set(terms) == {"collision/robot/self", "unresolved/robot/self"}
            assert scene.last_query_report["coverage_scope"] == "not_queried"


def test_robot_scope_returns_before_task_owner_poses_and_field_queries(tmp_path):
    scene = make_scene(tmp_path, occupied_background=True)
    with patch.object(scene, "_owner_poses", side_effect=AssertionError("Task owner pose outside robot scope")), \
         patch("manipulation.online_geometry.pair_residual", side_effect=AssertionError("Task field outside robot scope")), \
         patch.object(scene.robot_model, "query", wraps=scene.robot_model.query) as native:
        terms, _ = query(scene, "robot")
    assert native.call_count == 1
    assert native.call_args.kwargs["pairs"] is scene.robot_validator.required_pairs
    assert sum(terms.values()).item() == 0.


def test_full_scope_queries_self_once_before_owner_poses_and_keeps_field_failure(tmp_path):
    scene = make_scene(tmp_path, occupied_background=True)
    events = []
    original_query, original_owners = scene.robot_model.query, scene._owner_poses

    def native(**kwargs):
        events.append("self")
        return original_query(**kwargs)

    def owners(*args, **kwargs):
        events.append("owners")
        return original_owners(*args, **kwargs)

    with patch.object(scene.robot_model, "query", side_effect=native) as robot, \
         patch.object(scene, "_owner_poses", side_effect=owners):
        terms, evidence = query(scene, "full")
    assert events == ["self", "owners"] and robot.call_count == 1
    assert terms["collision/robot/self"].item() == 0.
    assert terms["collision/robot/background"].item() > 0.
    assert any(row["component"] == "collision/robot/background" and row["status"] == "intersection"
               for row in evidence)


def test_hand_controller_and_unknown_motion_retain_their_original_components(tmp_path, monkeypatch):
    scene = make_scene(tmp_path, gap=-.001)
    # The first state has measured coordinates; the second declares unknown
    # finger motion. Its dependent pair is unmodelled, not a measured collision.
    scene.gripper = SimpleNamespace(joint_names=("left_slide",),
        state=lambda progress, q, names, **kwargs: GripperState(q[..., :1], q[..., :1] * 0,
            progress * 0 + .125, torch.tensor([[False, True]])))
    monkeypatch.setattr(scene.robot_model, "geometry_depending_on", lambda names: ("first",))
    q = torch.tensor([[[.04, -.02], [.04, -.02]]], dtype=torch.float64)
    previous = None
    for scope in ("robot", "full"):
        terms, evidence = query(scene, scope, q=q)
        selected = {key: terms[key] for key in ("collision/robot/self", "unresolved/robot/self",
                                                "gripper/controller", "conditional/hand_geometry")}
        assert selected["collision/robot/self"][0, 0] > 0
        assert selected["collision/robot/self"][0, 1] == 0
        assert selected["conditional/hand_geometry"].tolist() == [[0., 1.]]
        assert selected["gripper/controller"].tolist() == [[.125, .125]]
        own = [row for row in evidence if row["component"] == "collision/robot/self"]
        assert [row["status"] for row in own] == ["intersection", "unmodelled_motion"]
        if previous is not None:
            for key in selected:
                torch.testing.assert_close(selected[key], previous[key], rtol=0, atol=0)
        previous = selected


def moving_scene(tmp_path):
    """Analytic prismatic FK matches the real native checker's URDF exactly."""
    scene = make_scene(tmp_path)
    local = np.eye(4)
    local[1, 3] = -.2
    checker = RobotGeometryChecker(tmp_path / "robot.urdf", base_frame="base", geometries=(
        NativePrimitive("first", "base", "box", (.02, .02, .02), np.eye(4), "analytic robot fixture"),
        NativePrimitive("second", "left", "box", (.02, .02, .02), local, "analytic robot fixture")))
    scene.robot_model = scene.robot_validator.checker = checker

    def poses(q, names):
        zero = q[..., 0] * 0
        quaternion = torch.stack((zero + 1, zero, zero, zero), -1)
        return {"base": (torch.stack((zero, zero, zero), -1), quaternion),
                "left": (torch.stack((q[..., names.index("left_slide")], zero + .2, zero), -1), quaternion)}

    scene.kinematics = SimpleNamespace(poses=poses)
    return scene


@pytest.mark.parametrize("scope", ["robot", "full"])
def test_native_self_scalar_and_witness_gradient_match_finite_difference(tmp_path, scope):
    scene = moving_scene(tmp_path)
    q = torch.tensor([[[.015, -.02]]], dtype=torch.float64, requires_grad=True)
    terms, _ = query(scene, scope, q)
    value = terms["collision/robot/self"].sum()
    derivative = torch.autograd.grad(value, q)[0]
    # The 2 cm boxes overlap by 5 mm along x. A positive left_slide motion
    # decreases that overlap; the independent right_slide does not move either.
    assert value.item() == pytest.approx(.005, abs=1e-10)
    np.testing.assert_allclose(derivative.detach().numpy(), [[[-1., 0.]]], atol=1e-10)
    step = torch.zeros_like(q)
    step[..., 0] = 1e-6
    plus = query(scene, scope, q=q.detach() + step)[0]["collision/robot/self"]
    minus = query(scene, scope, q=q.detach() - step)[0]["collision/robot/self"]
    torch.testing.assert_close(derivative[..., 0], (plus - minus) / 2e-6, rtol=1e-6, atol=1e-9)
# END no task model, source tolerance or collision policy changes across scopes.
