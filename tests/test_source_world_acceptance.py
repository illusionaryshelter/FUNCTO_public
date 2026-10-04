"""CPU counterexamples for the single source-world planning acceptance path."""

# BEGIN native geometry owns acceptance while differentiable proxies guide search.
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from manipulation.collision_records import decode_array
from manipulation.planning import _ProgressMap, _scene_boundary_evaluate
from manipulation.planning_problem import PosePath
from manipulation.planning_scene import (
    Attachment, CollisionOwner, ObservedSpace, PlanningScene,
    RobotOwnerContactPermission,
)
from manipulation.robot_geometry_check import NativePrimitive, RobotGeometryChecker
from manipulation.robot_validation import RobotStateValidator
from manipulation.tsdf_geometry import tsdf_geometry_from_samples
from manipulation.voxel_layers import LayeredVoxelQuery


URDF = """<robot name="source_world_slide">
  <link name="base"/><link name="tool"/>
  <joint name="slide" type="prismatic"><parent link="base"/><child link="tool"/>
    <axis xyz="1 0 0"/><limit lower="0" upper="1" effort="10" velocity="1"/>
  </joint>
</robot>"""


class ProxyQuery:
    """Deliberately disagree with source geometry to expose a wrong veto."""

    def __init__(self, owner, geometry, bias=-2.):
        self.owner_ids = frozenset((owner,))
        self.source = "explicit synthetic query and independent geometry"
        self.independent_geometry = deepcopy(geometry)
        self.bias = bias
        self.calls = 0

    def clone_for_rollout(self):
        return ProxyQuery(next(iter(self.owner_ids)), self.independent_geometry, self.bias)

    def __call__(self, spheres, *, key):
        self.calls += 1
        return torch.relu(self.bias+spheres[..., 3]-spheres[..., 0])


def empty_geometry():
    return {"solids": [], "query_gaps": []}


def box_geometry(x=0., size=.03):
    return {"solids": [{"name": "native_target", "kind": "cuboid",
        "dimensions": [size]*3, "pose": [x, 0., 0., 1., 0., 0., 0.],
        "source": "explicit native target box"}], "query_gaps": []}


def observed_plane():
    # Samples define z=0 exactly inside one observed cube. Positive samples
    # do not become material; the lower half and the zero plane are retained.
    values = np.broadcast_to(np.array([-.05, .05])[:, None, None], (2, 2, 2)).copy()
    return tsdf_geometry_from_samples(values, np.ones_like(values, bool),
        origin=[-.05]*3, voxel_size=.1, name="observed_plane", source="measured signed samples")


def owner(query, *, attachments=(), moving=False):
    path = PosePath(np.array([0., 1.]), np.repeat(np.eye(4)[None], 2, axis=0),
                    source={"kind": "explicit fixed owner pose"})
    return CollisionOwner("target", query, path, "explicit target geometry", True,
        moving=moving, attachments=attachments,
        local_spheres=np.array([[0., 0., 0., .03]]) if moving else None)


@pytest.fixture
def scene_factory(tmp_path):
    path = tmp_path / "source_world.urdf"
    path.write_text(URDF)
    checker = RobotGeometryChecker(path, base_frame="base", geometries=[
        NativePrimitive("tool_box", "tool", "box", (.02, .02, .02), np.eye(4),
                        "complete native moving box")])
    gate = RobotStateValidator(checker, joint_names=("slide",), arm_joint_names=("slide",),
        finger_joint_names=(), pairs=(), named_velocity_limits={"slide": 1.},
        reference_frame="world", T_reference_base=np.eye(4), tool_frame="tool", T_frame_tcp=np.eye(4),
        fk_translation_tolerance_m=1e-8, fk_rotation_tolerance_rad=1e-8,
        source="complete one-joint native model")
    problem = SimpleNamespace(reference_frame="world", tool_link="tool", base_pose=np.eye(4),
        tool_transform=np.eye(4), start_q=np.array([.2]), joint_names=("slide",),
        start_progress=0., phase_count=2)

    def make(background=None, owners=(), permissions=(), bodies=()):
        # Entirely unknown observation coverage is allowed. It must neither
        # invent occupied volume nor hide an explicit known geometry failure.
        current = PlanningScene(background=(ProxyQuery("background", empty_geometry())
            if background is None else background), owners=owners,
            coverage=ObservedSpace(np.zeros((5, 5, 5), bool), origin=[-1.]*3,
                voxel_size=.5, observation_ids=["unknown synthetic view"], reference_frame="world"),
            reference_frame="world", robot_sphere_materials=(None,),
            robot_geometry_source="complete native one-box cover", has_gripper=False,
            robot_owner_permissions=permissions, robot_collision_bodies=bodies)
        current.bind_robot_validator(gate, problem)
        current.bind_source_scene()
        return current

    return make


def states(joints, progress=None, *, dtype=torch.float64):
    q = torch.tensor(joints, dtype=dtype)
    s = torch.full_like(q, .5) if progress is None else torch.tensor(progress, dtype=dtype)
    p = torch.stack((q, torch.zeros_like(q), torch.zeros_like(q)), -1)
    quaternion = q.new_tensor([1., 0., 0., 0.]).expand(*q.shape, 4).clone()
    spheres = torch.cat((p, torch.full((*q.shape, 1), .02, dtype=dtype)), -1).unsqueeze(-2)
    return spheres, p, quaternion, s, q[..., None], ("slide",)


def source_terms(current, args, **kwargs):
    return current.source_components(*args, trajectory_time=args[3], **kwargs)


def test_positive_proxy_is_guidance_and_unknown_space_does_not_veto_source_clearance(scene_factory):
    current = scene_factory(ProxyQuery("background", observed_plane(), bias=1.))
    args = states([[.2]])
    args[0].requires_grad_()
    guidance = current.evaluate(*args, trajectory_time=args[3])
    assert guidance.item() > 0
    guidance.sum().backward()
    assert args[0].grad[..., 0].item() == -1.
    terms = source_terms(current, args)
    assert terms["source_world_collision"].item() == 0.
    assert any(name.startswith("unknown/") and value.item() > 0 for name, value in terms.items())
    assert current.evaluate_source(*args, trajectory_time=args[3]).item() == 0.
    assert current.background.calls == 1


def test_zero_proxy_still_queries_tsdf_surface_and_negative_material(scene_factory):
    current = scene_factory(ProxyQuery("background", observed_plane()))
    args = states([[0.]])
    assert current.evaluate(*args, trajectory_time=args[3]).item() == 0.
    evidence = {}
    assert source_terms(current, args, diagnostics=evidence)["source_world_collision"].item() == 1.
    assert evidence["checked_states"] == evidence["failed_states"] == 1
    assert evidence["worst_state"]["scene"]["violations"]


def test_missing_required_geometry_is_not_allowed_unknown_space(scene_factory):
    incomplete = {"solids": [], "query_gaps": ["explicit_source_geometry_missing"]}
    current = scene_factory(ProxyQuery("background", incomplete))
    evidence = {}
    assert source_terms(current, states([[.2]]), diagnostics=evidence)["source_world_collision"].item() == 1.
    assert any(row["reason"] == "explicit_source_geometry_missing"
               for row in evidence["worst_state"]["scene"]["query_gaps"])


def test_invalid_padding_never_becomes_a_source_robot_state(scene_factory):
    current = scene_factory(ProxyQuery("background", observed_plane()))
    args = list(states([[.2, .2]]))
    args[4][0, 1, 0] = float("nan")
    time = args[3].clone()
    time[0, 1] = float("nan")
    evidence = {}
    result = current.evaluate_source(*args, trajectory_time=time,
        valid=torch.tensor([[True, False]]), diagnostics=evidence)
    assert result.tolist() == [[0., 0.]]
    assert evidence["checked_states"] == 1


def test_candidate_release_poses_do_not_rebind_execution_or_shared_scene(scene_factory):
    relation = Attachment((0., 1.), np.array([.1, 0., 0.]), np.array([1., 0., 0., 0.]),
                          "candidate_hold", "candidate_release")
    target = owner(ProxyQuery("target", box_geometry()), attachments=(relation,), moving=True)
    current = scene_factory(owners=(target,))
    cloned = current.clone_for_rollout()
    assert current._source_scene_checker is cloned._source_scene_checker
    first = source_terms(current, states([[.2, .3]], [[1., 1.5]]))["source_world_collision"]
    second = source_terms(cloned, states([[.6, .3]], [[1., 1.5]]))["source_world_collision"]
    assert first.tolist() == [[0., 1.]]
    assert second.tolist() == [[0., 0.]]
    assert current._source_scene_checker._scene_release_predictions == {}
    assert not hasattr(current.robot_validator.checker, "_scene_binding")


def test_source_boundary_preserves_arrival_permission_and_expiry(scene_factory):
    permission = RobotOwnerContactPermission(rule_id="arrived_target", target_owner="target",
        native_geometry_names=("tool_box",), sphere_indices=(0,), interval=(.5, .9),
        arrival_goal_id=("pose_events", 1),
        activation_event={"skill_id": "test", "event_id": "close", "occurrence": 0, "kind": "close"},
        source="explicit arrival policy")
    current = scene_factory(owners=(owner(ProxyQuery("target", box_geometry())),),
                            permissions=(permission,))
    args = states([[0., 0., 0.]], [[.499, .5, .9]], dtype=torch.float32)
    result = _scene_boundary_evaluate(current, SimpleNamespace(start_progress=0., phase_count=1),
        _ProgressMap(np.array([0., 1.]), np.array([0., 1.])), *args,
        trajectory_time=args[3], normalized_time=args[3], source_geometry=True)
    assert result.tolist() == [[1., 0., 1.]]


def test_source_geometry_consumes_the_exact_archived_whole_body_layer(scene_factory):
    shape = (21, 21, 21)
    region = {"pose": [.2, 0., 0., 1., 0., 0., 0.], "shape_xyz": list(shape),
        "voxel_size_m": .01, "numerical_support_m": .02,
        "measurement_allowance_m": .001, "measurement_evidence": "explicit test margin",
        "source": "complete observed synthetic fine region",
        "source_bits_hex": np.full(shape, 3, np.uint8).tobytes().hex(),
        "raw_free_packbits_hex": np.packbits(np.zeros(shape, bool)).tobytes().hex()}
    layered = LayeredVoxelQuery(ProxyQuery("background", box_geometry(.28, .04)),
        ProxyQuery("background", empty_geometry()), region)
    bodies = ({"body_id": "tool", "sphere_indices": [0], "geometry_names": ["tool_box"]},)
    current = scene_factory(layered, bodies=bodies)
    archived = {}
    residual = source_terms(current, states([[.2, .28]]), layer_selection_out=archived)["source_world_collision"]
    assert decode_array(archived["robot/background"]["body_fine"]).tolist() == [[[True], [False]]]
    assert residual.tolist() == [[0., 1.]]
# END known geometry, exact native states and declared owner policy determine acceptance.
