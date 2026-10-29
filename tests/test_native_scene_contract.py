"""Scene integration of native model policy and observed task fields."""

# BEGIN use real native queries with a small analytic FK fixture.
from types import SimpleNamespace

from pair_evidence_fixtures import PairBatch, records

import numpy as np
import pytest
import torch

from manipulation.online_geometry import OnlinePlanningScene
from manipulation.observed_mapping import FrozenObservedField
from manipulation.planning_scene import ObservedSpace
from manipulation.planning_scene import CollisionOwner
from manipulation.planning_problem import PosePath
from manipulation.robot_geometry_check import NativePrimitive, PairQuery, RobotGeometryChecker
from test_robot_geometry_check import URDF


def make_scene(tmp_path, *, gap=.001, fixed=(), occupied_background=False, policy="required",
               second_pose=None):
    path = tmp_path / "robot.urdf"
    path.write_text(URDF)
    second = np.eye(4)
    second[0, 3] = .02 + gap
    if second_pose is not None:
        second = second_pose
    checker = RobotGeometryChecker(path, base_frame="base", geometries=(
        NativePrimitive("first", "base", "box", (.02, .02, .02), np.eye(4), "source"),
        NativePrimitive("second", "base", "box", (.02, .02, .02), second, "source")))
    background = FrozenObservedField(-np.ones((2, 2, 2)),
        np.full((2, 2, 2), occupied_background), weights=np.ones((2, 2, 2)),
        origin=[-.005, -.005, -.005], voxel_size=.01, name="background", source="RGB-D")
    space = ObservedSpace(np.zeros((2, 2, 2), bool), origin=[0, 0, 0], voxel_size=.01,
        observation_ids=("view",), reference_frame="world")
    scene = OnlinePlanningScene(background=background, owners=(), coverage=space,
        reference_frame="world", robot_model=checker, fixed_geometry_names=fixed, tool_geometry_names=(),
        has_gripper=False, unknown_policy={"conditional_regions": [], "outside_regions": "forbid"})
    pair = PairQuery("first", "second", policy, "source native pair")
    scene.robot_pairs = (pair,)
    # BEGIN the fixture exposes the same preselected tuple as RobotStateValidator.
    scene.robot_validator = SimpleNamespace(checker=checker, T_reference_base=np.eye(4),
        required_pairs=tuple(item for item in scene.robot_pairs if item.policy == "required"))
    # END the full scene policy remains separately available for serialization.

    class BaseKinematics:
        def poses(self, q, names, *, fk_session=None):
            zero = q[..., :1] * 0
            return {"base": (zero.expand(*q.shape[:-1], 3),
                             torch.cat((zero + 1, zero.expand(*q.shape[:-1], 3)), -1))}

    scene.kinematics = BaseKinematics()
    return scene


def evaluate(scene):
    """Run the same original entity conditions for every scene consumer."""
    q = torch.tensor([[[.04, -.02]]], dtype=torch.float64, requires_grad=True)
    return scene.components(None, torch.zeros((1, 1, 3), dtype=torch.float64),
        torch.tensor([[[1., 0., 0., 0.]]], dtype=torch.float64),
        torch.zeros((1, 1), dtype=torch.float64), q, ("left_slide", "right_slide"),
        trajectory_time=torch.zeros((1, 1)))


# BEGIN registered attachments keep their original scene-owned validation.
@pytest.mark.parametrize("change", [None, "carrier", "transform", "start", "release"])
def test_registered_attachment_matches_the_task_declaration(tmp_path, change):
    from manipulation.planning_scene import Attachment

    world = make_scene(tmp_path)
    transform = np.eye(4)
    transform[0, 3] = -.1
    condition = dict(kind="declared_attachment", condition_id="relation",
        carrier_id="material", progress=0., T_carrier_eef=transform)
    release = dict(kind="predicted_release", condition_id="release", progress=1.)
    field = FrozenObservedField(np.zeros((2, 2, 2)), np.zeros((2, 2, 2), bool),
        weights=np.zeros((2, 2, 2)), origin=[0.] * 3, voxel_size=.01,
        name="material", source="empty observed support")
    world.owners = (CollisionOwner("material", field,
        PosePath(np.array([0., 1.]), np.stack((np.eye(4), np.eye(4))), "declared path"),
        "observed field", attachments=(Attachment((0., 1.), [.1, 0., 0.],
            [1., 0., 0., 0.], "relation", "release"),)),)
    if change == "carrier":
        condition["carrier_id"] = "another_material"
    elif change == "transform":
        condition["T_carrier_eef"][0, 3] = -.2
    elif change == "start":
        condition["progress"] = .1
    elif change == "release":
        release["progress"] = .9
    problem = SimpleNamespace(reference_frame="world", conditions=(condition, release),
        pose_events=(), events=())
    if change is None:
        world.validate_problem(problem)
    else:
        with pytest.raises(ValueError):
            world.validate_problem(problem)
# END execution consumes the accepted scene instead of repeating these checks.


def test_native_structure_gap_is_independent_of_task_acceptance_margin(tmp_path):
    scene = make_scene(tmp_path)
    costs = evaluate(scene)
    assert costs["collision/robot/self"].item() == 0
    assert costs["unresolved/robot/self"].item() == 0
    assert scene.last_query_report["status_counts"]["separated"] == 1
    assert not any("coverage" in key or "unknown" in key for key in costs)


# BEGIN original task-field queries retain nonpenetration semantics.
@pytest.mark.parametrize("gap,status", [(1e-3, "clear"), (1e-7, "clear"),
                                      (0., "unresolved"), (-1e-3, "intersection")])
def test_task_field_gap_and_boundary_retain_original_nonpenetration_meaning(tmp_path, gap, status):
    from manipulation.implicit_distance import ObservedField
    scene = make_scene(tmp_path)
    scene.background = ObservedField(-np.ones((2, 2, 2)), np.ones((2, 2, 2), bool),
                                     origin=[-.005, .01 + gap, -.005], voxel_size=.01)
    costs = evaluate(scene)
    assert (sum(costs.values()).item() > 0) == (status == "intersection")
    assert scene.last_query_report["status_counts"].get(status, 0) > 0
    assert scene.last_query_report["collision_rule"] == "nonpenetration"
    if status == "unresolved":
        assert any(row["status"] == "unresolved"
                   for row in scene.last_query_report["nonclear_pairs"])
# END numerical uncertainty is visible without creating a clearance gate.


@pytest.mark.parametrize("gap,collision,unresolved", [(-.001, True, False), (0., False, True)])
def test_self_penetration_and_numerical_boundary_remain_distinct(tmp_path, gap, collision, unresolved):
    costs = evaluate(make_scene(tmp_path, gap=gap))
    assert (costs["collision/robot/self"].item() > 0) == collision
    assert (costs["unresolved/robot/self"].item() > 0) == unresolved


def test_mechanical_contact_policy_does_not_become_a_required_pair(tmp_path):
    costs = evaluate(make_scene(tmp_path, gap=-.001, policy="mechanical_contact"))
    assert costs["collision/robot/self"].item() == 0
    assert costs["unresolved/robot/self"].item() == 0


def test_fixed_installation_role_is_explicit_even_with_constant_link_fk(tmp_path):
    robot = make_scene(tmp_path, occupied_background=True)
    assert evaluate(robot)["collision/robot/background"].item() > 0
    fixed = make_scene(tmp_path, occupied_background=True, fixed=("first", "second"))
    assert "collision/robot/background" not in evaluate(fixed)


def test_scene_record_retains_model_policy_and_calibration(tmp_path):
    scene = make_scene(tmp_path, fixed=("second",))
    saved = scene.to_execution_binding()
    assert saved["fixed_geometry_names"] == ["second"]
    assert saved["robot_model"]["base_frame"] == "base"
    assert saved["robot_model"]["T_reference_base"] == np.eye(4).tolist()
    assert saved["robot_model"]["pairs"][0]["policy"] == "required"
    assert saved["robot_model"]["geometries"][1]["T_frame_geometry"][0][3] == .021
    assert "robot_bodies" not in saved


# BEGIN displayed collision locations retain each component's actual query clock.
def test_nonclear_diagnostics_keep_first_and_maximum_per_component(tmp_path):
    scene = make_scene(tmp_path, occupied_background=True)
    local = np.eye(4)
    local[:3, 3] = [-.04, -.2, 0.]
    checker = RobotGeometryChecker(tmp_path / "robot.urdf", base_frame="base", geometries=(
        NativePrimitive("first", "base", "box", (.02, .02, .02), np.eye(4), "source"),
        NativePrimitive("second", "left", "box", (.02, .02, .02), local, "source")))
    scene.robot_model = scene.robot_validator.checker = checker

    def poses(q, names, *, fk_session=None):
        zero = q[..., :1] * 0
        quaternion = torch.cat((zero + 1, zero.expand(*q.shape[:-1], 3)), -1)
        return {"base": (zero.expand(*q.shape[:-1], 3), quaternion),
                "left": (torch.cat((q[..., :1], zero + .2, zero), -1), quaternion)}

    scene.kinematics = SimpleNamespace(poses=poses)
    q = torch.tensor([[[.059, -.02]] * 39 + [[.045, -.02]]], dtype=torch.float64)
    progress = torch.linspace(.1, .9, 40, dtype=torch.float64)[None]
    permission = progress + .05
    times = torch.linspace(2., 6., 40, dtype=torch.float64)[None]
    position, quaternion = poses(q, ())['base']
    complete_evidence = PairBatch()
    costs = scene.components(None, position, quaternion, progress, q,
        ("left_slide", "right_slide"), trajectory_time=times,
        contact_permission_progress=permission, geometry_evidence_out=complete_evidence)
    complete_evidence = records(complete_evidence)
    report = scene.last_query_report
    assert report['pair_queries'] == len(complete_evidence) == 120
    rows = report['nonclear_pairs']
    assert {row['component'] for row in rows} == {
        'collision/robot/self', 'collision/robot/background'}
    self_rows = [row for row in rows if row['component'] == 'collision/robot/self']
    assert len(self_rows) == 2
    first, maximum = self_rows
    assert first['selection'] == ['first'] and first['state_index'] == [0, 0]
    assert maximum['selection'] == ['maximum_pair_residual']
    assert maximum['state_index'] == [0, 39]
    assert maximum['pair_residual_value'] == pytest.approx(costs['collision/robot/self'][0, 39])
    for row in rows:
        index = tuple(row['state_index'])
        assert row['time_s'] == times[index].item()
        assert row['progress'] == progress[index].item()
        assert row['permission_progress'] == permission[index].item()
    field_rows = [row for row in rows if row['component'] == 'collision/robot/background']
    assert len(field_rows) == 1
    assert field_rows[0]['selection'] == ['first', 'maximum_pair_residual']
# END native self collisions cannot consume the observed-field display capacity.


def test_primitive_fixed_rotation_and_translation_are_applied_once(tmp_path, monkeypatch):
    from scipy.spatial.transform import Rotation
    from manipulation.online_geometry import _matrices
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler("xyz", [.3, -.7, .2]).as_matrix()
    transform[:3, 3] = [.2, .3, .4]
    scene = make_scene(tmp_path, second_pose=transform)
    consumed = []

    def inspect(first, first_pose, first_matrix, second, second_pose, second_matrix, **kwargs):
        consumed.append(first_matrix)
        np.testing.assert_allclose(_matrices(first_pose), first_matrix, atol=1e-14)
        return first_pose[0].sum() * 0, {"status": "clear"}

    monkeypatch.setattr("manipulation.online_geometry.pair_residual", inspect)
    evaluate(scene)
    reference = scene.robot_model.query(joint_names=("left_slide", "right_slide"),
        joint_positions=(.04, -.02), T_reference_base=np.eye(4), pairs=())
    np.testing.assert_allclose(consumed[1], reference["geometry_poses"]["second"], atol=1e-14)


def test_fixed_installation_still_constrains_a_moving_observed_owner(tmp_path):
    scene = make_scene(tmp_path, fixed=("first", "second"))
    field = FrozenObservedField(-np.ones((2, 2, 2)), np.ones((2, 2, 2), bool),
        weights=np.ones((2, 2, 2)), origin=[-.005] * 3, voxel_size=.01,
        name="moving", source="observed task entity")
    scene.owners = (CollisionOwner("moving", field,
        PosePath(np.array([0., 1.]), np.stack((np.eye(4), np.eye(4))), "declared path"),
        "observed task entity", moving=True),)
    costs = evaluate(scene)
    assert costs["collision/robot/moving"].item() > 0
    rows = scene.last_query_report["nonclear_pairs"]
    assert any(row["first"] == "first" and row["second"] == "moving" for row in rows)
# END no rendered robot pixels participate in the native model's validity.


# BEGIN performance changes retain policy identity and every zero derivative.
@pytest.mark.parametrize("policy", ["required", "excluded", "mechanical_contact"])
def test_scene_queries_only_required_pairs_and_serializes_complete_policy(tmp_path, monkeypatch, policy):
    scene = make_scene(tmp_path, policy=policy)
    # BEGIN inspect the numeric pair request consumed by this scene query.
    query = scene.robot_model.query_pair_batches
    submitted = []

    def inspect(**kwargs):
        submitted.append(kwargs["pairs"])
        return query(**kwargs)

    monkeypatch.setattr(scene.robot_model, "query_pair_batches", inspect)
    evaluate(scene)
    assert submitted == [scene.robot_validator.required_pairs]
    assert len(submitted[0]) == int(policy == "required")
    assert scene.to_execution_binding()["robot_model"]["pairs"][0]["policy"] == policy
    # END compact execution preserves the complete serialized pair policy.


def test_scene_shares_batch_zero_across_pairs_and_retains_all_position_gradients(tmp_path, monkeypatch):
    from manipulation.planning_scene import Attachment
    import manipulation.online_geometry as geometry

    scene = make_scene(tmp_path)
    field = FrozenObservedField(np.zeros((2, 2, 2)), np.zeros((2, 2, 2), bool),
        weights=np.zeros((2, 2, 2)), origin=[0.] * 3, voxel_size=.01,
        name="entity", source="empty observed support")
    scene.owners = (CollisionOwner("entity", field,
        PosePath(np.array([0., 1.]), np.stack((np.eye(4), np.eye(4))), "declared path"),
        "observed entity", attachments=(Attachment((0., 1.), [.1, 0., 0.],
            [1., 0., 0., 0.], "declared_attachment", None),)),)
    scene.tool_geometry_names = ("first",)
    q = torch.tensor([[[.04, -.02], [.05, -.03]]], dtype=torch.float64, requires_grad=True)
    tool_position = torch.zeros((1, 2, 3), dtype=torch.float64, requires_grad=True)
    robot_position = torch.zeros((1, 2, 3), dtype=torch.float64, requires_grad=True)
    rotation = torch.tensor([[[1., 0., 0., 0.], [1., 0., 0., 0.]]],
                            dtype=torch.float64, requires_grad=True)
    scene.kinematics = SimpleNamespace(poses=lambda *args, **kwargs: {"base": (robot_position, rotation)})
    pair_query = geometry.pair_residual
    shared_zeros = []

    def inspect(*args, zero, **kwargs):
        shared_zeros.append(zero)
        return pair_query(*args, zero=zero, **kwargs)

    monkeypatch.setattr(geometry, "pair_residual", inspect)
    progress = torch.tensor([[.25, .75]], dtype=torch.float64)
    terms = scene.components(None, tool_position, rotation, progress, q,
        ("left_slide", "right_slide"), trajectory_time=progress)
    total = sum(value.sum() for value in terms.values())
    assert total.item() == 0
    gradients = torch.autograd.grad(total, (q, tool_position, robot_position, rotation))
    assert all(torch.count_nonzero(value) == 0 for value in gradients)
    assert len(shared_zeros) > progress.numel()
    assert len({value.untyped_storage().data_ptr() for value in shared_zeros}) == 1
    assert all(value.requires_grad for value in shared_zeros)
    assert scene.last_query_report["internal_assembly_pair_count"] == progress.numel()
# END shared storage leaves explicit internal membership and model policy intact.
