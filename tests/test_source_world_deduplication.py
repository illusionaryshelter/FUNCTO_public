"""CPU differential evidence for exact, call-local source-world reuse."""

# BEGIN original-order singleton queries remain an independent no-reuse oracle.
from copy import deepcopy
import sys
from types import MethodType, SimpleNamespace

import numpy as np
import pytest
import torch

from manipulation.planning import _ProgressMap, _scene_boundary_evaluate
from manipulation.planning_scene import Attachment, PlanningScene, RobotOwnerContactPermission
from manipulation.voxel_layers import LayeredVoxelQuery
from test_source_world_acceptance import (
    ProxyQuery, box_geometry, empty_geometry, observed_plane, owner, scene_factory, states,
)


def singleton_reference(scene, joints, names, times, progress, *, geometry_query,
                        robot_owner_permission_progress, valid=None, diagnostics=None):
    """Preserve the original row order without permitting within-call reuse.

    Owner motion and layer choices have already been computed for the whole
    candidate. Slice those exact inputs, rather than recomputing release FK
    or choosing a layer independently for the one-state reference.
    """
    result = torch.zeros_like(times)
    for batch in range(times.shape[0]):
        for step in range(times.shape[1]):
            one = lambda value: value[batch:batch+1, step:step+1]
            query = {
                "nominal_owner_poses": {name: tuple(one(value) for value in pose)
                    for name, pose in geometry_query["nominal_owner_poses"].items()},
                "layer_selection": {name: SimpleNamespace(body_ids=selection.body_ids,
                    **{field: one(getattr(selection, field)) for field in (
                        "body_fine", "influence_aabb", "index_lower", "index_upper")})
                    for name, selection in geometry_query["layer_selection"].items()},
            }
            result[batch, step] = PlanningScene.check_world_geometry(
                scene, one(joints), names, one(times), one(progress), geometry_query=query,
                robot_owner_permission_progress=one(robot_owner_permission_progress),
                valid=None if valid is None else one(valid), diagnostics=diagnostics)[0, 0]
    return result


def recorded_query(joints, names, arguments):
    """Keep consumed query values and raw typed layer records for comparison."""
    return {"joints": joints.tolist(), "names": list(names),
        **{name: float(arguments[name]) for name in (
            "time_s", "progress", "robot_owner_permission_progress")},
        "owners": {name: pose.tolist() for name, pose in arguments["nominal_owner_poses"].items()},
        "layers": deepcopy(arguments["layer_selection"])}


def recording_scene(report_factory=None):
    """Exercise the public batch method without native geometry cost."""
    calls = []

    def check(joints, names, **arguments):
        query = recorded_query(joints, names, arguments)
        calls.append(query)
        if report_factory is not None:
            return report_factory(joints, query)
        return {"residual": 1., "unmodelled_geometry_names": [],
            "scene": {"query_gaps": [], "pairs": [{"signed_distance_m": -float(joints[0])}]},
            "consumed_query": query}

    current = SimpleNamespace(robot_validator=SimpleNamespace(check_planning_world=check),
        _source_scene_checker=object(), _self_collision_owner_ids=frozenset(), gripper=None)
    current.check_world_geometry = MethodType(PlanningScene.check_world_geometry, current)
    return current, calls


def batch_inputs(dtype=torch.float64):
    """Two batches, two owners and two layers detect omitted tail fields."""
    shape = (2, 3)
    joints = torch.full((*shape, 2), .25, dtype=dtype)
    times = torch.full(shape, .5, dtype=dtype)
    progress = torch.full(shape, .75, dtype=dtype)
    permission = progress.clone()
    poses, layers = {}, {}
    for name in ("first", "second"):
        position = torch.full((*shape, 3), .125, dtype=dtype)
        quaternion = torch.tensor([1., 0., 0., 0.], dtype=dtype).expand(*shape, 4).clone()
        poses[name] = (position, quaternion)
        layers["robot/"+name] = SimpleNamespace(body_ids=("first_body", "second_body"),
            body_fine=torch.zeros((*shape, 2), dtype=torch.bool),
            influence_aabb=torch.tensor([[-1.]*3, [1.]*3], dtype=torch.float64).expand(*shape, 2, 2, 3).clone(),
            index_lower=torch.zeros((*shape, 2, 3), dtype=torch.int64),
            index_upper=torch.full((*shape, 2, 3), 4, dtype=torch.int64))
    return (joints, ("first_joint", "second_joint"), times, progress), {
        "geometry_query": {"nominal_owner_poses": poses, "layer_selection": layers},
        "robot_owner_permission_progress": permission}


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
@pytest.mark.parametrize("changed", (
    "joints", "time", "progress", "permission", "owner_position", "owner_quaternion",
    "quaternion_sign", "signed_zero", "body_fine", "aabb", "index_lower", "index_upper",
))
def test_every_consumed_field_distinguishes_states_without_rounding(dtype, changed):
    args, kwargs = batch_inputs(dtype)
    query = kwargs["geometry_query"]
    layer = query["layer_selection"]["robot/second"]
    position, quaternion = query["nominal_owner_poses"]["second"]
    fields = {"joints": args[0], "time": args[2], "progress": args[3],
        "permission": kwargs["robot_owner_permission_progress"], "owner_position": position,
        "owner_quaternion": quaternion, "body_fine": layer.body_fine,
        "aabb": layer.influence_aabb, "index_lower": layer.index_lower, "index_upper": layer.index_upper}
    if changed == "quaternion_sign":
        quaternion[1, 0] *= -1  # Geometrically equal is still a different typed input.
    elif changed == "signed_zero":
        quaternion[1, 0, 3] = -0.
    else:
        value = fields[changed][1, 0]
        # Select the last scalar so testing never relies only on field prefixes.
        index = (1, 0)+tuple(size-1 for size in value.shape)
        array = fields[changed]
        array[index] = (not bool(array[index]) if array.dtype == torch.bool else
            array[index]+1 if not array.is_floating_point() else
            torch.nextafter(array[index], array.new_tensor(float("inf"))))
    current, calls = recording_scene()
    expected_scene, expected_calls = recording_scene()
    actual_diagnostics, expected_diagnostics = {}, {}
    actual = current.check_world_geometry(*args, **kwargs, diagnostics=actual_diagnostics)
    expected = singleton_reference(expected_scene, *args, **kwargs, diagnostics=expected_diagnostics)
    torch.testing.assert_close(actual, expected, rtol=0., atol=0.)
    assert actual_diagnostics == expected_diagnostics
    assert len(calls) == 2 and len(expected_calls) == 6
    assert calls == [expected_calls[0], expected_calls[3]]
    assert actual_diagnostics["checked_states"] == actual_diagnostics["failed_states"] == 6


@pytest.mark.parametrize("all_invalid", (False, True))
def test_padding_is_filtered_before_identity_and_pose_conversion(all_invalid):
    args, kwargs = batch_inputs()
    valid = torch.zeros((2, 3), dtype=torch.bool)
    if not all_invalid:
        valid[0, 0] = valid[1, 1] = True
    for tensor in (args[0], args[2], args[3], kwargs["robot_owner_permission_progress"]):
        tensor[~valid] = float("nan")
    for position, quaternion in kwargs["geometry_query"]["nominal_owner_poses"].values():
        position[~valid] = float("nan")
        quaternion[~valid] = 0.  # Rotation.from_quat must never consume padding.
    for selection in kwargs["geometry_query"]["layer_selection"].values():
        selection.influence_aabb[~valid] = float("nan")
    current, calls = recording_scene()
    reference, _ = recording_scene()
    actual_diagnostics, expected_diagnostics = {}, {}
    actual = current.check_world_geometry(*args, **kwargs, valid=valid, diagnostics=actual_diagnostics)
    expected = singleton_reference(reference, *args, **kwargs, valid=valid, diagnostics=expected_diagnostics)
    torch.testing.assert_close(actual, expected, rtol=0., atol=0.)
    assert actual_diagnostics == expected_diagnostics
    assert len(calls) == (0 if all_invalid else 1)
    assert actual_diagnostics["checked_states"] == int(valid.sum())
    assert not bool(actual[~valid].any())


@pytest.mark.parametrize("prior_worst", (False, True))
def test_logical_counts_unmodelled_union_and_first_tied_worst_survive_reuse(prior_worst):
    def report(joints, query):
        identity = int(joints[0])
        return {"residual": float(identity > 0), "unmodelled_geometry_names": [f"part_{identity}"],
            "scene": {"query_gaps": [{"reason": "synthetic_gap"}] if identity else [],
                      "pairs": [{"signed_distance_m": -.02}] if identity else []},
            "consumed_query": query}

    args, kwargs = batch_inputs()
    args[0][..., 0] = torch.tensor([[2., 1., 2.], [0., 1., 0.]])
    current, calls = recording_scene(report)
    reference, _ = recording_scene(report)
    prior = {"checked_states": 7, "failed_states": 3, "unmodelled_geometry_names": ["earlier_part"]}
    if prior_worst:
        # A same-rank earlier chunk must remain the witness, even though this
        # batch has failures with different physical coordinates.
        older = deepcopy(args)
        older[0][..., 0] = 1.
        singleton_reference(reference, *older, **kwargs, diagnostics=prior)
        prior["worst_state"]["trajectory_time_s"] = 9.
    initial_count, initial_failed = prior["checked_states"], prior["failed_states"]
    actual_diagnostics, expected_diagnostics = deepcopy(prior), deepcopy(prior)
    actual = current.check_world_geometry(*args, **kwargs, diagnostics=actual_diagnostics)
    expected = singleton_reference(reference, *args, **kwargs, diagnostics=expected_diagnostics)
    torch.testing.assert_close(actual, expected, rtol=0., atol=0.)
    assert actual_diagnostics == expected_diagnostics
    assert len(calls) == 3
    assert [row["joints"][0] for row in calls] == [2., 1., 0.]
    assert actual_diagnostics["checked_states"] == initial_count+6
    assert actual_diagnostics["failed_states"] == initial_failed+4
    assert actual_diagnostics["unmodelled_geometry_names"] == ["earlier_part", "part_0", "part_1", "part_2"]
    assert actual_diagnostics["worst_state"]["trajectory_time_s"] == (9. if prior_worst else .5)
    assert actual_diagnostics["worst_state"]["planning_joint_positions"][0] == (1. if prior_worst else 2.)


def test_first_original_error_is_preserved_without_partial_diagnostic_commit():
    def report(joints, query):
        if joints[0] < 0:
            raise ValueError("first invalid joint state")
        return {"residual": 0., "unmodelled_geometry_names": [],
                "scene": {"pairs": [], "query_gaps": []}}

    args, kwargs = batch_inputs()
    args[0][0, 2, 0] = -1.
    kwargs["geometry_query"]["nominal_owner_poses"]["second"][1][1, 0] = 0.
    current, calls = recording_scene(report)
    diagnostics = {"checked_states": 9, "failed_states": 2, "unrelated": "retained"}
    before = deepcopy(diagnostics)
    with pytest.raises(ValueError, match="^first invalid joint state$"):
        current.check_world_geometry(*args, **kwargs, diagnostics=diagnostics)
    assert [query["joints"][0] for query in calls] == [.25, -1.]
    assert diagnostics == before


def native_differential(current, operation, monkeypatch):
    """Compare every native report, plus the aggregate/public query records."""
    native = current.robot_validator.check_planning_world
    records = []

    def capture(joints, names, **arguments):
        report = native(joints, names, **arguments)
        records.append((recorded_query(joints, names, arguments), deepcopy(report)))
        return report

    monkeypatch.setattr(current.robot_validator, "check_planning_world", capture)
    actual_diagnostics, actual_layers, actual_components = {}, {}, {}
    actual = operation(current, actual_diagnostics, actual_layers, actual_components)
    actual_records = records[:]
    records.clear()
    # Override only the batch dispatcher. Full-candidate source preparation,
    # release witnesses, three-mode expansion and layer choices stay native.
    with monkeypatch.context() as context:
        context.setattr(current, "check_world_geometry", MethodType(singleton_reference, current))
        expected_diagnostics, expected_layers, expected_components = {}, {}, {}
        expected = operation(current, expected_diagnostics, expected_layers, expected_components)
    torch.testing.assert_close(actual, expected, rtol=0., atol=0.)
    assert actual_diagnostics == expected_diagnostics
    assert actual_layers == expected_layers
    assert actual_components.keys() == expected_components.keys()
    for name in actual_components:
        torch.testing.assert_close(actual_components[name], expected_components[name], rtol=0., atol=0.)
    assert 0 < len(actual_records) < len(records)
    # Compare complete distance/contact/permission/query-gap evidence for
    # every logical reference row, not only the chosen worst report.
    assert all(item in actual_records for item in records)
    assert all(item in records for item in actual_records)
    return actual, actual_diagnostics, actual_layers


def source_operation(args):
    return lambda current, diagnostics, layers, components: current.evaluate_source(
        *args, trajectory_time=args[3], diagnostics=diagnostics,
        layer_selection_out=layers, components_out=components)


def test_real_tsdf_native_reports_equal_original_order_checks(scene_factory, monkeypatch):
    current = scene_factory(ProxyQuery("background", observed_plane()))
    args = states([[.2, 0., .2, 0.]])
    residual, diagnostics, _ = native_differential(current, source_operation(args), monkeypatch)
    assert residual.tolist() == [[0., 1., 0., 1.]]
    assert diagnostics["checked_states"] == 4 and diagnostics["failed_states"] == 2


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
@pytest.mark.parametrize("plateau", (False, True))
def test_real_boundary_modes_preserve_arrival_expiry_and_plateau(scene_factory, monkeypatch, dtype, plateau):
    permission = RobotOwnerContactPermission(rule_id="arrived_target", target_owner="target",
        native_geometry_names=("tool_box",), sphere_indices=(0,), interval=(.5, .9),
        arrival_goal_id=("pose_events", 1),
        activation_event={"skill_id": "test", "event_id": "close", "occurrence": 0, "kind": "close"},
        source="explicit arrival policy")
    current = scene_factory(owners=(owner(ProxyQuery("target", box_geometry())),), permissions=(permission,))
    if plateau:
        mapping = _ProgressMap(np.array([0., .25, .75, 1.]), np.array([0., .5, .5, 1.]))
        args = states([[0., 0., 0.]], [[.5, .5, .5]], dtype=dtype)
        clock = torch.tensor([[.25, .5, .75]], dtype=dtype)
    else:
        mapping = _ProgressMap(np.array([0., 1.]), np.array([0., 1.]))
        args = states([[0.]*5], [[.499, .5, .9, .5, .499]], dtype=dtype)
        clock = args[3]

    def operation(scene, diagnostics, layers, components):
        return _scene_boundary_evaluate(scene, SimpleNamespace(start_progress=0., phase_count=1),
            mapping, *args, trajectory_time=clock, normalized_time=clock, source_geometry=True,
            diagnostics=diagnostics, layer_selection_out=layers, source_components_out=components)

    residual, diagnostics, layers = native_differential(current, operation, monkeypatch)
    assert residual.tolist() == ([[0., 0., 0.]] if plateau else [[1., 0., 1., 0., 1.]])
    assert diagnostics["checked_states"] == args[3].numel()*3
    assert layers["query_order"] == ["exact", "left", "right"]


def test_real_candidate_release_pose_is_not_reused_across_calls(scene_factory, monkeypatch):
    relation = Attachment((0., 1.), np.array([.1, 0., 0.]), np.array([1., 0., 0., 0.]),
                          "candidate_hold", "candidate_release")
    current = scene_factory(owners=(owner(ProxyQuery("target", box_geometry()),
                                         attachments=(relation,), moving=True),))
    for anchor, expected in ((.2, [[0., 1., 1., 0.]]), (.6, [[0., 0., 0., 0.]])):
        args = states([[anchor, .3, .3, anchor]], [[1., 1.5, 1.5, 1.]])
        # Each differential owns its wrapper so the second candidate cannot
        # accidentally capture the previous spy as its native implementation.
        with monkeypatch.context() as context:
            residual, _, _ = native_differential(current, source_operation(args), context)
        assert residual.tolist() == expected
    assert current._source_scene_checker._scene_release_predictions == {}


def test_real_layer_records_and_complete_native_evidence_are_unchanged(scene_factory, monkeypatch):
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
    args = states([[.2, .28, .2, .28]])
    residual, _, _ = native_differential(current, source_operation(args), monkeypatch)
    assert residual.tolist() == [[0., 1., 0., 1.]]


# BEGIN measure the actual production dictionary rather than restating its budget.
def test_source_reservation_bounds_real_unique_key_storage_as_horizon_grows(scene_factory, monkeypatch):
    shape = (21, 21, 21)
    region = {"pose": [.2, 0., 0., 1., 0., 0., 0.], "shape_xyz": list(shape),
        "voxel_size_m": .01, "numerical_support_m": .02,
        "measurement_allowance_m": .001, "measurement_evidence": "explicit test margin",
        "source": "complete observed synthetic fine region",
        "source_bits_hex": np.full(shape, 3, np.uint8).tobytes().hex(),
        "raw_free_packbits_hex": np.packbits(np.zeros(shape, bool)).tobytes().hex()}
    layered = LayeredVoxelQuery(ProxyQuery("background", empty_geometry()),
        ProxyQuery("background", empty_geometry()), region)
    bodies = ({"body_id": "tool", "sphere_indices": [0], "geometry_names": ["tool_box"]},)
    target = owner(ProxyQuery("target", box_geometry()), moving=True)
    current = scene_factory(layered, owners=(target,), bodies=bodies)
    checker = current._source_scene_checker

    def cheap_verdict(*args, **kwargs):
        # Geometry correctness has separate native tests. Here only the
        # production byte keys, tuple/dict allocation and residuals matter.
        return {"residual": float(False), "unmodelled_geometry_names": []}

    monkeypatch.setattr(current.robot_validator, "check_planning_world", cheap_verdict)
    captured = []

    def capture_dictionary(frame, event, argument):
        if event == "return" and frame.f_code is PlanningScene.check_world_geometry.__code__:
            captured.append(frame.f_locals["state_residuals"])

    def owned_bytes(value, seen=None):
        seen = set() if seen is None else seen
        if id(value) in seen:
            return 0
        seen.add(id(value))
        total = sys.getsizeof(value)
        if isinstance(value, dict):
            total += sum(owned_bytes(key, seen)+owned_bytes(item, seen) for key, item in value.items())
        elif isinstance(value, tuple):
            total += sum(owned_bytes(item, seen) for item in value)
        return total

    measured_sizes, reservations = [], []
    for count in (1, 31, 257):
        args = states([np.linspace(.2, .4, count).tolist()])
        query = {}
        current.components(*args, trajectory_time=args[3], geometry_query_out=query)
        admitted = current.estimated_workspace_allocation(1, count, robot_coverage_horizon=0)
        with monkeypatch.context() as context:
            context.setattr(current, "_source_scene_checker", None)
            original = current.estimated_workspace_allocation(1, count, robot_coverage_horizon=0)
        # Unbinding removes both the pre-existing one-state native scratch
        # and the new keys. Subtract only that known old scratch to isolate
        # the resource increment; no key-size formula is reproduced here.
        reserved = admitted["live"]-original["live"]-checker.scene_query_workspace_nbytes
        previous_profile = sys.getprofile()
        try:
            sys.setprofile(capture_dictionary)
            current.check_world_geometry(args[4], args[5], args[3], args[3],
                geometry_query=query, robot_owner_permission_progress=args[3])
        finally:
            sys.setprofile(previous_profile)
        dictionary = captured.pop()
        assert len(dictionary) == count  # Exercise the worst case: no reuse.
        measured = owned_bytes(dictionary)
        assert measured <= reserved
        measured_sizes.append(measured)
        reservations.append(reserved)
    assert measured_sizes[0] < measured_sizes[1] < measured_sizes[2]
    assert reservations[0] < reservations[1] < reservations[2]
# END the measured key inventory includes actual owner and whole-body layer fields.
# END equality covers accepted decisions and the raw evidence explaining them.
