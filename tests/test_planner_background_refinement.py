"""CPU integration checks for the planner's map replacement boundary."""

# BEGIN witnesses retain the native curve/clock through a solver rebuild.
from types import SimpleNamespace
from unittest.mock import Mock
from copy import deepcopy
import math

import numpy as np
import pytest
import torch

from manipulation.physical_joint_domain import CubicJointDomain
from manipulation.planning_observation import _FrozenBackgroundRefinement
from manipulation.planning_scene import ObservedSpace, PlanningScene
from manipulation.planning import (
    PlanningBudget, ResourceLimitError, _CuroboPlanner,
    _planning_workspace_bytes, _require_planning_memory, _PlanningWorkspace,
)
from manipulation.voxel_layers import LayeredVoxelQuery
from test_source_world_acceptance import ProxyQuery, observed_plane, owner, scene_factory, states
from test_source_world_deduplication import assert_report_bytes_equal, assert_business_report_bytes_equal
from test_voxel_layers import NativeStub, region


# BEGIN CPU fixtures expose each preflight boundary without touching CUDA.
@pytest.fixture(autouse=True)
def admission(monkeypatch):
    result = SimpleNamespace(storage=Mock(return_value={"encoded_record_bytes_upper": 7}),
        workspace=Mock(return_value=_PlanningWorkspace(0, 0, 101, 0, 103, False)),
        host=Mock(), gpu=Mock(side_effect=lambda runtime, device, workspace, **limits: workspace))
    monkeypatch.setattr("manipulation.planning._validation_storage_admission", result.storage)
    monkeypatch.setattr("manipulation.planning._planning_workspace_bytes", result.workspace)
    monkeypatch.setattr("manipulation.planning._require_planning_host_memory", result.host)
    monkeypatch.setattr("manipulation.planning._require_planning_memory", result.gpu)
    return result
# END the separately imported capacity functions remain available for pure tests.


def planner_fixture(replace=True):
    planner = _CuroboPlanner.__new__(_CuroboPlanner)
    planner.torch = torch
    planner.device_cfg = SimpleNamespace(device="cpu", to_device=lambda value: torch.as_tensor(value, dtype=torch.float32))
    planner.budget = PlanningBudget()
    planner.robot_metadata = {"active_joint_names": ["joint"], "links": 1, "spheres": 1}
    planner.joint_names = ("joint",)
    planner.JointState = SimpleNamespace(from_position=lambda q, **kwargs: SimpleNamespace(position=q))
    planner.problem = SimpleNamespace(tool_link="tcp", transform_tool_pose=lambda p, r: (p, r),
                                     transform_robot_spheres=lambda value: value)
    planner.physical_domain = CubicJointDomain(torch.tensor([-2.]), torch.tensor([2.]),
        torch.tensor([[.2]]), torch.tensor([[.05]]), .1, 4)
    planner.progress_map = SimpleNamespace(u=np.array([0., 1.]), s=np.array([0., 2.]),
        anchor_parameters=lambda anchors: np.array([[.137, .274], [.811, 1.622]]))
    planner.anchors = np.array([.274, 1.622])
    planner.refinement_parameters = np.array([.477])
    planner.formal_clock = {"duration": .8}
    planner.trajopt = SimpleNamespace(horizon=9, destroy=Mock())
    background = LayeredVoxelQuery.__new__(LayeredVoxelQuery)
    background.refinement_source = object()
    seen = []

    def fk(state):
        q = state.position
        centers = torch.cat((q, torch.zeros_like(q), torch.zeros_like(q)), -1)
        radii = torch.full_like(q, .01)
        orientation = torch.zeros((*q.shape[:2], 1, 4))
        orientation[..., 0] = 1.
        return SimpleNamespace(robot_spheres=torch.cat((centers, radii), -1).unsqueeze(-2),
            tool_poses=SimpleNamespace(tool_frames=["tcp"], position=centers.unsqueeze(-2),
                                       quaternion=orientation))

    def witnesses(spheres, position, quaternion, progress, joints, names, *, trajectory_time, include_satisfied):
        assert include_satisfied is True
        seen.append((joints.clone(), trajectory_time.clone(), progress.clone()))
        return [{"time_s": float(trajectory_time[0, 0]), "body_id": "body", "positive_spheres": 1,
                 "fine_supported": False, "guidance_weight": 0., "coarse_cost": .2,
                 "fine_cost": .1, "guidance_cost": .2}]

    verification = object()
    replacement = SimpleNamespace(clone_for_rollout=Mock(return_value=verification),
                                  background_refinement_witnesses=witnesses)
    planner.scene = SimpleNamespace(background=background, background_refinement_witnesses=witnesses,
        refine_background=Mock(return_value=(replacement if replace else None, {"reason": "test"})))
    planner._verification_fk = SimpleNamespace(compute_kinematics=fk)
    planner._verification_scene = object()
    planner._check_validation_work = Mock()
    planner.refine_validation = Mock()
    raw = torch.full((1, 1, 4, 1), .3)
    seed = planner.physical_domain.initialize(raw)
    return planner, seed, seen, replacement, verification


def test_witness_queries_use_actual_cubic_exact_times_and_replace_before_rebuild(admission):
    planner, seed, seen, replacement, verification = planner_fixture()
    old_scene, old_verification, old_solver = planner.scene, planner._verification_scene, planner.trajopt
    stages = []

    def gpu_check(runtime, device, workspace, **limits):
        assert planner.scene is old_scene
        assert planner._verification_scene is old_verification
        assert planner.trajopt is old_solver
        replacement.clone_for_rollout.assert_not_called()
        admission.storage.assert_called_once_with(planner.problem, replacement, planner.robot_metadata, planner.budget)
        admission.workspace.assert_called_once()
        stages.append("admitted")
        return workspace

    admission.gpu.side_effect = gpu_check

    def rebuild(parameters):
        assert stages == ["admitted"]
        assert planner.scene is replacement
        assert planner._verification_scene is verification
        np.testing.assert_array_equal(parameters, [.477])
        stages.append("rebuilt")

    planner.refine_validation.side_effect = rebuild
    result = planner.refine_background_for_seed(seed)
    joints, times, progress = seen[0]
    u = times.numpy()[0]/.8
    for anchor in (.137, .811, .477):
        assert np.any(np.isclose(u, anchor, atol=1e-7, rtol=0))
    expected = planner.physical_domain.evaluate(seed[0].numpy(), u)
    np.testing.assert_allclose(joints.numpy(), expected, atol=1e-7, rtol=0)
    np.testing.assert_allclose(progress.numpy()[0], 2*u, atol=1e-7, rtol=0)
    assert result["updated"] is True
    planner.refine_validation.assert_called_once()
    assert planner._check_validation_work.call_count >= 3
    assert stages == ["admitted", "rebuilt"]
    assert result["planner_admission"]["scene_workspace_bytes"] == 101


def test_refused_refinement_retains_scene_and_solver():
    planner, seed, _, _, _ = planner_fixture(replace=False)
    original = planner.scene
    result = planner.refine_background_for_seed(seed)
    assert result["updated"] is False
    assert planner.scene is original
    planner.refine_validation.assert_not_called()


# BEGIN exercise the real no-growth refiner through both scene/planner boundaries.
@pytest.mark.parametrize("supported", (False, True))
def test_contained_native_grid_witnesses_do_not_rebuild_scene_or_solver(admission, monkeypatch, supported):
    planner, _, _, _, _ = planner_fixture()
    shape = (81, 43, 45)
    bits = np.full(shape, 3 if supported else 1, np.uint8)
    if supported:
        # An interior unknown plane bounds an already certified cuboid. The
        # body stays in its overlap transition, so it still proposes witnesses
        # even though neither the physical grid nor this cuboid can grow.
        bits[40] = 1
    record = region()
    record.update(shape_xyz=list(shape), voxel_size_m=float(np.float32(.002)),
        numerical_support_m=.01, measurement_allowance_m=.0001,
        source_bits_hex=bits.tobytes().hex(),
        raw_free_packbits_hex=np.packbits(np.zeros(shape, bool)).tobytes().hex())
    background = LayeredVoxelQuery(NativeStub(10., 2.), NativeStub(1., 3.), record)
    background.mapping_report = {}
    bodies = ({"body_id": "body", "sphere_indices": [0], "geometry_names": ["native"]},)
    if supported:
        initial = torch.tensor([[[[-.02, 0., 0., .01]]]], dtype=torch.float32)
        selected = background.region.select(initial, bodies)
        background.region.prepare_guidance_boxes(selected.influence_aabb.numpy().reshape(-1, 2, 3))
        weight = background.region.guidance_weights(initial, bodies)[1].item()
        assert 0 < weight < 1
    views = {"camera": {"image": np.zeros((3, 3, 3), np.uint8), "depth": np.ones((3, 3)),
                        "K": np.eye(3), "T_output_camera": np.eye(4)}}
    source = _FrozenBackgroundRefinement(views, {"camera": {None: np.ones((3, 3), bool)}},
        SimpleNamespace(device="cuda:0"), {"voxel_size_m": .002, "measurement_allowance_m": .0001,
            "measurement_evidence": "explicit calibration", "source": "original RGB-D"}, .0001)
    background.refinement_source = source
    scene = PlanningScene(background=background, owners=(),
        coverage=ObservedSpace(np.ones((5, 5, 5), bool), origin=[-1.]*3, voxel_size=.5,
            observation_ids=["camera"], reference_frame="world"), reference_frame="world",
        robot_sphere_materials=(None,), robot_geometry_source="complete synthetic native cover",
        robot_collision_bodies=bodies, has_gripper=False)
    # These identities make an accidental source rebind observable. Actual
    # source queries are unnecessary for gathering the original coarse costs.
    scene.robot_validator = object()
    scene._source_scene_checker = object()
    planner.scene = scene
    planner.physical_domain = CubicJointDomain(torch.tensor([-2.]), torch.tensor([2.]),
        torch.tensor([[-.02]]), torch.tensor([[.0005]]), .1, 4)
    seed = planner.physical_domain.initialize(torch.full((1, 1, 4, 1), -.0195))
    planner._charge_candidate_queries = Mock()
    planner.optimize = Mock()
    fk = Mock(wraps=planner._verification_fk.compute_kinematics)
    planner._verification_fk.compute_kinematics = fk
    witnesses = Mock(wraps=scene.background_refinement_witnesses)
    monkeypatch.setattr(scene, "background_refinement_witnesses", witnesses)
    refine = Mock(wraps=source.refine)
    monkeypatch.setattr(source, "refine", refine)
    forbidden = []
    for name in ("_fine_mapper_config", "_map_partition", "_require_mapping_memory"):
        guard = Mock(side_effect=AssertionError("Contained witnesses must not allocate another map"))
        monkeypatch.setattr("manipulation.planning_observation."+name, guard)
        forbidden.append(guard)
    bind = Mock(side_effect=AssertionError("An unchanged source scene must not be rebound"))
    monkeypatch.setattr(PlanningScene, "bind_source_scene", bind)
    original = (planner.scene, planner._verification_scene, planner.trajopt, scene._source_scene_checker)
    original_boxes = background.region.guidance_boxes.copy()
    cuda_before = torch.cuda.is_initialized()
    for _ in range(2):
        result = planner.refine_background_for_seed(seed)
        assert result["updated"] is False and result["mapping_reused"] is True
        assert result["status"] == ("unchanged_source_supported_cuboids" if supported
                                    else "observation_support_unavailable")
        assert result["active_witnesses"] == result["sampled_times"] > 0
        assert result["shape_xyz"] == list(shape) and result["center_m"] == [0., 0., 0.]
        assert result["estimated_mapping_bytes"] == 0
        assert (planner.scene, planner._verification_scene, planner.trajopt,
                scene._source_scene_checker) == original
    # The real cubic/FK/witness pass still runs for each request; only the
    # unnecessary map, source-scene and solver replacement are absent.
    assert fk.call_count == witnesses.call_count == refine.call_count == 2
    assert all(call.kwargs["include_satisfied"] is True for call in witnesses.call_args_list)
    assert all(not call.args and not call.kwargs for call in planner._check_validation_work.call_args_list)
    assert planner._check_validation_work.call_count == 4
    assert scene.background is background
    np.testing.assert_array_equal(background.region.guidance_boxes, original_boxes)
    for unused in (*forbidden, bind, admission.storage, admission.workspace, admission.host, admission.gpu,
                   planner.refine_validation, planner._charge_candidate_queries,
                   planner.optimize, planner.trajopt.destroy):
        unused.assert_not_called()
    assert torch.cuda.is_initialized() == cuda_before
# END actual float32 spacing cannot create an otherwise unnecessary rebuild.


# BEGIN a changed optimizer cuboid is not a changed source-geometry snapshot.
def real_guidance_refinement_scene(scene_factory):
    shape = (81, 43, 45)
    bits = np.full(shape, 3, np.uint8)
    bits[40] = 1
    record = region()
    record.update(shape_xyz=list(shape), voxel_size_m=float(np.float32(.002)),
        numerical_support_m=.01, measurement_allowance_m=.0001,
        source_bits_hex=bits.tobytes().hex(),
        raw_free_packbits_hex=np.packbits(np.zeros(shape, bool)).tobytes().hex())
    background = LayeredVoxelQuery(ProxyQuery("background", observed_plane(), bias=1.),
        ProxyQuery("background", observed_plane(), bias=.1), record)
    background.region.prepare_guidance_boxes(np.array([[[-.04, -.01, -.01], [-.02, .01, .01]]]))
    background.mapping_report = {"source": "same original mapped RGB-D", "shape_xyz": list(shape)}
    views = {"camera": {"image": np.zeros((3, 3, 3), np.uint8), "depth": np.ones((3, 3)),
                        "K": np.eye(3), "T_output_camera": np.eye(4)}}
    background.refinement_source = _FrozenBackgroundRefinement(views,
        {"camera": {None: np.ones((3, 3), bool)}}, SimpleNamespace(device="cuda:0"),
        {"voxel_size_m": .002, "measurement_allowance_m": .0001,
         "measurement_evidence": "explicit calibration", "source": "original RGB-D"}, .0001)
    bodies = ({"body_id": "tool", "sphere_indices": [0], "geometry_names": ["tool_box"]},)
    current = scene_factory(background, bodies=bodies,
        owners=(owner(ProxyQuery("target", observed_plane())),))
    current.observation_report = {"partitions": {"observed_background": {
        "fine": deepcopy(background.mapping_report)}}}
    current.bind_source_scene()
    return current


def test_guidance_only_refinement_preserves_native_shapes_cache_and_complete_reports(scene_factory, monkeypatch):
    current = real_guidance_refinement_scene(scene_factory)
    old_binding = current.to_execution_binding()
    old_boxes = current.background.region.guidance_boxes.copy()
    checker = current._source_scene_checker
    # Exercise both a coarse-selected collision and a fine-selected separation
    # before changing only the smooth guidance weights on the right component.
    query = states([[.005, .03]], [[.25, .75]])
    old_evidence = {"capture_components": True}
    old_residual = current.evaluate_source(*query, trajectory_time=query[3], diagnostics=old_evidence)
    assert old_residual.tolist() == [[1., 0.]]
    shapes = tuple(row["shape"] for row in checker._scene_shapes)
    caches = tuple(shape._collision_cache for shape in shapes)
    entries = tuple(tuple(cache.items()) for cache in caches)
    assert any(entries)
    witness_query = states([[.03]])
    witnesses = current.background_refinement_witnesses(*witness_query,
        trajectory_time=witness_query[3], include_satisfied=True)
    assert len(witnesses) == 1 and witnesses[0]["guidance_weight"] == 0.
    previous_cost = current.background.background_guidance(witness_query[0], key="before",
        groups=current.robot_collision_bodies)
    bind = Mock(side_effect=AssertionError("Unchanged source inputs must keep their existing checker"))
    mapper = Mock(side_effect=AssertionError("An existing supported component needs no new Mapper"))
    monkeypatch.setattr(PlanningScene, "bind_source_scene", bind)
    monkeypatch.setattr("manipulation.planning_observation._map_partition", mapper)
    replacement, report = current.refine_background(witnesses)
    assert replacement is not None and report["mapping_reused"] is True
    assert replacement._source_scene_checker is checker
    assert len(replacement.background.region.guidance_boxes) == 2
    np.testing.assert_array_equal(current.background.region.guidance_boxes, old_boxes)
    assert replacement.background.region.guidance_weights(witness_query[0], current.robot_collision_bodies)[1].item() > 0
    assert replacement.background.background_guidance(witness_query[0], key="after",
        groups=current.robot_collision_bodies).item() < previous_cost.item()
    new_binding = replacement.to_execution_binding()
    # Optimization history is exported from the new scene, while every field
    # actually consumed by source geometry/permissions remains identical.
    old_log, new_log = old_binding.pop("observation_report"), new_binding.pop("observation_report")
    assert_report_bytes_equal(new_binding, old_binding)
    assert "background_refinements" not in old_log
    assert new_log["background_refinements"] == [report]
    assert new_log["partitions"] == old_log["partitions"]
    assert current.observation_report == old_log
    new_evidence = {"capture_components": True}
    new_residual = replacement.evaluate_source(*query, trajectory_time=query[3], diagnostics=new_evidence)
    torch.testing.assert_close(new_residual, old_residual, rtol=0., atol=0.)
    # BEGIN unchanged geometry preserves evidence while warm hints reduce work.
    assert_business_report_bytes_equal(new_evidence, old_evidence)
    # END native shape/cache identity is checked independently below.
    for row, shape, cache, saved in zip(checker._scene_shapes, shapes, caches, entries):
        assert row["shape"] is shape and shape._collision_cache is cache
        assert all(cache[key] is value for key, value in saved)
    bind.assert_not_called()
    mapper.assert_not_called()


@pytest.mark.parametrize("remap", (False, True))
def test_refinement_binds_a_new_checker_for_new_map_or_unbound_scene(scene_factory, monkeypatch, remap):
    current = real_guidance_refinement_scene(scene_factory)
    old_checker = current._source_scene_checker
    original_shape = current.background.region.shape
    if remap:
        padding = current.background.region.support+6*current.background.region.size
        old_top = original_shape[2]*current.background.region.size/2
        envelope = [[.02, -.005, old_top-padding-.006], [.04, .005, old_top-padding+.004]]
        witnesses = [{"body_id": "tool", "positive_spheres": 1, "coarse_cost": 1.,
                      "envelope_world_m": envelope}]

        def mapped(*args):
            # Only the CUDA Mapper boundary is replaced. The real refiner
            # decides expansion/shape and the real source checker binds the
            # resulting finite TSDF and changed source-support grid.
            request = args[-1]
            shape = tuple(math.ceil(value/request["voxel_size_m"]) for value in request["extent_m"])
            native_pose = np.asarray(request["center_m"], dtype=np.float32).astype(float)
            record = deepcopy(current.background.region.record)
            record.update(shape_xyz=list(shape), pose=[*native_pose, 1., 0., 0., 0.],
                source_bits_hex=np.full(shape, 3, np.uint8).tobytes().hex(),
                raw_free_packbits_hex=np.packbits(np.zeros(shape, bool)).tobytes().hex())
            changed_geometry = observed_plane()
            changed_geometry["solids"][0]["pose"][2] = .001
            return ProxyQuery("background", changed_geometry, bias=.1), {"shape_xyz": list(shape)}, record

        monkeypatch.setattr("manipulation.planning_observation._fine_mapper_config",
                            Mock(return_value=SimpleNamespace(device="cuda:0")))
        monkeypatch.setattr("manipulation.planning_observation._mapper_memory_bound", lambda _: 1000)
        monkeypatch.setattr("manipulation.planning_observation._require_mapping_memory", Mock())
        mapper = Mock(side_effect=mapped)
    else:
        current._source_scene_checker = None
        query = states([[.03]])
        witnesses = current.background_refinement_witnesses(*query, trajectory_time=query[3])
        mapper = Mock(side_effect=AssertionError("Guidance-only first binding still needs no Mapper"))
    monkeypatch.setattr("manipulation.planning_observation._map_partition", mapper)
    original_bind = PlanningScene.bind_source_scene
    rebound = []

    def bind(scene, **kwargs):
        rebound.append(scene)
        original_bind(scene, **kwargs)

    monkeypatch.setattr(PlanningScene, "bind_source_scene", bind)
    replacement, report = current.refine_background(witnesses)
    assert replacement is not None and report["mapping_reused"] is (not remap)
    assert rebound == [replacement]
    assert replacement._source_scene_checker is not None and replacement._source_scene_checker is not old_checker
    if remap:
        mapper.assert_called_once()
        assert replacement.background.region.shape[2] > original_shape[2]
        assert replacement._source_scene_checker._scene_layers["background"]["region"].shape == replacement.background.region.shape
        # Only the changed fine source is rebuilt. The old scene and its
        # rollout references still share the unchanged coarse/owner maps.
        for new, old in zip(replacement._source_scene_checker._scene_shapes, old_checker._scene_shapes):
            if new["owner"] == "background" and new["layer"] == "fine":
                assert new["shape"] is not old["shape"]
            else:
                assert new["shape"] is old["shape"]
        assert current._source_scene_checker is old_checker
    else:
        mapper.assert_not_called()
        assert current._source_scene_checker is None
# END a real geometry change and a first binding still create their own checker.


def test_builder_api_error_is_not_silently_treated_as_no_refinement():
    planner, seed, _, _, _ = planner_fixture()
    planner.scene.refine_background.side_effect = RuntimeError("native mapper failure")
    with pytest.raises(RuntimeError, match="native mapper failure"):
        planner.refine_background_for_seed(seed)
    planner.refine_validation.assert_not_called()


# BEGIN pre-construction host refusal leaves every old planner object usable.
def test_builder_host_capacity_refusal_preserves_running_scene(admission):
    planner, seed, _, _, _ = planner_fixture()
    original = planner.scene, planner._verification_scene, planner.trajopt
    planner.scene.refine_background.side_effect = ResourceLimitError("host_process_rss", 101, 100)
    result = planner.refine_background_for_seed(seed)
    assert result["updated"] is False and result["status"] == "resource_refused"
    assert result["resource_refusal"]["resource"] == "host_process_rss"
    assert (planner.scene, planner._verification_scene, planner.trajopt) == original
    assert result["before"]["body_queries"] > 0
    for unused in (admission.storage, admission.workspace, admission.host, admission.gpu,
                   planner.refine_validation, planner.trajopt.destroy):
        unused.assert_not_called()
# END only an explicit resource refusal is converted to a retained-scene result.


# BEGIN a refused trial cannot invalidate the already running planner.
@pytest.mark.parametrize("boundary", ["workspace", "host", "gpu"])
def test_candidate_capacity_refusal_preserves_all_running_objects(admission, boundary):
    planner, seed, _, replacement, _ = planner_fixture()
    old_scene, old_verification, old_solver = planner.scene, planner._verification_scene, planner.trajopt
    getattr(admission, boundary).side_effect = ResourceLimitError("test_scene_budget", 933, 644)
    was_cuda_initialized = torch.cuda.is_initialized()
    result = planner.refine_background_for_seed(seed)
    assert result["updated"] is False
    assert result["status"] == "resource_refused"
    assert result["resource_refusal"]["resource"] == "test_scene_budget"
    assert result["resource_refusal"]["required_bytes"] == 933
    assert result["resource_refusal"]["limit_bytes"] == 644
    assert planner.scene is old_scene
    assert planner._verification_scene is old_verification
    assert planner.trajopt is old_solver
    old_solver.destroy.assert_not_called()
    replacement.clone_for_rollout.assert_not_called()
    planner.refine_validation.assert_not_called()
    assert torch.cuda.is_initialized() == was_cuda_initialized


@pytest.mark.parametrize("boundary", ["storage", "workspace", "host", "gpu"])
def test_plain_preflight_value_error_keeps_its_identity(admission, boundary):
    planner, seed, _, _, _ = planner_fixture()
    old_scene, old_verification, old_solver = planner.scene, planner._verification_scene, planner.trajopt
    error = ValueError("invalid API layout, not a byte refusal")
    getattr(admission, boundary).side_effect = error
    with pytest.raises(ValueError) as captured:
        planner.refine_background_for_seed(seed)
    assert captured.value is error
    assert planner.scene is old_scene
    assert planner._verification_scene is old_verification
    assert planner.trajopt is old_solver
    old_solver.destroy.assert_not_called()
    planner.refine_validation.assert_not_called()


def test_capacity_functions_report_the_exact_failed_byte_comparison():
    scene = SimpleNamespace(required_progress_anchors=(), estimated_workspace_allocation=lambda *args, **kwargs:
        {"gpu_retained": 0, "gpu_live": 11, "gpu_shared": 0, "cpu_shared": 0, "cpu_live": 0, "gpu_graph_live": 0})
    with pytest.raises(ResourceLimitError) as captured:
        _planning_workspace_bytes(scene, {}, PlanningBudget(max_scene_gpu_workspace_bytes=10), 2)
    assert (captured.value.resource, captured.value.required_bytes, captured.value.limit_bytes) == (
        "scene_gpu_workspace", 11, 10)
    reserve = 2*1024**3
    cuda = SimpleNamespace(mem_get_info=Mock(return_value=(reserve-1, 12*1024**3)),
        memory_reserved=Mock(return_value=0), memory_allocated=Mock(return_value=0))
    with pytest.raises(ResourceLimitError) as captured:
        _require_planning_memory(SimpleNamespace(cuda=cuda), "cpu",
            _PlanningWorkspace(0, 0, 100, 0, 200, False), max_device_used_bytes=8*1024**3)
    assert (captured.value.resource, captured.value.required_bytes, captured.value.limit_bytes) == (
        "gpu_native_reserve", reserve, reserve-1)
    cuda.mem_get_info.return_value = reserve+50, 12*1024**3
    with pytest.raises(ResourceLimitError) as captured:
        _require_planning_memory(SimpleNamespace(cuda=cuda), "cpu",
            _PlanningWorkspace(0, 0, 100, 0, 200, False), max_device_used_bytes=8*1024**3)
    assert (captured.value.resource, captured.value.required_bytes, captured.value.limit_bytes) == (
        "gpu_planner_workspace", reserve+300, reserve+50)
# END shape/runtime failures propagate; byte-cap refusals preserve the old scene.
# END map proposals never acquire a trajectory acceptance verdict.
