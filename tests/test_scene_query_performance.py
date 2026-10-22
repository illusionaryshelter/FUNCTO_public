"""CPU checks for scene-query work reduction and unchanged derivatives."""

# BEGIN tensor-only manager fixtures exercise the production query boundary.
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from manipulation.implicit_distance import ObservedField
from manipulation.observed_collision import pair_residual
from manipulation.planning import PlanningBudget, _ProgressMap, _process_manager_type


@pytest.fixture
def manager_fixture(monkeypatch):
    """Use CPU cost buffers and analytic FK, without loading a robot asset."""
    import curobo.kinematics
    import curobo._src.rollout.cost_manager.cost_manager_robot as manager_module
    import curobo._src.cost.cost_cspace_base as cspace_module
    import manipulation.planning as planning
    from curobo._src.robot.types.joint_limits import JointLimits
    from curobo.types import DeviceCfg

    def reject_cuda(*args, **kwargs):
        raise AssertionError("CPU query tests must not initialize CUDA")

    monkeypatch.setattr(torch.cuda, "_lazy_init", reject_cuda)
    monkeypatch.setattr(manager_module, "create_cuda_stream_pair", lambda: (None, None))
    monkeypatch.setattr(manager_module, "synchronize_cuda_streams", lambda *args: None)
    monkeypatch.setattr(manager_module.RobotCostManager, "_stream_context", lambda *args: nullcontext())
    monkeypatch.setattr(cspace_module, "init_warp", lambda: None)
    monkeypatch.setattr(planning, "_orientation_vjp_type",
                        lambda: SimpleNamespace(apply=lambda q, rotation, jacobian: rotation))
    device = DeviceCfg("cpu")

    def kinematics(joints):
        q = joints.position
        position = torch.cat((q, torch.zeros_like(q[..., :1])), -1)
        quaternion = q.new_zeros((*q.shape[:2], 1, 4))
        quaternion[..., 0] = 1.
        return SimpleNamespace(tool_poses=SimpleNamespace(position=position.unsqueeze(-2),
            quaternion=quaternion, tool_frames=["tool"]),
            tool_jacobians=q.new_zeros((*q.shape[:2], 1, 6, 2)),
            robot_spheres=q.new_zeros((*q.shape[:2], 0, 4)))

    monkeypatch.setattr(curobo.kinematics, "Kinematics",
                        lambda *args, **kwargs: SimpleNamespace(compute_kinematics=kinematics))

    @contextmanager
    def factory(mode, *, required, capacity, task=None):
        names = ("first_joint", "second_joint")
        bounds = torch.tensor([[-20., -20.], [20., 20.]])
        limits = JointLimits(list(names), bounds, bounds, bounds, bounds, bounds, device)
        robot = SimpleNamespace(tool_frames=["tool"], config=object(), total_spheres=0,
                                get_self_collision_config=lambda: SimpleNamespace(num_spheres=0))
        transition = SimpleNamespace(robot_model=robot, interpolation_steps=1, action_dim=2,
            teleport_mode=False, joint_names=names, get_state_bounds=lambda: limits)
        problem = task
        if problem is None:
            problem = SimpleNamespace(tool_link="tool", start_progress=0., end_progress=1.,
                mechanisms=(), stop_progress=(), transform_tool_pose=lambda p, r: (p, r),
                transform_robot_spheres=lambda value: value, functional_constraint_keys=(),
                functional_inequalities=lambda *args, **kwargs: ({}, {}),
                residuals=lambda p, *args, **kwargs: {"process": p[..., 0] * 0},
                optimizer_preferences=lambda p, *args, **kwargs: p[..., 0] * 0)
        scene = SimpleNamespace(required_progress_anchors=(), contacts=(),
                                optimizer_preferences=lambda progress, *args: progress * 0)
        # BEGIN the analytic scene explicitly evaluates its empty self-pair set.
        # Scope tests use the same manager entry as real robot scenes. This
        # fixture declares no collision entities, so its differentiable self
        # components are exactly zero in either requested geometry scope.
        scene.queries = []

        def clear_scene(spheres, position, quaternion, progress, joints, names,
                        *, components_out, geometry_scope, **kwargs):
            scene.queries.append(geometry_scope)
            zero = joints.sum(-1) * 0
            components_out.update({"collision/robot/self": zero,
                                   "unresolved/robot/self": zero})
            return zero

        scene.evaluate = clear_scene
        # END individual regressions can replace this explicit declared model.
        scene.clone_for_rollout = lambda: scene
        configuration = _process_manager_type(problem, scene,
            _ProgressMap(np.array([0., 1.]), np.array([0., 1.])), required,
            PlanningBudget(constraint_weight=777., max_task_anchors=capacity),
            clock={"duration": 2.}, seed_parameter=torch.zeros((1, 1)), bound_reference=None)
        config = configuration.create({"process_mode": mode}, device_cfg=device)
        manager = config.class_type(device)
        manager.initialize_from_config(config, transition)
        yield SimpleNamespace(manager=manager, scene=scene, kinematics=kinematics,
                              problem=problem, configuration=configuration)

    yield factory


def evaluate_manager(factory, mode, counts):
    """A declared release contributes its first exact joint witness to all states."""
    from curobo.types import JointState
    from curobo._src.state.state_robot import RobotState

    with factory(mode, required=(.5,), capacity=4) as fixture:
        q = torch.tensor([[[-1., 0.], [.2, 0.], [1., 0.], [9., 0.], [13., 0.]],
                          [[-1.2, 0.], [-.6, 0.], [.4, 0.], [.8, 0.], [1.2, 0.]]],
                         requires_grad=True)
        counts = torch.tensor(counts)
        joints = JointState.from_position(q, joint_names=list(fixture.manager._joint_names))
        joints.velocity = torch.zeros_like(q)
        joints.dt = 2. / (counts - 1)
        state = RobotState(joint_state=joints, cuda_robot_model_state=fixture.kinematics(joints))
        seen = []

        def evaluate(spheres, position, quaternion, progress, joint_position, joint_names,
                     *, trajectory_time, components_out, **kwargs):
            seen.append(SimpleNamespace(progress=progress.detach().clone(),
                joints=joint_position.detach().clone(), times=trajectory_time.detach().clone()))
            release_index = (progress == .5).to(torch.long).argmax(1)
            release = joint_position[..., 0].gather(1, release_index[:, None])
            residual = joint_position[..., 0].square() + release.square()
            components_out["collision/observed/entities"] = residual
            return residual

        fixture.scene.required_progress_anchors = (.5,)
        fixture.scene.contacts = ()
        fixture.scene.evaluate = evaluate
        fixture.manager.set_dense_counts(counts)
        costs = fixture.manager.compute_costs(state)
        value = costs.values[costs.names.index("process_scene")].sum((1, 2)) / 777.
        gradient = torch.autograd.grad(value.sum(), q)[0]
        # Independent polynomial: each real sample plus one exact event query
        # contributes q_x^2 and the first event's q_x^2. Mode-side ties retain
        # that same scalar function and its derivative.
        expected = torch.stack([
            q[row, :count, 0].square().sum()
            + (count + 2) * q[row, (count - 1) // 2, 0].square()
            for row, count in enumerate(counts.tolist())])
        expected_gradient = torch.autograd.grad(expected.sum(), q)[0]
        torch.testing.assert_close(value, expected, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(gradient, expected_gradient, rtol=1e-6, atol=1e-6)
        if mode == "trajectory_metrics":
            source, evidence = fixture.manager._scene_evidence
            assert source is costs.values[costs.names.index("process_scene")]
            torch.testing.assert_close(evidence[..., 0].sum(1) / 777., expected.detach())
            assert not evidence[..., 1:].any()
        else:
            assert fixture.manager._optimizer_evidence[-1] is costs.values[costs.names.index("process_scene")]
            assert not fixture.manager._optimizer_evidence[1].any()
        return seen, dict(fixture.manager._query_counts), gradient


@pytest.mark.parametrize("mode", ["trajectory_metrics", "trajectory_optimizer"])
def test_dense_padding_is_removed_before_scene_queries_and_evidence(mode, manager_fixture):
    seen, counts, gradient = evaluate_manager(manager_fixture, mode, (3, 3))
    assert len(seen) == 1
    assert counts["exact_states"] == 8  # Three real samples and one event per row.
    assert counts["left_states"] == counts["right_states"] == 4
    assert counts["evaluated_states"] == 16
    assert seen[0].joints.shape == (2, 8, 2)
    assert seen[0].joints.abs().max() <= 1.2
    assert not gradient[:, 3:].any()


def test_dense_compaction_keeps_columns_needed_by_another_batch_row(manager_fixture):
    seen, counts, gradient = evaluate_manager(manager_fixture, "trajectory_metrics", (3, 5))
    assert len(seen) == 1
    # Different sorted valid layouts require the union of all six columns.
    assert counts["exact_states"] == 12
    assert seen[0].joints.shape[1] == counts["evaluated_states"] // 2
    assert not gradient[0, 3:].any()
    assert gradient[1, 3:, 0].abs().min() > 0
# END padding removal retains event history, evidence layout and joint gradients.


# BEGIN raw observed fields exercise shared and standalone zero baselines.
@pytest.mark.parametrize("shift,status", [(.09, "clear"), (.01, "unresolved"), (.07, "intersection")])
def test_pair_shared_zero_preserves_classification_value_and_gradient(shift, status):
    axis = np.linspace(-.05, .05, 7)
    points = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), -1)
    field = ObservedField(np.max(np.abs(points), -1) - .04, np.ones((7, 7, 7), bool),
                          origin=[-.05] * 3, voxel_size=axis[1] - axis[0])
    if status == "unresolved":
        # A constant negative supported cell touches its translated copy at
        # one exact cell face; no interpolated zero-location approximation is
        # used to construct this boundary case.
        field = ObservedField(-np.ones((2, 2, 2)), np.ones((2, 2, 2), bool),
                              origin=[0.] * 3, voxel_size=.01)
    quaternion = torch.tensor([1., 0., 0., 0.], dtype=torch.float64)
    matrix = np.eye(4)
    matrix[0, 3] = shift
    results = []
    for shared in (False, True):
        first = torch.zeros((2, 3), dtype=torch.float64, requires_grad=True)
        second = torch.tensor([[shift, 0., 0.], [0., 0., 0.]],
                              dtype=torch.float64, requires_grad=True)
        zero = (first.sum(-1) * 0 + second.sum(-1) * 0)[0] if shared else None
        residual, evidence = pair_residual(field, (first[0], quaternion), np.eye(4),
            field, (second[0], quaternion), matrix, zero=zero)
        assert evidence["status"] == status
        if shared and status != "intersection":
            assert residual is zero
        gradients = torch.autograd.grad(residual, (first, second))
        results.append((residual.detach(), evidence, gradients))
    assert results[0][1] == results[1][1]
    torch.testing.assert_close(results[0][0], results[1][0], rtol=0, atol=0)
    for actual, expected in zip(results[0][2], results[1][2]):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
# END batch sharing changes only zero construction, including unresolved queries.
