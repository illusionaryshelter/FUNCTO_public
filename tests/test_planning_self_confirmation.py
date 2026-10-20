"""Self proxies share candidate retention, exact clocks and hard acceptance."""

# BEGIN CPU integration checks for the model-only self confirmation patch.
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from manipulation.online_native_confirmation import OnlineCollisionConfirmation
from manipulation.planning import (_candidate_confirmation_times, _confirm_native_self_collection,
                                   _scene_boundary_evaluate, _ProgressMap, MotionPlanningError)
import test_manipulation_planning as _planning_fixtures
from test_native_confirmation_selection import selection_fixture, completed


def test_native_collection_preserves_nonself_and_nonfinite_self_samples():
    from curobo._src.rollout.metrics import CostCollection
    checked = Mock(return_value={'passed': True, 'residual': 0., 'unmodelled_geometry_names': ['future_finger']})
    adapter = OnlineCollisionConfirmation(SimpleNamespace(check_planning_self=checked),
                                          SimpleNamespace(gripper=None))
    self_proxy = torch.tensor([[[.1], [float('nan')], [0.], [float('-inf')]]])
    contact = torch.full_like(self_proxy, .2)
    collection = CostCollection(values=[self_proxy, contact], names=['self_collision', 'contact'])
    q, time = torch.zeros((1, 4, 1)), torch.arange(4, dtype=torch.float32)[None]
    _confirm_native_self_collection(collection, adapter, q, ('joint',), time)
    assert collection.values[0][0, 0, 0] == 0.
    assert torch.isnan(collection.values[0][0, 1, 0])
    assert collection.values[0][0, 2, 0] == 0.
    assert torch.isneginf(collection.values[0][0, 3, 0])
    assert collection.values[1] is contact and self_proxy[0, 0, 0] == .1
    assert checked.call_count == 1
    assert adapter.self_report['states'][0]['unmodelled_geometry_names'] == ['future_finger']


@pytest.mark.parametrize('residual', [0., .1])
def test_absent_confirmation_capability_retains_native_hard_self_gate(residual):
    from curobo._src.rollout.metrics import CostCollection
    value = torch.full((1, 2, 1), residual)
    collection = CostCollection(names=['self_collision'], values=[value])
    result = _confirm_native_self_collection(collection, None, torch.zeros((1, 2, 1)),
                                            ('joint',), torch.tensor([[0., 1.]]))
    assert result is collection and result.values[0] is value
    assert bool((result.values[0] <= 0).all()) is (residual == 0.)


@pytest.mark.parametrize('storage', ['constraints', 'hybrid'])
@pytest.mark.parametrize('source', ['self_collision', 'anchor_self_collision'])
def test_finite_self_proxy_is_pending_in_both_native_hard_collections(storage, source):
    from curobo._src.rollout.metrics import CostCollection, CostsAndConstraints, RolloutResult
    case = _planning_fixtures.NativeCostManagerOwnershipTests()
    case.setUp()
    try:
        with case.anchor_self_collision_fixture('trajectory_optimizer', required=(0., 1.), capacity=4) as fixture:
            costs = fixture.manager.compute_costs(fixture.state)
            # Preserve all same-forward task/scene evidence. Only this native
            # sphere channel is positive, whether hard or hybrid storage owns it.
            hybrid = CostCollection()
            if storage == 'constraints':
                index = costs.names.index(source)
                costs.values[index] = torch.full_like(costs.values[index], .25)
            else:
                hybrid.add(torch.full_like(costs.values[0], .25), source)
            result = RolloutResult(costs_and_constraints=CostsAndConstraints(
                constraints=costs, hybrid_costs_constraints=hybrid))
            eligible, feasible, _, _ = fixture.manager.optimizer_evidence(result)
            assert eligible.all() and not feasible.any()
            target = costs if storage == 'constraints' else hybrid
            target.values[target.names.index(source)][0, 0, 0] = float('-inf')
            eligible, feasible, _, _ = fixture.manager.optimizer_evidence(result)
            assert not eligible[0] and not feasible[0]
    finally:
        case.doCleanups()


@pytest.mark.parametrize('anchor_clear', [True, False])
def test_candidate_manager_confirms_regular_and_unfolded_anchor_states(anchor_clear):
    from curobo._src.state.state_robot import RobotState
    from curobo.types import JointState
    case = _planning_fixtures.NativeCostManagerOwnershipTests()
    case.setUp()
    try:
        with case.anchor_self_collision_fixture('trajectory_optimizer', required=(.5,), capacity=4) as fixture:
            q = torch.full_like(fixture.q, .1)
            joints = JointState.from_position(q, joint_names=['first_joint', 'second_joint'])
            joints.velocity, joints.dt = torch.zeros_like(q), torch.ones(2)
            state = RobotState(joint_state=joints, cuda_robot_model_state=fixture.kinematics(joints))
            times = torch.tensor([[.1, .7]]).expand(2, -1)
            anchor_u = torch.tensor([[.25]]).expand(2, -1)
            anchor_times = anchor_u*2.
            anchor_q = torch.full((2, 1, 2), .2)
            def check(position, names, *, time_s, gripper):
                assert tuple(names) == ('first_joint', 'second_joint')
                clear = anchor_clear or time_s != .5
                return {'passed': clear, 'residual': 0. if clear else .01}
            checker = Mock(side_effect=check)
            confirmation = OnlineCollisionConfirmation(
                SimpleNamespace(check_planning_self=checker), fixture.scene)
            fixture.manager._candidate_query = {
                'scene': fixture.scene, 'mapping': _ProgressMap(np.array([0., 1.]), np.array([0., 1.])),
                'duration': 2., 'times': times, 'progress': times/2.,
                'anchor_u': anchor_u, 'anchor_s': anchor_u, 'anchor_times': anchor_times,
                'anchor_q': anchor_q, 'anchor_dq': torch.zeros_like(anchor_q),
                'scene_components': {}, 'native_confirmation': confirmation}
            costs = fixture.manager.compute_costs(state)
            values = dict(zip(costs.names, costs.values))
            assert not values['self_collision'].any()
            assert bool(values['anchor_self_collision'].any()) is (not anchor_clear)
            assert checker.call_count == 3  # Equal states in the two batches share proof.
            actual = [(call.args[0].copy(), call.kwargs['time_s']) for call in checker.call_args_list]
            for position, time_s in actual:
                np.testing.assert_array_equal(position, np.full(2, .2 if time_s == .5 else .1, np.float32))
            assert {row[1] for row in actual} == {float(times[0, 0]), float(times[0, 1]), .5}
            assert all(bool((row['value'] > 0).all()) for row in fixture.forwarded)
            torch.testing.assert_close(joints.position, q, rtol=0, atol=0)
    finally:
        case.doCleanups()


def test_exact_prismatic_violation_cannot_become_a_pending_sphere_candidate():
    # The tiny exact violation remains hard beside a huge sphere objective;
    # aggregate subtraction would lose it in float32.
    def evaluate(spheres, p, r, progress, q, names, *, components_out, **kwargs):
        components_out.update({'collision/robot/self': torch.full_like(progress, 1e20),
                               'collision/robot/self_prismatic': torch.full_like(progress, 1e-30)})
        return sum(components_out.values())
    scene = SimpleNamespace(required_progress_anchors=(), evaluate=evaluate,
        native_collision_components=frozenset({'collision/robot/self'}))
    time = torch.tensor([[0., 1.]])
    evidence = {}
    _scene_boundary_evaluate(scene, SimpleNamespace(start_progress=0., end_progress=1.),
        _ProgressMap(np.array([0., 1.]), np.array([0., 1.])),
        torch.zeros((1, 2, 1, 4)), torch.zeros((1, 2, 3)), torch.zeros((1, 2, 4)), time,
        torch.zeros((1, 2, 1)), ('q',), trajectory_time=time, normalized_time=time,
        scene_evidence_out=evidence, optimizer_candidate_evidence=True)
    assert evidence['optimizer_hard_invalid'].all()


@pytest.mark.parametrize('source', ['self_collision', 'anchor_self_collision'])
def test_self_only_native_rejection_reaches_full_candidate_confirmation(source):
    planner, result = selection_fixture(scene=(0., 0.), task=(0., .1))
    result.metrics.costs_and_constraints.constraints.add(
        torch.tensor([.25, 0.])[:, None, None].expand(2, 9, 1).clone(), source)
    planner._candidate_metrics.side_effect = lambda curve, scene, *, native_query: completed(
        accepted=True, residual=0., points=9)
    _, curve, report = planner.select_candidate(result)
    assert report['accepted'] and curve.seed_index == 0
    assert report['confirmation_validation_points'] == 9
    assert report['sphere_proposal_metrics']['rollout']['constraints'][source] == .25
    planner._candidate_metrics.assert_called_once_with(curve, planner.scene, native_query=True)


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), float('-inf')])
def test_nonfinite_self_sample_cannot_hide_behind_clear_neighbours_or_success_flag(bad):
    planner, result = selection_fixture(scene=(0., 0.), task=(0., .1), flags=(True, False))
    values = torch.zeros((2, 9, 1))
    values[0, 4, 0] = bad
    result.metrics.costs_and_constraints.constraints.add(values, 'self_collision')
    _, _, report = planner.select_candidate(result)
    assert not report['accepted']
    planner._candidate_metrics.assert_not_called()


def test_confirmation_union_keeps_regular_dense_and_source_times_with_unchanged_capacity():
    trajectory = SimpleNamespace(times=np.array([0., .7, 1.7]), duration=1.7)
    budget = SimpleNamespace(interpolation_dt=.13, max_validation_points=128, interpolation_buffer_size=128)
    times = _candidate_confirmation_times(trajectory, budget, regular_count=7, regular_dt=1.7/6,
        dense_count=10, dense_dt=1.7/9)
    represented = set(times)
    for count in (7, 10):
        original = np.arange(count, dtype=float)*(trajectory.duration/(count-1))
        assert set(original) <= represented
    assert .7 in represented
    assert times[0] == 0. and times[-1] == trajectory.duration
    budget.interpolation_buffer_size = len(times)-1
    with pytest.raises(MotionPlanningError, match='Both original metric grids'):
        _candidate_confirmation_times(trajectory, budget, regular_count=7, regular_dt=1.7/6,
        dense_count=10, dense_dt=1.7/9)


def test_confirmation_canonical_nested_native_grids_fit_original_buffer():
    dt = float(np.float32(.17))
    dense_dt = dt/8
    trajectory = SimpleNamespace(times=np.arange(641)*dense_dt, duration=80*dt)
    budget = SimpleNamespace(interpolation_dt=dense_dt, max_validation_points=4096,
                             interpolation_buffer_size=768)
    times = _candidate_confirmation_times(trajectory, budget, regular_count=81, regular_dt=dt,
        dense_count=641, dense_dt=dense_dt)
    np.testing.assert_array_equal(times, trajectory.times)
    assert len(times) == 641


def test_entrance_self_proxy_requires_actual_model_clear_before_acceptance():
    case = _planning_fixtures.InitialStatePreflightTests()
    planner, _, collision = case.fixture(native_collision=1.)
    # The original fixture's default source checker rejects its declared
    # collision. Replace only that independent model result for the false-
    # positive counterexample; q/dq and every scene requirement are unchanged.
    planner.robot_validator.check_planning_self = Mock(return_value={
        'passed': True, 'residual': 0., 'unmodelled_geometry_names': []})
    before = planner.start.position.clone(), planner.start.velocity.clone()
    report = planner.initial_state_diagnostics()
    assert report['passed'] and report['residuals']['self_collision'] == 0.
    assert collision.forward.return_value.item() == 1.
    checked = planner.robot_validator.check_planning_self.call_args
    np.testing.assert_array_equal(checked.args[0], before[0][0].numpy())
    assert checked.kwargs['time_s'] == 0.
    torch.testing.assert_close(planner.start.velocity, before[1], rtol=0, atol=0)
    assert report['native_confirmation']['self_collision']['states'][0]['sphere_residuals'] == {
        'self_collision': 1.}
# END no source geometry permission is inferred from the optimization proxy.
