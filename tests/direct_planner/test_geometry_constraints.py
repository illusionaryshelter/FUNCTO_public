"""Small original-model checks for fixed geometry and sparse pose derivatives.

The maintained CPU fixture supplies registered analytic robot entities and
observed TSDF arrays. These checks cover scalar/support meanings, current
maximum derivatives, declaration clocks, first-release anchors and native batch cancellation.
They run neither a full scene plan nor a simulator.
"""

# BEGIN fixed-query maxima and original event-clock regression checks.
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import coal
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import torch

EXPERIMENT = Path(__file__).resolve().parent
PROJECT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(Path(__file__).resolve().parents[2]),
                 str(Path(__file__).resolve().parents[2] / 'tests'),
                 str(Path(__file__).resolve().parents[2] / 'tests/planning')]

from manipulation.direct_planner.differentiable_kinematics import DifferentiablePinKinematics
from manipulation.direct_planner.geometry_constraints import GeometryConstraints, _query_records
from manipulation.implicit_distance import ObservedField
from manipulation.native_field_distance import _transform
from manipulation.observed_mapping import FrozenObservedField
from manipulation.planning import _ProgressMap
from manipulation.planning_problem import PosePath
from manipulation.planning_scene import Attachment, CollisionOwner
from manipulation.sampled_geometry_objective import MaterialSamples
from pair_evidence_fixtures import declared_state, state_record
from manipulation.direct_planner.support_linearization import linearized_robot_state
from manipulation.direct_planner.terminal_binding import bind_terminal_hold

spec = importlib.util.spec_from_file_location('fixed_geometry_fixture',
    PROJECT/'tests/preference_planner/test_geometry.py')
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)


# BEGIN CPU is an explicit test provider; production keeps its declared CUDA device.
@pytest.fixture(autouse=True)
def cpu_batch_distance(monkeypatch):
    from manipulation.direct_planner import geometry_constraints as module
    from manipulation.direct_planner.batch_distance import BatchDistance
    monkeypatch.setattr(module, 'BatchDistance',
        lambda geometries, *, query_sources: BatchDistance(geometries,
            query_sources=query_sources, device='cpu'))
# END the same field values and native models are used by both devices.


def computation(tmp_path, *, field_value=-1.):
    base, controls = fixture.computation(tmp_path, value=field_value)
    base.kinematics = DifferentiablePinKinematics(base.scene.robot_model.model,
        frames=base.kinematics.frames, T_reference_base=np.eye(4))
    base.scene.check_request = lambda: None
    return base, controls


def bind(base):
    mapping = _ProgressMap(np.array([0., 1.]), np.array([0., 1.]))
    geometry = GeometryConstraints(base, progress_map=mapping)
    terminal = bind_terminal_hold(base.domain._curve_map, [0., 1.], [0., 1.], source='fixture clock')
    return geometry, terminal, geometry.bind_solver(terminal)


def dense(point):
    result = np.zeros(point.bound.pattern.shape)
    result[point.bound.pattern.rows, point.bound.pattern.columns] = point.jacobian_values()
    return result


def difference(bound, terminal, controls, epsilon=1e-7):
    reduced = terminal.reduce(controls)
    columns = []
    for index in range(reduced.size):
        step = np.zeros_like(reduced)
        step.flat[index] = epsilon
        columns.append((bound.evaluate(terminal.expand(reduced+step)).values
                        -bound.evaluate(terminal.expand(reduced-step)).values)/(2*epsilon))
    return np.stack(columns, axis=1)








# BEGIN signed near-boundary values and empty-support tails retain distinct meanings.


# END a finite negative tail is not a complete scene classification.




def test_fixed_original_geometry_values_and_sparse_jacobian_match_difference(tmp_path):
    base, controls = computation(tmp_path)
    geometry, terminal, bound = bind(base)
    controls = controls + np.arange(len(controls))[:, None]*np.array([.0003, -.0001])
    point = bound.evaluate(controls)
    expected = difference(bound, terminal, controls)
    np.testing.assert_allclose(dense(point), expected, atol=3e-7, rtol=2e-5)
    assert geometry.row_count == len(bound.records)
    assert len(bound.records) == len(base.parameters)*3
    assert geometry.describe()['fixed_sample_counts']['robot/first'] > 0
    np.testing.assert_array_equal(point.values, point.spatial_values)


# BEGIN an interior collision must contribute to the actual optimizer rows.
def test_native_interval_finds_collision_between_clear_endpoints_and_differentiates(tmp_path):
    from manipulation.cubic_constraints import CubicAffineMap
    base, _ = computation(tmp_path, field_value=1.)
    base.parameters = np.array([0., 1.])
    base.times = torch.tensor([[0., 1.2]], dtype=torch.float64)
    base.progress = torch.tensor([[0., 1.]], dtype=torch.float64)
    curve = CubicAffineMap([.04, -.02], [.16/1.2, 0.], 2, 1.2)
    base.domain._curve_map = curve
    controls = np.tile(curve.start_q, (2, 1))
    equations, targets = [], []
    for order, target in ((0, .04), (1, -.16/1.2)):
        weights, offset = curve.query(np.array([1.]), order)
        equations.append(weights[0]); targets.append(target-offset[0, 0])
    controls[:, 0] = np.linalg.solve(equations, targets)
    geometry, terminal, bound = bind(base)
    point = bound.evaluate(controls)
    rows = bound.native_rows['robot/first', 'robot/second']
    assert len(rows) == 2
    np.testing.assert_allclose(curve.evaluate(controls, [0., 1.])[:, 0], [.04, .04], atol=1e-15)
    # Coal's signed primitive distance includes its approximately 1 pm numeric offset.
    assert point.native_distances_m[rows[0]] == pytest.approx(-.01, abs=2e-12)
    assert point.query_times[rows[0]] == pytest.approx(.6, abs=1e-10)
    assert point.native_distances_m[rows[1]] > 0.
    assert point.values[rows[0]] > 0.
    np.testing.assert_allclose(dense(point)[rows], difference(bound, terminal, controls)[rows],
                               atol=2e-6, rtol=2e-5)
# END both physical endpoints were clear in the original finite-time provider.


# BEGIN the same interval row remains continuous as its minimum reaches a boundary.
def test_native_interval_keeps_right_endpoint_when_stationary_point_leaves(tmp_path):
    from manipulation.cubic_constraints import CubicAffineMap
    base, _ = computation(tmp_path, field_value=1.)
    base.parameters = np.array([0., 1.])
    base.times = torch.tensor([[0., 1.2]], dtype=torch.float64)
    base.progress = torch.tensor([[0., 1.]], dtype=torch.float64)
    curve = CubicAffineMap([.04, -.02], [.06, 0.], 2, 1.2)
    base.domain._curve_map = curve
    geometry, terminal, bound = bind(base)
    weights = np.vstack([curve.query(np.array([1.]), order)[0][0] for order in (0, 1)])
    offsets = np.array([curve.query(np.array([1.]), order)[1][0, 0] for order in (0, 1)])
    results = []
    for end_rate in (-1e-5, 0., 1e-5):
        controls = np.tile(curve.start_q, (2, 1))
        controls[:, 0] = np.linalg.solve(weights, np.array([.08, end_rate])-offsets)
        point = bound.evaluate(controls)
        rows = bound.native_rows['robot/first', 'robot/second']
        results.append(point.values[rows[0]])
        assert point.native_distances_m[rows[0]] == pytest.approx(-.01, abs=2e-9)
        if end_rate >= 0.:
            assert point.query_times[rows[0]] == 1.2
            np.testing.assert_allclose(dense(point)[rows[0]], dense(point)[rows[1]], atol=1e-12)
    assert np.ptp(results) < 4e-7
# END neighboring rows and a sign-changing root retain one numerical value limit.




def test_released_owner_derivative_uses_first_release_frame_and_sparse_columns(tmp_path):
    base, controls = computation(tmp_path)
    field = FrozenObservedField(-1e-6*np.ones((2, 2, 2)), np.ones((2, 2, 2), bool),
        weights=np.ones((2, 2, 2)), origin=[-.001]*3, voxel_size=.002,
        name='material', source='analytical observed fixture')
    base.scene.owners = (CollisionOwner('material', field,
        PosePath(np.array([0., 1.]), np.stack((np.eye(4), np.eye(4))), 'declared reference'),
        'observed material', moving=True, attachments=(Attachment((0., .5), [.055, -.2, 0.],
            [1., 0., 0., 0.], 'declared attachment', 'declared release'),)),)
    geometry, terminal, bound = bind(base)
    controls = controls + np.arange(len(controls))[:, None]*np.array([.0003, 0.])
    point = bound.evaluate(controls)
    row = next(i for i, record in enumerate(bound.records) if record['first'] == 'robot/second'
               and record['second'] == 'owner/material' and record['progress'] == 1.)
    column = int(bound.record_columns[row])
    assert bound.sources['owner/material', column] == ('left', 1)
    analytical = dense(point)
    np.testing.assert_allclose(analytical, difference(bound, terminal, controls), atol=2e-6, rtol=1e-5)
    assert np.any(analytical[row] != 0.)
    # The held material's current state at the final query is a different
    # coefficient row; release uses only the first exact event's row.
    assert not np.array_equal(bound.coefficients[1], bound.coefficients[column])


def test_original_mode_sides_keep_physical_clock_and_arrival_permission(tmp_path):
    base, _ = computation(tmp_path)
    base.scene.required_progress_anchors = (.5,)
    base.scene.contacts = (SimpleNamespace(actor='robot', interval=(.5, 1.)),)
    rows = _query_records(base, _ProgressMap(np.array([0., 1.]), np.array([0., 1.])))
    assert len(rows) == 5
    left, right = rows[-2:]
    assert left['time_s'] == right['time_s'] == .6
    assert left['progress'] < .5 < right['progress']
    assert left['permission_progress'] == .5
    assert right['permission_progress'] == right['progress']


def test_shared_context_reuses_completed_base_fk(tmp_path):
    base, controls = computation(tmp_path)
    _, terminal, bound = bind(base)
    arrays = [base.domain._curve_map.query(base.parameters, order) for order in (0, 1)]
    q, dq = [(torch.tensor(row) @ torch.tensor(controls) + torch.tensor(offset))[None]
             for row, offset in arrays]
    robot, jacobians = linearized_robot_state(base, q, dq)
    shared = dict(robot_state=robot, frame_jacobians=jacobians)
    expected = bound.evaluate(controls)
    def unexpected(*args, **kwargs):
        raise AssertionError('base FK must be reused')
    base.kinematics.states_with_jacobians = unexpected
    actual = bound.evaluate(controls, shared_context=shared)
    np.testing.assert_array_equal(actual.values, expected.values)
    np.testing.assert_array_equal(actual.jacobian_values(), expected.jacobian_values())
    assert bound.owner.last_evaluation['query_state_count'] == len(base.parameters)
# END fixed-query maxima and original event-clock regression checks.


# BEGIN native batching preserves active rows, their order and the original Jacobian.
def native_binding(tmp_path, monkeypatch):
    """Use real fixture geometry with shuffled ordered-pair and clock identities."""
    from manipulation.direct_planner import geometry_constraints as module
    monkeypatch.setattr(module, 'MAX_BATCH_SIZE', 2)
    base, controls = computation(tmp_path)
    geometry = GeometryConstraints(base,
        progress_map=_ProgressMap(np.array([0., 1.]), np.array([0., 1.])))
    template = next(row for row in geometry.records if row['component'] == 'collision/robot/self')
    states = [(1.2, 1., 1.), (.6, .5, .5), (0., 0., 0.),
              (.6, .5, np.nextafter(.5, np.inf)), (.3, .25, .25), (.9, .75, .75)]
    geometry.records = [{**template, 'time_s': time, 'progress': progress,
        'permission_progress': permission,
        'first': template['second'] if index % 2 else template['first'],
        'second': template['first'] if index % 2 else template['second']}
        for index, (time, progress, permission) in enumerate(states)]
    original_context = geometry._context

    def context(*args, **kwargs):
        result = original_context(*args, **kwargs)
        # A current hand-model uncertainty sample disables both orientations
        # of the same registered pair through the original _permission API.
        result['hand'] = SimpleNamespace(uncertain=torch.tensor([[row['permission_progress'] == .25
            for row in result['states']]], dtype=torch.bool))
        return result

    geometry._context = context
    geometry._unknown_geometries = {'first'}
    terminal = bind_terminal_hold(base.domain._curve_map, [0., 1.], [0., 1.], source='fixture clock')
    bound = geometry.bind_solver(terminal)
    calls = []
    for group in bound.native_groups:
        pair, table = group['pairs'], group['table']
        class RecordedTable:
            storage_nbytes = table.storage_nbytes

            def __init__(self, original, identity):
                self.original, self.identity = original, identity

            def query(self, poses, projection, request):
                calls.append(dict(pair=self.identity, poses=poses.copy(), projection=projection.copy()))
                return self.original.query(poses, projection, request)

        group['table'] = RecordedTable(table, pair)
    return base, geometry, terminal, bound, controls, calls


def test_native_batches_restore_original_rows_and_skip_only_unmodelled_states(tmp_path, monkeypatch):
    base, geometry, terminal, bound, controls, calls = native_binding(tmp_path, monkeypatch)
    controls = controls+np.arange(len(controls))[:, None]*np.array([.0003, -.0001])
    # BEGIN the reference native queries consume the same fixed clock binding.
    context = geometry._context(controls, bound)
    # END independent Coal calls below still supply the expected numeric results.
    expected, active = [], []
    for ordinal, (record, column) in enumerate(zip(bound.records, bound.record_columns)):
        _, mode = geometry._permission(context, record, int(column))
        if mode == 'active':
            request, result = coal.DistanceRequest(), coal.DistanceResult()
            request.enable_signed_distance = True
            first, second = record['first'], record['second']
            expected.append(-coal.distance(geometry.geometries[first],
                _transform(context['matrices'][first][0, column]), geometry.geometries[second],
                _transform(context['matrices'][second][0, column]), request, result)/bound.distance_unit)
            active.append(ordinal)
        else:
            expected.append(-1.)  # inactive finite inequalities are constant feasible rows
    point = bound.evaluate(controls)
    # BEGIN each active native interval includes both original physical endpoints.
    expected = np.array(expected)
    for intervals in bound.temporal.intervals.values():
        for left, right in intervals:
            if left in active and right in active:
                expected[left] = max(expected[left], expected[right])
    # END interior-root behavior is covered by the independent analytic tests.
    np.testing.assert_array_equal(point.spatial_values, np.maximum(expected, -1.))
    far = np.array(expected) < -1.
    assert np.all(point.native_distances_m[far] > bound.distance_unit)
    np.testing.assert_array_equal(point.factors[far], 0.)
    assert sum(len(call['poses']) for call in calls) == len(active) == 5
    assert all(len(call['poses']) <= 2 and not call['projection'].any() for call in calls)
    assert geometry.last_evaluation['inactive_count'] == 1
    assert geometry.last_evaluation['native_pair_queries_completed'] == 5
    assert bound.records[1]['time_s'] == bound.records[3]['time_s']
    assert bound.record_columns[1] != bound.record_columns[3]
    derivative = dense(point)
    np.testing.assert_array_equal(point.factors[4], np.zeros_like(point.factors[4]))
    np.testing.assert_allclose(derivative, difference(bound, terminal, controls), atol=2e-7, rtol=1e-5)
    assert all(group['poses'].nbytes == 2*2*16*8 for group in bound.native_groups)
    assert geometry.describe()['native_pair_table_count'] == 2


def test_native_batch_cancellation_preserves_original_exception_and_completed_work(tmp_path, monkeypatch):
    _, geometry, _, bound, controls, calls = native_binding(tmp_path, monkeypatch)
    failure = RuntimeError('original request budget boundary')

    def check():
        if calls:
            raise failure

    geometry.scene.check_request = check
    with pytest.raises(RuntimeError) as caught:
        bound.evaluate(controls)
    assert caught.value is failure
    assert len(calls) == 1 and len(calls[0]['poses']) == 2
    report = geometry.last_evaluation
    assert report['state'] == 'producing'
    assert report['native_batches_submitted'] == report['native_batches_completed'] == 1
    assert report['native_pair_queries_submitted'] == report['native_pair_queries_completed'] == 2
    assert report['seconds']['native_query'] > 0.
# END all batches use the installed NativePairTable and original scalar Coal oracle.


# BEGIN equal-clock pairs exercise the existing native multi-pair interface.
def test_native_equal_clock_pairs_share_calls_and_preserve_derivatives(tmp_path, monkeypatch):
    from manipulation.direct_planner import geometry_constraints as module
    monkeypatch.setattr(module, 'MAX_BATCH_SIZE', 2)
    base, controls = computation(tmp_path)
    geometry = GeometryConstraints(base,
        progress_map=_ProgressMap(np.array([0., 1.]), np.array([0., 1.])))
    originals = [row for row in geometry.records if row['component'] == 'collision/robot/self']
    geometry.records = [entry for row in originals for entry in
        (row, {**row, 'first': row['second'], 'second': row['first']})]
    terminal = bind_terminal_hold(base.domain._curve_map, [0., 1.], [0., 1.], source='fixture clock')
    bound = geometry.bind_solver(terminal)
    assert len(bound.native_groups) == 1
    group = bound.native_groups[0]
    original_table, calls = group['table'], []

    class RecordedTable:
        def query(self, poses, projection, request):
            result = original_table.query(poses, projection, request)
            calls.append((poses.copy(), projection.copy(), result.copy()))
            return result

    group['table'] = RecordedTable()
    controls = controls+np.arange(len(controls))[:, None]*np.array([.0003, -.0001])
    point = bound.evaluate(controls)
    assert len(calls) == 2
    assert sum(result.shape[0]*result.shape[1] for _, _, result in calls) == 6
    for poses, projection, numeric in calls:
        assert not projection.any()
        for state in range(len(poses)):
            for pair_index, (first, second) in enumerate(group['pairs']):
                request, result = coal.DistanceRequest(), coal.DistanceResult()
                request.enable_signed_distance = True
                a, b = group['indices'][pair_index]
                distance = coal.distance(geometry.geometries[first], _transform(poses[state, a]),
                    geometry.geometries[second], _transform(poses[state, b]), request, result)
                assert numeric[state, pair_index, 0] == distance
    np.testing.assert_allclose(dense(point), difference(bound, terminal, controls),
                               atol=2e-7, rtol=1e-5)
# END batching changes call count, while original Coal and finite differences are oracles.
