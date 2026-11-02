"""Actual IPOPT calls, sparse delivery and original interruption types.

The analytical fixture implements the same fixed problem protocol as the
trajectory computation. Native cubic/robot checks live beside these interface
checks; no scene geometry or robot execution is replaced by this fixture.
"""

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import casadi as ca

EXPERIMENT = Path(__file__).resolve().parent
PROJECT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(Path(__file__).resolve().parents[2]),
                 str(Path(__file__).resolve().parents[2] / 'tests'),
                 str(Path(__file__).resolve().parents[2] / 'tests/planning')]

from manipulation.planning import MotionPlanningError, PlanningWorkExhausted
from manipulation.direct_planner.shared_solver import _Oracle, solve_trajectory
from manipulation.direct_planner.sparse_rows import SparseRows


class IdentityTerminal:
    @staticmethod
    def expand(value):
        return np.asarray(value).copy()


class QuadraticComputation:
    """A declared equality and signed bounds with minimizer (1,.5).

    This is the new numerical protocol used by lifecycle tests as well. It contains no
    geometry query; one bound object supplies a fixed sparse row layout.
    """

    def __init__(self):
        self.max_host_rss_bytes = 4*1024**3
        self.terminal = IdentityTerminal()
        self.geometry = SimpleNamespace()
        self.trial_history = []
        self.evaluated = []
        self.fixed_feasible = True

    def bind_solver(self, initial, *, check_request):
        owner = self
        pattern = SparseRows(([0, 1], [0], [1]), 2)

        class Point:
            def __init__(self, controls):
                x, y = np.asarray(controls).ravel()
                self.coordinates = np.array([x, y])
                self.objective_value = (x-1.)**2+2.*(y-.5)**2
                self.objective_components = {'known_quadratic': self.objective_value}
                # BEGIN exact-point relation provenance has one current schema.
                self.scene_relations = {'same_controls': [[float(x), float(y)]]}
                # END the quadratic fixture carries no physical relation claim.
                self.constraint_values = np.array([x-2.*y, -x-.5, -y-.5])

            def objective_gradient(self):
                check_request()
                x, y = self.coordinates
                return np.array([2.*(x-1.), 4.*(y-.5)])

            def jacobian_values(self):
                check_request()
                return np.array([1., -2., -1., -1.])

        class Bound:
            def __init__(self):
                self.computation = owner
                self.shape = np.asarray(initial).shape
                self.pattern = pattern
                self.geometry = SimpleNamespace(row_keys=())
                self.problem_identity = (id(owner),)
                self.row_keys = (('equality', 'coupling'), ('bound', 'x'), ('bound', 'y'))
                self.lower = np.array([0., -np.inf, -np.inf])
                self.upper = np.zeros(3)
                self.fixed_feasible = owner.fixed_feasible
                self.report = {'solver_task_equalities': 1, 'solver_task_inequalities': 2,
                               'solver_jacobian_nonzeros': pattern.nnz}
                # BEGIN the fixed-problem protocol records lifecycle snapshots.
                self.producer_snapshots = {}

            def snapshot_producers(self, label):
                self.producer_snapshots[label] = {'evaluations': len(owner.evaluated)}
            # END this analytical producer records its actual evaluation count.

            def evaluate(self, controls):
                check_request()
                owner.evaluated.append(np.asarray(controls).copy())
                return Point(controls)

            # BEGIN a nontrivial map checks original-coordinate solver delivery.
            def solver_coordinates(self, point):
                return np.array([[.25, .1], [0., 2.]]), {'kind': 'declared_analytical_coordinates'}
            # END the analytical objective and constraint values are unchanged.

        return Bound()


def run(computation, *, check=lambda: None, started=lambda: None, report=None, initial=None):
    initial = np.zeros((1, 2)) if initial is None else initial
    bound = computation.bind_solver(initial, check_request=check)
    return solve_trajectory(bound, initial, iterations=40,
        check_request=check, remaining_seconds=lambda: 5., on_solver_start=started,
        pending_geometry_bytes=0, report_out={} if report is None else report)


def test_feasible_start_still_optimizes_the_declared_quadratic():
    computation, starts = QuadraticComputation(), []
    result = run(computation, started=lambda: starts.append(True))
    np.testing.assert_allclose(result.controls, [[1., .5]], atol=2e-5)
    assert starts == [True]
    assert result.report['success'] and result.report['status'] == 'Solve_Succeeded'
    assert result.report['termination'] == 'optimizer_stopping_condition'
    assert result.report['backend'] == 'casadi.nlpsol/ipopt'
    assert result.report['solver_jacobian_nonzeros'] == 4
    assert result.report['same_point_hits'] > 0
    assert len(computation.evaluated) == result.report['point_evaluations']
    assert len(computation.trial_history) == 1
    assert 'slsqp_workspaces' not in result.report['solver_storage_bytes']


def test_derivative_first_and_repeat_values_share_exact_point_and_sparse_ccs():
    computation = QuadraticComputation()
    owner = _Oracle(computation.bind_solver(np.zeros((1, 2)), check_request=lambda: None), lambda: None)
    jacobian = owner.jacobian()
    point = np.array([.2, .7])
    gradient, sparse = jacobian(point, 0., np.zeros(3))
    owner(point)
    jacobian(point.copy(), 0., np.zeros(3))
    assert len(computation.evaluated) == 1
    assert sparse.nnz() == 4
    np.testing.assert_array_equal(np.asarray(sparse), [[1., -2.], [-1., 0.], [0., -1.]])
    np.testing.assert_allclose(np.asarray(gradient).ravel(), [-1.6, .8])
    point[0] += .1
    owner(point)
    assert len(computation.evaluated) == 2


# BEGIN the delivered feasible point is independent of subsequent trial quality.
def test_original_feasible_incumbent_improves_without_accepting_later_infeasibility():
    computation = QuadraticComputation()
    owner = _Oracle(computation.bind_solver(np.zeros((1, 2)), check_request=lambda: None), lambda: None)
    owner.point(np.array([0., 0.]))
    first = owner.best_feasible
    owner.point(np.array([.6, .3]))
    improved = owner.best_feasible
    assert improved.objective_value < first.objective_value
    owner.point(np.array([.9, .5]))  # lower objective, but the declared equality is violated
    assert owner.best_feasible is improved
    owner.point(np.array([.2, .1]))  # feasible, higher objective
    assert owner.best_feasible is improved
    assert len(owner.feasible_improvements) == 2
    np.testing.assert_array_equal(owner.best_feasible_record['controls'], [[.6, .3]])
# END source coefficients and retained original values refer to the same point.


def test_fixed_preflight_rejection_does_not_charge_ipopt():
    computation, starts = QuadraticComputation(), []
    computation.fixed_feasible = False
    result = run(computation, started=lambda: starts.append(True))
    assert not starts and not computation.evaluated
    assert result.report['status'] == 'Fixed_Constraint_Infeasible'
    assert not result.report['solver_called']


@pytest.mark.parametrize('error', [PlanningWorkExhausted('request work exhausted'),
    MotionPlanningError('cancelled', 'explicit cancellation'), RuntimeError('producer failure')])
def test_callback_exception_is_rethrown_with_original_type_and_identity(error):
    computation, report, starts = QuadraticComputation(), {}, []

    def check():
        if starts:
            raise error

    with pytest.raises(type(error)) as caught:
        run(computation, check=check, started=lambda: starts.append(True), report=report)
    assert caught.value is error
    assert starts == [True] and report['solver_called']
    assert report['state'] == 'interrupted'
    assert report['abort_record']['error_type'] == type(error).__name__
    assert report['reason'] == str(error)
    assert report['latest_complete_trial']['objective'] == pytest.approx(1.5)


# BEGIN explicit physical scales are independent of numerical initialization.
def test_distinct_initial_values_keep_declared_scale_and_feasible_result():
    computation = QuadraticComputation()
    first = run(computation)
    second = run(computation, initial=np.array([[3., -2.]]))
    for result in (first, second):
        assert result.report['objective_scale_source'] == 'explicit_physical_residual_scales'
        assert result.report['nlp_scaling_method'] == 'declared_original_units'
        point = computation.bind_solver(result.controls, check_request=lambda: None).evaluate(result.controls)
        assert abs(point.constraint_values[0]) < 1e-12
        assert np.all(point.constraint_values[1:] <= 0.)
    np.testing.assert_array_equal(computation.final_point.coordinates, second.controls.ravel())
# END the final point is the same numerical result handed to the candidate.
