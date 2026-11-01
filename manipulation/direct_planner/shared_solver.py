"""One IPOPT backend for fixed sparse trajectory problems.

Original computation producers own every objective and signed condition.
This adapter owns callback lifetime, exact-x reuse, CCS delivery, numerical
termination and original request exceptions across the Python/C++ boundary.
Complete scene qualification remains the caller's existing candidate exit.
"""

from time import perf_counter

import casadi as ca
import numpy as np

from manipulation.resource_admission import require_host_memory
from manipulation.unified_solver import ConstrainedSolution


# BEGIN fixed sparse callbacks with explicit first-order and request ownership.
class _Jacobian(ca.Callback):
    """Deliver the original first derivatives in precompiled CCS order."""

    def __init__(self, name, owner, input_names, output_names):
        ca.Callback.__init__(self)
        self.owner = owner
        self.input_names, self.output_names = input_names, output_names
        self.construct(name, {'enable_fd': False})

    def get_n_in(self):
        return 3

    def get_n_out(self):
        return 2

    def get_name_in(self, index):
        return self.input_names[index]

    def get_name_out(self, index):
        return self.output_names[index]

    def get_sparsity_in(self, index):
        return ca.Sparsity.dense((self.owner.variables, 1, self.owner.constraints)[index], 1)

    def get_sparsity_out(self, index):
        return self.owner.patterns[index]

    def eval(self, arguments):
        def evaluate():
            self.owner.counts['jacobian_callbacks'] += 1
            point = self.owner.point(np.asarray(arguments[0]).ravel())
            self.owner.latest['task_jacobian_status'] = 'running'
            gradient = point.objective_gradient()
            values = point.jacobian_values()[self.owner.bound.pattern.ccs_order]
            self.owner.latest['task_jacobian_status'] = 'completed'
            return [ca.DM(self.owner.patterns[0], gradient), ca.DM(self.owner.patterns[1], values)]
        return self.owner.guard(evaluate)


class _Oracle(ca.Callback):
    """Keep values, derivatives and failure reasons attached to the exact x."""

    def __init__(self, bound, check_request):
        ca.Callback.__init__(self)
        self.bound, self.check_request = bound, check_request
        self.variables = bound.pattern.shape[1]
        self.constraints = bound.pattern.shape[0]
        self.patterns = (ca.Sparsity.dense(1, self.variables), ca.Sparsity.triplet(
            self.constraints, self.variables, bound.pattern.rows.tolist(), bound.pattern.columns.tolist()))
        self.children = []
        self.cache_x = self.cache = None
        self.latest = self.last_complete = self.last_iterate = None
        self.first_feasible_iterate = None
        # BEGIN one incumbent belongs to this exact fixed numerical problem.
        self.best_feasible = self.best_feasible_record = None
        self.feasible_improvements = []
        # END later line-search or iterate values cannot replace it by infeasibility.
        # BEGIN retain original inequality evidence independently of IPOPT slacks.
        self.iteration_records, self.started = [], perf_counter()
        # END these passive values introduce no additional scene evaluation.
        self.abort_exception = self.abort_record = None
        self.counts = {'value_callbacks': 0, 'jacobian_callbacks': 0, 'point_evaluations': 0,
                       'same_point_hits': 0, 'callback_iterations': 0}
        self.construct('trajectory_oracle', {'enable_fd': False})

    def get_n_in(self):
        return 1

    def get_n_out(self):
        return 2

    def get_name_in(self, index):
        return 'x'

    def get_name_out(self, index):
        return ('f', 'g')[index]

    def get_sparsity_in(self, index):
        return ca.Sparsity.dense(self.variables, 1)

    def get_sparsity_out(self, index):
        return ca.Sparsity.dense((1, self.constraints)[index], 1)

    def has_jac_sparsity(self, output, input_):
        return True

    def get_jac_sparsity(self, output, input_, symmetric):
        return self.patterns[output]

    def has_jacobian(self):
        return True

    def get_jacobian(self, name, input_names, output_names, options):
        child = _Jacobian(name, self, input_names, output_names)
        self.children.append(child)
        return child

    def record_abort(self, error):
        """Preserve the first Python cause before CasADi translates the error."""
        if self.abort_exception is None:
            self.abort_exception = error
            self.abort_record = {'error_type': type(error).__name__, 'reason': str(error),
                                 'trial_index': None if self.latest is None else self.latest['trial_index']}
            if self.latest is not None and self.latest['task_jacobian_status'] == 'running':
                self.latest['task_jacobian_status'] = 'interrupted'

    def guard(self, function):
        try:
            if self.abort_exception is not None:
                raise self.abort_exception
            self.check_request()
            result = function()
            self.check_request()
            return result
        except BaseException as error:
            self.record_abort(error)
            raise

    def point(self, flat):
        """Create only complete same-problem cache entries, never partial ones."""
        self.check_request()
        if self.cache_x is not None and np.array_equal(flat, self.cache_x):
            self.counts['same_point_hits'] += 1
            return self.cache
        self.cache_x = self.cache = None
        self.counts['point_evaluations'] += 1
        self.latest = {'trial_index': self.counts['point_evaluations'],
                       'controls': flat.reshape(self.bound.shape).copy(),
                       'completed_components': [], 'task_jacobian_status': 'not_requested'}
        point = self.bound.evaluate(flat.reshape(self.bound.shape))
        violation = np.maximum(self.bound.lower-point.constraint_values,
                               point.constraint_values-self.bound.upper)
        largest = np.argsort(violation)[-5:][::-1]
        self.latest.update(completed_components=['objective', 'task', 'geometry', 'physical'],
                           objective=point.objective_value, objective_components=point.objective_components,
                           scene_relations=point.scene_relations,
                           largest_constraints=[dict(identity=self.bound.row_keys[index],
                               value=float(point.constraint_values[index]), violation=float(violation[index]))
                               for index in largest])
        self.last_complete = self.latest
        self.cache_x, self.cache = flat.copy(), point
        # BEGIN compare only fully computed original conditions and objective.
        feasible = (np.isfinite(point.objective_value) and np.isfinite(point.constraint_values).all()
            and np.all(point.constraint_values >= self.bound.lower)
            and np.all(point.constraint_values <= self.bound.upper))
        if feasible and (self.best_feasible is None
                or point.objective_value < self.best_feasible.objective_value):
            self.best_feasible, self.best_feasible_record = point, self.latest
            self.feasible_improvements.append(dict(trial_index=self.latest['trial_index'],
                elapsed_seconds=perf_counter()-self.started,
                objective=point.objective_value,
                maximum_original_violation=float(violation.max()),
                reduced_controls=flat.reshape(self.bound.shape).tolist()))
        # END the optimizer continues its declared work after each improvement.
        return point

    def eval(self, arguments):
        def evaluate():
            self.counts['value_callbacks'] += 1
            point = self.point(np.asarray(arguments[0]).ravel())
            return [point.objective_value, point.constraint_values]
        return self.guard(evaluate)

    @staticmethod
    def snapshot(record):
        if record is None:
            return None
        return {**record, 'controls': record['controls'].tolist()}


class _Iteration(ca.Callback):
    """Retain the current iterate and stop for the original request boundary."""

    def __init__(self, owner, coordinate_map):
        ca.Callback.__init__(self)
        self.owner = owner
        self.coordinate_map = coordinate_map
        self.construct('trajectory_iteration', {})

    def get_n_in(self):
        return ca.nlpsol_n_out()

    def get_n_out(self):
        return 1

    def get_name_in(self, index):
        return ca.nlpsol_out(index)

    def get_sparsity_in(self, index):
        size = {'x': self.owner.variables, 'f': 1, 'g': self.owner.constraints,
                'lam_x': self.owner.variables, 'lam_g': self.owner.constraints, 'lam_p': 0}[ca.nlpsol_out(index)]
        return ca.Sparsity.dense(size, 1)

    def get_sparsity_out(self, index):
        return ca.Sparsity.scalar()

    def eval(self, arguments):
        # BEGIN iterator evidence is always expressed in original coefficients.
        flat = np.asarray(self.coordinate_map(arguments[0])).ravel()
        # END IPOPT's numerical coordinates are not a different physical curve.
        self.owner.counts['callback_iterations'] += 1
        matched = (self.owner.latest if self.owner.latest is not None
                   and np.array_equal(self.owner.latest['controls'].ravel(), flat) else None)
        self.owner.last_iterate = {'callback_index': self.owner.counts['callback_iterations'],
            'controls': flat.reshape(self.owner.bound.shape).copy(),
            'trial_index': None if matched is None else matched['trial_index'],
            'completed_components': [] if matched is None else list(matched['completed_components']),
            'task_jacobian_status': None if matched is None else matched['task_jacobian_status'],
            'largest_constraints': [] if matched is None else matched['largest_constraints']}
        try:
            self.owner.check_request()
        except BaseException as error:
            self.owner.record_abort(error)
            return [1]
        # BEGIN feasibility is recorded independently of objective convergence.
        # A feasible curve can still concentrate an avoidable tool turn; the
        # ordinary optimizer stopping rule also considers objective progress.
        values = np.asarray(arguments[2]).ravel()
        feasible = (np.isfinite(values).all() and np.all(values >= self.owner.bound.lower)
                    and np.all(values <= self.owner.bound.upper))
        # BEGIN original conditions and physical objective label each iteration.
        violation = np.maximum(self.owner.bound.lower-values, values-self.owner.bound.upper)
        self.owner.iteration_records.append(dict(
            callback_index=self.owner.counts['callback_iterations'],
            elapsed_seconds=perf_counter()-self.owner.started,
            objective=float(arguments[1]), maximum_original_violation=float(violation.max()),
            original_conditions_feasible=bool(feasible)))
        # END IPOPT's internal slack residual remains in its own statistics.
        if feasible and self.owner.first_feasible_iterate is None:
            self.owner.first_feasible_iterate = self.owner.last_iterate.copy()
        return [int(self.owner.abort_exception is not None)]
        # END no extra geometric or task query is performed by this observer.
# END no complete scene query or geometry-layout mutation occurs in callbacks.


def sparse_solver_storage_bytes(*, variables, constraints, nonzeros):
    """Count owned numeric buffers and a sparse solver work estimate.

    The symbolic factorization's fill depends on the submitted pattern and the
    bundled linear solver; its measured process RSS remains the runtime value.
    This estimate accounts callback/cache arrays, sparse numeric copies,
    primal/dual vectors and IPOPT's six-pair limited-memory history.
    """
    n, m, z = map(int, (variables, constraints, nonzeros))
    return {'callback_values_gradients_and_slots': 8*(m+3*n+2*z),
            'ipopt_sparse_values_and_vectors': 8*(4*z+24*(n+m)),
            'limited_memory_pairs': 8*12*n}


def solve_trajectory(bound, initial, *, iterations, check_request, remaining_seconds,
                     on_solver_start, pending_geometry_bytes, report_out, geometry_workspace=None,
                     ):
    """Solve one fixed problem and return the original ConstrainedSolution.

    The supplied bound problem also owns numerical proposal scoring.
    Preflight and graph construction do not call on_solver_start. The caller
    increments its request ledger at that hook immediately before the actual
    nlpsol call. Original wall-time and cancellation checks run at every callback and
    producer boundary, including every line-search query.
    """
    computation = bound.computation
    initial = np.asarray(initial, dtype=float)
    report, started, owner = report_out, perf_counter(), None
    report.update(backend='casadi.nlpsol/ipopt', state='binding', solver_called=False,
                  success=False, callback_iterations=0, latest_trial=None,
                  latest_complete_trial=None, last_callback_iterate=None,
                  hessian_approximation='limited-memory', limited_memory_history=6)
    solver = None
    try:
        check_request()
        require_host_memory(int(pending_geometry_bytes), pending_existing_bytes=0,
                            max_rss_bytes=computation.max_host_rss_bytes)
        report.update(bound.report)
        if not bound.fixed_feasible:
            report.update(state='completed', status='Fixed_Constraint_Infeasible',
                          message='A declared constant task row is infeasible', iterations=0,
                          objective=None, maximum_control_update=0., unchanged_from_seed=True)
            return ConstrainedSolution(initial.copy(), report)
        owner = _Oracle(bound, check_request)
        point = owner.guard(lambda: owner.point(initial.ravel()))
        gradient = owner.guard(point.objective_gradient)
        # BEGIN the declared physical residual scales are the solver scales.
        report.update(initial_objective=point.objective_value,
                      initial_gradient_linf=float(np.abs(gradient).max()),
                      objective_scale_source='explicit_physical_residual_scales',
                      nlp_scaling_method='declared_original_units')
        # END neither the current seed nor a previous solve defines a scale.
        memory = sparse_solver_storage_bytes(variables=owner.variables,
            constraints=owner.constraints, nonzeros=bound.pattern.nnz)
        report.update(solver_storage_bytes=memory,
                      resident_sparse_pattern_bytes=bound.pattern.storage_bytes,
                      solver_memory_scope='owned sparse buffers plus estimate; native factorization fill is process RSS')
        require_host_memory(sum(memory.values()), pending_existing_bytes=0,
                            max_rss_bytes=computation.max_host_rss_bytes)
        variables = ca.MX.sym('trajectory_controls', owner.variables)
        # BEGIN a fixed invertible map conditions the same physical problem.
        transform, coordinate_report = bound.solver_coordinates(point)
        memory['coefficient_coordinate_matrix'] = transform.nbytes
        coordinate_map = ca.Function('original_trajectory_controls', [variables],
            [ca.DM(initial.ravel())+ca.sparsify(ca.DM(transform))@variables])
        report['coefficient_coordinates'] = coordinate_report
        observer = _Iteration(owner, coordinate_map)
        objective, constraints = owner(coordinate_map(variables))
        # END all objective and row values still come from the original oracle.
        check_request()
        seconds_left = float(remaining_seconds())
        if seconds_left <= 0.:
            # The original request check owns the exception type and reason.
            check_request()
            raise ValueError('The request supplied no positive numerical wall-time allowance')
        # BEGIN the maintained interior-point method owns numerical convergence.
        # External registered geometry supplies first derivatives. Its complete
        # Lagrangian curvature is therefore approximated by IPOPT's own update.
        # Task tolerances remain in the submitted rows. The acceptable stopping
        # rule additionally requires sustained small objective changes; the
        # first feasible connection is only a recorded numerical milestone.
        options = {'print_time': False, 'error_on_fail': False,
                   'ipopt.print_level': 0, 'ipopt.sb': 'yes',
                   # BEGIN use the measured sparse symmetric factorization.
                   'ipopt.linear_solver': 'mumps',
                   # END the nonlinear problem and IPOPT method are unchanged.
                   'ipopt.max_iter': int(iterations), 'ipopt.max_wall_time': seconds_left,
                   'ipopt.hessian_approximation': 'limited-memory',
                   'ipopt.limited_memory_max_history': 6,
                   'ipopt.mu_strategy': 'adaptive', 'ipopt.nlp_scaling_method': 'none',
                   'ipopt.bound_relax_factor': 0.,
                   'ipopt.tol': 1e-6, 'ipopt.constr_viol_tol': 1e-9,
                   # BEGIN acceptable optimality and original feasibility differ.
                   'ipopt.acceptable_tol': 1., 'ipopt.acceptable_dual_inf_tol': 1.,
                   'ipopt.acceptable_constr_viol_tol': 1e-9,
                   'ipopt.acceptable_compl_inf_tol': 1e-3,
                   'ipopt.acceptable_obj_change_tol': 1e-3, 'ipopt.acceptable_iter': 3,
                   # END candidate task/geometry bounds remain their original zero.
                   'iteration_callback': observer}
        report['numerical_termination'] = {name: value for name, value in options.items()
            if name.startswith('ipopt.acceptable_') or name in ('ipopt.tol', 'ipopt.constr_viol_tol')}
        # END numerical convergence is separate from original metric acceptance.
        arguments = {'x0': np.zeros(initial.size), 'lbg': bound.lower, 'ubg': bound.upper}
        solver = ca.nlpsol('trajectory', 'ipopt', {'x': variables, 'f': objective, 'g': constraints}, options)
        check_request()
        on_solver_start()
        report.update(state='running', solver_called=True)
        try:
            result = solver(**arguments)
        except BaseException:
            if owner.abort_exception is not None:
                raise owner.abort_exception
            raise
        stats = solver.stats()
        report['optimizer_statistics'] = stats
        if owner.abort_exception is not None:
            raise owner.abort_exception
        check_request()
        returned_controls = np.asarray(coordinate_map(result['x'])).reshape(initial.shape)
        returned_point = owner.guard(lambda: owner.point(returned_controls.ravel()))
        # BEGIN deliver the lowest objective among this solve's feasible points.
        # The original point owns every value needed by the single candidate
        # outlet. Solver termination remains its own reported numerical fact.
        if owner.best_feasible is not None:
            computation.final_point = owner.best_feasible
            matched = owner.best_feasible_record
            controls = matched['controls'].copy()
        else:
            computation.final_point = returned_point
            matched, controls = owner.latest, returned_controls
        # END an all-infeasible solve retains a diagnostic candidate for rejection.
        report.update(state='completed', success=bool(stats['success']), status=stats['return_status'],
            message=stats['return_status'], iterations=int(stats['iter_count']),
            termination='optimizer_stopping_condition',
            first_feasible_callback_index=None if owner.first_feasible_iterate is None
                else owner.first_feasible_iterate['callback_index'],
            objective=computation.final_point.objective_value,
            numerical_return_objective=float(result['f']),
            selected_feasible=owner.best_feasible is not None,
            selected_trial_index=matched['trial_index'],
            candidate_selection='Lowest original objective among fully computed feasible points of this solve',
            maximum_control_update=float(np.max(np.abs(controls-initial))),
            unchanged_from_seed=bool(np.array_equal(controls, initial)),
            # BEGIN exact-x scene relationship provenance replaces removed pair cost.
            scene_relations={} if matched is None else matched['scene_relations'],
            scene_relation_trial_index=None if matched is None else matched['trial_index'],
            # END geometry inequalities remain in the fixed constraint records.
            objective_components={} if matched is None else matched['objective_components'])
        # One completed numerical boundary owns detailed coefficient provenance.
        computation.trial_history.append({'objective_query_index': len(computation.trial_history),
            'controls': computation.terminal.expand(controls).tolist(),
            'objective_components': report['objective_components'],
            'geometry_constraint_rows': len(bound.geometry.row_keys),
            'classification': 'solver_return_candidate_not_qualified'})
        return ConstrainedSolution(controls.copy(), report)
    except BaseException as error:
        report.update(state='interrupted', interrupted=True, error_type=type(error).__name__, reason=str(error))
        raise
    finally:
        # Completed calls own native solver statistics. Interrupted calls
        # retain the original producer exception and already completed points.
        if owner is not None:
            report.update(owner.counts, latest_trial=owner.snapshot(owner.latest),
                          latest_complete_trial=owner.snapshot(owner.last_complete),
                          last_callback_iterate=owner.snapshot(owner.last_iterate),
                          abort_record=owner.abort_record,
                          iteration_records=owner.iteration_records,
                          first_feasible_callback_index=None if owner.first_feasible_iterate is None
                            else owner.first_feasible_iterate['callback_index'],
                          feasible_improvements=owner.feasible_improvements)
            # Solver lifetime has ended. Drop its native references before
            # breaking Python owner/derivative cycles and any final trial graph.
            solver = None
            owner.cache = owner.cache_x = None
            owner.children.clear()
            owner.abort_exception = None
        report['elapsed_seconds'] = perf_counter()-started
