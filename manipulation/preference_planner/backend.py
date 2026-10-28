"""Preference planner using the existing robot, task and scene interfaces.

The independent request lifecycle retains original accounting, connection search
and binding APIs. This backend binds event-aware native setup, a common joint
objective, terminal equations and observed-geometry constraint updates.
The output remains the original JointTrajectory or the original failed result.
"""

from copy import deepcopy
from threading import get_ident

import numpy as np
from scipy.interpolate import PPoly
import torch

from manipulation.planning import (
    MotionBackend, MotionPlanningError, _ConstrainedPlanner, _ProgressMap,
    _phase_aligned_parameters, _planning_workspace_bytes,
)
from manipulation.unified_solver import ConstrainedSolution
from manipulation.completed_evidence import completed_evidence

from .geometry import GeometryConstraints
from .joint_computation import JointComputation
# BEGIN explicit event priors share initialization, solving and comparison APIs.
from .native_setup import NativeSetup
from .objective import trajectory_objective_components
from .pose_roles import pose_objective
# END the original full reference remains part of the problem and initializer.
# BEGIN local derivative adapter; the original SciPy solver and lifecycle are retained.
from .solver import solve_trajectory
# END explicit adapter import.
from .terminal import TerminalHold
from .termination_clock import terminal_clock



def _hold_polynomial_report(trajectory, start_s):
    """Compute complete suffix position/rate extrema of the saved polynomial.

    Each scalar cubic's velocity roots locate position extrema; acceleration
    roots locate velocity extrema.  The original breakpoints and requested
    suffix boundaries are also queried.  This is an independent check of the
    exported curve, using the same existing numerical joint/rate tolerances.
    """
    end = trajectory.duration
    if start_s == end:
        return dict(start_s=float(start_s), end_s=float(end),
                    maximum_position_change_rad=0., maximum_velocity_rad_s=0.,
                    position_witness=None, velocity_witness=None)
    maximum_position = maximum_velocity = 0.
    position_witness = velocity_witness = None
    for column, name in enumerate(trajectory.joint_names):
        position = PPoly(trajectory._curve.c[..., column], trajectory._curve.x, extrapolate=False)
        velocity = position.derivative()
        for quantity, derivative in (('position', velocity), ('velocity', velocity.derivative())):
            roots = derivative.roots(extrapolate=False)
            times = np.unique(np.r_[start_s, end, position.x,
                                    roots[np.isfinite(roots)]])
            times = times[(times >= start_s) & (times <= end)]
            values = (np.abs(position(times) - position(start_s)) if quantity == 'position'
                      else np.abs(velocity(times)))
            index = int(np.argmax(values))
            value = float(values[index])
            witness = dict(joint=name, time_s=float(times[index]), value=value)
            if quantity == 'position' and value > maximum_position:
                maximum_position, position_witness = value, witness
            elif quantity == 'velocity' and value > maximum_velocity:
                maximum_velocity, velocity_witness = value, witness
    return dict(start_s=float(start_s), end_s=float(end),
                maximum_position_change_rad=maximum_position,
                maximum_velocity_rad_s=maximum_velocity,
                position_witness=position_witness, velocity_witness=velocity_witness)


class PreferencePlanner(NativeSetup, _ConstrainedPlanner):
    """Use declared pose priors, terminal equations and observed-model rows."""

    def __init__(self, *args, incoming_terminal_contract=None, **kwargs):
        self.incoming_terminal_contract = incoming_terminal_contract
        self.binding_continuation = kwargs.get('continuation')
        super().__init__(*args, **kwargs)

    def rebind(self, problem, scene, budget, *, continuation=None, goal_identities=None):
        """Reset request-local computations through the original rebinding API."""
        self.binding_continuation = continuation
        return super().rebind(problem, scene, budget, continuation=continuation,
                              goal_identities=goal_identities)

    def _bind_computation(self, extra_parameters=()):
        """Bind one original full calculation and its explicit reduced problem."""
        super()._bind_computation(extra_parameters)
        base = self.computation
        u, z, source = terminal_clock(problem=self.problem, budget=self.budget,
            progress_map=self.progress_map, reference_time_map=self.reference_time_map,
            duration=self.formal_clock['duration'], continuation=self.binding_continuation,
            incoming_terminal_contract=self.incoming_terminal_contract)
        self.terminal = TerminalHold.from_clock(self.physical_domain._curve_map, u, z, source=source)
        # BEGIN the explicit functional event geometry defines pose preferences.
        self.pose_objectives = [dict(goal_id=['pose_events', index], **pose_objective(event))
                                for index, event in enumerate(self.problem.pose_events)]
        self.geometry_constraints = GeometryConstraints(base)
        self.computation = JointComputation(base=base, terminal=self.terminal,
                                             geometry=self.geometry_constraints)
        # END model geometry defines collision queries independently of preference scales.
        self.solve_history, self.candidate_history = [], []
        self._current_control_curve = self._current_full_controls = None
        self._candidate_evidence_scene = None
        self._candidate_added_rows = 0
        self.last_candidate_report = None
        self.route = 'preference_pose_prior_terminal_geometry'

    def _controls_curve(self, actions):
        """Keep the full coefficients associated with this exact exported curve."""
        trajectory = super()._controls_curve(actions)
        self._current_control_curve = trajectory
        self._current_full_controls = (actions.detach().cpu().numpy().copy()
            if isinstance(actions, torch.Tensor) else np.asarray(actions, float).copy())
        return trajectory

    def reference_seeds(self, joint_reference=None):
        """Project the original numerical initializer into the declared subspace."""
        seeds, counts = super().reference_seeds(joint_reference=joint_reference)
        original = seeds.detach().cpu().numpy()
        projected = self.terminal.expand(self.terminal.reduce(original))
        self.reference_initialization['reference_termination'] = {
            **self.terminal.report,
            'original_full_controls': original[0].tolist(),
            'projected_full_controls': projected[0].tolist(),
            'maximum_control_projection_rad': float(np.max(np.abs(projected - original))),
            'scope': 'numerical initializer projection; original task and geometry checks still apply',
        }
        seeds = self.torch.as_tensor(projected, dtype=seeds.dtype, device=seeds.device)
        self.initialization_candidate = self._controls_curve(seeds[0, 0])
        return seeds, counts

    def _candidate_metrics(self, trajectory, candidate_scene, *, record_work):
        """Consume the same completed entity evidence for necessary-row updates."""
        result = super()._candidate_metrics(trajectory, candidate_scene, record_work=record_work)
        self._candidate_evidence_scene = None
        if trajectory is self._current_control_curve and candidate_scene is self.scene:
            self._candidate_evidence_scene = result[3][0]
            update = self.geometry_constraints.add_evidence(self._current_full_controls, result[3][1])
            self._candidate_added_rows += update['added_rows']
        return result

    def _candidate_domain_validation(self, trajectory):
        """Qualify terminal motion on the same saved curve, including initializers."""
        report = super()._candidate_domain_validation(trajectory)
        start_u = self.terminal.report['requested_hold_start_u']
        # An explicitly supplied joint curve may carry a different clock.
        # The existing phase-alignment API maps the same source occurrence;
        # no fitting or new curve is used to qualify that supplied polynomial.
        if (trajectory.duration != self.formal_clock['duration']
                or not np.array_equal(trajectory.progress_u, self.progress_map.u)
                or not np.array_equal(trajectory.progress_s, self.progress_map.s)):
            mapping = _ProgressMap(trajectory.progress_u, trajectory.progress_s)
            start_u = float(_phase_aligned_parameters(self.progress_map, mapping,
                                                       np.asarray([start_u]))[0])
        start_s = start_u * trajectory.duration
        measured = _hold_polynomial_report(trajectory, start_s)
        trajectory.residuals.update(
            reference_terminal_position=max(0., measured['maximum_position_change_rad'] - self.problem.numerical.joint),
            reference_terminal_velocity=max(0., measured['maximum_velocity_rad_s'] - self.problem.numerical.velocity))
        report['reference_termination'] = {**self.terminal.report,
            'requested_hold_start_u': start_u, 'requested_hold_start_s': start_s,
            'hold_end_s': trajectory.duration, 'exported_polynomial': measured}
        return report

    def _candidate_objective(self, trajectory, candidate_scene):
        """Score the saved curve with the solver's fixed physical computation.

        Exact event anchors and every extra bound query are retained. A
        supplied curve's different progress clock uses the existing phase
        alignment API to identify the corresponding occurrences. Ordinary
        physical-time samples own motion integration in both callers.
        """
        # BEGIN use one objective implementation on the same complete query layout.
        if (trajectory.duration == self.formal_clock['duration']
                and np.array_equal(trajectory.progress_u, self.progress_map.u)
                and np.array_equal(trajectory.progress_s, self.progress_map.s)):
            # Reuse exact binary64 times. Reconstructing them with another
            # linspace/multiplication order can duplicate a point at one ULP
            # and therefore change the summed process or geometry objective.
            times = self.computation.times.numpy()[0]
            progress = self.computation.progress.numpy()[0]
            indices = self.computation.ordinary_indices
        else:
            mapping = _ProgressMap(trajectory.progress_u, trajectory.progress_s)
            # The same query identities follow the candidate's clock. Adding
            # another uniform grid would change the number of summed process
            # and geometry terms and therefore change candidate weighting.
            times = _phase_aligned_parameters(self.progress_map, mapping,
                                              self.computation.parameters) * trajectory.duration
            progress = self.computation.progress.numpy()[0]
            indices = self.computation.ordinary_indices
        with torch.no_grad():
            states, robot_state = self._candidate_robot_state(
                trajectory, candidate_scene, times, progress)
            tensor_progress, tensor_times = torch.as_tensor(progress)[None], torch.as_tensor(times)[None]
            tensors = trajectory_objective_components(problem=self.problem, scene=candidate_scene,
                states=states, robot_state=robot_state, progress=tensor_progress, times=tensor_times,
                ordinary_indices=indices, bounds=self.computation.bounds,
                motion_configuration=self.motion_configuration, native_dt=self.formal_clock['native_dt'],
                constraint_weight=self.budget.constraint_weight)
            components = {name: float(value) for name, value in tensors.items()}
        return dict(fixed_objective=sum(components.values()), objective_components=components,
                    objective_scope='common physical objective and bound event queries')
        # END qualification and comparison consume the same unscaled objective.

    # BEGIN consume existing scene evidence while the original checker owns sampling.
    def _candidate_geometry(self, trajectory, candidate_scene, *, initial_states, remaining_points, record_work):
        """Reuse bounded interval queries and collect their necessary witnesses.

        Only the current control curve can supply rows for this numerical
        problem.  A rollout clone retains its candidate-specific release and
        hand state; decorating that clone leaves the shared scene untouched.
        The existing evaluate output already contains every queried entity
        pair, including the actual progress and permission-progress sides.
        Each record supplies one witness state without another scene query.
        Pair records are retained by reference until the parent completes its
        accounting; only then are necessary rows added for the next solve.

        The parent owns subdivision, budgets, residuals and unresolved-interval
        reporting.  Finite/adaptive sampling is not a whole-curve certificate.
        """
        batches = []
        if (trajectory is self._current_control_curve
                and candidate_scene is self._candidate_evidence_scene):
            candidate_scene = candidate_scene.clone_for_rollout()
            evaluate = candidate_scene.evaluate

            def collect(*args, geometry_evidence_out, **kwargs):
                value = evaluate(*args, geometry_evidence_out=geometry_evidence_out, **kwargs)
                batches.append(geometry_evidence_out)
                return value

            candidate_scene.evaluate = collect
        report = super()._candidate_geometry(trajectory, candidate_scene,
            initial_states=initial_states, remaining_points=remaining_points, record_work=record_work)
        # BEGIN each completed query batch retains its original witness order.
        for batch in batches:
            update = self.geometry_constraints.add_evidence(self._current_full_controls, pair_batch=batch)
            self._candidate_added_rows += update["added_rows"]
        # END the common interval checker still owns all query decisions.
        return report
    # END original same-curve geometry evidence supplies both consumers once.

    def evaluate_candidate(self, trajectory, *, record_work, candidate_scene=None, objective=None):
        """Keep original task/domain qualification and capture its row discoveries."""
        self._candidate_added_rows = 0
        result = super().evaluate_candidate(trajectory, record_work=record_work,
            candidate_scene=candidate_scene, objective=objective)
        report = result[1]
        # BEGIN one immutable record owns this candidate's completed geometry.
        # The parent has finished every query and residual update. Reports,
        # selector snapshots and formal bindings can now share this value;
        # later candidates allocate their own records instead of changing it.
        if 'continuous_geometry' in report:
            geometry = completed_evidence(report['continuous_geometry'])
            report['continuous_geometry'] = geometry
            if self.last_validation is not None:
                self.last_validation['continuous_geometry'] = geometry
        # END mutable controls, ledger counters and report headers stay separate.
        # BEGIN interval scope stays diagnostic; original scene residuals own acceptance.
        # The parent's report retains confirmed intersections, unresolved
        # intervals and the declared unknown-space channels for this curve.
        # An unavailable separation proof adds no independent task condition.
        # END no additional geometry verdict replaces the original qualifier.
        report['preference_joint_problem'] = dict(
            pose_objectives=self.pose_objectives,
            reference_termination=self.terminal.report,
            geometry_constraint_rows=self.geometry_constraints.row_count,
            added_geometry_rows=self._candidate_added_rows)
        self.last_candidate_report = report
        self.candidate_history.append(dict(accepted=report['accepted'],
            residuals=deepcopy(report['residuals']),
            objective_components=deepcopy(report['objective_components']),
            added_geometry_rows=self._candidate_added_rows,
            controls=(self._current_full_controls.tolist()
                      if trajectory is self._current_control_curve else None)))
        return result

    def optimize(self, seed, *, iterations=None):
        """Solve reduced variables and exchange entity witnesses within the ledger.

        The original outer planner charges the first invocation.  Additional
        witness solves are charged here before they start, using its same
        request counter and check.  Yielding gives the original selector its
        complete curve qualification; only a rejected curve with new necessary
        rows can consume another solve slot.  Connection search remains the
        original planner's responsibility when no new direction is available.
        """
        self.last_optimizer_diagnostics, self.last_optimizer_runtime = [], []
        self.diagnostic_optimizer_candidate = self.diagnostic_optimizer_provenance = None
        original_strategies = tuple(self.seed_strategies)
        self.seed_strategies = []
        solve_index = 0
        for original_seed_index, row in enumerate(seed[0]):
            full = row.detach().cpu().numpy()
            exchange_index = 0
            while True:
                reduced = self.terminal.reduce(full)
                self.seed_strategies.append(original_strategies[original_seed_index]
                    if exchange_index == 0 else 'same_curve_registered_geometry_constraints')
                workspace = _planning_workspace_bytes(self.scene, self.robot_metadata,
                    self.budget, self.anchor_capacity, clock_nodes=len(self.progress_map.u))
                geometry_workspace = self.geometry_constraints.workspace_bytes()
                runtime = dict(seed_index=solve_index, original_seed_index=original_seed_index,
                    exchange_index=exchange_index,
                    reference_termination=self.terminal.report,
                    pose_objectives=self.pose_objectives,
                    geometry_constraint_rows=self.geometry_constraints.row_count,
                    geometry_constraint_workspace_bytes=geometry_workspace)
                self.last_optimizer_runtime.append(runtime)
                history_start = len(self.computation.trial_history)
                try:
                    if reduced.size:
                        result = solve_trajectory(self.computation, reduced,
                            iterations=self.budget.trajectory_iterations if iterations is None else iterations,
                            check_request=self.check_request, pending_geometry_bytes=(workspace.pending_cpu_bytes
                                + sum(geometry_workspace.values())),
                            report_out=runtime)
                        full = self.terminal.expand(result.controls)
                    else:
                        # With no free coefficient the declared problem has
                        # one numerical curve.  The same original qualifier
                        # evaluates it; an optimizer has no variable to update.
                        full = self.terminal.expand(reduced)
                        runtime.update(state='completed', solver_called=False, zero_free_variables=True,
                            reason='terminal_equations_determine_unique_curve', iterations=0)
                        result = ConstrainedSolution(reduced, runtime)
                except BaseException:
                    self._expand_snapshots(runtime)
                    keys = ('latest_complete_trial', 'last_callback_iterate', 'latest_trial')
                    selected = next(((key, runtime[key]) for key in keys
                                     if runtime.get(key) is not None), (None, None))
                    snapshot_key, snapshot = selected
                    if snapshot is not None and np.isfinite(snapshot['controls']).all():
                        try:
                            self.diagnostic_optimizer_candidate = self._controls_curve(np.asarray(snapshot['controls']))
                            self.diagnostic_optimizer_provenance = {
                                **deepcopy(snapshot), 'source': ('last_solver_iterate'
                                    if snapshot_key == 'last_callback_iterate' else 'diagnostic_optimizer_trial'),
                                'seed_index': solve_index, 'validation_status': 'not_completed'}
                        except Exception as export_error:
                            # Preserve the original solver interruption even
                            # if its finite trial overflows curve export.
                            runtime['diagnostic_curve_export_error'] = {
                                'error_type': type(export_error).__name__, 'reason': str(export_error)}
                    raise
                else:
                    self._expand_snapshots(runtime)
                finally:
                    runtime['objective_query_range'] = [history_start, len(self.computation.trial_history)]
                    self.solve_history.append(runtime)
                self.last_optimizer_diagnostics.append(result.report)
                used_rows = self.geometry_constraints.row_count
                yield ConstrainedSolution(np.asarray(full).copy(), result.report)
                solve_index += 1
                extra_rows = self.geometry_constraints.row_count - used_rows
                used = (self.request_work['trajectory_solves']
                        - self.request_work_baseline.get('trajectory_solves', 0))
                if (self.last_candidate_report['accepted'] or extra_rows == 0
                        or used >= self.budget.maximum_trajectory_solves):
                    break
                self.check_request('trajectory_solves')
                self.request_work['trajectory_solves'] += 1
                exchange_index += 1

    def _expand_snapshots(self, runtime):
        """Convert diagnostic coordinates once before original curve export."""
        for key in ('latest_trial', 'latest_complete_trial', 'last_callback_iterate'):
            snapshot = runtime.get(key)
            if snapshot is not None:
                reduced = np.asarray(snapshot['controls'])
                snapshot['reduced_controls'] = snapshot['controls']
                snapshot['controls'] = self.terminal.expand(reduced).tolist()
        runtime['solver_control_layout'] = 'terminal-reduced [control,joint]'
        runtime['diagnostic_control_layout'] = 'original full [control,joint]'

    def diagnostic_snapshot(self):
        """Return owned CPU evidence for this request, after its planning verdict.

        The request lifecycle attaches this record to PlanningResult. File
        formats and destinations belong to callers. Nonfinite numerical
        trials remain diagnostic values. Snapshotting does not
        evaluate a curve, query geometry or change candidate qualification.
        """
        return deepcopy(dict(joint_problem=dict(
            reference_termination=self.terminal.report,
            pose_objectives=self.pose_objectives,
            geometry=self.geometry_constraints.describe(), solves=self.solve_history,
            candidates=self.candidate_history),
            objective_trials=self.computation.trial_history))


class PreferenceMotionBackend(MotionBackend):
    """Own reusable native resources for the independent preference planner."""

    def __init__(self, robot, *, robot_validator, budget=None,
                 incoming_terminal_contract=None, device='cuda:0'):
        super().__init__(robot, robot_validator=robot_validator, budget=budget, device=device)
        self.incoming_terminal_contract = incoming_terminal_contract

    def bind_request(self, problem, scene, *, budget, continuation=None, goal_identities=None):
        """Retain original ownership/capacity rules and choose the local class."""
        if self._request_owner != get_ident():
            raise MotionPlanningError('backend_busy', 'Request binding requires ownership of the native buffers')
        if self._closed:
            raise MotionPlanningError('backend_closed', 'MotionBackend is closed')
        for name in ('knots', 'interpolation_steps', 'interpolation_dt', 'interpolation_buffer_size',
                     'ik_seeds', 'trajectory_seeds', 'max_task_anchors', 'ik_iterations', 'trajectory_iterations'):
            if getattr(budget, name) != getattr(self.budget, name):
                raise MotionPlanningError('capacity_exceeded', f'Resident backend cannot change {name}')
        if self._planner is None:
            self._planner = PreferencePlanner(problem, scene, self.robot, budget, self.device,
                robot_validator=self.robot_validator, continuation=continuation,
                goal_identities=goal_identities,
                incoming_terminal_contract=self.incoming_terminal_contract)
        else:
            self._planner.rebind(problem, scene, budget, continuation=continuation,
                                  goal_identities=goal_identities)
        return self._planner

    def plan(self, problem, scene, *, continuation=None, request_ledger=None, rebind=None,
             cancelled=None, deadline=None, joint_reference=None):
        """Call this planner's lifecycle while retaining resident native models.

        A complete problem carries explicitly bound pose roles. The original
        request ledger, measured continuation and cancellation semantics are
        forwarded without replenishing any computation allowance.
        """
        from .lifecycle import plan_trajectory
        return plan_trajectory(problem, scene, self.robot, robot_validator=self.robot_validator,
            budget=self.budget, device=self.device, continuation=continuation,
            request_ledger=request_ledger, rebind=rebind, backend=self,
            cancelled=cancelled, deadline=deadline, joint_reference=joint_reference)
