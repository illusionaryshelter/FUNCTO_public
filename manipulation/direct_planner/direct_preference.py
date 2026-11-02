"""Bind one sparse IPOPT joint problem and one same-point candidate result.

The request supplies reference preference scales and a relative weight. The
backend binds the original robot, scene, physical clock and terminal equations.
Package components share one objective and fixed derivative layout. One IPOPT
invocation supplies the final numerical states and geometry values to the
candidate result. The fixed physical query layout does not change.
"""
from copy import deepcopy
from threading import get_ident
from time import perf_counter

import numpy as np
from scipy.spatial.transform import Rotation

from manipulation.planning import (
    MotionPlanningError, _ConstrainedPlanner, _ProgressMap,
    _phase_aligned_parameters, _planning_workspace_bytes,
)
from manipulation.preference_planner.backend import (
    PreferenceMotionBackend, _hold_polynomial_report)
from manipulation.preference_planner.pose_roles import pose_objective
from manipulation.preference_planner.termination_clock import terminal_clock
from .direct_setup import DirectSetup
from .direct_computation import DirectComputation
from .direct_lifecycle import plan_trajectory
from .geometry_constraints import GeometryConstraints
from .nominal_hand_geometry import bind_position_controlled_finger_geometry
from .reference_objective import trajectory_objective_components
from .shared_solver import ConstrainedSolution, solve_trajectory
from .terminal_binding import bind_terminal_hold
from .reference_initialization import SymbolicKinematics, event_reference_samples
from .reference_projection import ReferenceProjection
from .candidate_result import evaluate_candidate as evaluate_numerical_candidate


class DirectPreferencePlanner(DirectSetup, _ConstrainedPlanner):
    """Optimize supplied curve coefficients under one physical objective."""

    # BEGIN explicit request data is available before native binding/rebinding.
    def __init__(self, *args, reference_objective, initialization,
                 incoming_terminal_contract=None, support_presolve=True, **kwargs):
        self.initialization_declaration = initialization
        self.reference_objective_declaration = reference_objective
        self.incoming_terminal_contract = incoming_terminal_contract
        # BEGIN caller selects solver-row elimination for this backend lifetime.
        self.support_presolve = support_presolve
        # END task production and complete candidate evaluation are shared.
        self.binding_continuation = kwargs.get('continuation')
        super().__init__(*args, **kwargs)
    # END the declaration has no implicit scale or weight.

    # BEGIN direct request ownership uses native setup without the old qualifier.
    def rebind(self, problem, scene, budget, *, continuation=None, goal_identities=None):
        self.binding_continuation = continuation
        return super().rebind(problem, scene, budget, continuation=continuation,
            goal_identities=goal_identities)

    def _controls_curve(self, controls):
        """Associate the original Hermite export with its exact coefficient array."""
        trajectory = super()._controls_curve(controls)
        self._current_control_curve = trajectory
        self._current_full_controls = (controls.detach().cpu().numpy().copy()
            if isinstance(controls, self.torch.Tensor) else np.asarray(controls, float).copy())
        return trajectory

    def _candidate_domain_validation(self, trajectory):
        """Read original continuous joint bounds and the bound terminal suffix."""
        report = super()._candidate_domain_validation(trajectory)
        start_u = self.terminal.report['requested_hold_start_u']
        start_s = start_u*trajectory.duration
        measured = _hold_polynomial_report(trajectory, start_s)
        trajectory.residuals.update(
            reference_terminal_position=max(0., measured['maximum_position_change_rad']-self.problem.numerical.joint),
            reference_terminal_velocity=max(0., measured['maximum_velocity_rad_s']-self.problem.numerical.velocity))
        report['reference_termination'] = dict(self.terminal.report,
            requested_hold_start_s=start_s, hold_end_s=trajectory.duration, exported_polynomial=measured)
        return report
    # END task and scene evaluation belong only to evaluate_candidate below.

    # BEGIN construct the native calculation once, then bind the local problem.
    def _bind_computation(self, extra_parameters=()):
        """Share calibrated FK and one original curve across all consumers.

        The fixed robot and installed end effector use their official model:
        their shapes and component relationships are reusable across tasks.
        Calibration and named joint feedback place that model. The original
        constrained setup owns the query grid; the local terminal, geometry
        and objective are each constructed once from that same calculation.
        """
        # BEGIN DirectSetup already owns the calibrated differentiable FK.
        _ConstrainedPlanner._bind_computation(self, extra_parameters)
        # END a second temporary PinKinematics construction is unnecessary.
        base = self.computation
        u, z, source = terminal_clock(problem=self.problem, budget=self.budget,
            progress_map=self.progress_map, reference_time_map=self.reference_time_map,
            duration=self.formal_clock['duration'], continuation=self.binding_continuation,
            incoming_terminal_contract=self.incoming_terminal_contract)
        self.terminal = bind_terminal_hold(base.domain._curve_map, u, z, source=source)
        self.pose_objectives = [dict(goal_id=['pose_events', index], **pose_objective(event))
                                for index, event in enumerate(self.problem.pose_events)]
        self.geometry_constraints = GeometryConstraints(base, progress_map=self.progress_map)
        reference = self.reference_objective_declaration.bind(self.bound_reference,
            base.times[0, base.ordinary_indices])
        self.computation = DirectComputation(base=base, terminal=self.terminal,
            geometry=self.geometry_constraints, reference_objective=reference,
            support_presolve=self.support_presolve)
        self.solve_history, self.candidate_history = [], []
        self._current_control_curve = self._current_full_controls = None
        self.last_candidate_report = None
        # BEGIN request-local passive timing belongs to this bound computation.
        self.candidate_evaluations = []
        # END a later request cannot inherit another curve's evaluation duration.
        self.route = 'ipopt_full_reference_and_registered_geometry'
    # END first binding and request rebinding share the same component ownership.

    # BEGIN exported curves and numerical controls consume the same cost definition.
    def _candidate_objective(self, trajectory, candidate_scene):
        """Align original query identities and bind the candidate's actual clock."""
        # BEGIN task relations reuse the sole immutable material inventory.
        started = perf_counter()
        relation_report = {}
        material_samples = {}
        if any(rule.required for rule in candidate_scene.contacts):
            self.geometry_constraints._prepare_materials()
            material_samples = self.geometry_constraints.scene._objective_material_samples
        # END an explicit curve without required contact does not prepare materials.
        indices = self.computation.ordinary_indices
        progress = self.computation.progress.numpy()[0]
        if (trajectory.duration == self.formal_clock['duration']
                and np.array_equal(trajectory.progress_u, self.progress_map.u)
                and np.array_equal(trajectory.progress_s, self.progress_map.s)):
            times = self.computation.times.numpy()[0]
            reference = self.computation.reference_objective
        else:
            mapping = _ProgressMap(trajectory.progress_u, trajectory.progress_s)
            times = _phase_aligned_parameters(self.progress_map, mapping,
                self.computation.parameters) * trajectory.duration
            reference = self.reference_objective_declaration.bind_clock(self.bound_reference,
                problem=self.problem, progress_u=mapping.u, progress_s=mapping.s,
                duration=trajectory.duration, physical_times=times[indices])
        with self.torch.no_grad():
            states, robot_state = self._candidate_robot_state(
                trajectory, candidate_scene, times, progress)
            costs = trajectory_objective_components(reference_objective=reference,
                problem=self.problem, scene=candidate_scene, states=states, robot_state=robot_state,
                progress=self.torch.as_tensor(progress)[None], times=self.torch.as_tensor(times)[None],
                ordinary_indices=indices, bounds=self.computation.bounds,
                motion_configuration=self.motion_configuration, native_dt=self.formal_clock['native_dt'],
                constraint_weight=self.budget.constraint_weight, material_samples=material_samples,
                scene_relation_report=relation_report)
            components = {name: float(value) for name, value in costs.items()}
        return dict(fixed_objective=sum(components.values()), objective_components=components,
            objective_scope='shared motion, task, pose, reference and declared scene-relation objective',
            scene_relations=relation_report,
            objective_evaluation_seconds=perf_counter()-started,
            reference_objective=reference.report(self.budget.constraint_weight))

    def diagnostic_snapshot(self):
        """Attach the actual reference weighting to the original owned records."""
        record = deepcopy(dict(joint_problem=dict(reference_termination=self.terminal.report,
            pose_objectives=self.pose_objectives, geometry=self.geometry_constraints.describe(),
            solves=self.solve_history, candidates=self.candidate_history),
            objective_trials=self.computation.trial_history))
        record['joint_problem']['reference_objective'] = self.computation.reference_objective.report(
            self.budget.constraint_weight)
        record['joint_problem']['candidate_evaluations'] = deepcopy(self.candidate_evaluations)
        return record
    # END scoring and diagnostic export perform no additional scene qualification.

    # BEGIN this explicit override owns the only solver-candidate exit.
    def evaluate_candidate(self, trajectory, *, record_work, objective):
        """Read the retained solver point through the local candidate contract.

        The inherited selector supplies this exact exported curve and its
        common objective. Task errors, continuous joint extrema and retained
        scene queries form the one result consumed by execution binding.
        """
        return evaluate_numerical_candidate(self, trajectory,
            record_work=record_work, objective=objective)
    # END the parent's material-witness and interval-query methods are unused.

    # BEGIN measured-entry initialization and one numerical solve own this request.
    def reference_seeds(self, joint_reference=None):
        """Rank measured-entry proposals with the same problem used by the NLP.

        Bounded terminal fits produce explicit alternatives. Every connection
        is fitted to the original curve and terminal map. Its actual signed
        conditions and complete objective determine the initial numerical
        branch; all values come from the same immutable bound problem later
        consumed by the optimizer. Ranking is not candidate authorization.
        """
        started = perf_counter()
        self.reference_ik_queries, self.joint_reference_report = [], None
        if joint_reference is not None:
            seeds, successes = super().reference_seeds(joint_reference=joint_reference)
            projected = self.terminal.expand(self.terminal.reduce(seeds.detach().numpy()))
            seeds = self.torch.as_tensor(projected, dtype=seeds.dtype)
            self.numerical_problem = self.computation.bind_solver(
                self.terminal.reduce(seeds[0, 0].detach().numpy()), check_request=self.check_request)
            self.initialization_candidate = self._controls_curve(projected[0, 0])
            # BEGIN record the supplied-reference initialization boundary too.
            self.numerical_problem.snapshot_producers('initialization_scored')
            # END this passive timestamp does not change reference selection.
            return seeds, successes
        if self.terminal.matrix.shape[1]:
            curve = self.physical_domain._curve_map
            declaration = self.initialization_declaration
            kinematics = SymbolicKinematics(self.robot_validator.checker.model,
                joint_names=self.joint_names,
                fixed_positions=dict(zip(self.problem.joint_names, self.problem.start_q)),
                frame=self.problem.tool_link, reference_from_base=self.problem.base_pose,
                frame_from_tool=self.problem.tool_transform)
            times = np.unique(np.r_[0., self.computation.times.numpy()[0]])
            position, quaternion = self.bound_reference.sample(self.torch.tensor(times))
            targets = np.broadcast_to(np.eye(4), (len(times), 4, 4)).copy()
            targets[:, :3, 3] = position.numpy()
            targets[:, :3, :3] = Rotation.from_quat(quaternion.numpy()[:, [1, 2, 3, 0]]).as_matrix()
            bounds = self.computation.bounds.numpy()
            event_times = self.progress_map.anchor_parameters(
                [event.progress for event in self.problem.pose_events]).ravel()*self.formal_clock['duration']
            proposals, provenance = event_reference_samples(kinematics, times=times,
                reference=targets, start_q=curve.start_q, start_dq=curve.start_dq,
                position_bounds=bounds[0], velocity_bounds=bounds[1], event_times=event_times,
                position_scale=declaration['pose_fit_position_scale_m'],
                rotation_scale=declaration['pose_fit_rotation_scale_rad'],
                random_seed=self.budget.random_seed, proposal_count=declaration['proposal_count'],
                maximum_evaluations=declaration['maximum_evaluations'], check_request=self.check_request)
            candidates, fits = [], []
            for samples in proposals:
                fitted, fit = self.physical_domain.fit_samples(times/self.formal_clock['duration'], samples)
                candidates.append(self.terminal.reduce(fitted))
                fits.append(fit)
            # BEGIN every terminal branch is fitted to the complete reference.
            # This stage produces only numerical proposals. The fixed original
            # task/geometry/physical problem below supplies their actual scores.
            connections = list(candidates)
            context = self.computation.shared_context(self.torch.as_tensor(candidates[0]))
            projection = ReferenceProjection(self.computation, context, check_request=self.check_request)
            candidates, reference_fits = [], []
            for connection in connections:
                fitted, fit = projection.fit(connection,
                    maximum_evaluations=declaration['maximum_evaluations'])
                candidates.append(fitted)
                reference_fits.append(fit)
            self.numerical_problem = self.computation.bind_solver(candidates[0], check_request=self.check_request)
            # END all proposals are scored in the same problem later used by IPOPT.
            scores = []
            for candidate in candidates:
                point = self.numerical_problem.evaluate(candidate)
                # BEGIN implied-row removal only changes the solver's layout.
                scores.append(point.initialization_score())
                # END the original (full violation, full objective) ranking remains.
            selected = min(range(len(candidates)), key=lambda i: (
                scores[i]['total_positive_violation'], scores[i]['objective']))
            controls = self.terminal.expand(candidates[selected])
            provenance.update(strategy='bounded_terminal_proposals_complete_reference_fit',
                reference_projection=projection.report, reference_fits=reference_fits,
                fitted_connection_controls=[self.terminal.expand(value).tolist() for value in connections],
                selected_proposal=selected, problem_scores=scores,
                selection='Original total positive signed violation, then original complete objective',
                fitted_proposal_controls=[self.terminal.expand(value).tolist() for value in candidates])
            position_row, position_offset = curve.query(times/self.formal_clock['duration'])
            self.reference_initialization = dict(status='same_problem_proposals_ranked', fits=fits,
                initialization=provenance, reference_termination=self.terminal.report,
                sample_times_s=times.tolist(), joint_samples=(position_row@controls+position_offset).tolist())
            strategy = provenance['strategy']
        else:
            controls, strategy = self.terminal.offset.copy(), 'terminal_unique_curve'
            self.reference_initialization = dict(status=strategy, reference_termination=self.terminal.report)
            self.numerical_problem = self.computation.bind_solver(
                self.terminal.reduce(controls), check_request=self.check_request)
        self.initial_seed_strategies = self.seed_strategies = [strategy]
        self.initialization_timing = dict(total_seconds=perf_counter()-started,
            scope='Registered-model terminal proposals, complete-reference fits and original fixed problem ranking')
        seeds = self.torch.as_tensor(controls, dtype=self.torch.float64)[None, None]
        self.initialization_candidate = self._controls_curve(controls)
        # BEGIN the shared counters already contain both initialization scores.
        self.numerical_problem.snapshot_producers('initialization_scored')
        # END main-solver timing uses a later snapshot of these same counters.
        return seeds, []
    # END the chosen curve and all subsequent trials share one bound problem.

    def optimize(self, seed, *, iterations=None):
        """Solve each explicitly supplied initial coefficient array once."""
        self.last_optimizer_diagnostics, self.last_optimizer_runtime = [], []
        self.diagnostic_optimizer_candidate = self.diagnostic_optimizer_provenance = None
        for index, row in enumerate(seed[0]):
            full = row.detach().cpu().numpy()
            reduced = self.terminal.reduce(full)
            workspace = _planning_workspace_bytes(self.scene, self.robot_metadata,
                self.budget, self.anchor_capacity, clock_nodes=len(self.progress_map.u))
            geometry_workspace = self.geometry_constraints.workspace_bytes()
            runtime = dict(seed_index=index, seed_provenance=self.seed_strategies[index],
                reference_termination=self.terminal.report, pose_objectives=self.pose_objectives,
                geometry_constraint_rows=self.geometry_constraints.row_count,
                geometry_constraint_workspace_bytes=geometry_workspace)
            self.last_optimizer_runtime.append(runtime)
            history_start = len(self.computation.trial_history)

            def charge_solver():
                self.check_request('trajectory_solves')
                self.request_work['trajectory_solves'] += 1

            # BEGIN count only this solve's increments in the shared producers.
            self.numerical_problem.snapshot_producers(f'main_{index}_started')
            # END initialization counts remain available at their own boundary.
            try:
                if reduced.size:
                    result = solve_trajectory(self.numerical_problem, reduced,
                        iterations=self.budget.trajectory_iterations if iterations is None else iterations,
                        check_request=self.check_request,
                        pending_geometry_bytes=workspace.pending_cpu_bytes+sum(geometry_workspace.values()),
                        report_out=runtime, geometry_workspace=geometry_workspace,
                        remaining_seconds=self.remaining_request_seconds, on_solver_start=charge_solver)
                    full = self.terminal.expand(result.controls)
                else:
                    bound = self.numerical_problem
                    self.computation.final_point = bound.evaluate(reduced)
                    runtime.update(state='completed', solver_called=False, zero_free_variables=True, iterations=0)
                    result = ConstrainedSolution(reduced, runtime)
            except BaseException:
                self._expand_snapshots(runtime)
                self._retain_diagnostic_trial(runtime, index)
                raise
            else:
                self._expand_snapshots(runtime)
            finally:
                # BEGIN preserve completed work even when the solver is interrupted.
                self.numerical_problem.snapshot_producers(f'main_{index}_finished')
                # END the outer planning-call clock includes all these stages.
                runtime['objective_query_range'] = [history_start, len(self.computation.trial_history)]
                self.solve_history.append(runtime)
            self.last_optimizer_diagnostics.append(result.report)
            self.last_candidate_report = None
            if np.isfinite(full).all():
                self.diagnostic_optimizer_candidate = self._controls_curve(full)
                self.diagnostic_optimizer_provenance = dict(source='solver_return_candidate', seed_index=index,
                    controls=full.tolist(), validation_status='not_completed', solver_state=result.report['state'])
            yield ConstrainedSolution(np.asarray(full).copy(), result.report)
            self.diagnostic_optimizer_candidate = self.diagnostic_optimizer_provenance = None
    # END no material revision, second qualification chain or implicit solve restart.

    # BEGIN numerical diagnostics use the original full curve coordinate layout.
    def _expand_snapshots(self, runtime):
        for key in ('latest_trial', 'latest_complete_trial', 'last_callback_iterate'):
            snapshot = runtime.get(key)
            if snapshot is not None:
                snapshot['reduced_controls'] = snapshot['controls']
                snapshot['controls'] = self.terminal.expand(np.asarray(snapshot['controls'])).tolist()
        runtime['solver_control_layout'] = 'terminal-reduced [control,joint]'
        runtime['diagnostic_control_layout'] = 'original full [control,joint]'
    # END numerical evidence is converted once at the result boundary.

    def _retain_diagnostic_trial(self, runtime, solve_index):
        """Keep an already evaluated curve when further work is interrupted."""
        keys = ('latest_complete_trial', 'last_callback_iterate', 'latest_trial')
        snapshot_key, snapshot = next(((key, runtime[key]) for key in keys
            if runtime.get(key) is not None), (None, None))
        if snapshot is not None and np.isfinite(snapshot['controls']).all():
            try:
                self.diagnostic_optimizer_candidate = self._controls_curve(np.asarray(snapshot['controls']))
                self.diagnostic_optimizer_provenance = {
                    **deepcopy(snapshot), 'source': ('last_solver_iterate'
                        if snapshot_key == 'last_callback_iterate' else 'diagnostic_optimizer_trial'),
                    'seed_index': solve_index, 'validation_status': 'not_completed'}
            except Exception as export_error:
                # Polynomial export may overflow even for finite controls;
                # keep its evidence beside the original request interruption.
                runtime['diagnostic_curve_export_error'] = {
                    'error_type': type(export_error).__name__, 'reason': str(export_error)}
    # END original selector and binding retain final candidate ownership.


class DirectPreferenceBackend(PreferenceMotionBackend):
    """Own native resources and one explicit reference preference declaration."""

    # BEGIN reuse the maintained backend ownership and capacity contract.
    def __init__(self, *args, reference_objective, initialization, support_presolve=True, **kwargs):
        self.initialization_declaration = initialization
        self.reference_objective = reference_objective
        # BEGIN one explicit option selects the retained NLP row set.
        self.support_presolve = support_presolve
        # END rebinding retains this backend configuration.
        super().__init__(*args, **kwargs)

    def bind_request(self, problem, scene, *, budget, continuation=None, goal_identities=None):
        """Bind the local computation while retaining resident resource checks."""
        if self._request_owner != get_ident():
            raise MotionPlanningError('backend_busy', 'Request binding requires ownership of the native buffers')
        if self._closed:
            raise MotionPlanningError('backend_closed', 'MotionBackend is closed')
        for name in ('knots', 'interpolation_steps', 'interpolation_dt', 'interpolation_buffer_size',
                     'ik_seeds', 'trajectory_seeds', 'max_task_anchors', 'ik_iterations', 'trajectory_iterations'):
            if getattr(budget, name) != getattr(self.budget, name):
                raise MotionPlanningError('capacity_exceeded', f'Resident backend cannot change {name}')
        # BEGIN declare one position-controlled geometry model on this scene.
        # The resource setup binds its first hand clock inside the planner
        # constructor. Installing the factory beforehand makes initialization,
        # the joint problem and cloned candidate scenes share that model.
        # Commands, encoder feedback and relation conditions remain inputs.
        hand = scene.gripper
        if hand is not None and hand.controller == 'timed_reference':
            def nominal_geometry(duration, progress_map, command_reference):
                if command_reference.duration != duration:
                    raise ValueError('Nominal hand and command require the same bound physical clock')
                return bind_position_controlled_finger_geometry(command_reference,
                    problem.conditions, progress_map.event_clock,
                    progress_interval=(progress_map.s[0], progress_map.s[-1]),
                    source='Direct backend declared bounded position-control hand model; '
                           'original attachment, command and contact occurrence inputs')

            hand.nominal_finger_trajectory_factory = nominal_geometry
        # END every request rebind installs a factory for its current problem.
        if self._planner is None:
            self._planner = DirectPreferencePlanner(problem, scene, self.robot, budget,
                self.device, robot_validator=self.robot_validator,
                continuation=continuation, goal_identities=goal_identities,
                incoming_terminal_contract=self.incoming_terminal_contract,
                reference_objective=self.reference_objective, initialization=self.initialization_declaration,
                support_presolve=self.support_presolve)
        else:
            self._planner.rebind(problem, scene, budget, continuation=continuation,
                goal_identities=goal_identities)
        return self._planner

    def plan(self, problem, scene, *, continuation=None, request_ledger=None, rebind=None,
             cancelled=None, deadline=None, joint_reference=None):
        """Use the local initialization/optimization lifecycle with original inputs."""
        return plan_trajectory(problem, scene, backend=self, budget=self.budget,
            continuation=continuation, request_ledger=request_ledger, rebind=rebind,
            cancelled=cancelled, deadline=deadline, joint_reference=joint_reference)
    # END the prepared runner retains ownership of backend closure.
