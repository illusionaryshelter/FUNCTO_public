"""Preference-planner request lifecycle returning the first fully qualified candidate.

The static request implementation retains FUNCTO's original ledger, binding,
search and resource checks. Reference IK guides initialization; necessary
repairs use the common objective. Original qualification and execution binding
finish the request independently of the declared pose objective roles.
"""

from copy import deepcopy
from dataclasses import asdict, replace
import time

import numpy as np

from manipulation.execution_binding import bind_candidate, curve_snapshot, plain
from manipulation.planning import (
    ContinuationBindingError, JointTrajectory, MotionPlanningError,
    PlanningBudget, PlanningResult, PlanningWorkExhausted, _bind_motion_work_ledger,
    _check_work_budget, _fixed_spline_grid, _obligation_contract, _prepare_continuation,
    _reference_clock_issues, _request_work_ledger, _restore_original_direction_origins,
    _unknown_policy_failure, _validate_bound_event_times, _verified_dependency_outcomes,
)
from manipulation.planning_problem import TimeBindingError, event_occurrence_key
from .backend import PreferenceMotionBackend


def plan_trajectory(problem, scene, robot, *, robot_validator, budget=None, device="cuda:0", continuation=None,
                    request_ledger=None, rebind=None, backend=None, cancelled=None, deadline=None, joint_reference=None):
    """Return the first fully checked candidate in the existing bounded search.

    ``request_ledger`` continues failed planning attempts. ``continuation``
    carries actual execution facts. An explicit ``rebind`` rebuilds the soft
    schedule of remaining work or creates selected retry occurrences; ordinary
    continuation preserves the original frozen clock. Every request admitted
    including initialization failure carries the cumulative ledger. No GPU stage is
    claimed to be preemptible. Only completed qualification and execution
    binding publish a successful result. The first such result finishes this
    request; reference IK and the backend's objective retain their respective
    initialization and necessary-repair roles.
    """
    started = time.monotonic()
    prior = planner = ledger = trajectory = None
    # BEGIN reference provenance follows the exact selected curve identity.
    reference_trajectory = None
    # END rejected candidates retain their own provenance during necessary repair.
    initial_seconds = 0.
    motion_ready, motion_baseline = False, {}
    owned_backend = backend is None
    backend_claimed = False
    diagnostics = {"attempts": []}
    # BEGIN passive request milestones distinguish lazy native initialization from solving.
    diagnostics["wallclock"] = {"scope": "One planning request; Unix timestamps join processes, elapsed seconds use the request monotonic clock", "events": {}}
    def mark_wallclock(name):
        diagnostics["wallclock"]["events"][name] = {
            "unix_time_ns": time.time_ns(), "elapsed_seconds": time.monotonic()-started}
    mark_wallclock("planning_request_started")
    # END timing never changes the cumulative work ledger or triggers GPU synchronization.
    outcome = PlanningResult(False, None, diagnostics)
    stage = "request_binding"
    try:
        # BEGIN every request owns its mutable online scene and cumulative ledger.
        scene = scene.clone_for_rollout()
        # END clones share the immutable model policy, not candidate hand clocks or query evidence.
        previous = request_ledger
        if previous is None and isinstance(continuation, dict) and isinstance(continuation.get("original_binding"), dict):
            previous = continuation.get("request_ledger", continuation["original_binding"].get("request_ledger"))
        if previous is not None:
            original_budget = PlanningBudget(**previous["planning_budget"])
        else:
            original_budget = PlanningBudget() if budget is None else budget
        # BEGIN execution facts enter a trusted work ledger only after validation.
        # An executor may already have appended its new observations to the
        # supplied ledger. Keep their computation history, but restore the
        # last frozen binding's facts until this continuation passes. A failed
        # request must never return unverified completion IDs as trusted facts.
        execution_fields = ("history", "event_records", "timing_events", "timing_checks",
            "completed_goal_ids", "goal_evidence", "completed_direction_ids", "direction_evidence",
            "effects", "effect_history")
        ledger_input = previous
        if isinstance(continuation, dict) and isinstance(continuation.get("original_binding"), dict):
            original_ledger = continuation["original_binding"]["request_ledger"]
            ledger_input = deepcopy(previous)
            for name in execution_fields:
                ledger_input[name] = deepcopy(original_ledger[name])
            diagnostics["continuation_facts_pending_validation"] = {
                name: deepcopy(continuation[name]) for name in (*execution_fields, "actual_feedback")
                if name in continuation}
        ledger = _request_work_ledger(problem, original_budget, ledger_input)
        if continuation is not None:
            ledger["requires_continuation"] = True
        # END pending facts remain diagnostics, not completed obligations.
        consumed = ledger["consumed_planning"]
        initial_seconds = consumed["planning_seconds"]
        entrance_consumed = deepcopy(consumed)
        request_budget = asdict(original_budget)
        diagnostics.update(planning_budget=request_budget, consumed_planning=consumed)
        def check(next_stage=None):
            if cancelled is not None and cancelled():
                raise MotionPlanningError("cancelled", "Motion request was cancelled")
            if deadline is not None and time.monotonic() >= deadline:
                raise MotionPlanningError("deadline_exceeded", "Motion request deadline elapsed")
            consumed["planning_seconds"] = initial_seconds+time.monotonic()-started
            if motion_ready:
                relative = {name: value-motion_baseline.get(name, 0) for name, value in consumed.items()}
                _check_work_budget(original_budget, relative, started=started,
                    initial_seconds=initial_seconds-motion_baseline.get("planning_seconds", 0.), stage=next_stage)
        # BEGIN the request ledger owns started attempts and completed queries.
        def record_validation_work(*, passes=0, points=0):
            if passes:
                check("validation_passes")
                consumed["validation_passes"] += passes
            consumed["validation_queries"] += points
            check()
        # END query accounting precedes the post-query deadline/budget check.
        check("binding_attempts")
        consumed["binding_attempts"] += 1
        # BEGIN explicit allocation changes preserve the existing request ledger.
        # Validate and retain the old ledger before considering the new limits,
        # so an invalid change still returns this attempt's cumulative work.
        if previous is not None and budget is not None and asdict(budget) != asdict(original_budget):
            ledger = _request_work_ledger(problem, budget, ledger)
            original_budget = budget
            consumed = ledger["consumed_planning"]
            request_budget = asdict(original_budget)
            diagnostics.update(planning_budget=request_budget, consumed_planning=consumed)
        # END allocation admission never buys a new solve, second or obligation.
        # BEGIN each admitted binding owns mutable problem and device state.
        # Initial clock construction can bind a hand before native allocation
        # fails. A private problem is therefore required for both first and
        # subsequent attempts, just as for remaining-obligation filtering.
        problem = deepcopy(problem)
        if previous is not None:
            _restore_original_direction_origins(problem, ledger["obligation_contract"])
        # END each admitted binding owns mutable problem and device state.
        if _obligation_contract(problem) != ledger["obligation_contract"]:
            raise ContinuationBindingError("A same request cannot remove, reorder or change original goal/process obligations")
        if continuation is not None:
            if not isinstance(continuation, dict) or not isinstance(continuation.get("original_binding"), dict):
                raise ContinuationBindingError("Continuation requires its original formal execution binding")
            if previous is None:
                raise ContinuationBindingError("Continuation lacks the original cumulative request ledger")
            # Binding owns its filtered obligations and hand occurrences. The
            # caller retains the full source problem after any failed attempt;
            # the scene clone shares only immutable geometry and the existing
            # cumulative robot oracle, never a reset query allowance.
            current = deepcopy(continuation)
            current["consumed_planning"] = deepcopy(consumed)
            budget, prior = _prepare_continuation(problem, scene, original_budget, current, rebind=rebind)
            # BEGIN commit actual facts only after their identities, clock and proofs pass.
            ledger = _request_work_ledger(problem, original_budget, ledger, continuation)
            consumed = ledger["consumed_planning"]
            diagnostics["consumed_planning"] = consumed
            diagnostics.pop("continuation_facts_pending_validation", None)
            # END a rejected continuation still retains its real planning work counters.
            prior["history"] = deepcopy(ledger["history"])
            prior["timing_events"] = deepcopy(ledger["timing_events"])
        else:
            if ledger["requires_continuation"] or any(ledger[name] for name in execution_fields if name != "timing_events"):
                raise ContinuationBindingError("An executed request needs its actual continuation facts before another binding")
            if plain(problem.events) != ledger["original_events"]:
                raise ContinuationBindingError("A failed-request retry changed the original event/source requirements")
            if rebind is not None and (not rebind.get("source") or rebind.get("kind") != "remaining"):
                raise ContinuationBindingError("Creating retry occurrences requires an actual execution continuation")
            budget = original_budget
        stage = "initialization"
        satisfied_dependencies = _verified_dependency_outcomes(continuation)
        try:
            problem, block_record = problem.bound_motion_block(
                goal_identities=None if prior is None else prior["goal_identities"],
                satisfied_dependencies=satisfied_dependencies)
        except ValueError as error:
            if "requires fresh dependency evidence" in str(error):
                raise MotionPlanningError("needs_rebind", str(error)) from error
            raise
        motion_baseline = _bind_motion_work_ledger(ledger, block_record, continuation, entrance_consumed)
        motion_ready = True
        check()
        budget = replace(budget, search_seconds=max(0., original_budget.search_seconds-
            (consumed["search_seconds"]-motion_baseline.get("search_seconds", 0.))))
        diagnostics["motion_block"] = block_record
        # BEGIN reject an unrepresented source occurrence before native allocation.
        if isinstance(joint_reference, JointTrajectory) and prior is None:
            stage = "reference_clock_binding"
            issues = _reference_clock_issues(problem, joint_reference,
                None if scene.gripper is None else scene.gripper.control_timestep_s)
            if issues:
                diagnostics.update(failure="reference_clock_outside_curve", failed_stage=stage,
                    reference_clock_issues=issues, diagnostic_trajectory=joint_reference,
                    diagnostic_candidate_scope="Supplied curve has no complete event clock and no execution authorization",
                    original_reference_candidate={"accepted": False, "clock_issues": issues,
                        "scope": "Required occurrence lies outside supplied curve; online geometry was not evaluated"},
                    joint_reference={"source_curve_preserved": True, "candidate_kind": "original_joint_curve",
                        "source_duration_s": joint_reference.duration, "bound_duration_s": None})
                return outcome
        # END this refusal does not invent a hold, source fit, or collision verdict.
        stage = "initialization"
        if backend is None:
            backend = PreferenceMotionBackend(robot, robot_validator=robot_validator,
                budget=original_budget, device=device, incoming_terminal_contract=(None if continuation is None
                    else continuation["original_binding"]["validation"]["export"]["reference_termination"]))
        check("initializations")
        consumed["initializations"] += 1
        backend._claim()
        backend_claimed = True
        if scene.gripper is not None:
            occurrences = {event_occurrence_key(event) for event in problem.events}
            scene.gripper.commands = tuple(row for row in scene.gripper.commands
                                           if event_occurrence_key(row) in occurrences)
        # BEGIN bind_request creates native solvers on first use or rebinds resident ones.
        mark_wallclock("backend_binding_started")
        planner = backend.bind_request(problem, scene, budget=budget, continuation=prior,
                                       goal_identities=block_record["goal_identities"])
        mark_wallclock("backend_binding_completed")
        # END failed initialization retains only the start milestone.
        planner.check_request = check
        # BEGIN native reference and event IK share actual submitted work.
        planner.request_work = consumed
        # BEGIN absolute request counters retain the active motion-block baseline.
        planner.request_work_baseline = motion_baseline
        # END native query admission uses the same verified boundary as this ledger.
        def remaining_request_seconds():
            check()
            remaining = original_budget.planning_seconds-(consumed["planning_seconds"]
                - motion_baseline.get("planning_seconds", 0.))
            return max(0., remaining if deadline is None else min(remaining, deadline-time.monotonic()))
        planner.remaining_request_seconds = remaining_request_seconds
        # END this is the same dictionary retained by request continuation.
        planner.satisfied_dependencies = tuple(sorted(satisfied_dependencies))
        check()
        diagnostics.update(backend="scipy_slsqp_with_native_ik", batch_size=1, ik_seeds=budget.ik_seeds,
            trajectory_seeds=budget.trajectory_seeds, used_ompl=False,
            route=planner.route, recovery=None,
            timing_policy="explicit_rebind_then_frozen" if rebind is not None else "source_proposal_then_frozen_binding",
            native_trajectory_dt=planner.budget.maximum_trajectory_dt,
            scene=scene.describe(), scene_workspace_bytes=planner.scene_workspace_bytes,
            scene_cpu_workspace_bytes=planner.scene_cpu_workspace_bytes,
            scene_gpu_workspace_bytes=planner.scene_gpu_workspace_bytes,
            anchor_fk_workspace_bytes=planner.anchor_fk_workspace_bytes, exact_anchor_capacity=planner.anchor_capacity,
            validation_storage=deepcopy(getattr(planner, "validation_storage", {})))
        # BEGIN describe the actual native state computation, including effort.
        diagnostics["joint_state_computation"] = {
            "kinematic_derivative_orders": [0, 1, 2, 3],
            "effort": "not_computed"}
        # END zero effort buffers in a kinematic rollout are not torque evidence.
        limits = planner._verification_native.get_cost("cspace").config.joint_limits
        diagnostics["verification_joint_limits"] = {"joint_names": list(limits.joint_names),
            **{name: getattr(limits, name).detach().cpu().numpy().tolist()
               for name in ("position", "velocity", "acceleration", "jerk")}}
        stage = "initial_state_verification"
        consumed["validation_queries"] += 1
        diagnostics["initial_state"] = planner.initial_state_diagnostics()
        check()
        if not diagnostics["initial_state"]["passed"]:
            # BEGIN distinguish a policy decision from a measured physical violation.
            diagnostics["failure"] = (_unknown_policy_failure(diagnostics["initial_state"],
                scene.unknown_policy) or "initial_state_infeasible")
            # END no optimization can change the measured entrance.
            return outcome
        # BEGIN seed, block and final candidates share one complete binding boundary.
        def qualify_candidate(trajectory, validation, attempt, *, binding_scene=None):
            if not attempt["accepted"] or validation is None:
                raise ValueError("Only a completely validated candidate may enter execution binding")
            bound_scene = scene if binding_scene is None else binding_scene
            # BEGIN the binding owns the validation tree at its conversion boundary.
            # bind_candidate reads the candidate's arrays synchronously, then
            # plain(validation) creates independent containers and array values.
            binding = bind_candidate(trajectory, problem=problem, scene=scene if binding_scene is None else binding_scene,
                validation=validation, native_report=attempt)
            # END the source validation remains owned by this active candidate.
            for event in problem.events:
                if event.timing and not any(event_occurrence_key(row) == event_occurrence_key(event)
                                            for row in ledger["timing_events"]):
                    ledger["timing_events"].append(plain(event))
            retry_goal_ids = set() if prior is None else set(prior["retry_goal_ids"])
            retry_direction_ids = set() if prior is None else set(prior["retry_direction_ids"])
            binding.update(motion_block=deepcopy(block_record), planning_budget=deepcopy(request_budget),
                request_events=deepcopy(ledger["original_events"]),
                consumed_events=[] if prior is None else prior["consumed_events"],
                goal_identities=deepcopy(block_record["goal_identities"]),
                # BEGIN no physical geometry score enters an execution binding.
                completed_goal_ids=[], goal_evidence=[],
                # END independent effect and direction histories stay separate.
                direction_identities=([[(index, column) for column in range(len(mechanism.directions))]
                    for index, mechanism in enumerate(getattr(problem, "mechanisms", ()))]
                    if prior is None else deepcopy(prior["direction_identities"])),
                completed_direction_ids=[deepcopy(value) for value in
                    (ledger["completed_direction_ids"] if prior is None else prior["completed_direction_ids"])
                    if tuple(value) not in retry_direction_ids],
                retry_direction_ids=sorted(retry_direction_ids),
                direction_evidence=deepcopy(ledger["direction_evidence"]),
                history=deepcopy(ledger["history"]), effect_history=deepcopy(ledger["effect_history"]),
                timing_events=deepcopy(ledger["timing_events"]), timing_checks=deepcopy(ledger["timing_checks"]),
                request_time_origin_s=0. if prior is None else prior["request_time_origin_s"],
                request_time_origin_step=0 if prior is None else prior["request_time_origin_step"])
            if prior is not None:
                binding.update({name: deepcopy(prior[name]) for name in (
                    "physics_budget", "consumed_physics_steps", "remaining_physics_steps",
                    "device_policy", "active_finger_interval")})
            # BEGIN the first complete binding publishes one owned result.
            # Reference initialization or necessary optimization has produced
            # this candidate; its objective remains part of the owned report.
            # Complete the existing request check before publishing success;
            # caller return and the original selector stop all further proposals.
            binding["verification_joint_limits"] = deepcopy(diagnostics["verification_joint_limits"])
            owned_curve, owned_report = deepcopy(trajectory), deepcopy(attempt)
            check()
            outcome.success, outcome.trajectory, outcome.execution_binding = True, owned_curve, binding
            diagnostics.update(owned_report, returned_candidate=deepcopy(owned_report),
                scene=bound_scene.describe(), qualified_candidate_elapsed_seconds=time.monotonic()-started,
                executable_until=owned_curve.executable_until, next_verification=owned_curve.next_verification)
            mark_wallclock("qualified_candidate_ready")
            # END result ownership survives backend reuse and final ledger publication.

        def qualify_native(result, attempt):
            """Export immediately while the selected native validation still belongs to this curve."""
            if not attempt["accepted"]:
                return
            check()
            curve, _ = planner.export(result, attempt)
            qualify_candidate(curve, planner.last_validation, attempt)

        # BEGIN preserve a supplied polynomial before creating native proposals.
        if joint_reference is not None:
            if prior is not None:
                raise MotionPlanningError("unsupported_reference", "A supplied complete curve cannot replace an actual continuation's frozen suffix")
            stage = "reference_curve_verification"
            candidate, candidate_scene, candidate_map = planner.bind_reference_candidate(joint_reference)
            diagnostics["event_time_checks"] = _validate_bound_event_times(problem, candidate_map, candidate.duration,
                0., ledger["history"], None, ledger["timing_events"])
            candidate, reference_attempt, _ = planner.evaluate_candidate(candidate,
                candidate_scene=candidate_scene, record_work=record_validation_work)
            diagnostics["original_reference_candidate"] = deepcopy(reference_attempt)
            diagnostics["joint_reference"] = {"source_curve_preserved": True,
                "candidate_kind": "original_joint_curve", "source_duration_s": joint_reference.duration,
                "bound_duration_s": candidate.duration}
            diagnostics["reference_clock"] = {"duration": candidate.duration,
                "sample_times_s": candidate.times.tolist(),
                "scope": "Supplied polynomial retains its physical clock and bound progress map"}
            check()
            if reference_attempt["accepted"]:
                qualify_candidate(candidate, planner.last_validation, reference_attempt, binding_scene=candidate_scene)
                diagnostics.update(reference_candidate=deepcopy(reference_attempt), route="candidate_metrics")
                # BEGIN a fully bound supplied curve completes the request for every role.
                return outcome
                # END repairs are entered only for rejected supplied curves.
            # BEGIN explicit observation requests stop at the completed reference verdict.
            policy_failure = _unknown_policy_failure(reference_attempt, candidate_scene.unknown_policy)
            if policy_failure == "needs_observation":
                diagnostics.update(failure=policy_failure, reference_candidate=deepcopy(reference_attempt),
                    diagnostic_trajectory=candidate,
                    failure_scope="Supplied curve requires observation under the selected policy; other routes were not evaluated")
                return outcome
            # END forbidden coverage may still use the existing bounded repair path.
            missing = {name: value for name, value in reference_attempt.get("scene_components", {}).items()
                if name.startswith("scene_update/") and (not np.isfinite(value["maximum"]) or value["maximum"] > 0)}
            if missing:
                diagnostics.update(failure="reference_outside_observation_scope", reference_scope_gaps=missing,
                    diagnostic_trajectory=candidate,
                    diagnostic_candidate_scope="Supplied joint curve; no execution authorization",
                    failure_scope="Supplied curve lacks current model support; other routes were not evaluated")
                return outcome
        # END rejected supplied curves retain the same bounded repair route.
        grid = _fixed_spline_grid(intervals=planner.budget.knots+4,
            native_dt=planner.budget.maximum_trajectory_dt, interpolation_steps=planner.budget.interpolation_steps,
            interpolation_dt=planner.budget.interpolation_dt)
        # BEGIN report the admitted clock without claiming unchanged source timestamps.
        source_seconds = problem.source_time_map["source_times"]
        source_duration = float(source_seconds[-1]-source_seconds[0])
        diagnostics["reference_clock"] = {**grid, "source_duration_s": source_duration,
            "duration_change_s": grid["duration"]-source_duration,
            "scope": "Source duration is a soft proposal; native, dense, event and finger consumers share the bound clock"}
        # END hard event deadlines are checked separately on this exact binding.
        diagnostics["event_time_checks"] = _validate_bound_event_times(problem, planner.progress_map, grid["duration"],
            0. if prior is None else prior["request_time_origin_s"], ledger["history"],
            None if prior is None else prior["goal_identities"], ledger["timing_events"])

        check("ik_batches")
        # BEGIN task initialization and main optimization share the first solve slot.
        check("trajectory_solves")
        consumed["ik_batches"] += 1
        stage = "reference_initialization"
        seeds, ik_success = planner.reference_seeds(joint_reference=joint_reference)
        check()
        diagnostics.update(reference_ik_success_counts=ik_success,
            ik_exit_early=bool(planner.reference_ik.config.exit_early),
            initial_seed_strategies=planner.initial_seed_strategies,
            reference_ik_queries=deepcopy(planner.reference_ik_queries))
        diagnostics["functional_ik_candidates"] = deepcopy(getattr(planner, "goal_ik_candidates", {}))
        diagnostics["joint_reference"] = deepcopy(planner.joint_reference_report)
        # BEGIN every completed candidate is frozen before another solve or query.
        def on_candidate(result, candidate_report):
            nonlocal trajectory, attempt
            trajectory, attempt = planner.delivery_candidate, candidate_report
            if stage in {"main_solve", "recovery_solve"}:
                diagnostics["attempts"].append(deepcopy(candidate_report))
            qualify_native(result, candidate_report)

        def select_candidate(result):
            return planner.select_candidate(result, record_work=record_validation_work,
                                            on_candidate=on_candidate)

        reference_result = planner.evaluate_reference(seeds)
        native_result, trajectory, attempt = select_candidate(reference_result)
        # BEGIN checked seed evidence remains bound to its own saved polynomial.
        reference_trajectory = trajectory
        diagnostics['reference_candidate'] = {
            **deepcopy(attempt), "source": "validated_reference_seed", "validation_status": "completed",
            "curve": curve_snapshot(trajectory)}
        # END later numerical trials have separate geometry and validation status.
        # BEGIN complete reference qualification finishes the request for every role.
        if outcome.success:
            return outcome
        # END rejected references retain the original optimization and search route.
        observation_required = _unknown_policy_failure(attempt, scene.unknown_policy) == 'needs_observation'
        recovery = None
        search_used = False

        def geometry_failed(candidate_report):
            """Search only from explicit collision or forbidden-coverage residuals."""
            return any(np.isfinite(value) and value > 0
                for name in ("collision_residuals", "coverage_residuals")
                for value in candidate_report[name].values())

        def search_initializers(candidate):
            """Spend the single search opportunity and preserve the original local seed."""
            nonlocal recovery, search_used, stage
            search_used, recovery, stage = True, "ompl", "bounded_seed_search"
            planner.last_search_report = {}
            diagnostics["used_ompl"] = True
            search_started = time.monotonic()
            try:
                recovered, report = planner.ompl_seed(seeds, candidate)
                planner.last_search_report = report
            finally:
                # Every started state query and all elapsed search work is
                # committed exactly once, including cancellation and errors.
                report = planner.last_search_report
                elapsed = time.monotonic()-search_started
                consumed["search_seconds"] += elapsed
                consumed["search_queries"] += report.get("state_queries", 0)
                diagnostics["ompl"] = {**deepcopy(report), "elapsed_seconds": elapsed}
            check()
            if recovered is None:
                recovery = "ompl_failed"
                return seeds
            # Search is the priority proposal; K bounds actual distinct sources.
            # The local curve remains the second source only when K permits it.
            count = budget.trajectory_seeds
            planner.seed_strategies = (["ompl"]+list(planner.initial_seed_strategies))[:count]
            return planner.torch.cat((recovered.to(seeds), seeds), dim=1)[:, :count]

        # Initial geometry evidence uses search before the main optimization.
        # A failed connection leaves the same local proposal and solve allowance.
        if not observation_required:
            if geometry_failed(attempt) and budget.search_seconds > 0:
                seeds = search_initializers(trajectory)
            check()
            check("trajectory_solves")
            consumed["trajectory_solves"] += 1
            stage = "main_solve"
            native_result, trajectory, attempt = select_candidate(planner.optimize(
                seeds, iterations=planner.main_trajectory_iterations))
            # BEGIN the original selector's first bound repair finishes this request.
            if outcome.success:
                diagnostics.update(route=planner.route, recovery=recovery,
                    used_ompl=recovery in {"ompl", "ompl_failed"})
                return outcome
            # END no completed repair launches additional seeds or geometric search.
            check()
            observation_required = _unknown_policy_failure(attempt, scene.unknown_policy) == "needs_observation"
            if (not attempt["accepted"] and not observation_required and not search_used
                    and consumed["trajectory_solves"]-motion_baseline.get("trajectory_solves", 0) < original_budget.maximum_trajectory_solves):
                # Newly discovered geometry failure can use the still-unspent
                # search and optimization batch. Failed search adds no batch.
                if geometry_failed(attempt) and budget.search_seconds > 0:
                    seeds = search_initializers(trajectory)
                    if recovery == "ompl":
                        check("trajectory_solves")
                        consumed["trajectory_solves"] += 1
                        stage = "recovery_solve"
                        native_result, trajectory, attempt = select_candidate(planner.optimize(seeds))
                        # BEGIN a bound geometric recovery also completes immediately.
                        if outcome.success:
                            diagnostics.update(route=planner.route, recovery=recovery,
                                used_ompl=recovery in {"ompl", "ompl_failed"})
                            return outcome
                        # END subsequent failure accounting applies only to rejected curves.
                        check()
        # END task, contact, unknown-space and continuous-domain checks are unchanged.
        diagnostics.update(attempt, route=planner.route, recovery=recovery, used_ompl=recovery in {"ompl", "ompl_failed"})
        # BEGIN completed finite search reports its selected failure evidence.
        candidate_source = ("validated_reference_seed" if trajectory is reference_trajectory
                            else "validated_optimizer_candidate")
        diagnostics.update(failure=(_unknown_policy_failure(attempt, scene.unknown_policy)
            or "no_feasible_solution_found"), diagnostic_trajectory=trajectory,
            diagnostic_candidate_source=candidate_source, diagnostic_validation_status="completed",
            diagnostic_candidate_report={**deepcopy(attempt), "source": candidate_source,
                                         "validation_status": "completed"},
            failure_scope="The selected candidate failed the declared requirements or unknown-space policy; "
                "its component evidence does not establish that every route to the fixed target fails")
        return outcome
        # END an executable result requires the completed qualification and binding above.
    except Exception as error:
        # BEGIN every error precedes completed delivery and retains its own type.
        # BEGIN preserve the original failure stack without repeating any query.
        import traceback
        diagnostics.update(failure=(error.reason if isinstance(error, MotionPlanningError)
            else "cumulative_planning_budget_exhausted" if isinstance(error, PlanningWorkExhausted)
            else "continuation_binding_gap" if isinstance(error, ContinuationBindingError)
            else "event_time_requirement_unsatisfied" if isinstance(error, TimeBindingError)
            else "planning_stage_failed"), failed_stage=stage, reason=str(error), error_type=type(error).__name__,
            failure_traceback=traceback.format_exc())
        # END no unsuccessful binding or interrupted solve authorizes execution.
        # END stage context and native API failures remain directly diagnosable.
        # BEGIN interruption diagnostics identify the curve that produced each record.
        candidate = getattr(planner, "diagnostic_optimizer_candidate", None)
        provenance = getattr(planner, "diagnostic_optimizer_provenance", None)
        if candidate is not None:
            diagnostics.update(diagnostic_candidate_source=provenance["source"],
                diagnostic_validation_status="not_completed", diagnostic_optimizer_provenance=deepcopy(provenance))
        else:
            candidate = getattr(planner, "delivery_candidate", None)
            if candidate is not None and reference_trajectory is not None:
                diagnostics.update(diagnostic_candidate_source=("validated_reference_seed"
                    if candidate is reference_trajectory else "validated_optimizer_candidate"
                    if candidate is trajectory else "diagnostic_optimizer_trial"),
                    diagnostic_validation_status="completed" if candidate is trajectory else "not_completed")
                if candidate is trajectory:
                    diagnostics["diagnostic_candidate_report"] = {
                        **deepcopy(attempt), "source": diagnostics["diagnostic_candidate_source"],
                        "validation_status": "completed"}
        candidate = trajectory if candidate is None else candidate
        if candidate is None:
            candidate = getattr(planner, "initialization_candidate", None)
        if isinstance(candidate, JointTrajectory):
            diagnostics["diagnostic_trajectory"] = candidate
            diagnostics["diagnostic_candidate_scope"] = "joint curve retained for failure diagnostics; qualification not implied"
        # END no diagnostic candidate enters the execution binding.
        # BEGIN interrupted qualification retains the original failed result and evidence.
        outcome.success, outcome.trajectory, outcome.execution_binding = False, None, None
        return outcome
        # END no exception can substitute an earlier, already-delivered request result.
    finally:
        # BEGIN interrupted initialization retains its completed actual coefficients.
        if planner is not None:
            for name in ("reference_initialization", "reference_ik_queries", "reference_ik_runtime",
                         "initial_seed_strategies", "last_optimizer_runtime"):
                if hasattr(planner, name):
                    diagnostics[name] = deepcopy(getattr(planner, name))
            diagnostics["functional_ik_candidates"] = deepcopy(planner.goal_ik_candidates)
            # BEGIN the library returns passive evidence before native resources close.
            diagnostics["preference_planner"] = planner.diagnostic_snapshot()
            # END artifact writers consume owned CPU records without a second solve.
        # END diagnostics do not qualify a partial candidate for execution.
        if backend_claimed:
            # Native kernels/iteration blocks have returned before this point.
            # A timeout cannot release storage while their call is in flight.
            backend._release()
        if owned_backend and backend is not None:
            backend.close()
        diagnostics["elapsed_seconds"] = time.monotonic()-started
        if ledger is not None:
            ledger["consumed_planning"]["planning_seconds"] = initial_seconds+diagnostics["elapsed_seconds"]
            request_seconds = ledger["consumed_planning"]["planning_seconds"]-motion_baseline.get("planning_seconds", 0.)
            if request_seconds >= ledger["planning_budget"]["planning_seconds"]:
                diagnostics["wall_time_overrun_s"] = request_seconds-ledger["planning_budget"]["planning_seconds"]
                diagnostics["planning_budget_exhausted"] = True
            # BEGIN distinguish planning-only retries from actual-world continuation.
            # A previous work ledger carries cumulative computation, but does
            # not by itself supply executed state, events or a physical clock.
            # Append the accurate mode without relabelling historical records.
            ledger["binding_records"].append({
                "candidate_id": None if outcome.execution_binding is None else outcome.execution_binding.get("candidate_id"),
                "mode": "initial" if previous is None else
                        "planning_retry" if continuation is None and rebind is None else
                        "continuation" if rebind is None else
                        rebind.get("kind") if isinstance(rebind, dict) else "invalid_rebind",
                "source": deepcopy(rebind.get("source")) if isinstance(rebind, dict) else None,
                "request_time_origin_s": (None if continuation is not None and prior is None else
                                           0. if prior is None else prior["request_time_origin_s"]),
                "accepted": outcome.success, "failed_stage": diagnostics.get("failed_stage"),
                "failure": diagnostics.get("failure"), "reason": diagnostics.get("reason"),
                "consumed_planning": deepcopy(ledger["consumed_planning"])})
            # END the mode describes this attempt, not all earlier request work.
            outcome.request_ledger = deepcopy(ledger)
            if outcome.execution_binding is not None:
                outcome.execution_binding["consumed_planning"] = deepcopy(ledger["consumed_planning"])
                outcome.execution_binding["request_ledger"] = deepcopy(ledger)
