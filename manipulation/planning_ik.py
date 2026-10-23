"""Finite IK proposal policies with explicit reference and task semantics.

The caller owns static task/geometry checks and the native IK budget. These
CPU helpers neither create task goals nor certify connections or trajectories.
"""

# BEGIN numerical reference and functional candidate policies retain separate evidence.
from __future__ import annotations

import numpy as np


# BEGIN numerical reference selection is separate from functional qualification.
def select_ik_tracking_candidate(candidates, candidate_indices, pose_costs, previous,
                                 *, reports):
    """Select one warm start from the original static and native IK evidence.

    ``candidates`` contains named-order joint positions with shape ``[N,D]``;
    ``candidate_indices`` identifies the caller's finite, position-domain
    proposals. ``pose_costs[N]`` contains their finite native pose energies
    for the same reference query. ``previous[1,D]`` is the preceding tracking
    state, initially the measured entrance. ``reports[N]`` contains the same
    query's task/model components and original native pose-success flags.

    Original native success precedes numerical preferences. Every proposal
    then minimizes Euclidean joint displacement before using pose energy as
    a tiebreaker. Failed native outputs retain their approximation status and
    residual diagnostics under that same continuity-first numerical rule.
    Static task/model evidence breaks only equal numerical choices; an
    unqueried failed pose cannot displace a solved numerical reference.
    The original candidate index breaks exact ties. The returned index names
    at most one warm start. The native SeedManager fills the remaining fixed
    seed budget. An empty input index set has no tracking proposal.
    """
    values = np.asarray(candidates, dtype=float)
    costs = np.asarray(pose_costs, dtype=float)
    _validate_reports(reports, len(values))
    solved = np.asarray([row["native_pose_ik_success"] for row in reports], dtype=bool)
    prior = np.asarray(previous, dtype=float)[0]
    # BEGIN one reference ordering applies to solved and approximate proposals.
    selected = min(candidate_indices, default=None,
        key=lambda index: (not solved[index],
                           float(np.linalg.norm(values[index]-prior)),
                           float(costs[index]),
                           *_static_qualification(reports[index]), int(index)))
    # END pose residuals cannot override the preceding continuity choice.
    return None if selected is None else int(selected)
# END one selected numerical state leaves the remaining native seed slots free.


def _validate_reports(reports, count):
    """Require separate evidence for task checks, source geometry and pose IK."""
    if len(reports) != count:
        raise ValueError("IK candidates and static reports must align")
    for report in reports:
        if any(type(report.get(name)) not in (bool, np.bool_)
               for name in ("task_satisfied", "model_valid", "native_pose_ik_success")):
            raise ValueError("IK task, source and native pose validity must be explicit booleans")


def _measurement(report, name):
    """Return missingness and a finite measurement without inventing infinity."""
    value = report.get(name)
    if value is None:
        return 1, 0.
    number = float(value)
    return (0, number) if np.isfinite(number) else (1, 0.)


def _quality(report):
    """Supply additive, independent quality tiebreakers in documented units."""
    values = []
    for name in ("environment_clearance_m", "self_clearance_m", "joint_limit_margin_fraction"):
        missing, measurement = _measurement(report, name)
        values.extend((missing, -measurement))
    return tuple(values)


# BEGIN interpret original component semantics without another geometry query.
def _static_qualification(report):
    """Return known task/model failures and separate unestablished verdicts.

    ``planning._ConstrainedPlanner._rank_ik_candidates`` owns these reports. A
    completed task/native check provides finite positive violation values;
    NaN and infinities provide no definite violation measurement. The scene
    component families below are the definite constraints produced by
    ``OnlinePlanningScene.components``: registered-entity intersections,
    required contact, declared initial support and gripper control. Unknown
    coverage, numerical boundaries and missing model relations retain their
    original components but cannot supply a definite failure here.

    A failure and an unestablished verdict are mutually exclusive ranking
    states. An explicit violation wins when the same report also contains
    unknown relations; the report itself retains both pieces of evidence.
    Numerical reference consumers use these values only after their
    numerical criteria. Functional consumers use this original allowed-set
    evidence first and do not require convergence to a nominal pose centre.
    """
    evaluated = report.get("static_model_query") == "evaluated"

    def positive(value):
        return np.isfinite(value) and value > 0

    task_failed = evaluated and any(positive(value)
        for value in report.get("task_violations", {}).values())
    model_failed = evaluated and (
        any(positive(value) for value in report.get("native_violations", {}).values())
        or any(positive(component["maximum"])
               for name, component in report.get("scene_components", {}).items()
               if name.startswith(("collision/", "contact/", "initial_support/"))
               or name == "gripper/controller"))
    return (bool(task_failed), bool(model_failed),
            not task_failed and (not evaluated or not report["task_satisfied"]),
            not model_failed and (not evaluated or not report["model_valid"]))
# END a stored False without its definite component remains unestablished.


def rank_ik_candidates(candidates, reports, previous, *, count, joint_tolerance, query_kind):
    """Return bounded distinct proposals, without assembling a trajectory.

    Reference proposals compare original native success and numerical
    continuity before static evidence. Functional proposals compare definite
    task/model failures and unestablished verdicts before continuity, without
    using nominal-centre convergence. Both kinds retain finite proposals, explicitly
    leaving their failed checks in the reports; they never reject a skill.

    Distance to the nearest supplied previous state precedes
    extra environment clearance, self clearance and joint-limit margin. These
    optional quality measurements cannot buy a large posture jump. Missing or
    nonfinite measurements lose their quality tiebreak against finite ones.
    Numerical duplicates are removed after ordering so better qualified
    evidence owns the representative. This local ordering is only for bounded
    proposals/warm starts; ``select_ik_chains`` chooses complete reference chains.
    """
    values = np.asarray(candidates, dtype=float)
    prior = np.asarray(previous, dtype=float)
    if (values.ndim != 2 or values.shape[1] < 1 or prior.ndim != 2
            or prior.shape[1] != values.shape[1] or not len(prior)
            or not np.isfinite(prior).all() or type(count) is not int or count < 1
            or not np.isfinite(joint_tolerance) or joint_tolerance <= 0
            or query_kind not in ("reference", "functional")):
        raise ValueError("IK ranking needs finite previous states, positive bounds and an explicit query kind")
    _validate_reports(reports, len(values))

    def key(index):
        report = reports[index]
        continuity = float(np.linalg.norm(prior-values[index], axis=-1).min())
        static = _static_qualification(report)
        prefix = ((not report["native_pose_ik_success"], continuity, *static)
                  if query_kind == "reference" else (*static, continuity))
        return (*prefix, *_quality(report), int(index))

    selected = []
    for index in sorted(np.flatnonzero(np.isfinite(values).all(axis=-1)), key=key):
        if selected and np.any(np.max(np.abs(values[selected]-values[index]), axis=-1) <= joint_tolerance):
            continue
        selected.append(int(index))
        if len(selected) == count:
            break
    return np.asarray(selected, dtype=int)


def select_ik_chains(candidates_by_query, reports_by_query, previous, query_times_s,
                     velocity_limits, *, position_limits, pose_costs_by_query,
                     count, joint_tolerance):
    """Form bounded coarse joint paths from finite position-domain IK states.

    Every numerical node retains its native pose-success flag. A legal
    nonconverged state is an approximate proposal, not a solved reference.
    'pose_costs_by_query' supplies the untruncated native quadratic pose
    energy in the same candidate order. Position bounds admit nodes; velocity
    bounds only normalize the polyline's continuity cost. Neither the nodes
    nor their connecting edges certify the later continuous trajectory.

    Dynamic programming first minimizes native-failure count, then normalized
    joint velocity energy, failed-node pose energy and successful-node pose
    energy. Summed static evidence only breaks equal numerical paths.
    Unqueried or unresolved checks remain unestablished verdicts. Each last node keeps
    'count' histories. Samples with no finite legal proposal are filled
    between available nodes and measured q0, without inventing pose success.
    The resulting values initialize the common constrained trajectory solve;
    they never become mandatory event interpolation conditions.
    """
    pools = [np.asarray(values, dtype=float) for values in candidates_by_query]
    reports = list(reports_by_query)
    costs = [np.asarray(values, dtype=float) for values in pose_costs_by_query]
    start = np.asarray(previous, dtype=float)
    times = np.asarray(query_times_s, dtype=float)
    bounds = np.asarray(velocity_limits, dtype=float)
    positions_bounds = np.asarray(position_limits, dtype=float)
    if (not pools or len(pools) != len(reports) or len(pools) != len(costs)
            or start.ndim != 2 or start.shape[0] != 1 or start.shape[1] < 1
            or not np.isfinite(start).all() or times.shape != (len(pools),)
            or not np.isfinite(times).all() or times[0] <= 0 or np.any(np.diff(times) <= 0)
            or bounds.shape != (2, start.shape[1]) or not np.isfinite(bounds).all()
            or np.any(bounds[0] >= bounds[1]) or positions_bounds.shape != bounds.shape
            or not np.isfinite(positions_bounds).all()
            or np.any(positions_bounds[0] >= positions_bounds[1])
            or type(count) is not int or count < 1
            or not np.isfinite(joint_tolerance) or joint_tolerance <= 0):
        raise ValueError("IK chains need measured start, increasing physical times and named physical bounds")
    scales = np.max(np.abs(bounds), axis=0)
    layers = []
    for query, (values, rows, energy) in enumerate(zip(pools, reports, costs)):
        if values.ndim != 2 or values.shape[1] != start.shape[1] or energy.shape != (len(values),):
            raise ValueError("IK states, pose energies and reports must share candidate and joint axes")
        _validate_reports(rows, len(values))
        for row in rows:
            if row.get("candidate_source") != "native_ik":
                raise ValueError("Reference chain nodes must retain their native_ik source")
        legal = (np.isfinite(values).all(-1) & np.isfinite(energy)
                 & (values >= positions_bounds[0]).all(-1)
                 & (values <= positions_bounds[1]).all(-1))
        retained = []
        # BEGIN duplicate representatives follow the ordinary reference order.
        # The nearest preceding retained state supplies this local continuity
        # comparison. Complete-path continuity is subsequently evaluated by
        # dynamic programming with its actual predecessor and physical dt.
        preceding = start if not layers else pools[layers[-1][0]][layers[-1][1]]
        continuity = np.min(np.sum(((values[:, None]-preceding[None])/scales)**2, axis=-1), axis=-1)
        order = sorted(np.flatnonzero(legal), key=lambda index:
            (not rows[index]["native_pose_ik_success"], continuity[index], energy[index],
             *_static_qualification(rows[index]), index))
        # END tiny residual differences cannot own a farther duplicate state.
        for index in order:
            if retained and np.any(np.max(np.abs(values[retained]-values[index]), axis=-1) <= joint_tolerance):
                continue
            retained.append(int(index))
        if retained:
            layers.append((query, retained))
    if not layers:
        return []

    # BEGIN fixed node evidence and physical-time continuity define one ranking.
    # Pose energy and continuity remain separate lexicographic quantities; no
    # task-specific conversion weight or extra rejection threshold is added.
    states = {None: [((0, 0., 0., 0., 0, 0, 0, 0), ())]}
    previous_time = 0.
    normalizer = times[-1]*start.shape[1]
    for layer, (query, indices) in enumerate(layers):
        dt = float(times[query]-previous_time)
        next_states = {}
        for index in indices:
            report = reports[query][index]
            static = _static_qualification(report)
            options = []
            for preceding_index, paths in states.items():
                preceding = start[0] if layer == 0 else pools[layers[layer-1][0]][preceding_index]
                with np.errstate(over="ignore", invalid="ignore"):
                    scaled_velocity = (pools[query][index]-preceding)/(scales*dt)
                    edge = float(dt*np.dot(scaled_velocity, scaled_velocity)/normalizer)
                if not np.isfinite(edge):
                    continue
                for cost, path in paths:
                    solved = report["native_pose_ik_success"]
                    energy = float(costs[query][index])
                    combined = (cost[0]+int(not solved), cost[1]+edge,
                                cost[2]+(0. if solved else energy), cost[3]+(energy if solved else 0.),
                                *(cost[4+item]+int(value) for item, value in enumerate(static)))
                    options.append((combined, (*path, index)))
            next_states[index] = sorted(options)[:count]
        states = next_states
        previous_time = times[query]
    # END the unified solve enforces task requirements and scores motion quality.

    result = []
    for cost, path in sorted(item for paths in states.values() for item in paths):
        anchor_queries = [query for query, _ in layers]
        anchors = np.stack([pools[query][index] for query, index in zip(anchor_queries, path)])
        anchor_times = np.r_[0., times[anchor_queries]]
        anchor_positions = np.concatenate((start, anchors))
        positions = np.column_stack([np.interp(times, anchor_times, anchor_positions[:, joint])
                                     for joint in range(start.shape[1])])
        if any(np.max(np.abs(np.asarray(prior["joint_positions"])-positions)) <= joint_tolerance
               for prior in result):
            continue
        indices = [None]*len(pools)
        for query, index in zip(anchor_queries, path):
            indices[query] = int(index)
        statuses = [("INTERPOLATED_PROPOSAL" if index is None else
                     "SOLVED" if reports[query][index]["native_pose_ik_success"]
                     else "APPROXIMATE_PROPOSAL")
                    for query, index in enumerate(indices)]
        # BEGIN every finite chain state is a coefficient proposal, not an observation.
        # Native success and original pool indices retain the selection
        # evidence. The common curve fit consumes the selected sample times.
        result.append({"candidate_indices": indices, "joint_positions": positions.tolist(),
            "anchor_query_indices": anchor_queries, "query_statuses": statuses,
            "selection_cost": {"static_task_failed_count": int(cost[4]),
                "static_model_failed_count": int(cost[5]),
                "nonconverged_candidate_count": int(cost[0]),
                "static_task_not_established_count": int(cost[6]),
                "static_model_not_established_count": int(cost[7]),
                "failed_pose_energy": float(cost[2]), "normalized_velocity_energy": float(cost[1]),
                "successful_pose_energy": float(cost[3])},
            "interpolated_queries": [query for query, index in enumerate(indices) if index is None],
            "reference_success": all(status == "SOLVED" for status in statuses)})
        # END the full task-space reference remains independent of proposal status.
        if len(result) == count:
            break
    return result
# END candidate evidence initializes fixed-clock curves; it never accepts executable motion.
