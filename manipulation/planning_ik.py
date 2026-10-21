"""Finite IK proposal policies with explicit reference and task semantics.

The caller owns static task/geometry checks and the native IK budget. These
CPU helpers neither create task goals nor certify connections or trajectories.
"""

# BEGIN reference qualification and complete-chain continuity replace clearance-first greed.
from __future__ import annotations

import numpy as np


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


def _tier(report, query_kind):
    """Keep task/source checks independent of optional nominal pose success."""
    return (not report["task_satisfied"], not report["model_valid"],
            not report["native_pose_ik_success"] if query_kind == "reference" else False)


def rank_ik_candidates(candidates, reports, previous, *, count, joint_tolerance, query_kind):
    """Return bounded distinct proposals, without assembling a trajectory.

    ``query_kind='reference'`` prefers successful nominal-pose IK within the
    same task/source screening tier. ``query_kind='functional'`` does not use
    that flag: the declared allowed set can pass despite failure to reach its
    nominal centre. Both kinds retain finite recovery proposals, explicitly
    leaving their failed checks in the reports; they never reject a skill.

    Within one tier, distance to the nearest supplied previous state precedes
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
        return (*_tier(report, query_kind),
                float(np.linalg.norm(prior-values[index], axis=-1).min()),
                *_quality(report), int(index))

    selected = []
    for index in sorted(np.flatnonzero(np.isfinite(values).all(axis=-1)), key=key):
        if selected and np.any(np.max(np.abs(values[selected]-values[index]), axis=-1) <= joint_tolerance):
            continue
        selected.append(int(index))
        if len(selected) == count:
            break
    return np.asarray(selected, dtype=int)


def select_ik_chains(candidates_by_query, reports_by_query, previous, query_times_s,
                     velocity_limits, *, count, joint_tolerance, query_kind="reference"):
    """Connect qualified reference anchors and interpolate missing seed samples.

    Inputs retain the caller's candidate indices. ``previous`` is the measured
    start of shape ``[1,D]`` at t=0; strictly increasing ``query_times_s`` are
    physical seconds, and ``velocity_limits`` contains named-order native
    lower/upper bounds of shape ``[2,D]``. No wraparound, retiming or extra IK
    query is introduced. Each report requires an explicit ``candidate_source``;
    only ``native_ik`` may report native pose success.

    Finite native IK successes can be anchors even when static task or source
    checks fail. Native success includes the native solver's constraints and
    convergence; it does not establish observation coverage or task success.
    Failed native queries do not become graph vertices or joint targets.
    The graph connects successive SOLVED queries using their actual time gap;
    missing samples are interpolated between the measured start and selected
    anchors, with constant continuation after the last anchor. Those samples
    are explicitly FALLBACK_SEED, not solutions to their requested tool pose.
    With no qualified anchor, return no reference chain: the caller owns its
    independent measured-state initializer. Every emitted chain still needs
    whole-curve optimization and validation.

    ``query_kind='functional'`` instead requires a task-qualified state at
    every supplied event and rejects connections whose average velocity is
    outside any joint's signed model bounds. Native pose-centre convergence
    is irrelevant to these continuous allowed sets. Static scene failures
    remain explicit proposal quality; the whole curve still owns acceptance.
    Reference samples are soft observations and do not impose those mandatory
    interpolation nodes or a hard velocity test on a reference polyline.

    Dynamic programming retains at most ``count`` paths at each last candidate.
    Failed static task/model check counts rank proposals first, without
    removing the only available transferred-reference chain. Edges then
    integrate squared joint velocity divided by the square of the
    joint's largest absolute native velocity bound. Dividing by the final
    query time and D yields the normalized velocity energy of the selected
    joint-space polyline. This first-order additive cost makes the bounded
    dynamic program exact for these supplied candidates. Clearance and joint
    margin break only equal-energy ties. They never certify the edges.
    """
    pools = [np.asarray(values, dtype=float) for values in candidates_by_query]
    reports = list(reports_by_query)
    start = np.asarray(previous, dtype=float)
    times = np.asarray(query_times_s, dtype=float)
    bounds = np.asarray(velocity_limits, dtype=float)
    if (not pools or len(pools) != len(reports) or start.ndim != 2 or start.shape[0] != 1
            or start.shape[1] < 1 or not np.isfinite(start).all()
            or times.shape != (len(pools),) or not np.isfinite(times).all()
            or times[0] <= 0 or np.any(np.diff(times) <= 0)
            or bounds.shape != (2, start.shape[1]) or not np.isfinite(bounds).all()
            or np.any(bounds[0] >= bounds[1]) or type(count) is not int or count < 1
            or not np.isfinite(joint_tolerance) or joint_tolerance <= 0
            or query_kind not in {"reference", "functional"}):
        raise ValueError("IK chains need finite measured start, increasing physical times and ordered velocity bounds")
    scales = np.max(np.abs(bounds), axis=0)
    if np.any(scales <= 0):
        raise ValueError("IK chain velocity scales must be positive")

    # BEGIN only native failure excludes a reference initializer from the graph.
    layers = []
    for query, (values, rows) in enumerate(zip(pools, reports)):
        if values.ndim != 2 or values.shape[1] != start.shape[1]:
            raise ValueError("Every IK query pool must use the same named joint layout")
        _validate_reports(rows, len(values))
        for row in rows:
            source = row.get("candidate_source")
            if not isinstance(source, str) or not source:
                raise ValueError("IK chain candidates require an explicit candidate_source")
            if source != "native_ik" and row["native_pose_ik_success"]:
                raise ValueError("A recovery source cannot claim native pose IK success")
        # BEGIN task events use their actual allowed-set qualification.
        qualified = [int(index) for index in np.flatnonzero(np.isfinite(values).all(axis=-1))
                     if (rows[index]["task_satisfied"] if query_kind == "functional" else
                         rows[index]["candidate_source"] == "native_ik"
                         and rows[index]["native_pose_ik_success"])]
        retained = []
        for index in sorted(qualified,
                            key=lambda index: (*_tier(rows[index], query_kind), *_quality(rows[index]), index)):
            if retained and np.any(np.max(np.abs(values[retained]-values[index]), axis=-1) <= joint_tolerance):
                continue
            retained.append(index)
        if retained:
            layers.append((query, retained))
        elif query_kind == "functional":
            return []
        # END a missing event candidate cannot be replaced by interpolation.
    if not layers:
        return []
    # END all discarded native results remain in the caller's original query reports.

    # State tuples contain the additive rank and original candidate-index
    # sequence. Keeping K predecessors per terminal node suffices for K best
    # complete paths because all future edge costs depend only on that node.
    # BEGIN static failures rank proposals; they cannot erase all native successes.
    states = {None: [((0, 0, 0., 0, 0., 0, 0., 0, 0.), ())]}
    previous_time = 0.
    normalizer = times[-1]*start.shape[1]
    for layer, (query, indices) in enumerate(layers):
        dt = float(times[query]-previous_time)
        next_states = {}
        for index in indices:
            quality = _quality(reports[query][index])
            task_failure = int(not reports[query][index]["task_satisfied"])
            model_failure = int(not reports[query][index]["model_valid"])
            options = []
            for preceding_index, paths in states.items():
                preceding = start[0] if layer == 0 else pools[layers[layer-1][0]][preceding_index]
                # BEGIN integrate the signed joint-speed bounds over this edge.
                # This is a necessary condition at fixed physical times; a
                # passing edge still needs continuous curve validation.
                displacement = pools[query][index]-preceding
                if query_kind == "functional" and (np.any(displacement < bounds[0]*dt)
                                                     or np.any(displacement > bounds[1]*dt)):
                    continue
                # END no collision proxy or task-specific margin is used.
                with np.errstate(over="ignore", invalid="ignore"):
                    scaled_velocity = displacement/(scales*dt)
                    edge = float(dt*np.dot(scaled_velocity, scaled_velocity)/normalizer)
                if not np.isfinite(edge):
                    continue
                for cost, path in paths:
                    combined = (cost[0]+task_failure, cost[1]+model_failure, cost[2]+edge,
                                *(old+new for old, new in zip(cost[3:], quality)))
                    if np.isfinite(combined).all():
                        options.append((combined, (*path, index)))
            next_states[index] = sorted(options)[:count]
        states = next_states
        previous_time = times[query]

    result = []
    for cost, path in sorted(item for paths in states.values() for item in paths):
        # BEGIN gap samples derive only from measured q0 and successful native anchors.
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
        recovery = [{"query_index": query, "candidate_index": None,
                     "candidate_source": "interpolated_success_anchors",
                     "reasons": ["reference_query_has_no_qualified_native_candidate"]}
                    for query, index in enumerate(indices) if index is None]
        result.append({"candidate_indices": indices, "joint_positions": positions.tolist(),
            "anchor_query_indices": anchor_queries,
            "query_statuses": ["FALLBACK_SEED" if index is None else "SOLVED" for index in indices],
            "selection_cost": {"static_task_failure_count": int(cost[0]),
                "static_model_failure_count": int(cost[1]), "normalized_velocity_energy": float(cost[2]),
                "environment_clearance_missing_count": int(cost[3]),
                "environment_clearance_sum_m": float(-cost[4]),
                "self_clearance_missing_count": int(cost[5]),
                "self_clearance_sum_m": float(-cost[6]),
                "joint_margin_missing_count": int(cost[7]),
                "joint_margin_sum": float(-cost[8])},
            "recovery_queries": recovery, "reference_success": not recovery})
        # END interpolated joint samples initialize a curve; they never claim pose convergence.
        if len(result) == count:
            break
    return result
    # END finite proposal ranking leaves static failures visible for full-curve checks.
# END candidate evidence initializes fixed-clock curves; it never accepts executable motion.
