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
               for name in ("task_satisfied", "source_valid", "native_pose_ik_success")):
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
    return (not report["task_satisfied"], not report["source_valid"],
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
                     velocity_limits, *, count, joint_tolerance):
    """Choose up to ``count`` distinct complete chains from finite query pools.

    Inputs retain the caller's candidate indices. ``previous`` is the measured
    start of shape ``[1,D]`` at t=0; strictly increasing ``query_times_s`` are
    physical seconds, and ``velocity_limits`` contains named-order native
    lower/upper bounds of shape ``[2,D]``. No wraparound, retiming or extra IK
    query is introduced. Each report requires an explicit ``candidate_source``;
    only ``native_ik`` may report native pose success.

    At each query, retain the best available task/source/reference-success
    tier and remove numerical duplicates. A failed nominal query is therefore
    an explicit recovery layer, while a successful candidate cannot lose to
    a failed one with more clearance or a shorter connection. These are only
    initializer policies: a tier failing actual task/source checks remains
    unqualified, and every emitted chain still needs whole-curve optimization.

    Dynamic programming retains at most ``count`` paths at each last candidate.
    Edges integrate squared joint velocity divided by the square of the
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
            or not np.isfinite(joint_tolerance) or joint_tolerance <= 0):
        raise ValueError("IK chains need finite measured start, increasing physical times and ordered velocity bounds")
    scales = np.max(np.abs(bounds), axis=0)
    if np.any(scales <= 0):
        raise ValueError("IK chain velocity scales must be positive")

    layers = []
    for values, rows in zip(pools, reports):
        if values.ndim != 2 or values.shape[1] != start.shape[1]:
            raise ValueError("Every IK query pool must use the same named joint layout")
        _validate_reports(rows, len(values))
        for row in rows:
            source = row.get("candidate_source")
            if not isinstance(source, str) or not source:
                raise ValueError("IK chain candidates require an explicit candidate_source")
            if source != "native_ik" and row["native_pose_ik_success"]:
                raise ValueError("A recovery source cannot claim native pose IK success")
        finite = np.flatnonzero(np.isfinite(values).all(axis=-1))
        if not len(finite):
            # The caller may offer a separately screened carried/measured
            # state; this policy must never manufacture one or its evidence.
            return []
        best_tier = min(_tier(rows[index], "reference") for index in finite)
        retained = []
        for index in sorted((int(index) for index in finite
                             if _tier(rows[index], "reference") == best_tier),
                            key=lambda index: (*_quality(rows[index]), index)):
            if retained and np.any(np.max(np.abs(values[retained]-values[index]), axis=-1) <= joint_tolerance):
                continue
            retained.append(index)
        layers.append(retained)

    # State tuples contain the additive rank and original candidate-index
    # sequence. Keeping K predecessors per terminal node suffices for K best
    # complete paths because all future edge costs depend only on that node.
    states = {None: [((0., 0, 0., 0, 0., 0, 0.), ())]}
    previous_time = 0.
    normalizer = times[-1]*start.shape[1]
    for query, indices in enumerate(layers):
        dt = float(times[query]-previous_time)
        next_states = {}
        for index in indices:
            quality = _quality(reports[query][index])
            options = []
            for preceding_index, paths in states.items():
                preceding = start[0] if query == 0 else pools[query-1][preceding_index]
                with np.errstate(over="ignore", invalid="ignore"):
                    scaled_velocity = (pools[query][index]-preceding)/(scales*dt)
                    edge = float(dt*np.dot(scaled_velocity, scaled_velocity)/normalizer)
                if not np.isfinite(edge):
                    continue
                for cost, path in paths:
                    combined = (cost[0]+edge, *(old+new for old, new in zip(cost[1:], quality)))
                    if np.isfinite(combined).all():
                        options.append((combined, (*path, index)))
            next_states[index] = sorted(options)[:count]
        states = next_states
        previous_time = times[query]

    result = []
    for cost, path in sorted(item for paths in states.values() for item in paths):
        positions = np.stack([pools[query][index] for query, index in enumerate(path)])
        if any(np.max(np.abs(np.asarray(prior["joint_positions"])-positions)) <= joint_tolerance
               for prior in result):
            continue
        recovery = []
        for query, index in enumerate(path):
            report = reports[query][index]
            reasons = []
            if not report["task_satisfied"]:
                reasons.append("static_task_requirements_not_satisfied")
            if not report["source_valid"]:
                reasons.append("static_source_requirements_not_satisfied")
            if not report["native_pose_ik_success"]:
                reasons.append("reference_pose_ik_not_successful")
            if report["candidate_source"] != "native_ik":
                reasons.append("reference_query_has_no_finite_native_candidate")
            if reasons:
                recovery.append({"query_index": query, "candidate_index": int(index),
                    "candidate_source": report["candidate_source"], "reasons": reasons})
        result.append({"candidate_indices": list(path), "joint_positions": positions.tolist(),
            "selection_cost": {"normalized_velocity_energy": float(cost[0]),
                "environment_clearance_missing_count": int(cost[1]),
                "environment_clearance_sum_m": float(-cost[2]),
                "self_clearance_missing_count": int(cost[3]),
                "self_clearance_sum_m": float(-cost[4]),
                "joint_margin_missing_count": int(cost[5]),
                "joint_margin_sum": float(-cost[6])},
            "recovery_queries": recovery, "reference_success": not recovery})
        if len(result) == count:
            break
    return result
# END candidate evidence initializes fixed-clock curves; it never accepts executable motion.
