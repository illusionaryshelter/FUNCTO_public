"""Small CPU policies for ranking inverse-kinematics initialization candidates.

These policies consume completed checks from the caller. They do not perform
geometry queries, create task constraints, or certify a trajectory.
"""

# BEGIN lexicographic IK selection preserves feasibility before continuity.
import numpy as np


def rank_ik_candidates(candidates, reports, previous, *, count, joint_tolerance):
    """Return up to ``count`` numerically distinct finite candidate indices.

    Each report must explicitly provide ``task_satisfied`` and ``source_valid``
    booleans. These refer only to the caller's stated static point checks; a
    successful reference-pose IK is not a replacement for either check. The
    optional metre-valued ``environment_clearance_m`` and ``self_clearance_m``
    and dimensionless ``joint_limit_margin_fraction`` rank larger values first.
    A missing or nonfinite measurement ranks after every finite measurement,
    including a negative one. It must never masquerade as infinite clearance.

    The exact priority is task validity, source validity, environment clearance,
    self clearance, joint margin, then distance to the nearest previous branch.
    Numerical duplicates are removed only after sorting, so their representative
    retains the best available evidence. Different indices or configurations do
    not establish different kinematic branches. ``count`` bounds returned CPU
    records and never changes a solver's number of seeds.
    """
    values = np.asarray(candidates, dtype=float)
    prior = np.asarray(previous, dtype=float)
    if (values.ndim != 2 or values.shape[1] < 1 or prior.ndim != 2
            or prior.shape[1] != values.shape[1] or not len(prior)
            or not np.isfinite(prior).all() or len(reports) != len(values)
            or type(count) is not int or count < 1
            or not np.isfinite(joint_tolerance) or joint_tolerance <= 0):
        raise ValueError("IK ranking needs aligned candidate reports, finite previous states and positive bounds")
    for report in reports:
        if any(type(report.get(name)) not in (bool, np.bool_)
               for name in ("task_satisfied", "source_valid")):
            raise ValueError("IK task and source validity must be explicit booleans")

    def measured(report, name):
        value = report.get(name)
        if value is None:
            return True, 0.
        number = float(value)
        return (False, -number) if np.isfinite(number) else (True, 0.)

    def key(index):
        report = reports[index]
        return (not report["task_satisfied"], not report["source_valid"],
                *measured(report, "environment_clearance_m"),
                *measured(report, "self_clearance_m"),
                *measured(report, "joint_limit_margin_fraction"),
                float(np.linalg.norm(prior-values[index], axis=-1).min()), int(index))

    selected = []
    for index in sorted(np.flatnonzero(np.isfinite(values).all(axis=-1)), key=key):
        if selected and np.any(np.max(np.abs(values[selected]-values[index]), axis=-1) <= joint_tolerance):
            continue
        selected.append(int(index))
        if len(selected) == count:
            break
    return np.asarray(selected, dtype=int)
# END point evidence selects initializers, never complete executable motion.
