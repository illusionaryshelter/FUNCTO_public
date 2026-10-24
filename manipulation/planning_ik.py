"""Finite functional IK ordering using original task and model evidence.

The caller owns event declarations, native solves and scene checks. These
helpers order proposals and never certify a connecting path or trajectory.
"""

# BEGIN only functional candidate ordering remains after single-seed reference IK.
import numpy as np


def _validate_reports(reports, count):
    """Require separate evidence for task checks, source geometry and pose IK."""
    if len(reports) != count:
        raise ValueError("IK candidates and static reports must align")
    for report in reports:
        if any(type(report.get(name)) not in (bool, np.bool_)
               for name in ("task_satisfied", "model_valid")):
            raise ValueError("IK task and model validity must be explicit booleans")


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
    Functional consumers use this original allowed-set evidence first.
    The search branch rechecks scene qualification using its own prefix.
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


def rank_ik_candidates(candidates, reports, previous, *, count, joint_tolerance):
    """Return distinct finite functional proposals ordered by original evidence.

    Definite task/model failures and unestablished verdicts precede distance
    to the supplied numerical seed. Current branch validity owns the final
    scene decision. Source index breaks exact ties; no nominal-pose convergence
    flag or unmeasured quality proxy changes this allowed-set ordering.
    """
    values = np.asarray(candidates, dtype=float)
    prior = np.asarray(previous, dtype=float)
    if (values.ndim != 2 or values.shape[1] < 1 or prior.ndim != 2
            or prior.shape[1] != values.shape[1] or not len(prior)
            or not np.isfinite(prior).all() or type(count) is not int or count < 1
            or not np.isfinite(joint_tolerance) or joint_tolerance <= 0):
        raise ValueError("IK ranking needs finite previous states and positive bounds")
    _validate_reports(reports, len(values))

    def key(index):
        continuity = float(np.linalg.norm(prior-values[index], axis=-1).min())
        return (*_static_qualification(reports[index]), continuity, int(index))

    selected = []
    for index in sorted(np.flatnonzero(np.isfinite(values).all(axis=-1)), key=key):
        if selected and np.any(np.max(np.abs(values[selected]-values[index]), axis=-1) <= joint_tolerance):
            continue
        selected.append(int(index))
        if len(selected) == count:
            break
    return np.asarray(selected, dtype=int)
# END actual branch geometry and complete trajectory acceptance remain separate.
