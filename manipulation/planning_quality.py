"""Compare validated cubic trajectories independently of optimizer stage costs.

This CPU-only score uses errors from the same candidate's completed independent
validation. It does not establish feasibility, query robot geometry, or compute
new forward kinematics. The caller must retain the validation and curve together.
"""

# BEGIN one stage-independent quality measure for already accepted trajectories.
from __future__ import annotations

import numpy as np


def trajectory_quality(trajectory, measurements, problem, *, limit_joint_names,
                       velocity_bounds, acceleration_bounds, jerk_bounds):
    """Return task-center and dimensionless motion quality components.

    Functional event errors keep the original position norm, SO(3) angle and
    lifted-coordinate semantics recorded by ``PlanningProblem.measurements``.
    Each error is divided by its requested tolerance before squaring.
    Only goals remaining in this request contribute; an earlier event is not
    recreated for a continuation.

    For each derivative order 1, 2 and 3, integrate the squared joint derivative
    divided by the square of that joint's largest absolute corresponding bound,
    then divide by duration and joint count. Three-point Gauss-Legendre quadrature
    integrates these polynomial terms exactly on each saved cubic segment:
    squared velocity has degree four, acceleration degree two, and jerk degree
    zero. This measures piecewise jerk, not impulses at acceleration jumps.

    The final score is the sum of task-center and the three motion terms. It is
    a fixed acceptance preference, not the optimizer's weighted penalty sum;
    collision proxies, constraint penalties, reference poses and sampling counts
    do not contribute. Native derivative bounds are explicit CPU arrays of shape
    ``(2, len(limit_joint_names))`` and may be in a different named joint order.
    """
    names, bound_names = tuple(trajectory.joint_names), tuple(limit_joint_names)
    if (not names or len(set(names)) != len(names) or len(set(bound_names)) != len(bound_names)
            or any(name not in bound_names for name in names)):
        raise ValueError("Quality requires unique named trajectory joints and matching derivative bounds")
    columns = [bound_names.index(name) for name in names]
    times = np.asarray(trajectory.times, dtype=float)
    if (times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all()
            or times[0] != 0. or np.any(np.diff(times) <= 0)
            or not np.array_equal(times, trajectory._curve.x)
            or not np.isfinite(trajectory._curve.c).all()
            or trajectory._curve.c.shape != (4, len(times)-1, len(names))):
        raise ValueError("Quality requires one finite, complete named cubic on its original time grid")
    duration = float(times[-1])

    def error_value(key):
        if key not in measurements:
            raise ValueError(f"Quality is missing this candidate's validated task measurement: {key}")
        value = np.asarray(measurements[key], dtype=float)
        if value.shape != () or not np.isfinite(value) or value < 0:
            raise ValueError(f"Quality requires a finite nonnegative scalar task measurement: {key}")
        return float(value)

    task_center = 0.
    for index, goal in enumerate(problem.pose_events):
        if goal.progress < problem.start_progress:
            continue
        task_center += (error_value(f"function:{index}:position_error_m")/goal.position_tolerance_m)**2
        task_center += (error_value(f"function:{index}:rotation_error_rad")/goal.rotation_tolerance_rad)**2
    for index, goal in enumerate(problem.coordinate_events):
        if goal.progress < problem.start_progress:
            continue
        unit = "m" if goal.kind == "prismatic" else "rad"
        task_center += (error_value(f"coordinate:{index}:error_{unit}")/goal.tolerance)**2

    nodes, weights = np.polynomial.legendre.leggauss(3)
    widths = np.diff(times)
    moments = times[:-1, None]+widths[:, None]*(nodes[None]+1.)/2.
    motion_components = {}
    for order, (label, supplied) in enumerate((("velocity", velocity_bounds),
            ("acceleration", acceleration_bounds), ("jerk", jerk_bounds)), start=1):
        bounds = np.asarray(supplied, dtype=float)
        if (bounds.shape != (2, len(bound_names)) or not np.isfinite(bounds).all()
                or np.any(bounds[0] >= bounds[1])):
            raise ValueError(f"Quality requires finite ordered named {label} bounds")
        scale = np.max(np.abs(bounds[:, columns]), axis=0)
        values = trajectory._curve(moments, order)/scale
        with np.errstate(over="ignore", invalid="ignore"):
            integral = np.sum(widths[:, None]/2.*weights[None]*np.sum(values*values, axis=-1))
        value = float(integral/(duration*len(names)))
        if not np.isfinite(value):
            raise ValueError(f"Quality produced a nonfinite {label} integral")
        motion_components[label] = value
    motion = sum(motion_components.values())
    score = task_center+motion
    if not np.isfinite(score):
        raise ValueError("Quality produced a nonfinite task or total score")
    return {"score": float(score), "task_center": float(task_center), "motion": float(motion),
            "motion_components": motion_components}
# END independent geometry acceptance remains the caller's prerequisite.
