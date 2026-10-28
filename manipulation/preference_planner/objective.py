"""One fixed objective for numerical search and exported-curve comparison.

The inputs are sampled states of the same physical curve, a fixed physical
clock, the current explicit task declarations, and the registered observed
scene. Tool poses map the declared TCP into the problem's reference frame;
positions are metres and quaternions are unit wxyz. No simulator state is
introduced by this calculation. Each caller supplies the same ordinary grid
and exact event queries so the resulting unscaled totals are comparable.
"""

import torch

from manipulation.planning_problem import _first_event_query
from manipulation.unified_solver import motion_cost, time_mean_weights

from .pose_roles import pose_objective, pose_objective_components, pose_objective_scales


def pose_prior_cost(problem, position, quaternion, progress, times, *, goal_ids=None):
    """Return dimensionless declared prior cost on the supplied event layout.

    All tensors share batch/time axes. ``goal_ids`` optionally selects
    ``('pose_events', index)`` in the supplied problem, as in event IK. Formal
    request identities are mapped by the existing motion-block caller. Each
    prior contributes at its earliest exact event occurrence once per batch.
    Position, swing, and finite twist use the existing event geometry; a free
    twist has no component. Required events retain their original hard rows.

    Scales are explicit positive objective data, separate from hard bounds.
    The first controlled comparison uses the old quality-scale numbers; this
    preserves that declared weighting, not an uncertainty interpretation.
    The caller applies the existing constraint_weight exactly once.
    """
    result = torch.zeros_like(progress)
    for index, event in enumerate(problem.pose_events):
        if (not problem.start_progress <= event.progress <= problem.end_progress
                or (goal_ids is not None and ('pose_events', index) not in goal_ids)
                or pose_objective(event)['role'] != 'prior'):
            continue
        errors = pose_objective_components(event, position, quaternion)
        scales = pose_objective_scales(event)
        preference = sum((errors[name] / scale).square() for name, scale in scales.items())
        mask = _first_event_query(progress, event.progress, times)
        result = result + torch.where(mask, preference, 0.)
    return result


def trajectory_objective_components(*, problem, scene, states, robot_state,
        progress, times, ordinary_indices, bounds, motion_configuration,
        native_dt, constraint_weight, include_geometry=True, geometry_report=None):
    """Evaluate the four original-coordinate preference terms once.

    ``states`` holds q/dq/ddq/dddq for the optimized named joints on all queries.
    ``robot_state`` is the existing robot_tool_states tuple: completed q/dq,
    joint names, robot-link poses, TCP pose, and TCP linear/angular velocity.
    ``ordinary_indices`` selects motion integration samples; exact event
    anchors additionally participate in task, prior, and geometry queries.

    Motion uses the maintained joint-state weights and fixed native interval.
    Process preferences use PlanningProblem's feasibility calculation. Prior
    scales and the existing constraint_weight define the pose weighting.
    Observed geometry uses the supplied scene's original differentiable cost.
    Hard task/domain/geometry qualification remains a separate calculation.
    Returned scalar tensors retain derivatives with respect to robot states.
    """
    q, _, names, poses, tool, twist = robot_state
    ordinary = tuple(value[:, ordinary_indices] for value in states)
    weights = states[0].new_tensor(time_mean_weights(
        times.detach().cpu().numpy()[0, ordinary_indices]))[None]
    costs = {'motion_quality': motion_cost(ordinary, bounds, motion_configuration,
                                           native_dt, weights)}
    preferences = problem.optimizer_preferences(*tool, progress,
        trajectory_time=times, tool_linear_velocity=twist[0], tool_angular_velocity=twist[1],
        phase='feasibility')
    costs['task_preferences'] = constraint_weight * preferences.sum()
    costs['pose_prior'] = constraint_weight * pose_prior_cost(
        problem, *tool, progress, times).sum()
    if include_geometry:
        costs['observed_geometry'] = constraint_weight * scene.optimization_objective(
            *tool, progress, q, names, trajectory_time=times,
            robot_link_poses=poses, report_out=geometry_report)
    return costs
