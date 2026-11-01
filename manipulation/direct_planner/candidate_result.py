"""Read a solved trajectory through the same task and geometry computations.

The solver point owns sampled robot states, registered-shape distances and
scene event sides. Exact polynomial extrema remain the physical joint test.
Owner coverage uses the same finite occupied coordinates and measured free
voxels; conditional regions retain their declared policy meaning.
"""

# BEGIN same-model candidate results replace recursive material requalification.
from copy import deepcopy
from time import perf_counter

import numpy as np
import torch

from manipulation.observed_collision import query_region_intersection


def _maximum(value):
    array = value.detach().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
    return float(np.max(array)) if np.isfinite(array).all() else float('inf')


def scene_conditions(point):
    """Evaluate declared relationship and unknown policy on retained states."""
    geometry, context = point.geometry, point.geometry.context
    owner = geometry.bound.owner
    scene = owner.scene
    costs = {key: _maximum(value) for key, value in context['relationship_components'].items()}
    conditional, coverage = {}, {}
    if context['hand'] is not None:
        costs['gripper/controller'] = _maximum(context['hand'].controller_residual)
    for rule in scene.contacts:
        if not rule.required:
            continue
        policy = context['policies'][rule.rule_id]
        normal, uncertain = context['contact_conditions'][rule.rule_id]
        active = policy['active'].numpy()[0]
        gaps, unresolved = [], []
        for index in np.flatnonzero(active):
            target = context['matrices']['owner/'+rule.target_owner][0, index]
            world_normals = policy['normals'][0, index].numpy() @ target[:3, :3].T
            world_bounds = policy['bounds'][0, index].numpy()+world_normals@target[:3, 3]
            evidence = query_region_intersection(owner.geometries['owner/'+rule.target_owner],
                target, world_normals, world_bounds, check_request=scene.check_request)
            gaps.append(float(evidence['status'] == 'missing_contact'))
            unresolved.append(float(evidence['status'] == 'unresolved'))
        costs['contact/'+rule.rule_id] = _maximum(normal*policy['active']*(1-uncertain))+max(gaps, default=0.)
        costs['unresolved/contact/'+rule.rule_id] = max(unresolved, default=0.)
        conditional['contact/'+rule.rule_id] = _maximum(uncertain)
    for model in scene._initial_support_models.values():
        components = model.components(context['owner_poses'][model.relation.actor],
            context['owner_poses'][model.relation.target_owner], context['progress'],
            check_request=scene.check_request)
        costs.update({key: _maximum(value) for key, value in components.items()})

    known = scene.coverage._known
    shape = np.asarray(known.shape[::-1])
    origin, h = scene.coverage.origin, scene.coverage.voxel_size
    for entity in scene.owners:
        if not entity.moving:
            continue
        local = owner.distance.inventory('owner/'+entity.owner_id, ()).points
        poses = context['matrices']['owner/'+entity.owner_id][0]
        world = np.einsum('nij,pj->npi', poses[:, :3, :3], local)+poses[:, None, :3, 3]
        flat = world.reshape(-1, 3)
        cells = np.floor((flat-origin)/h+.5).astype(np.int64)
        inside = np.all((cells >= 0) & (cells < shape), axis=1)
        observed = np.zeros(len(flat), bool)
        valid = cells[inside]
        observed[inside] = known[valid[:, 2], valid[:, 1], valid[:, 0]]
        declared = np.zeros(len(flat), bool)
        for region in scene.unknown_policy['conditional_regions']:
            declared |= np.all((flat >= region['lower_m']) & (flat <= region['upper_m']), axis=1)
        outside = ~observed & ~declared
        rows = np.flatnonzero(outside)
        coverage[entity.owner_id] = dict(query_points=len(flat), observed_points=int(observed.sum()),
            declared_region_points=int((~observed & declared).sum()), unknown_points=len(rows),
            first_unknown=None if not len(rows) else dict(
                state_index=int(rows[0]//len(local)), world_point=flat[rows[0]].tolist()))
        channel = conditional if scene.unknown_policy['outside_regions'] == 'allow_attempt' else costs
        channel['unknown/forbidden/'+entity.owner_id] = float(bool(len(rows)))
    return costs, conditional, coverage


def evaluate_candidate(planner, trajectory, *, record_work, objective):
    """Form one finite-query verdict and the exact curve's binding metadata."""
    started = perf_counter()
    record_work(passes=1)
    computation = planner.computation
    point = computation.final_point
    full = point.full_controls
    if not np.array_equal(full, planner._current_full_controls):
        raise ValueError('Candidate controls differ from the retained numerical result')
    planner.delivery_candidate, planner.last_validation = trajectory, None
    q, dq, names, tool, twist = point.candidate_state
    with torch.no_grad():
        terms = planner.problem.residuals(*tool, computation.progress, q, names, dq,
            trajectory_time=computation.times, tool_linear_velocity=twist[0], tool_angular_velocity=twist[1])
    # BEGIN final pose accuracy is posthoc evidence, independent of delivery.
    # The optimizer still consumes the original required pose-event constraints.
    # Here their metric excess is retained separately so neither the candidate
    # verdict nor execution_binding's validation violations recreates a pose gate.
    pose_residuals = {name: _maximum(value) for name, value in terms.items()
                      if name.startswith('function_pose:')}
    residuals = {name: _maximum(value) for name, value in terms.items()
                 if not name.startswith('function_pose:')}
    with torch.no_grad():
        measurements = planner.problem.measurements(*tool, computation.progress,
            trajectory_time=computation.times)
    pose_accuracy = dict(
        status='warning' if any(not np.isfinite(value) or value > 0.
            for value in pose_residuals.values()) else 'within_tolerance',
        scope='Posthoc candidate pose-event accuracy; original nominal solver constraints retained',
        residuals=pose_residuals,
        maximum_absolute_measurements={name: _maximum(value.abs())
            for name, value in measurements.items() if name.startswith('function:')},
        goals=[dict(goal_id=['pose_events', index], progress=goal.progress,
            component_tolerances=goal.component_tolerances)
            for index, goal in enumerate(planner.problem.pose_events)])
    # END all remaining task, joint and scene conditions keep their own verdict.
    issues = planner.problem.verification_issues(computation.progress)
    residuals['query_completeness'] = float(bool(issues))
    validation = planner._candidate_domain_validation(trajectory)
    residuals.update(trajectory.residuals)
    scene, conditional, coverage = scene_conditions(point)
    residuals.update(scene)
    geometry = point.geometry
    # BEGIN exported geometry belongs to the selected feasible candidate.
    # Later numerical trials may have changed the producer's last-query field;
    # the candidate owns an independent complete report for its own controls.
    geometry.bound.owner.last_evaluation = deepcopy(geometry.report)
    # END diagnostic export and scene markers use the delivered curve identity.
    collision = {}
    for index, row in enumerate(geometry.bound.records):
        collision[row['component']] = max(collision.get(row['component'], 0.), float(geometry.spatial_values[index]))
    residuals['process_scene'] = max(collision.values(), default=0.)
    report = dict(**objective, residuals=residuals, pose_accuracy=pose_accuracy, scene_components=scene,
        collision_residuals=collision, scene_update_residuals={},
        coverage_residuals={k: v for k, v in scene.items() if k.startswith('unknown/')},
        conditional_unknown_residuals=conditional, owner_coverage=coverage,
        observed_geometry=deepcopy(geometry.report), task_query_issues=issues,
        geometry_scope='Original scene queries and native interval endpoint/stationary minima',
        unknown_tsdf_query_count=int(geometry.unknown_counts.sum()),
        preference_joint_problem=dict(pose_objectives=planner.pose_objectives,
            reference_termination=planner.terminal.report,
            geometry_constraint_rows=len(geometry.values)))
    report['accepted'] = bool(np.isfinite(report['fixed_objective']) and
        all(np.isfinite(value) and value <= 0. for value in residuals.values()))
    parameters = computation.parameters
    validation.update(u=parameters.copy(), violations=np.full(len(parameters),
        max(residuals.values(), default=0.)), observed_geometry=deepcopy(geometry.report),
        scope='Finite task and original geometry queries; exact continuous joint polynomial bounds')
    record_work(points=len(parameters))
    planner.last_validation = validation if report['accepted'] else None
    planner.last_candidate_report = report
    planner.candidate_history.append(dict(accepted=report['accepted'], residuals=deepcopy(residuals),
        pose_accuracy=deepcopy(pose_accuracy),
        objective_components=deepcopy(report['objective_components']), controls=full.tolist()))
    planner.candidate_evaluations.append(dict(completed=True, accepted=report['accepted'],
        elapsed_seconds=perf_counter()-started, scope=validation['scope']))
    return trajectory, report, len(parameters)
# END conditional diagnostics remain distinct from hard task and collision values.
