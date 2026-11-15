"""Measure completed original cubic curves for fixed trajectory preferences.

Joint extrema and integrals are analytic polynomial calculations. Robot-pair
clearance and whole-tool motion use saved-curve samples and local refinements
with the reusable official robot model. The planner's original success flag
owns eligibility; these measurements preserve that completed decision.
"""
from __future__ import annotations
import numpy as np
from scipy.interpolate import CubicHermiteSpline, PPoly
from scipy.optimize import minimize_scalar
from scipy.signal import find_peaks
from scipy.spatial.transform import Rotation
from .physical_joint_domain import cubic_position_report

def polynomial_extrema(curve):
    """Measure both segment endpoints and all real stationary points.

    Each segment is evaluated independently, so both one-sided derivatives
    at a knot are included. This matters for piecewise cubic jerk, whose
    jumps do not represent a globally continuous jerk signal.
    """
    rows = []
    for joint in range(curve.c.shape[-1]):
        candidates = []
        for segment in range(len(curve.x)-1):
            piece = PPoly(curve.c[:, segment:segment+1, joint],
                          curve.x[segment:segment+2], extrapolate=False)
            stationary = piece.derivative().roots(extrapolate=False)
            times = np.r_[piece.x, stationary[np.isfinite(stationary)]]
            candidates.extend((float(piece(t)), float(t), segment) for t in times)
        minimum = min(candidates)
        maximum = max(candidates)
        rows.append(dict(minimum=minimum[0], minimum_time_s=minimum[1],
            minimum_segment=minimum[2], maximum=maximum[0],
            maximum_time_s=maximum[1], maximum_segment=maximum[2]))
    return rows


def polynomial_square_integral(curve):
    """Integrate each squared scalar polynomial exactly in physical seconds."""
    total = np.zeros(curve.c.shape[-1])
    for segment, width in enumerate(np.diff(curve.x)):
        for joint in range(len(total)):
            squared = np.polymul(curve.c[:, segment, joint],
                                 curve.c[:, segment, joint])
            integral = np.polyint(squared)
            total[joint] += np.polyval(integral, width)-np.polyval(integral, 0.)
    return total


def joint_metrics(curve, names, limits):
    """Use the planner's saved named model bounds for all four derivatives."""
    columns = [limits['joint_names'].index(name) for name in names]
    position_bounds = np.asarray(limits['position'])[:, columns]
    ranges = position_bounds[1]-position_bounds[0]
    duration = float(curve.x[-1]-curve.x[0])
    results = {}
    for order, key in enumerate(('position', 'velocity', 'acceleration', 'jerk')):
        derivative = curve.derivative(order)
        bounds = np.asarray(limits[key])[:, columns]
        extrema = polynomial_extrema(derivative)
        for index, row in enumerate(extrema):
            lower, upper = bounds[:, index]
            low_margin, high_margin = row['minimum']-lower, upper-row['maximum']
            use_lower = low_margin <= high_margin
            row.update(joint=names[index], lower_bound=float(lower),
                upper_bound=float(upper), margin=float(min(low_margin, high_margin)),
                margin_fraction=float(min(low_margin, high_margin)/(upper-lower)),
                margin_side='lower' if use_lower else 'upper',
                margin_time_s=row['minimum_time_s'] if use_lower else row['maximum_time_s'])
        closest = min(extrema, key=lambda row: row['margin'])
        closest_fraction = min(extrema, key=lambda row: row['margin_fraction'])
        saved_validation = cubic_position_report(derivative, *bounds)
        results[key] = dict(per_joint=extrema, minimum_margin=closest,
            minimum_fractional_margin=closest_fraction,
            maximum_absolute_value=max(max(abs(row['minimum']), abs(row['maximum'])) for row in extrema),
            maximum_bound_excess=saved_validation['maximum_bound_excess'],
            mean_squared_joint_range_normalized=float(np.mean(
                polynomial_square_integral(derivative)/ranges**2)/duration))
    return results


def robot_pair_metrics(validator, curve, fingers, times):
    """Query every required official pair, retaining pair and witness identity.

    The registered robot and its installed hand have fixed reusable official
    shapes and component relationships. Named joint curves and calibration
    place those shapes. Task objects remain in the original observed scene.
    A uniform clock plus saved knots and original scene witness time gives
    brackets; SciPy refines sampled distance minima. This numerical review
    describes these saved robot curves and leaves original acceptance intact.
    """
    checker, pairs = validator.checker, validator.required_pairs
    names = validator.arm_joint_names + validator.finger_joint_names
    positions = np.column_stack((curve(times), fingers(times)))
    numeric = np.concatenate(list(checker.query_pair_batches(
        joint_names=names, joint_positions=positions,
        T_reference_base=validator.T_reference_base, pairs=pairs,
        projection_mask=np.zeros((len(times), len(pairs)), bool))))
    distances = numeric[..., 0]
    boundary = max(checker.numerics['gjk_tolerance'], checker.numerics['epa_tolerance'])
    minima = []
    for column, pair in enumerate(pairs):
        values = distances[:, column]
        initial = int(values.argmin())
        best = float(values[initial]), float(times[initial])
        def query(t):
            report = checker.query(joint_names=names,
                joint_positions=np.r_[curve(t), fingers(t)],
                T_reference_base=validator.T_reference_base, pairs=(pair,))
            return report['pairs'][0]
        peaks, _ = find_peaks(-values, prominence=boundary)
        for index in peaks:
            refined = minimize_scalar(lambda t: query(t)['signed_distance_m'],
                bounds=(times[index-1], times[index+1]), method='bounded',
                options=dict(xatol=1e-8))
            best = min(best, (float(refined.fun), float(refined.x)))
        evidence = query(best[1])
        minima.append(dict(time_s=best[1], evidence=evidence))
    return dict(minimum=min(minima, key=lambda r: r['evidence']['signed_distance_m']),
        pair_minima=minima, sample_count=len(times), pair_count=len(pairs),
        numerical_boundary_m=boundary,
        scope='all required official robot pairs; saved-curve samples and bracketed local minima'), distances



def movement(poses):
    """Measure full sampled TCP travel; SO(3) angles use consecutive matrices."""
    poses = np.asarray(poses)
    return dict(length_m=float(np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1).sum()),
        rotation_travel_deg=float(np.degrees(Rotation.from_matrix(
            poses[:-1, :3, :3].transpose(0, 2, 1) @ poses[1:, :3, :3]).magnitude().sum())))


def deviation(poses, reference, times):
    """Compare the same physical instants in calibrated W, not Euler differences."""
    p = np.linalg.norm(poses[:, :3, 3] - reference[:, :3, 3], axis=1)
    r = np.degrees(Rotation.from_matrix(reference[:, :3, :3].transpose(0, 2, 1) @ poses[:, :3, :3]).magnitude())
    return dict(position_rms_m=float(np.sqrt(np.mean(p*p))), position_max_m=float(p.max()),
        position_max_time_s=float(times[p.argmax()]), rotation_rms_deg=float(np.sqrt(np.mean(r*r))),
        rotation_max_deg=float(r.max()), rotation_max_time_s=float(times[r.argmax()]))


def robot_tool_poses(fixture, feedback, names, positions):
    """Evaluate official URDF FK at each original named joint sample.

    This is the same fixed robot model used by the original recorded-curve
    measurement. It supplies robot kinematics only; task entities retain
    their independent RGB-D fields. Each sample uses a local mutable FK data
    object, so one hypothesis cannot overwrite another hypothesis's poses.
    """
    import pinocchio as pin
    model = pin.buildModelFromUrdf(fixture['urdf_path'])
    data, q = model.createData(), pin.neutral(model)
    slots = [model.joints[model.getJointId(str(name))].idx_q for name in names]
    tcp = model.getFrameId(feedback['tool_link'])
    base, tool = np.asarray(feedback['base_pose']), np.asarray(feedback['tool_transform'])
    poses = []
    for values in positions:
        q[slots] = values
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        poses.append(base @ data.oMf[tcp].homogeneous @ tool)
    return np.asarray(poses)


def measure(result, *, hypothesis_id, fixture, feedback, source_report,
            source_curve, curve=None):
    """Measure a formal or diagnostic original curve without changing status.

    A caller may supply the producer's diagnostic curve for a rejected result.
    Missing curves produce a missing measurement record. Robot required-pair
    queries use 1001 uniform times, original arm/finger knots and the original
    scene witness, followed by the unchanged bracketed scalar refinement.
    Arm-joint position margins retain radians and exclude prismatic fingers.
    """
    from .grounding_robot_geometry import build_robot_validator

    curve = result.trajectory if curve is None else curve
    base = dict(id=hypothesis_id, accepted=bool(result.success),
                source_report=str(source_report),
                source_curve=None if source_curve is None else str(source_curve),
                physical_status='missing_not_executed')
    if curve is None:
        return {**base, 'metrics_status': 'missing_curve',
                'failure': result.diagnostics.get('failure')}, {}
    d = result.diagnostics
    arm = CubicHermiteSpline(curve.times, curve.position, curve.velocity)
    finger = curve.finger_trajectory
    fingers = CubicHermiteSpline(finger.times, finger.position, finger.velocity)
    witness = d['observed_geometry']['maximum_condition']
    times = np.unique(np.r_[np.linspace(0, curve.times[-1], 1001), curve.times,
                            finger.times, witness['identity']['time_s']])
    validator = build_robot_validator(fixture, reference_frame=feedback['reference_frame'],
        T_reference_base=np.asarray(feedback['base_pose']), tool_frame=feedback['tool_link'],
        T_frame_tcp=np.asarray(feedback['tool_transform']))
    native, distances = robot_pair_metrics(validator, arm, fingers, times)
    poses = robot_tool_poses(fixture, feedback,
        list(curve.joint_names)+list(finger.joint_names),
        np.column_stack((arm(times), fingers(times))))
    values = dict(**base, metrics_status='completed', duration_s=curve.duration,
        original_objective=d['fixed_objective'], objective_components=d['objective_components'],
        joint=joint_metrics(arm, list(curve.joint_names), d['verification_joint_limits']),
        tool_movement=movement(poses), robot_self=native,
        original_geometry_condition_slack=-float(witness['value']),
        original_geometry_witness=witness, scene_collision_residuals=d['collision_residuals'],
        scene_observed_query_counts={key: d['observed_geometry']['observed_queries'][key]
            for key in ('query_count', 'unknown_query_count', 'disjoint_source_band_states')},
        pose_accuracy=d['pose_accuracy'],
        optimizer={key: d['optimizer'][key] for key in
                   ('iterations', 'status', 'selected_feasible', 'selected_trial_index')},
        planning_core_seconds=d['elapsed_seconds'],
        measurement_scope=dict(joints='Exact saved arm cubic extrema and polynomial integrals',
            robot_pairs=native['scope'], tool='Official FK on the saved measurement clock'))
    return values, dict(times=times, tool_poses=poses,
                        required_pair_distances_m=distances)
