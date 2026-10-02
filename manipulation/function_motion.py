"""The released function-frame optimizer with an explicit geometric entry.

Input poses already describe a fixed, physically grounded function reference.
There is no requirement for named function/center/grasp points here. Keeping
that function origin matters: shifting it to the carrier origin changes both
the position objective and the translation increments under tool rotation.

The optimizer below deliberately preserves the released xyz-Euler objective
and sampled increment constraints. It is not an SO(3)-intrinsic optimizer, a
collision planner, or a time-parameterized velocity/acceleration controller.
It produces an optional initialization reference. A FunctionReferenceError
does not invalidate a separately bound functional goal.
"""

# BEGIN function-frame numerical kernel separated from semantic keypoints.
import casadi as ca
import numpy as np
from scipy.spatial.transform import Rotation

from .motion import _errors, _pose_array, _poses_array


# BEGIN isolate optional reference fitting from independently bound task goals.
class FunctionReferenceError(ValueError):
    """The sampled reference kernel failed; this is not task infeasibility."""
# END isolate optional reference fitting from independently bound task goals.


def _pose_pair(value, name):
    value = np.asarray(value, dtype=float)
    if value.shape != (2, 3) or not np.isfinite(value).all():
        raise ValueError(f"{name} must contain finite position and xyz-Euler vectors")
    return value.copy()


def _keyframes(function_keyframes, n_points):
    """Validate the single explicit event-list contract in the numerical kernel."""
    if not isinstance(function_keyframes, (list, tuple)) or not function_keyframes:
        raise ValueError("function_keyframes must be a nonempty sequence")
    result, seen = [], set()
    for value in function_keyframes:
        if not isinstance(value, dict):
            raise ValueError("Function keyframes must be records with index and pose")
        index = value.get("index")
        if (isinstance(index, (bool, np.bool_))
                or not isinstance(index, (int, np.integer))
                or not 0 < int(index) < n_points):
            raise ValueError("Function keyframe index must be an integer after the initial node and in range")
        if int(index) in seen:
            raise ValueError("Duplicate function keyframe indices are not allowed")
        seen.add(int(index))
        result.append({"index": int(index), "pose": _pose_pair(value["pose"], "Function keyframe")})
    return sorted(result, key=lambda item: item["index"])


def optimize_trajectory(reference_trajectory, static_pose, function_keyframes):
    """Preserve the original sampled objective and equality constraints.

    ``function_keyframes=[{'index': i, 'pose': (position, xyz_angles)}, ...]``
    supplies every hard function event. Physical contact is not required
    (pouring is an example). The trajectory length is taken from the supplied
    reference, and the original strict angular increment limits are fixed.
    This function performs one solve and returns an (N,2,3) pose-pair array.
    Solver failures propagate; there is no weaker constraint mode or retry.
    """
    reference = np.asarray(reference_trajectory, dtype=float)
    if (reference.ndim != 3 or reference.shape[1:] != (2, 3)
            or len(reference) < 2 or not np.isfinite(reference).all()):
        raise ValueError("reference_trajectory must contain at least two finite pose pairs")
    n_points = len(reference)
    static = _pose_pair(static_pose, "static_pose")
    events = _keyframes(function_keyframes, n_points)
    opti = ca.Opti()
    static_pos, static_ori = static
    positions = opti.variable(n_points, 3)
    euler_angles = opti.variable(n_points, 3)

    # These expressions, the 40%-sample split, and the function-origin metric
    # are the released numerical problem. Do not substitute a body origin,
    # uniformly timed velocity, geodesic loss, or softer event penalty here.
    cost = 0
    for t in range(n_points):
        ref_pos, ref_euler = reference[t]
        step_cost = ca.sumsqr(positions[t, :] - np.array(ref_pos).reshape((1, 3))) + \
            ca.sumsqr(ca.fmod(euler_angles[t, :] - np.array(ref_euler).reshape((1, 3))
                             + np.pi, np.pi * 2) - np.pi)
        if t / n_points > 0.4:
            cost += step_cost
            if t < n_points - 1:
                opti.subject_to(ca.sumsqr(positions[t + 1, :] - positions[t, :]) <= 0.01 ** 2)
                opti.subject_to(ca.sumsqr(ca.fmod(euler_angles[t + 1, :] - euler_angles[t, :]
                                               + np.pi, np.pi * 2) - np.pi) <= (np.pi / 72) ** 2)
        else:
            opti.subject_to(ca.sumsqr(positions[t + 1, :] - positions[t, :]) <= 0.01 ** 2)
            opti.subject_to(ca.sumsqr(ca.fmod(euler_angles[t + 1, :] - euler_angles[t, :]
                                           + np.pi, np.pi * 2) - np.pi) <= (np.pi / 36000) ** 2)

    opti.subject_to(positions[0, :] == np.array(static_pos).reshape((1, 3)))
    opti.subject_to(euler_angles[0, :] == np.array(static_ori).reshape((1, 3)))
    for event in events:
        function_pos, function_ori = event["pose"]
        opti.subject_to(positions[event["index"], :] == np.array(function_pos).reshape((1, 3)))
        opti.subject_to(euler_angles[event["index"], :] == np.array(function_ori).reshape((1, 3)))
    opti.minimize(cost)
    # The early squared angular bound is only (pi/36000)^2 ~= 7.6e-9.
    # IPOPT's default bound relaxation is comparable to that complete bound,
    # so its success status alone need not satisfy the intended angle limit.
    # Tighten numerical convergence and disable relaxation, keeping exactly
    # the same objective/equalities/inequalities and strict angular limits.
    opti.solver("ipopt", {"print_time": False}, {
        "print_level": 0, "sb": "yes", "tol": 1e-10,
        "constr_viol_tol": 1e-12, "bound_relax_factor": 0., "acceptable_iter": 0,
    })
    # BEGIN distinguish a numerical reference-fit failure from malformed input.
    try:
        solution = opti.solve()
    except RuntimeError as error:
        raise FunctionReferenceError("Function reference optimization failed") from error
    # END distinguish a numerical reference-fit failure from malformed input.
    return np.stack((solution.value(positions), solution.value(euler_angles)), axis=1)


def transfer_function_motion(reference_poses, initial_pose, function_goals):
    """Optimize explicit function frames with one or more hard function events.

    ``reference_poses`` is (N,4,4) and ``initial_pose`` is (4,4), all in one
    grounded reference frame. ``function_goals`` contains ``{'index': i,
    'pose': T_reference_function}``; indices are array rows, after the initial
    node and before N. The caller maps original event frame IDs to these rows
    without replacing source timestamps or dropping intermediate events.

    All events are hard equalities in one solve with the strict original limits. The
    returned independent SE(3) checks compare against the supplied goals, not
    against redefined goals derived from the optimizer's output. Passing these
    checks establishes geometry at saved nodes, not task/robot success.
    """
    reference = _poses_array(reference_poses, "reference_poses")
    initial = _pose_array(initial_pose, "initial_pose")
    if len(reference) < 2:
        raise ValueError("Function motion needs at least two trajectory nodes")
    if not isinstance(function_goals, (list, tuple)) or not function_goals:
        raise ValueError("function_goals must include at least one explicit function event")
    pose_goals, raw_goals = [], []
    for event in function_goals:
        if not isinstance(event, dict):
            raise ValueError("Each function goal must contain index and pose")
        goal = _pose_array(event["pose"], "Function goal pose")
        pose_goals.append({"index": event.get("index"), "pose": goal})
        raw_goals.append({"index": event.get("index"), "pose": (
            goal[:3, 3], Rotation.from_matrix(goal[:3, :3]).as_euler("xyz"))})
    normalized = _keyframes(raw_goals, len(reference))
    pose_goals.sort(key=lambda event: int(event["index"]))
    reference_pairs = np.stack((reference[:, :3, 3],
                                Rotation.from_matrix(reference[:, :3, :3]).as_euler("xyz")), axis=1)
    initial_pair = np.stack((initial[:3, 3], Rotation.from_matrix(initial[:3, :3]).as_euler("xyz")))
    # These are necessary reachability bounds implied by the preserved node
    # constraints. Failing them is a proof of infeasibility, not a reason to
    # loosen limits or spend another solve on the same conflicting goals.
    angular_limits = np.where(np.arange(len(reference) - 1) / len(reference) > .4,
                              np.pi / 72, np.pi / 36000)
    previous_index, previous_pose = 0, initial_pair
    for event in normalized:
        index, pair = event["index"], event["pose"]
        distance = np.linalg.norm(pair[0] - previous_pose[0])
        angle = np.linalg.norm((pair[1] - previous_pose[1] + np.pi) % (2 * np.pi) - np.pi)
        if (distance > .01 * (index - previous_index) + 1e-7
                or angle > angular_limits[previous_index:index].sum() + 1e-7):
            raise FunctionReferenceError("Function reference events exceed the kernel's per-node increment limits")
        previous_index, previous_pose = index, pair
    raw = optimize_trajectory(reference_pairs, initial_pair, normalized)
    raw = np.asarray(raw, dtype=float)
    if raw.shape != (len(reference), 2, 3) or not np.isfinite(raw).all():
        raise FunctionReferenceError("Function optimizer returned invalid pose pairs")
    output = np.repeat(np.eye(4)[None], len(raw), axis=0)
    output[:, :3, 3] = raw[:, 0]
    output[:, :3, :3] = Rotation.from_euler("xyz", raw[:, 1]).as_matrix()
    checks = []
    for index, goal in [(0, initial)] + [(item["index"], item["pose"]) for item in pose_goals]:
        position, rotation = _errors(goal[None], output[index:index + 1])
        check = {"index": int(index), "position_error_m": float(position[0]),
                 "rotation_error_rad": float(rotation[0]), "tolerance": 1e-6}
        if position[0] > 1e-6 or rotation[0] > 1e-6:
            raise FunctionReferenceError(f"Function optimizer failed independent equality verification at node {index}")
        checks.append(check)
    steps = np.linalg.norm(np.diff(raw[:, 0], axis=0), axis=1)
    euler_steps = np.linalg.norm(np.fmod(np.diff(raw[:, 1], axis=0) + np.pi,
                                       2 * np.pi) - np.pi, axis=1)
    translation_violation = float(max(0., (steps - .01).max()))
    angular_violation = float(max(0., (euler_steps - angular_limits).max()))
    # A solver success flag cannot certify intermediate constraints by itself.
    # Check the original increments independently, in metres and radians, with
    # declared numerical tolerances rather than silently accepting a bad node.
    if translation_violation > 1e-6 or angular_violation > 1e-6:
        raise FunctionReferenceError("Function optimizer violated independent per-node increment checks: "
                         f"{translation_violation:.6g} m, {angular_violation:.6g} rad")
    return {
        "status": "candidate",
        "poses": output,
        "goal_checks": checks[1:],
        "diagnostics": {
            "initial_check": checks[0],
            "kernel": "released_function_origin_xyz_euler_objective",
            "angular_constraint_flag": 2,
            "solve_attempts": 1,
            "function_event_indices": [event["index"] for event in normalized],
            "translation_increment_limit_m": .01,
            "angular_increment_limits_rad": angular_limits,
            "max_translation_increment_m": float(steps.max()),
            "max_translation_increment_violation_m": translation_violation,
            "max_angular_increment_violation_rad": angular_violation,
            "translation_increment_check_tolerance_m": 1e-6,
            "angular_increment_check_tolerance_rad": 1e-6,
            "frame_origin": "supplied fixed function reference; never substituted with carrier origin",
            "scope": ("saved-node function-frame geometry; increments are not physical velocities; "
                      "no complete-scene collision, IK, force, stability, or timing guarantee"),
        },
    }
# END function-frame numerical kernel separated from semantic keypoints.
