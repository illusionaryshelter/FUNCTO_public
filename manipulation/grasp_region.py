"""Check whether a proposed tool pose can reach its fixed contact permissions.

This is a known-robot/declared-domain query, with no target occupancy input.
One hypothetical finger configuration must allow every declared material to
intersect at least one of its permitted patches. An intersection is permission
compatibility only: it does not establish an observed surface, actual contact,
force closure, a future finger trajectory, or Holding.
"""

# BEGIN candidate-domain geometry reuses the existing native permission law.
from collections.abc import Mapping
from itertools import product
import math

import numpy as np
from scipy.optimize import linprog

from .motion import _pose_array
from .robot_geometry_check import _feature_region_halfspaces


def _array(value, shape, label):
    result = np.asarray(value)
    if (result.shape != shape or result.dtype.kind not in "iuf"
            or not np.isfinite(result).all()):
        raise ValueError(f"{label} must be finite with shape {shape}")
    return result.astype(float, copy=True)


def _rule_geometry(record, joint_count):
    """Validate only calibrated material/permission fields used by this query.

    Generation has no runtime sphere cover. Therefore complete ContactRule
    dictionaries and their explicitly calibrated geometric subset are both
    usable without inventing sphere IDs. Every physical parameter and both
    existing numerical tolerances are required; this layer supplies no new
    physical defaults. The full planner still validates its full ContactRule.
    """
    if not isinstance(record, Mapping):
        raise ValueError("Each material permission must be an explicit record")
    names = ("rule_id", "actor", "target_owner", "material_id", "source")
    required = (*names, "feature_kind", "feature_points", "feature_normal",
                "feature_joint_translation", "patch_origin", "patch_axes",
                "patch_half_size", "gap_bounds", "normal_tolerance_rad",
                "modeling_allowance_m", "geometry_error_m", "minimum_thickness_m",
                "numerical_position_tolerance_m", "numerical_angle_tolerance_rad")
    if not set(required) <= set(record):
        raise ValueError("Permission geometry must retain every declared material, patch and tolerance field")
    result = {key: record[key] for key in required}
    if any(not isinstance(result[key], str) or not result[key].strip() for key in names):
        raise ValueError("Material permissions require named rule, actor, owner, material and source")
    if result["actor"] != "robot":
        raise ValueError("A grasp candidate domain query only binds robot material")
    points = np.asarray(result["feature_points"])
    kind = result["feature_kind"]
    if (kind not in {"polygon", "edge"} or points.ndim != 2 or points.shape[1:] != (3,)
            or len(points) < (3 if kind == "polygon" else 2)
            or (kind == "edge" and len(points) != 2)):
        raise ValueError("Permission material needs its complete planar polygon or two-endpoint edge")
    result["feature_points"] = points = _array(points, points.shape, "feature_points")
    result["feature_normal"] = normal = _array(result["feature_normal"], (3,), "feature_normal")
    if (not np.isclose(np.linalg.norm(normal), 1., atol=1e-6)
            or np.max(np.abs((points-points[0]) @ normal)) > 1e-8):
        raise ValueError("Material vertices must lie in their explicit unit-normal plane")
    mapping = result["feature_joint_translation"]
    if mapping is not None:
        mapping = _array(mapping, (len(points), 3, joint_count), "feature_joint_translation")
        if not np.allclose(mapping, mapping[:1], atol=1e-10, rtol=0):
            raise ValueError("Prismatic material must translate rigidly in the shared finger coordinates")
        result["feature_joint_translation"] = mapping
    result["patch_origin"] = _array(result["patch_origin"], (3,), "patch_origin")
    result["patch_axes"] = axes = _array(result["patch_axes"], (3, 3), "patch_axes")
    if (not np.allclose(axes.T @ axes, np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(axes), 1., atol=1e-6)):
        raise ValueError("Permission patch axes must be a proper orthonormal frame")
    result["patch_half_size"] = half = _array(result["patch_half_size"], (2,), "patch_half_size")
    result["gap_bounds"] = gap = _array(result["gap_bounds"], (2,), "gap_bounds")
    if np.any(half <= 0) or gap[0] > gap[1]:
        raise ValueError("Permission patch extents and gap bounds must be valid")
    scalars = ("normal_tolerance_rad", "modeling_allowance_m", "geometry_error_m",
               "minimum_thickness_m", "numerical_position_tolerance_m", "numerical_angle_tolerance_rad")
    for key in scalars:
        value = result[key]
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number))
                or not np.isfinite(value)):
            raise ValueError(f"{key} must be an explicit finite scalar")
        result[key] = float(value)
    if (not 0 <= result["normal_tolerance_rad"] < math.pi/2
            or not 0 <= result["modeling_allowance_m"] <= result["geometry_error_m"]
            or result["minimum_thickness_m"] <= 0
            or result["modeling_allowance_m"] >= result["minimum_thickness_m"]/2
            or min(result["numerical_position_tolerance_m"], result["numerical_angle_tolerance_rad"]) < 0):
        raise ValueError("Permission allowances must retain their declared bounded physical meaning")
    return result


def check_grasp_region_compatibility(contact_rules, *, T_reference_tool,
                                     T_reference_owner, finger_position_limits):
    """Check existential compatibility with fixed, material-specific patches.

    ``contact_rules`` contains only the robot rules bound to this grasp arrival
    and one owner. Their feature coordinates use the named tool frame, patch
    coordinates use the owner frame, and affine finger columns use the same
    coordinate order as ``finger_position_limits`` (J by 2, metres). The caller
    binds those identities and supplies existing calibrated geometry. Rule
    ``required`` flags, clocks and collision permissions are never changed.

    Rules for one material are alternatives (OR); all materials must fit at
    one common hypothetical q (AND). A separate intersection point per material
    and shared continuous q variables give a linear feasibility problem for
    each rule combination. No finite set of sampled openings substitutes for
    the full declared coordinate box. The query does not assume synchronous
    fractional closure or assert that physical contact will stop at its q.

    ``compatible`` has a hypothetical witness; ``contradicted`` means every
    finite permission combination is geometrically incompatible; solver
    incompleteness gives ``unresolved``. Missing observed target sides cannot
    cause rejection because observed target geometry is not an input here.
    A caller must check the final planned arrival poses too if it lets the
    trajectory optimizer move within a tolerance around this nominal pose.
    """
    tool = _pose_array(T_reference_tool, "T_reference_tool")
    owner = _pose_array(T_reference_owner, "T_reference_owner")
    limits = np.asarray(finger_position_limits)
    if limits.ndim != 2 or limits.shape[1:] != (2,):
        raise ValueError("finger_position_limits must have shape [joints, 2]")
    limits = _array(limits, limits.shape, "finger_position_limits")
    if np.any(limits[:, 0] > limits[:, 1]):
        raise ValueError("Finger coordinate limits must be ordered")
    records = [_rule_geometry(row, len(limits)) for row in contact_rules]
    if (not records or len({row["rule_id"] for row in records}) != len(records)
            or len({row["target_owner"] for row in records}) != 1):
        raise ValueError("Grasp domain needs distinct rules for exactly one explicitly bound owner")
    grouped = {}
    for row in records:
        grouped.setdefault(row["material_id"], []).append(row)
    # One material name cannot silently identify different calibrated faces
    # in two alternative patches. Permissions vary; the actual material does not.
    fixed = ("feature_kind", "feature_points", "feature_normal", "feature_joint_translation")
    for alternatives in grouped.values():
        first = alternatives[0]
        if any(not np.array_equal(row[key], first[key]) for row in alternatives[1:] for key in fixed):
            raise ValueError("Alternative permissions must retain the same calibrated material")

    material_ids = sorted(grouped)
    report = {"status": "unresolved", "material_ids": material_ids,
              "target_owner": records[0]["target_owner"], "checked_rule_combinations": 0,
              "normal_rejected_rule_ids": [], "incomplete_rule_combinations": [],
              "scope": "Existential known-material compatibility with fixed declared permission patches",
              "actual_contact_verified": False, "observed_target_surface_verified": False,
              "holding_verified": False, "finger_trajectory_predicted": False,
              "numerical_certificate": False}
    q_reference = limits.mean(axis=1)
    choices = []
    for material in material_ids:
        valid = []
        for rule in grouped[material]:
            region, derivative, reason = _feature_region_halfspaces(rule, tool, owner, q_reference)
            if reason == "material_normal_outside_permission":
                report["normal_rejected_rule_ids"].append(rule["rule_id"])
                continue
            if reason is not None:
                # Valid affine-q inputs currently have no other unresolved
                # branch, but an incomplete native query must remain unknown.
                report.update(reason=reason)
                return report
            if derivative.shape[1] == 0:
                derivative = np.zeros((len(region[1]), len(limits)))
            valid.append((rule, region, derivative))
        if not valid:
            report.update(status="contradicted", reason="One material has no direction-compatible permission",
                          incompatible_material_id=material)
            return report
        choices.append(valid)

    joint_count, material_count = len(limits), len(material_ids)
    variable_count = joint_count+3*material_count
    variable_bounds = list(map(tuple, limits-q_reference[:, None])) + [(None, None)]*(3*material_count)
    for selected in product(*choices):
        matrices, bounds = [], []
        for index, (_, region, derivative) in enumerate(selected):
            # A x <= b + db_dq (q-q_reference). Each material has its own x;
            # every row shares the same q variables, including coupled fingers.
            block = np.zeros((len(region[1]), variable_count))
            block[:, :joint_count] = -derivative
            block[:, joint_count+3*index:joint_count+3*index+3] = region[0]
            matrices.append(block)
            bounds.append(region[1])
        matrix, bound = np.concatenate(matrices), np.concatenate(bounds)
        solved = linprog(np.zeros(variable_count), A_ub=matrix, b_ub=bound,
                         bounds=variable_bounds, method="highs",
                         options={"primal_feasibility_tolerance": 1e-9,
                                  "dual_feasibility_tolerance": 1e-9})
        report["checked_rule_combinations"] += 1
        identities = [row[0]["rule_id"] for row in selected]
        if solved.success:
            report.update(status="compatible", reason="All declared materials have one common hypothetical finger configuration",
                          selected_rule_ids=identities,
                          hypothetical_finger_positions_m=(q_reference+solved.x[:joint_count]).tolist(),
                          hypothetical_reference_contact_points=solved.x[joint_count:].reshape(material_count, 3).tolist(),
                          maximum_linear_inequality_excess_m=float(np.max(matrix@solved.x-bound)))
            return report
        if solved.status != 2:
            report["incomplete_rule_combinations"].append({"rule_ids": identities,
                                                         "solver_status": int(solved.status)})
    if report["incomplete_rule_combinations"]:
        report.update(reason="A finite permission-combination query did not complete")
    else:
        report.update(status="contradicted", reason="No shared finger configuration intersects all declared material domains")
    return report
# END hypothetical permissions never assert observed target contact or Holding.
