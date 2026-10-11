"""Check a declared standalone tool against calibrated fixture material faces.

This recorded-input boundary compares known robot geometry only. It does not
read object state, estimate finger feedback, alter contact permissions, or
authorize execution. All feature coordinates use the actual controlled TCP;
affine columns and joint limits must already use the profile's declared native
finger-coordinate order. A mismatch is an input error, never a compensation.
"""

# BEGIN explicit native-tool consistency before learned grasp generation.
from itertools import product

import numpy as np


# Match the existing material-calibration absolute geometric tolerance. This
# is a numerical comparison between two known robot descriptions in metres,
# not a new contact allowance or an adjustable task acceptance threshold.
_POSITION_ATOL_M = 1e-9
_NORMAL_ATOL = 1e-9


def _array(value, shape, name):
    result = np.asarray(value)
    if (result.shape != shape or result.dtype.kind not in "fiu"
            or not np.isfinite(result).all()):
        raise ValueError(f"{name} must have finite numeric shape {shape}")
    return result.astype(float, copy=True)


def _corner_order(observed, expected):
    """Find an unambiguous vertex correspondence, independent of input order."""
    distance = np.linalg.norm(observed[:, None, :] - expected[None, :, :], axis=2)
    matches = distance <= _POSITION_ATOL_M
    if not (np.all(matches.sum(axis=0) == 1) and np.all(matches.sum(axis=1) == 1)):
        return None
    # Row i of observed is expected[order[i]]. Requiring a unique bijection
    # also excludes repeated vertices disguised as a full material rectangle.
    return matches.argmax(axis=1)


def _materials(contact_rules):
    unique = {}
    for rule in contact_rules:
        name = rule.get("material_id")
        if not isinstance(name, str) or not name:
            raise ValueError("Every fixture contact rule requires a named material")
        if rule.get("feature_kind") != "polygon":
            raise ValueError("Tool matching requires the complete native inner pad face")
        record = {
            "points": _array(rule["feature_points"], (4, 3), "native pad face points"),
            "normal": _array(rule["feature_normal"], (3,), "native pad face normal"),
            "affine": _array(rule["feature_joint_translation"], (4, 3, 2), "native pad joint translation"),
        }
        if not np.isclose(np.linalg.norm(record["normal"]), 1., atol=_NORMAL_ATOL, rtol=0):
            raise ValueError("Native material normal must already be unit length")
        if name in unique:
            # Multiple permission patches can reference one material; those
            # patches cannot quietly replace the material's calibrated motion.
            existing = unique[name]
            order = _corner_order(record["points"], existing["points"])
            if (order is None
                    or not np.allclose(record["normal"], existing["normal"], atol=_NORMAL_ATOL, rtol=0)
                    or not np.allclose(record["affine"], existing["affine"][order], atol=_NORMAL_ATOL, rtol=0)):
                raise ValueError("Alternative patches disagree on one native material geometry")
        else:
            unique[name] = record
    if len(unique) != 2:
        raise ValueError("The declared parallel-jaw profile requires exactly two distinct inner-pad materials")
    return unique


def _tool_faces(tool, coordinates):
    state = tool.at(coordinates)
    if len(state["pads"]) != 2:
        raise ValueError("The standalone profile must expose two actual inner pad faces")
    result = []
    for pad in state["pads"]:
        half = pad["half_sizes_xy_m"]
        transform = pad["T_tool_pad"]
        local = np.array([[x * half[0], y * half[1], 0.]
                          for x, y in product((-1., 1.), repeat=2)])
        result.append({"points": local @ transform[:3, :3].T + transform[:3, 3],
                       "normal": transform[:3, 2].copy(), "geom_id": int(pad["geom_id"])})
    return result, float(state["signed_gap_m"])


def validate_tool_binding(tool, contact_rules, finger_position_limits):
    """Require the fixture and standalone model to describe the same two pads.

    Full closed/open/midpoint configurations are checked without clamping the
    native negative closed gap. The coordinate-box vertices additionally vary
    fingers independently: synchronous closure alone cannot reveal erroneous
    affine coupling terms that cancel when the two finger coordinates oppose.
    For the calibrated affine features, these vertices cover the full bounded
    coordinate domain. Pad and corner order are immaterial, but their unique
    geometric correspondence must remain the same across all configurations.

    The caller selects the robot profile explicitly. No joint-prefix matching,
    joint permutation, coordinate sign correction, mesh replacement or pose
    fitting is performed here.
    """
    limits = _array(finger_position_limits, (2, 2), "fixture finger position limits")
    profile_limits = _array(tool.joint_ranges_m, (2, 2), "profile finger position limits")
    if (np.any(limits[:, 0] >= limits[:, 1])
            or not np.allclose(limits, profile_limits, atol=_POSITION_ATOL_M, rtol=0)):
        raise ValueError("Fixture finger coordinate domain differs from the declared robot profile")
    materials = _materials(contact_rules)
    # Export also validates the standalone base axes, rigid TCP binding and
    # exact open/mid boxes used by the model, before any neural weights load.
    geometry = tool.grasp_generation_geometry()
    opened = _array(tool.open_joint_positions_m, (2,), "profile open coordinates")
    closed = _array(tool.closed_joint_positions_m, (2,), "profile closed coordinates")
    probes = [("open", opened), ("mid", (opened + closed) / 2), ("closed", closed)]
    for index, endpoint in enumerate(product(*limits)):
        q = np.asarray(endpoint)
        if not any(np.array_equal(q, previous) for _, previous in probes):
            probes.append((f"independent_vertex_{index}", q))

    assignment = None
    names = sorted(materials)
    maximum_position_error = 0.
    maximum_normal_error = 0.
    checked = []
    for label, coordinates in probes:
        expected, gap = _tool_faces(tool, coordinates)
        observed = [materials[name]["points"] + np.einsum(
            "vcj,j->vc", materials[name]["affine"], coordinates) for name in names]
        compatible = np.zeros((2, 2), dtype=bool)
        errors = {}
        for row, name in enumerate(names):
            for column, face in enumerate(expected):
                order = _corner_order(observed[row], face["points"])
                normal_error = float(np.max(np.abs(materials[name]["normal"] - face["normal"])))
                if order is not None and normal_error <= _NORMAL_ATOL:
                    compatible[row, column] = True
                    errors[row, column] = (
                        float(np.max(np.linalg.norm(observed[row] - face["points"][order], axis=1))),
                        normal_error)
        if not (np.all(compatible.sum(axis=0) == 1) and np.all(compatible.sum(axis=1) == 1)):
            raise ValueError(f"Fixture inner pad geometry differs from the declared robot profile at {label}")
        current = compatible.argmax(axis=1)
        if assignment is None:
            assignment = current
        elif not np.array_equal(current, assignment):
            raise ValueError("Material-to-profile correspondence changes with finger coordinates")
        for row, column in enumerate(current):
            position_error, normal_error = errors[row, column]
            maximum_position_error = max(maximum_position_error, position_error)
            maximum_normal_error = max(maximum_normal_error, normal_error)
        checked.append({"configuration": label, "joint_positions_m": coordinates.tolist(),
                        "native_signed_gap_m": gap})

    return {
        "status": "matched_native_tool_geometry",
        "material_to_profile_pad_index": {name: int(index) for name, index in zip(names, assignment)},
        "profile_joint_names": list(tool.joint_names), "finger_position_limits": limits.tolist(),
        "checked_configurations": checked,
        "maximum_corner_error_m": maximum_position_error,
        "maximum_normal_component_error": maximum_normal_error,
        "tool_geometry_source": geometry["source"],
        "scope": "Known native inner-pad geometry and finger-coordinate domain only",
        "target_contact_verified": False, "holding_verified": False,
    }
# END fixture geometry is checked, never replaced or fitted to the profile.
