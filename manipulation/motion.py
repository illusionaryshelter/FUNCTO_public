"""Rigid-object motion descriptions and demonstration transfer.

All matrices map one fixed material/body frame into the supplied reference
frame.  They describe objects or moving parts, never an inferred hand pose.
The four constrained models are small geometric families; a good fit does not
establish that a physical mechanism enforces the corresponding constraint.

Coordinates use metres and radians.  No timing, acceleration, collision, or
robot-execution model is introduced here.  Angular sequences are lifted to a
continuous branch, which requires consecutive observed rotations below pi.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


_DIMENSIONS = {
    "free": 6,
    "planar": 3,
    "translation_plane": 2,
    "revolute": 1,
    "prismatic": 1,
}
_CANDIDATES = ("prismatic", "revolute", "translation_plane", "planar", "free")
_ENDPOINT_TOLERANCE = 1e-8


def _poses_array(value: Any, name: str = "poses") -> np.ndarray:
    """Validate SE(3) without projecting reflections or invalid rotations."""
    poses = np.array(value, dtype=float, copy=True)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or len(poses) == 0:
        raise ValueError(f"{name} must have nonempty shape (N, 4, 4)")
    if not np.isfinite(poses).all():
        raise ValueError(f"{name} contains non-finite values")
    if not np.allclose(poses[:, 3], [0, 0, 0, 1], atol=1e-9, rtol=0):
        raise ValueError(f"{name} has an invalid homogeneous last row")
    rotations = poses[:, :3, :3]
    products = np.swapaxes(rotations, 1, 2) @ rotations
    if not np.allclose(products, np.eye(3), atol=1e-6, rtol=0):
        raise ValueError(f"{name} rotations are not orthonormal")
    if not np.allclose(np.linalg.det(rotations), 1, atol=1e-6, rtol=0):
        raise ValueError(f"{name} rotations must be proper (no reflections)")
    return poses


def _pose_array(value: Any, name: str) -> np.ndarray:
    pose = np.asarray(value, dtype=float)
    if pose.shape != (4, 4):
        raise ValueError(f"{name} must have shape (4, 4)")
    return _poses_array(pose[None], name)[0]


def _tolerances(position: float, rotation: float) -> tuple[float, float]:
    position, rotation = float(position), float(rotation)
    if not np.isfinite(position) or position <= 0:
        raise ValueError("position_tolerance must be finite and positive")
    if not np.isfinite(rotation) or not 0 < rotation < np.pi:
        raise ValueError("rotation_tolerance must lie strictly between 0 and pi")
    return position, rotation


def _inverse(pose: np.ndarray) -> np.ndarray:
    result = np.eye(4)
    result[:3, :3] = pose[:3, :3].T
    result[:3, 3] = -result[:3, :3] @ pose[:3, 3]
    return result


def _errors(observed: np.ndarray, predicted: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    positions = np.linalg.norm(observed[:, :3, 3] - predicted[:, :3, 3], axis=1)
    relative = observed[:, :3, :3] @ np.swapaxes(predicted[:, :3, :3], 1, 2)
    angles = Rotation.from_matrix(relative).magnitude()
    return positions, angles


def _basis(axis: np.ndarray, reference_rotation: np.ndarray) -> np.ndarray:
    """Choose a reproducible right-handed basis using the body-frame gauge.

    Axis sign and in-plane basis are coordinate conventions, not estimates of
    opening direction or a semantic zero.  Using the reference orientation
    makes the convention equivariant under a common world-frame transform.
    """
    axis = np.asarray(axis, dtype=float)
    norm = np.linalg.norm(axis)
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError("Cannot define a basis from a zero axis")
    axis = axis / norm
    dots = reference_rotation.T @ axis
    strongest = np.flatnonzero(np.abs(dots) >= np.max(np.abs(dots)) - 1e-12)[0]
    if dots[strongest] < 0:
        axis = -axis
    projected = reference_rotation - axis[:, None] * (axis @ reference_rotation)[None]
    lengths = np.linalg.norm(projected, axis=0)
    index = np.flatnonzero(lengths >= lengths.max() - 1e-12)[0]
    first = projected[:, index] / lengths[index]
    return np.column_stack((first, np.cross(axis, first), axis))


def _position_svd(positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centered = positions - positions.mean(axis=0)
    # Economy SVD avoids an N-by-N allocation for long demonstrations.  Plane
    # identification is rejected below when fewer than two directions exist.
    _, singular, vectors = np.linalg.svd(centered, full_matrices=False)
    singular = np.pad(singular, (0, max(0, 3 - len(singular))))
    return singular[:3] / np.sqrt(len(positions)), vectors


def _angles(relative_rotations: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """Closest rotation about axis, then a continuous observed angle lift.

    atan2 uses the trace-maximizing projection onto the axis subgroup.  Its
    residual is checked separately; projection does not establish compatibility.
    A half-turn between samples has two equally plausible motion directions.
    """
    skew_vectors = np.column_stack(
        (
            relative_rotations[:, 2, 1] - relative_rotations[:, 1, 2],
            relative_rotations[:, 0, 2] - relative_rotations[:, 2, 0],
            relative_rotations[:, 1, 0] - relative_rotations[:, 0, 1],
        )
    )
    sine = 0.5 * (skew_vectors @ axis)
    cosine = 0.5 * (
        np.trace(relative_rotations, axis1=1, axis2=2)
        - np.einsum("i,nij,j->n", axis, relative_rotations, axis)
    )
    principal = np.arctan2(sine, cosine)
    increments = np.angle(np.exp(1j * np.diff(principal)))
    if np.any(np.abs(increments) >= np.pi - 1e-8):
        raise ValueError("Angular branch is ambiguous: a sampled step is a half-turn")
    return np.unwrap(principal)


@dataclass
class MotionModel:
    """One identifiable motion family fitted to a fixed-body pose sequence.

    ``coordinates`` contains the fitted reference sequence, or the returned
    sequence after ``transfer_motion``.  For free motion its columns are world
    position and principal rotation vector, not an angular interpolation rule.
    For planar motion columns are x/y in ``basis[:, :2]`` and lifted yaw.
    Translation-plane columns are x/y; revolute/prismatic use one angle/distance.
    ``axis`` is ``basis[:, 2]``.  A revolute axis passes through ``axis_point``.
    """

    kind: str
    coordinates: np.ndarray
    reference_pose: np.ndarray
    basis: np.ndarray
    axis_point: np.ndarray | None
    position_tolerance: float
    rotation_tolerance: float
    diagnostics: dict[str, Any]

    @property
    def dimension(self) -> int:
        return _DIMENSIONS[self.kind]

    @property
    def axis(self) -> np.ndarray:
        return self.basis[:, 2].copy()

    def poses(self, coordinates: Any) -> np.ndarray:
        """Evaluate FK at coordinates; interpolation, when used, belongs here.

        No world-position chord is substituted for a revolute arc.  A single
        coordinate vector is accepted and still returns shape (1, 4, 4).
        """
        values = np.asarray(coordinates, dtype=float)
        if values.ndim == 1:
            values = values[None]
        if values.ndim != 2 or values.shape[1] != self.dimension or len(values) == 0:
            raise ValueError(f"{self.kind} coordinates must have shape (N, {self.dimension})")
        if not np.isfinite(values).all():
            raise ValueError("coordinates contain non-finite values")
        output = np.repeat(np.eye(4)[None], len(values), axis=0)
        position = self.reference_pose[:3, 3]
        orientation = self.reference_pose[:3, :3]
        if self.kind == "free":
            output[:, :3, 3] = values[:, :3]
            output[:, :3, :3] = Rotation.from_rotvec(values[:, 3:]).as_matrix()
        elif self.kind in ("planar", "translation_plane"):
            output[:, :3, 3] = position + values[:, :2] @ self.basis[:, :2].T
            if self.kind == "planar":
                rotations = Rotation.from_rotvec(values[:, 2, None] * self.axis).as_matrix()
                output[:, :3, :3] = rotations @ orientation
            else:
                output[:, :3, :3] = orientation
        elif self.kind == "prismatic":
            output[:, :3, 3] = position + values * self.axis
            output[:, :3, :3] = orientation
        elif self.kind == "revolute":
            rotations = Rotation.from_rotvec(values * self.axis).as_matrix()
            output[:, :3, :3] = rotations @ orientation
            output[:, :3, 3] = self.axis_point + np.einsum(
                "nij,j->ni", rotations, position - self.axis_point
            )
        else:
            raise ValueError(f"Unknown motion kind: {self.kind}")
        return output

    def _coordinates_unchecked(self, poses: np.ndarray) -> np.ndarray:
        positions = poses[:, :3, 3] - self.reference_pose[:3, 3]
        if self.kind == "free":
            return np.column_stack(
                (poses[:, :3, 3], Rotation.from_matrix(poses[:, :3, :3]).as_rotvec())
            )
        if self.kind == "prismatic":
            return (positions @ self.axis)[:, None]
        if self.kind == "translation_plane":
            return positions @ self.basis[:, :2]
        relative = poses[:, :3, :3] @ self.reference_pose[:3, :3].T
        angle = _angles(relative, self.axis)
        if self.kind == "revolute":
            return angle[:, None]
        return np.column_stack((positions @ self.basis[:, :2], angle))

    def coordinates_of(self, poses: Any) -> np.ndarray:
        """Return coordinates only when full poses match this family.

        Compatibility uses the declared fitting tolerances.  Returned angles
        are unwrapped in sequence order; a single pose has a principal branch.
        ``transfer_motion`` separately checks exact endpoint reachability and
        chooses the goal branch nearest the reference terminal coordinate.
        """
        observed = _poses_array(poses)
        coordinates = self._coordinates_unchecked(observed)
        position_error, rotation_error = _errors(observed, self.poses(coordinates))
        if position_error.max() > self.position_tolerance + 1e-10:
            raise ValueError(
                f"Poses incompatible with {self.kind}: position residual "
                f"{position_error.max():.6g} m exceeds {self.position_tolerance:.6g}"
            )
        if rotation_error.max() > self.rotation_tolerance + 1e-10:
            raise ValueError(
                f"Poses incompatible with {self.kind}: rotation residual "
                f"{rotation_error.max():.6g} rad exceeds {self.rotation_tolerance:.6g}"
            )
        return coordinates

    def to_dict(self) -> dict[str, Any]:
        """JSON-compatible model parameters, coordinates, and evidence."""
        return {
            "kind": self.kind,
            "dimension": self.dimension,
            "reference_pose": self.reference_pose.tolist(),
            "basis": self.basis.tolist(),
            "axis_point": None if self.axis_point is None else self.axis_point.tolist(),
            "coordinates": self.coordinates.tolist(),
            "position_tolerance": self.position_tolerance,
            "rotation_tolerance": self.rotation_tolerance,
            "diagnostics": self.diagnostics,
            "pose_semantics": "fixed object/part frame in reference frame; not end effector",
        }


def _fit_one(poses: np.ndarray, kind: str, ptol: float, rtol: float) -> MotionModel:
    reference = poses[0].copy()
    positions = poses[:, :3, 3]
    relative = poses[:, :3, :3] @ reference[:3, :3].T
    rotation_excitation = float(Rotation.from_matrix(relative).magnitude().max())
    translation_excitation = float(np.linalg.norm(positions - positions[0], axis=1).max())
    spread, vectors = _position_svd(positions)
    basis = np.eye(3)
    axis_point = None
    diagnostics: dict[str, Any] = {
        "position_spread_rms_m": spread.tolist(),
        "translation_excitation_m": translation_excitation,
        "rotation_excitation_rad": rotation_excitation,
        "interpretation": "compatible motion description, not identified physical freedom",
        "coordinates_zero": "initial object pose; not a semantic open/closed state",
    }

    if kind == "prismatic":
        if translation_excitation <= ptol:
            raise ValueError("Insufficient translation to identify a prismatic direction")
        basis = _basis(vectors[0], reference[:3, :3])
    elif kind in ("translation_plane", "planar"):
        if kind == "planar" and rotation_excitation > rtol:
            rotation_constraints = np.concatenate(relative - np.eye(3), axis=0)
            _, singular, right = np.linalg.svd(rotation_constraints, full_matrices=False)
            normal = right[-1]
            diagnostics["rotation_axis_singular_values"] = singular.tolist()
        else:
            if spread[1] <= ptol:
                raise ValueError("Insufficient two-dimensional motion to identify a plane")
            normal = vectors[-1]
        basis = _basis(normal, reference[:3, :3])
    elif kind == "revolute":
        if rotation_excitation <= rtol:
            raise ValueError("Insufficient rotation to identify a fixed rotation axis")
        constraints = np.concatenate(np.eye(3) - relative, axis=0)
        _, singular, right = np.linalg.svd(constraints, full_matrices=False)
        basis = _basis(right[-1], reference[:3, :3])
        # Solve for the axis point relative to p0 in its perpendicular plane.
        # This fixes the unobservable shift along the axis and is equivariant
        # under rigid changes of world coordinates.
        system = constraints @ basis[:, :2]
        offsets, _, rank, axis_singular = np.linalg.lstsq(
            system, (positions - positions[0]).reshape(-1), rcond=None
        )
        if rank != 2 or axis_singular[-1] < 1e-10:
            raise ValueError("Insufficient rotation to locate the rotation axis")
        axis_point = positions[0] + basis[:, :2] @ offsets
        diagnostics["rotation_axis_singular_values"] = singular.tolist()
        diagnostics["axis_location_singular_values"] = axis_singular.tolist()
        diagnostics["axis_location_noise_gain"] = float(1 / axis_singular[-1])

    model = MotionModel(kind, np.empty((0, _DIMENSIONS[kind])), reference,
                        basis, axis_point, ptol, rtol, diagnostics)
    model.coordinates = model.coordinates_of(poses)
    position_error, rotation_error = _errors(poses, model.poses(model.coordinates))
    diagnostics.update({
        "position_residuals_m": position_error.tolist(),
        "rotation_residuals_rad": rotation_error.tolist(),
        "max_position_residual_m": float(position_error.max()),
        "max_rotation_residual_rad": float(rotation_error.max()),
    })
    if kind in ("planar", "revolute"):
        diagnostics["angular_branch"] = "continuous observed lift; adjacent steps below pi"
    return model


# BEGIN observed structure: report compatible geometry independently of task constraints.
def _candidate_models(observed: np.ndarray, ptol: float, rtol: float):
    """Fit the existing families without interpreting them as physical joints."""
    accepted, rejected = [], {}
    for candidate in _CANDIDATES:
        try:
            accepted.append(_fit_one(observed, candidate, ptol, rtol))
        except ValueError as error:
            rejected[candidate] = str(error)
    # Input SE(3) validity is checked before this helper. The free family must
    # therefore succeed; it is a complete representation, not an error fallback.
    free = next(model for model in accepted if model.kind == "free")
    excited = (free.diagnostics["translation_excitation_m"] > ptol
               or free.diagnostics["rotation_excitation_rad"] > rtol)
    return accepted, rejected, excited


def analyze_motion(
    poses: Any,
    position_tolerance: float = 0.005,
    rotation_tolerance: float = 0.035,
) -> dict[str, Any]:
    """Describe observed motion without selecting an applied task constraint.

    Every candidate is checked against the complete SE(3) sequence, including
    orientation. A line may be compatible with a prismatic description while
    remaining an unconstrained carry task. Similarly, a planar observation
    alone cannot establish contact or a physical planar joint. Accepted models
    retain all fitted coordinates and per-frame residuals; no terminal-only
    compression replaces the observed motion. Near-static observations remain
    valid poses but report that their motion structure is not identifiable.

    Invalid poses raise before candidate fitting. In particular, ``free`` does
    not repair invalid rotations, missing observations or a lost track.
    """
    observed = _poses_array(poses)
    ptol, rtol = _tolerances(position_tolerance, rotation_tolerance)
    accepted, rejected, excited = _candidate_models(observed, ptol, rtol)
    return {
        "status": "observed_compatible" if excited else "insufficient_excitation",
        "preferred_description": accepted[0].kind if excited else None,
        "compatible_candidates": [model.kind for model in accepted],
        "candidates": {model.kind: model.to_dict() for model in accepted},
        "rejected_candidates": rejected,
        "position_tolerance_m": ptol, "rotation_tolerance_rad": rtol,
        "interpretation": "observed geometry only; task and reference evidence determine applied constraints",
    }
# END observed structure.


def fit_motion(
    poses: Any,
    kind: str = "auto",
    position_tolerance: float = 0.005,
    rotation_tolerance: float = 0.035,
) -> MotionModel:
    """Fit one of five rigid motion descriptions using complete poses.

    ``auto`` selects the lowest-dimensional identifiable compatible family and
    reports all tested alternatives.  Free SE(3) containment is not evidence
    against a constrained description, nor is a constrained fit proof of a
    physical joint.  Near-static input cannot identify a family in auto mode.
    Tolerances are explicit acceptance scales, not estimated sensor precision.
    """
    observed = _poses_array(poses)
    ptol, rtol = _tolerances(position_tolerance, rotation_tolerance)
    if kind not in (*_DIMENSIONS, "auto"):
        raise ValueError(f"Unknown motion kind: {kind}")
    if kind != "auto":
        return _fit_one(observed, kind, ptol, rtol)

    # BEGIN share the same candidate fits with non-constraining observation analysis.
    accepted, rejected, excited = _candidate_models(observed, ptol, rtol)
    if not excited:
        raise ValueError("Static or insufficiently excited observations cannot identify a motion mode")
    # END shared candidate fits.
    selected = accepted[0]  # Valid input always admits the free family.
    selected.diagnostics["selection"] = {
        "requested": "auto",
        "selected": selected.kind,
        "compatible_candidates": [model.kind for model in accepted],
        "rejected_candidates": rejected,
        "ambiguous": len(accepted) > 1,
        "rule": "smallest identifiable compatible description; listed order breaks dimension ties",
    }
    return selected


def transfer_motion(
    demo_poses: Any,
    initial_pose: Any,
    reference_transform: Any = np.eye(4),
    kind: str = "free",
    goal_pose: Any | None = None,
    position_tolerance: float = 0.005,
    rotation_tolerance: float = 0.035,
) -> dict[str, Any]:
    """Transfer object motion, preserving the initial pose and reachable goal.

    A maps demonstration-reference coordinates to test-reference coordinates:
    ``T[k] = A @ Td[k] @ inv(Td[0]) @ inv(A) @ initial_pose``.
    The supplied mapping is an input assumption; it is not evidence that an
    unseen mechanism shares the demonstration axis or shape.

    A terminal correction is distributed uniformly over the sample phase in
    model coordinates.  For free motion, translation uses an additive world
    correction and orientation uses a left-multiplied Slerp correction.  For
    constrained motion no incompatible terminal pose is silently projected.
    No physical timestamps or end-effector control are inferred.
    """
    demo = _poses_array(demo_poses, "demo_poses")
    initial = _pose_array(initial_pose, "initial_pose")
    transform = _pose_array(reference_transform, "reference_transform")
    goal = None if goal_pose is None else _pose_array(goal_pose, "goal_pose")
    mapped = transform @ demo @ _inverse(demo[0]) @ _inverse(transform) @ initial
    mapped[0] = initial
    model = fit_motion(mapped, kind, position_tolerance, rotation_tolerance)
    coordinates = model.coordinates.copy()
    diagnostics: dict[str, Any] = {
        "reference_transform": transform.tolist(),
        "goal_supplied": goal is not None,
        "phase": "uniform sample index; no physical timing inferred",
        "pose_semantics": "object/part pose, not end-effector trajectory",
    }

    if goal is not None:
        if len(demo) < 2:
            p_error, r_error = _errors(initial[None], goal[None])
            if max(p_error[0], r_error[0]) > _ENDPOINT_TOLERANCE:
                raise ValueError("A one-node trajectory cannot preserve different initial and goal poses")
        phase = np.linspace(0, 1, len(demo))
        if model.kind == "free":
            output = model.poses(coordinates)
            output[:, :3, 3] += phase[:, None] * (goal[:3, 3] - output[-1, :3, 3])
            correction = goal[:3, :3] @ output[-1, :3, :3].T
            rotations = Slerp([0, 1], Rotation.from_matrix(np.stack((np.eye(3), correction))))(phase)
            output[:, :3, :3] = rotations.as_matrix() @ output[:, :3, :3]
            coordinates = model.coordinates_of(output)
        else:
            goal_coordinates = model.coordinates_of(goal[None])[0]
            angular_column = 0 if model.kind == "revolute" else 2 if model.kind == "planar" else None
            if angular_column is not None:
                turns = np.round((coordinates[-1, angular_column] - goal_coordinates[angular_column]) / (2 * np.pi))
                goal_coordinates[angular_column] += 2 * np.pi * turns
                diagnostics["goal_angle_branch"] = "nearest reference terminal angle"
            projected_goal = model.poses(goal_coordinates)
            p_error, r_error = _errors(goal[None], projected_goal)
            if p_error[0] > _ENDPOINT_TOLERANCE or r_error[0] > _ENDPOINT_TOLERANCE:
                raise ValueError("goal_pose is not exactly reachable in the fitted motion model")
            coordinates += phase[:, None] * (goal_coordinates - coordinates[-1])
            output = model.poses(coordinates)
    else:
        output = model.poses(coordinates)

    initial_error = _errors(initial[None], output[:1])
    if initial_error[0][0] > _ENDPOINT_TOLERANCE or initial_error[1][0] > _ENDPOINT_TOLERANCE:
        raise ValueError("Motion transfer did not preserve the initial pose")
    output[0] = initial
    if goal is not None:
        endpoint_error = _errors(goal[None], output[-1:])
        if endpoint_error[0][0] > _ENDPOINT_TOLERANCE or endpoint_error[1][0] > _ENDPOINT_TOLERANCE:
            raise ValueError("Motion transfer did not preserve the goal pose")
        output[-1] = goal
    model.coordinates = coordinates
    diagnostics["model_fit"] = model.diagnostics
    return {"poses": output, "model": model, "diagnostics": diagnostics}
