"""Bind declared tool-reference boundaries to the actual planning clock.

The nominal paths retain their original composition and mechanism coordinates.
All corrections act at the TCP origin in the reference frame W: p=p_n+b and
R=Exp(phi) R_n. Endpoint derivatives are physical linear/angular velocities,
including the current progress map's holds and changes of slope.
"""
# BEGIN one physical-time reference is shared by IK, costs and saved results.
from copy import deepcopy

import numpy as np
import pinocchio as pin
from scipy.interpolate import CubicHermiteSpline
from scipy.spatial.transform import Rotation

from .motion import _pose_array
from .planning_problem import (
    _interpolate, _tensor, quaternion_conjugate, quaternion_multiply,
)


class BoundToolReference:
    """One explicit clock and the TCP pose/twist measured at its entrance.

    ``initial_twist`` is [v_W, omega_W], in metres/second and radians/second.
    Phase metadata ``reference_boundaries`` declares an initial pose and
    nominal event goals in that phase's [0,1] coordinates. An event's common
    output velocity is the right-side nominal physical velocity; its left
    correction cancels any nominal derivative jump at that declared join.
    Other nominal corners remain part of the supplied reference.

    The source generator only declares geometric boundaries. This constructor
    binds their actual first-arrival times and constructs the corrections once.
    Plain phases without that declaration retain their nominal reference.
    """

    def __init__(self, problem, progress_u, progress_s, duration, initial_pose, initial_twist):
        self.progress_u = np.asarray(progress_u, float).copy()
        self.progress_s = np.asarray(progress_s, float).copy()
        self.duration = float(duration)
        self.initial_pose = _pose_array(initial_pose, "initial tool pose")
        self.initial_twist = np.asarray(initial_twist, float).copy()
        if (self.progress_u.ndim != 1 or len(self.progress_u) < 2
                or self.progress_s.shape != self.progress_u.shape
                or not np.isfinite(self.progress_u).all() or not np.isfinite(self.progress_s).all()
                or np.any(np.diff(self.progress_u) <= 0) or np.any(np.diff(self.progress_s) < 0)
                or self.progress_u[0] != 0 or self.progress_u[-1] != 1
                or not np.isfinite(self.duration) or self.duration <= 0
                or self.initial_twist.shape != (6,) or not np.isfinite(self.initial_twist).all()):
            raise ValueError("A bound tool reference needs a finite physical clock and explicit six-component entrance twist")
        self.nominal_phases = tuple(phase.reference for phase in problem.phases)
        self.boundaries = []
        self.source_boundaries = []
        declared = []
        for index, phase in enumerate(problem.phases):
            declaration = phase.metadata.get("reference_boundaries")
            if declaration is None:
                continue
            self.source_boundaries.append({"phase_index": index, "declaration": deepcopy(declaration)})
            if not declaration.get("source"):
                raise ValueError("Reference boundaries require their declared source")
            if self.progress_s[0] <= index <= self.progress_s[-1]:
                declared.append({"progress": float(index), "pose": declaration["initial_pose"],
                                 "source": declaration["source"], "kind": "phase_entrance"})
            for goal in declaration["goals"]:
                progress = index + float(goal["progress"])
                if not 0 <= float(goal["progress"]) <= 1 or not goal.get("source"):
                    raise ValueError("A reference goal needs phase-local progress and source evidence")
                if self.progress_s[0] <= progress <= self.progress_s[-1]:
                    declared.append({"progress": progress, "pose": goal["pose"],
                                     "source": goal["source"], "kind": "nominal_event"})
        self._starts = np.empty(0)
        self._ends = np.empty(0)
        self._translation = np.empty((0, 4, 3))
        self._rotation = np.empty((0, 4, 3))
        self._restore_suffix = True
        self.diagnostics = {"clock": "seconds from the bound request entrance",
            "position_frame": "TCP origin in reference frame W", "angular_velocity_frame": "W",
            "continuity_scope": "declared joins; other nominal corners retain their supplied derivatives",
            "status": "nominal_reference", "joins": [], "entrance_twist_bound": False,
            "declared_boundaries_in_interval": len(declared)}
        if declared:
            self._bind_boundaries(declared)

    def _first_time(self, progress):
        """Invert only a declared event's first arrival, preserving hold spans."""
        index = int(np.searchsorted(self.progress_s, progress, side="left"))
        if self.progress_s[index] == progress:
            return self.duration*self.progress_u[index]
        fraction = ((progress-self.progress_s[index-1])
                    / (self.progress_s[index]-self.progress_s[index-1]))
        return self.duration*(self.progress_u[index-1]+fraction*(self.progress_u[index]-self.progress_u[index-1]))

    def _nominal(self, times, *, side="right"):
        """Sample original composed paths using the specified one-sided clock."""
        import torch
        progress = _interpolate(self.progress_u*self.duration, self.progress_s, times, side=side)
        position, quaternion = self.nominal_phases[0].sample(progress.clamp(0, 1), side=side)
        for index, path in enumerate(self.nominal_phases[1:], 1):
            p, q = path.sample((progress-index).clamp(0, 1), side=side)
            selected = ((progress >= index) if side == "right" else (progress > index))[..., None]
            position, quaternion = torch.where(selected, p, position), torch.where(selected, q, quaternion)
        return position, quaternion

    def _nominal_boundary(self, time, side):
        """Use analytic Torch derivatives, including rotating TCP lever arms."""
        import torch
        value = torch.tensor([time], dtype=torch.float64)
        # Forward AD differentiates the physical query once. The reverse-over-
        # reverse JVP asks for second derivatives of norms at identity rotations,
        # which the boundary calculation does not mathematically require.
        (position, quaternion), (linear, derivative) = torch.func.jvp(
            lambda t: self._nominal(t, side=side), (value,), (torch.ones_like(value),))
        angular = 2*quaternion_multiply(derivative, quaternion_conjugate(quaternion))[..., 1:]
        pose = np.eye(4)
        pose[:3, 3] = position[0].detach().numpy()
        pose[:3, :3] = Rotation.from_quat(quaternion[0].detach().numpy()[[1, 2, 3, 0]]).as_matrix()
        return pose, np.r_[linear[0].detach().numpy(), angular[0].detach().numpy()]

    @staticmethod
    def _rotation_coordinate(matrix, previous):
        """Choose a continuous correction lift without logging the source path.

        Only the finite boundary correction is represented in exponential
        coordinates. Source mechanism turns and parent/attachment motion are
        evaluated directly by the original PosePath at every query.
        """
        principal = Rotation.from_matrix(matrix).as_rotvec()
        angle = np.linalg.norm(principal)
        if previous is None or angle == 0:
            return principal
        axis = principal/angle
        turns = int(np.rint((np.dot(previous, axis)-angle)/(2*np.pi)))
        candidates = [principal+2*np.pi*k*axis for k in (turns-1, turns, turns+1)]
        return min(candidates, key=lambda candidate: np.linalg.norm(candidate-previous))

    @staticmethod
    def _correction(nominal, nominal_twist, target, target_twist, previous=None):
        rotation = target[:3, :3] @ nominal[:3, :3].T
        phi = BoundToolReference._rotation_coordinate(rotation, previous)
        b = target[:3, 3]-nominal[:3, 3]
        db = target_twist[:3]-nominal_twist[:3]
        # Pinocchio Jexp3 is right-trivialized. Its value at -phi is the
        # left Jacobian: spatial omega_E=J_l(phi) phi_dot in the W frame.
        dphi = np.linalg.solve(pin.Jexp3(-phi), target_twist[3:]-rotation@nominal_twist[3:])
        return b, phi, db, dphi

    def _bind_boundaries(self, declared):
        rows = []
        for declaration in sorted(declared, key=lambda row: row["progress"]):
            row = {**deepcopy(declaration), "time_s": self._first_time(declaration["progress"]),
                   "pose": _pose_array(declaration["pose"], "declared reference boundary")}
            if row["time_s"] == 0.:
                # Current FK owns a new connection's entrance. A source event
                # at this same instant retains its independent task meaning;
                # FK-minus-nominal error cannot reject a continuation request.
                continue
            if rows and rows[-1]["time_s"] == row["time_s"]:
                if not np.allclose(rows[-1]["pose"], row["pose"], atol=1e-9, rtol=0):
                    raise ValueError("Reference boundaries sharing one physical time need one pose")
                continue
            rows.append(row)
        if not rows:
            self.diagnostics["adaptation_scope"] = "no future declared join; original nominal suffix"
            return
        final_nominal, _ = self._nominal_boundary(rows[-1]["time_s"], "right")
        self._restore_suffix = bool(np.allclose(rows[-1]["pose"], final_nominal, atol=1e-9, rtol=0))
        if rows[-1]["time_s"] < self.duration and not self._restore_suffix:
            raise ValueError("The final declared join must restore its nominal suffix pose")
        beginning = {"time_s": 0., "progress": float(self.progress_s[0]), "pose": self.initial_pose,
                     "twist": self.initial_twist, "source": "named robot feedback, calibration and kinematics",
                     "kind": "robot_entrance"}
        for row in rows:
            _, row["twist"] = self._nominal_boundary(row["time_s"], "right")
        boundaries = [beginning, *rows]
        translations, rotations, starts, ends = [], [], [], []
        prior_phi = None
        for left, right in zip(boundaries[:-1], boundaries[1:]):
            left_nominal, left_twist = self._nominal_boundary(left["time_s"], "right")
            right_nominal, right_twist = self._nominal_boundary(right["time_s"], "left")
            a = self._correction(left_nominal, left_twist, left["pose"], left["twist"], prior_phi)
            b = self._correction(right_nominal, right_twist, right["pose"], right["twist"], a[1])
            # A nominal return uses the identity chart at the join. The full
            # source rotation remains untouched, including all observed turns.
            if np.allclose(right["pose"], right_nominal, atol=1e-12, rtol=0):
                b = self._correction(right_nominal, right_twist, right["pose"], right["twist"])
            times = [left["time_s"], right["time_s"]]
            translations.append(CubicHermiteSpline(times, [a[0], b[0]], [a[2], b[2]]).c[:, 0])
            rotations.append(CubicHermiteSpline(times, [a[1], b[1]], [a[3], b[3]]).c[:, 0])
            starts.append(times[0]); ends.append(times[1]); prior_phi = b[1]
            self.diagnostics["joins"].append({"time_s": times[1], "progress": right["progress"],
                "nominal_left_twist": right_twist.tolist(), "output_twist": right["twist"].tolist(),
                "left_translation_derivative": b[2].tolist(), "left_rotation_coordinate_derivative": b[3].tolist(),
                "source": deepcopy(right["source"])})
        self._starts, self._ends = np.asarray(starts), np.asarray(ends)
        self._translation, self._rotation = np.asarray(translations), np.asarray(rotations)
        self.boundaries = [{**row, "pose": row["pose"].tolist(), "twist": row["twist"].tolist()} for row in boundaries]
        self.diagnostics.update(status="physical_time_boundary_correction", final_join_time_s=float(self._ends[-1]),
                                entrance_twist_bound=True)

    def sample(self, times_s):
        """Return W-frame TCP position (...,3) and unit wxyz quaternion (...,4)."""
        import torch
        position, quaternion = self._nominal(times_s)
        if not len(self._starts):
            return position, quaternion
        starts = _tensor(self._starts, times_s)
        index = (torch.searchsorted(starts, times_s.contiguous(), right=True)-1).clamp(0, len(starts)-1)
        delta = (times_s-starts[index])[..., None]
        def polynomial(coefficients):
            values = _tensor(coefficients, times_s)[index]
            return ((values[..., 0, :]*delta+values[..., 1, :])*delta+values[..., 2, :])*delta+values[..., 3, :]
        translation, phi = polynomial(self._translation), polynomial(self._rotation)
        angle = torch.linalg.vector_norm(phi, dim=-1, keepdim=True)
        correction = torch.cat((torch.cos(angle/2), .5*torch.sinc(angle/(2*torch.pi))*phi), -1)
        corrected_position = position+translation
        corrected_quaternion = quaternion_multiply(correction, quaternion)
        active = ((times_s < self._ends[-1]) if self._restore_suffix
                  else (times_s <= self._ends[-1]))[..., None]
        # Exact nominal values own the suffix; no additional interpolation or
        # small residual correction rewrites its parent/attachment geometry.
        return (torch.where(active, corrected_position, position),
                torch.where(active, corrected_quaternion, quaternion))

    def sample_numpy(self, times_s):
        """Return homogeneous TCP poses with the input time-array shape."""
        import torch
        times = torch.as_tensor(np.asarray(times_s), dtype=torch.float64)
        position, quaternion = self.sample(times)
        poses = np.broadcast_to(np.eye(4), (*times.shape, 4, 4)).copy()
        poses[..., :3, 3] = position.detach().numpy()
        poses[..., :3, :3] = Rotation.from_quat(quaternion.detach().numpy().reshape(-1, 4)[:, [1, 2, 3, 0]]).as_matrix().reshape(*times.shape, 3, 3)
        return poses
# END progress remains task semantics; tool derivatives use physical seconds.
