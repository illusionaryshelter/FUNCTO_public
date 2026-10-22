"""Fixed observed-geometry objectives for one native optimization block.

The records contain coordinates in the original entities, not new collision
geometry. Actual pair and contact queries retain all feasibility decisions.
One seed shares its records across every line-search particle and rollout.
"""

from dataclasses import dataclass, field
from contextlib import contextmanager

import numpy as np
import torch
from torch.nn import functional as torch_functional

from .planning_problem import quaternion_rotate


def _point_world(point, pose):
    position, quaternion = pose
    local = torch.as_tensor(point, dtype=position.dtype, device=position.device)
    return position + quaternion_rotate(quaternion, local)


@dataclass(frozen=True)
class LocalPairModel:
    """A material point and an observed zero-section tangent in local frames.

    ``surface_on_second`` chooses which pair entity owns the zero point and
    normal; the other entity owns ``point``. All coordinates and the positive
    smoothing length are metres, except the unit normal. The normal rotates
    with its entity in every trial, including its rotation derivative.
    """

    point: tuple
    surface_point: tuple
    surface_normal: tuple
    surface_on_second: bool
    length_scale_m: float

    def value(self, first_pose, second_pose):
        point_pose, surface_pose = ((first_pose, second_pose) if self.surface_on_second
                                    else (second_pose, first_pose))
        normal = torch.as_tensor(self.surface_normal, dtype=surface_pose[0].dtype,
                                 device=surface_pose[0].device)
        normal_world = quaternion_rotate(surface_pose[1], normal)
        signed = ((_point_world(self.point, point_pose)
                   - _point_world(self.surface_point, surface_pose)) * normal_world).sum(-1)
        return self.length_scale_m * torch_functional.softplus(-signed / self.length_scale_m)


@dataclass(frozen=True)
class LocalContactModel:
    """One fixed observed zero point in the target entity's coordinates."""

    point: tuple

    def value(self, normals, bounds):
        """Return the norm of the declared region's half-space excesses.

        Both region coefficients and the saved point are in the target frame.
        This is a continuous optimization objective, not contact evidence or
        an exact Euclidean distance to the intersection of half-spaces.
        """
        point = torch.as_tensor(self.point, dtype=normals.dtype, device=normals.device)
        excess = torch.relu((normals * point).sum(-1) - bounds)
        return torch.linalg.vector_norm(excess, dim=-1)


@dataclass
class _StateModels:
    pairs: dict = field(default_factory=dict)
    contacts: dict = field(default_factory=dict)


class LocalGeometryModels:
    """Capture and freeze local objectives on actual optimization base states.

    The caller begins capture, evaluates the repeated base actions without
    consuming their cost, then finishes capture before evaluating cost and
    gradient. A state key uses seed identity, physical time, relation progress
    and permission progress. It never depends on candidate joint values.
    Recording empty states distinguishes a valid empty model from a rollout
    whose time or relation layout was never captured.
    """

    def __init__(self):
        self.num_seeds = 0
        self.num_particles = 0
        self.capturing = False
        self._states = {}
        self._pending = None
        self._previous_layout = None
        # BEGIN only an explicitly repeated base may share original queries.
        self.repeated_base = False
        # END ordinary line-search particles retain separate hard evidence.

    # BEGIN the solver owns this scope because it constructs repeated actions.
    @contextmanager
    def repeated_base_evaluation(self):
        """Declare identical particle inputs while keeping independent gradients.

        The scene reuses numeric query records within one call only. It still
        forms each row's scalar from that row's differentiable entity poses.
        A trial evaluation never enters this scope merely because it has the
        same physical-time layout as a base evaluation.
        """
        previous = self.repeated_base
        self.repeated_base = True
        try:
            yield
        finally:
            self.repeated_base = previous
    # END exceptions cannot leave a later trial in the repeated-base scope.

    def begin_capture(self, num_seeds, num_particles):
        """Start replacement from one actual base trajectory per seed."""
        if self.capturing:
            raise RuntimeError("Finish the current local geometry capture before starting another")
        if num_seeds < 1 or num_particles < 1:
            raise ValueError("Local geometry capture requires positive seed and particle counts")
        self._previous_layout = self.num_seeds, self.num_particles
        self.num_seeds, self.num_particles = int(num_seeds), int(num_particles)
        self._pending = {}
        self.capturing = True

    def state_key(self, row, time_s, progress, permission_progress):
        """Map repeated particle rows onto the same semantic seed state."""
        if not self.num_particles:
            raise RuntimeError("Local geometry models require an explicit base capture")
        seed = int(row) // self.num_particles
        if seed >= self.num_seeds:
            raise ValueError("The optimizer batch does not match the captured seed layout")
        return seed, float(time_s), float(progress), float(permission_progress)

    def state(self, key):
        """Read one frozen state, or create its replacement during capture."""
        if self.capturing:
            return self._pending.setdefault(key, _StateModels())
        try:
            return self._states[key]
        except KeyError as error:
            raise RuntimeError("The optimizer queried a time/relation state absent from its base capture") from error

    def captures_row(self, row):
        """Repeated copies of a base seed need only one local direction solve."""
        return self.capturing and int(row) % self.num_particles == 0

    def finish_capture(self):
        """Freeze the replacement and report exactly which seeds changed."""
        if not self.capturing:
            raise RuntimeError("No local geometry capture is active")
        changed = np.zeros(self.num_seeds, dtype=bool)
        for key in self._states.keys() | self._pending.keys():
            seed = key[0]
            if seed < self.num_seeds and self._states.get(key) != self._pending.get(key):
                changed[seed] = True
        self._states = self._pending
        self._pending = None
        self._previous_layout = None
        self.capturing = False
        return changed

    def abort_capture(self):
        """Discard an interrupted replacement and restore its previous layout.

        The solver exits on the original interruption; this method only closes
        the capture lifecycle and does not evaluate an earlier objective.
        """
        if self.capturing:
            self.num_seeds, self.num_particles = self._previous_layout
            self._previous_layout = None
            self._pending = None
            self.capturing = False
