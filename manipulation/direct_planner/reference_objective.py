"""One physical-time TCP reference term for solving and candidate comparison.

The request declares the two soft error scales and a weight relative to the
existing event prior. BoundToolReference owns source path composition, entry
pose/twist correction and the physical clock. This module freezes its values
on the existing ordinary integration grid, then consumes the same TCP FK
output as the original objective. Original material-event and declared scene
relationship terms retain their maintained FUNCTO computations.
"""

from dataclasses import asdict, dataclass

import numpy as np
import torch

from manipulation.planning_problem import (
    quaternion_conjugate, quaternion_multiply, quaternion_rotvec,
)
from manipulation.preference_planner.objective import (
    trajectory_objective_components as original_objective_components,
)
from manipulation.timed_reference import BoundToolReference
from manipulation.unified_solver import time_mean_weights
# BEGIN the common objective composes only the declared scene relationships.
from .scene_relationships import scene_relationship_objective
# END pair nonpenetration is represented by the fixed geometry constraints.


@dataclass(frozen=True)
class ReferenceObjective:
    """Explicit dimensionless preference data, separate from hard bounds.

    ``position_scale_m`` and ``rotation_scale_rad`` normalize TCP errors.
    ``relative_weight`` multiplies their time mean relative to the existing
    unweighted event sum. The common objective applies ``constraint_weight``
    once to both terms. Zero relative weight is an explicit ablation value;
    no implicit scale or weight is supplied by this component.
    """

    position_scale_m: float
    rotation_scale_rad: float
    relative_weight: float
    # BEGIN motion weights are declared once for the complete trajectory.
    angular_speed_weight_s2: float
    joint_motion_weight: float
    # END pose-event tolerances remain independent hard task data.

    def __post_init__(self):
        values = (self.position_scale_m, self.rotation_scale_rad, self.relative_weight,
                  self.angular_speed_weight_s2, self.joint_motion_weight)
        if (not np.isfinite(values).all() or self.position_scale_m <= 0
                or self.rotation_scale_rad <= 0 or min(values[2:]) < 0):
            raise ValueError('Reference objective needs positive finite scales and a finite nonnegative relative weight')

    def bind(self, bound_reference, physical_times):
        """Freeze one reference on ordinary times in seconds from entry.

        The caller supplies ``computation.times[0, ordinary_indices]``.
        Event anchors stay in the original full task layout. The maintained
        time_mean_weights function owns the increasing-grid requirement and
        normalized trapezoid rule; reference values carry no trial gradient.
        """
        times = torch.as_tensor(physical_times, dtype=torch.float64).detach().clone()
        if times.ndim != 1 or not torch.isfinite(times).all():
            raise ValueError('Reference objective requires one finite physical-time grid')
        weights = time_mean_weights(times.cpu().numpy())
        with torch.no_grad():
            position, quaternion = bound_reference.sample(times)
        return BoundReferenceObjective(self, times, position.detach().clone(),
            quaternion.detach().clone(), position.new_tensor(weights))

    def bind_clock(self, source_reference, *, problem, progress_u, progress_s,
                   duration, physical_times):
        """Bind a supplied candidate clock through the original reference API.

        ``problem`` is the planner's same bound problem: its original phases,
        composed PosePath objects and reference_boundaries remain the source.
        The measured entry pose and twist come from ``source_reference``.
        This constructs the original physical boundary correction once for
        the candidate clock, then freezes the caller's mapped ordinary grid.
        Same-clock candidates reuse their existing BoundReferenceObjective.
        """
        reference = BoundToolReference(problem, progress_u, progress_s, duration,
            source_reference.initial_pose, source_reference.initial_twist)
        return self.bind(reference, physical_times)


@dataclass(frozen=True)
class BoundReferenceObjective:
    """Fixed reference samples and quadrature for one physical query layout.

    Positions use the bound reference frame and metres; quaternions are unit
    wxyz. The ordinary time axis is last before XYZ/wxyz. The planner supplies
    one candidate, optionally with its leading singleton batch axis.
    """

    declaration: ReferenceObjective
    physical_times: torch.Tensor
    position: torch.Tensor
    quaternion: torch.Tensor
    weights: torch.Tensor

    def cost(self, position, quaternion):
        """Return relative_weight times the dimensionless TCP error mean.

        The rotation term is half the squared Frobenius distance of the two
        rotation matrices, expressed as 4*sin(theta/2)**2. It is smooth at a
        half-turn, invariant to quaternion sign and equal to theta**2 to
        second order near the reference. Hard events and reported angular
        errors keep their original principal-angle definitions.
        """
        translation = (position - self.position) / self.declaration.position_scale_m
        relative = quaternion_multiply(quaternion_conjugate(self.quaternion), quaternion)
        # BEGIN the complete reference uses one smooth rotation-matrix metric.
        rotation = (4*relative[..., 1:].square().sum(-1)/relative.square().sum(-1)
                    /self.declaration.rotation_scale_rad**2)
        error = translation.square().sum(-1) + rotation
        # END the normalized quaternion expression has no choice of angle sign.
        return self.declaration.relative_weight * (self.weights * error).sum()

    def report(self, constraint_weight):
        """Describe the actual declared weighting and frozen physical grid."""
        return dict(**asdict(self.declaration), constraint_weight=float(constraint_weight),
            effective_absolute_weight=float(constraint_weight * self.declaration.relative_weight),
            ordinary_sample_count=len(self.physical_times),
            physical_times_s=self.physical_times.cpu().tolist(),
            time_weight_sum=float(self.weights.sum()),
            frame='TCP in the bound reference frame', quaternion_order='wxyz',
            rotation_metric='half_squared_rotation_matrix_frobenius_distance',
            weighting='constraint_weight * relative_weight * normalized_reference_mean')

    # BEGIN actual neighbouring orientations define a tool motion integral.
    def angular_motion_cost(self, quaternion):
        """Integrate squared finite angular speed over the same physical clock.

        Each principal relative rotation is the shortest rotation between
        neighbouring query poses. Dividing its squared angle by the interval
        and total duration gives the time mean of squared angular speed on
        that interval. This term penalizes concentrated turns independently
        of their direction relative to the reference. It does not constrain
        every intermediate orientation to the source path.
        """
        relative = quaternion_multiply(quaternion_conjugate(quaternion[..., :-1, :]),
                                       quaternion[..., 1:, :])
        angle = quaternion_rotvec(relative)
        dt = self.physical_times[1:]-self.physical_times[:-1]
        duration = self.physical_times[-1]-self.physical_times[0]
        return self.declaration.angular_speed_weight_s2 * (angle.square().sum(-1)/dt).sum()/duration
    # END the term differentiates through the original robot pose computation.


# BEGIN numerical and saved-curve scoring use the same five objective terms.
def trajectory_objective_components(*, reference_objective, material_samples, problem, scene,
        states, robot_state, progress, times, ordinary_indices, bounds,
        motion_configuration, native_dt, constraint_weight,
        scene_relation_report=None, producer=None):
    """Compose original task costs, declared scene relations and TCP reference.

    Both numerical controls and saved candidate curves call this function.
    ``robot_state`` is their existing robot_tool_states result; its TCP is
    already transformed by the declared fixed tool offset. Reusing it keeps
    the reference and original material-event terms on the same FK query.
    Exact event queries retain the original prior counting and coordinates.
    ``producer`` is BoundComputation.produce during numerical evaluation and
    records the three existing work boundaries; saved-curve scoring evaluates
    them directly. Its timings are children of the complete objective timing.
    """
    def produce(name, function):
        return function() if producer is None else producer(name, function)

    costs = produce('objective_base', lambda: original_objective_components(
        problem=problem, scene=scene, states=states,
        robot_state=robot_state, progress=progress, times=times,
        ordinary_indices=ordinary_indices, bounds=bounds,
        motion_configuration=motion_configuration, native_dt=native_dt,
        constraint_weight=constraint_weight, include_geometry=False))
    # BEGIN an explicit multiplier controls the retained native joint-state cost.
    costs['motion_quality'] = costs['motion_quality'] * reference_objective.declaration.joint_motion_weight
    # END no iterate-dependent objective scaling participates in this definition.
    q, _, names, _, _, _ = robot_state
    position, quaternion = robot_state[4]
    costs['scene_relations'] = constraint_weight * produce('scene_relations', lambda:
        scene_relationship_objective(scene, position, quaternion, progress, q, names,
            trajectory_time=times, material_samples=material_samples,
            report_out=scene_relation_report))
    costs['reference_prior'] = constraint_weight * produce('reference_prior', lambda:
        reference_objective.cost(position[:, ordinary_indices], quaternion[:, ordinary_indices]))
    costs['tool_angular_motion'] = produce('tool_angular_motion', lambda:
        reference_objective.angular_motion_cost(quaternion[:, ordinary_indices]))
    return costs
# END relation normalization and reference quadrature retain their own clocks.
