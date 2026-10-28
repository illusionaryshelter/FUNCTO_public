"""One reduced cubic problem using the existing differentiable solver API.

Terminal motion equations are eliminated before SciPy sees the variables.
Task, joint and geometry rows all evaluate the expanded original cubic.  The
declared pose priors and the original motion/process costs share that curve
and the already bound physical time.  No second trajectory is reconstructed.
"""

import torch

from manipulation.unified_solver import TrajectoryComputation
from .objective import trajectory_objective_components


class ReducedJointDomain:
    """Pull back the existing complete continuous joint inequalities.

    ``full_domain`` owns the original position/velocity/acceleration/jerk
    bounds.  ``terminal`` defines C = C0 + E Z, with named joint columns.
    The solver receives the original values and the exact Jacobian in Z.
    """

    def __init__(self, full_domain, terminal):
        self.full_domain = full_domain
        self.terminal = terminal
        self.intervals = full_domain.intervals
        self._curve_map = full_domain._curve_map
        self._constraint_bounds = full_domain._constraint_bounds

    def signed_constraints(self, controls, *, with_witnesses=False):
        values = self.full_domain.signed_constraints(self.terminal.expand(controls),
                                                     with_witnesses=with_witnesses)
        return (values[0], self.terminal.pullback(values[1]), *values[2:])


class JointComputation(TrajectoryComputation):
    """Bind the original task calculation to terminal and geometry equations.

    ``base`` is an original, full-coefficient TrajectoryComputation.  Geometry
    keeps that explicit full calculation for arbitrary witness times; this
    class exposes reduced variables to the local solve_trajectory adapter.
    Explicit event roles and preference scales belong to ``base.problem``.
    The same objective function scores this calculation and exported curves.
    """

    def __init__(self, *, base, terminal, geometry):
        super().__init__(problem=base.problem, scene=base.scene, kinematics=base.kinematics,
            domain=base.domain, joint_names=base.joint_names, parameters=base.parameters,
            progress=base.progress.numpy()[0], ordinary_indices=base.ordinary_indices,
            motion_configuration=base.motion_configuration, native_dt=base.native_dt,
            constraint_weight=base.constraint_weight, max_host_rss_bytes=base.max_host_rss_bytes)
        self.full_domain = base.domain
        self.terminal = terminal
        self.geometry = geometry
        self.domain = ReducedJointDomain(base.domain, terminal)
        self.trial_history = []

    def states(self, controls):
        """Evaluate q and its first three derivatives from the same full C."""
        return super().states(self.terminal.expand(controls))

    def task(self, controls, *, solver_rows_out=None):
        """Append fixed necessary geometry rows to the original task rows.

        The original task producer supplies its complete values and masks.
        Witness rows stay fixed during each SLSQP invocation.  A subsequent
        complete entity query may generate more rows for a later invocation,
        charged to the same planning request.  A sampled row is a necessary
        condition, not a replacement for the complete entity calculation.
        """
        equalities, inequalities = super().task(controls, solver_rows_out=solver_rows_out)
        geometry = self.geometry.rows(self.terminal.expand(controls))
        if solver_rows_out is not None:
            solver_rows_out[1]['registered_geometry_witnesses'] = torch.ones_like(geometry, dtype=torch.bool)
        return equalities, torch.cat((inequalities, geometry))

    def objective_components(self, controls, *, include_geometry=True):
        """Compute all preferences once on the same robot states.

        Motion/process and finite geometric direction terms retain the
        maintained formulas. Pose priors use the original functional-frame
        errors, explicit objective scales and exact event occurrences.
        Numerical trials are recorded with their full coefficients so a
        rejected line-search trial is distinguishable from a delivered curve.
        """
        states = self.states(controls)
        # BEGIN one physical objective serves optimization and candidate selection.
        costs = trajectory_objective_components(problem=self.problem, scene=self.scene,
            states=states, robot_state=self.robot_states(*states[:2]), progress=self.progress,
            times=self.times, ordinary_indices=self.ordinary_indices, bounds=self.bounds,
            motion_configuration=self.motion_configuration, native_dt=self.native_dt,
            constraint_weight=self.constraint_weight, include_geometry=include_geometry,
            geometry_report=self.geometry_report)
        # END event membership is supplied by the same explicit problem declaration.
        self.trial_history.append({
            'objective_query_index': len(self.trial_history),
            'controls': self.terminal.expand(controls).detach().cpu().numpy().tolist(),
            'objective_components': {key: float(value.detach()) for key, value in costs.items()},
            'geometry_constraint_rows': self.geometry.row_count,
            'classification': 'numerical_objective_trial_not_qualified',
        })
        return costs
