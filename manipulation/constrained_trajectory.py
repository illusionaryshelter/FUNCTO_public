"""Explicit SLSQP constraints on the exact joint curve delivered to execution.

cuRobo supplies FK and its derivatives. SciPy updates the free cubic controls;
neither collision clearance nor joint bounds are finite objective penalties.
This module owns one evaluation cache, not a history of replacement candidates.
"""

# BEGIN explicit physical-constraint trajectory solve.
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize, OptimizeResult

from .physical_curve import PHYSICAL_FTOL, aggregate_violation


@dataclass
class CurveEvaluation:
    actions: np.ndarray
    objective: float
    raw_objective: float
    objective_gradient: np.ndarray
    slack: np.ndarray
    jacobian: np.ndarray
    names: tuple
    fixed_slack: np.ndarray
    fixed_names: tuple
    physical: object
    scene_report: dict

    @property
    def violation(self):
        return aggregate_violation(self.slack)+aggregate_violation(self.fixed_slack)

    @property
    def accepted(self):
        return bool(np.isfinite(self.objective) and self.violation <= PHYSICAL_FTOL)


class ConstrainedTrajectory:
    """One fixed clock, objective and constraint layout for a planning request.

    Each geometry group has a fixed-state minimum and a variable-state minimum.
    Fixed rows have zero Jacobians; both kinds enter SLSQP's same vector.
    Taking a minimum preserves that group's
    feasible set without a dense Jacobian row for every sampled state. Raw
    per-state signed distances and validity masks remain in the report.
    """

    def __init__(self, physical, *, problem, scene, progress_map, anchors, budget,
                 fk, joint_state_type, device, smooth_config, dynamics=None,
                 effort_bounds=None):
        import torch
        from .planning import _orientation_vjp_type, _native_gradient_layout_type

        self.torch, self.physical = torch, physical
        self.problem, self.scene, self.budget = problem, scene, budget
        # The old native objective multiplies nominal costs by this fixed
        # request weight. SLSQP's initial quadratic model is scale sensitive;
        # express the WHOLE objective in that existing cost unit. Relative
        # preferences, constraints and their numerical acceptance do not move.
        self.objective_scale = budget.constraint_weight
        self.fk, self.JointState, self.device = fk, joint_state_type, device
        self.orientation_vjp = _orientation_vjp_type()
        self.native_layout = _native_gradient_layout_type()
        self.dynamics, self.effort_bounds = dynamics, effort_bounds
        self.check_request = lambda: None
        self.last = None
        self.calls = self.states = 0
        regular = np.arange((budget.knots+4)*budget.interpolation_steps+1)*float(np.float32(budget.maximum_trajectory_dt))
        pairs = progress_map.anchor_parameters(anchors)
        anchor_times = pairs[:, 0]*physical.duration
        self.query_times = np.unique(np.r_[physical.times, regular, anchor_times])
        objective_times = np.r_[regular, anchor_times]
        objective_progress = np.r_[np.interp(regular/physical.duration, progress_map.u, progress_map.s), pairs[:, 1]]
        order = np.argsort(objective_times, kind="stable")
        self.objective_indices = np.searchsorted(self.query_times, objective_times[order])
        self.regular_indices = np.searchsorted(self.query_times, regular)
        scene_times = np.unique(np.r_[physical.times, anchor_times])
        self.scene_indices = np.searchsorted(self.query_times, scene_times)
        scene_progress = np.interp(scene_times/physical.duration, progress_map.u, progress_map.s)
        scene_progress[np.searchsorted(scene_times, anchor_times)] = pairs[:, 1]
        self.objective_times = self.tensor(objective_times[order])[None]
        self.objective_progress = self.tensor(objective_progress[order])[None]
        self.scene_times, self.scene_progress = self.tensor(scene_times)[None], self.tensor(scene_progress)[None]
        self.weights = [self.tensor(physical.sample_map(self.query_times, derivative)[0]) for derivative in range(4)]
        variable = np.any(physical.sample_map(scene_times, 0)[0] != 0., axis=1)
        self.variable_mask = torch.as_tensor(variable, device=device)[None, :, None].expand(1, -1, physical.dof)
        self.smooth_weights = np.asarray(smooth_config["squared_l2_regularization_weight"], dtype=float).copy()
        self.dt = float(np.float32(budget.maximum_trajectory_dt))
        if smooth_config["retime_regularization_weights"]:
            self.smooth_weights *= [self.dt, self.dt**2, self.dt**3, 1., self.dt]
        if dynamics is not None:
            dynamics.setup_batch_size(1, len(self.query_times))

    def tensor(self, value):
        return self.torch.as_tensor(value, dtype=self.torch.float64, device=self.device)

    def evaluate(self, actions):
        """Evaluate the actual Hermite64 values and their affine derivatives.

        Native FK requires float32 inputs; Coal and joint constraints receive
        the actual float64 curve. The zero-valued linear term attaches the
        analytic derivative without replacing reconstructed Hermite values by
        a separately rounded affine reconstruction.
        """
        from .planning import _nominal_task_cost

        actions = np.asarray(actions, dtype=np.float64).reshape(self.physical.controls, self.physical.dof)
        if self.last is not None and np.array_equal(actions, self.last.actions):
            return self.last
        self.check_request()
        if not np.isfinite(actions).all():
            raise ValueError("SLSQP produced nonfinite cubic controls")
        torch = self.torch
        x = self.tensor(actions).requires_grad_(True)
        states = [(self.tensor(self.physical.samples(actions, self.query_times, derivative))
                   + weight@(x-x.detach()))[None] for derivative, weight in enumerate(self.weights)]
        q, dq, acceleration, jerk = states
        native_q = q.to(torch.float32).contiguous()
        joint = self.JointState.from_position(native_q, joint_names=list(self.physical.joint_names))
        state = self.fk.compute_kinematics(joint)
        tool_index = state.tool_poses.tool_frames.index(self.problem.tool_link)
        link_q = self.orientation_vjp.apply(native_q, state.tool_poses.quaternion.detach(),
            state.tool_jacobians[..., 3:, :].detach())[:, :, tool_index].to(torch.float64)
        link_p = self.native_layout.apply(state.tool_poses.position)[:, :, tool_index].to(torch.float64)
        position, quaternion = self.problem.transform_tool_pose(link_p, link_q)
        spheres = self.problem.transform_robot_spheres(self.native_layout.apply(state.robot_spheres).to(torch.float64))
        linear = angular = None
        if any(direction.kind == "local_rate" for mechanism in self.problem.mechanisms for direction in mechanism.directions):
            jacobian = self.native_layout.apply(state.tool_jacobians)[:, :, tool_index].to(torch.float64)
            twist = torch.einsum("...kd,...d->...k", jacobian, dq)
            linear, angular = self.problem.transform_tool_velocity(twist[..., :3], twist[..., 3:], link_q)
        idx = self.objective_indices
        nominal = _nominal_task_cost(self.problem, position[:, idx], quaternion[:, idx], self.objective_progress,
            q[:, idx], self.physical.joint_names, dq[:, idx], trajectory_time=self.objective_times,
            tool_linear_velocity=None if linear is None else linear[:, idx],
            tool_angular_velocity=None if angular is None else angular[:, idx])
        nominal = nominal+self.scene.optimizer_preferences(self.objective_progress, q[:, idx], self.physical.joint_names)
        objective = self.budget.constraint_weight*nominal.sum()
        for weight, value in zip(self.smooth_weights[:3], (dq, acceleration, jerk)):
            objective = objective+.5*float(weight)*value[:, self.regular_indices].square().sum()
        variable_terms, variable_names, fixed_values, fixed_names = [], [], [], []
        if self.dynamics is not None:
            native_state = self.JointState(position=native_q, velocity=dq.to(torch.float32).contiguous(),
                acceleration=acceleration.to(torch.float32).contiguous(), joint_names=list(self.physical.joint_names))
            effort = self.dynamics.compute_inverse_dynamics(native_state).to(torch.float64)
            selected = effort[:, self.regular_indices]
            objective = objective+.5*float(self.smooth_weights[3])*selected.square().sum()
            objective = objective+float(self.smooth_weights[4])*(selected*dq[:, self.regular_indices]*self.dt).square().sum()
            for side, sign, bound in (("lower", 1., self.effort_bounds[0]), ("upper", -1., self.effort_bounds[1])):
                values = sign*(effort-self.tensor(bound))
                for joint_index, name in enumerate(self.physical.joint_names):
                    variable_terms.append(values[..., joint_index].min())
                    variable_names.append(f"effort/{name}/{side}")
        idx = self.scene_indices
        signed = self.scene.signed_constraints(spheres[:, idx], position[:, idx], quaternion[:, idx], self.scene_progress,
            q[:, idx], self.physical.joint_names, trajectory_time=self.scene_times,
            joint_variable_mask=self.variable_mask)
        scene_report = {"groups": {}, "fixed_failures": {}, "diagnostics": signed.diagnostics}
        for name, values in signed.slack.items():
            valid = signed.valid[name]
            scene_report["groups"][name] = {"slack": values.detach().cpu().numpy().tolist(),
                "valid": valid.detach().cpu().numpy().tolist(),
                "variable": signed.variable[name].detach().cpu().numpy().tolist()}
            for slot in (0, 1):
                selected = values[..., slot][valid[..., slot]]
                # An empty group has no physical inequality. Keeping a zero
                # placeholder makes SLSQP's row layout stable; validity in the
                # report prevents interpreting it as a measured clearance.
                minimum = selected.min() if selected.numel() else x.sum()*0.
                if slot:
                    variable_terms.append(minimum)
                    variable_names.append(f"scene/{name}/variable")
                else:
                    fixed_values.append(float(minimum.detach().item()))
                    fixed_names.append(f"scene/{name}/fixed")
        for name, failure in signed.fixed_failures.items():
            value = float(failure.detach().max().item())
            fixed_values.append(-value)
            fixed_names.append(f"scene/{name}/input")
            scene_report["fixed_failures"][name] = value
        physical = self.physical.constraints(actions)
        raw_objective = float(objective.detach().item())
        objective = objective/self.objective_scale
        objective_gradient = torch.autograd.grad(objective, x, retain_graph=bool(variable_terms))[0].detach().clone()
        scene_jacobian = []
        for index, term in enumerate(variable_terms):
            gradient = (torch.autograd.grad(term, x, retain_graph=index+1 < len(variable_terms))[0]
                        if term.requires_grad else torch.zeros_like(x))
            # cuRobo reuses native backward buffers; retain each completed VJP
            # before the next constraint asks the same graph for a gradient.
            scene_jacobian.append(gradient.detach().clone().reshape(-1).cpu().numpy())
        extra_slack = np.asarray([float(term.detach().item()) for term in variable_terms])
        jacobian = np.concatenate((physical.jacobian, np.asarray(scene_jacobian).reshape(-1, actions.size)))
        result = CurveEvaluation(actions.copy(), float(objective.detach().item()), raw_objective,
            objective_gradient.reshape(-1).cpu().numpy(), np.r_[physical.slack, extra_slack], jacobian,
            physical.names+tuple(variable_names), np.r_[physical.fixed_slack, fixed_values],
            physical.fixed_names+tuple(fixed_names), physical, scene_report)
        if not (np.isfinite(result.objective) and np.isfinite(result.objective_gradient).all()
                and np.isfinite(result.slack).all() and np.isfinite(result.jacobian).all()
                and np.isfinite(result.fixed_slack).all()):
            raise ValueError("Physical trajectory evaluation produced nonfinite values or derivatives")
        self.calls += 1
        self.states += len(self.scene_indices)
        self.last = result
        self.check_request()
        return result

    def solve(self, actions):
        """Run the installed SLSQP with the public physical accuracy contract."""
        initial = np.asarray(actions, dtype=np.float64).reshape(-1)
        first = self.evaluate(initial)
        scene_fixed = first.fixed_slack[len(first.physical.fixed_slack):]
        fixed_violation = self.physical.fixed_violation_lower_bound+aggregate_violation(scene_fixed)
        if fixed_violation > PHYSICAL_FTOL:
            result = OptimizeResult(x=initial, success=False, status=4, nit=0,
                message="Fixed physical inputs violate the request constraints")
        else:
            def objective(value):
                evaluated = self.evaluate(value)
                return evaluated.objective, evaluated.objective_gradient
            # Every reported fixed row enters the same vector. Its derivative
            # is structurally zero; float rounding of a rigid world transform
            # cannot be an optimization degree of freedom or change row layout.
            def slack(value):
                evaluated = self.evaluate(value)
                return np.r_[evaluated.slack, evaluated.fixed_slack]
            def jacobian(value):
                evaluated = self.evaluate(value)
                return np.concatenate((evaluated.jacobian,
                    np.zeros((len(evaluated.fixed_slack), initial.size))))
            result = minimize(objective, initial, method="SLSQP", jac=True,
                constraints={"type": "ineq", "fun": slack, "jac": jacobian},
                callback=lambda _: self.check_request(),
                options={"ftol": PHYSICAL_FTOL, "maxiter": self.budget.trajectory_iterations})
        final = self.evaluate(result.x)
        report = {"method": "scipy.optimize.SLSQP", "success": bool(result.success),
            "status": int(result.status), "message": str(result.message), "iterations": int(result.nit),
            "initial_objective": first.objective, "final_objective": final.objective,
            "initial_raw_objective": first.raw_objective, "final_raw_objective": final.raw_objective,
            "objective_scale": self.objective_scale,
            "objective_units": "whole original objective divided by the existing nominal weight",
            "initial_violation": first.violation, "final_violation": final.violation,
            "maximum_control_update": float(np.max(np.abs(final.actions.reshape(-1)-initial))),
            "ftol": PHYSICAL_FTOL}
        return final, report
# END explicit physical-constraint trajectory solve.
