"""Bounded native coordinate-rate and gradient probe with one Franka model.

This checks real CUDA FK/Jacobian outputs and their backward pass. It is not a
contact experiment or a proof of direction between the finite curve queries.
Run serially in recovery with the reviewed local cuRobo source on PYTHONPATH.
"""

# BEGIN native J(q) dq direction gradients through calibrated rigid transforms.
import argparse
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def transform(translation, angles):
    value = np.eye(4)
    value[:3, :3] = Rotation.from_euler("xyz", angles).as_matrix()
    value[:3, 3] = translation
    return value


def run(output):
    import torch
    from curobo.kinematics import Kinematics, KinematicsCfg
    from curobo._src.state.state_joint import JointState
    from manipulation.motion import MotionModel
    from manipulation.planning import _native_gradient_layout_type, _orientation_vjp_type
    from manipulation.planning_problem import (CoordinateDirectionRequirement,
        MechanismRequirement, MotionPhase, PlanningProblem, PosePath)

    if not torch.cuda.is_available():
        raise RuntimeError("This probe requires native CUDA; no CPU substitute is reported")
    torch.set_num_threads(1)
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(512*1024**2/total, 1.))
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    robot = Kinematics(KinematicsCfg.from_robot_yaml_file("franka.yml"), compute_jacobian=True)
    names = tuple(robot.joint_names)
    actual = robot.default_joint_state.position.detach().reshape(-1)
    q = actual[None, None].repeat(2, 3, 1)
    offset = torch.linspace(-.035, .04, q.numel(), device=q.device, dtype=q.dtype).reshape_as(q)
    q = q+offset
    dq = torch.linspace(-.09, .11, q.numel(), device=q.device, dtype=q.dtype).reshape_as(q)
    reference = PosePath(np.array([0., 1.]), np.stack((np.eye(4), np.eye(4))), "unused reference fixture")
    problem = PlanningProblem(names, actual.cpu().numpy(), np.zeros(len(names)), robot.tool_frames[0],
        "independently calibrated probe world", transform([.1, -.2, .3], [.3, -.4, .2]),
        transform([.03, -.02, .07], [.2, -.1, .3]), (MotionPhase("reference", reference),))
    layout, orientation = _native_gradient_layout_type(), _orientation_vjp_type()
    basis = Rotation.from_euler("xyz", [.3, -.2, .5]).as_matrix()
    attachment = transform([.04, .03, -.02], [-.2, .3, .1])
    times = q.new_tensor([[0., .4, 1.], [0., .4, 1.]])
    progress = q.new_tensor([[0., .5, 1.], [0., .5, 1.]])
    cases = []
    for kind in ("prismatic", "revolute"):
        model = MotionModel(kind, np.zeros((1, 1)), np.eye(4), basis, np.zeros(3),
                            1e-4, 1e-4, {"source": "independent analytic rate-coordinate fixture"})
        requirement = MechanismRequirement(0., 1., model, attachment, None, [0.], "explicit probe mechanism",
            (CoordinateDirectionRequirement(0., 1., 0, "local_rate", 1, 1., 0., "explicit rate threshold"),))

        def objective(q_input, dq_input, gradient=False):
            positions = q_input.detach().clone().requires_grad_(gradient)
            velocities = dq_input.detach().clone().requires_grad_(gradient)
            state = robot.compute_kinematics(JointState.from_position(positions, joint_names=list(names)))
            quaternion = orientation.apply(positions, state.tool_poses.quaternion.detach(),
                                           state.tool_jacobians[..., 3:, :].detach())[:, :, 0]
            jacobian = layout.apply(state.tool_jacobians)[:, :, 0]
            twist = torch.einsum("...kd,...d->...k", jacobian, velocities)
            linear, angular = problem.transform_tool_velocity(twist[..., :3], twist[..., 3:], quaternion)
            position, quaternion = problem.transform_tool_pose(layout.apply(state.tool_poses.position)[:, :, 0], quaternion)
            residual = next(iter(requirement.direction_residuals(position, quaternion, progress,
                trajectory_time=times, tool_linear_velocity=linear, tool_angular_velocity=angular).values()))
            loss = residual.square().sum()
            derivatives = (None, None) if not gradient else tuple(value.detach().clone() for value in
                torch.autograd.grad(loss, (positions, velocities)))
            return float(loss.detach().cpu()), derivatives

        loss, gradients = objective(q, dq, True)
        row = {"kind": kind, "loss": loss, "differences": []}
        for variable, value, derivative in zip(("q", "dq"), (q, dq), gradients):
            flat = int(derivative.abs().reshape(-1).argmax().item())
            index = np.unravel_index(flat, tuple(value.shape))
            expected = float(derivative[index].cpu())
            # Two explicit physical perturbations report float32 stability;
            # the larger one is the fixed acceptance test for native CUDA.
            for epsilon in (.002, .0005):
                upper, lower = value.clone(), value.clone()
                upper[index] += epsilon; lower[index] -= epsilon
                plus, _ = objective(upper, dq) if variable == "q" else objective(q, upper)
                minus, _ = objective(lower, dq) if variable == "q" else objective(q, lower)
                finite_difference = (plus-minus)/(2*epsilon)
                allowed = .003+.03*abs(finite_difference)
                row["differences"].append({"variable": variable, "index": list(map(int, index)),
                    "epsilon": epsilon, "autograd": expected, "central_difference": finite_difference,
                    "absolute_error": abs(expected-finite_difference), "allowed_error": allowed,
                    "passed": bool(np.isfinite(expected) and abs(expected-finite_difference) <= allowed)})
        row["passed"] = all(item["passed"] for item in row["differences"] if item["epsilon"] == .002)
        cases.append(row)
    report = {"scope": __doc__, "cases": cases, "passed": all(row["passed"] for row in cases),
        "elapsed_s": time.monotonic()-started, "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        "peak_torch_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_torch_reserved_bytes": torch.cuda.max_memory_reserved(), "physical_execution_tested": False}
    output.mkdir(parents=True, exist_ok=False)
    (output/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps(report), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    raise SystemExit(run(parser.parse_args().output))
# END the same public direction residual consumes actual native poses and twist.
