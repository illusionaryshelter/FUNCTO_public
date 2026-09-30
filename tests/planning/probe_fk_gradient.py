"""Check native FK position/quaternion Jacobians against finite differences."""

# BEGIN native FK gradient isolation; no environment or library mutations.
import argparse
import json
from pathlib import Path
import resource
import time

def run(output):
    # BEGIN load the native GPU backend only for an explicit probe run.
    # Importing this helper or requesting --help must not load vendor kernels.
    import torch
    from curobo.kinematics import Kinematics, KinematicsCfg
    from curobo.types import JointState
    # END deferred GPU imports.
    started = time.monotonic()
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(2*1024**3/torch.cuda.get_device_properties(0).total_memory)
    kinematics = Kinematics(KinematicsCfg.from_robot_yaml_file("franka.yml"), compute_jacobian=True)
    q = kinematics.default_joint_position.reshape(1, 1, -1).clone().requires_grad_(True)

    def forward(value):
        state = kinematics.compute_kinematics(JointState.from_position(value, joint_names=kinematics.joint_names))
        return torch.cat((state.tool_poses.position[0, 0, 0], state.tool_poses.quaternion[0, 0, 0]))

    pose = forward(q)
    native = kinematics.compute_kinematics(JointState.from_position(q, joint_names=kinematics.joint_names))
    jacobian = native.tool_jacobians[0, 0, 0].detach().clone()
    # d(wxyz)/dq = .5 * [-v^T; wI-[v]x] * J_omega for
    # angular velocity expressed in the robot base (world) frame.
    rotation = pose[3:].detach().clone()
    w, x, y, z = rotation
    mapping = .5*torch.stack((torch.stack((-x, -y, -z)),
                            torch.stack((w, z, -y)),
                            torch.stack((-z, w, x)),
                            torch.stack((y, -x, w))))
    geometric_quaternion_jacobian = mapping @ jacobian[3:]
    analytic = torch.stack([torch.autograd.grad(pose[i], q, retain_graph=True)[0].reshape(-1).clone()
                            for i in range(7)]).detach()
    finite = []
    for i in range(q.shape[-1]):
        delta = torch.zeros_like(q)
        delta[..., i] = 1e-3
        with torch.no_grad():
            plus = forward(q.detach()+delta).clone()
            minus = forward(q.detach()-delta).clone()
        finite.append((plus-minus)/(2e-3))
    finite = torch.stack(finite, -1)
    report = {"analytic": analytic.cpu().tolist(), "finite_difference": finite.cpu().tolist(),
              "position_max_error": float((analytic[:3]-finite[:3]).abs().max()),
              "quaternion_max_error": float((analytic[3:]-finite[3:]).abs().max()),
              "geometric_position_max_error": float((jacobian[:3]-finite[:3]).abs().max()),
              "geometric_quaternion_max_error": float((geometric_quaternion_jacobian-finite[3:]).abs().max()),
              "elapsed_seconds": time.monotonic()-started,
              "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "torch_peak_reserved_bytes": torch.cuda.max_memory_reserved()}
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
# END native FK gradient isolation; no environment or library mutations.
