"""Bounded native regression for self collision at added Hermite samples.

The synthetic robot has one prismatic joint and two explicitly enabled spheres.
Three regular samples can be separated while the added u=0.25 sample overlaps.
The production ProcessCostManager, CUDA FK and CUDA self-collision cost are all
used unchanged; unrelated task and scene terms return zero. A second case has
both regular and added collisions, so its analytic gradient detects overwritten
native workspaces. Run this probe serially with other GPU work.
"""

# BEGIN isolated native anchor-collision regression; no environment mutation.
import argparse
from copy import deepcopy
import json
from pathlib import Path
import resource
import sys
import time
import traceback
from types import SimpleNamespace

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def _robot_config(output):
    """Declare a two-sphere test model, without loading a task or scene asset."""
    urdf = output / "two_sphere_slider.urdf"
    urdf.write_text('''<?xml version="1.0"?>
<robot name="anchor_collision_slider">
  <link name="base"/>
  <link name="slider"/>
  <joint name="slide" type="prismatic">
    <parent link="base"/><child link="slider"/>
    <origin xyz="0 0 0" rpy="0 0 0"/><axis xyz="1 0 0"/>
    <limit lower="-1" upper="1" effort="10" velocity="2"/>
  </joint>
</robot>
''')
    return {"robot_cfg": {"kinematics": {
        "format_version": 2.0, "urdf_path": str(urdf.resolve()),
        "asset_root_path": str(output.resolve()), "base_link": "base",
        "tool_frames": ["slider"], "collision_link_names": ["base", "slider"],
        "collision_spheres": {
            "base": [{"center": [.03, 0., 0.], "radius": .04}],
            "slider": [{"center": [0., 0., 0.], "radius": .04}]},
        "collision_sphere_buffer": 0., "self_collision_buffer": {"base": 0., "slider": 0.},
        # This is an intentional synthetic collision pair. No adjacency ignore
        # is inferred: moving the slider through the fixed sphere must collide.
        "self_collision_ignore": {},
        "cspace": {"joint_names": ["slide"], "default_joint_position": [-.3],
                   "cspace_distance_weight": [1.], "null_space_weight": [1.],
                   "max_acceleration": 10., "max_jerk": 100.}}}}


def run(output):
    """Exercise production wiring and compare native backward to exact algebra."""
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    report = {"status": "running", "scope": __doc__, "cases": []}
    try:
        # Importing this file or requesting --help must not initialize CUDA.
        import torch
        from curobo.kinematics import Kinematics, KinematicsCfg
        from curobo.types import DeviceCfg, JointState
        from curobo._src.state.state_robot import RobotState
        from manipulation.planning import PlanningBudget, _ProgressMap, _process_manager_type

        torch.set_num_threads(1)
        torch.cuda.set_per_process_memory_fraction(
            min(.5, 2*1024**3/torch.cuda.get_device_properties(0).total_memory))
        torch.cuda.reset_peak_memory_stats()
        device = DeviceCfg("cuda:0")
        configuration = _robot_config(output)
        fk = Kinematics(KinematicsCfg.from_robot_yaml_file(deepcopy(configuration), device_cfg=device),
                        compute_jacobian=True)
        if tuple(fk.joint_names) != ("slide",) or fk.total_spheres != 2:
            raise AssertionError("The native model must retain exactly the declared joint and spheres")
        pair_config = fk.get_self_collision_config()
        if pair_config.collision_pairs.shape[0] != 1:
            raise AssertionError("The synthetic sphere pair was unexpectedly excluded")

        # Only unrelated task/scene behavior is replaced by a zero-valued
        # adapter. Both FK models and both self-collision costs remain native.
        problem = SimpleNamespace(
            tool_link="slider", start_progress=0., phase_count=1., mechanisms=(), stop_progress=(),
            base_pose=np.eye(4), transform_tool_pose=lambda p, r: (p, r),
            transform_robot_spheres=lambda value: value,
            evaluate=lambda p, r, progress, *args, **kwargs: torch.zeros_like(progress),
            optimizer_preferences=lambda p, r, progress, **kwargs: torch.zeros_like(progress))
        scene = SimpleNamespace(
            gripper=None, robot_coverage=None, robot_owner_permissions=(), required_progress_anchors=(),
            evaluate=lambda spheres, p, r, progress, *args, **kwargs: torch.zeros_like(progress),
            optimizer_preferences=lambda progress, *args: torch.zeros_like(progress))
        scene.clone_for_rollout = lambda: scene
        mapping = _ProgressMap(np.array([0., 1.]), np.array([0., 1.]))
        transition = SimpleNamespace(robot_model=fk, joint_names=fk.joint_names, interpolation_steps=4)
        manager_type = _process_manager_type(problem, scene, mapping, (0., 1.), PlanningBudget(),
            refinement_parameters=(.25,), clock={"duration": 1.}, robot_dict=configuration)

        def manager(mode):
            config = manager_type.create({"process_mode": mode, "self_collision_cfg": {
                "weight": 8., "convert_to_binary": mode == "trajectory_metrics",
                "use_grad_input": False}}, device_cfg=device)
            result = config.class_type(device)
            result.initialize_from_config(config, transition)
            if result.get_cost("self_collision")._weight.item() != 2.:
                raise AssertionError("Native interpolation normalization must occur exactly once")
            if not torch.equal(result._anchor_u, device.to_device([[0., .25, 1.]])):
                raise AssertionError("The added refinement time must coexist with original anchors")
            return result

        def evaluate(cost_manager, positions, *, backward):
            q = device.to_device(positions).unsqueeze(-1).requires_grad_(backward)
            dq = torch.zeros_like(q, requires_grad=backward)
            js = JointState(position=q, velocity=dq, dt=device.to_device([.5, .5]),
                            joint_names=list(fk.joint_names))
            native_state = fk.compute_kinematics(js)
            costs = cost_manager.compute_costs(RobotState(joint_state=js, cuda_robot_model_state=native_state))
            named = dict(zip(costs.names, costs.values))
            if "anchor_self_collision" not in named:
                raise AssertionError("Production manager omitted self collision at its extra samples")
            regular, added = named["self_collision"], named["anchor_self_collision"]
            if regular.shape != (2, 3, 1) or added.shape != regular.shape:
                raise AssertionError("Both collision terms must retain the native [batch,horizon,1] layout")
            if bool((added[:, :-1] != 0).any()):
                raise AssertionError("Added collision costs must be folded only into the final column")
            # Capture values before backward or the next native forward can
            # reuse its persistent outputs. The following sum is deliberately
            # unweighted, honoring SelfCollisionCost.use_grad_input=False.
            values = regular.detach().clone(), added.detach().clone()
            gradients = None
            if backward:
                (regular.sum()+added.sum()).backward()
                gradients = q.grad.detach().clone(), dq.grad.detach().clone()
                if not all(bool(torch.isfinite(value).all()) for value in gradients):
                    raise AssertionError("Native combined q/dq gradients must be finite")
            return values, gradients

        def analytic(positions, *, binary):
            # At u=.25 the first dt=.5 segment is exactly at local a=.5.
            # Its Hermite position is (q0+q1)/2 + dt*(dq0-dq1)/8.
            # Derive this independently instead of calling the production
            # Hermite helper whose integration this probe is checking.
            q = device.to_device(positions).unsqueeze(-1).requires_grad_(not binary)
            dq = torch.zeros_like(q, requires_grad=not binary)
            midpoint = .5*(q[:, 0]+q[:, 1])+.0625*(dq[:, 0]-dq[:, 1])
            anchors = torch.stack((q[:, 0], midpoint, q[:, 2]), dim=1)

            def cost(position):
                raw = .5*2.*torch.relu(.08**2-(position-.03).square())
                return torch.where(raw > 0, raw.clamp(max=1.)+1., raw) if binary else raw

            regular, anchors = cost(q), cost(anchors)
            added = torch.cat((torch.zeros_like(regular[:, :-1]), anchors.sum(1, keepdim=True)), dim=1)
            gradients = None
            if not binary:
                (regular.sum()+added.sum()).backward()
                gradients = q.grad, dq.grad
            return (regular.detach(), added.detach()), gradients

        optimizer, metrics = manager("trajectory_optimizer"), manager("trajectory_metrics")
        cases = (
            ("regular_grid_clear_added_sample_collides", [[-.30, .30, .40], [-.25, .35, .45]]),
            ("regular_and_added_gradients_coexist", [[0., .13, .40], [.01, .15, .45]]),
        )
        for name, positions in cases:
            native, gradients = evaluate(optimizer, positions, backward=True)
            expected, expected_gradients = analytic(positions, binary=False)
            for actual, reference in zip((*native, *gradients), (*expected, *expected_gradients)):
                torch.testing.assert_close(actual, reference, atol=2e-6, rtol=2e-5)
            if not bool((native[1][:, -1] > 0).all()) or not bool((gradients[0].abs() > 0).any()):
                raise AssertionError("Added collisions must produce positive optimizer cost and a q gradient")
            if name == "regular_grid_clear_added_sample_collides" and bool((native[0] != 0).any()):
                raise AssertionError("The missed-sample fixture must be clear on every original grid state")
            metric_values, _ = evaluate(metrics, positions, backward=False)
            metric_expected, _ = analytic(positions, binary=True)
            for actual, reference in zip(metric_values, metric_expected):
                torch.testing.assert_close(actual, reference, atol=2e-6, rtol=2e-5)
            if not bool((metric_values[1][:, -1] > 1.).all()):
                raise AssertionError("Native binary metrics must reject each added collision")
            report["cases"].append({
                "name": name, "q_m": positions, "regular_cost": native[0].cpu().tolist(),
                "added_cost": native[1].cpu().tolist(), "joint_position_gradient": gradients[0].cpu().tolist(),
                "joint_velocity_gradient": gradients[1].cpu().tolist(),
                "metrics_added_cost": metric_values[1].cpu().tolist(),
                "max_analytic_gradient_error": max(float((a-b).abs().max())
                    for a, b in zip(gradients, expected_gradients))})
        torch.cuda.synchronize()
        report.update(status="passed", batch_size=2, regular_samples=3, added_samples=3,
            native_weight_after_interpolation=2., native_pair_count=1,
            torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            torch_peak_reserved_bytes=torch.cuda.max_memory_reserved())
    except Exception as error:
        report.update(status="error", error=f"{type(error).__name__}: {error}")
        (output / "exception.txt").write_text(traceback.format_exc())
        raise
    finally:
        report.update(elapsed_seconds=time.monotonic()-started,
                      peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
        print(json.dumps(report, allow_nan=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
# END isolated native anchor-collision regression.
