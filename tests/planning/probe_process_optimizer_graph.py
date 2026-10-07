"""Compare real ProcessCostManager/native-robot IK and trajectory graph blocks.

The recorded source robot and installation geometry are real fixture inputs.
The reach goal, empty background and nominal fixed hand are synthetic, declared
test inputs. Each current eager core is compared with an independent graphed
core on the same native rollout objects. No plan, candidate metrics, dense
metrics or simulation is run; archived arithmetic is checked by the separate
analytic optimizer and component differential probes.
"""

# BEGIN bounded full-manager integration, separate from acceptance certification.
import argparse
from copy import deepcopy
import json
from pathlib import Path
import resource
import sys
import time
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    import torch
    from curobo.kinematics import Kinematics, KinematicsCfg
    from curobo.types import DeviceCfg, JointState
    from curobo._src.optim.gradient.lbfgs import LBFGSOpt
    from curobo._src.solver.manager_goal import GoalManager
    from curobo._src.solver.solve_mode import SolveMode
    from curobo._src.solver.solve_state import SolveState
    from manipulation.planning import PlanningBudget, _CuroboPlanner
    from manipulation.planning_problem import MotionPhase, PlanningProblem, PosePath, PoseEventRequirement
    from manipulation.curobo_solver_components import bind_solver_components
    from probe_robot_fixture import load_probe_fixture, fixed_probe_hand
    from probe_process_planner import no_obstacle_scene
    from probe_optimizer_graph_equivalence import (
        observe_iterations, state_tensors, history_tensors, tracker_tensors, compare_tree)

    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError("This explicit native-robot capture probe requires CUDA access")
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(.5, 2*1024**3/total), 0)
    torch.cuda.reset_peak_memory_stats(0)
    device = DeviceCfg("cuda:0")
    begin = time.perf_counter()
    fixture, robot, validator = load_probe_fixture(arguments.fixture, reference_frame="robot_base")
    full_positions = np.asarray(robot["robot_cfg"]["kinematics"]["cspace"]["default_joint_position"])
    kinematics = Kinematics(KinematicsCfg.from_robot_yaml_file(deepcopy(robot), device_cfg=device))
    initial = kinematics.default_joint_position.reshape(-1).clone()
    target = initial.clone()
    target[0] += .025
    native = kinematics.compute_kinematics(JointState.from_position(
        torch.stack((initial, target))[None].contiguous(), joint_names=kinematics.joint_names))
    index = kinematics.tool_frames.index(fixture["tool_link"])
    p = native.tool_poses.position[0, :, index].detach().cpu().numpy().copy()
    q = native.tool_poses.quaternion[0, :, index].detach().cpu().numpy().copy()
    poses = np.repeat(np.eye(4)[None], 2, axis=0)
    poses[:, :3, 3] = p
    poses[:, :3, :3] = Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()
    poses = poses @ validator.T_frame_tcp
    problem = PlanningProblem(tuple(kinematics.joint_names), initial.cpu().numpy(),
        np.zeros(len(initial)), fixture["tool_link"], "robot_base", np.eye(4), validator.T_frame_tcp,
        (MotionPhase("explicit_capture_probe", PosePath(np.array([0., 1.]), poses,
            "synthetic endpoint from the supplied native robot")),),
        pose_events=(PoseEventRequirement(1., np.eye(4), poses[-1], .001, .005,
            "explicit synthetic reach for optimizer graph differentiation"),),
        source_timestamps=np.array([0., 1.]), source_progress=np.array([0., 1.]))
    hand, owners, hand_report = fixed_probe_hand(fixture, robot, problem, device=device,
                                                full_current_positions=full_positions)
    scene = no_obstacle_scene(kinematics.total_spheres, hand=hand, installation_owners=owners,
                              geometry_source=fixture["robot_geometry_source"])
    budget = PlanningBudget(ik_seeds=1, trajectory_seeds=1, knots=6,
        ik_iterations=20, trajectory_iterations=25, max_robot_spheres=kinematics.total_spheres,
        maximum_trajectory_solves=1, search_seconds=0.)
    del native, kinematics
    planner = None
    paired_optimizers = []
    try:
        planner = _CuroboPlanner(problem, scene, robot, budget, "cuda:0", robot_validator=validator)
        # BEGIN bind the same real functional goal or goal-free native route.
        seed = planner.start.position[:, None].expand(1, planner.trajopt.action_horizon, -1).clone()
        planner.physical_domain.validate(seed)
        solve_state, goal = planner._native_request(seed[:, None])
        seed_goal = JointState.from_position(seed[:, -1:, :].clone())
        seed_goal.dt = torch.full((1, 1), budget.maximum_trajectory_dt, device=device.device)
        # END no reference endpoint is installed as a storage-only pose goal.
        goal_buffer, _ = planner.trajopt._prepare_goal_buffer(
            solve_state, goal, planner.start, False, seed_goal, None)
        planner.trajopt._update_rollout_params(goal_buffer)
        template = planner.trajopt.core.optimizers[0]
        if template.config.inner_iters != 25 or template.config.num_iters != 25:
            raise AssertionError("The probe must run exactly one unchanged native 25-step block")
        # The paired cores share the same concrete rollout instances and goal
        # buffers, evaluated sequentially. Their L-BFGS histories are distinct.
        # This removes independent scene/FK construction from the comparison.
        def compare_optimizers(label, template, domain, goal_buffer, seed):
            pair = []
            for use_graph in (False, True):
                optimizer = LBFGSOpt(deepcopy(template.config), template.get_all_rollout_instances())
                holder = SimpleNamespace(core=SimpleNamespace(config=SimpleNamespace(use_cuda_graph=False),
                    device_cfg=device, optimizers=[optimizer], goal_registry_manager=GoalManager(device)))
                bind_solver_components(holder, domain, use_optimizer_graph=use_graph)
                optimizer.update_rollout_params(goal_buffer)
                observe_iterations(optimizer)
                paired_optimizers.append(optimizer)
                pair.append(optimizer)
            cases = []
            for case in range(2):
                # Same layout, a changed legal seed on replay. The measured
                # entrance and the clock still come from the original problem.
                values = seed.clone()
                if case:
                    values[:, :, 0] += .002
                domain.validate(values)
                answers = []
                for optimizer in pair:
                    optimizer._core.probe_blocks = []
                    optimizer.reinitialize(values)
                    entrance = state_tensors(optimizer._core._iteration_state)
                    answer = optimizer.optimize(values)
                    answers.append({"entrance": entrance, "answer": answer.clone(),
                        "blocks": optimizer._core.probe_blocks, "history": history_tensors(optimizer),
                        "tracker": tracker_tensors(optimizer)})
                compare_tree(*answers, f"{label}/case={case}")
                cases.append({"changed_seed": bool(case), "all_tensor_bytes_equal": True})
            result = {"cases": cases, "native_iterations": template.config.num_iters,
                "native_inner_iterations": template.config.inner_iters,
                "graph_capture_bytes": pair[1]._core.graph_capture_bytes,
                "graph_extra_workspace_bytes": pair[1]._core.extra_workspace_bytes}
            for optimizer in pair:
                optimizer.reset_cuda_graph()
            return result

        trajectory = compare_optimizers("trajectory_optimizer", template,
            planner.physical_domain, goal_buffer, seed)
        # Seed IK has the same independently timed native finger geometry but
        # deliberately omits the complete trajectory's scene/process terms.
        ik_goal = planner._goal(problem.phase_count)
        ik_state = SolveState(SolveMode.SINGLE, num_ik_seeds=1, batch_size=1,
            num_envs=1, num_goalset=1, tool_frames=ik_goal.tool_frames)
        current = planner.ik._fill_current_state_dt(planner.start)
        ik_buffer, _ = planner.ik._prepare_goal_buffer(ik_state, ik_goal, current)
        planner.ik._update_rollout_params(ik_buffer, include_auxiliary_rollout=False)
        planner.seed_parameter.fill_(.5)
        ik = compare_optimizers("seed_optimizer", planner.ik.core.optimizers[0],
            planner.position_domain, ik_buffer, planner.start.position[:, None].clone())
        report = {"scope": __doc__, "trajectory_optimizer": trajectory, "seed_optimizer": ik,
            "production_graph_admission": {
                "seed": planner.ik.core.optimizers[0].use_cuda_graph,
                "trajectory": planner.trajopt.core.optimizers[0].use_cuda_graph},
            "capture_unit": "one native step; original 20/25-step block iterations and fault boundary",
            "joint_count": len(planner.joint_names), "sphere_count": planner.robot_metadata["spheres"],
            "knots": budget.knots,
            "peak_torch_gpu_bytes": torch.cuda.max_memory_allocated(0),
            "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            "seconds": time.perf_counter()-begin, "fixed_hand_hypothesis": hand_report,
            "candidate_or_dense_metrics_run": False, "simulation_run": False}
        # A graph also retains CPU node metadata. The unchanged 25-step block
        # now replays one native step graph. This probe's 2 GiB device/4 GiB
        # host guards are stricter than the user's 8/10 GiB maxima; always save
        # evidence on a guard fault, without weakening numeric comparisons.
        report["limits"] = {"gpu_bytes": 2*1024**3, "process_rss_bytes": 4*1024**3}
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(json.dumps(report, indent=2)+"\n")
        print(json.dumps(report, indent=2))
        if (report["peak_torch_gpu_bytes"] > report["limits"]["gpu_bytes"]
                or report["peak_process_rss_bytes"] > report["limits"]["process_rss_bytes"]):
            raise RuntimeError("Native-manager probe exceeded its declared device/host allocation")
    finally:
        for optimizer in paired_optimizers:
            optimizer.reset_cuda_graph()
        if planner is not None:
            planner.close()


if __name__ == "__main__":
    main()
# END passing this synthetic fixture does not certify an arbitrary scene graph.
