"""Bounded numerical/performance probe of real native coverage guidance.

The fixture and archived raw free grid are verified by the production adapter.
Eight synthetic rigid translations of its measured link poses exercise the
guidance expression and its derivatives. These are numerical query inputs,
not robot trajectories, and neither coverage acceptance nor physics is run.
"""

# BEGIN serial, explicitly bounded native-guidance experiment.
import argparse
from copy import copy
import json
import math
from pathlib import Path
import resource
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from manipulation.planning import PlanningBudget, _planning_workspace_bytes
import manipulation.planning_scene as scene_module
from manipulation.planning_scene import ObservedSpace, PlanningScene
from robot_coverage_fixture import build_robot_coverage


def read_json(path):
    if path.stat().st_size > 16 * 1024**2:
        raise ValueError("Probe metadata must fit 16 MiB")
    return json.loads(path.read_text())


def run(args):
    torch.set_num_threads(1)
    fixture, feedback, audit = map(read_json, (args.fixture, args.feedback, args.workspace_audit))
    with np.load(args.observed_free, allow_pickle=False) as saved:
        coverage = ObservedSpace(saved["known_free"], origin=saved["origin"],
            voxel_size=float(saved["voxel_size"]), observation_ids=saved["observation_ids"].tolist(),
            reference_frame=feedback["reference_frame"])
    oracle = build_robot_coverage(fixture, feedback, coverage,
        initialization_source=args.initial_state_assumption_source)
    scene = PlanningScene(background=SimpleNamespace(owner_ids=()), owners=(), coverage=coverage,
        reference_frame=feedback["reference_frame"], robot_sphere_materials=(None,)*audit["robot_metadata"]["spheres"],
        robot_geometry_source="Verified native fixture; guidance-only numerical probe",
        robot_coverage=oracle, has_gripper=False)
    solids = scene._coverage_geometry["solids"]
    counts = [len(row["baseline_halfspaces_world"]) for row in solids]
    # Only these declared counts enter the allocation expression. Their
    # placeholders are never submitted to any geometric query or optimizer.
    allocation_scene = copy(scene)
    # BEGIN declare every field consumed by current allocation accounting.
    allocation_scene.owners = tuple(SimpleNamespace(moving=False, owner_id=f"static-{index}", query=None)
                                    for index in range(args.static_owners))
    allocation_scene.gripper = SimpleNamespace(self_collision_pairs=range(args.hand_pairs),
        finger_trajectory=None, nominal_finger_trajectory=None)
    # END these metadata placeholders still never enter geometric evaluation.
    budget = PlanningBudget(**audit["unchanged_budget"])
    scene_bytes, fk_bytes, trajectory_graph = _planning_workspace_bytes(
        allocation_scene, audit["robot_metadata"], budget, audit["reserved_anchor_capacity"])
    report = {"scope": __doc__, "fixture": str(args.fixture.resolve()),
        "feedback": str(args.feedback.resolve()), "observed_free": str(args.observed_free.resolve()),
        "gpu_used": not args.cpu_only, "coverage_queries": 0,
        "solid_plane_counts": counts, "solid_sample_counts": [len(p) for p in scene._coverage_samples],
        "grid_shape": list(coverage.known_free.shape), "static_owner_count": args.static_owners,
        "hand_pair_count": args.hand_pairs, "scene_workspace_bytes": scene_bytes,
        "trajectory_optimizer_graph_admitted": trajectory_graph,
        "scene_workspace_mib": scene_bytes/1024**2, "anchor_fk_bytes": fk_bytes,
        "scene_limit_bytes": budget.max_scene_workspace_bytes, "methods": {}}
    if not args.cpu_only:
        device = torch.device("cuda:0")
        free, total = torch.cuda.mem_get_info(device)
        cap = 1024**3
        if free < 2*cap:
            raise ValueError("Bounded standalone probe requires 2 GiB free device memory")
        torch.cuda.set_per_process_memory_fraction(cap/total, device)
        report["torch_allocator_cap_bytes"] = cap
        frames = oracle.required_link_names
        baseline = {row["frame"]: row["baseline_pose"] for row in solids}
        translations = np.array([[0, 0, 0], [.003, -.002, .001], [.02, .03, -.01],
            [-.02, .04, .01], [.1, -.06, .02], [-.1, .2, -.03], [.4, .3, .1], [1., -.3, .2]])
        base_p = np.array([baseline[name][:3, 3] for name in frames])
        # Match the production FK pose contract: rotations are WXYZ unit
        # quaternions, not rotation matrices. Derivatives remain numerical
        # input derivatives; this does not vary physical joint coordinates.
        base_r = Rotation.from_matrix(np.array([baseline[name][:3, :3] for name in frames])).as_quat()[:, [3, 0, 1, 2]]
        positions = torch.tensor(base_p[None, None]+translations[:, None, None], device=device, dtype=torch.float32)
        rotations = torch.tensor(np.broadcast_to(base_r, (8, 1)+base_r.shape).copy(), device=device, dtype=torch.float32)
        progress = torch.zeros((8, 1), device=device)
        original_chunk = scene_module._coverage_plane_chunk_size
        original_max = scene_module._coverage_halfspace_max

        def old_halfspace_max(points, halfspaces):
            normal, offset = halfspaces[:, :3], halfspaces[:, 3]
            signed = (points @ normal.T + offset)/torch.linalg.vector_norm(normal, dim=-1)
            return signed.amax(dim=-1)

        results = {}
        for method in ("fixed_32_original_amax", "bounded_batch_argmax"):
            chunk = ((lambda states, points, planes: min(32, planes))
                     if method == "fixed_32_original_amax" else original_chunk)
            reduce = old_halfspace_max if method == "fixed_32_original_amax" else original_max
            times, peaks = [], []
            with patch.object(scene_module, "_coverage_plane_chunk_size", chunk), \
                    patch.object(scene_module, "_coverage_halfspace_max", reduce):
                for repeat in range(4):
                    p, r = positions.clone().requires_grad_(), rotations.clone().requires_grad_()
                    poses = {name: (p[:, :, i], r[:, :, i]) for i, name in enumerate(frames)}
                    torch.cuda.synchronize(device)
                    torch.cuda.reset_peak_memory_stats(device)
                    start = time.perf_counter()
                    value = scene.robot_coverage_guidance(poses, progress)
                    gradient = torch.autograd.grad(value.sum(), (p, r))
                    torch.cuda.synchronize(device)
                    elapsed = time.perf_counter()-start
                    if not all(torch.isfinite(t).all().item() for t in (value, *gradient)):
                        raise ValueError("Guidance value or derivative is not finite")
                    if repeat:
                        times.append(elapsed)
                        peaks.append(torch.cuda.max_memory_allocated(device))
                results[method] = tuple(t.detach().cpu().numpy() for t in (value, *gradient))
            report["methods"][method] = {"states": 8, "warmup_calls": 1, "measured_calls": 3,
                "forward_backward_seconds": times, "median_seconds": float(np.median(times)),
                "maximum_torch_allocated_bytes": max(peaks), "finite_values_and_gradients": True,
                "point_plane_chunks_per_objective": sum(math.ceil(n/chunk(8, len(p), n))
                    for p in scene._coverage_samples for n in counts),
                "values": results[method][0].tolist()}
        old, new = (results[name] for name in report["methods"])
        report["maximum_absolute_differences"] = dict(zip(("value", "position_gradient", "quaternion_gradient"),
            (float(np.max(np.abs(a-b))) for a, b in zip(old, new))))
        # FP32 GEMM reduction order can vary with chunk dimensions. Compare
        # values tightly; ties may legitimately choose another subgradient.
        if not np.allclose(old[0], new[0], atol=1e-6, rtol=1e-6):
            raise ValueError("Batching changed the numerical guidance values beyond FP32 tolerance")
        report["median_speedup"] = (report["methods"]["fixed_32_original_amax"]["median_seconds"]
                                   / report["methods"]["bounded_batch_argmax"]["median_seconds"])
    report["maximum_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    report["status"] = "passed"
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({key: report[key] for key in ("status", "gpu_used", "scene_workspace_mib", "maximum_rss_mib")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("fixture", "feedback", "observed-free", "workspace-audit", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--static-owners", type=int, required=True)
    parser.add_argument("--hand-pairs", type=int, required=True)
    parser.add_argument("--initial-state-assumption-source", required=True)
    parser.add_argument("--cpu-only", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.static_owners <= 32 or not 0 <= args.hand_pairs <= 65536:
        parser.error("Explicit archived geometry counts exceed the bounded diagnostic")
    if args.output.exists():
        parser.error("Preserve prior reports; the output path must not exist")
    run(args)
# END no model, observation, optimization budget, or acceptance rule is changed.
