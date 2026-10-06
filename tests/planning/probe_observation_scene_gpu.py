"""Small real cuRobo RGB-D mapper to production planning-scene experiment.

One 64x64 synthetic depth plane has a selected central owner patch. The two
native partitions share raw observation coverage and use explicit calibrated
frames. This is an API/evidence counterexample, not a real-scene certificate.
"""

# BEGIN bounded native RGB-D observation and partition query experiment.
import argparse
import json
from pathlib import Path
import resource
import sys
import time
import traceback

import numpy as np

# BEGIN resolve imports from this FUNCTO checkout, independent of its directory name.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
# END project-local imports.


def run(output):
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    report = {"scope": "synthetic current RGB-D, actual Mapper and owner-separated SceneCollision",
              "production_environment_validation": False, "status": "running"}
    try:
        import torch
        from curobo.perception import MapperCfg
        from manipulation.planning_observation import build_scene_from_rgbd
        from manipulation.planning_scene import _local_spheres

        if not torch.cuda.is_available():
            raise RuntimeError("This experiment requires explicitly authorized CUDA access")
        torch.set_num_threads(2)
        torch.cuda.set_per_process_memory_fraction(1024**3/torch.cuda.get_device_properties(0).total_memory)
        torch.cuda.reset_peak_memory_stats()
        mask = np.zeros((64, 64), bool)
        mask[22:42, 22:42] = True
        observation = {"image": np.zeros((64, 64, 3), np.uint8), "depth": np.ones((64, 64)),
                       "K": np.array([[48., 0., 31.5], [0., 48., 31.5], [0., 0., 1.]]),
                       "T_output_camera": np.eye(4), "reference_frame": "camera0_current",
                       "owner_masks": {"surface_patch": mask}, "robot_mask": np.zeros((64, 64), bool)}
        owner_pose = np.eye(4)
        owner_pose[2, 3] = 1.
        config = MapperCfg(extent_meters_xyz=(1.28, 1.28, 1.28), voxel_size=.04,
                           extent_esdf_meters_xyz=(1.28, 1.28, 1.28), esdf_voxel_size=.04,
                           grid_center=torch.tensor([0., 0., .8]), truncation_distance=.08,
                           image_height=64, image_width=64, depth_minimum_distance=.1,
                           depth_maximum_distance=2., device="cuda:0")
        scene = build_scene_from_rgbd(
            {"plane_now": observation}, {"surface_patch": {"T_output_owner": owner_pose,
                                                             "source": "current plane depth and fixed observed frame"}},
            mapper_cfg=config, depth_error_m=.005, reference_frame="camera0_current",
            robot_sphere_materials=(None,), robot_geometry_source="synthetic probe sphere",
            # BEGIN synthetic owners use the same explicit per-owner API.
            has_gripper=False, owner_voxel_sizes_m={"surface_patch": .04},
            max_mapping_bytes=256*1024**2,
            max_host_rss_bytes=8*1024**3, max_device_used_bytes=6*1024**3)
            # END the analytic patch does not infer resolution from its name.
        report["observation_build"] = scene.observation_report
        if scene.owners[0].geometry_complete:
            raise AssertionError("Observed surface was incorrectly declared complete")
        points = torch.tensor([[[[0., 0., .913, .12], [.45, 0., .913, .12]]]], device="cuda:0", requires_grad=True)
        background = scene.background(points, key="partition_probe")
        p, q = scene.owners[0].predicted_pose(torch.tensor([[0.]], device="cuda:0"))
        owned = scene.owners[0].query(_local_spheres(points, p, q), key="partition_probe")
        report["partition_costs"] = {"background": background.detach().cpu().tolist(),
                                      "selected_owner": owned.detach().cpu().tolist()}
        if not (float(owned[0, 0, 0].detach()) > .005 and float(background[0, 0, 1].detach()) > .005):
            raise AssertionError("Each observed surface must be present in its own native field")
        if not (float(background[0, 0, 0].detach()) < 1e-5 and float(owned[0, 0, 1].detach()) < 1e-5):
            raise AssertionError("Selected owner pixels leaked across native geometry partitions")
        owned.sum().backward()
        derivative = points.grad[0, 0, 0, 2]
        if not torch.isfinite(points.grad).all() or float(derivative) <= .1:
            raise AssertionError("Mapped owner query did not backpropagate through its calibrated transform")
        finite = []
        for offset in (-.002, .002):
            shifted = points.detach().clone()
            shifted[..., 2] += offset
            finite.append(float(scene.owners[0].query(_local_spheres(shifted, p, q), key="partition_probe")[0, 0, 0]))
        central = (finite[1]-finite[0])/.004
        if abs(float(derivative)-central) > .01:
            raise AssertionError("Native mapped plane gradient disagrees with central finite difference")
        report["owner_z_gradient"] = {"native": float(derivative), "central_difference": central}
        balls = torch.tensor([[[[0., 0., .5, .025], [0., 0., 1.2, .025]]]], device="cuda:0")
        coverage = scene.coverage.violation(balls)
        if not (float(coverage[0, 0, 0]) == 0. and float(coverage[0, 0, 1]) > 0.):
            raise AssertionError("Original depth ray coverage must preserve the hidden back as unknown")
        report["coverage_violation"] = coverage.cpu().tolist()
        torch.cuda.synchronize()
        report.update(status="passed", torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      torch_peak_reserved_bytes=torch.cuda.max_memory_reserved())
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        (output / "exception.txt").write_text(traceback.format_exc())
        raise
    finally:
        report.update(elapsed_seconds=time.monotonic()-started,
                      max_cpu_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        (output / "report.json").write_text(json.dumps(report, indent=2)+"\n")
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
# END bounded native RGB-D observation and partition query experiment.
