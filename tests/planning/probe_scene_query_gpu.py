"""Bounded actual cuRobo sphere query and rigid-transform gradient experiment.

This uses one synthetic cuboid, at most two spheres and no robot optimizer.
It validates the production query against an analytic box on CUDA. It does
not certify RGB-D reconstruction, full robot geometry, or physical contact.
"""

# BEGIN bounded native collision and attachment-transform gradient experiment.
import argparse
import json
from pathlib import Path
import resource
import sys
import time
import traceback

# BEGIN resolve imports from this FUNCTO checkout, independent of its directory name.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
# END project-local imports.


def run(output):
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    report = {"scope": "synthetic actual cuRobo V2 cuboid query and rigid transforms",
              "production_environment_validation": False, "status": "running"}
    try:
        import torch
        from curobo._src.geom.collision.collision_scene import SceneCollision, SceneCollisionCfg
        from curobo._src.geom.types import Cuboid, SceneCfg
        from curobo.types import DeviceCfg
        from manipulation.planning_scene import CuroboSphereQuery, _local_spheres, _world_spheres

        if not torch.cuda.is_available():
            raise RuntimeError("This experiment requires explicitly authorized CUDA access")
        torch.set_num_threads(2)
        torch.cuda.set_per_process_memory_fraction(512*1024**2/torch.cuda.get_device_properties(0).total_memory)
        torch.cuda.reset_peak_memory_stats()
        config = DeviceCfg(device="cuda:0")
        native = SceneCollision.from_config(SceneCollisionCfg(
            device_cfg=config, scene_model=SceneCfg(cuboid=[Cuboid(
                name="observed_test_box", pose=[0., 0., 0., 1., 0., 0., 0.], dims=[.2, .3, .4])]),
            cache={"cuboid": 1}))
        query = CuroboSphereQuery(native, owner_ids=("test_box",), source="synthetic analytic box",
                                 max_query_spheres=2, max_buffers=8, max_cache_bytes=4096)
        balls = config.to_device([[[[.14, .02, 0., .05], [.25, .02, 0., .05]]]]).requires_grad_()
        value = query(balls, key="clearance")
        # The native activation-zero output is penetration cost. Report the
        # signed analytic clearances separately; zero cost is not a distance.
        torch.testing.assert_close(value, value.new_tensor([[[.01, 0.]]]), atol=2e-6, rtol=0)
        value.sum().backward()
        torch.testing.assert_close(balls.grad[..., 0], value.new_tensor([[[-1., 0.]]]), atol=2e-5, rtol=0)
        report["clearance"] = {"analytic_signed_clearance_m": [-.01, .10],
                               "native_penetration_cost_m": value.detach().cpu().tolist(),
                               "native_x_gradient": balls.grad[..., 0].cpu().tolist()}

        # Each independent evaluation builds its own graph; the production
        # wrapper must also preserve gradients across repeated kernel slots.
        x1 = config.to_device([[[[.14, .02, 0., .05]]]]).requires_grad_()
        x2 = config.to_device([[[[.02, .19, 0., .05]]]]).requires_grad_()
        first, second = query(x1, key="reused"), query(x2, key="reused")
        (2*first + 3*second).sum().backward()
        torch.testing.assert_close(x1.grad[..., :3], x1.new_tensor([[[[-2., 0., 0.]]]]), atol=2e-5, rtol=0)
        torch.testing.assert_close(x2.grad[..., :3], x2.new_tensor([[[[0., -3., 0.]]]]), atol=2e-5, rtol=0)
        report["reused_workspace_gradients"] = [x1.grad.cpu().tolist(), x2.grad.cpu().tolist()]

        def quaternion(angle):
            zero = angle*0
            return torch.stack((torch.cos(angle/2), zero, zero, torch.sin(angle/2))).reshape(1, 1, 4)

        def owner_cost(parameters):
            tx, angle = parameters.unbind()
            position = torch.stack((tx, tx*0, tx*0)).reshape(1, 1, 3)
            world = config.to_device([[[[.16, .02, 0., .05]]]])
            return query(_local_spheres(world, position, quaternion(angle)), key="owner").sum()

        def attached_cost(parameters):
            tx, angle = parameters.unbind()
            position = torch.stack((tx, tx*0, tx*0)).reshape(1, 1, 3)
            payload = _world_spheres([[.12, .02, 0., .05]], position, quaternion(angle))
            return query(payload, key="attached").sum()

        for label, objective, expected in (("owner_inverse_transform", owner_cost, [1., -.02]),
                                            ("attached_tool_transform", attached_cost, [-1., .02])):
            parameters = config.to_device([.02, 0.]).requires_grad_()
            cost = objective(parameters)
            gradient, = torch.autograd.grad(cost, parameters)
            finite = []
            for index in range(2):
                delta = torch.zeros_like(parameters)
                delta[index] = 1e-4
                finite.append(float(((objective(parameters.detach()+delta)
                                      - objective(parameters.detach()-delta))/(2e-4)).cpu()))
            torch.testing.assert_close(gradient, gradient.new_tensor(expected), atol=3e-5, rtol=0)
            torch.testing.assert_close(gradient, gradient.new_tensor(finite), atol=2e-4, rtol=0)
            report[label] = {"cost_m": float(cost.detach().cpu()),
                             "gradient": gradient.cpu().tolist(), "central_difference": finite}
        torch.cuda.synchronize()
        report.update(status="passed", torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      torch_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                      bounded_query_cache_bytes=query._cache_bytes)
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
# END bounded native collision and attachment-transform gradient experiment.
