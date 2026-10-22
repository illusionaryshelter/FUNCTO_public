"""Bounded native GPU checks for zero-step history and physical-boundary recovery.

Run explicitly in the existing recovery environment. The objective is analytic,
with 16 cubic controls and seven coordinates; no robot, scene or model is loaded.
The probe compares the same guarded implementation with and without CUDA graphs.
"""

# BEGIN explicit small fused-kernel/graph lifecycle probe; no import-time CUDA.
import argparse
import json
from pathlib import Path
import resource
import sys
from types import SimpleNamespace


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.output.exists():
        raise FileExistsError(arguments.output)
    import torch
    from curobo import runtime
    from curobo._src.curobolib.cuda_ops.optimization import LBFGScu
    from curobo._src.optim.components.quasi_newton_buffers import QuasiNewtonBuffers
    from curobo._src.optim.gradient.lbfgs import LBFGSOpt, LBFGSOptCfg
    from curobo._src.solver.manager_goal import GoalManager
    from curobo._src.types.device_cfg import DeviceCfg
    from manipulation.curobo_solver_components import bind_solver_components
    from manipulation.physical_joint_domain import CubicJointDomain

    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError("This explicit probe requires CUDA access")
    if not runtime.cuda_graphs:
        raise RuntimeError("The explicitly requested graph probe requires the native graph runtime")
    device = DeviceCfg("cuda:0")
    total = torch.cuda.get_device_properties(device.device).total_memory
    torch.cuda.set_per_process_memory_fraction(min(.5, 512*1024**2/total), device.device)
    torch.cuda.reset_peak_memory_stats(device.device)
    output = {}

    # Observe what the installed fused kernel actually does at zero s/y.
    # This is a diagnostic, not an assumed explanation of an earlier solve.
    history = QuasiNewtonBuffers(device, 27)
    history.resize(1, 112)
    q = torch.zeros((1, 112), device=device.device)
    gradient = torch.linspace(1., 1e5, 112, device=device.device).reshape(1, 1, 112)
    history.set_reference(q[..., None], gradient.transpose(-1, -2))
    raw = LBFGScu.apply(history.step_q_buffer, history.rho, history.y, history.s,
        q, gradient, history.x_0, history.grad_0, .01, True, True)
    output["native_fused_zero_step"] = {
        "dimension": 112, "history": 27,
        "direction_over_negative_gradient": float((-raw/gradient.reshape_as(raw)).median()),
        "history_displacement_max": float(history.s.abs().max()),
        "history_gradient_change_max": float(history.y.abs().max()),
    }
    del history

    class Objective:
        def __init__(self, domain):
            self.domain = domain
            self.action_horizon, self.action_dim = 16, 7
            self.action_bound_lows, self.action_bound_highs = domain.lower.float(), domain.upper.float()
            self.target = torch.full((7,), .5, device=device.device)
            self.target[0] = 4.

        def update_batch_size(self, batch_size):
            self.batch_size = batch_size

        def evaluate_action(self, action, use_cuda_graph=False):
            self.domain.validate(action)
            costs = ((action-self.target)**2).sum(dim=(-1, -2))
            return SimpleNamespace(costs_and_constraints=SimpleNamespace(
                get_sum_cost_and_constraint=lambda sum_horizon: costs))

    def construct(graph):
        # BEGIN numerical recovery declares the same four model derivatives as the domain.
        bounds = torch.tensor([[-1., 1.], [-10., 10.], [-100., 100.], [-1000., 1000.]],
                              device=device.device)[:, :, None].expand(-1, -1, 7)
        domain = CubicJointDomain(bounds,
            torch.zeros(7, device=device.device), torch.zeros(7, device=device.device), .2, 16)
        # END one declared domain serves both eager and captured native updates.
        optimizer = LBFGSOpt(LBFGSOptCfg(device_cfg=device, num_iters=8, inner_iters=2,
            history=27, line_search_scale=[0., .1, .5, 1.], step_scale=.98,
            use_cuda_kernel_step_direction=True, use_cuda_kernel_line_search=True,
            use_cuda_kernel_shared_buffers=True, initial_step_scale=.001, fixed_iters=True),
            [Objective(domain), Objective(domain)])
        solver = SimpleNamespace(core=SimpleNamespace(device_cfg=device,
            config=SimpleNamespace(use_cuda_graph=False), optimizers=[optimizer],
            goal_registry_manager=GoalManager(device)))
        bind_solver_components(solver, domain, use_optimizer_graph=graph)
        return optimizer, domain

    records = []
    for graph in (False, True):
        optimizer, domain = construct(graph)
        core, history = optimizer._core, optimizer._qn
        seed = torch.zeros((1, 16, 7), device=device.device)
        seed[..., 0] = 1.
        optimizer.reinitialize(seed)
        # The protected callback must skip zero s even with valid older pairs.
        history.s.fill_(.01)
        history.y.fill_(.02)
        history.rho.fill_(1.)
        before = {name: getattr(history, name).clone() for name in ("s", "y", "rho", "alpha")}
        direction = core._guarded_step_direction(core._iteration_state)
        torch.testing.assert_close(direction, -.001*core._iteration_state.exploration_gradient, rtol=0, atol=0)
        for name, saved in before.items():
            torch.testing.assert_close(getattr(history, name), saved, rtol=0, atol=0)
        optimizer.reinitialize(seed)
        result = optimizer.optimize(seed)
        domain.validate(result)
        report = core.solve_diagnostics[0]
        if report["recovery_accepted"] < 1 or not bool((result[..., 1:] != seed[..., 1:]).any()):
            raise AssertionError("A blocked boundary coordinate prevented recovery of free coordinates")
        records.append({"result": result.detach().cpu(), "report": report,
                        "history": {name: getattr(history, name).detach().cpu().clone()
                                    for name in ("s", "y", "rho", "x_0", "grad_0")}})
        output["graph" if graph else "eager"] = report
        core.reset_cuda_graph()
        del optimizer, core, domain, history
    torch.testing.assert_close(records[0]["result"], records[1]["result"], rtol=0, atol=0)
    for name in records[0]["history"]:
        torch.testing.assert_close(records[0]["history"][name], records[1]["history"][name], rtol=0, atol=0)
    if records[0]["report"] != records[1]["report"]:
        raise AssertionError("Capture warmup or replay changed the recovery lifecycle")
    output.update(passed=True, graph_eager_equal=True,
        peak_torch_gpu_bytes=torch.cuda.max_memory_allocated(device.device),
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(output, indent=2)+"\n", encoding="utf8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
# END the probe never loads a scene or changes installed native libraries.
