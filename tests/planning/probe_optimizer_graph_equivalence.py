"""Bounded bytewise check of the archived eager core and owned native-step graph.

Run explicitly in recovery; importing this file creates no device context. The
probe exercises native L-BFGS with four coordinates and eight cubic controls,
not a robot planner. It retains each native iteration's action, cost, gradient,
best/exploration fields and quasi-Newton buffers so an unchanged final success
flag cannot hide a numerical difference earlier in the solve.

Both source snapshots must expose the same four-derivative physical domain.
The explicit reference directory names that arithmetic baseline; historical
position-only snapshots describe a different optimization problem.
"""

# BEGIN finite optimizer-only differential probe; no simulation or model loading.
import argparse
from contextlib import nullcontext
from dataclasses import fields
import importlib.util
import json
from pathlib import Path
import resource
import sys
import time
from types import MethodType, ModuleType, SimpleNamespace
from unittest.mock import patch


PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))


def load_reference(directory):
    package = ModuleType("optimizer_probe_reference")
    package.__path__ = [str(PROJECT / "manipulation")]
    sys.modules[package.__name__] = package
    modules = []
    for name in ("physical_joint_domain", "curobo_solver_components"):
        qualified = f"{package.__name__}.{name}"
        specification = importlib.util.spec_from_file_location(
            qualified, directory / f"reference_{name}.py")
        module = importlib.util.module_from_spec(specification)
        sys.modules[qualified] = module
        specification.loader.exec_module(module)
        modules.append(module)
    return modules


def bytes_equal(first, second, label):
    import torch

    if first.shape != second.shape or first.dtype != second.dtype:
        raise AssertionError(f"{label}: shape/dtype changed")
    a = first.detach().contiguous().cpu()
    b = second.detach().contiguous().cpu()
    a_bytes, b_bytes = a.reshape(-1).view(torch.uint8), b.reshape(-1).view(torch.uint8)
    differences = (a_bytes != b_bytes).nonzero().flatten()
    if differences.numel():
        byte = int(differences[0])
        index = byte // a.element_size()
        raise AssertionError(f"{label}: first changed byte {byte}, value {index}: "
                             f"{a.reshape(-1)[index].item()!r} -> {b.reshape(-1)[index].item()!r}")


def state_tensors(state):
    import torch

    return {field.name: getattr(state, field.name).detach().clone()
            for field in fields(state) if isinstance(getattr(state, field.name), torch.Tensor)}


def history_tensors(optimizer):
    return {name: getattr(optimizer._qn, name).detach().clone()
            for name in ("s", "y", "rho", "x_0", "grad_0", "alpha", "step_q_buffer")}


def tracker_tensors(optimizer):
    return {name: getattr(optimizer._core._best, name).detach().clone()
            for name in ("cost", "action", "iteration", "current_iteration",
                         "previous_step_direction", "converged")}


def observe_iterations(optimizer):
    """Add only copying kernels to both versions; do not alter their arithmetic."""
    core = optimizer._core
    original_record = core._record_iteration_state
    original_iters = core._opt_iters
    original_dispatch = core._dispatch
    core.probe_iterations = []
    core.probe_blocks = []
    core.probe_recording_iterations = False

    def record(self, state):
        original_record(state)
        # The native outer optimizer also records its entry state. Observe
        # only iterations of the unchanged inner loop, after each eager step
        # or graph replay, so both traces have the same temporal meaning.
        if self.probe_recording_iterations:
            self.probe_iterations.append(state_tensors(state))

    def iters(self, state):
        self.probe_iterations = []
        self.probe_recording_iterations = True
        try:
            return original_iters(state)
        finally:
            self.probe_recording_iterations = False

    def dispatch(self, method_name, *args):
        result = original_dispatch(method_name, *args)
        if method_name == "_opt_iters":
            # Native recording runs after every step replay. Preserve this
            # block's clones before the next block resets the observation list.
            self.probe_blocks.append({
                "iterations": [{key: value.clone() for key, value in item.items()}
                               for item in self.probe_iterations],
                "history": history_tensors(optimizer),
                "tracker": tracker_tensors(optimizer),
                "result": state_tensors(result),
            })
        return result

    core._record_iteration_state = MethodType(record, core)
    core._opt_iters = MethodType(iters, core)
    core._dispatch = MethodType(dispatch, core)
    # Install the current core's executor grain; CPU/reference allocate none.
    core.finish_init()


def compare_tree(first, second, label):
    import torch

    if isinstance(first, torch.Tensor):
        bytes_equal(first, second, label)
    elif isinstance(first, dict):
        if first.keys() != second.keys():
            raise AssertionError(f"{label}: fields changed")
        for key in first:
            compare_tree(first[key], second[key], f"{label}/{key}")
    elif isinstance(first, list):
        if len(first) != len(second):
            raise AssertionError(f"{label}: count changed {len(first)} -> {len(second)}")
        for index, (a, b) in enumerate(zip(first, second)):
            compare_tree(a, b, f"{label}/{index}")
    elif first != second:
        raise AssertionError(f"{label}: metadata changed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    parser.add_argument("--reference-dir", type=Path, required=True)
    arguments = parser.parse_args()
    import torch
    from curobo import runtime
    from curobo._src.optim.gradient.lbfgs import LBFGSOpt, LBFGSOptCfg
    from curobo._src.solver.manager_goal import GoalManager
    from curobo._src.types.device_cfg import DeviceCfg
    from manipulation import physical_joint_domain as updated_domain
    from manipulation import curobo_solver_components as updated_components

    torch.set_num_threads(1)
    device = DeviceCfg(arguments.device)
    use_graph = device.device.type == "cuda"
    if use_graph:
        if not torch.cuda.is_available():
            raise RuntimeError("The explicitly requested optimizer probe needs CUDA access")
        if not runtime.cuda_graphs:
            raise ValueError("The probe does not override a disabled native CUDA graph runtime")
        total = torch.cuda.get_device_properties(device.device).total_memory
        torch.cuda.set_per_process_memory_fraction(min(.5, 2*1024**3/total), device.device)
        torch.cuda.reset_peak_memory_stats(device.device)
    old_domain, old_components = load_reference(arguments.reference_dir)
    lower = torch.tensor([-1., -.8, .1, -2.], device=device.device)
    upper = torch.tensor([1., .9, 1., 3.], device=device.device)
    q0 = torch.tensor([.2, .1, .4, .0], device=device.device)
    dq0 = torch.tensor([.03, -.02, .01, 0.], device=device.device)
    # BEGIN both implementations receive the same explicit physical derivatives.
    derivative_bounds = torch.tensor([[-10., 10.], [-100., 100.], [-1000., 1000.]], device=device.device)
    physical_bounds = torch.cat((torch.stack((lower, upper))[None],
        derivative_bounds[:, :, None].expand(-1, -1, 4)))
    # END the probe compares execution mode under one unchanged model domain.

    class Objective:
        def __init__(self, domain, horizon):
            self.domain = domain
            self.action_horizon, self.action_dim = horizon, 4
            self.action_bound_lows, self.action_bound_highs = lower, upper
            self.target = upper + .25

        def update_batch_size(self, batch_size):
            self.batch_size = batch_size

        def evaluate_action(self, action, use_cuda_graph=False):
            # Independent observation at the evaluated-action boundary. The
            # updated domain records failures in the device scope; eager old
            # code raises immediately. No valid input is numerically changed.
            valid = self.domain.validate(action)
            if valid is not None:
                action = torch.where(valid, action, q0.expand_as(action))
            costs = ((action-self.target)**2).sum(dim=(-1, -2))
            return SimpleNamespace(costs_and_constraints=SimpleNamespace(
                get_sum_cost_and_constraint=lambda sum_horizon: costs))

    def construct(module, components, horizon):
        domain = (module.JointPositionDomain(lower, upper) if horizon == 1 else
            module.CubicJointDomain(physical_bounds, q0, dq0, .2, horizon))
        rollouts = [Objective(domain, horizon) for _ in range(2)]
        optimizer = LBFGSOpt(LBFGSOptCfg(device_cfg=device, num_iters=4, inner_iters=2,
            history=4, line_search_scale=[0., .2, 1.], step_scale=.98,
            use_cuda_kernel_step_direction=use_graph, use_cuda_kernel_line_search=use_graph,
            use_cuda_kernel_shared_buffers=True, fixed_iters=True), rollouts)
        solver = SimpleNamespace(core=SimpleNamespace(device_cfg=device,
            config=SimpleNamespace(use_cuda_graph=False), optimizers=[optimizer],
            goal_registry_manager=GoalManager(device)))
        kwargs = {"use_optimizer_graph": use_graph} if components is updated_components else {}
        components.bind_solver_components(solver, domain, **kwargs)
        observe_iterations(optimizer)
        return solver, optimizer, domain, rollouts

    begin = time.perf_counter()
    completed = []
    # CPU mode is an arithmetic/lifecycle preflight only and never asks the
    # native CUDA event timer to initialize a CUDA context.
    timer_scope = patch("curobo._src.runtime.cuda_event_timers", False) if not use_graph else nullcontext()
    with timer_scope:
        for horizon in (1, 8):
            original = construct(old_domain, old_components, horizon)
            updated = construct(updated_domain, updated_components, horizon)
            for case, (count, offset) in enumerate(((2, 0.), (2, .03), (1, .01), (2, -.02))):
                outputs = []
                for solver, optimizer, domain, rollouts in (original, updated):
                    if optimizer.config.num_problems != count:
                        optimizer.update_num_problems(count)
                    seed = q0.repeat(count, horizon, 1) + offset
                    optimizer._core.probe_blocks = []
                    optimizer.reinitialize(seed)
                    initial = state_tensors(optimizer._core._iteration_state)
                    result = optimizer.optimize(seed)
                    outputs.append({"initial": initial, "result": result.detach().clone(),
                                    "blocks": optimizer._core.probe_blocks,
                                    "history": history_tensors(optimizer),
                                    "tracker": tracker_tensors(optimizer)})
                compare_tree(*outputs, f"horizon={horizon}/case={case}")
                completed.append({"horizon": horizon, "batch": count, "offset": offset,
                                  "bytewise_equal": True})
            # The normal reinitialize path prepares its initial state eagerly.
            # Also exercise the distinct native initial-state graph, retaining
            # existing quasi-Newton history exactly as a direct optimize does.
            for index, record in enumerate((original, updated)):
                _, optimizer, _, _ = record
                optimizer._core.probe_blocks = []
                result = optimizer.optimize(q0.repeat(2, horizon, 1)+.01)
                outputs[index] = {"result": result.detach().clone(),
                    "blocks": optimizer._core.probe_blocks, "history": history_tensors(optimizer),
                    "tracker": tracker_tensors(optimizer)}
            compare_tree(*outputs, f"horizon={horizon}/direct-optimize")
            completed.append({"horizon": horizon, "without_reinitialize": True, "bytewise_equal": True})
            # Rebind a changed physical interval with the same tensor shapes.
            for record, module, components in ((original, old_domain, old_components),
                                               (updated, updated_domain, updated_components)):
                solver, optimizer, _, rollouts = record
                changed_upper = upper-.1
                changed_bounds = physical_bounds.clone()
                changed_bounds[0, 1] = changed_upper
                domain = (module.JointPositionDomain(lower, changed_upper) if horizon == 1 else
                    module.CubicJointDomain(changed_bounds, q0, dq0, .2, horizon))
                kwargs = {"use_optimizer_graph": use_graph} if components is updated_components else {}
                components.bind_solver_components(solver, domain, **kwargs)
                for rollout in rollouts:
                    rollout.domain = domain
                optimizer._core.probe_blocks = []
                seed = q0.repeat(2, horizon, 1)
                optimizer.reinitialize(seed)
                result = optimizer.optimize(seed)
                outputs[0 if record is original else 1] = {"result": result.detach().clone(),
                    "blocks": optimizer._core.probe_blocks, "history": history_tensors(optimizer),
                    "tracker": tracker_tensors(optimizer)}
            compare_tree(*outputs, f"horizon={horizon}/rebound-domain")
            completed.append({"horizon": horizon, "rebound_domain": True, "bytewise_equal": True})
            for _, optimizer, _, _ in (original, updated):
                try:
                    optimizer.reinitialize(torch.full((2, horizon, 4), float("nan"), device=device.device))
                except ValueError as error:
                    if type(error).__name__ != "JointDomainInitializationError":
                        raise
                else:
                    raise AssertionError("An invalid seed entered an optimizer")
            del original, updated, outputs

    peak_gpu = torch.cuda.max_memory_allocated(device.device) if use_graph else 0
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    report = {"device": arguments.device, "native_step_graph": use_graph,
              "fault_check_boundary": "once per original native inner-iteration block",
              "comparison": "identical tensor bytes including signed zero; no tolerances",
              "cases": completed, "seconds": time.perf_counter()-begin,
              "peak_process_rss_bytes": peak_rss, "peak_torch_gpu_bytes": peak_gpu,
              "no_simulation": True, "native_runtime_cuda_graphs": bool(runtime.cuda_graphs)}
    if max(peak_rss, peak_gpu) > 2*1024**3:
        raise RuntimeError(f"The bounded probe exceeded its 2 GiB resource allocation: {report}")
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
# END this probe does not authorize enabling a production scene's graph scope.
