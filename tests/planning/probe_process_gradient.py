"""Measure the native first search direction without changing production knobs.

This bounded synthetic probe reuses the production planner experiment.  All
rollout evaluations are sequential so native FK work buffers remain valid.
"""

# BEGIN independent finite-difference and line-search scale experiment.
import argparse
import json
from pathlib import Path
from unittest.mock import patch

import probe_process_planner as experiment


def run(output):
    import torch
    from manipulation import planning
    from curobo._src.optim.gradient.line_search_strategy import LineSearchStrategy

    original = planning._CuroboPlanner.optimize

    def measured(planner, seed):
        # Native solve establishes each line-search particle's goal mapping.
        initial = seed.detach().clone()
        result = original(planner, seed)
        rollout = planner.trajopt.optimizer_rollouts[0]
        optimizer = planner.trajopt.optimizer.optimizers[0]
        particles = optimizer.config.num_particles
        x = initial.reshape(-1, initial.shape[-2], initial.shape[-1]).repeat_interleave(particles, 0)

        def objective(value, gradient=False):
            value = value.detach().clone().requires_grad_(gradient)
            evaluated = rollout.evaluate_action(value, use_cuda_graph=False)
            collection = evaluated.costs_and_constraints
            loss = collection.get_sum_cost_and_constraint(sum_horizon=True).sum()
            derivative = torch.autograd.grad(loss, value)[0].detach().clone() if gradient else None
            return float(loss.detach().cpu()), derivative

        baseline, gradient = objective(x, True)
        # Use exactly the library's initialization and clipping expression.
        step = -optimizer.config.initial_step_scale * gradient
        maximum = optimizer.action_step_max
        direction = LineSearchStrategy.scale_action(
            step, maximum, optimizer.config.step_scale,
            optimizer.config.fix_terminal_action, step.shape[1])
        # BEGIN use the same physical-domain restriction as native line search.
        # A diagnostic rollout must not present an out-of-domain trial as a
        # direction that the production optimizer would actually evaluate.
        alphas = (0., 1e-5, 1e-4, 1e-3, .01, .1, .5, 1.)
        direction = planner.physical_domain.limit_step(x, direction,
            x.new_tensor(alphas).reshape(1, len(alphas), 1, 1))
        # END this preserves the common scale ratios without clipping q(t).
        report = {"baseline": baseline, "gradient_max": float(gradient.abs().max().cpu()),
                  "raw_step_max": float(step.abs().max().cpu()),
                  "scaled_step_max": float(direction.abs().max().cpu()),
                  "search": []}
        for alpha in alphas:
            loss, _ = objective(x + alpha * direction)
            report["search"].append({"alpha": alpha, "loss": loss})
        # BEGIN only defined, feasible symmetric probes support differences.
        unit = gradient / gradient.abs().max().clamp_min(torch.finfo(gradient.dtype).eps)
        for epsilon in (1e-4, 1e-5):
            if not bool(planner.physical_domain.feasible(torch.stack((x+epsilon*unit, x-epsilon*unit))).all()):
                report[f"directional_derivative_{epsilon}"] = {
                    "status": "not_evaluated", "reason": "Symmetric difference leaves the physical joint domain"}
                continue
            upper, _ = objective(x + epsilon*unit)
            lower, _ = objective(x - epsilon*unit)
            report[f"directional_derivative_{epsilon}"] = (upper-lower)/(2*epsilon)
        # END absent finite-difference evidence is never replaced by clipping.
        report["analytic_directional_derivative"] = float((gradient*unit).sum().cpu())
        output.mkdir(parents=True, exist_ok=True)
        (output / "gradient.json").write_text(json.dumps(report, indent=2)+"\n")
        print(json.dumps(report), flush=True)
        return result

    # BEGIN keep diagnostic instrumentation local to this invocation.
    # Imported probe helpers can be called again in the same process. Restore
    # the exact prior method even if the experiment or observer raises.
    with patch.object(planning._CuroboPlanner, "optimize", new=measured):
        experiment.run(output, False)
    # END scoped diagnostic instrumentation.


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
# END independent finite-difference and line-search scale experiment.
