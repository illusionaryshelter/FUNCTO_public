"""Compare the production curve objective/constraint derivatives numerically.

This bounded synthetic probe uses the same evaluate interface as SLSQP.
Infeasible trial controls are evaluated and reported; they are never clipped
or promoted to executable candidates by this diagnostic.
"""

# BEGIN independent differences on the current constrained-curve interface.
import argparse
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

import probe_process_planner as experiment


def gradient_diagnostics(solver, controls):
    """Check one deterministic direction against the actual signed rows.

    A mixed direction also tests constraint derivatives at a stationary task
    objective. This is a finite-difference experiment, not an attempted copy
    of SLSQP's internal search direction or a new acceptance rule.
    """
    controls = np.asarray(controls, dtype=np.float64)
    baseline = solver.evaluate(controls)
    direction = np.cos(np.arange(controls.size, dtype=np.float64)+.5)
    direction /= np.max(np.abs(direction))
    objective_derivative = float(baseline.objective_gradient @ direction)
    constraint_derivative = baseline.jacobian @ direction
    report = {"baseline_objective": baseline.objective,
        "baseline_violation": baseline.violation, "baseline_accepted": baseline.accepted,
        "objective_gradient_max": float(np.abs(baseline.objective_gradient).max()),
        "direction": direction.tolist(), "analytic_objective_derivative": objective_derivative,
        "constraint_names": list(baseline.names), "baseline_slack": baseline.slack.tolist(),
        "analytic_constraint_derivative": constraint_derivative.tolist(),
        "fixed_constraints": dict(zip(baseline.fixed_names, baseline.fixed_slack.tolist())),
        "physical_residuals": baseline.physical.raw_residuals,
        "curve_duration_s": solver.physical.duration,
        "curve_saved_node_count": len(solver.physical.times), "differences": []}
    direction = direction.reshape(controls.shape)
    for epsilon in (1e-4, 1e-5):
        positive, negative = (solver.evaluate(controls+sign*epsilon*direction) for sign in (1., -1.))
        objective = (positive.objective-negative.objective)/(2*epsilon)
        constraint = (positive.slack-negative.slack)/(2*epsilon)
        report["differences"].append({"epsilon": epsilon,
            "objective_derivative": float(objective),
            "objective_absolute_error": abs(float(objective)-objective_derivative),
            "constraint_derivative": constraint.tolist(),
            "constraint_absolute_error": np.abs(constraint-constraint_derivative).tolist(),
            "positive_violation": positive.violation, "negative_violation": negative.violation,
            "positive_accepted": positive.accepted, "negative_accepted": negative.accepted})
    return report


def run(output, *, fixture_path):
    from manipulation import planning

    original = planning._CuroboPlanner.optimize

    def measured(planner, seed):
        initial = seed.detach().cpu().numpy().copy()
        result = original(planner, seed)
        report = {"scope": "objective and signed physical constraints; diagnostic only",
            "seeds": [{"seed_index": index, **gradient_diagnostics(planner.trajectory_solver, controls)}
                      for index, controls in enumerate(initial[0])]}
        output.mkdir(parents=True, exist_ok=True)
        (output / "gradient.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
        print(json.dumps(report, allow_nan=False), flush=True)
        return result

    # Scoped instrumentation restores the exact prior method on exceptions.
    with patch.object(planning._CuroboPlanner, "optimize", new=measured):
        experiment.run(output, False, fixture_path=fixture_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    args = parser.parse_args()
    run(args.output, fixture_path=args.fixture)
# END no private optimizer layout or physical-domain clipping participates.
