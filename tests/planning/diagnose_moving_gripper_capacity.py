"""CPU-only capacity bound for the fixed moving-gripper command.

This is a diagnostic of the native cubic function space, not a collision
backend, production initializer, or claimed robot plan. SciPy HiGHS solves a
small minimax LP with at most 24 controls and 16,386 inequalities. The command
is exactly the existing fixed tracking_command(8); only native capacity varies.
No native cuRobo model, CUDA context, or GPU query is constructed.
"""

# BEGIN bounded scalar approximation audit using the actual progress clock.
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import linprog
import torch

# BEGIN resolve imports from this FUNCTO checkout, independent of its directory name.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
# END project-local imports.
from manipulation.planning import _ProgressMap, _progress_for_cubic_boundaries
from probe_moving_gripper_gpu import tracking_command


def continuous_maximum(native_spline, command_spline, progress_map, knots):
    """Check every exact polynomial error extremum of the saved finite map.

    Both the native q(u) and command q(s) are piecewise cubic. The saved s(u)
    is piecewise linear. Split at all three sets of boundaries, then each
    error is exactly cubic; its extrema are endpoints or quadratic derivative
    roots. The underlying unlinearized cubic clock could instead produce a
    ninth-degree composition, which explains the increased native demand.
    """
    command_breaks = progress_map.anchor_parameters(np.linspace(0., 1., 11))[:, 0]
    keys = np.unique(np.r_[progress_map.u, command_breaks, np.arange(knots+5)/(knots+4)])
    maximum, worst_u = 0., 0.
    native_derivatives = [native_spline.derivative(i) for i in range(4)]
    command_derivatives = [command_spline.derivative(i) for i in range(4)]
    for left, right in zip(keys[:-1], keys[1:]):
        middle, radius = (left+right)/2, (right-left)/2
        index = np.searchsorted(progress_map.u, middle, side="right")-1
        slope = ((progress_map.s[index+1]-progress_map.s[index])
                 / (progress_map.u[index+1]-progress_map.u[index]))
        s = np.interp(middle, progress_map.u, progress_map.s)
        derivatives = [float(native_derivatives[i](middle*(knots+4)))*(knots+4)**i
                       - float(command_derivatives[i](1.+10*s))*(10*slope)**i for i in range(4)]
        coefficients = np.array([derivatives[3]/2, derivatives[2], derivatives[1]])
        roots = np.roots(np.trim_zeros(coefficients, "f")) if np.any(coefficients) else []
        offsets = [-radius, radius]
        offsets.extend(float(root.real) for root in roots
                       if abs(root.imag) < 1e-10 and -radius <= root.real <= radius)
        for offset in offsets:
            error = abs(derivatives[0]+derivatives[1]*offset+derivatives[2]*offset**2/2
                        + derivatives[3]*offset**3/6)
            if error > maximum:
                maximum, worst_u = error, middle+offset
    return maximum, worst_u


def run(output):
    command = tracking_command(8)
    command_coefficients = np.r_[np.full(4, .035), np.linspace(.035, .015, 9)[1:], np.full(3, .015)]
    command_spline = BSpline(np.arange(-3., 16.), command_coefficients, 3)
    u = np.linspace(0., 1., 8193)
    results = []
    for knots in (8, 12, 16, 24):
        progress_map, _ = _progress_for_cubic_boundaries(
            _ProgressMap(np.array([0., 1.]), np.array([0., 1.])), knots, np.zeros(9),
            stop_progress=np.array([1.]), anchors=np.linspace(0., 1., 5))
        progress = np.interp(u, progress_map.u, progress_map.s)
        desired = command(torch.tensor(progress, dtype=torch.float64))[0][:, 0].numpy()
        basis = BSpline.design_matrix(u*(knots+4), np.arange(-3., knots+8.), 3).toarray()
        # Actual cubic start boundary at q0=.035,dq0=ddq0=0: four fixed
        # coefficients. Native replicate-goal mode repeats the last free
        # control three more times. Letting that endpoint vary is a relaxation
        # of exact final feedback, so its minimax optimum is still a valid
        # impossibility lower bound for a stricter physical terminal state.
        matrix = basis[:, 4:4+knots].copy()
        matrix[:, -1] += basis[:, 4+knots:].sum(1)
        target = (desired-.035)/.02
        inequality = np.r_[np.c_[matrix, -np.ones(len(u))], np.c_[-matrix, -np.ones(len(u))]]
        rhs = np.r_[target, -target]
        solution = linprog(np.r_[np.zeros(knots), 1.], A_ub=inequality, b_ub=rhs,
                           bounds=[(-1.75, .25)]*knots+[(0., None)], method="highs",
                           options={"primal_feasibility_tolerance": 1e-9,
                                    "dual_feasibility_tolerance": 1e-9})
        if not solution.success:
            raise RuntimeError(solution.message)
        # A directly evaluable dual certificate: a convex combination of
        # signed residuals cannot exceed their maximum. Minimize that linear
        # combination over the bounded controls to get a numerical lower
        # bound independent of the reported primal optimizer solution.
        weights = np.maximum(-solution.ineqlin.marginals, 0.)
        weights /= weights.sum()
        signed_weights = weights[:len(u)]-weights[len(u):]
        gradient = matrix.T@signed_weights
        certificate = (np.minimum(-1.75*gradient, .25*gradient).sum()-signed_weights@target)*.02
        controls = .035+.02*solution.x[:-1]
        native_spline = BSpline(np.arange(-3., knots+8.), np.r_[np.full(4, .035), controls,
                                                                      np.full(3, controls[-1])], 3)
        maximum, worst_u = continuous_maximum(native_spline, command_spline, progress_map, knots)
        results.append({"native_knots": knots, "fixed_command_knots": 8,
                        "grid_minimax_lower_bound_m": float(solution.fun*.02),
                        "dual_certificate_lower_bound_m": float(certificate),
                        "continuous_error_for_lp_controls_m": maximum, "continuous_worst_u": worst_u,
                        "native_controls_m": controls.tolist(), "progress_map_keys": len(progress_map.u),
                        "required_interpolation_buffer_size": (knots+4)*33+1})
    report = {"scope": "CPU scalar cubic capacity audit; not a robot plan or gradient certification",
              "fixed_command": "tracking_command(8), piecewise cubic in progress, .035m to .015m",
              "grid_points": len(u), "native_finger_tolerance_m": 1e-5,
              "lower_bound": "8193-point minimax and normalized nonnegative dual weights; terminal value relaxed",
              "upper_bound": "all piecewise-cubic error extrema for saved finite progress map; float64 arithmetic",
              "results": results}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps(report, allow_nan=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
# END scalar approximation audit.
