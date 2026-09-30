"""Independent CPU oracle for progress versus native cubic boundary geometry.

This tests the scalar base-angle subclass of the synthetic paired-branch
fixture. It does not prove full nonlinear Franka infeasibility and does not
execute IK, CUDA, collision queries or a robot. SciPy supplies both the exact
uniform cubic basis and the minimax linear-program optimizer.
"""

# BEGIN bounded scalar oracle for native progress-clock representation.
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import linprog


def native_basis(count):
    """Installed V2 layout: four prefix controls, N actions, three tail copies."""
    coefficients = np.zeros((count+7, count))
    coefficients[4:4+count] = np.eye(count)
    coefficients[-3:, -1] = 1.
    return BSpline(np.arange(-3., count+8.), coefficients, 3)


def linearization(clock, count, subdivisions):
    """Compare dense samples with a proved second-difference error bound.

    For uniform knot spacing one, the cubic's second derivative is a convex
    combination of second differences of its coefficients. Linear sampling
    at spacing 1/L therefore has sup error at most max|diff(c,2)|/(8 L^2).
    """
    intervals = count+4
    keys = np.linspace(0., 1., intervals*subdivisions+1)
    values = clock(keys*intervals)
    query = np.linspace(0., 1., 20001)
    difference = np.abs(np.interp(query, keys, values)-clock(query*intervals))
    second = float(np.max(np.abs(np.diff(clock.c, n=2))))
    return {"subdivisions_per_native_interval": int(subdivisions), "keys": len(keys),
            "observed_normalized_parameter_error": float(difference.max()),
            "proved_normalized_parameter_error_bound": second/(8*subdivisions**2),
            "maximum_second_coefficient_difference": second}


def old_linear_clock_minimax(count):
    """Give a lower bound with no joint/velocity/acceleration restrictions.

    The old clock holds only the already-constant first/last native intervals
    and is otherwise linear from source progress zero to two. At progress one
    a real grasp stop requires scalar angle and velocity zero; future carry
    requires theta=.45*(s-1). All N scalar controls are otherwise unrestricted.
    """
    intervals = count+4
    basis = native_basis(count)
    first, finish = 1./intervals, (intervals-1.)/intervals
    start = (first+finish)/2
    keys, values = [0., first, finish, 1.], [0., 0., 2., 2.]
    query = np.linspace(start, finish, 2001)
    target = .45*(np.interp(query, keys, values)-1.)
    design = basis(query*intervals)
    inequalities = np.r_[np.column_stack((design, -np.ones(len(query)))),
                        np.column_stack((-design, -np.ones(len(query))))]
    equality = np.column_stack((np.stack((basis(start*intervals),
                                         basis.derivative()(start*intervals))), [0., 0.]))
    result = linprog(np.r_[np.zeros(count), 1.], A_ub=inequalities, b_ub=np.r_[target, -target],
                     A_eq=equality, b_eq=[0., 0.],
                     bounds=[(None, None)]*count+[(0., None)], method="highs")
    if not result.success:
        raise RuntimeError(result.message)
    dense = np.linspace(start, finish, 20001)
    defect = np.abs(basis(dense*intervals)@result.x[:-1]-.45*(np.interp(dense, keys, values)-1.))
    return {"free_controls": count, "grasp_u": start, "terminal_arrival_u": finish,
            "grid_minimax_lower_bound_rad": float(result.fun),
            "independent_dense_error_rad": float(defect.max()),
            "peak_u": float(dense[np.argmax(defect)]),
            "scalar_controls": result.x[:-1].tolist()}


def native_internal_stop_clock():
    """An exact feasible scalar counterexample in the same N=16 basis.

    The parameter warp r runs from zero to one. Source progress is s=2r.
    Seven ascending coefficients reach r=.5; three extra copies produce four
    equal coefficients and a C2 stop at s=1. Six remaining coefficients reach
    r=1. The carry angle controls use the same scalar relation .45*max(s-1,0).
    """
    count = 16
    controls = np.r_[np.arange(1, 8)/14., [.5]*3, .5+np.arange(1, 7)/12.]
    coefficients = np.r_[np.zeros(4), controls, np.ones(3)]
    clock = BSpline(np.arange(-3., count+8.), coefficients, 3)
    angle = native_basis(count)
    angle_controls = .45*np.maximum(2*controls-1., 0.)
    query = np.linspace(0., 1., 20001)
    mapped = clock(query*(count+4))
    actual = angle(query*(count+4))@angle_controls
    expected = .45*np.maximum(2*mapped-1., 0.)
    stop_query = np.linspace(10., 11., 101)
    second = float(np.max(np.abs(np.diff(coefficients, n=2))))
    bounded_subdivisions = int(np.ceil(np.sqrt(second/(8e-6))))
    return {
        "free_controls": count, "parameter_controls": controls.tolist(),
        "angle_controls": angle_controls.tolist(),
        "source_grasp_progress": 1., "grasp_stop_u": [.5, .55],
        "maximum_required_angle_error_rad": float(np.max(np.abs(actual-expected))),
        "maximum_stop_parameter_velocity_native_time": float(np.max(np.abs(clock.derivative()(stop_query)))),
        "maximum_stop_parameter_acceleration_native_time": float(np.max(np.abs(clock.derivative(2)(stop_query)))),
        "linearization": [linearization(clock, count, value) for value in (16, 32, bounded_subdivisions)],
        "scope": "exact scalar subclass witness, not an injected production robot seed",
    }


def run(output):
    report = {"scope": "independent CPU scalar native-cubic oracle; no full nonlinear robot infeasibility claim",
              "rotation_component_allowance_rad": .005,
              "rotation_norm_envelope_rad": float(np.sqrt(3)*.005),
              "old_clock": [old_linear_clock_minimax(count) for count in (12, 16, 24)],
              "same_basis_with_internal_stop": native_internal_stop_clock()}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps(report, allow_nan=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
# END bounded scalar progress-clock oracle.
