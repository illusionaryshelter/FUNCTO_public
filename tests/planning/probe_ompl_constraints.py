"""Small CPU-only OMPL 2.0.1 binding probe, not a robot planner.

The toy state is (x, y, phase), constrained to x*x + y*y = 1.  It checks
Python projection callbacks, directed MotionValidator callbacks, the broken
discreteGeodesic return binding, and a native PathGeometric export route.
The explicit edge samples are an API demonstration, not continuous collision
certification or a choice of physical skill tolerance.
"""

# BEGIN isolated OMPL constrained-planning API probe.
import argparse
import gc
import json
from pathlib import Path
import resource
import time

import numpy as np
from ompl import base as ob, geometric as og, util as ou


def probe():
    """Keep native ownership local and return ordinary JSON-compatible data."""
    counts = {"function": 0, "jacobian": 0, "valid": 0, "motion": 0}
    options = {"tail_obstacle": False}

    class Cylinder(ob.Constraint):
        """Use the library's Newton projection with an analytic toy Jacobian."""

        def __init__(self):
            super().__init__(3, 1, 1e-8)

        def function(self, x, out):
            counts["function"] += 1
            out[0] = x[0] * x[0] + x[1] * x[1] - 1

        def jacobian(self, x, out):
            counts["jacobian"] += 1
            out[:] = [[2 * x[0], 2 * x[1], 0]]

    constraint = Cylinder()
    projected = np.array([0.8, 0.8, 0.5], dtype=np.float64)
    projection_success = bool(constraint.project(projected))
    result = {
        "projection_success": projection_success,
        "projection_residual": float(projected[0] ** 2 + projected[1] ** 2 - 1),
        "projection_callbacks": counts.copy(),
    }
    ambient = ob.RealVectorStateSpace(3)
    bounds = ob.RealVectorBounds(3)
    bounds.setLow(-2)
    bounds.setHigh(2)
    bounds.setLow(2, 0)
    bounds.setHigh(2, 1)
    ambient.setBounds(bounds)
    space = ob.ProjectedStateSpace(ambient, constraint)
    space.setDelta(0.1)
    information = ob.ConstrainedSpaceInformation(space)

    def vector(state):
        return np.array([state[i] for i in range(3)], dtype=np.float64)

    def valid_array(values):
        if (not np.isfinite(values).all() or np.any(np.abs(values[:2]) > 2)
                or not 0 <= values[2] <= 1):
            return False
        return not (options["tail_obstacle"] and 0.975 < values[2] < 0.99)

    def valid(state):
        counts["valid"] += 1
        return bool(valid_array(vector(state)))

    information.setStateValidityChecker(valid)
    information.setup()
    # Retain the original C++ checker: its bool result has a valid binding.
    native_validator = information.getMotionValidator()
    start = space.allocState()
    start.copy([1.0, 0.0, 0.0])
    goal = space.allocState()
    goal.copy([0.0, 1.0, 1.0])

    try:
        answer = space.discreteGeodesic(start, goal, False)
        result["direct_geodesic"] = {"callable": True, "type": type(answer).__name__}
    except TypeError as error:
        result["direct_geodesic"] = {"callable": False, "error": str(error)}

    def native_points(a, b):
        """Use the existing C++ path container to own and expose its states.

        interpolate() itself skips validity. A successful native check must
        precede this function; all exported points and the true tail still
        require the same edge acceptance checks.
        """
        path = og.PathGeometric(information)
        path.append(a)
        path.append(b)
        path.interpolate()
        return np.array([vector(path.getState(i)) for i in range(path.getStateCount())])

    result["native_forward"] = bool(native_validator.checkMotion(start, goal))
    exported = native_points(start, goal)
    result["path_container_export"] = {
        "nodes": len(exported), "first": exported[0].tolist(),
        "last": exported[-1].tolist(), "penultimate": exported[-2].tolist(),
        "maximum_node_residual": float(np.max(np.abs((exported[:, :2] ** 2).sum(1) - 1))),
    }

    def checked_forward_edge(a, b):
        """Check the proposed edge during tree growth, before accepting it.

        This probe uses bounded projected samples between the library's
        geodesic nodes, including its unchecked delta-length terminal tail.
        Constraint.project supplies the projection, so this is not a custom
        RRT or inverse-kinematics implementation. Real robot discretization,
        contact geometry and export equivalence are outside this experiment.
        """
        first, last = vector(a), vector(b)
        if (last[2] < first[2] or not valid_array(first) or not valid_array(last)
                or not constraint.isSatisfied(first) or not constraint.isSatisfied(last)):
            return None
        if not native_validator.checkMotion(a, b):
            return None
        vertices = native_points(a, b)
        if (len(vertices) > 256 or not np.array_equal(vertices[0], first)
                or not np.array_equal(vertices[-1], last)):
            return None
        checked = [first]
        for left, right in zip(vertices[:-1], vertices[1:]):
            intervals = max(1, int(np.ceil(np.linalg.norm(right - left) / 0.01)))
            if intervals > 256:
                return None
            for index in range(1, intervals + 1):
                # Preserve each native node exactly at the interval endpoint;
                # algebraically equivalent floating-point interpolation need
                # not reproduce its final operand bit for bit.
                point = (right.copy() if index == intervals else
                         left + (right - left) * (index / intervals))
                if (not constraint.project(point) or not constraint.isSatisfied(point)
                        or not valid_array(point) or point[2] < checked[-1][2] - 1e-12):
                    return None
                checked.append(point)
        samples = np.asarray(checked)
        # A projection changing the caller's true endpoint is not a successful
        # edge export, even if a nearby point is on the same constraint surface.
        if not np.array_equal(samples[0], first) or not np.array_equal(samples[-1], last):
            return None
        return samples

    class ForwardValidator(ob.MotionValidator):
        def __init__(self, info):
            super().__init__(info)

        def checkMotion(self, a, b, *args):
            counts["motion"] += 1
            return checked_forward_edge(a, b) is not None

    validator = ForwardValidator(information)
    information.setMotionValidator(validator)
    try:
        result["checked_forward"] = bool(information.checkMotion(start, goal))
        result["checked_reverse"] = bool(information.checkMotion(goal, start))
        options["tail_obstacle"] = True
        result["tail_obstacle"] = {
            "phase_band": [0.975, 0.99],
            "native_accepts": bool(native_validator.checkMotion(start, goal)),
            "checked_accepts": bool(information.checkMotion(start, goal)),
        }
        options["tail_obstacle"] = False
        definition = ob.ProblemDefinition(information)
        definition.setStartAndGoalStates(start, goal)
        planner = og.RRTConnect(information)
        planner.setIntermediateStates(False)
        planner.setProblemDefinition(definition)
        planner.setup()
        before = counts.copy()
        status = planner.solve(0.2)
        result["rrtconnect"] = {
            "status": str(status), "exact": bool(definition.hasExactSolution()),
            "motion_callbacks_during_solve": counts["motion"] - before["motion"],
            "valid_callbacks_during_solve": counts["valid"] - before["valid"],
        }
        if definition.hasExactSolution():
            path = definition.getSolutionPath()
            values = np.asarray([vector(path.getState(i)) for i in range(path.getStateCount())])
            # Reuse the identical edge exporter to inspect the returned path.
            edges = [checked_forward_edge(path.getState(i), path.getState(i + 1))
                     for i in range(path.getStateCount() - 1)]
            result["rrtconnect"].update(
                path_nodes=len(values), phase_monotone=bool(np.all(np.diff(values[:, 2]) >= 0)),
                all_edges_exported=all(edge is not None for edge in edges),
                endpoint_exact=bool(np.array_equal(values[0], vector(start))
                                    and np.array_equal(values[-1], vector(goal))),
                maximum_node_residual=float(np.max(np.abs((values[:, :2] ** 2).sum(1) - 1))),
            )
            if all(edge is not None for edge in edges):
                samples = np.concatenate(edges)
                result["rrtconnect"].update(
                    exported_sample_count=len(samples),
                    all_exported_edge_endpoints_exact=all(
                        np.array_equal(edge[0], values[index])
                        and np.array_equal(edge[-1], values[index + 1])
                        for index, edge in enumerate(edges)),
                    exported_phase_monotone=bool(np.all(np.diff(samples[:, 2]) >= -1e-12)),
                    maximum_exported_sample_residual=float(
                        np.max(np.abs((samples[:, :2] ** 2).sum(1) - 1))),
                    all_exported_samples_valid=all(valid_array(point) for point in samples),
                )
    finally:
        # Python callbacks capture native objects. Break the ownership cycle
        # explicitly; suppressing nanobind warnings would hide a real leak.
        information.setMotionValidator(native_validator)
    result["counts"] = counts
    result["scope"] = "toy constrained-space binding and directed edge probe; no robot or continuous collision guarantee"
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    ou.RNG.setSeed(7)
    report = probe()
    gc.collect()
    report.update(elapsed_s=time.monotonic() - started,
                  peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  gpu_called=False, curobo_imported=False)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, allow_nan=False))
# END isolated OMPL constrained-planning API probe.
