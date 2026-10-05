"""CPU checks of a phase-independent preference over validated cubic curves."""

# BEGIN exact polynomial and functional-event quality contracts without robot FK.
from types import SimpleNamespace
import unittest

import numpy as np
from scipy.interpolate import CubicHermiteSpline

from manipulation.planning_quality import trajectory_quality


class PlanningQualityTest(unittest.TestCase):
    def curve(self, times=(0., 1.)):
        times = np.asarray(times, float)
        # Two named joints traverse the same normalized cubic with different
        # physical derivative limits. The polynomial is independent of its
        # knot subdivision and of the order used to provide limit columns.
        position = times[:, None]**3*np.array([1., 2.])
        velocity = 3*times[:, None]**2*np.array([1., 2.])
        return SimpleNamespace(joint_names=("first", "second"), times=times,
            _curve=CubicHermiteSpline(times, position, velocity, axis=0))

    def limits(self):
        return dict(limit_joint_names=("second", "first"),
            velocity_bounds=np.array([[-6., -3.], [6., 3.]]),
            acceleration_bounds=np.array([[-12., -6.], [12., 6.]]),
            jerk_bounds=np.array([[-12., -6.], [12., 6.]]))

    def problem(self, **changes):
        values = dict(start_progress=0., pose_events=(), coordinate_events=())
        values.update(changes)
        return SimpleNamespace(**values)

    def test_exact_derivative_integrals_do_not_depend_on_cubic_subdivision_or_named_order(self):
        expected = dict(velocity=1/5, acceleration=1/3, jerk=1.)
        for times in ((0., 1.), (0., .17, .91, 1.)):
            value = trajectory_quality(self.curve(times), {}, self.problem(), **self.limits())
            for name, exact in expected.items():
                self.assertAlmostEqual(value["motion_components"][name], exact, places=12)
            self.assertAlmostEqual(value["score"], sum(expected.values()), places=12)
            self.assertEqual(value["task_center"], 0.)
        limits = self.limits()
        limits["limit_joint_names"] = tuple(reversed(limits["limit_joint_names"]))
        for name in ("velocity_bounds", "acceleration_bounds", "jerk_bounds"):
            limits[name] = limits[name][:, ::-1]
        reordered = trajectory_quality(self.curve(), {}, self.problem(), **limits)
        self.assertAlmostEqual(reordered["score"], sum(expected.values()), places=12)

    def test_quality_uses_real_goal_tolerances_and_ignores_reference_and_proxy_measurements(self):
        request = self.problem(
            pose_events=(SimpleNamespace(progress=.5, position_tolerance_m=.01, rotation_tolerance_rad=.02),),
            coordinate_events=(SimpleNamespace(progress=1., kind="revolute", tolerance=.03),))
        measurements = {"function:0:position_error_m": .005, "function:0:rotation_error_rad": .005,
                        "coordinate:0:error_rad": .03, "phase:0:original_reference_position_error_m": 999.,
                        "process_scene": 999.}
        value = trajectory_quality(self.curve(), measurements, request, **self.limits())
        self.assertAlmostEqual(value["task_center"], .5**2+.25**2+1., places=12)
        measurements["function:0:position_error_m"] = .0025
        improved = trajectory_quality(self.curve(), measurements, request, **self.limits())
        self.assertLess(improved["score"], value["score"])
        self.assertEqual(improved["motion"], value["motion"])
        # A suffix cannot recreate an already completed event, and no stale
        # measurement is needed for that removed part of the original request.
        request.start_progress = .6
        del measurements["function:0:position_error_m"]
        del measurements["function:0:rotation_error_rad"]
        suffix = trajectory_quality(self.curve(), measurements, request, **self.limits())
        self.assertEqual(suffix["task_center"], 1.)

    def test_quality_rejects_missing_or_nonfinite_evidence_and_invalid_limit_names(self):
        request = self.problem(pose_events=(SimpleNamespace(progress=1.,
            position_tolerance_m=.01, rotation_tolerance_rad=.02),))
        for measurements in ({}, {"function:0:position_error_m": np.nan},
                             {"function:0:position_error_m": -.1}):
            with self.subTest(measurements=measurements), self.assertRaisesRegex(ValueError, "measurement"):
                trajectory_quality(self.curve(), measurements, request, **self.limits())
        limits = self.limits(); limits["limit_joint_names"] = ("first", "unrelated")
        with self.assertRaisesRegex(ValueError, "matching derivative bounds"):
            trajectory_quality(self.curve(), {}, self.problem(), **limits)
        limits = self.limits(); limits["velocity_bounds"][1, 0] = np.inf
        with self.assertRaisesRegex(ValueError, "finite ordered named velocity"):
            trajectory_quality(self.curve(), {}, self.problem(), **limits)
        broken = self.curve(); broken._curve.c[0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "complete named cubic"):
            trajectory_quality(broken, {}, self.problem(), **self.limits())
# END this preference neither checks geometry nor authorizes execution.


if __name__ == "__main__":
    unittest.main()
