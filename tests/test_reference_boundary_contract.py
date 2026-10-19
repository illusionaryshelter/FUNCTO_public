"""Independent boundary and coordinate counterexamples for grounded references.

The fixtures prescribe a continuous physical source motion before choosing a
body coordinate chart.  They therefore expose changes caused by resampling or
coordinate re-expression instead of merely checking the implementation formula.
These are CPU geometry checks, not claims about collision or robot execution.
"""

# BEGIN current-entry and interaction-boundary reference regressions.
from copy import deepcopy
import unittest

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from manipulation.motion import MotionModel, _inverse
from manipulation.representation import build_representation, generate_reference


def pose(position=(0., 0., 0.), angles=(0., 0., 0.)):
    """Create independently specified world translation and XYZ orientation."""
    value = np.eye(4)
    value[:3, :3] = Rotation.from_euler("xyz", angles).as_matrix()
    value[:3, 3] = position
    return value


def interpolate_track(knots, poses, queries):
    """Sample a prescribed physical-origin track using SciPy, without FUNCTO."""
    query = np.atleast_1d(np.asarray(queries, dtype=float))
    values = np.repeat(np.eye(4)[None], len(query), axis=0)
    for axis in range(3):
        values[:, axis, 3] = np.interp(query, knots, poses[:, axis, 3])
    values[:, :3, :3] = Slerp(knots, Rotation.from_matrix(poses[:, :3, :3]))(query).as_matrix()
    return values[0] if np.asarray(queries).ndim == 0 else values


def at_time(reference, name, times):
    """Use each exported path's own physical interval, including release."""
    own = reference[name + "_timestamps"]
    query = np.asarray(times, dtype=float)
    if np.any(query < own[0]) or np.any(query > own[-1]):
        raise AssertionError("The test attempted to extrapolate a production path")
    progress = (query - own[0]) / (own[-1] - own[0])
    return reference[name + "_path"].sample_numpy(progress)


class ReferenceBoundaryContractTest(unittest.TestCase):
    def setUp(self):
        # Nonuniform time, a non-node interaction and a non-node release catch
        # nearest-frame goals and sample-index-dependent endpoint corrections.
        self.times = np.array([1., 1.12, 1.8, 3.])
        self.interaction = 2.37
        self.release = 2.71
        self.displacements = np.array([
            pose(),
            pose((.03, -.02, .06), (.1, -.15, .2)),
            pose((.18, .04, .2), (-.25, .3, .6)),
            pose((.31, -.03, .08), (.35, -.2, 1.1)),
        ])
        self.source_origin = pose((.07, -.09, .04), (.2, -.3, .15))
        self.function = pose((.55, -.23, .41), (-.2, .35, -.6))
        self.J = pose((.04, -.03, .08), (.3, -.25, .4))
        self.A = pose((.03, .06, -.05), (-.3, .2, -.4))
        self.start_body = pose((-.17, .13, .67), (.5, -.1, .7))
        self.events = [
            {"event_id": "opening_intention", "timestamp": 2.19,
             "command": "open", "source": "synthetic recorded intention"},
            {"event_id": "actual_release", "timestamp": self.release,
             "kind": "release", "source": "synthetic measured contact end"},
        ]

    def source_body(self, queries):
        """A fixed physical trajectory, with a separately chosen body chart."""
        return interpolate_track(self.times, self.displacements, queries) @ self.source_origin

    def representation(self, *, times=None, source_chart=None, interaction=True):
        clock = self.times if times is None else np.asarray(times, dtype=float)
        body = self.source_body(clock)
        if source_chart is not None:
            body = body @ source_chart
        return build_representation(
            kind="manipulation", frame_indices=np.arange(len(clock)), timestamps=clock,
            T_world_carrier=body,
            T_world_reference=np.repeat(np.eye(4)[None], len(clock), axis=0),
            L_R=np.eye(4), approach_time=float(clock[0]),
            interaction_time=self.interaction if interaction else None,
            release_time=self.release, events=deepcopy(self.events))

    def grounded(self, representation=None, **changes):
        options = dict(T_world_function=self.function, J=self.J, A=self.A,
                       T_world_body=self.start_body,
                       T_world_eef=self.start_body @ _inverse(self.A))
        options.update(changes)
        return generate_reference(self.representation() if representation is None else representation, **options)

    def test_actual_start_and_interaction_are_distinct_boundaries(self):
        result = self.grounded()
        self.assertEqual(result["status"], "reference")
        np.testing.assert_allclose(at_time(result, "object", self.times[0]), self.start_body, atol=2e-10)
        np.testing.assert_allclose(at_time(result, "eef", self.times[0]),
                                   self.start_body @ _inverse(self.A), atol=2e-10)
        target = self.function @ self.source_body(self.interaction) @ self.J
        np.testing.assert_allclose(at_time(result, "object", self.interaction), target, atol=2e-10)
        np.testing.assert_allclose(at_time(result, "eef", self.interaction),
                                   target @ _inverse(self.A), atol=2e-10)

        # The initial correction is substantial and the interior changes too.
        # Keeping the target while changing the entry cannot be one rigid move.
        raw_entry = self.function @ self.source_body(self.times[0]) @ self.J
        self.assertGreater(np.linalg.norm(self.start_body[:3, 3] - raw_entry[:3, 3]), .1)
        self.assertGreater(Rotation.from_matrix(self.start_body[:3, :3] @ raw_entry[:3, :3].T).magnitude(), .1)
        middle = 1.63
        raw_middle = self.function @ self.source_body(middle) @ self.J
        self.assertGreater(np.linalg.norm(at_time(result, "object", middle)[:3, 3] - raw_middle[:3, 3]), .01)
        rigidly_moved_target = self.start_body @ _inverse(raw_entry) @ target
        self.assertGreater(np.linalg.norm(rigidly_moved_target - target), .1)

    def test_changing_target_keeps_the_same_actual_entry(self):
        moved_function = pose((.21, .46, .57), (.45, -.3, .8))
        original, changed = self.grounded(), self.grounded(T_world_function=moved_function)
        for result in (original, changed):
            np.testing.assert_allclose(at_time(result, "object", self.times[0]), self.start_body, atol=2e-10)
        changed_target = moved_function @ self.source_body(self.interaction) @ self.J
        np.testing.assert_allclose(at_time(changed, "object", self.interaction), changed_target, atol=2e-10)
        self.assertGreater(np.linalg.norm(at_time(original, "object", self.interaction) - changed_target), .1)

    def test_sampling_density_does_not_change_the_continuous_motion(self):
        # Insert observations on the SAME prescribed motion, including near
        # event times. Sampling composed body origins would define another path.
        dense_times = np.unique(np.r_[self.times, 1.06, 1.39, 2.02, 2.36, 2.58, 2.9])
        sparse, dense = self.grounded(), self.grounded(self.representation(times=dense_times))
        query = np.unique(np.r_[np.linspace(self.times[0], self.release, 43), self.interaction])
        for name in ("object", "eef"):
            np.testing.assert_allclose(at_time(sparse, name, query), at_time(dense, name, query), atol=3e-9)
            for result in (sparse, dense):
                np.testing.assert_allclose(at_time(result, name, result[name + "_timestamps"]),
                                           result[name + "_poses"], atol=2e-10)

    def test_source_and_current_body_chart_changes_leave_physical_motion_unchanged(self):
        source_chart = pose((-.08, .06, .035), (-.4, .25, -.3))
        current_chart = pose((.09, -.07, .045), (.35, -.2, .6))
        original = self.grounded()
        source_changed = self.grounded(self.representation(source_chart=source_chart),
                                       J=_inverse(source_chart) @ self.J)
        current_changed = self.grounded(J=self.J @ current_chart,
            A=self.A @ current_chart, T_world_body=self.start_body @ current_chart)
        query = np.unique(np.r_[np.linspace(self.times[0], self.release, 37), self.interaction])
        for changed in (source_changed, current_changed):
            np.testing.assert_allclose(at_time(changed, "eef", query), at_time(original, "eef", query), atol=3e-9)
        np.testing.assert_allclose(at_time(source_changed, "object", query), at_time(original, "object", query), atol=3e-9)
        np.testing.assert_allclose(at_time(current_changed, "object", query),
                                   at_time(original, "object", query) @ current_chart, atol=3e-9)

    # BEGIN persisted source summaries and generated goals share one physical event.
    def test_source_summary_uses_same_non_node_physical_event_as_reference(self):
        source_chart = pose((-.08, .06, .035), (-.4, .25, -.3))
        expected_source = self.source_body(self.interaction)
        expected_goal = self.function @ expected_source @ self.J
        for chart in (np.eye(4), source_chart):
            with self.subTest(body_chart=chart.tolist()):
                rep = self.representation(source_chart=chart)
                summary = rep["summaries"]["H_m"]
                # The independent fixture fixes a physical motion before the
                # body chart is chosen; the reported pose must transform with
                # that chart even when the event lies between source samples.
                np.testing.assert_allclose(summary["interaction_pose"], expected_source @ chart, atol=2e-10)
                displacement = ((expected_source @ chart)[:3, 3]
                                - (self.source_body(self.times[0]) @ chart)[:3, 3])
                np.testing.assert_allclose(summary["approach_direction"],
                                           displacement / np.linalg.norm(displacement), atol=2e-10)
                result = self.grounded(rep, J=_inverse(chart) @ self.J)
                np.testing.assert_allclose(result["boundary_checks"]["object"]["nominal_goal"], expected_goal, atol=2e-10)
                np.testing.assert_allclose(self.function @ summary["interaction_pose"] @ _inverse(chart) @ self.J,
                                           expected_goal, atol=2e-10)
    # END changing a body chart cannot silently change the source event or target.

    def test_new_grasp_changes_only_eef_conversion_at_fixed_body_state(self):
        changed_A = pose((-.08, .04, .12), (.4, -.5, .1))
        original = self.grounded()
        changed = self.grounded(A=changed_A, T_world_eef=self.start_body @ _inverse(changed_A))
        query = np.linspace(self.times[0], self.release, 41)
        np.testing.assert_allclose(at_time(original, "object", query), at_time(changed, "object", query), atol=2e-10)
        self.assertGreater(np.linalg.norm(at_time(original, "eef", 1.63) - at_time(changed, "eef", 1.63)), .1)
        for result, attachment in ((original, self.A), (changed, changed_A)):
            np.testing.assert_allclose(at_time(result, "eef", query) @ attachment,
                                       at_time(result, "object", query), atol=2e-10)

    def test_existing_attachment_and_either_observed_pose_determine_the_entry(self):
        full = self.grounded()
        body_only = self.grounded(T_world_eef=None)
        eef_only = self.grounded(T_world_body=None)
        query = np.linspace(self.times[0], self.release, 17)
        for result in (body_only, eef_only):
            self.assertEqual(result["status"], "reference")
            for name in ("object", "eef"):
                np.testing.assert_allclose(at_time(result, name, query), at_time(full, name, query), atol=2e-10)

    def test_no_attachment_preserves_only_the_adapted_object_output(self):
        result = self.grounded(A=None, T_world_eef=None)
        self.assertNotEqual(result["status"], "reference")
        self.assertIn("eef_poses", result["missing"])
        self.assertNotIn("eef_path", result)
        self.assertNotIn("eef_poses", result)
        np.testing.assert_allclose(at_time(result, "object", self.times[0]), self.start_body, atol=2e-10)
        np.testing.assert_allclose(at_time(result, "object", self.interaction),
                                   self.function @ self.source_body(self.interaction) @ self.J, atol=2e-10)

    def test_missing_entry_or_interaction_never_returns_an_unadapted_production_path(self):
        cases = [self.grounded(T_world_body=None, T_world_eef=None),
                 self.grounded(self.representation(interaction=False))]
        for result in cases:
            with self.subTest(missing=result["missing"]):
                self.assertNotEqual(result["status"], "reference")
                self.assertTrue(result["missing"])
                for key in ("object_path", "object_poses", "eef_path", "eef_poses"):
                    self.assertNotIn(key, result)

    def test_inconsistent_current_body_eef_and_attachment_are_rejected(self):
        wrong_eef = pose((.11, -.04, .02), (.2, .1, -.3)) @ self.start_body @ _inverse(self.A)
        with self.assertRaises(ValueError):
            self.grounded(T_world_eef=wrong_eef)

    def test_coincident_entry_and_interaction_cannot_overwrite_conflicting_boundaries(self):
        rep = self.representation()
        rep["interaction_time"] = float(self.times[0])
        with self.assertRaises(ValueError):
            self.grounded(rep)

    def test_release_limits_production_paths_but_retains_original_evidence(self):
        rep = self.representation()
        result = self.grounded(rep)
        for name in ("object", "eef"):
            self.assertAlmostEqual(result[name + "_timestamps"][-1], self.release)
            self.assertLess(result[name + "_timestamps"][-1], rep["timestamps"][-1])
        np.testing.assert_array_equal(result["timestamps"], rep["timestamps"])
        np.testing.assert_array_equal(result["frame_indices"], rep["frame_indices"])
        np.testing.assert_array_equal(result["H_m"], rep["H_m"])
        self.assertEqual(result["events"], self.events)
        self.assertIn("bound_object_poses", result)
        self.assertGreaterEqual(len(result["bound_object_poses"]), len(rep["timestamps"]))

        # The source suffix after the interaction remains the demonstrated
        # relation until release. The opening intention is not its goal time.
        suffix = np.array([self.interaction, 2.51, self.release])
        expected = self.function @ self.source_body(suffix) @ self.J
        np.testing.assert_allclose(at_time(result, "object", suffix), expected, atol=3e-9)

    def test_grasp_reference_also_starts_at_actual_eef(self):
        rep = build_representation(kind="grasp", frame_indices=[2, 4, 9, 17], timestamps=self.times,
            T_world_carrier=np.repeat(np.eye(4)[None], len(self.times), axis=0),
            T_world_eef=self.displacements, L_G=np.eye(4), interaction_time=self.interaction)
        actual_eef = pose((-.2, .3, .9), (.4, -.1, .2))
        result = generate_reference(rep, T_world_function=self.function, T_world_eef=actual_eef)
        self.assertEqual(result["status"], "reference")
        np.testing.assert_allclose(at_time(result, "eef", self.times[0]), actual_eef, atol=2e-10)
        target = self.function @ interpolate_track(self.times, self.displacements, self.interaction)
        np.testing.assert_allclose(at_time(result, "eef", self.interaction), target, atol=2e-10)
        missing = generate_reference(rep, T_world_function=self.function)
        self.assertNotEqual(missing["status"], "reference")
        self.assertNotIn("eef_path", missing)
        self.assertNotIn("eef_poses", missing)

    def test_known_mechanism_uses_existing_current_coordinate_without_extra_pose_input(self):
        model = MotionModel("prismatic", np.array([[0.], [1.]]), pose((.4, -.2, .3)),
                            np.eye(3), np.zeros(3), 1e-4, 1e-4, {})
        rep = build_representation(kind="manipulation", frame_indices=np.arange(len(self.times)),
            timestamps=self.times, interaction_time=self.interaction, release_time=self.release)
        source = np.array([.1, .15, .3, .45])
        current = .23
        result = generate_reference(rep, motion_model=model, source_coordinates=source,
                                    current_coordinates=current, A=self.A)
        self.assertEqual(result["status"], "reference")
        query = np.array([self.times[0], 1.43, self.interaction, self.release])
        expected_coordinates = current + np.interp(query, self.times, source) - source[0]
        expected = model.poses(expected_coordinates[:, None])
        np.testing.assert_allclose(at_time(result, "object", query), expected, atol=2e-10)
        np.testing.assert_allclose(at_time(result, "eef", query) @ self.A, expected, atol=2e-10)


if __name__ == "__main__":
    unittest.main()
# END current-entry and interaction-boundary reference regressions.
