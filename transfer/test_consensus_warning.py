"""CPU regressions for treating a low inlier fraction as quality evidence.

The warning permits a localization candidate; it does not certify the matched
position. Exact policy boundaries inject a registration result deliberately.
The final counterexample instead runs the real robust solver with fixed data
and seed, demonstrating that all remaining gates can accept a wrong position.
No test initializes a renderer, neural model, or CUDA device.
"""

# BEGIN LOW_CONSENSUS_QUALITY_WARNING_REGRESSIONS
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from transfer.core import TransferConfig, fps_indices, infer_region, prepare_anchors
from transfer.experiment import store_prediction
from transfer.registration import RegistrationResult
from transfer.test_core import (
    _controls, _grid_pixels, _mirror_fixture, _plane_grid, _surface_fixture,
)


WARNING = "low_correspondence_inlier_ratio"


def _injected_preparation(source, target, source_ids, target_ids, count, *,
                          mode="automatic", reason=None):
    """Isolate the consumer's exact count/ratio boundary from robust sampling.

    The solver's returned residuals and support mask are explicit test inputs,
    not a claim that arbitrary supplied pairs would produce this solution.
    Actual-solver coverage is provided separately by the fixed plane example.
    """
    inliers = np.arange(len(source_ids)) < count

    def registered(*args, **kwargs):
        if reason is not None:
            return RegistrationResult(None, None, None, None,
                np.zeros(len(source_ids), dtype=bool), reason, {})
        residuals = np.where(inliers, 0.0, 2 * kwargs["tolerance_m"])
        return RegistrationResult(1.0, np.eye(3), np.zeros(3), residuals,
                                  inliers, None, {})

    with mock.patch("transfer.core.register_sim3", side_effect=registered):
        return prepare_anchors(source, target, source_ids, target_ids, mode=mode)


def _padded_ids(first, available, count):
    """Keep designated support first, then add distinct unused pair indices."""
    return np.r_[first, np.setdiff1d(available, first)[:count - len(first)]]


def _fixed_plane_correspondences(offset_steps):
    """Return fixed matched evidence whose geometric truth is identity.

    Twelve pairs agree on a translation of ``offset_steps * 5 mm``. Sixty-eight
    other pairs are unrelated. A nonzero offset represents coherent matching
    mistakes on a repeated planar surface, not the independently defined truth.
    The source annotation and pixel topology are fixed before matching.
    """
    x_ticks, y_ticks = np.arange(-40, 41), np.arange(-30, 31)
    xx, yy = np.meshgrid(x_ticks, y_ticks, indexing="ij")
    points = np.column_stack((xx.ravel() * 0.005, yy.ravel() * 0.005,
                              np.zeros(xx.size)))
    pixels = np.column_stack((np.repeat(np.arange(len(x_ticks)), len(y_ticks)),
                               np.tile(np.arange(len(y_ticks)), len(x_ticks))))
    region = np.linalg.norm(points - [-0.075, 0., 0.], axis=1) <= 0.02 + 1e-12
    index = {tuple(pair): i for i, pair in enumerate(zip(xx.ravel(), yy.ravel()))}
    support_xy = [(x, y) for x in [-25, -15, -5] for y in [-20, -7, 9, 20]]
    source_good = np.array([index[(x, y)] for x, y in support_xy])
    target_good = np.array([index[(x + offset_steps, y)] for x, y in support_xy])
    rng = np.random.default_rng(0)
    source_bad = rng.choice(np.setdiff1d(np.arange(len(points)), source_good),
                            68, replace=False)
    target_bad = rng.choice(np.setdiff1d(np.arange(len(points)), target_good),
                            68, replace=False)
    return points, pixels, region, np.r_[source_good, source_bad], np.r_[target_good, target_bad]


class TestConsensusWarningPolicy(unittest.TestCase):
    """Warnings cannot replace any independent geometric rejection condition."""

    def test_low_ratio_accepts_automatic_and_direct_candidates_with_warning(self):
        points = _plane_grid()
        ids = fps_indices(points, 50)
        # All source pairs are inside the annotation for direct_local, as its
        # public contract requires; both modes receive the same source plane.
        region = np.ones(len(points), dtype=bool)
        for mode, method in (("automatic", "relational"), ("direct", "direct_local")):
            with self.subTest(mode=mode):
                prepared = _injected_preparation(points, points, ids, ids, 9, mode=mode)
                result = infer_region(region, prepared, method=method, **_controls(points))
                self.assertEqual(prepared.diagnostics["inlier_ratio"], 0.18)
                self.assertEqual(prepared.diagnostics["quality_warnings"], [WARNING])
                self.assertEqual(prepared.reasons, ())
                self.assertTrue(result["accepted"], result["reasons"])
                self.assertEqual(result["diagnostics"]["quality_warnings"], [WARNING])

    def test_exactly_twenty_percent_has_no_low_ratio_warning(self):
        points = _plane_grid()
        ids = fps_indices(points, 50)
        prepared = _injected_preparation(points, points, ids, ids, 10)
        result = infer_region(np.ones(len(points), dtype=bool), prepared,
                              method="relational", **_controls(points))
        self.assertEqual(prepared.diagnostics["inlier_ratio"], 0.2)
        self.assertEqual(prepared.diagnostics["quality_warnings"], [])
        self.assertTrue(result["accepted"], result["reasons"])

    def test_fewer_than_six_final_inliers_remain_a_hard_rejection(self):
        points = _plane_grid()
        ids = fps_indices(points, 50)
        prepared = _injected_preparation(points, points, ids, ids, 5)
        result = infer_region(np.ones(len(points), dtype=bool), prepared,
                              method="relational", **_controls(points))
        self.assertIsNone(prepared.scale)
        self.assertEqual(prepared.reasons, ("weak_correspondence_consensus",))
        self.assertEqual(result["diagnostics"]["quality_warnings"], [])
        self.assertFalse(result["accepted"])
        self.assertEqual(result["modes"], [])

    def test_backend_hard_failures_are_not_reclassified_by_reason_string(self):
        points = _plane_grid()
        ids = fps_indices(points, 50)
        for reason in ("weak_correspondence_consensus", "degenerate_anchors"):
            with self.subTest(reason=reason):
                prepared = _injected_preparation(points, points, ids, ids, 9, reason=reason)
                result = infer_region(np.ones(len(points), dtype=bool), prepared,
                                      method="relational", **_controls(points))
                self.assertFalse(result["accepted"])
                self.assertEqual(result["reasons"], [reason])
                self.assertEqual(result["diagnostics"]["quality_warnings"], [])

    def test_low_ratio_keeps_opposite_face_and_insufficient_region_rejections(self):
        points = _plane_grid()
        ids = fps_indices(points, 50)
        prepared = _injected_preparation(points, points, ids, ids, 9)
        cases = (
            (np.ones(len(points), dtype=bool), _controls(points, target_camera=[0, 0, -1]),
             "no_valid_oriented_surface"),
            (np.arange(len(points)) < 5, _controls(points), "insufficient_source_region"),
        )
        for region, controls, reason in cases:
            with self.subTest(reason=reason):
                result = infer_region(region, prepared, method="relational", **controls)
                self.assertFalse(result["accepted"])
                self.assertIn(reason, result["reasons"])
                self.assertEqual(result["diagnostics"]["quality_warnings"], [WARNING])

    def test_low_ratio_keeps_indistinguishable_surface_rejection(self):
        points, region, anchors = _mirror_fixture()
        ids = _padded_ids(anchors, np.arange(len(points)), 50)
        prepared = _injected_preparation(points, points, ids, ids, len(anchors))
        result = infer_region(region, prepared, method="relation_only", **_controls(points))
        self.assertFalse(result["accepted"])
        self.assertIn("surface_ambiguity", result["reasons"])
        self.assertEqual(result["diagnostics"]["quality_warnings"], [WARNING])

    def test_low_ratio_cannot_project_onto_distant_visible_support(self):
        source, region, anchors, centroid = _surface_fixture()
        visible = np.linalg.norm(source - centroid, axis=1) > 0.04
        visible[anchors] = True
        remap = np.full(len(source), -1, dtype=int)
        remap[visible] = np.arange(visible.sum())
        ids = _padded_ids(anchors, np.flatnonzero(visible), 150)
        prepared = _injected_preparation(source, source[visible], ids, remap[ids], len(anchors))
        result = infer_region(region, prepared, method="global_sim3",
            **_controls(source, target_pixels=_grid_pixels(source)[visible]))
        self.assertFalse(result["accepted"])
        self.assertIn("predicted_center_without_nearby_surface", result["reasons"])
        self.assertEqual(result["diagnostics"]["quality_warnings"], [WARNING])

    def test_oracle_keeps_an_empty_warning_list_without_robust_registration(self):
        points = _plane_grid()
        ids = fps_indices(points, 3)
        with mock.patch("transfer.core.register_sim3") as registered:
            prepared = prepare_anchors(points, points, ids, ids, mode="oracle")
        registered.assert_not_called()
        result = infer_region(np.ones(len(points), dtype=bool), prepared,
                              method="oracle_relational", **_controls(points))
        self.assertTrue(result["accepted"], result["reasons"])
        self.assertEqual(prepared.diagnostics["quality_warnings"], [])
        self.assertEqual(result["diagnostics"]["quality_warnings"], [])

    def test_prediction_json_preserves_warning_for_accepted_and_rejected_results(self):
        points = _plane_grid()
        ids = fps_indices(points, 50)
        prepared = _injected_preparation(points, points, ids, ids, 9)
        with tempfile.TemporaryDirectory() as temporary:
            for camera_z in (1, -1):
                with self.subTest(camera_z=camera_z):
                    result = infer_region(np.ones(len(points), dtype=bool), prepared,
                        method="relational", **_controls(points, target_camera=[0, 0, camera_z]))
                    output = Path(temporary) / f"prediction_{camera_z}"
                    store_prediction(output, result, {})
                    restored = json.loads(output.with_suffix(".json").read_text())
                    self.assertEqual(restored["accepted"], result["accepted"])
                    self.assertEqual(restored["reasons"], result["reasons"])
                    self.assertEqual(restored["diagnostics"]["quality_warnings"], [WARNING])
                    self.assertEqual(restored["diagnostics"]["inlier_ratio"], 0.18)

    # BEGIN multiview consumer preserves warning and hard-rejection semantics.
    def test_multiview_preserves_warning_without_erasing_a_hard_rejection(self):
        from dataclasses import replace
        from transfer.multiview_grounding import infer_multiview_region
        from transfer.test_multiview_grounding import _observation, _prepared, _targets

        source = _observation()
        targets = _targets(_observation(), _observation())
        bank, prepared = _prepared(source, targets)
        # This consumer test injects only the quality metadata of an already
        # prepared result. Low-ratio creation and actual solving are covered
        # above; multiview inference must neither reinterpret that warning as
        # rejection nor erase an independently supplied hard rejection reason.
        prepared = replace(prepared, diagnostics={**prepared.diagnostics,
                                                  "quality_warnings": [WARNING]})
        for reasons in ((), ("weak_correspondence_consensus",)):
            with self.subTest(reasons=reasons):
                supplied = replace(prepared, reasons=reasons)
                result = infer_multiview_region(source, targets, supplied, bank)
                self.assertEqual(result["accepted"], not reasons)
                self.assertEqual(result["reasons"], list(reasons))
                self.assertEqual(result["diagnostics"]["quality_warnings"], [WARNING])
                self.assertTrue(result["modes"])
                np.testing.assert_array_equal(result["sim3_translation"], prepared.translation)
        self.assertEqual(prepared.diagnostics["quality_warnings"], [WARNING])
    # END multiview consumer warning and hard-rejection regression.


class TestActualLowConsensusGeometry(unittest.TestCase):
    """A deterministic solver example documents both utility and limits."""

    def test_actual_solver_can_accept_a_correct_low_consensus_location(self):
        points, pixels, region, source_ids, target_ids = _fixed_plane_correspondences(0)
        prepared = prepare_anchors(points, points, source_ids, target_ids)
        result = infer_region(region, prepared, method="relational",
                              **_controls(points, target_pixels=pixels))
        self.assertGreaterEqual(prepared.diagnostics["inliers"], 6)
        self.assertLess(prepared.diagnostics["inlier_ratio"], 0.2)
        self.assertEqual(result["diagnostics"]["quality_warnings"], [WARNING])
        self.assertTrue(result["accepted"], result["reasons"])
        np.testing.assert_allclose(result["modes"][0]["center"], points[region].mean(0), atol=1e-12)

    def test_actual_wrong_consensus_can_pass_every_other_gate_at_100_mm_error(self):
        points, pixels, region, source_ids, target_ids = _fixed_plane_correspondences(20)
        prepared = prepare_anchors(points, points, source_ids, target_ids)
        self.assertEqual(prepared.diagnostics["inliers"], 12)
        self.assertEqual(prepared.diagnostics["inlier_ratio"], 0.15)
        np.testing.assert_allclose(prepared.translation, [0.1, 0., 0.], atol=1e-12)
        truth_center = points[region].mean(0)
        # Disabling only the warning threshold must not change the registration
        # or any output geometry. This is a comparison control, not a fallback.
        without_warning = prepare_anchors(points, points, source_ids, target_ids,
                                         config=TransferConfig(min_inlier_ratio=0.0))
        np.testing.assert_array_equal(prepared.rotation, without_warning.rotation)
        np.testing.assert_array_equal(prepared.translation, without_warning.translation)
        np.testing.assert_array_equal(prepared.src_anchor_idx, without_warning.src_anchor_idx)
        np.testing.assert_array_equal(prepared.tgt_anchor_idx, without_warning.tgt_anchor_idx)
        self.assertEqual(prepared.scale, without_warning.scale)
        for method in ("global_sim3", "relational", "relation_only"):
            with self.subTest(method=method):
                controls = _controls(points, target_pixels=pixels)
                result = infer_region(region, prepared, method=method, **controls)
                control = infer_region(region, without_warning, method=method, **controls)
                self.assertTrue(result["accepted"], result["reasons"])
                self.assertEqual(result["reasons"], [])
                self.assertEqual(result["diagnostics"]["quality_warnings"], [WARNING])
                self.assertEqual(control["diagnostics"]["quality_warnings"], [])
                self.assertAlmostEqual(np.linalg.norm(result["modes"][0]["center"] - truth_center), 0.1)
                self.assertLess(result["diagnostics"]["center_signature_score"], 1e-12)
                self.assertEqual(len(result["modes"]), len(control["modes"]))
                for mode, unchanged in zip(result["modes"], control["modes"]):
                    for field in ("seed_index", "center", "normal", "mask", "sample_indices"):
                        np.testing.assert_array_equal(mode[field], unchanged[field])
                    for field in ("score", "relation_score", "type", "diagnostics"):
                        self.assertEqual(mode[field], unchanged[field])


if __name__ == "__main__":
    unittest.main()
# END LOW_CONSENSUS_QUALITY_WARNING_REGRESSIONS
