"""CPU checks for independent camera topology and shared-center sampling.

These fixtures supply known observed planes and exact correspondence identities;
they never load SAM, MARCO weights, a simulator, or target evaluation files.
Feature fusion has its own tests.  Here the boundary is a frozen registration
result plus two original camera observations and their explicit index maps.
"""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
from scipy.spatial.transform import Rotation

from transfer import core
from transfer.evaluation import evaluate_prediction
from transfer.experiment import store_prediction
from transfer.multiview_grounding import infer_multiview_region


# BEGIN independent dual-view neighborhood regression fixtures and tests.
def _observation(side=9, spacing=0.005, shift=(0.0, 0.0, 0.0), keep=None):
    """Observe a frontoparallel plane with exact optical depth of one metre."""
    y, x = np.mgrid[:side, :side]
    pixels = np.column_stack((x.ravel(), y.ravel())).astype(np.int64)
    center = (side - 1) / 2
    points = np.column_stack(((x.ravel() - center) * spacing,
                              (y.ravel() - center) * spacing, np.ones(x.size)))
    points += np.asarray(shift)
    pose = np.eye(4)
    pose[:3, 3] = np.asarray(shift)
    mask = np.ones((side, side), dtype=bool)
    if keep is not None:
        keep = np.asarray(keep, dtype=bool)
        mask.flat[:] = keep
        points, pixels = points[keep], pixels[keep]
    return {"xyz": points, "pixels": pixels, "object_mask": mask,
            "K": np.array([[1 / spacing, 0, center], [0, 1 / spacing, center], [0, 0, 1.]]),
            "T_world_camera_cv": pose, "region_mask": np.ones(len(points), dtype=bool)}


def _moved(observation, scale, rotation, translation):
    """Move the measured points and viewing pose consistently with a known Sim(3)."""
    moved = {key: value.copy() for key, value in observation.items()}
    moved["xyz"] = scale * observation["xyz"] @ rotation.T + translation
    moved["T_world_camera_cv"][:3, :3] = rotation @ observation["T_world_camera_cv"][:3, :3]
    moved["T_world_camera_cv"][:3, 3] = scale * rotation @ observation["T_world_camera_cv"][:3, 3] + translation
    return moved


def _targets(a, b):
    """No feature or depth arrays are required by the already-grounded sampler."""
    return [{"id": "a", "observation": a}, {"id": "b", "observation": b}]


def _bank(targets):
    """Declare both index domains independently of the sampler implementation."""
    ordered = sorted(targets, key=lambda target: target["id"])
    counts = [len(target["observation"]["xyz"]) for target in ordered]
    observed = np.concatenate([target["observation"]["xyz"] for target in ordered])
    points, first, inverse = np.unique(observed, axis=0, return_index=True, return_inverse=True)
    return {"xyz": points, "observation_xyz": observed,
            "observation_view_index": np.repeat(np.arange(2), counts),
            "observation_local_index": np.concatenate([np.arange(count) for count in counts]),
            "observation_candidate_index": inverse, "candidate_first_observation_index": first,
            "view_ids": [target["id"] for target in ordered],
            "view_offsets": np.r_[0, np.cumsum(counts)]}


def _prepared(source, targets, *, algorithm="ransac", source_indices=None, config=None):
    """Use independently specified source-to-A correspondences, never geometry truth files."""
    bank = _bank(targets)
    source_indices = np.arange(len(source["xyz"])) if source_indices is None else np.asarray(source_indices)
    a = next(target["observation"] for target in targets if target["id"] == "a")
    # Fixtures explicitly declare the matching A observation order; this is a
    # known synthetic correspondence input, not nearest-neighbor inference.
    if len(source_indices) != len(a["xyz"]):
        raise ValueError("The fixture correspondence order must describe every A observation")
    candidate_indices = bank["observation_candidate_index"][:len(a["xyz"])]
    config = core.TransferConfig(ransac_iterations=40) if config is None else config
    prepared = core.prepare_anchors(source["xyz"], bank["xyz"], source_indices, candidate_indices,
        mode="direct", config=config, registration_algorithm=algorithm,
        correspondence_scores=np.linspace(1.0, 0.5, len(source_indices)))
    return bank, prepared


class TestMultiviewNeighborhoods(unittest.TestCase):
    """Preserve geometric acceptance checks without faking a combined camera."""

    def test_known_similarity_all_three_backends_and_no_second_solver_call(self):
        source = _observation()
        rotation = Rotation.from_rotvec([0.21, -0.15, 0.33]).as_matrix()
        translation = np.array([0.17, -0.23, 0.08])
        scale = 1.25
        targets = _targets(_moved(source, scale, rotation, translation),
            _moved(_observation(shift=(0.01, 0, 0)), scale, rotation, translation))
        expected = scale * rotation @ source["xyz"].mean(0) + translation
        for algorithm in ("ransac", "prosac_lo", "teaserpp"):
            with self.subTest(algorithm=algorithm):
                bank, prepared = _prepared(source, targets, algorithm=algorithm)
                self.assertIsNotNone(prepared.scale)
                with mock.patch("transfer.core.register_sim3", side_effect=AssertionError("second fit")):
                    result = infer_multiview_region(source, targets, prepared, bank)
                self.assertTrue(result["accepted"], result["reasons"])
                np.testing.assert_allclose(result["modes"][0]["center"], expected, atol=1e-12, rtol=0)
                np.testing.assert_allclose(bank["observation_xyz"][result["tgt_anchor_idx"]],
                                           bank["xyz"][prepared.tgt_anchor_idx], atol=0, rtol=0)
                self.assertEqual(result["diagnostics"]["registration_target_candidate_indices"],
                                 prepared.tgt_anchor_idx.tolist())
                self.assertLessEqual(len(result["modes"]), 3)
                for mode in result["modes"]:
                    # Every returned connected neighborhood belongs to one
                    # original image, even where the physical planes overlap.
                    self.assertTrue(np.all(bank["observation_view_index"][mode["mask"]] == mode["view_index"]))
                    view = targets[mode["view_index"]]["observation"]
                    np.testing.assert_array_equal(mode["sample_pixels"], view["pixels"][mode["sample_local_indices"]])

    def test_duplicate_view_does_not_consume_mode_budget_and_order_is_canonical(self):
        source = _observation()
        targets = _targets(_observation(), _observation())
        bank, prepared = _prepared(source, targets)
        forward = infer_multiview_region(source, targets, prepared, bank)
        reverse = infer_multiview_region(source, list(reversed(targets)), prepared, bank)
        self.assertTrue(forward["accepted"])
        self.assertEqual(len(forward["modes"]), 1)
        self.assertEqual(len(reverse["modes"]), 1)
        np.testing.assert_array_equal(forward["modes"][0]["sample_indices"], reverse["modes"][0]["sample_indices"])
        self.assertEqual(forward["modes"][0]["view_id"], "a")

    def test_identical_pixels_in_distant_cameras_never_form_one_component(self):
        source = _observation()
        targets = _targets(_observation(), _observation(shift=(0.1, 0, 0)))
        bank, prepared = _prepared(source, targets)
        result = infer_multiview_region(source, targets, prepared, bank, max_modes=2)
        self.assertTrue(result["accepted"], result["reasons"])
        self.assertEqual(len(result["modes"]), 2)
        self.assertEqual({mode["view_id"] for mode in result["modes"]}, {"a", "b"})
        for mode in result["modes"]:
            self.assertEqual(len(np.unique(bank["observation_view_index"][mode["mask"]])), 1)

    def test_missing_pixel_column_stays_a_hole_with_an_explicit_empty_second_view(self):
        source = _observation()
        keep = source["pixels"][:, 0] != 4
        targets = _targets(_observation(keep=keep), _observation(keep=np.zeros(81, dtype=bool)))
        bank, prepared = _prepared(source, targets, source_indices=np.flatnonzero(keep))
        result = infer_multiview_region(source, targets, prepared, bank)
        self.assertTrue(result["accepted"], result["reasons"])
        self.assertGreaterEqual(len(result["modes"]), 2)
        for mode in result["modes"]:
            self.assertEqual(mode["view_id"], "a")
            columns = targets[0]["observation"]["pixels"][mode["mask"][:keep.sum()], 0]
            self.assertFalse(np.any(columns < 4) and np.any(columns > 4))
        self.assertEqual(result["diagnostics"]["per_view_surface_geometry"][1]["observed_points"], 0)

    def test_empty_evidence_and_failed_registration_remain_rejected(self):
        source = _observation()
        empty = _observation(keep=np.zeros(81, dtype=bool))
        targets = _targets(empty, empty)
        bank = _bank(targets)
        prepared = core.prepare_anchors(source["xyz"], bank["xyz"], np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64), mode="direct")
        result = infer_multiview_region(source, targets, prepared, bank)
        self.assertFalse(result["accepted"])
        self.assertEqual(result["reasons"], ["insufficient_correspondences"])
        self.assertEqual(result["modes"], [])

    def test_existing_consensus_failure_is_not_erased_by_visible_surface(self):
        source = _observation()
        targets = _targets(_observation(), _observation())
        bank, prepared = _prepared(source, targets)
        prepared = replace(prepared, reasons=("weak_correspondence_consensus",))
        result = infer_multiview_region(source, targets, prepared, bank)
        self.assertFalse(result["accepted"])
        self.assertIn("weak_correspondence_consensus", result["reasons"])
        self.assertTrue(result["modes"])

    def test_far_surface_projection_is_rejected_without_snapping_the_transform(self):
        source = _observation()
        targets = _targets(_observation(), _observation())
        bank, prepared = _prepared(source, targets)
        prepared = replace(prepared, translation=prepared.translation + [0, 0, 0.05])
        result = infer_multiview_region(source, targets, prepared, bank)
        self.assertFalse(result["accepted"])
        self.assertIn("predicted_center_without_nearby_surface", result["reasons"])
        self.assertAlmostEqual(result["diagnostics"]["center_projection_distance_m"], 0.05)
        np.testing.assert_array_equal(result["sim3_translation"], prepared.translation)

    def test_different_surface_ambiguity_is_checked_beyond_mode_budget(self):
        source = _observation()
        targets = _targets(_observation(), _observation(shift=(0, 0, 0.02)))
        bank, prepared = _prepared(source, targets)
        prepared = replace(prepared, translation=prepared.translation + [0, 0, 0.01])
        result = infer_multiview_region(source, targets, prepared, bank, max_modes=1)
        self.assertFalse(result["accepted"])
        self.assertIn("surface_ambiguity", result["reasons"])
        self.assertTrue(result["diagnostics"]["surface_ambiguity"])
        self.assertEqual(len(result["modes"]), 1)
        self.assertIsNotNone(result["diagnostics"]["selected_alternate_surface_index"])

    def test_source_normal_still_requires_observable_orientation(self):
        source = _observation()
        targets = _targets(_observation(), _observation())
        bank, prepared = _prepared(source, targets)
        source["T_world_camera_cv"][:3, 3] = source["xyz"].mean(0) + [1, 0, 0]
        result = infer_multiview_region(source, targets, prepared, bank)
        self.assertFalse(result["accepted"])
        self.assertIn("source_normal_orientation_unobservable", result["reasons"])
        self.assertEqual(result["modes"], [])

    def test_index_domain_mismatches_and_missing_view_are_explicit(self):
        source = _observation()
        targets = _targets(_observation(), _observation())
        bank, prepared = _prepared(source, targets)
        with self.assertRaisesRegex(ValueError, "Exactly two"):
            infer_multiview_region(source, targets[:1], prepared, bank)
        wrong = dict(bank, observation_candidate_index=np.roll(bank["observation_candidate_index"], 1))
        with self.assertRaisesRegex(ValueError, "changes measured coordinates"):
            infer_multiview_region(source, targets, prepared, wrong)
        wrong = dict(bank, view_ids=["b", "a"])
        with self.assertRaisesRegex(ValueError, "canonical target ID order"):
            infer_multiview_region(source, targets, prepared, wrong)
        wrong = replace(prepared, mode="automatic")
        with self.assertRaisesRegex(ValueError, "direct PreparedAnchors"):
            infer_multiview_region(source, targets, wrong, bank)

    def test_existing_prediction_storage_and_evaluation_use_observation_indices(self):
        source = _observation()
        targets = _targets(_observation(), _observation(shift=(0.01, 0, 0)))
        bank, prepared = _prepared(source, targets)
        result = infer_multiview_region(source, targets, prepared, bank)
        truth = {"support_id": np.zeros(len(bank["observation_xyz"]), dtype=np.int64),
                 "preferred_support_id": np.asarray(0, dtype=np.int64)}
        metrics = evaluate_prediction(result, bank["observation_xyz"], truth, source["xyz"].mean(0))
        self.assertTrue(metrics["accepted_top1_preferred_support_hit"])
        self.assertLess(metrics["center_error_m"], 1e-12)
        self.assertLessEqual(metrics["maximum_tangent_radius_m"], 0.03 * (1 + 1e-12))
        self.assertGreaterEqual(metrics["minimum_sample_spacing_m"], 0.005 * (1 - 1e-12))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "prediction"
            store_prediction(output, result, metrics)
            metadata = json.loads(output.with_suffix(".json").read_text())
            self.assertEqual(metadata["diagnostics"]["point_index_domain"], "candidate_bank.observation_xyz")
            with np.load(output.with_suffix(".npz"), allow_pickle=False) as arrays:
                np.testing.assert_array_equal(arrays["tgt_anchor_idx"], result["tgt_anchor_idx"])
                np.testing.assert_array_equal(arrays["mode_0_mask"], result["modes"][0]["mask"])
# END independent dual-view neighborhood regression fixtures and tests.


if __name__ == "__main__":
    unittest.main()
