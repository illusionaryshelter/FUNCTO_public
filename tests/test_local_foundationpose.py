"""CPU contracts for native rigid-pose state, geometry checks and file transport."""

# BEGIN FoundationPose adapter regression tests; no model, GPU or API is used.
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import trimesh

from local_models import foundationpose as fp
from local_models import runtime


class FakeEstimator:
    """Implement the actual register/track_one boundary with independent state."""

    def __init__(self, entity, mask):
        self.entity = entity
        self.mask = mask
        self.diameter = .2
        self.model_center = np.array([.03, -.02, .01])
        self.pose_last = None
        self.calls = []

    def register(self, *, K, rgb, depth, ob_mask, iteration):
        self.calls.append(("register", iteration))
        np.testing.assert_array_equal(ob_mask, self.mask)
        assert rgb.dtype == np.uint8 and rgb[0, 0, 2] == 255
        assert depth.dtype == np.float32
        np.testing.assert_allclose(depth, 1.)
        self.pose_last = np.eye(4)
        self.pose_last[:3, 3] = [.1 * self.entity, -.05, 1.]
        self.scores = np.arange(252, dtype=np.float32)
        return self.pose_last.copy()

    def track_one(self, *, rgb, depth, K, iteration):
        self.calls.append(("track", iteration))
        assert self.pose_last is not None
        self.pose_last[0, 3] += .01 * (self.entity + 1)
        return self.pose_last.copy()


class FoundationPoseContractTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.frames, self.depths = [], []
        for index in range(2):
            rgb_path, depth_path = self.root / f"rgb-{index}.png", self.root / f"depth-{index}.png"
            Image.new("RGB", (32, 24), (index, 128, 255)).save(rgb_path)
            Image.fromarray(np.full((24, 32), 1000, np.uint16)).save(depth_path)
            self.frames.append(rgb_path)
            self.depths.append(depth_path)
        self.K = np.array([[100., 0, 16], [0, 100, 12], [0, 0, 1]])
        self.masks = np.zeros((2, 24, 32), dtype=bool)
        self.masks[0, 2:12, 2:12] = True
        self.masks[1, 12:22, 20:30] = True
        mesh = trimesh.creation.box(extents=(.1, .12, .08))
        mesh.visual.vertex_colors = [180, 80, 20, 255]
        mesh_path = self.root / "model.ply"
        mesh.export(mesh_path)
        self.models = [{"mesh": str(mesh_path), "scale": 1.} for _ in range(2)]

    def run_sequence(self, estimators=None, renderer=None):
        estimators = estimators or [FakeEstimator(i, mask) for i, mask in enumerate(self.masks)]
        def render(estimator, pose, K, H, W):
            self.assertEqual((H, W), (24, 32))
            return estimator.mask.astype(np.float32)
        return fp._track_sequence(self.frames, self.depths, self.K, .001, self.masks,
                                  estimators, renderer or render), estimators

    def test_native_registration_tracking_and_state_are_isolated_by_model(self):
        result, estimators = self.run_sequence()
        self.assertEqual(result["camera_from_model"].shape, (2, 2, 4, 4))
        self.assertTrue(result["validated"].all())
        np.testing.assert_array_equal(result["registration_candidate_count"], [252, 252])
        np.testing.assert_array_equal(result["support_pixels"], 100)
        np.testing.assert_array_equal(result["depth_median_error_m"], 0)
        for est in estimators:
            self.assertEqual(est.calls, [("register", 5), ("track", 2)])
        np.testing.assert_allclose(result["camera_from_model"][:, :, 0, 3], [[0, .1], [.01, .12]])

    def test_public_worker_contract_contains_explicit_models_and_typed_masks(self):
        arrays, _ = self.run_sequence()
        with patch("local_models.runtime.run_file_worker", return_value=arrays) as worker:
            result = fp.track(self.frames, self.depths, K=self.K, depth_scale=.001,
                              masks=self.masks, models=self.models)
        self.assertIs(result["camera_from_model"], arrays["camera_from_model"])
        call = worker.call_args.kwargs
        self.assertEqual(call["kind"], "foundationpose")
        self.assertEqual(call["paths"], self.frames)
        np.testing.assert_array_equal(call["options"]["masks"], self.masks)
        self.assertEqual(call["options"]["depth_paths"], [str(path) for path in self.depths])
        self.assertEqual(call["options"]["models"][0]["scale"], [1., 1., 1.])

    def test_finite_guess_pose_is_rejected_when_registration_did_not_complete(self):
        class Incomplete(FakeEstimator):
            def register(self, **kwargs):
                self.scores = np.ones(252)  # Even stale scores cannot grant tracking permission.
                return np.eye(4)
        estimators = [Incomplete(0, self.masks[0]), FakeEstimator(1, self.masks[1])]
        with self.assertRaisesRegex(ValueError, "registration did not complete"):
            self.run_sequence(estimators)
        self.assertEqual(estimators[1].calls, [])

    def test_rigid_matrix_without_rendered_depth_support_does_not_pass(self):
        calls = []
        def render(estimator, pose, K, H, W):
            calls.append((estimator.entity, len(estimator.calls)))
            depth = estimator.mask.astype(float)
            if len(estimator.calls) > 1:
                depth *= 1.2
            return depth
        with self.assertRaisesRegex(ValueError, "frame 1.*No sufficient model/depth support"):
            self.run_sequence(renderer=render)
        self.assertEqual(calls, [(0, 1), (1, 1), (0, 2)])

    def test_partial_occlusion_reports_all_depth_errors_without_requiring_all_pixels(self):
        render = np.ones((10, 10))
        measured = render.copy()
        measured[:8] = .5
        result = fp._geometry(measured, render, .01)
        self.assertEqual(result["support_pixels"], 20)
        self.assertEqual(result["support_fraction"], .2)
        self.assertEqual(result["nonoccluded_pixels"], 20)
        self.assertEqual(result["occluded_pixels"], 80)
        self.assertEqual(result["free_space_conflict_pixels"], 0)
        self.assertEqual(result["nonoccluded_support_fraction"], 1.)
        self.assertEqual(result["depth_median_error_m"], .5)
        self.assertEqual(result["depth_p90_error_m"], .5)

    def test_farther_measured_depth_is_free_space_evidence_not_an_occluder(self):
        rendered = np.full((20, 20), 2.)
        measured = np.full_like(rendered, 3.)
        measured[:2] = 2.
        # Ten percent correct depth must not hide the remaining model surface
        # occupying space through which the camera actually saw farther depth.
        with self.assertRaisesRegex(ValueError, "360 free-space conflicts"):
            fp._geometry(measured, rendered, .01)
        measured[2:] = 1.
        visible_mask = np.zeros_like(measured, dtype=bool)
        visible_mask[:2] = True
        result = fp._geometry(measured, rendered, .01, visible_mask)
        self.assertEqual(result["support_fraction"], .1)
        self.assertEqual(result["occluded_pixels"], 360)
        self.assertEqual(result["nonoccluded_support_fraction"], 1.)
        self.assertEqual(result["initial_mask_depth_coverage"], 1.)
        # Exactly half is not a majority, even if the absolute pixel count is high.
        measured[:] = 3.
        measured[:10] = 2.
        with self.assertRaisesRegex(ValueError, "majority of its nonoccluded projection"):
            fp._geometry(measured, rendered, .01)

    def test_initial_instance_support_is_bidirectional_not_a_small_patch_match(self):
        measured = np.ones((20, 20))
        undersized_render = np.zeros_like(measured)
        undersized_render[:2] = 1.
        with self.assertRaisesRegex(ValueError, "majority of the initial mask"):
            fp._geometry(measured, undersized_render, .01, np.ones_like(measured, dtype=bool))
        initial_mask = undersized_render > 0
        # A coplanar background can support a wrong oversized model in depth;
        # that agreement is outside the selected instance and cannot count.
        with self.assertRaisesRegex(ValueError, "majority of its nonoccluded projection"):
            fp._geometry(measured, measured, .01, initial_mask)

    def test_mask_overlap_and_invalid_render_are_explicit_failures(self):
        projected = np.zeros((10, 20))
        projected[:, :10] = 1
        opposite_mask = projected == 0
        with self.assertRaisesRegex(ValueError, "initial mask"):
            fp._geometry(np.ones_like(projected), projected, .01, opposite_mask)
        with self.assertRaisesRegex(ValueError, "invalid depth"):
            fp._geometry(np.ones((10, 10)), np.full((10, 10), np.nan), .01)
        with self.assertRaisesRegex(ValueError, "No sufficient"):
            fp._geometry(np.zeros((10, 10)), np.ones((10, 10)), .01)

    def test_render_frame_converts_original_mesh_pose_to_internal_centred_mesh(self):
        pose = np.eye(4)
        pose[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        pose[:3, 3] = [.3, -.2, .7]
        centre = np.array([.04, .05, -.03])
        original_points = np.array([[.1, .2, .3], [-.2, .4, .2]])
        expected = original_points @ pose[:3, :3].T + pose[:3, 3]
        centred_pose = fp._centered_pose(pose, centre)
        actual = (original_points - centre) @ centred_pose[:3, :3].T + centred_pose[:3, 3]
        np.testing.assert_allclose(actual, expected, atol=1e-15)
        np.testing.assert_array_equal(pose[:3, 3], [.3, -.2, .7])

    def test_mesh_scaling_preserves_the_original_local_origin(self):
        mesh = trimesh.creation.box(extents=(2, 4, 6))
        mesh.apply_translation([10, -20, 30])
        path = self.root / "offset.ply"
        mesh.export(path)
        loaded = fp._load_mesh({"mesh": str(path), "scale": [.01, .02, .03]})
        np.testing.assert_allclose(loaded.bounds.mean(axis=0), [.1, -.4, .9], atol=1e-8)
        np.testing.assert_allclose(loaded.extents, [.02, .08, .18], atol=1e-8)

    def test_point_cloud_is_not_silently_treated_as_a_triangle_model(self):
        path = self.root / "points.ply"
        trimesh.PointCloud([[0, 0, 0], [1, 0, 0], [0, 1, 0]]).export(path)
        with self.assertRaisesRegex(ValueError, "one triangle mesh"):
            fp._load_mesh({"mesh": str(path), "scale": [1, 1, 1]})

    def test_invalid_model_mask_units_and_skew_fail_before_worker(self):
        valid = dict(K=self.K, depth_scale=.001, masks=self.masks, models=self.models)
        skew = self.K.copy()
        skew[0, 1] = .2
        cases = [dict(valid, K=skew), dict(valid, depth_scale=True), dict(valid, depth_scale=0),
                 dict(valid, masks=self.masks.astype(np.uint8)), dict(valid, models=[]),
                 dict(valid, models=[dict(self.models[0], scale=-1), self.models[1]]),
                 dict(valid, models=[dict(self.models[0], scale=[1, 2]), self.models[1]]),
                 dict(valid, models=[dict(self.models[0], scale=True), self.models[1]])]
        with patch("local_models.runtime.run_file_worker") as worker:
            for kwargs in cases:
                with self.subTest(kwargs=list(kwargs)), self.assertRaises(ValueError):
                    fp.track(self.frames, self.depths, **kwargs)
            with self.assertRaises(FileNotFoundError):
                fp.track(self.frames, self.depths, **dict(valid, models=[
                    {"mesh": str(self.root / "missing.obj")}, self.models[1]]))
            worker.assert_not_called()

    def test_worker_output_requires_explainable_support_for_each_pose(self):
        valid, _ = self.run_sequence()
        bad = []
        for key, value in [("validated", np.array([[True, True], [False, True]])),
                           ("support_pixels", np.zeros((2, 2), dtype=int)),
                           ("nonoccluded_support_fraction", np.full((2, 2), .5)),
                           ("free_space_conflict_pixels", np.full((2, 2), 100)),
                           ("initial_mask_iou", np.zeros(2)),
                           ("initial_mask_depth_coverage", np.full(2, .5)),
                           ("registration_candidate_count", np.ones(2, dtype=int)),
                           ("camera_from_model", np.full((2, 2, 4, 4), np.nan))]:
            bad.append(dict(valid, **{key: value}))
        bad.append({key: value for key, value in valid.items() if key != "validated"})
        for output in bad:
            with self.subTest(keys=list(output)), self.assertRaises(ValueError):
                fp._outputs(output, 2, 2)
        reflected = np.eye(4)
        reflected[0, 0] = -1
        with self.assertRaisesRegex(ValueError, "invalid rigid"):
            fp._pose(reflected)

    def test_file_runtime_writes_masks_as_npy_and_never_starts_a_model_in_test(self):
        options = {"depth_paths": [str(path) for path in self.depths], "K": self.K.tolist(),
                   "depth_scale": .001, "masks": self.masks, "models": self.models}
        def fake_subprocess(command, **kwargs):
            request = json.loads(Path(command[-1]).read_text())
            self.assertEqual(request["kind"], "foundationpose")
            self.assertNotIn("masks", request["options"])
            np.testing.assert_array_equal(np.load(request["options"]["masks_path"], allow_pickle=False),
                                          self.masks)
            self.assertEqual(kwargs["env"]["HF_HUB_OFFLINE"], "1")
            self.assertEqual(kwargs["env"]["PYTHONNOUSERSITE"], "1")
            np.savez_compressed(request["output"], checked=np.array([True]))
        with patch("local_models.runtime.subprocess.run", side_effect=fake_subprocess) as process:
            result = runtime.run_file_worker("foundationpose", self.frames, options)
        process.assert_called_once()
        self.assertTrue(result["checked"][0])
        self.assertIs(options["masks"], self.masks)


if __name__ == "__main__":
    unittest.main()
# END FoundationPose adapter regression tests.
