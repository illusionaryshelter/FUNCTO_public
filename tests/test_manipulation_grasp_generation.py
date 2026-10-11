"""CPU checks of candidate evidence, robot-frame binding and model failures."""

# BEGIN independent evidence and exact-prior candidate regressions.
import builtins
from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

from manipulation.grasp_generation import (
    GraspGenerationRequest, GraspGenXBackend, generate_grasp_candidates)
from manipulation.tool_geometry import load_robosuite_panda_tool


def pose(position=(0., 0., 0.), rotation=(0., 0., 0.)):
    result = np.eye(4)
    result[:3, 3] = position
    result[:3, :3] = Rotation.from_rotvec(rotation).as_matrix()
    return result


class Backend:
    """Model substitute only; it makes no claims about physical validation."""

    def __init__(self, generated, quality=.4, binding=None):
        self.T_tool_grasp = np.eye(4) if binding is None else binding
        self.generated = generated
        self.quality = quality
        self.scored = None
        self.prepared = None
        self.calls = []

    def prepare_cloud(self, points, *, seed):
        self.calls.append("prepare")
        self.prepared = points[:100].copy()
        return self.prepared, {"model_points": len(self.prepared)}

    def generate(self, points, *, seed):
        self.calls.append("generate")
        if points is not self.prepared:
            raise AssertionError("generation must use the shared prepared cloud")
        return self.generated, ["diffusion", "obb"], {}

    def score(self, grasps, points, *, seed):
        self.calls.append("score")
        if points is not self.prepared:
            raise AssertionError("scoring must use the shared prepared cloud")
        self.scored = grasps.copy()
        values = np.full(len(grasps), self.quality)
        values[0] = 0.  # A zero-score prior remains a candidate, without priority.
        values[-1] = 1.
        return values


class GraspGenerationTest(unittest.TestCase):
    def setUp(self):
        self.points = np.random.default_rng(4).uniform(-.02, .02, (200, 3))
        self.prior = pose((.317283945123, -.142684128134, .89370156273), (.2, -.3, .1))
        self.binding = pose((.02, -.01, -.097), (.1, .2, -.4))
        self.alternative = pose((.35, -.18, .90), (-.3, .2, .7))
        self.generated = np.stack((self.alternative @ self.binding,
                                   self.alternative @ pose(rotation=(0., 0., np.pi)) @ self.binding))

    def request(self, **kwargs):
        return GraspGenerationRequest(self.points, self.prior, self.binding, **kwargs)

    def test_exact_prior_and_all_three_sources_share_scoring_without_quality_pruning(self):
        backend = Backend(self.generated, binding=self.binding)
        result = generate_grasp_candidates(self.request(), backend, seed=17)
        # BEGIN preservation is independent of the candidate's search rank.
        exact = next(row for row in result["candidates"] if row["source"] == "demo_exact")
        np.testing.assert_array_equal(exact["T_reference_tool"], self.prior)
        self.assertEqual(exact["quality"], 0.)
        self.assertGreater(exact["rank"], 0)
        self.assertGreater(result["candidates"][0]["ranking_score"], exact["ranking_score"])
        scores = [row["ranking_score"] for row in result["candidates"]]
        self.assertTrue(all(a >= b for a, b in zip(scores, scores[1:])))
        # END all sources compete using their existing shared score.
        self.assertEqual(backend.calls, ["prepare", "generate", "score"])
        self.assertEqual(len(backend.scored), 195)
        self.assertEqual(len(result["candidates"]), 195)
        self.assertEqual(result["diagnostics"]["source_counts"],
                         {"demo_exact": 1, "demo_local": 192, "diffusion": 1, "obb": 1})
        self.assertFalse(result["diagnostics"]["execution_validated"])
        self.assertEqual(result["rejected_candidates"], [])

    # BEGIN ordinary prior preference and deterministic ties need no source privilege.
    def test_best_scoring_exact_prior_still_wins_without_a_special_branch(self):
        class PriorBest(Backend):
            def score(self, grasps, points, *, seed):
                values = np.zeros(len(grasps))
                values[0] = 1.
                return values

        result = generate_grasp_candidates(self.request(), PriorBest(self.generated, binding=self.binding))
        self.assertEqual(result["candidates"][0]["source"], "demo_exact")
        self.assertEqual(result["candidates"][0]["ranking_score"], 1.)

    def test_soft_prior_can_outweigh_a_farther_candidates_higher_model_quality(self):
        class SoftPreference(Backend):
            def score(self, grasps, points, *, seed):
                values = np.zeros(len(grasps))
                values[0], values[-1] = .8, 1.
                return values

        far = np.stack((pose((1., 0., 0.)), pose((2., 0., 0.))))
        request = GraspGenerationRequest(self.points, np.eye(4), np.eye(4))
        result = generate_grasp_candidates(request, SoftPreference(far))
        exact = next(row for row in result["candidates"] if row["source"] == "demo_exact")
        distant = next(row for row in result["candidates"] if row["source"] == "obb")
        self.assertLess(exact["quality"], distant["quality"])
        self.assertGreater(exact["ranking_score"], distant["ranking_score"])
        self.assertLess(exact["rank"], distant["rank"])

    def test_equal_scores_use_stable_ids_and_repeated_seed_keeps_the_order(self):
        class EqualQuality(Backend):
            def score(self, grasps, points, *, seed):
                return np.ones(len(grasps))

        request = GraspGenerationRequest(self.points, np.eye(4), np.eye(4))
        generated = np.stack((np.eye(4), np.eye(4)))
        first = generate_grasp_candidates(request, EqualQuality(generated), seed=17)
        second = generate_grasp_candidates(request, EqualQuality(generated), seed=17)
        tied = [row for row in first["candidates"] if row["ranking_score"] == 1.]
        self.assertEqual([row["source"] for row in tied], ["demo_exact", "diffusion", "obb"])
        self.assertEqual([row["candidate_id"] for row in tied], ["grasp_0000", "grasp_0193", "grasp_0194"])
        self.assertEqual([row["candidate_id"] for row in first["candidates"]],
                         [row["candidate_id"] for row in second["candidates"]])
    # END model quality is a soft ordering input, never an execution certificate.

    def test_nontrivial_robot_binding_and_reference_transform_are_applied_once(self):
        result = generate_grasp_candidates(self.request(), Backend(self.generated, binding=self.binding), seed=5)
        diff = next(item for item in result["candidates"] if item["source"] == "diffusion")
        np.testing.assert_allclose(diff["T_reference_tool"], self.alternative, atol=1e-12)
        world_change = pose((.7, -.6, .4), (.3, -.4, .2))
        shifted = GraspGenerationRequest(
            self.points @ world_change[:3, :3].T + world_change[:3, 3],
            world_change @ self.prior, self.binding)
        other = generate_grasp_candidates(shifted, Backend(world_change @ self.generated, binding=self.binding), seed=5)
        a = {item["candidate_id"]: item for item in result["candidates"]}
        b = {item["candidate_id"]: item for item in other["candidates"]}
        for key in a:
            np.testing.assert_allclose(b[key]["T_reference_tool"],
                                       world_change @ a[key]["T_reference_tool"], atol=1e-12)
            self.assertAlmostEqual(a[key]["ranking_score"], b[key]["ranking_score"], places=12)

    def test_explicit_hard_angle_keeps_rejection_evidence_and_is_not_relaxed(self):
        # Identity bindings make the inward axis unambiguous. Both generated
        # candidates oppose +Z and must be scored, recorded and then rejected.
        upside_down = pose(rotation=(np.pi, 0., 0.))
        request = GraspGenerationRequest(self.points, np.eye(4), np.eye(4),
                                         approach_reference=np.array([0., 0., 1.]),
                                         hard_max_approach_angle_rad=.2)
        backend = Backend(np.stack((upside_down, upside_down)))
        result = generate_grasp_candidates(request, backend)
        self.assertEqual(len(backend.scored), 195)
        generated_rejections = [item for item in result["rejected_candidates"]
                                if item["source"] in {"diffusion", "obb"}]
        self.assertEqual(len(generated_rejections), 2)
        self.assertTrue(all(item["rejection_reason"] == "explicit_hard_approach_angle"
                            for item in generated_rejections))
        self.assertTrue(all(item["approach_angle_rad"] <= .2 for item in result["candidates"]))
        unrestricted = GraspGenerationRequest(self.points, np.eye(4), np.eye(4),
                                              approach_reference=np.array([0., 0., 1.]))
        result = generate_grasp_candidates(unrestricted, Backend(np.stack((upside_down, upside_down))))
        self.assertEqual(len(result["candidates"]), 195)

    def test_inadmissible_exact_prior_does_not_override_explicit_hard_angle(self):
        request = GraspGenerationRequest(self.points, pose(rotation=(np.pi, 0., 0.)), np.eye(4),
                                         approach_reference=np.array([0., 0., 1.]),
                                         hard_max_approach_angle_rad=.2)
        result = generate_grasp_candidates(request, Backend(np.stack((np.eye(4), np.eye(4)))))
        self.assertNotIn("demo_exact", [item["source"] for item in result["candidates"]])
        self.assertIn("demo_exact", [item["source"] for item in result["rejected_candidates"]])

    def test_model_exception_is_not_replaced_by_prior_or_empty_feasibility(self):
        class Broken(Backend):
            def score(self, *args, **kwargs):
                raise RuntimeError("model failure evidence")
        with self.assertRaisesRegex(RuntimeError, "model failure evidence"):
            generate_grasp_candidates(self.request(), Broken(self.generated, binding=self.binding))

    def test_invalid_direction_and_scores_fail_before_claiming_a_candidate_pool(self):
        for kwargs in ({"hard_max_approach_angle_rad": .2},
                       {"approach_reference": np.zeros(3)},
                       {"approach_reference": [0, 0, 1], "hard_max_approach_angle_rad": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                generate_grasp_candidates(self.request(**kwargs), Backend(self.generated, binding=self.binding))
        with self.assertRaisesRegex(ValueError, "finite"):
            generate_grasp_candidates(self.request(), Backend(self.generated, quality=np.nan, binding=self.binding))

    def test_non_axis_aligned_exact_direction_is_accepted_at_zero_hard_angle(self):
        grasp = self.prior @ self.binding
        request = self.request(approach_reference=grasp[:3, 2].copy(),
                               hard_max_approach_angle_rad=0.)
        result = generate_grasp_candidates(request, Backend(self.generated, binding=self.binding))
        # BEGIN exact direction acceptance does not imply unconditional rank zero.
        exact = next(row for row in result["candidates"] if row["source"] == "demo_exact")
        self.assertEqual(exact["approach_angle_rad"], 0.)
        # END the explicit angle gate is independent of the common ranking.

    def test_mismatched_robot_binding_fails_before_model_inference(self):
        backend = Backend(self.generated)
        with self.assertRaisesRegex(ValueError, "native robot binding"):
            generate_grasp_candidates(self.request(), backend)
        self.assertEqual(backend.calls, [])

    def test_native_asset_controls_grasp_base_and_both_conditioning_boxes(self):
        tool = load_robosuite_panda_tool(tool_id="robot_tool", tool_frame="measured_site")
        output = tool.grasp_generation_geometry()
        expected = np.eye(4)
        expected[2, 3] = -.097
        np.testing.assert_allclose(output["T_tool_grasp"], expected, atol=1e-12)
        params = output["sweep_volume_params"]
        np.testing.assert_allclose(params["extents_open"], [.079, .016, .016], atol=1e-12)
        np.testing.assert_allclose(params["extents_mid"], [.039, .016, .016], atol=1e-12)
        np.testing.assert_allclose(params["offset_open"], [0., 0., .0934], atol=1e-12)
        np.testing.assert_allclose(params["offset_mid"], params["offset_open"], atol=1e-12)
        self.assertAlmostEqual(params["fingertip_depth"], .0934, places=12)
        # Moving either native finger before the export must not change this
        # fixed asset calibration or turn feedback into model conditioning.
        tool.at([.015, -.028])
        again = tool.grasp_generation_geometry()
        np.testing.assert_array_equal(again["T_tool_grasp"], output["T_tool_grasp"])
        self.assertEqual(output["source"]["kind"], "standalone_native_tool_asset")

    # BEGIN CPU construction checks of the actual installed model API boundary.
    def test_public_sweep_constructor_loads_real_local_config_without_serving_or_cuda(self):
        from graspgenx.x_grippers import XGripperInfo

        tool = load_robosuite_panda_tool(tool_id="known_tool", tool_frame="known_tcp")
        geometry = tool.grasp_generation_geometry()
        checkpoints = Path(__file__).resolve().parents[2] / "models/graspgenx_checkpoints/release"
        original_import = builtins.__import__

        def without_serving(name, *args, **kwargs):
            if name.startswith("graspgenx.serving") or name == "msgpack_numpy":
                raise AssertionError("Offline model construction must not import the optional messaging stack")
            return original_import(name, *args, **kwargs)

        # Only the weight-loading/CUDA sampler constructor is replaced. Config
        # loading and the public XGripperInfo factory are the installed library.
        with patch("graspgenx.grasp_server.GraspGenXSampler") as sampler:
            with patch("builtins.__import__", side_effect=without_serving):
                backend = GraspGenXBackend(checkpoints_dir=checkpoints, gripper_geometry=geometry)
        sampler.assert_called_once()
        kwargs = sampler.call_args.kwargs
        self.assertIsInstance(kwargs["gripper_info"], XGripperInfo)
        self.assertFalse(kwargs["use_tensorrt"])
        self.assertEqual(backend.model_point_limit, 3500)
        np.testing.assert_allclose(kwargs["gripper_info"].sweep_volume,
                                   [.079, .016, .016, 0., 0., .0934], atol=1e-8)
        np.testing.assert_array_equal(backend.T_tool_grasp, geometry["T_tool_grasp"])

    def test_asset_dependent_backbones_and_invalid_sweep_geometry_fail_before_sampler(self):
        from omegaconf import OmegaConf

        geometry = load_robosuite_panda_tool(tool_id="known_tool", tool_frame="known_tcp").grasp_generation_geometry()
        base_cfg = {"data": {"num_points": 3500},
                    "diffusion": {"gripper_backbone": "sweep_volume_v2"},
                    "discriminator": {"gripper_backbone": "sweep_volume_v2"}}
        with patch("graspgenx.grasp_server.GraspGenXSampler") as sampler:
            for side in ("diffusion", "discriminator"):
                cfg = deepcopy(base_cfg)
                cfg[side]["gripper_backbone"] = "pointnet"
                with self.subTest(side=side), patch("graspgenx.utils.checkpoint_io.load_model_cfg", return_value=OmegaConf.create(cfg)):
                    with self.assertRaisesRegex(ValueError, "requires real gripper assets"):
                        GraspGenXBackend(checkpoints_dir="unused", gripper_geometry=geometry)
            for invalid in ({"extents_open": [0., .016, .016]},
                            {"offset_mid": [0., np.nan, .1]},
                            {"fingertip_depth": None}, {"gripper_type": True}):
                bad = deepcopy(geometry)
                bad["sweep_volume_params"].update(invalid)
                with self.subTest(invalid=invalid), patch("graspgenx.utils.checkpoint_io.load_model_cfg", return_value=OmegaConf.create(base_cfg)):
                    with self.assertRaises(ValueError):
                        GraspGenXBackend(checkpoints_dir="unused", gripper_geometry=bad)
        sampler.assert_not_called()
    # END config and geometry checks allocate no model weights or GPU storage.


if __name__ == "__main__":
    unittest.main()
# END independent evidence and exact-prior candidate regressions.
