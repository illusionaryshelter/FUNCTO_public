"""Analytic geometry and small RGB-D sequences exercise observation contracts.

No model inference or external data is used. Tests independently construct the
motions, including camera motion, so a plausible-looking fitted path alone is
not the correctness criterion.
"""

# BEGIN calibrated rigid-part observation tests.
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from manipulation.observations import (
    backproject, load_calibration, load_rgbd, poses_from_models, poses_from_tracks,
    rigid_registration, track_rgbd,
)


def transform(points, matrix):
    return (matrix @ np.column_stack((points, np.ones(len(points)))).T).T[:, :3]


def pose(angles=(0, 0, 0), translation=(0, 0, 0)):
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_euler("xyz", angles).as_matrix()
    matrix[:3, 3] = translation
    return matrix


def write_camera(root, width=96, height=80, scale=1000):
    matrix = np.array([[100., 0., width / 2], [0., 110., height / 2], [0., 0., 1.]])
    (root / "intrinsic.json").write_text(json.dumps({
        "width": width, "height": height,
        "intrinsic_matrix": matrix.ravel(order="F").tolist(),
    }))
    (root / "config.json").write_text(json.dumps({"depth_scale": scale}))
    return matrix


class CalibrationAndDepthTest(unittest.TestCase):
    def test_open3d_column_major_and_divisor_scale(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = write_camera(root, width=4, height=3, scale=2000)
            camera = load_calibration(root / "intrinsic.json")
            np.testing.assert_array_equal(camera.K, expected)
            self.assertEqual(camera.depth_scale, 0.0005)
            (root / "rgb").mkdir()
            (root / "depth").mkdir()
            Image.new("RGB", (4, 3)).save(root / "rgb/00000.jpg")
            raw = np.full((3, 4), 4000, dtype=np.uint16)
            raw[0, 0] = 0
            Image.fromarray(raw).save(root / "depth/00000.png")
            _, depth, _ = load_rgbd(root)
            self.assertEqual(depth[1, 1], 2.0)
            cloud = backproject(depth, expected)
            self.assertEqual(cloud.shape, (11, 3))
            np.testing.assert_allclose(cloud[:, 2], 2.0)

    def test_explicit_metric_units_and_conflicts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_camera(root)
            (root / "config.json").write_text(json.dumps({"depth_unit_m": 1.0}))
            self.assertEqual(load_calibration(root).depth_scale, 1.0)
            (root / "config.json").write_text(json.dumps({"depth_unit_m": 1.0, "depth_scale": 1000}))
            with self.assertRaisesRegex(ValueError, "Conflicting"):
                load_calibration(root)

    def test_invalid_depth_is_filtered_only_for_unordered_clouds(self):
        depth = np.array([[0., -1., np.nan], [np.inf, 2., 3.]])
        matrix = np.array([[2., 0., 1.], [0., 3., 0.5], [0., 0., 1.]])
        result = backproject(depth, matrix)
        np.testing.assert_allclose(result, [[0., 1 / 3, 2.], [1.5, 0.5, 3.]])
        for uv in ([0., 0.], [1., 0.], [2., 0.], [0., 1.]):
            with self.subTest(uv=uv), self.assertRaisesRegex(ValueError, "invalid depth"):
                backproject(depth, matrix, [uv])
        for uv in ([-0.1, 1.], [3., 1.], [1., 2.]):
            with self.subTest(uv=uv), self.assertRaisesRegex(ValueError, "out of bounds"):
                backproject(depth, matrix, [uv])
        with self.assertRaisesRegex(ValueError, "finite N-by-2"):
            backproject(depth, matrix, [[np.nan, 1.]])
        selected = np.zeros_like(depth, dtype=bool)
        selected[1, 1] = True
        with self.assertRaisesRegex(ValueError, "outside the mask"):
            backproject(depth, matrix, [[2., 1.]], mask=selected)

    def test_subpixel_rays_and_calibration_shape(self):
        depth = np.full((4, 5), 2.)
        matrix = np.array([[2., 0., 1.], [0., 4., 1.], [0., 0., 1.]])
        np.testing.assert_allclose(backproject(depth, matrix, [[1.25, 1.5]]), [[0.25, 0.25, 2.]])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_camera(root, width=4, height=3)
            (root / "rgb").mkdir()
            (root / "depth").mkdir()
            Image.new("RGB", (4, 3)).save(root / "rgb/00000.png")
            Image.fromarray(np.ones((2, 4), dtype=np.uint16)).save(root / "depth/00000.png")
            with self.assertRaisesRegex(ValueError, "dimensions must agree"):
                load_rgbd(root)


class CorrespondenceMotionTest(unittest.TestCase):
    def test_planar_correspondences_observe_rotation_with_outliers(self):
        rng = np.random.default_rng(18)
        source = np.column_stack((rng.uniform(-0.3, 0.3, (80, 2)), np.full(80, 1.4)))
        expected = pose((0.4, -0.25, 0.8), (0.12, -0.2, 0.04))
        target = transform(source, expected)
        target += rng.normal(0, 0.0002, target.shape)
        target[:18] = rng.uniform(-2, 2, (18, 3))
        fitted, diagnostics = rigid_registration(source, target, max_error=0.002)
        np.testing.assert_allclose(fitted[:3, :3], expected[:3, :3], atol=6e-4)
        np.testing.assert_allclose(fitted[:3, 3], expected[:3, 3], atol=6e-4)
        self.assertGreaterEqual(diagnostics["num_inliers"], 60)
        self.assertFalse(diagnostics["inlier_mask"][:18].any())

    def test_minimal_triangle_is_valid_but_collinear_and_nonrigid_are_not(self):
        source = np.array([[0., 0., 1.], [0.2, 0., 1.], [0., 0.3, 1.]])
        expected = pose((0.5, 0.3, -0.2), (0.1, -0.2, 0.3))
        fitted, _ = rigid_registration(source, transform(source, expected))
        np.testing.assert_allclose(fitted, expected, atol=1e-12)
        line = np.column_stack((np.arange(8), np.zeros((8, 2))))
        with self.assertRaisesRegex(ValueError, "noncollinear"):
            rigid_registration(line, line)
        rng = np.random.default_rng(29)
        random = rng.uniform(-1, 1, (40, 3))
        with self.assertRaisesRegex(ValueError, "consensus"):
            rigid_registration(random, random * 1.7, max_error=0.001)

    def test_unordered_arrays_do_not_pass_as_known_correspondence(self):
        rng = np.random.default_rng(11)
        source = rng.uniform(-1, 1, (45, 3))
        # Both clouds contain exactly the same geometry. A nearest-neighbour
        # cloud score would be perfect, but these supplied feature IDs are wrong.
        with self.assertRaisesRegex(ValueError, "consensus"):
            rigid_registration(source, source[rng.permutation(len(source))], max_error=0.001)

# BEGIN original-mesh pose and fixed-material-point contracts.
class ModelPoseTest(unittest.TestCase):
    def test_arbitrary_camera_motion_is_removed_without_losing_object_rotation(self):
        rng = np.random.default_rng(17)
        material_points = rng.uniform(-0.15, 0.15, (35, 3)) + [0.4, -0.2, 0.3]
        bodies = np.asarray([pose((.1*f, -.06*f, .13*f), (.2+.01*f, .1, 1.2))
                             for f in range(8)])
        cameras = np.asarray([pose(rng.uniform(-.5, .5, 3), rng.uniform(-.2, .2, 3))
                              for _ in bodies])
        reference = pose((.6, -.4, .2), (-.4, .3, 1.8))
        observations = cameras @ bodies
        references = cameras @ reference
        initial_points = transform(material_points, observations[0])
        result = poses_from_models(observations, initial_points,
                                   reference_camera_from_model=references,
                                   frame_indices=np.arange(8)*2)
        np.testing.assert_allclose(result["poses"][0], pose(translation=initial_points.mean(0)), atol=1e-12)
        for frame, world_from_body in enumerate(bodies):
            # Expected material coordinates are constructed in the physical
            # world, then expressed in the first camera independently of the
            # body-origin convention chosen by poses_from_models.
            expected = transform(material_points, cameras[0] @ world_from_body)
            actual = transform(result["points"], result["poses"][frame])
            np.testing.assert_allclose(actual, expected, atol=1e-12)
        np.testing.assert_array_equal(result["frame_indices"], np.arange(8)*2)
        self.assertIn("compensated", result["camera_assumption"])

    def test_moving_reference_defines_relative_motion_not_a_stationary_world(self):
        material_points = np.array([[.1, .2, .3], [-.1, .3, .2], [.2, -.2, .1]])
        bodies = np.array([pose((0, .1*f, 0), (.2, .1, 1.2)) for f in range(3)])
        refs = np.array([pose((0, 0, .2*f), (-.3, .03*f, 1.4)) for f in range(3)])
        cameras = np.array([pose((.1*f, -.1, .1), (.02*f, .2, -.1)) for f in range(3)])
        initial_points = transform(material_points, cameras[0] @ bodies[0])
        result = poses_from_models(cameras @ bodies, initial_points,
                                   reference_camera_from_model=cameras @ refs)
        for frame in range(3):
            expected = transform(material_points, cameras[0] @ refs[0] @ np.linalg.inv(refs[frame]) @ bodies[frame])
            np.testing.assert_allclose(transform(result["points"], result["poses"][frame]), expected, atol=1e-12)

    def test_nonzero_mesh_center_and_arbitrary_mesh_frame_do_not_change_body_motion(self):
        models = np.array([pose((.2, .3*f, -.1), (.4, .1*f, 1.6)) for f in range(4)])
        refs = np.array([pose((0, -.2, .05*f), (-.4, .2, 1.4)) for f in range(4)])
        points = np.array([[.2, .1, 1.4], [.3, .4, 1.2], [.5, -.2, 1.3]])
        result = poses_from_models(models, points, reference_camera_from_model=refs)
        mesh_change = pose((.6, -.4, .2), (1., -2., 3.))
        reference_change = pose((-.4, .8, -.6), (-.5, .7, -.9))
        reexpressed = poses_from_models(models @ np.linalg.inv(mesh_change), points,
                                      reference_camera_from_model=refs @ np.linalg.inv(reference_change))
        np.testing.assert_allclose(reexpressed["poses"], result["poses"], atol=1e-12)
        np.testing.assert_allclose(reexpressed["model_from_body"], mesh_change @ result["model_from_body"], atol=1e-12)
        np.testing.assert_array_equal(reexpressed["points"], result["points"])

    def test_stationary_camera_and_single_frame_bind_points_once(self):
        initial_model = pose((.4, -.5, .7), (.2, -.1, 1.5))
        material_points = np.array([[.2, .1, 0], [-.1, .2, .1], [.1, -.2, .2]])
        points = transform(material_points, initial_model)
        original_points = points.copy()
        single = poses_from_models(initial_model[None], points)
        np.testing.assert_allclose(transform(single["points"], single["poses"][0]), points, atol=1e-12)
        final_model = pose((-.2, .3, 1.), (.3, .2, 1.6))
        result = poses_from_models([initial_model, final_model], points)
        np.testing.assert_allclose(transform(result["points"], result["poses"][1]),
                                   transform(material_points, final_model), atol=1e-12)
        np.testing.assert_array_equal(points, original_points)
        self.assertEqual(result["camera_assumption"], "stationary camera required")

    def test_invalid_poses_and_point_geometry_are_rejected(self):
        points = np.array([[0., 0., 1.], [.1, 0., 1.], [0., .2, 1.]])
        invalid = [np.eye(4), np.empty((0, 4, 4))]
        for entry in ((0, 0, np.nan), (3, 0, .2), (0, 0, -1.), (1, 1, 2.)):
            matrix = np.eye(4)
            matrix[entry[:2]] = entry[2]
            invalid.append(matrix[None])
        for prediction in invalid:
            with self.subTest(shape=prediction.shape), self.assertRaises(ValueError):
                poses_from_models(prediction, points)
        with self.assertRaisesRegex(ValueError, "same frame sequence"):
            poses_from_models(np.array([np.eye(4)]*2), points, reference_camera_from_model=np.eye(4)[None])
        for geometry in ([], [[0, 1]], [[0, np.nan, 1]]):
            with self.subTest(geometry=geometry), self.assertRaisesRegex(ValueError, "initial_points"):
                poses_from_models(np.eye(4)[None], geometry)

    def test_frame_indices_preserve_unsigned_order_and_reject_invalid_values(self):
        models = np.array([np.eye(4), pose(translation=(.1, 0, 0))])
        points = np.array([[0., 0., 1.], [.1, 0., 1.], [0., .2, 1.]])
        for indices in ([-1, 1], [0, 0], [0., np.nan], ["0", "1"], np.array([2, 1], np.uint64)):
            with self.subTest(indices=indices), self.assertRaisesRegex(ValueError, "integer frame IDs"):
                poses_from_models(models, points, frame_indices=indices)
        # Comparing adjacent IDs, rather than subtracting them, avoids unsigned
        # wraparound and preserves legitimately large ordered identifiers.
        large = np.array([2**64-2, 2**64-1], dtype=np.uint64)
        result = poses_from_models(models, points, frame_indices=large)
        np.testing.assert_array_equal(result["frame_indices"], large)
# END original-mesh pose and fixed-material-point contracts.


# BEGIN fixed-ID TAPIP3D boundary and independent frame-gauge tests.
class TrackPoseTest(unittest.TestCase):
    def observations(self, *, count=6, with_reference=True, initial_bias=True):
        rng = np.random.default_rng(83)
        roles = ["carrier", "reference"] if with_reference else ["carrier"]
        geometries = {role:rng.uniform(-.1,.1,(64,3))+[.2*i, .1, 1.5]
                      for i,role in enumerate(roles)}
        E0 = pose((.3,-.2,.4), (.3,.1,-.15))
        cameras = np.array([pose((.02*f,-.01*f,.03*f), (.02*f,0,0)) @ E0 for f in range(count)])
        motions = {"carrier":np.array([pose((.04*f,-.02*f,.03*f), (.01*f,-.003*f,.002*f)) for f in range(count)]),
                   "reference":np.array([pose((-.01*f,.01*f,-.02*f), (-.003*f,.007*f,0)) for f in range(count)])}
        query_camera = np.concatenate(list(geometries.values()))
        query_world = transform(query_camera, np.linalg.inv(E0))
        owners = np.repeat(np.arange(len(roles)),64)
        coords = []
        for f in range(count):
            groups = []
            for i, role in enumerate(roles):
                bias = pose((.001,-.002,.003), (.0003,.0002,-.0001)) if initial_bias else np.eye(4)
                groups.append(transform(geometries[role], np.linalg.inv(E0) @ motions[role][f] @ bias))
            coords.append(np.concatenate(groups))
        tracked = {"coords":np.asarray(coords),"query_points":np.column_stack((np.zeros(len(query_world)),query_world)),
                   "query_owner":owners,"visibs":np.ones((count,len(owners)),bool),
                   "projection_valid":np.ones((count,len(owners)),bool),"camera_from_world":cameras}
        return tracked,geometries,motions

    def test_moving_camera_reference_and_query_frame_bias_keep_one_common_gauge(self):
        tracked,geometry,motions = self.observations()
        raw = tracked["coords"].copy()
        result = poses_from_tracks(tracked,geometry,frame_indices=np.arange(6)*3)
        for role in geometry:
            P0 = pose(translation=geometry[role].mean(0))
            np.testing.assert_allclose(result["body_poses"][role], motions[role] @ P0, atol=2e-12)
            np.testing.assert_array_equal(result["body_poses"][role][0],P0)
        P0 = pose(translation=geometry["carrier"].mean(0))
        expected = np.linalg.inv(motions["reference"]) @ motions["carrier"] @ P0
        np.testing.assert_allclose(result["poses"],expected,atol=2e-12)
        E = tracked["camera_from_world"]
        np.testing.assert_allclose(result["camera_from_initial_camera"],E @ np.linalg.inv(E[0]),atol=1e-12)
        np.testing.assert_array_equal(result["tracking_coords"],raw)
        np.testing.assert_array_equal(tracked["coords"],raw)
        np.testing.assert_array_equal(result["frame_indices"],np.arange(6)*3)
        # The first predicted coordinates are deliberately biased, while the
        # normalized fixed-body gauge still binds original semantic geometry.
        self.assertGreater(np.linalg.norm(raw[0]-tracked["query_points"][:,1:]),0.001)

    def test_reference_is_not_required_with_known_camera_motion(self):
        tracked,geometry,motions = self.observations(with_reference=False)
        result = poses_from_tracks(tracked,geometry)
        initial = pose(translation=geometry["carrier"].mean(0))
        np.testing.assert_allclose(result["poses"],motions["carrier"] @ initial,atol=2e-12)
        np.testing.assert_array_equal(result["points"],geometry["carrier"]-geometry["carrier"].mean(0))

    def test_visibility_is_a_filter_and_failed_fits_do_not_become_free_motion(self):
        tracked,geometry,_ = self.observations()
        tracked["visibs"][3,:64] = False
        tracked["visibs"][3,:5] = True
        with self.assertRaisesRegex(ValueError,"carrier frame 3.*insufficient"):
            poses_from_tracks(tracked,geometry)
        tracked,geometry,_ = self.observations(initial_bias=False)
        tracked["coords"][4,:64] *= 1.8
        with self.assertRaisesRegex(ValueError,"carrier frame 4 rigid fit failed"):
            poses_from_tracks(tracked,geometry)

    def test_visibility_and_correspondence_identity_shapes_are_enforced(self):
        tracked,geometry,_ = self.observations()
        changes = [{"query_owner":np.zeros(128,dtype=int)},
                   {"visibs":np.ones((6,128),float)},
                   {"projection_valid":np.ones((6,64),bool)},
                   {"query_points":np.ones((128,4))},
                   {"coords":np.full((6,128,3),np.nan)}]
        for change in changes:
            with self.subTest(change=list(change)),self.assertRaises(ValueError):
                poses_from_tracks(dict(tracked,**change),geometry)
        with self.assertRaisesRegex(ValueError,"integer frame IDs"):
            poses_from_tracks(tracked,geometry,frame_indices=[0,1,2,2,4,5])


class SmallRGBDSequenceTest(unittest.TestCase):
    def write_sequence(self,root,*,invalidate_second=False,stationary=True):
        matrix = write_camera(root,width=128,height=96)
        config = json.loads((root/"config.json").read_text())
        if stationary is not None:
            config["stationary_camera"] = stationary
        (root/"config.json").write_text(json.dumps(config))
        (root/"rgb").mkdir(); (root/"depth").mkdir()
        mask = np.zeros((96,128),bool); mask[14:82,16:94] = True
        for frame in range(3):
            Image.new("RGB",(128,96),(80+frame,90,100)).save(root/"rgb"/f"{frame:05d}.png")
            depth = np.full((96,128),1500,dtype=np.uint16)
            if frame==1 and invalidate_second:
                depth[:] = 0
            Image.fromarray(depth).save(root/"depth"/f"{frame:05d}.png")
        return mask,matrix

    @staticmethod
    def known_tracks(paths,depth_paths,*,K,depth_scale,masks,extrinsics,stationary_camera=False):
        from local_models.tapip3d import sample_queries
        with Image.open(depth_paths[0]) as image:
            depth = np.asarray(image,dtype=float)*depth_scale
        E = np.asarray(extrinsics)
        q,owners,_ = sample_queries(depth,K,masks,E[0])
        initial_camera = transform(q[:,1:],E[0])
        coords = []
        for frame in range(len(paths)):
            current = initial_camera.copy()
            for owner in range(len(masks)):
                delta = pose((0,0,.05*frame),(.015*frame,.003*frame,0)) if owner==0 else pose((0,0,-.02*frame),(-.005*frame,0,0))
                current[owners==owner] = transform(current[owners==owner],delta)
            coords.append(transform(current,np.linalg.inv(E[0])))
        coords = np.asarray(coords)
        camera = np.einsum("tij,tnj->tni",E[:,:3,:3],coords)+E[:,None,:3,3]
        uvz = camera @ np.asarray(K).T
        return {"coords":coords,"visibs":np.ones(coords.shape[:2],bool),
                "query_points":q,"query_owner":owners,"projection_valid":np.ones(coords.shape[:2],bool),
                "pixels":uvz[...,:2]/uvz[...,2:],"camera_from_world":E}

    def test_default_path_needs_no_mesh_and_preserves_query_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); mask,K = self.write_sequence(root)
            with patch("local_models.tapip3d.track",side_effect=self.known_tracks) as worker:
                result = track_rgbd(root,mask)
            worker.assert_called_once()
            self.assertNotIn("models",worker.call_args.kwargs)
            self.assertLessEqual(len(result["points"]),3000)
            self.assertEqual(result["tracking_coords"].shape,(3,64,3))
            relative = result["poses"] @ np.linalg.inv(result["poses"][0])
            expected = [pose((0,0,.05*f),(.015*f,.003*f,0)) for f in range(3)]
            np.testing.assert_allclose(relative,expected,atol=2e-12)
            self.assertEqual(result["camera_assumption"],"explicitly stationary camera")

    def test_reference_tracks_compensate_actual_reference_motion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); mask,_ = self.write_sequence(root)
            reference = np.zeros_like(mask); reference[10:85,100:124] = True
            with patch("local_models.tapip3d.track",side_effect=self.known_tracks) as worker:
                result = track_rgbd(root,mask,reference_mask=reference)
            np.testing.assert_array_equal(worker.call_args.kwargs["masks"],np.stack([mask,reference]))
            for f in range(3):
                ref_motion = pose((0,0,-.02*f),(-.005*f,0,0))
                body_motion = pose((0,0,.05*f),(.015*f,.003*f,0))
                expected = np.linalg.inv(ref_motion) @ body_motion @ result["poses"][0]
                np.testing.assert_allclose(result["poses"][f],expected,atol=2e-12)

    def test_camera_declaration_missing_files_and_worker_failure_are_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); mask,_ = self.write_sequence(root,stationary=None)
            with patch("local_models.tapip3d.track") as worker:
                with self.assertRaisesRegex(ValueError,"declare stationary_camera"):
                    track_rgbd(root,mask)
                worker.assert_not_called()
            with patch("local_models.tapip3d.track",side_effect=ValueError("tracking failed")):
                with self.assertRaisesRegex(ValueError,"tracking failed"):
                    track_rgbd(root,mask,stationary_camera=True)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); mask,_ = self.write_sequence(root,invalidate_second=True)
            with patch("local_models.tapip3d.track") as worker:
                with self.assertRaisesRegex(ValueError,"frame 1 has insufficient valid depth"):
                    track_rgbd(root,mask)
                worker.assert_not_called()

    def test_frame_subset_and_bad_output_length_are_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); mask,_ = self.write_sequence(root)
            with patch("local_models.tapip3d.track",side_effect=self.known_tracks) as worker:
                result = track_rgbd(root,mask,max_frames=2)
            self.assertEqual([len(x) for x in worker.call_args.args],[2,2])
            np.testing.assert_array_equal(result["frame_indices"],[0,1])
            for value in (1,True,2.5):
                with self.subTest(value=value),self.assertRaisesRegex(ValueError,"max_frames"):
                    track_rgbd(root,mask,max_frames=value)
            with patch("local_models.tapip3d.track",return_value={"coords":np.zeros((2,64,3))}):
                with self.assertRaisesRegex(ValueError,"every requested frame"):
                    track_rgbd(root,mask)
# END fixed-ID TAPIP3D boundary and independent frame-gauge tests.


if __name__ == "__main__":
    unittest.main()
# END calibrated rigid-part observation tests.
