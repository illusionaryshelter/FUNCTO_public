"""CPU tests of the recording boundary; no native tracking model is executed.

The fake pipeline returns deliberately identical matrices with different
registration evidence. These tests check the adapter's identity/time/frame
contract and failure records, not Point2Pose's tracking accuracy.
"""

# BEGIN fixed body coordinates and original observation evidence checks.
import json
import sys
from types import ModuleType, SimpleNamespace

import cv2
import numpy as np
from omegaconf import OmegaConf
import pytest
from scipy.spatial.transform import Rotation

from config.rgbd_bridge import RGBDPacket, MuJoCoRGBD, make_frame
from config.run_offline import run_offline, sequence_inputs, tracking_evidence


def transform(angle=0., xyz=(0., 0., 0.)):
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_euler("z", angle).as_matrix()
    pose[:3, 3] = xyz
    return pose


@pytest.fixture
def native_boundary(monkeypatch):
    # Match the native Frame's keyword-only data contract without importing
    # model packages, CUDA extensions or changing the installed environment.
    frame_module = ModuleType("point2pose.data_types.frame")
    frame_module.Frame = lambda **kwargs: SimpleNamespace(**kwargs)
    monkeypatch.setitem(sys.modules, frame_module.__name__, frame_module)
    pipeline_module = ModuleType("point2pose.pipeline.modular_pipeline")
    monkeypatch.setitem(sys.modules, pipeline_module.__name__, pipeline_module)
    return pipeline_module


def sequence(tmp_path, *, timed=True, world=True):
    directory = tmp_path / "sequence"
    for folder in ("rgb", "depth", "masks/obj_0"):
        (directory / folder).mkdir(parents=True)
    K = np.array([[31., 0., 2.5], [0., 29., 1.5], [0., 0., 1.]])
    camera = transform(.7, (.3, -.2, .9))
    meta = dict(K=K.tolist(), object_ids=["rigid_part"], frame_indices=[10, 17, 24])
    if timed:
        meta["timestamps_s"] = [1.2, 1.21, 1.43]
    if world:
        meta["T_world_camera_cv"] = camera.tolist()
    for index in range(3):
        name = f"{index:06}"
        cv2.imwrite(str(directory / "rgb" / (name + ".png")), np.full((4, 6, 3), [10, 20, 30], np.uint8))
        np.save(directory / "depth" / (name + ".npy"), np.full((4, 6), .8, np.float32))
        cv2.imwrite(str(directory / "masks/obj_0" / (name + ".png")), np.full((4, 6), 255, np.uint8))
    (directory / "camera.json").write_text(json.dumps(meta))
    config = tmp_path / "config.yaml"
    OmegaConf.save(OmegaConf.create(dict(pipeline=dict(params=dict(use_segmenter=True, max_num_obj=9,
        pose_save_path="unused", frame_reg_mode="f2m")), register=dict(params=dict(residual_thres=.01)),
        tracker=dict(params=dict(num_points=18)))), config)
    return directory, config, meta, K, camera


def pipeline_stub(native_boundary, *, fault=None):
    instances = []
    initial = transform(-.4, (.2, .5, -.1))
    motion = transform(.2, (.1, -.3, .2))

    class Pipeline:
        def __init__(self, config):
            self.config = config
            self.objects = [SimpleNamespace(id=0, pose=motion.copy(), init_pose=initial.copy(),
                                             lost=False, lost_streak=0)]
            self.frames, self.calls = [], []
            self.frontend = SimpleNamespace(frame_reg_mode="f2m", reg_residual_thres=.01)

            def step(frame):
                self.calls.append(frame.id)
                result = SimpleNamespace(frame_id=frame.id, obj_poses={0: motion.copy()},
                    reg_stats={0: dict(residuals=np.array([.002, .003]) if frame.id == 1 else np.array([]),
                                      inliers=np.array([True, True]), pose_jump_guard_info={"rejected": False})},
                    mean_residuals={0: .0025 if frame.id == 1 else -1.}, valid_indices={0: np.array([2, 7])})
                self.last_result = result
                return result

            self.original_step = self.frontend.step = step
            instances.append(self)

        def step(self, frame):
            self.frames.append(frame)
            if frame.id:
                # The wrapper must return exactly the object native consumers
                # received, not a replacement result with rewritten evidence.
                assert self.frontend.step(frame) is self.last_result
                if fault == "body":
                    self.objects[0].init_pose[0, 3] += .1
                elif fault == "identity":
                    self.objects[0].id = 1
                elif fault == "frame":
                    self.last_result.frame_id -= 1
            return np.stack([obj.pose for obj in self.objects])

    native_boundary.ModularPipeline = Pipeline
    return instances, initial, motion


def test_recording_preserves_original_clock_fixed_body_and_actual_evidence(tmp_path, native_boundary):
    directory, config, meta, K, camera = sequence(tmp_path)
    # A per-frame calibration/pose overrides explicitly supplied static values.
    (directory / "intrinsics").mkdir()
    (directory / "camera_poses").mkdir()
    K1 = K.copy()
    K1[0, 0] = 35.
    camera1 = transform(-.5, (.8, .4, -.2))
    np.save(directory / "intrinsics/000001.npy", K1)
    np.save(directory / "camera_poses/000001.npy", camera1)
    instances, initial, motion = pipeline_stub(native_boundary)
    output = tmp_path / "recording"
    report = run_offline(config, directory, output, device="cpu")
    pipeline = instances[0]
    assert pipeline.frontend.step is pipeline.original_step
    assert pipeline.calls == [1, 2]
    assert [frame.id for frame in pipeline.frames] == [0, 1, 2]
    assert [frame.timestamp for frame in pipeline.frames] == meta["timestamps_s"]
    np.testing.assert_array_equal(pipeline.frames[0].rgb[0, 0], [30, 20, 10])
    assert not pipeline.config.pipeline.params.use_segmenter
    assert pipeline.config.pipeline.params.max_num_obj == 1
    assert pipeline.config.register.params.residual_thres == .01
    assert pipeline.config.tracker.params.num_points == 18
    data = np.load(output / "estimated_poses_by_frame.npz", allow_pickle=False)
    np.testing.assert_array_equal(data["frame_indices"], [10, 17, 24])
    np.testing.assert_array_equal(data["tracker_frame_ids"], [0, 1, 2])
    np.testing.assert_array_equal(data["timestamp_s"], meta["timestamps_s"])
    np.testing.assert_allclose(data["T_initial_camera_B"][0], initial)
    np.testing.assert_allclose(data["T_camera_B"], np.repeat((motion @ initial)[None, None], 3, axis=0))
    np.testing.assert_allclose(data["T_world_B"][:, 0], np.stack([cam @ motion @ initial for cam in (camera, camera1, camera)]))
    np.testing.assert_array_equal(data["K"][1], K1)
    # Pose equality cannot decide whether the object was observed again.
    assert [row["objects"][0]["status"] for row in report["frames"]] == ["initialized", "frontend_condition_met", "stale"]
    assert report["frames"][1]["objects"][0]["valid_indices"] == [2, 7]
    saved = json.loads((output / "tracking_evidence.json").read_text())
    assert saved["status"] == "completed" and saved["processed_frames"] == 3


def test_missing_acquisition_time_and_world_pose_stay_missing(tmp_path, native_boundary):
    directory, config, _, _, _ = sequence(tmp_path, timed=False, world=False)
    instances, _, _ = pipeline_stub(native_boundary)
    output = tmp_path / "recording"
    report = run_offline(config, directory, output, device="cpu")
    assert all(frame.timestamp is None for frame in instances[0].frames)
    assert set(report["missing"]) == {"timestamps_s", "T_world_camera_cv"}
    data = np.load(output / "estimated_poses_by_frame.npz", allow_pickle=False)
    assert not data["timestamp_available"].any() and not data["world_camera_available"].any()
    assert np.isnan(data["timestamp_s"]).all() and np.isnan(data["T_world_B"]).all()
    assert np.isfinite(data["T_camera_B"]).all()


@pytest.mark.parametrize("fault, message", [("body", "fixed initial body"), ("identity", "instance order"),
                                            ("frame", "current tracker frame")])
def test_changed_body_identity_or_mismatched_evidence_saves_partial_failure(tmp_path, native_boundary, fault, message):
    directory, config, _, _, _ = sequence(tmp_path)
    instances, _, _ = pipeline_stub(native_boundary, fault=fault)
    output = tmp_path / "recording"
    with pytest.raises(ValueError, match=message):
        run_offline(config, directory, output, device="cpu")
    report = json.loads((output / "tracking_evidence.json").read_text())
    assert report["status"] == "failed" and report["failed_frame"] == 17
    assert report["processed_frames"] == 1
    np.testing.assert_array_equal(np.load(output / "estimated_poses_by_frame.npz")["frame_indices"], [10])
    assert instances[0].frontend.step is instances[0].original_step


def test_native_residual_modes_and_jump_rejection_remain_distinct():
    obj = SimpleNamespace(id=0, lost=False, lost_streak=0)
    result = SimpleNamespace(obj_poses={0: np.eye(4)}, reg_stats={0: {"residuals": np.array([.02])}},
                             mean_residuals={0: .02}, valid_indices={0: np.array([3])})

    def label(mode="f2m"):
        return tracking_evidence(result, [obj], first=False, frame_reg_mode=mode, residual_threshold_m=.01)[0]

    assert label()["status"] == "warning"
    assert label("f2f")["status"] == "stale"
    # Native cluster-register failure returns T0 plus a nonempty -1 sentinel
    # array; that array is not evidence of newly observed metric residuals.
    result.reg_stats[0]["residuals"] = np.full(3, -1.)
    sentinel = label()
    assert sentinel["status"] == "stale" and not sentinel["registration_residuals_observed"]
    assert sentinel["registration_stats"]["residuals"] == [-1., -1., -1.]
    result.reg_stats[0]["residuals"] = np.array([.002])
    result.mean_residuals[0] = .002
    result.reg_stats[0]["pose_jump_guard_info"] = {"rejected": True}
    assert label()["status"] == "stale"
    result.obj_poses.clear()
    assert label()["status"] == "missing"


def test_input_rejects_fabricated_identity_and_time_before_pipeline(tmp_path):
    directory, _, meta, _, _ = sequence(tmp_path)
    for key, bad in (("frame_indices", [False, 17, 24]), ("frame_indices", [10, 10, 24]),
                     ("timestamps_s", [1., 1., 2.]), ("object_ids", ["same", "same"])):
        invalid = dict(meta, **{key: bad})
        (directory / "camera.json").write_text(json.dumps(invalid))
        with pytest.raises(ValueError):
            sequence_inputs(directory)


def test_bridge_keeps_binary_masks_original_calibration_and_missing_time(native_boundary):
    K = np.array([[20., 0., 1.5], [0., 20., .5], [0., 0., 1.]])
    rgb = np.arange(24, dtype=np.uint8).reshape(2, 4, 3)
    depth = np.array([[1., 2., np.nan, -1.], [1., 2., 3., 4.]])
    packet = RGBDPacket(rgb, depth, K, None)
    mask = np.full((2, 4), 255, np.uint8)
    frame = make_frame(packet, mask, 4, 1, device="cpu")
    assert frame.id == 4 and frame.timestamp is None and frame.depth_factor == 1.
    np.testing.assert_array_equal(frame.intrinsics, K)
    np.testing.assert_array_equal(frame.mask.numpy()[0, 0], mask > 0)
    np.testing.assert_array_equal(frame.depth[0], [1., 2., 0., 0.])
    with pytest.raises(ValueError, match="binary"):
        make_frame(packet, np.full((2, 4), .7), 0, 1, device="cpu")
    K[0, 1] = .1
    with pytest.raises(ValueError, match="zero-skew"):
        make_frame(packet, mask, 0, 1, device="cpu")


def test_mujoco_bridge_uses_supplied_pixel_centers_and_one_capture_state(monkeypatch):
    class Renderer:
        def __init__(self, model, height, width):
            self.depth = False

        def update_scene(self, data, camera):
            self.data = data

        def enable_depth_rendering(self):
            self.depth = True

        def disable_depth_rendering(self):
            self.depth = False

        def render(self):
            return np.full((2, 4), .8) if self.depth else np.zeros((2, 4, 3), np.uint8)

        def close(self):
            pass

    mj = SimpleNamespace(__version__="3.5.0", mj_name2id=lambda *args: 0,
        mjtObj=SimpleNamespace(mjOBJ_CAMERA=0), mjtProjection=SimpleNamespace(mjPROJ_PERSPECTIVE=0), Renderer=Renderer)
    monkeypatch.setitem(sys.modules, "mujoco", mj)
    K = np.array([[20., 0., 1.5], [0., 20., .5], [0., 0., 1.]])
    data = SimpleNamespace(time=7.1, cam_xmat=np.eye(3)[None], cam_xpos=np.array([[.3, .4, .5]]))
    camera = MuJoCoRGBD(SimpleNamespace(cam_projection=[0]), data, "camera", width=4, height=2, K=K)
    packet = camera.capture()
    np.testing.assert_array_equal(packet.K, K)
    np.testing.assert_array_equal(packet.T_world_cam_cv[:3, :3], np.diag([1., -1., -1.]))
    np.testing.assert_array_equal(packet.T_world_cam_cv[:3, 3], [.3, .4, .5])
    assert packet.timestamp_s == 7.1 and data.time == 7.1
    with pytest.raises(TypeError, match="K"):
        MuJoCoRGBD(SimpleNamespace(cam_projection=[0]), data, "camera")
    camera.close()
# END these tests do not substitute synthetic results for a real tracking run.
