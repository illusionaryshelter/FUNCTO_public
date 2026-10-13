"""Render saved FUNCTO predictions without model inference or API requests.

All geometry follows the existing trajectory renderer: millimeter RGB-D,
``pc_viz`` filtering, target coordinates, function-point translation, and xyz
Euler angles in radians. Only display point counts are reduced. Numerical
checks use the original points and compare against the saved trajectory PLY.
The replay is a single-view rigid point cloud, not a simulation or robot path.
"""

import argparse
from dataclasses import dataclass
from html import escape
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation


POINT_NAMES = ("function", "center", "grasp")
POINT_COLORS = ("#ffc400", "#a24edd", "#00d9ff")


def read_json(path):
    return json.loads(Path(path).read_text())


def move_points(points, initial_pose, pose):
    """Apply an absolute tool pose around the *function point*, not its center.

    Each pose is [translation, xyz Euler angles]. With row-vector points the
    relative rotation is transposed. This equals the pipeline's sequential
    Open3D rotations about the previous function point, without drift.
    """
    r0 = Rotation.from_euler("xyz", initial_pose[1]).as_matrix()
    ri = Rotation.from_euler("xyz", pose[1]).as_matrix()
    return (np.asarray(points) - initial_pose[0]) @ (ri @ r0.T).T + pose[0]


def target_to_camera(points, frame):
    """Invert rotate-about-center followed by translate(-centroid)."""
    center = np.asarray(frame["rotation_center"])
    centroid = np.asarray(frame["centroid"])
    rotation = np.asarray(frame["rotation_matrix"])
    return (np.asarray(points) + centroid - center) @ rotation + center


def project_points(points, intrinsic):
    """Project camera-coordinate points; points behind the camera are invalid."""
    points = np.asarray(points)
    if np.any(points[:, 2] <= 0):
        raise ValueError("Cannot project points behind the camera")
    homogeneous = points @ np.asarray(intrinsic).T
    return homogeneous[:, :2] / homogeneous[:, 2:3]


def sample_cloud(points, colors, limit):
    """Deterministic display-only sampling, independent of model RNG state."""
    indices = np.random.default_rng(0).choice(len(points), min(len(points), limit), replace=False)
    return np.asarray(points)[indices], np.asarray(colors)[indices]


@dataclass
class SavedRun:
    root: Path
    demo: Path
    manifest: dict
    rgb: np.ndarray
    depth: np.ndarray
    masks: dict
    points_2d: dict
    intrinsic: np.ndarray
    frame: dict
    track: dict
    trajectory: np.ndarray
    contact: int
    scene: np.ndarray
    scene_colors: np.ndarray
    scene_tool_mask: np.ndarray
    scene_target_mask: np.ndarray
    tool: np.ndarray
    tool_colors: np.ndarray

    @property
    def initial_markers(self):
        return np.asarray([self.track[f"{name}_init"] for name in POINT_NAMES])

    def markers(self, index):
        return move_points(self.initial_markers, self.trajectory[0], self.trajectory[index])


def load_run(root, demo=None):
    """Load a completed run and reuse its original point-cloud reconstruction.

    The current pipeline uses CAM_MATRIX and a fixed 0.001 m depth unit. Check
    the saved intrinsics against that matrix instead of silently rendering a
    different calibration. RGB is read as RGB, without the old BGR plot swap.
    """
    from main import validate_outputs
    from tool_traj_transfer import pc_viz
    from utils_IL.params import CAM_MATRIX

    root = Path(root).resolve()
    manifest = read_json(root / "run.json")
    if manifest["status"] != "passed":
        raise ValueError("Visualization requires a completed run")
    validated = validate_outputs(root)
    demo = Path(demo if demo is not None else manifest["demo"]).resolve()
    trajectory = np.asarray(read_json(root / "tool_traj_transfer_output/test_tool_trajectory.json"))
    contact = int(read_json(demo / "key_point_track_output/keyframe_idx.json")["func"])
    if not 0 <= contact < len(trajectory):
        raise ValueError("Contact keyframe lies outside the trajectory")
    intrinsic_meta = read_json(root / "intrinsic.json")
    # Open3D serializes its 3x3 intrinsic matrix column by column.
    intrinsic = np.asarray(intrinsic_meta["intrinsic_matrix"]).reshape(3, 3, order="F")
    if not np.allclose(intrinsic, CAM_MATRIX, atol=1e-6, rtol=0):
        raise ValueError("Saved calibration differs from the pipeline's CAM_MATRIX")
    rgb = np.asarray(Image.open(root / "rgb/00000.jpg").convert("RGB"))
    depth = np.asarray(Image.open(root / "depth/00000.png"))
    if depth.shape != rgb.shape[:2] or depth.shape != (intrinsic_meta["height"], intrinsic_meta["width"]):
        raise ValueError("RGB, depth and calibration image dimensions disagree")
    masks = {role: np.load(root / "detection_output" / f"{role}_mask.npy", allow_pickle=False)
             for role in ("tool", "target")}
    # pc_viz uses exactly these depth rules and row-major point ordering. It
    # also applies the original tool outlier filter (20 neighbors, std=2).
    valid_depth = (depth > 0) & (depth * 0.001 <= 1.0)
    scene_cloud, tool_cloud = pc_viz(str(root))
    scene = np.asarray(scene_cloud.points).copy()
    if len(scene) != np.count_nonzero(valid_depth):
        raise ValueError("Point-cloud reconstruction no longer matches RGB-D ordering")
    return SavedRun(
        root, demo, manifest, rgb, depth, masks, validated["points_2d"], intrinsic,
        read_json(root / "cam_to_target_trans_output/cam_to_target_trans.json"),
        read_json(root / "tool_pose_transfer_output/test_tool_track.json"),
        trajectory, contact, scene, np.asarray(scene_cloud.colors).copy(),
        masks["tool"][valid_depth], masks["target"][valid_depth],
        np.asarray(tool_cloud.points).copy(), np.asarray(tool_cloud.colors).copy(),
    )


def diagnose(data):
    """Check actual geometry; these are consistency checks, not task scores."""
    import open3d as o3d
    from tool_traj_transfer import compute_function_plane_frame

    trajectory = data.trajectory
    expected = np.asarray(compute_function_plane_frame(*[
        [data.track[f"{name}_init"], data.track[f"{name}_func"]]
        for name in POINT_NAMES
    ]))
    chosen = trajectory[[0, data.contact]]
    position_residual = np.linalg.norm(chosen[:, 0] - expected[:, 0], axis=1)
    rotation_residual = (Rotation.from_euler("xyz", chosen[:, 1]) *
                         Rotation.from_euler("xyz", expected[:, 1]).inv()).magnitude()
    pixel_checks = {}
    projected = project_points(target_to_camera(data.initial_markers, data.frame), data.intrinsic)
    for name, marker_index in (("function", 0), ("grasp", 2)):
        point = np.asarray(data.points_2d[name])
        u, v = point.astype(int)
        pixel_checks[name] = {
            "matched_pixel_xy": point.tolist(),
            "in_tool_mask": bool(data.masks["tool"][v, u]),
            "depth_m": float(data.depth[v, u] * 0.001),
            "selected_3d_reprojection_xy": projected[marker_index].tolist(),
            "reprojection_difference_px": float(np.linalg.norm(point - projected[marker_index])),
        }
    # Independent evidence of rendering equivalence: the old PLY stores the
    # initial tool, full static scene, then tools at frames 10, 20, ... . Compare
    # each block with the absolute reconstruction, using all original points.
    old_cloud = o3d.io.read_point_cloud(str(data.root / "tool_traj_transfer_output/test_tool_traj_pc.ply"))
    old_points = np.asarray(old_cloud.points)
    snapshot_indices = [0] + list(range(10, len(trajectory), 10))
    expected_count = len(data.scene) + len(snapshot_indices) * len(data.tool)
    if len(old_points) != expected_count:
        raise ValueError("Saved PLY does not have the original trajectory snapshot layout")
    residuals = []
    offset = 0
    for index in snapshot_indices:
        points = move_points(data.tool, trajectory[0], trajectory[index])
        residuals.append(float(np.max(np.linalg.norm(old_points[offset:offset + len(points)] - points, axis=1))))
        offset += len(points)
        if index == 0:
            residuals.append(float(np.max(np.linalg.norm(old_points[offset:offset + len(data.scene)] - data.scene, axis=1))))
            offset += len(data.scene)
    rigid_lengths = np.linalg.norm(data.initial_markers[1:] - data.initial_markers[0], axis=1)
    contact_markers = data.markers(data.contact)
    contact_expected = np.asarray([data.track[f"{name}_func"] for name in POINT_NAMES])
    wrapped_delta = (np.diff(trajectory[:, 1], axis=0) + np.pi) % (2 * np.pi) - np.pi
    rotations = Rotation.from_euler("xyz", trajectory[:, 1])
    result = {
        "run_directory": str(data.root), "frame_count": len(trajectory), "contact_frame": data.contact,
        "pixel_checks": pixel_checks,
        "constraint_position_residual_m": position_residual.tolist(),
        "constraint_rotation_residual_rad": rotation_residual.tolist(),
        "contact_marker_residual_m": np.linalg.norm(contact_markers - contact_expected, axis=1).tolist(),
        "initial_marker_distances_m": rigid_lengths.tolist(),
        "saved_ply_max_residual_m": max(residuals),
        "saved_ply_checked_frames": snapshot_indices,
        "max_function_point_step_m": float(np.linalg.norm(np.diff(trajectory[:, 0], axis=0), axis=1).max()),
        "max_wrapped_euler_step_deg": float(np.rad2deg(np.linalg.norm(wrapped_delta, axis=1)).max()),
        "max_rotation_step_deg": float(np.rad2deg((rotations[1:] * rotations[:-1].inv()).magnitude()).max()),
        "contact_rotation_from_start_deg": float(np.rad2deg((rotations[data.contact] * rotations[0].inv()).magnitude())),
        "contact_function_xyz_m": trajectory[data.contact, 0].tolist(),
        "raw_tool_point_count": len(data.tool), "raw_scene_point_count": len(data.scene),
        "interpretation": "Saved-run consistency only; no task, collision, dynamics or robot-execution certification.",
    }
    if max(residuals) > 1e-6 or position_residual.max() > 1e-5 or rotation_residual.max() > 1e-5:
        raise ValueError("Saved trajectory and reconstructed geometry disagree")
    return result


def crop_limits(mask, margin=30):
    """Return an image viewport without rescaling coordinates or predictions."""
    y, x = np.where(mask)
    return (max(0, x.min() - margin), min(mask.shape[1], x.max() + margin),
            min(mask.shape[0], y.max() + margin), max(0, y.min() - margin))


def image_axis(axis, rgb, title, limits=None):
    axis.imshow(rgb)
    axis.set_title(title, fontsize=12)
    if limits is not None:
        axis.set_xlim(*limits[:2])
        axis.set_ylim(*limits[2:])
    axis.set_axis_off()


def draw_image_points(axis, points):
    for name, color in (("function", POINT_COLORS[0]), ("grasp", POINT_COLORS[2])):
        point = points[name]
        axis.scatter(*point, s=100, color=color, edgecolor="black", linewidth=1.2, zorder=5)
        axis.annotate(name, point, xytext=(9, 9), textcoords="offset points", color="white",
                      fontsize=10, bbox=dict(facecolor="black", alpha=0.7, pad=2), zorder=6)


def render_perception(data, output):
    """Keep source RGB colors, mask boundaries and 2D/3D differences visible."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    # BEGIN label the actual run configuration, including disabled refinement.
    # BEGIN describe the backend actually declared by this run.
    point_method = data.manifest["feature_backend"] if data.manifest["feature_matching"] else "Gemini"
    # END point-method label.
    # END actual point-method label.
    overlay = data.rgb.astype(float).copy()
    mask_colors = {"tool": np.array([0, 210, 230]), "target": np.array([255, 138, 45])}
    for name, color in mask_colors.items():
        overlay[data.masks[name]] = 0.7 * overlay[data.masks[name]] + 0.3 * color
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    image_axis(axes[0], overlay.astype(np.uint8), "SAM3.1: saved tool / target masks", crop_limits(data.masks["tool"] | data.masks["target"]))
    for name, color in mask_colors.items():
        axes[0].contour(data.masks[name], levels=[0.5], colors=[color / 255.0], linewidths=0.9)
    axes[0].legend(handles=[Patch(color=color / 255.0, label=name) for name, color in mask_colors.items()], loc="upper left")
    # BEGIN show final points and depth-selected points as different outputs.
    image_axis(axes[1], data.rgb, f"{point_method} points (circles) / adopted 3D points (crosses)", crop_limits(data.masks["tool"], 18))
    # END point output labels.
    draw_image_points(axes[1], data.points_2d)
    projected = project_points(target_to_camera(data.initial_markers, data.frame), data.intrinsic)
    axes[1].scatter(projected[[0, 2], 0], projected[[0, 2], 1], marker="x", s=100, color="white", linewidths=1.7, zorder=8)
    fig.suptitle("Saved local run: segmentation and keypoints", fontsize=17)
    fig.text(0.5, 0.04, "White crosses show the 3D points selected by the existing depth-neighbor heuristic; they need not equal the 2D matches.", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, 0.075, 1, 0.95))
    fig.savefig(output / "perception.png", dpi=140)
    plt.close(fig)

    # Preserve the VLM images exactly as saved; do not infer missing candidate
    # coordinates or regenerate the model selections during visualization.
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    for row, folder in enumerate(("func_point_transfer_output", "grasp_transfer_output")):
        label = "Function" if row == 0 else "Grasp"
        for col, filename in enumerate(("visual_prompting_candidates.png", "visual_prompting_result.png")):
            image_axis(axes[row, col], Image.open(data.root / folder / filename),
                       f"{label}: saved Gemini {'candidates' if col == 0 else 'selection'}")
        # BEGIN report the selected point method without implying refinement.
        image_axis(axes[row, 2], data.rgb, f"{label}: final {point_method} point", crop_limits(data.masks["tool"], 18))
        # END final point label.
        point = data.points_2d[label.lower()]
        axes[row, 2].scatter(*point, s=100, color=POINT_COLORS[0 if row == 0 else 2], edgecolor="black")
    # BEGIN title covers both refinement settings accepted by main.py.
    fig.suptitle("Semantic candidate selection and final keypoints", fontsize=16)
    # END configuration-independent title.
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output / "keypoint_selection.png", dpi=140)
    plt.close(fig)


def display_clouds(data):
    """Bound rendering cost; keep all trajectory poses and fixed camera limits."""
    tool, tool_colors = sample_cloud(data.tool, data.tool_colors, 4500)
    target_mask = data.scene_target_mask & ~data.scene_tool_mask
    target, target_colors = sample_cloud(data.scene[target_mask], data.scene_colors[target_mask], 6000)
    # Use the whole observed tool to set bounds, so display subsampling cannot
    # accidentally clip extremal geometry. Work one pose at a time to bound RAM.
    bounds = np.array([target.min(axis=0), target.max(axis=0)])
    for pose in data.trajectory:
        moved = move_points(data.tool, data.trajectory[0], pose)
        bounds[0] = np.minimum(bounds[0], moved.min(axis=0))
        bounds[1] = np.maximum(bounds[1], moved.max(axis=0))
    bounds += np.array([[-0.035], [0.035]])
    background_mask = ~data.scene_tool_mask & ~data.scene_target_mask
    background_mask &= ((data.scene >= bounds[0]) & (data.scene <= bounds[1])).all(axis=1)
    scene, scene_colors = sample_cloud(data.scene[background_mask], data.scene_colors[background_mask], 10000)
    return tool, tool_colors, target, target_colors, scene, scene_colors, bounds


def configure_3d(axis, bounds, elev=27, azim=-65):
    axis.set(xlim=bounds[:, 0], ylim=bounds[:, 1], zlim=bounds[:, 2],
             xlabel="X (m)", ylabel="Y (m)", zlabel="Z (m)")
    axis.set_box_aspect(bounds[1] - bounds[0])
    axis.view_init(elev=elev, azim=azim)
    axis.tick_params(labelsize=8)


def render_motion(data, output, clouds, fps):
    """Stream all frames into an MP4; FPS is playback only, not physical time."""
    import imageio.v2 as imageio
    import matplotlib.pyplot as plt

    tool, tool_colors, target, target_colors, scene, scene_colors, bounds = clouds
    fig = plt.figure(figsize=(14, 6), dpi=100)
    axes = [fig.add_subplot(121, projection="3d"), fig.add_subplot(122)]
    ax3, ax2 = axes
    configure_3d(ax3, bounds)
    ax3.scatter(*scene.T, c=scene_colors, s=0.8, alpha=0.32, depthshade=False)
    ax3.scatter(*target.T, c=target_colors, s=1.2, depthshade=False)
    tool_artist = ax3.scatter(*tool.T, c=tool_colors, s=2.0, depthshade=False)
    marker_artist = ax3.scatter(*data.initial_markers.T, c=POINT_COLORS, s=65, edgecolors="black", depthshade=False)
    path3, = ax3.plot([], [], [], color="#ee9b00", linewidth=1.6)
    ax3.set_title("Rigid tool replay in target coordinates")
    ax2.scatter(target[:, 0], target[:, 1], c=target_colors, s=2)
    top_tool = ax2.scatter(tool[:, 0], tool[:, 1], c=tool_colors, s=3)
    top_markers = ax2.scatter(data.initial_markers[:, 0], data.initial_markers[:, 1], c=POINT_COLORS, s=65, edgecolors="black")
    path2, = ax2.plot([], [], color="#ee9b00", linewidth=1.6)
    ax2.set(xlim=bounds[:, 0], ylim=bounds[:, 1], xlabel="X (m)", ylabel="Y (m)", title="Top view: function point relative to target")
    ax2.set_aspect("equal")
    ax2.grid(alpha=0.25)
    for name, color in zip(POINT_NAMES, POINT_COLORS):
        ax2.scatter([], [], color=color, edgecolors="black", label=name)
    ax2.legend(loc="upper right", fontsize=9)
    title = fig.suptitle("", fontsize=15)
    fig.text(0.5, 0.025, "Single-view RGB-D: hidden surfaces are unknown. Static original tool removed. Playback speed is illustrative.", ha="center", fontsize=10)
    fig.subplots_adjust(left=0.01, right=0.98, bottom=0.13, top=0.88, wspace=0.12)
    checkpoints = set(np.linspace(0, len(data.trajectory) - 1, 5).round().astype(int)) | {data.contact}
    with imageio.get_writer(output / "trajectory.mp4", fps=fps, codec="libx264", quality=8, macro_block_size=2,
                            ffmpeg_params=["-threads", "2"]) as writer:
        for index, pose in enumerate(data.trajectory):
            points = move_points(tool, data.trajectory[0], pose)
            markers = data.markers(index)
            tool_artist._offsets3d = tuple(points.T)
            marker_artist._offsets3d = tuple(markers.T)
            top_tool.set_offsets(points[:, :2])
            top_markers.set_offsets(markers[:, :2])
            trail = data.trajectory[:index + 1, 0]
            path3.set_data_3d(*trail.T)
            path2.set_data(trail[:, 0], trail[:, 1])
            title.set_text(f"Frame {index:02d} / {len(data.trajectory) - 1:02d}" +
                           ("  |  FUNCTION KEYFRAME" if index == data.contact else ""))
            fig.canvas.draw()
            writer.append_data(np.asarray(fig.canvas.buffer_rgba())[:, :, :3])
            if index in checkpoints:
                fig.savefig(output / f"frame_{index:03d}.png", dpi=120)
    plt.close(fig)
    # A static contact sheet allows review without a video player or WebGL.
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for axis, index in zip(axes.flat, sorted(checkpoints)):
        image_axis(axis, Image.open(output / f"frame_{index:03d}.png"), f"Frame {index}" + (" (function)" if index == data.contact else ""))
    for axis in list(axes.flat)[len(checkpoints):]:
        axis.set_axis_off()
    fig.tight_layout()
    fig.savefig(output / "trajectory_keyframes.png", dpi=130)
    plt.close(fig)


def render_interactive(data, output, clouds, fps):
    """Self-contained Plotly HTML with a persistent camera and every pose."""
    import plotly.graph_objects as go

    tool, tool_colors, target, target_colors, scene, scene_colors, bounds = clouds

    def trace(points, name, colors, size):
        rgb = [f"rgb({r},{g},{b})" for r, g, b in np.rint(colors * 255).astype(int)]
        return go.Scatter3d(x=points[:, 0], y=points[:, 1], z=points[:, 2], mode="markers", name=name,
                            marker=dict(color=rgb, size=size), hoverinfo="skip")

    fig = go.Figure(data=[trace(scene, "Static visible scene", scene_colors, 1.4),
                         trace(target, "Target", target_colors, 2), trace(tool, "Moving tool", tool_colors, 2),
                         go.Scatter3d(x=[], y=[], z=[], mode="markers+text", text=list(POINT_NAMES), name="Tool points",
                                      textposition="top center", marker=dict(color=list(POINT_COLORS), size=6)),
                         go.Scatter3d(x=[], y=[], z=[], mode="lines", name="Function-point path", line=dict(color="#ee9b00", width=5))])
    frames = []
    for index, pose in enumerate(data.trajectory):
        points, markers = move_points(tool, data.trajectory[0], pose), data.markers(index)
        trail = data.trajectory[:index + 1, 0]
        frames.append(go.Frame(name=str(index), traces=[2, 3, 4], data=[
            go.Scatter3d(x=points[:, 0], y=points[:, 1], z=points[:, 2]),
            go.Scatter3d(x=markers[:, 0], y=markers[:, 1], z=markers[:, 2]),
            go.Scatter3d(x=trail[:, 0], y=trail[:, 1], z=trail[:, 2]),
        ]))
    fig.frames = frames
    for index, update in zip([2, 3, 4], frames[0].data):
        fig.data[index].update(update)
    animation = dict(frame=dict(duration=1000 / fps, redraw=True), transition=dict(duration=0), fromcurrent=True, mode="immediate")
    fig.update_layout(
        title=f"Saved trajectory: {len(frames)} frames; function keyframe {data.contact}<br><sup>Drag to rotate, scroll to zoom. Axes in meters. Single-view geometry; playback FPS is illustrative.</sup>",
        scene=dict(**{f"{name}axis": dict(range=bounds[:, i].tolist(), title=f"{name.upper()} (m)") for i, name in enumerate("xyz")},
                   aspectmode="data", camera=dict(eye=dict(x=1.4, y=-1.6, z=1.0)), uirevision="saved-run"),
        margin=dict(l=0, r=0, b=110, t=85), height=820,
        updatemenus=[dict(type="buttons", direction="left", x=0.1, y=-0.1, buttons=[
            dict(label="Play", method="animate", args=[None, animation]),
            dict(label="Pause", method="animate", args=[[None], dict(mode="immediate", frame=dict(duration=0, redraw=False))]),
        ])],
        sliders=[dict(active=0, y=-0.02, currentvalue=dict(prefix="Frame: "), steps=[
            dict(label=str(i) + (" *" if i == data.contact else ""), method="animate",
                 args=[[str(i)], dict(mode="immediate", frame=dict(duration=0, redraw=True), transition=dict(duration=0))])
            for i in range(len(frames))
        ])],
    )
    fig.write_html(output / "trajectory.html", include_plotlyjs=True, auto_play=False)


def render_diagnostics(data, output):
    """Plot per-frame changes; do not report velocities without timestamps."""
    import matplotlib.pyplot as plt

    trajectory = data.trajectory
    frames = np.arange(len(trajectory))
    rotations = Rotation.from_euler("xyz", trajectory[:, 1])
    delta_euler = (np.diff(trajectory[:, 1], axis=0) + np.pi) % (2 * np.pi) - np.pi
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for i, name in enumerate("XYZ"):
        axes[0, 0].plot(frames, trajectory[:, 0, i], label=name)
    axes[0, 0].set(title="Function-point position", ylabel="Meters in target frame")
    axes[0, 0].legend()
    axes[0, 1].plot(frames, np.rad2deg((rotations * rotations[0].inv()).magnitude()))
    axes[0, 1].set(title="Tool rotation from initial pose", ylabel="SO(3) angle (degrees)")
    axes[1, 0].plot(frames[1:], 1000 * np.linalg.norm(np.diff(trajectory[:, 0], axis=0), axis=1))
    axes[1, 0].set(title="Translation between adjacent frames", ylabel="Millimeters / frame")
    axes[1, 1].plot(frames[1:], np.rad2deg(np.linalg.norm(delta_euler, axis=1)), label="Wrapped Euler norm (optimizer)")
    axes[1, 1].plot(frames[1:], np.rad2deg((rotations[1:] * rotations[:-1].inv()).magnitude()), label="SO(3) rotation angle")
    axes[1, 1].set(title="Orientation between adjacent frames", ylabel="Degrees / frame")
    axes[1, 1].legend(fontsize=8)
    for axis in axes.flat:
        axis.axvline(data.contact, color="#b34242", linestyle="--", alpha=0.6)
        axis.set_xlabel("Frame index (no physical timestamps)")
        axis.grid(alpha=0.2)
    fig.suptitle("Saved trajectory consistency; dashed line = function keyframe")
    fig.tight_layout()
    fig.savefig(output / "trajectory_diagnostics.png", dpi=140)
    plt.close(fig)


def write_index(data, output, diagnostics, fps):
    """Small offline entry page: images and video also work without WebGL."""
    # BEGIN describe the point method actually recorded by main.py.
    point_method = (data.manifest["feature_backend"] + " 匹配" if data.manifest["feature_matching"]
                    else "Gemini 选点（未启用特征匹配）")
    # END actual method description.
    rows = "".join(f"<tr><td>{escape(name)}</td><td>{info['matched_pixel_xy']}</td>"
                   f"<td>{info['in_tool_mask']}</td><td>{info['depth_m']:.3f} m</td>"
                   f"<td>{info['reprojection_difference_px']:.2f} px</td></tr>"
                   for name, info in diagnostics["pixel_checks"].items())
    content = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>FUNCTO 保存结果可视化核验</title>
<style>body{{max-width:1200px;margin:32px auto;padding:0 20px;font:16px/1.65 sans-serif;color:#223}}
img,video{{max-width:100%;border:1px solid #ddd}}th,td{{padding:8px 18px;border-bottom:1px solid #ddd}}
code{{overflow-wrap:anywhere}}a{{color:#0969b0}}</style>
<h1>FUNCTO 保存结果可视化核验</h1>
<p>来源：<code>{escape(str(data.root))}</code><br>任务：{escape(str(data.manifest['task']))}；
{len(data.trajectory)} 帧；功能关键帧 {data.contact}。仅读取实际保存结果，没有重新推理或修改轨迹。</p>
<h2>分割与二维 / 三维关键点</h2><img src="perception.png" alt="掩膜与关键点叠加">
<p>黄色：功能点；青色：抓取点；白色叉号：实际采用的三维点重投影。白叉与圆点的差异来自原有深度邻域选择。</p>
<table><tr><th>点</th><th>二维坐标</th><th>在工具掩膜内</th><th>二维点深度</th><th>三维重投影偏差</th></tr>{rows}</table>
<h2>语义选点与最终关键点</h2><img src="keypoint_selection.png" alt="保存的候选选择图与最终关键点">
<p>左两列为此次运行保存的 Gemini 候选和选择图，右列为最终{point_method}二维结果。原运行未保存相似度场，此处没有构造匹配热力图。</p>
<h2>完整轨迹回放</h2><p><a href="trajectory.html">打开可旋转、逐帧查看的三维点云（离线 HTML）</a> ·
<a href="trajectory.mp4">下载 MP4</a> · <a href="diagnostics.json">数值检查 JSON</a></p>
<video controls loop preload="metadata" src="trajectory.mp4"></video>
<p>视频 {fps:g} FPS 仅供观看，原轨迹没有时间戳。移动的是工具的可见点云；平移量是功能点位置，不能直接当作机器人末端位姿。
静态背景中已去掉原工具。遮挡空洞和不可见表面没有补造；目标坐标系原点是可见目标点云的包围盒中心。</p>
<img src="trajectory_keyframes.png" alt="轨迹关键帧"><h2>轨迹数值检查</h2>
<img src="trajectory_diagnostics.png" alt="轨迹位置和相邻帧变化">
<p>与原 PLY 保存的工具快照最大差异：{diagnostics['saved_ply_max_residual_m']:.3g} m。
关键帧约束和刚体回放一致性用于核验显示忠实于运行结果；这不构成分割精度、任务成功、无碰撞或机器人可执行性的证明。</p></html>"""
    (output / "index.html").write_text(content, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Completed main.py output directory")
    parser.add_argument("--demo-dir", type=Path, help="Override run.json's demonstration path if moved")
    parser.add_argument("--output-dir", type=Path, help="Defaults to RUN/visualization; rewrites only derived visuals")
    parser.add_argument("--fps", type=float, default=8, help="Illustrative playback FPS (not physical timing)")
    args = parser.parse_args()
    if not np.isfinite(args.fps) or args.fps <= 0:
        parser.error("--fps must be a finite positive number")
    import matplotlib
    matplotlib.use("Agg")

    print("Loading saved predictions and original point clouds...", flush=True)
    data = load_run(args.run_dir, args.demo_dir)
    diagnostics = diagnose(data)
    output = args.output_dir or data.root / "visualization"
    output.mkdir(parents=True, exist_ok=True)
    render_perception(data, output)
    render_diagnostics(data, output)
    clouds = display_clouds(data)
    diagnostics["display_point_counts"] = {"tool": len(clouds[0]), "target": len(clouds[2]), "background": len(clouds[4])}
    diagnostics["playback_fps_not_physical"] = args.fps
    print(f"Rendering all {len(data.trajectory)} frames on CPU...", flush=True)
    render_motion(data, output, clouds, args.fps)
    render_interactive(data, output, clouds, args.fps)
    (output / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2, allow_nan=False))
    write_index(data, output, diagnostics, args.fps)
    print(f"Visual review: {(output / 'index.html').resolve()}", flush=True)


if __name__ == "__main__":
    main()
