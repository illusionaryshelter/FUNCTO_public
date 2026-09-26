"""Offline, bounded-size views of object/part trajectories and their coordinates.

The renderer consumes fixed-body SE(3) poses. It does not infer hand poses,
landmarks, support planes, container interiors, or collision-free execution.
Each stage has its own axes because carriers may use different body frames.
"""

# BEGIN independent object-trajectory rendering and coordinate verification.
import json
from pathlib import Path

import numpy as np


def _points(value, name):
    points = np.asarray(value, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
        raise ValueError(f"{name} must contain nonempty finite Nx3 points")
    return points


def _poses(value, name):
    poses = np.asarray(value, dtype=float)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or not len(poses) or not np.isfinite(poses).all():
        raise ValueError(f"{name} must contain nonempty finite Nx4x4 poses")
    if not np.allclose(poses[:, 3], [0, 0, 0, 1], atol=1e-9, rtol=0):
        raise ValueError(f"{name} has an invalid homogeneous last row")
    rotations = poses[:, :3, :3]
    if (not np.allclose(rotations.transpose(0, 2, 1) @ rotations, np.eye(3), atol=1e-6, rtol=0)
            or not np.allclose(np.linalg.det(rotations), 1, atol=1e-6, rtol=0)):
        raise ValueError(f"{name} must contain proper rigid rotations")
    return poses


def transform_points(body_points, pose):
    """Apply ``reference_from_fixed_body`` without recentering local geometry.

    Row storage is only an array convention: the operation is the equivalent of
    ``pose @ [x_body, y_body, z_body, 1]`` on each column-vector body point.
    In particular, a nonzero body origin/centroid is not subtracted here.
    """
    points = _points(body_points, "body_points")
    matrix = np.asarray(pose, dtype=float)
    if matrix.shape != (4, 4):
        raise ValueError("pose must have shape 4x4")
    matrix = _poses(matrix[None], "pose")[0]
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def _indices(count, limit):
    return np.unique(np.linspace(0, count - 1, min(count, limit), dtype=int))


def _goal_geometry(stage):
    """Render only supplied bounded geometry, preserving its evidence source."""
    geometry = stage.get("goal_geometry")
    if geometry is None:
        return None, {"status": "not_supplied"}
    if not isinstance(geometry, dict):
        raise ValueError("goal_geometry must be a geometry dictionary or None")
    relation = stage["task"]["relation"]
    from scipy.spatial import ConvexHull, HalfspaceIntersection, QhullError
    from .goals import InteriorRegion, SupportSurface

    if relation == "on":
        surface = SupportSurface(
            origin=geometry["origin"], basis=geometry["basis"], normal=geometry["normal"],
            boundary=geometry["boundary"], source=geometry.get("source", "provided_support_geometry"))
        hull = ConvexHull(surface.boundary)
        vertices = surface.origin + surface.boundary[hull.vertices] @ surface.basis.T
        return {"kind": "support", "vertices": vertices, "faces": [list(range(len(vertices)))]}, {
            "status": "displayed", "kind": "support", "source": surface.source,
            "vertices_reference_m": vertices.tolist(),
        }
    if relation != "inside":
        raise ValueError("goal_geometry is only defined for on/inside relations")
    source = geometry.get("source")
    if "vertices" in geometry:
        region = InteriorRegion.from_vertices(geometry["vertices"], source)
        vertices = _points(geometry["vertices"], "Interior vertices")
    elif "halfspaces" in geometry:
        region = InteriorRegion(geometry["halfspaces"], source)
        from scipy.optimize import linprog

        normals, offsets = region.halfspaces[:, :3], region.halfspaces[:, 3]
        # Do not close an unbounded region by clipping it to the view box: that
        # would invent container walls. LP bounds along all three axes establish
        # that a finite polyhedron can actually be reconstructed for display.
        for direction in np.vstack((np.eye(3), -np.eye(3))):
            bound = linprog(direction, A_ub=normals, b_ub=-offsets,
                            bounds=[(None, None)] * 3, method="highs")
            if not bound.success:
                return None, {"status": "omitted", "kind": "interior", "source": source,
                              "reason": "halfspaces are unbounded, infeasible or numerically unresolved"}
        # A largest-inscribed-ball center is strictly feasible if this region
        # has volume. Region normals are unit length, so the fourth variable is
        # the radius in metres, not a guessed numerical interior point.
        center = linprog([0, 0, 0, -1], A_ub=np.column_stack((normals, np.ones(len(normals)))),
                         b_ub=-offsets, bounds=[(None, None)] * 3 + [(0, None)], method="highs")
        if not center.success or center.x[3] <= 1e-10:
            return None, {"status": "omitted", "kind": "interior", "source": source,
                          "reason": "no resolved strictly interior point for rendering"}
        try:
            vertices = HalfspaceIntersection(region.halfspaces, center.x[:3]).intersections
        except QhullError:
            return None, {"status": "omitted", "kind": "interior", "source": source,
                          "reason": "halfspace intersection is numerically unresolved"}
    else:
        raise ValueError("Interior goal geometry needs known vertices or halfspaces")
    vertices = _points(vertices, "Interior vertices")
    hull = ConvexHull(vertices)
    return {"kind": "interior", "vertices": vertices, "faces": hull.simplices.tolist()}, {
        "status": "displayed", "kind": "interior", "source": region.source,
        "vertices_reference_m": vertices.tolist(),
    }


def _background(stage):
    """Exclude the observed initial carrier using explicit point identities.

    A proximity threshold would also erase nearby reference surfaces. If no
    aligned carrier-membership mask is supplied, skip the optional background
    and record why, rather than showing an unexplained stationary carrier ghost.
    """
    if "scene_points" not in stage:
        return None, {"status": "not_supplied"}
    points = _points(stage["scene_points"], "scene_points")
    if "scene_carrier_mask" not in stage:
        return None, {"status": "omitted", "reason": "scene_carrier_mask not supplied",
                      "input_points": len(points)}
    membership = np.asarray(stage["scene_carrier_mask"])
    if membership.dtype != np.bool_ or membership.shape != (len(points),):
        raise ValueError("scene_carrier_mask must be a boolean membership mask for scene_points")
    remaining = points[~membership]
    if not len(remaining):
        return None, {"status": "empty_after_carrier_removal", "input_points": len(points),
                      "removed_carrier_points": int(membership.sum()), "displayed_points": 0}
    selected = remaining[_indices(len(remaining), 2500)]
    return selected, {"status": "displayed", "input_points": len(points),
                      "removed_carrier_points": int(membership.sum()), "displayed_points": len(selected)}


def _prepare_stage(stage):
    from scipy.spatial.distance import pdist
    from scipy.spatial.transform import Rotation

    if not isinstance(stage, dict):
        raise ValueError("Each stage must be a result dictionary")
    for field in ("carrier_id", "reference_id", "reference_frame"):
        if not isinstance(stage.get(field), str) or not stage[field].strip():
            raise ValueError(f"Stage needs a nonempty {field}")
    task = stage.get("task")
    if not isinstance(task, dict) or any(not isinstance(task.get(key), str) or not task[key]
                                         for key in ("relation", "interaction")):
        raise ValueError("Stage task needs relation and interaction")
    model = stage.get("model")
    if not isinstance(model, dict) or not isinstance(model.get("kind"), str):
        raise ValueError("Stage model needs its motion kind")
    poses = _poses(stage["poses"], "poses")
    points = _points(stage["body_points"], "body_points")
    goal = np.asarray(stage["goal_pose"], dtype=float)
    if goal.shape != (4, 4):
        raise ValueError("goal_pose must have shape 4x4")
    goal = _poses(goal[None], "goal_pose")[0]
    frame_ids = np.asarray(stage.get("frame_indices", np.arange(len(poses))))
    if (frame_ids.shape != (len(poses),) or frame_ids.dtype.kind not in "iu"
            or (frame_ids < 0).any() or (frame_ids[1:] <= frame_ids[:-1]).any()):
        # Compare adjacent IDs directly: int64 casting/subtraction can wrap
        # unsigned values and accept reversed IDs or reject valid large IDs.
        raise ValueError("frame_indices must be nonnegative strictly increasing integer IDs matching poses")
    goal_index = stage.get("goal_frame_index", len(poses) - 1)
    if (isinstance(goal_index, (bool, np.bool_)) or not isinstance(goal_index, (int, np.integer))
            or not 0 <= goal_index < len(poses)):
        raise ValueError("goal_frame_index must be a pose-array index within this stage")
    goal_index = int(goal_index)
    # At most five trajectory snapshots plus one target shape are displayed.
    snapshots = _indices(len(poses), 5)
    if goal_index not in snapshots:
        # The legacy tool subset binds its goal to an interior contact frame.
        # Keep both trajectory endpoints and replace the nearest interior sample
        # so that this goal event remains visible without exceeding the budget.
        nearest_interior = 1 + int(np.argmin(np.abs(snapshots[1:-1] - goal_index)))
        snapshots[nearest_interior] = goal_index
        snapshots.sort()
    body = points[_indices(len(points), 2500)]
    probes = points[_indices(len(points), 8)]
    local_distances = pdist(probes)
    transformed = [transform_points(body, poses[index]) for index in snapshots]
    goal_points = transform_points(body, goal)
    probe_reference = [transform_points(probes, poses[index]) for index in snapshots]
    goal_probes = transform_points(probes, goal)
    distance_error = max((float(np.max(np.abs(pdist(world) - local_distances)))
                          for world in [*probe_reference, goal_probes] if len(local_distances)), default=0.)
    rotation = poses[:, :3, :3]
    orthogonality = rotation.transpose(0, 2, 1) @ rotation - np.eye(3)
    geometry, geometry_report = _goal_geometry(stage)
    background, background_report = _background(stage)
    path_indices = _indices(len(poses), 1000)
    report = {
        "carrier_id": stage["carrier_id"], "reference_id": stage["reference_id"],
        "reference_frame": stage["reference_frame"], "task": task, "model_kind": model["kind"],
        "pose_convention": "column-vector reference_from_fixed_body; lengths in metres",
        "snapshot_semantics": "temporal copies of one rigid object/part, not multiple simultaneous objects",
        "pose_count": len(poses), "body_point_count": len(points), "displayed_body_points": len(body),
        "snapshot_indices": snapshots.tolist(), "snapshot_frame_ids": frame_ids[snapshots].tolist(),
        "goal_frame_index": goal_index, "goal_frame_id": int(frame_ids[goal_index]),
        "path_indices": path_indices.tolist(),
        "body_origin_reference_m": poses[snapshots, :3, 3].tolist(),
        "goal_body_origin_reference_m": goal[:3, 3].tolist(),
        "probe_body_points_m": probes.tolist(),
        "probe_reference_points_m": [world.tolist() for world in probe_reference],
        "goal_probe_reference_points_m": goal_probes.tolist(),
        "rigidity_probe_pair_count": len(local_distances),
        "max_probe_pair_distance_error_m": distance_error,
        "max_rotation_orthogonality_error": float(np.max(np.abs(orthogonality))),
        "goal_position_error_m": float(np.linalg.norm(poses[goal_index, :3, 3] - goal[:3, 3])),
        "goal_rotation_error_rad": float(Rotation.from_matrix(poses[goal_index, :3, :3] @ goal[:3, :3].T).magnitude()),
        "endpoint_position_error_m": float(np.linalg.norm(poses[-1, :3, 3] - goal[:3, 3])),
        "endpoint_rotation_error_rad": float(Rotation.from_matrix(poses[-1, :3, :3] @ goal[:3, :3].T).magnitude()),
        "goal_comparison": "goal checked at declared goal frame; endpoint distances are separate measurements",
        "provided_goal_checks": stage.get("goal_checks", {}),
        "goal_geometry": geometry_report, "background": background_report,
        "scope": "coordinate and sampled rigidity checks; not physical-execution validation",
    }
    return {"stage": stage, "poses": poses, "snapshots": snapshots, "frame_ids": frame_ids,
            "clouds": transformed, "goal_cloud": goal_points, "goal": goal,
            "path_indices": path_indices, "goal_index": goal_index,
            "geometry": geometry, "background": background,
            "report": report}


def _json_default(value):
    if isinstance(value, (np.ndarray, np.generic)):
        return value.tolist()
    raise TypeError(f"Cannot serialize coordinate evidence of type {type(value).__name__}")


def render_result(result, output_dir):
    """Write a static Agg PNG and coordinate-check JSON; return absolute paths.

    ``result`` is one stage or contains ``stages``. Each stage supplies poses,
    local body_points, goal_pose, task, model kind, and explicit frame/entity
    IDs. Optional goal_geometry uses the existing SupportSurface/InteriorRegion
    dictionary format. Optional scene_points are already in reference coordinates
    and require scene_carrier_mask for removing the initial carrier.
    goal_frame_index is an optional pose-array index (not an external frame ID);
    it defaults to the endpoint and can identify an intermediate tool-use event.

    Rendering is capped at 12 independent stage panels, 2500 body/background
    points per panel, five temporal snapshots plus the goal, and 1000 path nodes.
    The JSON records actual transformed probe coordinates, not image similarity.
    No global Matplotlib backend, model state, or working directory is changed.
    """
    if not isinstance(result, dict):
        raise ValueError("result must be a result dictionary")
    stages = result.get("stages", [result])
    if not isinstance(stages, (list, tuple)) or not 1 <= len(stages) <= 12:
        raise ValueError("Render between one and twelve stages per figure")
    prepared = [_prepare_stage(stage) for stage in stages]
    report = {"renderer": "object/part trajectory; CPU Agg",
              "stages": [item["report"] for item in prepared]}
    # Validate serializability before creating either output file.
    encoded = json.dumps(report, default=_json_default, allow_nan=False, indent=2)
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib import cm
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    columns = min(2, len(prepared))
    rows = (len(prepared) + columns - 1) // columns
    figure = Figure(figsize=(7.4 * columns, 6.6 * rows))
    FigureCanvasAgg(figure)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    png = destination / "trajectory.png"
    checks = destination / "coordinate_checks.json"
    try:
        for index, item in enumerate(prepared):
            stage, poses = item["stage"], item["poses"]
            axes = figure.add_subplot(rows, columns, index + 1, projection="3d")
            colors = cm.viridis(np.linspace(0.08, 0.88, len(item["snapshots"])))
            limits = [*item["clouds"], item["goal_cloud"], poses[:, :3, 3]]
            if item["background"] is not None:
                cloud = item["background"]
                axes.scatter(*cloud.T, c="0.7", s=1, alpha=0.18, label="reference/background (carrier removed)")
                limits.append(cloud)
            for snapshot, cloud, color in zip(item["snapshots"], item["clouds"], colors):
                axes.scatter(*cloud.T, c=[color], s=3, alpha=0.4,
                             label=f"temporal snapshot: frame {item['frame_ids'][snapshot]}")
            path = poses[item["path_indices"], :3, 3]
            axes.plot(*path.T, color="tab:blue", linewidth=1.5, label="sampled body-origin path")
            axes.scatter(*poses[0, :3, 3], c="tab:blue", s=42, marker="o", label="initial body origin")
            goal_frame_id = item["frame_ids"][item["goal_index"]]
            axes.scatter(*item["goal"][:3, 3], c="black", s=65, marker="*",
                         label=f"goal body origin (frame {goal_frame_id})")
            # Hollow markers distinguish the target shape from temporal copies.
            goal_display = item["goal_cloud"][_indices(len(item["goal_cloud"]), 300)]
            axes.scatter(*goal_display.T, facecolors="none", edgecolors="black", s=7,
                         linewidths=0.4, alpha=0.6, label="goal shape")
            geometry = item["geometry"]
            if geometry is not None:
                vertices = geometry["vertices"]
                faces = [vertices[face] for face in geometry["faces"]]
                axes.add_collection3d(Poly3DCollection(faces, facecolors="tab:orange", edgecolors="tab:orange",
                                                       linewidths=0.6, alpha=0.12))
                axes.plot([], [], [], color="tab:orange", label=f"provided {geometry['kind']} geometry")
                limits.append(vertices)
            minimum = np.min(np.vstack(limits), axis=0)
            maximum = np.max(np.vstack(limits), axis=0)
            center = (minimum + maximum) / 2
            span = max(float(np.max(maximum - minimum)), 0.02) * 1.15
            for dimension, setter in enumerate((axes.set_xlim, axes.set_ylim, axes.set_zlim)):
                setter(center[dimension] - span / 2, center[dimension] + span / 2)
            axes.set_box_aspect((1, 1, 1))
            axes.set(xlabel="reference X (m)", ylabel="reference Y (m)", zlabel="reference Z (m)")
            axes.set_title(
                f"Stage {index + 1}: object/part trajectory\n"
                f"{stage['carrier_id']} relative to {stage['reference_id']}\n"
                f"{stage['task']['relation']} / {stage['task']['interaction']} / {stage['model']['kind']}\n"
                f"reference frame: {stage['reference_frame']}", fontsize=10)
            axes.legend(loc="upper left", fontsize=6, framealpha=0.7)
            axes.text2D(0.02, -0.12, "Temporal copies; no hand pose or contact points inferred.\n"
                        "See coordinate_checks.json for transforms and goal-frame residuals.",
                        transform=axes.transAxes, fontsize=7)
        figure.subplots_adjust(left=0.04, right=0.94, bottom=0.14, top=0.87,
                               hspace=0.52, wspace=0.15)
        # 3-D axes adjust multiline title placement after projecting the view.
        # Include those final text extents so metric ranges cannot clip titles.
        figure.savefig(png, format="png", dpi=130, bbox_inches="tight", pad_inches=0.2)
        checks.write_text(encoded + "\n")
    finally:
        figure.clear()
    return {"trajectory_png": str(png.resolve()), "coordinate_checks_json": str(checks.resolve())}
# END independent object-trajectory rendering and coordinate verification.
