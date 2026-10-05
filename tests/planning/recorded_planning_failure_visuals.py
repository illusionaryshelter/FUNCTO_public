"""Display saved planning evidence without rerunning a planning or scene query.

This module extends the existing recorded-task renderer. It keeps numerical
solver failure, incomplete verification, and physical execution separate. The
archived query inputs do not contain query verdicts; a recorded time is never
promoted to a verified or collision-free time. CPU FK is display geometry only.
"""

# BEGIN bounded CPU views of saved planner diagnostics and partial evidence.
from __future__ import annotations

from base64 import b64encode
from io import BytesIO
import html
import json
from pathlib import Path

from visualize_recorded_task import configure_font, read_json
from visualize_recorded_scene_failure import observations, transform
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from PIL import Image
import plotly.graph_objects as go
from scipy.interpolate import CubicHermiteSpline
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
import trimesh
from yourdfpy import URDF

from manipulation.collision_records import decode_array


STAGE_NAMES = {
    "rgbd_scene_build": "RGB-D 场景构建",
    "planner_initialize": "规划器初始化",
    "initial_state_verification": "入口实体检查",
    "reference_ik": "参考关节种子",
    "trajectory_numeric_optimizer": "轨迹数值优化",
    "candidate_metrics": "候选指标检查",
    "dense_interpolation_metrics": "密集插值检查",
    "export_verification": "最终曲线验收",
}
PATH_COLOR, SOURCE_COLOR = "#d07728", "#657e99"
FAILED_COLOR, UNKNOWN_COLOR = "#b63e41", "#a9b1bc"
METRIC_MEANINGS = {
    "cspace": "cuRobo 原生关节位置及导数限制的合并约束",
    "source_self_collision": "机器人已建模原始实体之间的必检碰撞对",
    "skill_process": "PlanningProblem 定义的运动过程约束",
    "process_scene": "PlanningScene 场景约束的合并残差；此项不能独立定位实体对",
    "position": "末端位置收敛诊断；此表保存全时域最大值",
    "orientation": "末端朝向收敛诊断；此表保存全时域最大值",
}


def _plain(value):
    return html.escape(str(value))


def _embedded_png(path):
    """Keep each HTML evidence view readable when copied without its assets."""
    encoded = b64encode(path.read_bytes()).decode("ascii")
    return f'<img src="data:image/png;base64,{encoded}" style="width:100%" alt="规划诊断静态图">'


def stage_rows(stage_report):
    """Choose disjoint measured scopes; never sum nested solver/query timers."""
    if stage_report is None:
        return []
    stages = stage_report["stages"]
    rows = []
    for name, label in STAGE_NAMES.items():
        if name not in stages:
            continue
        value = float(stages[name]["inclusive_wall_seconds"])
        if not np.isfinite(value) or value < 0:
            raise ValueError("Stage durations must be finite nonnegative wall times")
        rows.append({"name": name, "label": label, "seconds": value,
                     "raised": int(stages[name]["raised"])})
    return rows


def captured_query_summary(snapshot):
    """Describe exactly recorded inputs, with no inference about their verdicts.

    The producer may record exact/left/right material-mode queries together.
    Count their common physical input rows once only when q and time bytes
    agree across all declared mode blocks. Progress deliberately may differ.
    This display check does not repeat geometry or certify continuous time.
    """
    capture = snapshot.get("scene_query_capture")
    if capture is None:
        return {"recorded": False, "verdicts_available": False, "sample_times": []}
    times, modes, physical_rows = [], 0, 0
    for call in capture["calls"]:
        time = decode_array(call["time_s"])
        position = decode_array(call["joint_position"])
        if (time.ndim != 2 or position.shape[:2] != time.shape
                or not np.isfinite(time).all() or not np.isfinite(position).all()):
            raise ValueError("Saved scene-query q/time inputs must be aligned and finite")
        modes += time.size
        order = call["query_order"]
        if not order or time.shape[1] % len(order):
            raise ValueError("Saved material-mode blocks must have an explicit aligned order")
        width = time.shape[1] // len(order)
        base_t, base_q = time[:, :width], position[:, :width]
        for column in range(1, len(order)):
            other_t = time[:, column*width:(column+1)*width]
            other_q = position[:, column*width:(column+1)*width]
            if (other_t.tobytes() != base_t.tobytes()
                    or other_q.tobytes() != base_q.tobytes()):
                raise ValueError("Material modes changed physical q/time; do not merge their display count")
        physical_rows += base_t.size
        times.append(base_t.reshape(-1))
    saved_times = np.concatenate(times) if times else np.empty(0)
    display = np.unique(saved_times)
    # Deterministic thinning changes only marker count, never the range/counts.
    display = display[::max(1, int(np.ceil(len(display)/1500)))]
    return {"recorded": True, "recording_status": capture["status"],
            "verdicts_available": False, "calls": len(capture["calls"]),
            "mode_rows": modes, "physical_input_rows": physical_rows,
            "time_range_s": ([float(saved_times.min()), float(saved_times.max())]
                             if len(saved_times) else None),
            "sample_times": display.tolist(),
            "scope": "recorded query inputs only; no saved per-query verdict or continuous certificate"}


def native_metric_rows(document, seed_index):
    """Keep raw names/values and the producer's own constraint classification.

    Convergence values are maxima across a horizon, while the solver checks
    terminal values. They are therefore shown as diagnostics, not reclassified
    by comparing a maximum against a terminal tolerance.
    """
    if document is None or seed_index is None:
        return []
    if document["schema"] != "functo.native_candidate_metrics.1":
        raise ValueError("Unsupported native metrics schema")
    rows = []
    for seed in document["seeds"]:
        if seed["seed_index"] != seed_index:
            continue
        for scope in ("rollout", "interpolated"):
            values = seed["metrics"].get(scope)
            if values is None:
                continue
            failures = seed["positive_constraint_names"][scope]
            for family in ("constraints", "convergence"):
                for name, value in values[family].items():
                    failed = family == "constraints" and name in failures
                    nonfinite = f"{scope}.{family}.{name}" in seed["nonfinite_metric_names"]
                    rows.append({"seed": seed["seed_index"], "scope": scope,
                                 "family": family, "name": name, "value": value,
                                 "status": ("非有限/未知" if nonfinite or value is None else
                                            "正约束项" if failed else
                                            "终点判定不可由此最大值推断" if family == "convergence"
                                            else "非正约束项")})
    return rows


def candidate_document(directory, record, key, schema):
    """Read only the explicit producer binding; an older absent field is unknown."""
    if key not in record:
        return None, None
    path = directory / "planning" / record[key]
    if not path.is_file():
        return path, None
    document = read_json(path)
    if (document["schema"] != schema or document["native_result"] != record["native_result"]):
        raise ValueError("Candidate diagnostic must name the exact native result and schema")
    if ("seed_index" in document and document["seed_index"] != record["attempt"].get("seed_index")):
        raise ValueError("Candidate diagnostic must name the exact native seed")
    return path, document


def validation_summary(document, duration):
    """Preserve sample-count evidence without turning its endpoints into a proof.

    Source world counts include material modes, while checked_points counts
    physical samples. The producer publishes only completed batches. Neither
    range endpoints nor validation_complete mean the candidate was accepted.
    """
    if document is None:
        return None
    if document["schema"] != "functo.export_validation_progress.1":
        raise ValueError("Unsupported export validation schema")
    interval = document["checked_time_interval_s"]
    checked, planned = document["checked_points"], document["planned_points"]
    if checked < 0 or (planned is not None and not checked <= planned):
        raise ValueError("Completed physical sample count must lie inside the planned grid")
    if interval is not None:
        if (len(interval) != 2 or not np.isfinite(interval).all()
                or not 0 <= interval[0] <= interval[1] <= duration+1e-6 or not checked):
            raise ValueError("Completed sample endpoints must be finite and inside this curve")
    elif checked:
        raise ValueError("A positive checked sample count requires its saved endpoints")
    return {key: document.get(key) for key in (
        "status", "validation_complete", "completed_batches", "planned_points", "checked_points",
        "curve_duration_s", "checked_time_interval_s", "source_world_collision",
        "source_self_collision", "robot_coverage", "analytic_acceleration",
        "process_constraints", "native_joint_constraints", "exception", "budget_exhausted",
        "full_validation_residuals", "scope")}


def failure_witnesses(partial):
    """Copy original rejected-pair points; nearest allowed pairs are not failures.

    Query gaps, unmodelled parts and joint-limit failures may have no spatial
    witness. Their original reports remain available in the evidence JSON.
    Rendering does not query distance again or derive contact from proximity.
    """
    witnesses = []
    if partial is None:
        return witnesses
    for scope in ("source_world_collision", "source_self_collision"):
        report = partial.get(scope) or {}
        state = report.get("worst_state")
        if state is None:
            continue
        issues = (state["scene"]["violations"] if scope == "source_world_collision"
                  else state["violations"])
        for issue in issues:
            evidence = issue.get("evidence", issue)
            if "witness_first" not in evidence or "witness_second" not in evidence:
                continue
            points = np.asarray([evidence["witness_first"], evidence["witness_second"]], float)
            if points.shape != (2, 3) or not np.isfinite(points).all():
                raise ValueError("Original spatial witnesses must contain two finite world points")
            names = ([evidence["actor_geometry"], evidence["target_geometry"]]
                     if scope == "source_world_collision" else [evidence["first"], evidence["second"]])
            witnesses.append({"scope": scope, "time_s": state["trajectory_time_s"],
                "geometry_names": names, "points_world": points.tolist(),
                "signed_distance_m": evidence["signed_distance_m"],
                "reason": issue.get("kind", "required_source_self_pair_not_separated"),
                "original_evidence": evidence})
    return witnesses


# BEGIN display native failures even when independent export was never started.
def native_failure_evidence(document, seed_index):
    """Summarize bound native samples and copy original rejected pair points.

    Component counts include material modes and are not distinct geometry
    calls. Spatial witnesses are deduplicated for display only; their complete
    raw rows remain in the linked native metrics file. No geometry is queried.
    """
    components, witnesses, seen = {}, [], set()
    if document is None:
        return [], []
    for seed in document["seeds"]:
        if seed["seed_index"] != seed_index:
            continue
        for scope in ("rollout", "interpolated"):
            trace = seed["metrics"].get(scope, {}).get("source_trace", {})
            if trace.get("status") != "recorded":
                continue
            for row in trace["component_failures"]:
                key = (scope, row["component"])
                summary = components.setdefault(key, {"scope": scope,
                    "component": row["component"], "failed_mode_samples": 0,
                    "time_range_s": [row["time_s"], row["time_s"]]})
                summary["failed_mode_samples"] += 1
                summary["time_range_s"][0] = min(summary["time_range_s"][0], row["time_s"])
                summary["time_range_s"][1] = max(summary["time_range_s"][1], row["time_s"])
                for state in row.get("geometry_witnesses", []):
                    partial = {"source_world_collision": {"worst_state": {
                        "trajectory_time_s": state["trajectory_time_s"],
                        "scene": {"violations": state["violations"]}}}}
                    for witness in failure_witnesses(partial):
                        identity = json.dumps([scope, witness["time_s"], witness["geometry_names"],
                                               witness["points_world"]], sort_keys=True)
                        if identity in seen:
                            continue
                        seen.add(identity)
                        witnesses.append({**witness, "scope": "native_"+scope,
                            "evidence_origin": "native_source_trace_not_independent_validation",
                            "joint_names": state["planning_joint_names"],
                            "joint_position": state["planning_joint_positions"],
                            "native_component": row["component"]})
    return list(components.values()), witnesses
# END only the selected seed's bound failing samples supply native evidence.


def _pose7(values):
    pose = np.eye(4)
    values = np.asarray(values, dtype=float)
    if values.shape != (7,) or not np.isfinite(values).all():
        raise ValueError("Archived geometry poses require finite XYZ/wxyz")
    pose[:3, :3] = Rotation.from_quat(values[[4, 5, 6, 3]]).as_matrix()
    pose[:3, 3] = values[:3]
    return pose


def _mesh(vertices):
    """Render the archived filled convex body, preserving its hull vertices."""
    vertices = np.asarray(vertices, dtype=float)
    hull = ConvexHull(vertices)
    used, inverse = np.unique(hull.simplices, return_inverse=True)
    return vertices[used], inverse.reshape(-1, 3)


def _primitive(kind, dimensions):
    """Analytic primitives are tessellated for display only, never acceptance."""
    if kind in {"box", "cuboid"}:
        mesh = trimesh.creation.box(extents=dimensions)
    elif kind == "cylinder":
        mesh = trimesh.creation.cylinder(radius=dimensions[0], height=dimensions[1], sections=32)
    elif kind == "sphere":
        mesh = trimesh.creation.icosphere(subdivisions=2, radius=dimensions[0])
    else:
        raise ValueError(f"No declared display geometry for {kind}")
    return np.asarray(mesh.vertices), np.asarray(mesh.faces)


def robot_display(snapshot, curve, feedback, display_times, *, joint_positions=None):
    """Build CPU-only FK poses from the exact archived URDF and solid vertices.

    It is intentionally not a replay of the GPU FK. Saved native endpoint
    positions supply the path. CPU reconstructed bodies are contextual display
    geometry; query witnesses, when supplied, retain their original positions.
    Unknown future finger geometry is omitted, rather than fixing the fingers
    at their entrance or mistaking commanded motion for observed motion.
    """
    source = snapshot["scene"]["robot_self_collision"]
    model = source["model"]
    robot = URDF.load(BytesIO(model["urdf_xml"].encode()), build_scene_graph=True,
                      load_meshes=False, build_collision_scene_graph=False,
                      load_collision_meshes=False)
    baseline = dict(zip(feedback["joint_names"], feedback["start_q"]))
    base = np.asarray(source["T_reference_base"], float)
    curve_poly = CubicHermiteSpline(curve["times"], curve["position"], curve["velocity"], axis=0)
    # BEGIN failure poses may use the exact original native q instead of resampling.
    if joint_positions is not None:
        joint_positions = np.asarray(joint_positions)
        if (joint_positions.shape != (len(display_times), len(curve["names"]))
                or not np.isfinite(joint_positions).all()):
            raise ValueError("Explicit display states must align with the named curve and times")
    # END CPU FK remains contextual display, never a new collision verdict.
    hand = snapshot["scene"]["gripper"]
    nominal = None if hand is None else hand["nominal_model"]
    nominal_poly = (None if nominal is None else CubicHermiteSpline(
        nominal["times"], nominal["position"], nominal["velocity"], axis=0))
    finger_names = () if hand is None else tuple(hand["joint_names"])
    independent = set(finger_names)-set(curve["names"])
    unknown_frames = set()
    frontier = [robot.joint_map[name].child for name in independent] if nominal is None else []
    while frontier:
        child = frontier.pop()
        if child in unknown_frames:
            continue
        unknown_frames.add(child)
        frontier.extend(joint.child for joint in robot.joint_map.values() if joint.parent == child)
    meshes = []
    for item in model["geometries"]:
        if item["kind"] == "convex":
            vertices, faces = _mesh(item["vertices_frame"])
            local = np.eye(4)
        else:
            vertices, faces = _primitive(item["kind"], item["dimensions"])
            local = np.asarray(item["T_frame_geometry"], float)
        meshes.append({"name": item["name"], "frame": item["frame"],
                       "vertices": vertices, "faces": faces, "local": local,
                       "unknown_after_entrance": item["frame"] in unknown_frames})
    states = []
    for display_index, time_s in enumerate(display_times):
        positions = dict(baseline)
        # BEGIN keep supplied native states byte-for-byte before CPU FK conversion.
        values = curve_poly(time_s) if joint_positions is None else joint_positions[display_index]
        positions.update(zip(curve["names"], values))
        # END the optional source is explicit at this display-only call site.
        if nominal_poly is not None:
            positions.update(zip(finger_names, nominal_poly(time_s)))
        robot.update_cfg(positions)
        states.append({"time_s": float(time_s), "poses": [base @ robot.get_transform(
            mesh["frame"], frame_from=source["base_frame"]) @ mesh["local"] for mesh in meshes]})
    return meshes, states


def source_display(snapshot, maximum_points=12000):
    """Show sampled archived zero geometry; retain owner and map-layer labels.

    These are the frozen entrance shapes, not a reconstructed later owner
    state. Missing GPU nominal owner poses are never recomputed as evidence.
    A point display of zero geometry is not a complete occupied-volume model.
    """
    groups = [("background", snapshot["scene"]["background"]["geometry"], np.eye(4))]
    for owner in snapshot["scene"]["owners"]:
        if owner["native_geometry_names"]:
            continue  # Already displayed through the same robot installation model.
        path = owner["pose_path"]
        pose = np.asarray(path["poses"][0], float) @ np.asarray(path["attachment"], float)
        groups.append((owner["owner_id"], owner["geometry"], pose))
    clouds, counts = [], []
    for identity, geometry, pose in groups:
        layers = geometry.get("layers", {"single": geometry})
        for label, contents in layers.items():
            for item in contents["solids"]:
                if item["kind"] == "tsdf":
                    vertices = decode_array(item["surface_vertices"])
                    original = len(vertices)
                    selected = vertices[::max(1, int(np.ceil(original/maximum_points)))]
                    kind = "零值几何顶点抽样"
                elif item["kind"] == "mesh":
                    selected = np.asarray(item["vertices"], float)
                    original, kind = len(selected), "实体顶点"
                else:
                    selected, _ = _primitive(item["kind"], item["dimensions"])
                    original, kind = len(selected), "实体显示顶点"
                if len(selected):
                    clouds.append({"name": f"{identity} / {label} / {kind}",
                                   "points": transform(selected, pose @ _pose7(item["pose"])),
                                   "color": SOURCE_COLOR if identity == "background" else "#259788",
                                   "visible": "legendonly" if label == "fine" else True})
                counts.append({"owner": identity, "layer": label, "vertices": original,
                               "display_vertices": len(selected), "geometry_role": kind})
    return clouds, counts


def _metric_table(rows):
    if not rows:
        return ("<p><b>命名约束未知：</b>这份产物没有归档原生约束/收敛指标。"
                "原生 success=false 不能用于推断碰撞实体或失败位置。</p>")
    body = ["<p>原生 seed 是 cuRobo 候选索引；rollout 表示求解器原生采样，"
            "interpolated 表示原生密集插值采样。两者均先于最终导出曲线验收。</p>"
            "<table><tr><th>原生 seed</th><th>检查网格</th><th>指标族</th>"
            "<th>原始名称</th><th>归档最大值</th><th>判读</th><th>对应检查</th></tr>"]
    for row in rows:
        color = f' style="color:{FAILED_COLOR}"' if row["status"] == "正约束项" else ""
        cells = [row[key] for key in ("seed", "scope", "family", "name", "value", "status")]
        cells.append(METRIC_MEANINGS.get(row["name"], "归档原始指标；无额外细分记录"))
        body.append(f"<tr{color}>"+"".join(f"<td>{_plain(value)}</td>" for value in cells)+"</tr>")
    body.append("</table><p>约束保留原判定：沿时间轴求和 ≤ 0。收敛列是全时域最大值，"
                "原生求解器使用终点值判定，不能用此表的最大值代替。</p>")
    return "".join(body)


def _timeline(axis, duration, captured, partial, *, validation_status):
    """Keep query recording and completed verification on separate rows."""
    for row in (0, 1):
        axis.broken_barh([(0, duration)], (row-.25, .5), facecolors=UNKNOWN_COLOR, alpha=.35)
    samples = np.asarray(captured["sample_times"], float)
    if len(samples):
        axis.plot(samples, np.zeros(len(samples)), "|", color=PATH_COLOR, markersize=10)
    # BEGIN omitted validation is a known lifecycle decision, not lost evidence.
    label = ("独立验收：原生拒绝后未启动" if validation_status == "skipped_native_rejection"
             else "原分批验收：未归档 / 未知")
    # END no sampled native verdict implies complete export validation.
    if partial is not None:
        label = f"原分批验收：{partial['checked_points']} / {partial['planned_points']} 采样"
        interval = partial["checked_time_interval_s"]
        if interval is not None:
            start, stop = interval
            axis.broken_barh([(start, stop-start)], (.75, .5), facecolors="#4d7eab", alpha=.6,
                             hatch="///", edgecolors="#355574")
            axis.plot(interval, [1, 1], "|", color="#244764", markersize=12)
            axis.annotate(f"最后完成采样 t={stop:.6f} s", (stop, 1.3), ha="right", fontsize=9)
    axis.set_yticks([0, 1], ["已记录查询输入", label])
    axis.set_xlim(0, duration)
    axis.set_ylim(-.55, 1.65)
    axis.set_xlabel("候选曲线时间 / s")
    axis.set_title("蓝色斜线仅标已检查离散采样的起止范围，不证明连续区间通过；灰色为未知")
    axis.grid(axis="x", alpha=.2)


def _mesh_trace(mesh, pose, visible):
    vertices = transform(mesh["vertices"], pose)
    faces = mesh["faces"]
    return go.Mesh3d(x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2], name=mesh["name"],
        color="#7794b2", opacity=.65, visible=visible, flatshading=True,
        showlegend=False, hovertemplate=f"{_plain(mesh['name'])}<extra>CPU FK 诊断几何</extra>")


def scene_figure(curve, meshes, states, clouds, captured, witnesses):
    figure = go.Figure()
    for cloud in clouds:
        points = cloud["points"]
        figure.add_trace(go.Scatter3d(x=points[:, 0], y=points[:, 1], z=points[:, 2],
            mode="markers", name=cloud["name"], marker={"size": 1.2, "color": cloud["color"], "opacity": .4},
            visible=cloud["visible"], hoverinfo="name"))
    if "eef_position" in curve:
        xyz = curve["eef_position"]
        figure.add_trace(go.Scatter3d(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
            mode="lines", name="保存候选末端路径 · 未执行", line={"color": PATH_COLOR, "width": 6},
            customdata=curve["times"], hovertemplate="t=%{customdata:.6f} s<extra>保存的 GPU FK 节点</extra>"))
    for witness in witnesses:
        points = np.asarray(witness["points_world"])
        label = " ↔ ".join(witness["geometry_names"])
        figure.add_trace(go.Scatter3d(x=points[:, 0], y=points[:, 1], z=points[:, 2],
            mode="lines+markers", name=f"原始失败见证 t={witness['time_s']:.6f}s · {label}",
            line={"color": FAILED_COLOR, "width": 7}, marker={"color": FAILED_COLOR, "size": 5},
            hovertemplate=(f"{_plain(label)}<br>t={witness['time_s']:.6f} s"
                           f"<br>signed distance={witness['signed_distance_m']:.9g} m"
                           "<extra>归档的原始见证；不随显示姿态重算</extra>")))
    for index, state in enumerate(states):
        for mesh, pose in zip(meshes, state["poses"]):
            visible = index == 0 and not (state["time_s"] > 0 and mesh["unknown_after_entrance"])
            figure.add_trace(_mesh_trace(mesh, pose, visible))
    buttons = []
    for index, state in enumerate(states):
        visible = [cloud["visible"] for cloud in clouds]
        if "eef_position" in curve:
            visible.append(True)
        visible.extend(True for _ in witnesses)
        for row_index, other in enumerate(states):
            visible.extend(row_index == index and not (other["time_s"] > 0 and mesh["unknown_after_entrance"])
                           for mesh in meshes)
        buttons.append({"label": f"t={state['time_s']:.3f} s", "method": "update",
                        "args": [{"visible": visible}, {"title": f"候选诊断姿态 t={state['time_s']:.6f} s · 未执行"}]})
    figure.update_layout(height=720, title="候选诊断姿态 t=0 s · 未执行",
        scene={"aspectmode": "data", "xaxis_title": "世界 X / m", "yaxis_title": "世界 Y / m",
               "zaxis_title": "世界 Z / m", "camera": {"eye": {"x": 1.4, "y": -1.6, "z": .9}}},
        margin={"l": 0, "r": 0, "b": 0, "t": 90},
        updatemenus=[{"buttons": buttons, "direction": "down", "x": 0, "y": 1.1}],
        legend={"orientation": "h", "y": -.08})
    return figure


def _static_scene(axis, curve, meshes, state, clouds):
    for cloud in clouds:
        if cloud["visible"] is not True:
            continue
        points = cloud["points"][::3]
        axis.scatter(points[:, 0], points[:, 1], points[:, 2], s=.4, color=cloud["color"], alpha=.22)
    for mesh, pose in zip(meshes, state["poses"]):
        if state["time_s"] > 0 and mesh["unknown_after_entrance"]:
            continue
        points = transform(mesh["vertices"], pose)
        axis.add_collection3d(Poly3DCollection(points[mesh["faces"]], facecolor="#7392b3",
                                               edgecolor="none", alpha=.55, rasterized=True))
    if "eef_position" in curve:
        xyz = curve["eef_position"]
        axis.plot(*xyz.T, color=PATH_COLOR, linewidth=2.0, label="保存候选末端路径（未执行）")
        axis.scatter(*xyz[0], color="#328f63", s=25, label="入口")
        axis.scatter(*xyz[-1], color=PATH_COLOR, s=25, marker="x", label="候选末端")
        # Keep a bounded view around the robot, curve and complete source crop.
        all_points = np.concatenate([xyz, *[transform(m["vertices"], p) for m,p in zip(meshes, state["poses"]) ]])
        lower, upper = all_points.min(0), all_points.max(0)
        centre, span = (lower+upper)/2, max(float((upper-lower).max()), .2)*.56
        axis.set(xlim=(centre[0]-span, centre[0]+span), ylim=(centre[1]-span, centre[1]+span),
                 zlim=(centre[2]-span, centre[2]+span))
    axis.set(xlabel="世界 X / m", ylabel="世界 Y / m", zlabel="世界 Z / m")
    axis.set_box_aspect((1, 1, 1)); axis.view_init(elev=25, azim=-58)
    axis.set_title(f"CPU 诊断姿态 t={state['time_s']:.3f} s\n原始场景固定在入口；非物理执行")
    axis.legend(loc="upper left", fontsize=8)


def static_summary(path, report, rows, captured, partial, metric_rows, curve,
                   meshes, states, clouds, image_path, witnesses):
    configure_font()
    figure = plt.figure(figsize=(17, 12), layout="constrained")
    grid = figure.add_gridspec(3, 2, height_ratios=[1.1, 1.4, .7])
    axis = figure.add_subplot(grid[0, 0])
    if rows:
        positions = np.arange(len(rows))
        axis.barh(positions, [r["seconds"] for r in rows], color="#477d9d")
        axis.set_yticks(positions, [r["label"] for r in rows]); axis.invert_yaxis()
        for y, row in zip(positions, rows):
            axis.text(row["seconds"], y, f"  {row['seconds']:.2f}", va="center", fontsize=9)
        axis.set_xlim(0, max(r["seconds"] for r in rows)*1.19)
    else:
        axis.text(.5, .5, "未提供阶段计时记录", ha="center", transform=axis.transAxes)
    axis.set_title("互斥阶段耗时（不重复累加父阶段）"); axis.set_xlabel("墙钟时间 / s")
    axis = figure.add_subplot(grid[0, 1]); axis.axis("off")
    failing = [r for r in metric_rows if r["status"] == "正约束项"]
    lines = ["规划诊断 / 非物理执行", f"候选：{report['candidate_id']}",
             f"规划成功：{report.get('planning_success')}",
             # BEGIN use the producer's exact validation lifecycle.
             "独立验收："+{"not_completed": "未完成", "completed": "已完成",
                 "skipped_native_rejection": "原生检查拒绝，未启动"}[curve["metadata"]["validation_status"]],
             # END native checks cannot imply completed independent acceptance.
             ""]
    exception = None if partial is None else partial.get("exception")
    error = curve["attempt"].get("export_error")
    if exception is not None:
        lines += [f"导出中断：{exception['type']}"]
    elif error is not None:
        lines += ["导出中断："+curve["attempt"].get("export_error_type", "异常类型未归档")]
    if ((partial is not None and partial.get("budget_exhausted"))
            or error == "The original cumulative planning wall-time budget is exhausted"):
        lines += ["原累计规划墙钟预算已耗尽；不能据此判定碰撞。"]
    if not metric_rows:
        lines += ["原生命名失败约束：未归档 / 未知", "无法据 success=false 判定哪一实体碰撞。"]
    elif failing:
        lines += ["原生命名正约束项："]+[f"{r['scope']} / {r['name']} = {r['value']:.5g}" for r in failing[:6]]
        if len(failing) > 6:
            lines.append(f"其余 {len(failing)-6} 项见 HTML 完整表格。")
    else:
        lines += ["原生约束表无正项；收敛终点失败原因", "需要对应终点指标，不能由全时域最大值推断。"]
    lines += ["", f"原始失败几何见证：{len(witnesses)} 对（另有局部图）" if witnesses
              else "原始失败几何见证：未提供；碰撞位置未知。"]
    axis.text(0, .98, "\n".join(lines), va="top", linespacing=1.35, fontsize=10)
    axis = figure.add_subplot(grid[1, 0], projection="3d")
    _static_scene(axis, curve, meshes, states[0], clouds)
    axis = figure.add_subplot(grid[1, 1]); axis.axis("off")
    if image_path is not None:
        axis.imshow(Image.open(image_path))
        axis.set_title("本候选入口视角 RGB · 橙：机器人；青：目标观测")
    else:
        axis.text(.5, .5, "观测图像未提供", ha="center", transform=axis.transAxes)
    axis = figure.add_subplot(grid[2, :])
    _timeline(axis, float(curve["times"][-1]), captured, partial,
              validation_status=curve["metadata"]["validation_status"])
    figure.suptitle("规划失败证据：计算耗时、验证范围与候选现象", fontsize=19)
    figure.savefig(path, dpi=135); plt.close(figure)


def static_witnesses(path, witnesses, curve, meshes, states, clouds):
    """Magnify original world points; retain CPU contextual geometry as such."""
    if not witnesses:
        return
    configure_font()
    shown = witnesses[:4]
    figure = plt.figure(figsize=(14, 6*int(np.ceil(len(shown)/2))), layout="constrained")
    for index, witness in enumerate(shown):
        # BEGIN keep original witness markers readable in a transparent close-up.
        axis = figure.add_subplot(int(np.ceil(len(shown)/2)), 2, index+1,
                                  projection="3d", computed_zorder=False)
        state = next(state for state in states if state["time_s"] == witness["time_s"])
        _static_scene(axis, curve, meshes, state, clouds)
        for collection in axis.collections:
            if isinstance(collection, Poly3DCollection):
                collection.set_alpha(.1)
        points = np.asarray(witness["points_world"])
        axis.plot(*points.T, "o-", color=FAILED_COLOR, linewidth=3, markersize=7, zorder=20)
        # END contextual bodies cannot cover the exact archived red points.
        center = points.mean(0)
        span = max(.06, float(np.linalg.norm(points[1]-points[0]))*1.5)
        axis.set(xlim=(center[0]-span, center[0]+span), ylim=(center[1]-span, center[1]+span),
                 zlim=(center[2]-span, center[2]+span))
        label = " ↔ ".join(witness["geometry_names"])
        axis.set_title(f"原始失败见证 t={witness['time_s']:.6f} s\n{label}\n"
                       f"signed distance={witness['signed_distance_m']:.9g} m", fontsize=10)
        if axis.legend_ is not None:
            axis.legend_.remove()
    figure.suptitle("红点/线来自原始失败记录；蓝色机器人仅为 CPU 诊断姿态", fontsize=15)
    figure.savefig(path, dpi=135); plt.close(figure)


def _partial_table(partial, witnesses):
    if partial is None:
        return "<p><b>分批验收证据未知：</b>未归档已完成采样数及失败见证；不能由查询输入反推。</p>"
    world = partial.get("source_world_collision") or {}
    local = partial.get("source_self_collision") or {}
    coverage = partial.get("robot_coverage") or {}
    values = {
        "记录状态 / 验收计算完成（不等于接受）": [partial["status"], partial["validation_complete"]],
        "完整批次 / 已检查物理采样 / 计划采样": [partial["completed_batches"], partial["checked_points"], partial["planned_points"]],
        "已检查采样起止时刻（非连续通过区间） / s": partial["checked_time_interval_s"],
        "世界实体检查状态数 / 失败状态数（含 material modes）": [world.get("checked_states"), world.get("failed_states")],
        "自碰撞检查状态数 / 未建模几何": [local.get("checked_states"), local.get("unmodelled_geometry_names")],
        "机器人覆盖检查通过 / 禁止尝试状态数": [coverage.get("passed"), coverage.get("attempt_disallowed_states")],
        "全曲线过程约束档案状态": partial.get("process_constraints"),
        "异常 / 原累计墙钟预算耗尽": [partial.get("exception"), partial.get("budget_exhausted")],
        "原始失败空间见证数": len(witnesses),
    }
    body = ["<table><tr><th>验收记录项</th><th>原始值</th></tr>"]
    for key, value in values.items():
        body.append(f"<tr><td>{_plain(key)}</td><td>{_plain(json.dumps(value, ensure_ascii=False))}</td></tr>")
    body.append("</table><p>世界实体状态数含 exact/left/right 材料模式，不等于独立 Coal 几何调用数。"
                "未提供空间见证的失败不能从画面推断接触位置。完整原始报告保留在证据 JSON 中。</p>")
    return "".join(body)


def render_failure_diagnostics(run, segment, manifest, output, run_index, stage_path=None):
    """Add a self-contained evidence panel to the existing current-run page."""
    planning = run["planning"]
    if planning is None:
        return "<p>规划诊断：尚无规划报告。</p>", []
    stage_report = None if stage_path is None else read_json(stage_path)
    rows = stage_rows(stage_report)
    body, evidence = [], []
    body.append("<h4>规划失败证据与诊断几何</h4><p>以下候选运动来自保存的曲线，"
                "不是机器人执行。命名指标、查询输入、已完成验收和未知范围分别显示。</p>")
    by_path = {curve["path"].name: curve for curve in run["curves"]}
    for index, record in enumerate(planning.get("candidates", [])):
        curve = by_path.get(record["path"])
        if curve is None:
            continue
        prefix = f"run_{run_index:03d}_candidate_{index:02d}"
        snapshot_record = record.get("collision_snapshot")
        if not snapshot_record or not snapshot_record.get("saved"):
            body.append(f"<p>{_plain(record['path'])}：没有保存来源场景快照，几何展示未知。</p>")
            continue
        snapshot_path = run["directory"] / "planning" / snapshot_record["path"]
        snapshot = read_json(snapshot_path)
        if snapshot["curve_path"] != curve["path"].name:
            raise ValueError("Scene snapshot must bind the exact displayed candidate")
        if snapshot["scene"]["reference_frame"] != "world":
            raise ValueError("Current RGB-D overlay requires an explicitly world-frame scene")
        captured = captured_query_summary(snapshot)
        # Release large mode/layer payloads as soon as their input range is read.
        # Their original bytes stay in the linked archive; no query is rerun.
        snapshot.pop("scene_query_capture", None)
        metric_path, metric_document = candidate_document(run["directory"], record,
            "native_metrics_path", "functo.native_candidate_metrics.1")
        metrics = native_metric_rows(metric_document, record["attempt"].get("seed_index"))
        partial_path, partial_document = candidate_document(run["directory"], record,
            "export_validation_path", "functo.export_validation_progress.1")
        partial = validation_summary(partial_document, float(curve["times"][-1]))
        # BEGIN native rejection still has original source-component evidence.
        components, native_witnesses = native_failure_evidence(
            metric_document, record["attempt"].get("seed_index"))
        all_witnesses = failure_witnesses(partial) + native_witnesses
        # Keep CPU display work bounded; the full evidence and raw native rows
        # remain saved. Show the earliest and the closest recorded pair first.
        witnesses = []
        if all_witnesses:
            ordered = [min(all_witnesses, key=lambda item: item["time_s"]),
                       min(all_witnesses, key=lambda item: item["signed_distance_m"]),
                       *all_witnesses]
            for witness in ordered:
                if witness not in witnesses:
                    witnesses.append(witness)
                if len(witnesses) == 4:
                    break
        # END graphics are a labelled subset, not a substitute for validation.
        display_times = np.unique(np.r_[curve["times"][0], np.linspace(0, curve["times"][-1], 5),
            [] if captured.get("time_range_s") is None else captured["time_range_s"][-1],
            [witness["time_s"] for witness in witnesses],
            [] if partial is None or partial["checked_time_interval_s"] is None else partial["checked_time_interval_s"][-1]])
        feedback = read_json(Path(run["report"]["feedback_path"]))
        meshes, states = robot_display(snapshot, curve, feedback, display_times)
        clouds, source_counts = source_display(snapshot)
        # Reuse the original CPU calibrated depth/mask renderer. Its crop and
        # deterministic thinning affect graphics only, not verification inputs.
        points = np.concatenate([transform(mesh["vertices"], pose) for mesh,pose in zip(meshes, states[0]["poses"])])
        if "eef_position" in curve:
            points = np.r_[points, curve["eef_position"]]
        lower, upper = points.min(0)-.2, points.max(0)+.2
        _, images, observation_counts = observations(manifest, segment, run["report"],
            run["directory"], output / "assets", lower, upper)
        preferred = run["report"]["candidate_id"].split("/")[0]
        selected_image = next((row for row in images if row["name"] == preferred), images[0] if images else None)
        image_path = None if selected_image is None else output / "assets" / selected_image["masks"]
        static_path = output / (prefix+"_summary.png")
        static_summary(static_path, run["report"], rows, captured, partial, metrics,
                       curve, meshes, states, clouds, image_path, witnesses)
        figure = scene_figure(curve, meshes, states, clouds, captured, witnesses)
        body.append(f"<h5>{_plain(record['path'])}</h5>"+_embedded_png(static_path))
        body.append("<p><b>原生求解标记：</b>"+_plain(record["attempt"].get("native_success"))+"；"
                    "<b>导出结果：</b>"+_plain(record["attempt"].get("export_error", "见原候选报告"))+"。</p>")
        body.append(_metric_table(metrics))
        # BEGIN display native component names/times without calling them export checks.
        if record["validation_status"] == "skipped_native_rejection":
            body.append("<p><b>独立验收未启动：</b>原生约束检查已经拒绝该候选。</p>")
        else:
            body.append(_partial_table(partial, witnesses))
        if components:
            body.append("<p>原生失败分量（次数包含材料模式；时间范围不表示连续碰撞）：</p><pre>"+
                        _plain(json.dumps(components, ensure_ascii=False, indent=2))+"</pre>")
        body.append(f"<p>空间见证展示 {len(witnesses)} / {len(all_witnesses)} 条；完整见证保存在证据 JSON。</p>")
        # END no skipped export is described as a missing or completed audit.
        witness_path = output / (prefix+"_witnesses.png")
        if witnesses:
            static_witnesses(witness_path, witnesses, curve, meshes, states, clouds)
            body.append(_embedded_png(witness_path))
        body.append(figure.to_html(full_html=False, include_plotlyjs=False,
                                   config={"displaylogo": False, "responsive": True}))
        body.append("<p>三维蓝色机器人由归档 URDF/实体顶点独立 CPU FK 展示；"
                    "橙线直接使用保存的 GPU 末端节点。灰/青点为入口场景零值几何的显示抽样，"
                    "未替代负值体积模型，也不证明未知空间为空。未来手指没有名义模型时隐藏其实体；"
                    "原始失败见证未提供时，不从画面接近关系推断碰撞。</p>")
        summary = {"candidate": curve["path"].name, "stage_report": None if stage_path is None else str(stage_path),
                   "scene_snapshot": str(snapshot_path), "captured_query_inputs": captured,
                   "native_metrics_path": str(metric_path) if metric_document is not None else None,
                   "native_metric_rows": metrics,
                   "partial_validation_path": None if partial_path is None else str(partial_path),
                   "partial_validation": partial, "original_failure_witnesses": all_witnesses,
                   "displayed_failure_witnesses": witnesses, "native_failure_components": components,
                   "source_display_counts": source_counts, "observation_display_counts": observation_counts,
                   "display_times_s": display_times.tolist(), "unknown_future_finger_geometry_hidden": True,
                   "geometry_evaluations_performed": 0, "gpu_used": False, "physics_steps": 0,
                   "cpu_geometry_is_original_gpu_query_evidence": False,
                   "static_png": static_path.name,
                   "witness_png": witness_path.name if witnesses else None}
        (output / (prefix+"_evidence.json")).write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n")
        evidence.append(summary)
    return "".join(body), evidence
# END no display operation produces an executable trajectory or query verdict.
