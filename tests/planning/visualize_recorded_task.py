#!/usr/bin/env python3
"""Visualize recorded-task input audits without implying robot execution.

The manifest and audit are explicit inputs produced by the recorded-task input
check. Paths in both documents are relative to ``manifest.workspace_root`` (or
absolute). The manifest's segment order is authoritative. Each recomputed
``reference_artifact`` is an NPZ containing ``eef_poses`` (N, 4, 4), original
``frame_indices`` (N,), and original ``timestamps`` (N,). These are task-space
references, not PlanningResult joint trajectories. Independent candidates are
rendered separately, and no gap is filled with demonstration motion.

This CPU-only report copies thumbnails and embeds Plotly's installed JavaScript
in index.html, so the resulting directory opens locally without a web server,
network access, GPU, video decoding, or a new dependency. It deliberately has
no planning/execution interface: actual PlanningResult visualization belongs to
the code that owns and validates that result.
"""

from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path
import textwrap
from typing import Any
from urllib.parse import quote

# Keep rendering within the same two-thread CPU budget as the input audit.
# Set these before importing NumPy or plotting libraries that initialize BLAS.
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/functo-recorded-task-matplotlib")

import matplotlib

matplotlib.use("Agg")
from matplotlib import font_manager
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs


NOTICE = "旧输入对比：参考/输入核验，未运行关节规划"
COLORS = {"blocked": "#b33e36", "passed": "#25755a", "warning": "#9b6b15"}
# Display labels describe the existing audit contract, not a task or dataset.
# Original codes and complete details remain visible in the HTML evidence.
BLOCKER_LABELS = {
    "named_joint_positions": "缺少命名关节位置反馈",
    "measured_joint_velocities": "缺少同步关节速度反馈",
    "robot_model_file_declared": "规划器尚未绑定机器人模型",
    "robot_base_pose_finite_4x4": "规划器尚未绑定机器人基座变换",
    "link_to_tcp_transform_finite_4x4": "规划器尚未绑定 link–TCP 变换",
    "robot_mask_files_aligned": "缺少各观测视角的机器人掩码",
    "collision_owner_contract_file_declared": "缺少完整碰撞 owner 合同",
    "free_interior_geometry": "缺少自由内部空间；inside 容纳未验证",
}
PHASE_LABELS = {
    "free approach reference": "接近参考",
    "protected grasp reference": "受保护抓取参考",
    "held object reference": "携带/操作参考",
    "released hand reference": "释放后撤退参考",
}


def read_json(path: Path) -> dict[str, Any]:
    """Read evidence without rewriting or normalizing its scientific content."""
    return json.loads(path.read_text(encoding="utf-8"))


def input_path(value: str, workspace: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else workspace / path


def evidence_link(value: str | None, workspace: Path, output: Path, label: str) -> str:
    """Keep source evidence inspectable; report images themselves are copied."""
    if value is None:
        return "未提供"
    relative = os.path.relpath(input_path(value, workspace), output)
    return f'<a href="{quote(relative, safe="/")}">{html.escape(label)}</a>'


def configure_font() -> None:
    """Use an already installed CJK font when present; never download fonts."""
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in ("WenQuanYi Micro Hei", "Noto Sans CJK SC", "Droid Sans Fallback"):
        if name in available:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            break
    plt.rcParams["axes.unicode_minus"] = False


def program_events(program: dict[str, Any]) -> list[dict[str, Any]]:
    """Expose recorded events; do not invent phase boundaries from EEF motion."""
    demo = program.get("demo", {})
    events = []
    for family, key in (("功能事件", "functional_events"),
                        ("夹爪意图", "gripper_events"),
                        ("物理证据", "physical_events")):
        for event in demo.get(key, []):
            if event.get("in_segment") is False:
                continue
            exact = event.get("frame_index")
            lower = event.get("frame_before_or_at")
            upper = event.get("frame_after_or_at")
            frame = str(exact) if exact is not None else f"[{lower}, {upper}]"
            # A physical event is bracketed by recorded samples, not assigned a
            # fabricated exact frame. The entire source is retained in the HTML.
            events.append({"family": family, "frame": frame,
                           "label": event.get("constraint_kind", event.get("command", event.get("kind", ""))),
                           "record": event})
    return events


def load_reference(path: Path) -> dict[str, np.ndarray]:
    """Check the plotting contract so malformed arrays cannot look successful."""
    with np.load(path, allow_pickle=False) as saved:
        poses = np.asarray(saved["eef_poses"], dtype=float)
        frames = np.asarray(saved["frame_indices"])
        times = np.asarray(saved["timestamps"], dtype=float)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or not len(poses):
        raise ValueError("eef_poses must have nonempty shape (N, 4, 4)")
    if frames.shape != (len(poses),) or times.shape != (len(poses),):
        raise ValueError("EEF poses, source frame indices, and timestamps must align")
    if not all(np.isfinite(values).all() for values in (poses, frames, times)):
        raise ValueError("reference arrays contain non-finite values")
    if np.any(np.diff(frames) <= 0) or np.any(np.diff(times) <= 0):
        raise ValueError("source frame indices and timestamps must increase strictly")
    if not np.allclose(poses[:, 3, :], [0, 0, 0, 1], atol=1e-8, rtol=0):
        raise ValueError("EEF poses do not have homogeneous-transform bottom rows")
    return {"xyz": poses[:, :3, 3], "frames": frames, "times": times}


def equal_axis_ranges(xyz: np.ndarray) -> np.ndarray:
    """Use the same metre scale on all axes without collapsing a near-line path."""
    middle = (xyz.min(axis=0) + xyz.max(axis=0)) / 2
    radius = max(float(np.ptp(xyz, axis=0).max()) * 0.55, 0.02)
    return np.column_stack((middle - radius, middle + radius))


def function_statuses(candidate: dict[str, Any]) -> list[str]:
    """Keep unverified function subchecks visible alongside recompute success."""
    statuses = []
    for check in candidate.get("function_checks", []):
        if "valid" in check:
            statuses.append(f"功能参考几何 valid={str(check['valid']).lower()}")
        for name, value in check.items():
            if isinstance(value, dict) and "status" in value:
                statuses.append(f"{name}: {value['status']}")
    return list(dict.fromkeys(statuses))


def reference_plot(reference: dict[str, np.ndarray], candidate: dict[str, Any]) -> str:
    """Render one candidate per coordinate system, with original frame hovers."""
    xyz, frames, times = reference["xyz"], reference["frames"], reference["times"]
    limits = equal_axis_ranges(xyz)
    custom = np.column_stack((frames, times))
    figure = go.Figure()
    figure.add_trace(go.Scatter3d(
        x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode="lines+markers",
        line={"color": "#346fc0", "width": 4},
        marker={"size": 2, "color": frames, "colorscale": "Viridis", "showscale": True,
                "colorbar": {"title": "源帧"}},
        customdata=custom, name="EEF reference",
        hovertemplate="源帧 %{customdata[0]:.0f}<br>原始时间 %{customdata[1]:.3f} s"
                      "<br>x=%{x:.4f}, y=%{y:.4f}, z=%{z:.4f} m<extra>参考 EEF</extra>"))
    for index, label, color in ((0, "参考起点", "#25755a"), (-1, "参考终点", "#b33e36")):
        figure.add_trace(go.Scatter3d(
            x=[xyz[index, 0]], y=[xyz[index, 1]], z=[xyz[index, 2]],
            mode="markers", name=label, marker={"size": 6, "color": color},
            hovertemplate=label + f" · 源帧 {int(frames[index])}<extra></extra>"))
    figure.update_layout(
        title={"text": html.escape(str(candidate["candidate_id"])) + " · EEF 参考（"
                       + html.escape(candidate["reference_frame"]) + "）", "font": {"size": 15}},
        height=400, margin={"l": 0, "r": 0, "t": 40, "b": 0},
        scene={"xaxis": {"title": "x (m)", "range": limits[0]},
               "yaxis": {"title": "y (m)", "range": limits[1]},
               "zaxis": {"title": "z (m)", "range": limits[2]}, "aspectmode": "cube"},
        legend={"orientation": "h", "y": 0}, paper_bgcolor="#ffffff")
    return figure.to_html(full_html=False, include_plotlyjs=False,
                          config={"displaylogo": False, "responsive": True})


def load_segments(manifest: dict[str, Any], audit: dict[str, Any],
                  workspace: Path, output: Path) -> list[dict[str, Any]]:
    """Collect display evidence in manifest order, preserving all failure cases."""
    manifest_ids = [row["id"] for row in manifest["segments"]]
    audit_rows = audit["segments"]
    audit_ids = [row["id"] for row in audit_rows]
    if len(set(manifest_ids)) != len(manifest_ids) or len(set(audit_ids)) != len(audit_ids):
        raise ValueError("manifest and audit segment IDs must be unique")
    if set(manifest_ids) != set(audit_ids):
        raise ValueError("manifest and audit must describe exactly the same segments")
    by_id = {row["id"]: row for row in audit_rows}
    assets = output / "assets"
    assets.mkdir(exist_ok=True)
    result = []
    for index, segment in enumerate(manifest["segments"]):
        row = {"manifest": segment, "audit": by_id[segment["id"]], "images": [],
               "references": [], "render_errors": [], "program": {}}
        for view_index, view in enumerate(segment["snapshot"]["views"]):
            source = input_path(view["rgb_path"], workspace)
            target = assets / f"segment_{index + 1:03d}_view_{view_index + 1:02d}.jpg"
            try:
                with Image.open(source) as image:
                    thumbnail = ImageOps.exif_transpose(image).convert("RGB")
                    thumbnail.thumbnail((960, 640), Image.Resampling.LANCZOS)
                    thumbnail.save(target, quality=88)
                row["images"].append({"view": view, "path": target})
            except (OSError, ValueError) as exc:
                row["render_errors"].append(f"RGB {view['id']}: {exc}")
        program_value = segment.get("program_path")
        if program_value:
            try:
                row["program"] = read_json(input_path(program_value, workspace))
            except (OSError, ValueError) as exc:
                row["render_errors"].append(f"program: {exc}")
        for candidate in row["audit"].get("grounding_recompute", []):
            artifact = candidate.get("reference_artifact")
            if artifact is None:
                continue
            try:
                if candidate["status"] != "passed" or candidate["reference_status"] != "reference_not_planned":
                    raise ValueError("reference artifact must be explicitly passed and reference_not_planned")
                reference = load_reference(input_path(artifact, workspace))
                source_start, source_end = segment["source_frame_range"]
                if reference["frames"][0] < source_start or reference["frames"][-1] > source_end:
                    raise ValueError("reference frames lie outside the source-program range")
                row["references"].append({"candidate": candidate, "data": reference})
            except (OSError, ValueError, KeyError) as exc:
                row["render_errors"].append(f"reference {candidate['candidate_id']}: {exc}")
        row["events"] = program_events(row["program"])
        row["phases"] = [(phase["name"], phase["start_frame"], phase["end_frame"])
                         for phase in row["audit"]["functional_phases"]]
        result.append(row)
    return result


def plot_overview(rows: list[dict[str, Any]], output: Path) -> None:
    """Give every segment equal space; no connection is drawn between scenes."""
    common = set.intersection(*({item["code"] for item in row["audit"]["planner_readiness"]["blockers"]}
                                for row in rows))
    common_order = [item["code"] for item in rows[0]["audit"]["planner_readiness"]["blockers"]
                    if item["code"] in common]
    figure = plt.figure(figsize=(19, 1.8 + 3.7 * len(rows)), facecolor="#f5f7fa")
    # Reserve real header space instead of letting the first 3D title overlap it.
    grid = figure.add_gridspec(len(rows), 3, width_ratios=(1.15, 1.0, 1.45),
                              left=0.025, right=0.99, bottom=0.02, top=0.915,
                              hspace=0.5, wspace=0.17)
    figure.suptitle(NOTICE + "\n各段独立历史观测；绿/红点为参考起/终点；没有段间执行衔接", fontsize=18, y=0.991)
    for line, first in enumerate(range(0, len(common_order), 3)):
        names = " · ".join(BLOCKER_LABELS.get(code, code) for code in common_order[first:first + 3])
        figure.text(0.04, 0.955 - line * 0.012, ("共同缺口：" if line == 0 else "　　　　　") + names,
                    fontsize=12, color=COLORS["blocked"], va="top")
    for index, row in enumerate(rows):
        segment, audit = row["manifest"], row["audit"]
        start, end = segment["recording_segment_frame_range"]
        source_start, source_end = segment["source_frame_range"]
        axis = figure.add_subplot(grid[index, 0])
        # A compact montage preserves all observed views without decoding any
        # source video. The full individual thumbnails are retained in HTML.
        thumbs = []
        for item in row["images"]:
            with Image.open(item["path"]) as image:
                thumbs.append(ImageOps.pad(image.convert("RGB"), (480, 320), color="#e8edf3"))
        if thumbs:
            montage = Image.new("RGB", (480 * len(thumbs), 320), "white")
            for offset, image in enumerate(thumbs):
                montage.paste(image, (480 * offset, 0))
            axis.imshow(montage)
        else:
            axis.text(0.5, 0.5, "没有可读取的 RGB 观测", ha="center", va="center")
        axis.set_title(f"{index + 1:02d} · {segment['name']}\n任务段 {start}–{end} · 参考源帧 {source_start}–{source_end}\n历史观测帧 {segment['snapshot']['entry_frame']}", fontsize=10)
        axis.set_xlabel(" / ".join(item["view"]["id"] for item in row["images"]), fontsize=9)
        axis.set_xticks([])
        axis.set_yticks([])

        path_axis = figure.add_subplot(grid[index, 1], projection="3d")
        if row["references"]:
            first = row["references"][0]
            xyz = first["data"]["xyz"]
            path_axis.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], color="#346fc0", linewidth=1.5)
            path_axis.scatter(*xyz[0], c="#25755a", s=23)
            path_axis.scatter(*xyz[-1], c="#b33e36", s=23)
            limits = equal_axis_ranges(xyz)
            path_axis.set_xlim(limits[0])
            path_axis.set_ylim(limits[1])
            path_axis.set_zlim(limits[2])
            path_axis.set_box_aspect((1, 1, 1))
            label = f"EEF 参考 · {first['candidate']['candidate_id']}\n仅展示该候选；未决选 · {len(xyz)} 个采样"
        else:
            path_axis.text2D(0.5, 0.5, "无可显示的 EEF 参考\n详见输入失败原因", ha="center", transform=path_axis.transAxes)
            label = "无参考曲线"
        path_axis.set_title(label, fontsize=10)
        path_axis.set_xlabel("x / m", fontsize=8, labelpad=0)
        path_axis.set_ylabel("y / m", fontsize=8, labelpad=0)
        path_axis.set_zlabel("z / m", fontsize=8, labelpad=0)
        path_axis.tick_params(labelsize=7, pad=0)
        for dimension in (path_axis.xaxis, path_axis.yaxis, path_axis.zaxis):
            dimension.set_major_locator(MaxNLocator(3))
        path_axis.view_init(elev=24, azim=-65)

        status_axis = figure.add_subplot(grid[index, 2])
        status_axis.axis("off")
        readiness = audit["planner_readiness"]
        candidates = audit.get("grounding_recompute", [])
        summary = [f"旧输入就绪检查：{'被阻塞' if readiness['status'] == 'blocked' else readiness['status']}；本页未运行关节规划",
                   f"参考重算：{len(row['references'])}/{len(candidates)}；本段图仅显示一个候选"]
        selected = segment["grounding"]["selected_candidate"]
        summary.append(f"旧候选：{len(candidates)} 个均未决选" if selected is None else f"旧输入声明选定：{selected}")
        summary.extend(f"{PHASE_LABELS.get(name, name)}：源帧 {first}–{last}" for name, first, last in row["phases"])
        for blocker in readiness.get("blockers", []):
            if blocker["code"] not in common:
                summary.append("额外缺口：" + BLOCKER_LABELS.get(blocker["code"], blocker["code"]))
        if any(candidate.get("contract_checks", {}).get("actual_holding_still_requires_verification")
               for candidate in candidates):
            summary.append("抓取/持有仍需同步观测验证")
        if any(candidate.get("contract_checks", {}).get("post_release_state_requires_observation")
               for candidate in candidates):
            summary.append("释放后的物体状态仍需新观测")
        summaries = [status for candidate in candidates for status in function_statuses(candidate)]
        summary.extend(list(dict.fromkeys(summaries)))
        for candidate in candidates:
            if candidate.get("error"):
                summary.append(f"• {candidate['candidate_id']}: {candidate['error']}")
        summary.append("共同缺口见页顶；全部证据见 index.html")
        # The overview keeps the readable blockers; the HTML retains complete
        # details and all events when the text exceeds one static panel.
        wrapped = []
        for line in summary:
            wrapped.extend(textwrap.wrap(line, width=66))
        if len(wrapped) > 18:
            wrapped = wrapped[:17] + ["…完整原因、阶段与证据见 index.html"]
        status_axis.text(0, 1, "\n".join(wrapped), va="top", fontsize=10.2,
                         color=COLORS.get(readiness["status"], "#263548"), linespacing=1.4)
    figure.savefig(output / "overview.png", dpi=135, facecolor=figure.get_facecolor())
    plt.close(figure)


def json_details(value: Any, label: str) -> str:
    return f"<details><summary>{html.escape(label)}</summary><pre>{html.escape(json.dumps(value, ensure_ascii=False, indent=2))}</pre></details>"


def robot_binding_html(audit: dict[str, Any], workspace: Path, output: Path) -> str:
    """Show the measured CPU model comparison without claiming native GPU FK."""
    audit_path = audit.get("robot_binding_audit_path")
    if audit_path is None:
        return ""
    binding = read_json(input_path(audit_path, workspace))
    pieces = ['<section id="robot-binding"><h2>机器人定义对照 · CPU</h2>',
              '<p>以下比较使用原始机器人模型定义，在 CPU 上核对 URDF 与仅含机器人的 MuJoCo 模型。此处没有运行 GPU 原生 FK、关节规划或机器人执行。</p>',
              f'<p>录制姿态：{len(binding["recorded_frame_indices"])} 个；关节范围探针：{binding["hypothetical_fk_probe_count"]} 个。最大偏差统计覆盖全部探针。对齐 TCP 不会修正其他 link 的碰撞几何。</p>',
              '<h3>对应 link 的最大原始偏差</h3><table><thead><tr><th>源机器人 body</th><th>URDF link</th><th>最大位置偏差 (mm)</th><th>最大旋转偏差 (rad)</th><th>相对首姿态的位置变差 (mm)</th></tr></thead><tbody>']
    for body, values in binding["mapped_link_relative_transforms"].items():
        pieces.append(f'<tr><td>{html.escape(body)}</td><td>{html.escape(values["urdf_link"])}</td><td>{1000 * values["maximum_translation_bias_m"]:.6g}</td><td>{values["maximum_rotation_bias_rad"]:.6g}</td><td>{1000 * values["maximum_translation_variation_m"]:.6g}</td></tr>')
    pieces.append('</tbody></table><h3>关节符号与范围</h3><table><thead><tr><th>源关节 → URDF 关节</th><th>q_urdf / q_native</th><th>源范围</th><th>URDF 范围</th></tr></thead><tbody>')
    for joint in binding["joint_binding"]:
        pieces.append(f'<tr><td>{html.escape(joint["recorded_name"])} → {html.escape(joint["urdf_name"])}</td><td>{joint["urdf_q_equals_native_q_times"]:+g}</td><td>{html.escape(str(joint["native_range"]))}</td><td>{html.escape(str(joint["urdf_range"]))}</td></tr>')
    pieces.append('</tbody></table><h3>碰撞球未覆盖的采样表面证据</h3><p>下表正值表示原始碰撞表面的采样顶点在所有机器人碰撞球的并集之外；这是未覆盖的具体证据，不是完整表面的最大误差上界。采样未发现缺口也不能证明完整覆盖。</p><table><thead><tr><th>原始碰撞 geom</th><th>对应 link</th><th>超出全部机器人球并集 (mm)</th><th>未覆盖顶点采样数</th></tr></thead><tbody>')
    for geom, values in binding["collision_sphere_surface_witnesses"].items():
        pieces.append(f'<tr><td>{html.escape(geom)}</td><td>{html.escape(values["urdf_link"])}</td><td>{1000 * values["maximum_all_robot_sphere_gap_m"]:.6g}</td><td>{values["uncovered_vertex_samples"]}</td></tr>')
    pieces.append('</tbody></table>')
    pieces.append(json_details({key: binding[key] for key in ("T_world_base", "T_urdf_tool_recorded_tcp", "rigid_tcp_corrected_vs_recorded_tcp", "rigid_tcp_correction_scope", "urdf_cpu_scope", "collision_coverage_certified")}, "基座/TCP 对齐数值及其适用范围"))
    pieces.append('<p>' + evidence_link(audit_path, workspace, output, "完整 CPU 模型对照 JSON") + '</p></section>')
    return "\n".join(pieces)


def write_html(rows: list[dict[str, Any]], audit: dict[str, Any],
               workspace: Path, output: Path) -> None:
    """Write one offline document; no CDN, fetch request, or guessed evidence."""
    sections = []
    for index, row in enumerate(rows):
        segment, checked = row["manifest"], row["audit"]
        start, end = segment["recording_segment_frame_range"]
        source_start, source_end = segment["source_frame_range"]
        readiness = checked["planner_readiness"]
        content = [f'<section id="segment-{index + 1}"><h2>{index + 1:02d} · {html.escape(segment["name"])}</h2>',
                   f'<p class="sub">{html.escape(segment["id"])} · 原始任务段帧 {start}–{end} · 参考程序源帧 {source_start}–{source_end} · 历史输入观测帧 {segment["snapshot"]["entry_frame"]}</p>',
                   f'<p class="badge">旧输入就绪检查：{html.escape(readiness["status"])} · {NOTICE}</p>',
                   '<div class="images">']
        for image in row["images"]:
            view = image["view"]
            content.append(f'<figure><img src="assets/{image["path"].name}" alt="{html.escape(segment["id"] + " " + view["id"], quote=True)} 的历史输入 RGB"><figcaption>{html.escape(view["id"])} · 历史独立观测</figcaption></figure>')
        content.append('</div><h3>输入缺口与重算结果</h3><ul>')
        for blocker in readiness.get("blockers", []):
            content.append(f'<li><strong>{html.escape(blocker["code"])}</strong> — {html.escape(blocker["detail"])}</li>')
        for error in row["render_errors"]:
            content.append(f'<li class="error">可视化读取失败：{html.escape(error)}</li>')
        content.append('</ul><p class="sub">参考重算 passed 表示相应输入与参考结果检查完成；候选筛选、在线持有验证及可执行性仍须单独验收。</p><table><thead><tr><th>候选</th><th>参考重算状态</th><th>registration_status</th><th>selected</th><th>功能参考检查</th><th>结果 / 失败原因 / 入口证据</th></tr></thead><tbody>')
        for candidate in checked.get("grounding_recompute", []):
            artifact = evidence_link(candidate.get("reference_artifact"), workspace, output, "重算参考 NPZ")
            detail = html.escape(str(candidate.get("error", "")))
            check_status = '<br>'.join(html.escape(value) for value in function_statuses(candidate))
            check_status += json_details(candidate.get("function_checks", []), "完整功能检查")
            holding = candidate.get("holding_source")
            if holding is not None:
                detail += json_details(holding, "受控 Holding 入口声明（非前段执行/在线验证）")
            content.append(f'<tr><td>{html.escape(candidate["candidate_id"])}</td><td>{html.escape(candidate["status"])}</td><td>{html.escape(candidate["registration_status"])}</td><td>{str(candidate["selected"]).lower()}</td><td>{check_status}</td><td>{artifact} {detail}</td></tr>')
        content.append('</tbody></table>')
        content.append('<h3>EEF 参考曲线</h3><p>以下曲线是每个候选各自场景坐标中的任务空间参考，单位为米；未验证关节运动、避障、段间衔接或执行。候选之间不连线，显示顺序不代表选定。拖动可旋转，悬停可检查原始帧与时间。</p><div class="plots">')
        for reference in row["references"]:
            content.append('<div class="plot">' + reference_plot(reference["data"], reference["candidate"]) + '</div>')
        if not row["references"]:
            content.append('<p class="error">没有通过绘图输入检查的参考曲线；未用源演示替代。</p>')
        content.append('</div><h3>功能阶段与事件证据</h3>')
        demo = row["program"].get("demo", {})
        content.append(f'<p>技能类型：{html.escape(str(row["program"].get("skill_kind", "未提供")))}；功能事件状态：<code>{html.escape(str(demo.get("functional_events_status", "未提供")))}</code>。夹爪命令表示意图；物理证据保留采样帧区间。</p>')
        content.append('<div class="phases">')
        for name, first, last in row["phases"]:
            content.append(f'<span>{html.escape(PHASE_LABELS.get(name, name))}<br>源帧 {first}–{last}</span>')
        content.append('</div><p class="sub">阶段来自源程序的显式边界，区间共享边界帧；不代表当前执行已达到该阶段。</p>')
        content.append('<table><thead><tr><th>类别</th><th>源帧 / 区间</th><th>事件</th><th>完整证据</th></tr></thead><tbody>')
        for event in row["events"]:
            content.append(f'<tr><td>{event["family"]}</td><td>{html.escape(event["frame"])}</td><td>{html.escape(str(event["label"]))}</td><td>{json_details(event["record"], "展开")}</td></tr>')
        if not row["events"]:
            content.append('<tr><td colspan="4">没有可读取的功能、夹爪或物理事件；没有推断事件或阶段。</td></tr>')
        content.append('</tbody></table>')
        content.append(json_details(checked, "本段完整输入核验记录（含阶段、时间与指标）"))
        content.append('<p class="sub">源程序：' + evidence_link(segment.get("program_path"), workspace, output, "程序文件")
                       + ' · 观测出处：' + evidence_link(segment["snapshot"].get("provenance_path"), workspace, output, "出处文件") + '</p></section>')
        sections.append("\n".join(content))
    nav = " ".join(f'<a href="#segment-{index + 1}">{index + 1:02d} {html.escape(row["manifest"]["name"])}</a>' for index, row in enumerate(rows))
    binding_section = robot_binding_html(audit, workspace, output)
    if binding_section:
        nav += ' <a href="#robot-binding">CPU 机器人定义对照</a>'
    document = """<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>旧输入对比 · 输入核验与参考可视化</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#edf1f6;color:#203047;font:16px/1.65 system-ui,sans-serif}
main{max-width:1450px;margin:auto;padding:24px}header,section{background:white;padding:24px;border-radius:12px;margin-bottom:24px}
h1{font-size:28px;line-height:1.3;margin:0 0 14px}h2{font-size:23px;margin:0}h3{margin:24px 0 10px;font-size:18px}
a{color:#235f9d}nav{display:flex;gap:8px;flex-wrap:wrap}nav a{padding:4px 10px;background:#edf3fa;border-radius:6px;text-decoration:none}
.notice{border-left:5px solid #b33e36;background:#fff1eb;padding:12px 16px;font-weight:650}.sub{color:#526175;font-size:14px}
.badge{display:inline-block;border-radius:5px;background:#fff0e9;color:#9d322b;padding:5px 10px}
.images,.plots{display:grid;grid-template-columns:repeat(auto-fit,minmax(350px,1fr));gap:16px}figure{margin:0}
img{width:100%;height:auto;display:block;border-radius:7px}figcaption{color:#526175;font-size:14px}table{width:100%;border-collapse:collapse;font-size:14px}
td,th{padding:9px;text-align:left;vertical-align:top;border-bottom:1px solid #dae1eb;overflow-wrap:anywhere}th{background:#f2f5f9}
pre{white-space:pre-wrap;overflow-wrap:anywhere;max-height:420px;overflow:auto;font:12px/1.5 ui-monospace,monospace;background:#f3f5f8;padding:12px}
code{overflow-wrap:anywhere}details{margin:8px 0}summary{cursor:pointer;color:#235f9d}.error{color:#9d322b}.plot{min-width:0}
.phases{display:flex;gap:4px;flex-wrap:wrap}.phases span{padding:9px 18px;background:#e8f0fa;border-left:4px solid #346fc0;flex:1}
@media(max-width:650px){main{padding:8px}header,section{padding:15px}.images,.plots{grid-template-columns:1fr}table{font-size:12px}}
</style><script>""" + get_plotlyjs() + """</script></head><body><main><header>
<h1>旧输入对比 · 输入核验与参考可视化</h1>
<p class="notice">""" + NOTICE + """</p>
<p>本页按源任务顺序对已保存的历史 RGB 输入快照重算 grounding 参考并核验。原有未决候选仅作旧输入对比，不表示新一轮按序 grounding 或规划失败。各段来自独立快照，不能证明上一段动作产生了下一段输入。</p>
<p>参考 EEF 曲线不包含关节规划结果。下方失败项保留实际核验原因，未将候选或静态曲线视为执行成功。</p>
<nav>""" + nav + """</nav><p><a href="overview.png">打开总览 PNG</a> · <a href="visualization-report.json">查看可视化完整性记录</a></p>
</header>""" + "\n".join(sections) + binding_section + "</main></body></html>"
    (output / "index.html").write_text(document, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest, audit = read_json(args.manifest), read_json(args.audit)
    if manifest["schema_version"] != 1:
        raise ValueError("only recorded-task manifest schema_version=1 is supported")
    if manifest.get("snapshots_are_independent") is not True:
        raise ValueError("this input report requires explicit independent snapshot provenance")
    if not manifest["segments"]:
        raise ValueError("manifest has no segments")
    workspace, output = Path(manifest["workspace_root"]), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    configure_font()
    rows = load_segments(manifest, audit, workspace, output)
    plot_overview(rows, output)
    write_html(rows, audit, workspace, output)
    report = {"notice": NOTICE, "evidence_scope": "historical_input_comparison",
              "reference_role": "reference_not_planned", "joint_planning_run": False,
              "manifest": str(args.manifest.resolve()), "audit": str(args.audit.resolve()),
              "segment_order": [row["manifest"]["id"] for row in rows],
              "segments": [{"id": row["manifest"]["id"], "image_count": len(row["images"]),
                            "reference_count": len(row["references"]),
                            "render_errors": row["render_errors"]} for row in rows]}
    (output / "visualization-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "segments": len(rows),
                      "images": sum(len(row["images"]) for row in rows),
                      "references": sum(len(row["references"]) for row in rows),
                      "render_errors": sum(len(row["render_errors"]) for row in rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
