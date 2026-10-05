#!/usr/bin/env python3
"""Render actual recorded-task planner and physics artifacts on the CPU.

The manifest determines segment order and the original RGB-D observations.
Only explicitly supplied run directories contribute current results. Each
simulation-report.json binds its trial to an exact saved trajectory path;
nearby filenames, historical candidate selection and elapsed time never stand
in for that identity or for a physical success decision. Missing segments stay
untested. The frozen input-reference visualization is never modified.

Example: --manifest inputs.json --audit input-audit.json --run-dir run-001
         --run-dir run-002 --output current-visualization
An external trial directory can additionally be supplied with --trial-dir.
"""

# BEGIN independent current-run visualization with explicit evidence identities.
from __future__ import annotations

import argparse
from collections import Counter
import html
import json
import os
from pathlib import Path
import resource
import subprocess
import time
from urllib.parse import quote

# Import the established CPU renderer before numerical libraries: it sets the
# two-thread budget and the noninteractive Matplotlib backend, without CUDA.
from visualize_recorded_task import (configure_font, equal_axis_ranges,
                                     input_path, load_reference, read_json)
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
import plotly.graph_objects as go
from plotly.colors import qualitative
from plotly.offline import get_plotlyjs
from plotly.subplots import make_subplots
from scipy.interpolate import CubicHermiteSpline


CLASSIFICATIONS = {
    "planner_accepted": ("规划接受", "#2677b8"),
    "rejected_diagnostic_not_executable": ("规划拒绝 · 仅诊断曲线", "#cb6e1e"),
}
PHYSICS = {"passed": "物理判定通过", "failed": "物理判定失败",
           "not_evaluated": "物理未判定", "unverified": "已执行 · 完整任务谓词未验证"}


def escape(value):
    return html.escape(str(value))


def link(path, output, label):
    """Link to existing local evidence; report assets stay relative and offline."""
    relative = os.path.relpath(Path(path).resolve(), output)
    return f'<a href="{quote(relative, safe="/")}">{escape(label)}</a>'


def details(label, value):
    return (f"<details><summary>{escape(label)}</summary><pre>"
            f"{escape(json.dumps(value, ensure_ascii=False, indent=2))}</pre></details>")


def plot_html(figure):
    return figure.to_html(full_html=False, include_plotlyjs=False,
                          config={"displaylogo": False, "responsive": True})


def load_curve(path, expected_classification):
    """Validate and preserve the exported named cubic and its knot-only FK."""
    metadata = read_json(path.with_suffix(".json"))
    with np.load(path, allow_pickle=False) as saved:
        curve = {name: saved[name].copy() for name in saved.files}
    classification = str(curve["classification"].item())
    if (classification not in CLASSIFICATIONS or classification != expected_classification
            or metadata["classification"] != classification):
        raise ValueError("Curve arrays, metadata and planning report disagree on classification")
    if metadata["validation_status"] not in {"completed", "not_completed"}:
        raise ValueError("Every current curve must declare whether final verification completed")
    if classification == "planner_accepted" and metadata["validation_status"] != "completed":
        raise ValueError("A curve without completed verification cannot be planner_accepted")
    times, position, velocity = (np.asarray(curve[key], float)
                                 for key in ("times", "position", "velocity"))
    names = tuple(curve["joint_names"].tolist())
    if (times.ndim != 1 or not 2 <= len(times) <= 8192 or times[0] != 0
            or np.any(np.diff(times) <= 0) or not names or len(set(names)) != len(names)
            or position.shape != (len(times), len(names)) or velocity.shape != position.shape
            or not all(np.isfinite(array).all() for array in (times, position, velocity))):
        raise ValueError("Exported times, named q and dq must be aligned and finite")
    if "eef_position" in curve:
        if (curve["eef_position"].shape != (len(times), 3)
                or not np.isfinite(curve["eef_position"]).all()):
            raise ValueError("Display FK must contain one finite XYZ per exported q knot")
    if not np.isclose(metadata["duration_seconds"], times[-1], atol=1e-10, rtol=0):
        raise ValueError("Curve duration differs from its saved metadata")
    stop = float(metadata["executable_until_seconds"])
    if not np.isfinite(stop) or not 0 <= stop <= times[-1]:
        raise ValueError("Saved evidence barrier must lie within the exported curve")
    return {**curve, "names": names, "metadata": metadata, "path": path.resolve(),
            "classification": classification, "trials": []}


def load_trial(path):
    """Read an explicit physical trial without deriving success from its clock."""
    report = read_json(path)
    if report["physics_outcome"] not in PHYSICS:
        raise ValueError("Physics outcome must retain the evaluator's explicit classification")
    expected_success = {"passed": True, "failed": False, "not_evaluated": None, "unverified": None}
    if report["physics_success"] is not expected_success[report["physics_outcome"]]:
        raise ValueError("Physics outcome and explicit success value disagree")
    result = {"path": path, "report": report, "physical": None,
              "execution": None, "evaluation": None, "error": None}
    try:
        for key, target in (("execution_report_path", "execution"),
                            ("evaluation_report_path", "evaluation")):
            # A failed setup declares its intended output paths before those
            # artifacts exist. That is an unexecuted trial, not a read failure.
            if report.get(key) is not None and Path(report[key]).is_file():
                result[target] = read_json(Path(report[key]))
        if (report.get("physical_trajectory_path") is not None
                and Path(report["physical_trajectory_path"]).is_file()):
            with np.load(report["physical_trajectory_path"], allow_pickle=False) as saved:
                physical = {name: saved[name].copy() for name in saved.files}
            elapsed, poses = physical["elapsed"], physical["actual_tool_pose"]
            names, actual_q = physical["joint_names"].tolist(), physical["actual_q"]
            actual_dq = physical["actual_dq"]
            if (elapsed.ndim != 1 or not 1 <= len(elapsed) <= 100001
                    or np.any(np.diff(elapsed) <= 0) or poses.shape != (len(elapsed), 4, 4)
                    or actual_q.shape != (len(elapsed), len(names))
                    or actual_dq.shape != actual_q.shape
                    or len(set(names)) != len(names)
                    or not all(np.isfinite(value).all() for value in (elapsed, poses, actual_q, actual_dq))
                    or not np.allclose(poses[:, 3], [0, 0, 0, 1], atol=1e-8, rtol=0)):
                raise ValueError("Physical samples must retain aligned time, named q and world TCP")
            result["physical"] = physical
    except (OSError, ValueError, KeyError) as error:
        result["error"] = f"{type(error).__name__}: {error}"
    return result


def load_runs(run_dirs, trial_dirs, manifest):
    """Retain failed runs and associate trials only through trajectory_path."""
    candidate_ids = {segment["id"]: {candidate["id"] for candidate in segment["grounding"]["candidates"]}
                     for segment in manifest["segments"]}
    runs, curves, trial_paths = [], {}, set()
    for directory in run_dirs:
        directory = directory.resolve()
        report = read_json(directory / "run-report.json")
        if report["segment_id"] not in candidate_ids:
            raise ValueError(f"Run segment is absent from this manifest: {directory}")
        if report["candidate_id"] not in candidate_ids[report["segment_id"]]:
            raise ValueError(f"Run candidate is absent from its declared manifest segment: {directory}")
        planning_path = directory / "planning" / "planning-report.json"
        planning = read_json(planning_path) if planning_path.is_file() else None
        run = {"directory": directory, "report": report, "planning": planning,
               "curves": [], "errors": []}
        for record in planning.get("candidates", []) if planning else []:
            path = directory / "planning" / record["path"]
            if not record["curve_saved"]:
                run["errors"].append(f"{path.name}: {record.get('diagnostic_error', '未保存曲线')}")
                continue
            try:
                curve = load_curve(path, record["classification"])
                if curve["metadata"]["validation_status"] != record["validation_status"]:
                    raise ValueError("Curve metadata and candidate report disagree on verification completion")
                curve["attempt"] = record["attempt"]
                curve["display_error"] = record.get("diagnostic_error")
                run["curves"].append(curve)
                curves[curve["path"]] = curve
            except (OSError, ValueError, KeyError) as error:
                run["errors"].append(f"{path.name}: {type(error).__name__}: {error}")
        # The returned accepted trajectory is a distinct public artifact. It
        # has no fabricated FK when only the selected diagnostic copy saved FK.
        returned = directory / "planning" / "trajectory.npz"
        if returned.is_file():
            try:
                curve = load_curve(returned, "planner_accepted")
                curve["attempt"] = {"artifact_role": "PlanningResult.trajectory"}
                run["curves"].append(curve)
                curves[curve["path"]] = curve
            except (OSError, ValueError, KeyError) as error:
                run["errors"].append(f"trajectory.npz: {type(error).__name__}: {error}")
        trial_paths.update(directory.rglob("simulation-report.json"))
        runs.append(run)
    for directory in trial_dirs:
        trial_paths.update(directory.resolve().rglob("simulation-report.json"))
    unmatched = []
    for path in sorted(trial_paths):
        trial = load_trial(path)
        source = Path(trial["report"]["trajectory_path"]).resolve()
        if source not in curves:
            unmatched.append(trial)
            continue
        curve = curves[source]
        trial_classification = trial["report"]["trajectory_classification"]
        # The simulation runner records None when setup failed before loading
        # the intended curve. Preserve that untested state without assigning a
        # planner classification to an executor that never read it.
        if trial_classification is not None and trial_classification != curve["classification"]:
            raise ValueError(f"Physics trial relabelled the planner result: {path}")
        if trial_classification is None and (trial["report"]["execution_tested"]
                or trial["report"]["physics_outcome"] != "not_evaluated"):
            raise ValueError(f"An executed or graded trial must retain its planner classification: {path}")
        curve["trials"].append(trial)
    return runs, unmatched


def planning_label(run):
    report, planning = run["report"], run["planning"]
    if report["status"] == "interrupted" or planning and planning["status"] == "interrupted":
        return "运行已中断"
    if report["status"] == "error" or planning and planning["status"] == "error":
        if run["curves"] and all(curve["metadata"]["validation_status"] == "not_completed"
                                 for curve in run["curves"]):
            return "运行报错 · 验证未完成"
        return "运行报错"
    if planning and planning["status"] == "completed":
        return "规划接受" if planning["success"] else "规划未通过"
    return "规划未调用" if not report["planning_called"] else "规划尚无最终结果"


def physics_label(run):
    trials = [trial for curve in run["curves"] for trial in curve["trials"]]
    # BEGIN current physical trials consume formal results, never label gates.
    labels = [PHYSICS[trial["report"]["physics_outcome"]] for trial in trials]
    # END rejected numerical candidates have no production execution mode.
    return "；".join(dict.fromkeys(labels)) if trials else "未执行物理试验"


def reference_for(segment_id, candidate_id, audit, workspace):
    """Look up an exact historical candidate; never choose a nearby candidate."""
    # BEGIN a missing independent input audit is an explicitly absent reference.
    if audit is None:
        return None
    # END current candidate evidence never borrows a different run's reference.
    for segment in audit["segments"]:
        if segment["id"] == segment_id:
            for candidate in segment.get("grounding_recompute", []):
                if candidate["candidate_id"] == candidate_id and candidate.get("reference_artifact"):
                    return load_reference(input_path(candidate["reference_artifact"], workspace)), candidate
    return None


def eef_plot(curve, reference_frame, historical):
    """Distinguish saved q-knot FK, world physical feedback, and old reference."""
    figure, displayed = go.Figure(), []
    label, color = CLASSIFICATIONS[curve["classification"]]
    if "eef_position" in curve:
        xyz = curve["eef_position"]
        displayed.append(xyz)
        figure.add_trace(go.Scatter3d(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
            mode="lines+markers", name=f"当前 q 节点 FK · {label}",
            line={"color": color, "width": 5}, marker={"size": 2},
            customdata=curve["times"], hovertemplate="规划时间 %{customdata:.3f} s<br>"
                "x=%{x:.4f}, y=%{y:.4f}, z=%{z:.4f} m<extra>导出 q 节点 FK</extra>"))
    for index, trial in enumerate(curve["trials"]):
        if trial["physical"] is not None and reference_frame == "world":
            physical = trial["physical"]
            xyz = physical["actual_tool_pose"][:, :3, 3]
            # Subsample display only. Full physics samples remain in the NPZ.
            indices = np.unique(np.linspace(0, len(xyz)-1, min(len(xyz), 2500)).astype(int))
            displayed.append(xyz)
            figure.add_trace(go.Scatter3d(x=xyz[indices, 0], y=xyz[indices, 1], z=xyz[indices, 2],
                mode="lines", name=f"真实物理 TCP · 试验 {index+1}",
                line={"color": "#853eab", "width": 5}, customdata=physical["elapsed"][indices],
                hovertemplate="物理 elapsed %{customdata:.3f} s<br>"
                    "x=%{x:.4f}, y=%{y:.4f}, z=%{z:.4f} m<extra>MuJoCo world TCP</extra>"))
    if not displayed:
        return "<p>当前产物尚无可展示的 EEF FK 或真实 TCP。</p>"
    if historical is not None and historical[1]["reference_frame"] == reference_frame:
        data = historical[0]
        xyz = data["xyz"]
        figure.add_trace(go.Scatter3d(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
            mode="lines", name="旧输入参考（点击查看）", visible="legendonly",
            line={"color": "#8b949e", "width": 3, "dash": "dash"},
            customdata=np.c_[data["frames"], data["times"]],
            hovertemplate="历史源帧 %{customdata[0]:.0f}<br>历史时间 %{customdata[1]:.3f} s"
                "<extra>旧输入参考</extra>"))
        displayed.append(xyz)
    limits = equal_axis_ranges(np.concatenate(displayed))
    figure.update_layout(title=f"EEF / TCP · {escape(reference_frame)} · 等比例米制坐标",
        height=440, margin={"l": 0, "r": 0, "t": 45, "b": 0},
        scene={axis+"axis": {"title": axis+" (m)", "range": limits[index], "nticks": 5}
               for index, axis in enumerate("xyz")}, legend={"orientation": "h", "y": -.05})
    figure.update_scenes(aspectmode="cube")
    return plot_html(figure)


def joint_plot(curve):
    """Sample the original Hermite cubic; physical traces retain their own time."""
    times = np.unique(np.r_[curve["times"], np.linspace(0, curve["times"][-1], 1200)])
    # BEGIN show both q and its exact cubic derivative against physical feedback.
    spline = CubicHermiteSpline(curve["times"], curve["position"], curve["velocity"], axis=0)
    samples = (spline(times), spline(times, 1))
    figure = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=.10,
                           subplot_titles=("关节位置 q", "关节速度 dq"))
    for column, name in enumerate(curve["names"]):
        color = qualitative.Plotly[column % len(qualitative.Plotly)]
        for derivative, label in enumerate(("q", "dq")):
            figure.add_trace(go.Scatter(x=times, y=samples[derivative][:, column], mode="lines",
                name=f"规划 · {name}", legendgroup=name, line={"color": color}, showlegend=derivative == 0,
                hovertemplate="曲线时间 %{x:.4f} s<br>"+label+"=%{y:.6f}<extra>"+escape(name)+"</extra>"),
                row=derivative+1, col=1)
            for index, trial in enumerate(curve["trials"]):
                physical = trial["physical"]
                if physical is None or name not in physical["joint_names"].tolist():
                    continue
                actual_column = physical["joint_names"].tolist().index(name)
                selection = np.unique(np.linspace(0, len(physical["elapsed"])-1,
                                                   min(len(physical["elapsed"]), 2500)).astype(int))
                figure.add_trace(go.Scatter(x=physical["elapsed"][selection],
                    y=physical["actual_"+label][selection, actual_column], mode="lines",
                    name=f"真实 · {name} · 试验 {index+1}", legendgroup=name, showlegend=derivative == 0,
                    line={"dash": "dot", "color": color},
                    hovertemplate="物理 elapsed %{x:.4f} s<br>"+label+"=%{y:.6f}<extra>"+escape(name)+"</extra>"),
                    row=derivative+1, col=1)
    figure.add_vline(x=curve["metadata"]["executable_until_seconds"], line_dash="dash",
                     line_color="#b33e36", annotation_text="原证据屏障 / 可执行终点", row=1, col=1)
    figure.add_vline(x=curve["metadata"]["executable_until_seconds"], line_dash="dash",
                     line_color="#b33e36", row=2, col=1)
    figure.update_layout(title="原导出 q(t) / dq(t) 与真实反馈", height=740,
        legend={"orientation": "h", "y": -.25}, margin={"t": 65, "b": 130})
    figure.update_yaxes(title_text="q（原生关节单位）", row=1, col=1)
    figure.update_yaxes(title_text="dq（原生关节单位 / s）", row=2, col=1)
    figure.update_xaxes(title_text="相对试验入口的时间 (s)", row=2, col=1)
    # END show both q and dq.
    return plot_html(figure)


def observation_panel(segment, workspace, output, run, run_index):
    """Show the actual input RGB and optical-z depth, with exact source links."""
    cards = []
    for view_index, view in enumerate(segment["snapshot"]["views"]):
        rgb_path, depth_path = (input_path(view[key], workspace) for key in ("rgb_path", "depth_path"))
        with Image.open(rgb_path) as source:
            rgb = ImageOps.exif_transpose(source).convert("RGB")
            rgb.thumbnail((720, 480), Image.Resampling.LANCZOS)
            target = output / "assets" / f"run_{run_index:03d}_view_{view_index:02d}.jpg"
            rgb.save(target, quality=88)
        depth = np.load(depth_path, allow_pickle=False)
        valid = np.isfinite(depth) & (depth > 0)
        if depth.ndim != 2 or not valid.any():
            raise ValueError(f"Observed depth must be a nonempty valid H,W optical-z map: {depth_path}")
        depth_target = target.with_name(target.stem+"_depth.png")
        figure, axis = plt.subplots(figsize=(6, 3.8))
        drawn = axis.imshow(np.ma.masked_where(~valid, depth), cmap="viridis")
        axis.set_axis_off()
        figure.colorbar(drawn, ax=axis, label="optical z (m)", fraction=.04)
        figure.tight_layout()
        figure.savefig(depth_target, dpi=110)
        plt.close(figure)
        mask_report = run["directory"] / "observations" / view["id"] / "robot-mask-report.json"
        mask_detail = details("机器人掩码证据", read_json(mask_report)) if mask_report.is_file() else ""
        cards.append(f'<div class="view"><b>{escape(view["id"])}</b>'
            f'<img src="assets/{target.name}" alt="当前规划输入 RGB">'
            f'<img src="assets/{depth_target.name}" alt="当前规划输入 optical-z 深度">'
            f'<p>{link(rgb_path, output, "原 RGB")} · {link(depth_path, output, "原深度")} · '
            f'{link(input_path(view["calibration_path"], workspace), output, "标定")}</p>{mask_detail}</div>')
    return '<div class="views">'+"".join(cards)+"</div>"


def prepare_video_previews(trial, output, trial_index):
    """Create browser-playable copies with existing FFmpeg; preserve all source frames.

    The source MP4, its fixed-rate clock and separate physical capture times
    remain evidence. Re-encoding changes only the page's playback copy, uses
    two CPU threads, and never invokes the simulator or a hardware encoder.
    """
    previews = []
    for index, video in enumerate(trial["report"].get("videos", [])):
        target = output / "assets" / f"trial_{trial_index:03d}_view_{index:02d}.mp4"
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-threads", "2",
                        "-i", str(video["path"]), "-map", "0:v:0", "-an", "-c:v", "libx264",
                        "-crf", "18", "-preset", "fast", "-pix_fmt", "yuv420p",
                        "-vsync", "0", "-threads", "2", "-filter_threads", "2",
                        "-movflags", "+faststart", str(target)], check=True, timeout=120)
        previews.append(target)
    trial["video_previews"] = previews


def trial_panel(trial, output):
    """Expose evaluator outcome alongside the executor's still-pending barriers."""
    report, execution = trial["report"], trial["execution"]
    body = [f'<div class="trial"><h4>{PHYSICS[report["physics_outcome"]]} · '
            f'{escape(report["execution_input"])}</h4>', link(trial["path"], output, "仿真报告")]
    body.append(f'<p>试验状态：{escape(report["status"])}；'
                f'执行器读入的轨迹分类：{escape(report["trajectory_classification"])}</p>')
    if report.get("error"):
        body.append(f'<p class="error">{escape(report["error"])}</p>')
    for key, label in (("physical_trajectory_path", "真实 q / TCP 轨迹"),
                       ("physics_evidence_path", "独立物理证据")):
        if report.get(key) is not None and Path(report[key]).is_file():
            body.append(" · "+link(report[key], output, label))
    if trial["error"]:
        body.append(f'<p class="error">产物读取问题：{escape(trial["error"])}</p>')
    if execution is not None:
        body.append(f'<p>执行状态：{escape(execution["status"])}；'
                    f'证据通过：{escape(execution["evidence_passed"])}；'
                    f'屏障命令已采样：{escape(execution["barrier_command_reached"])}</p>')
        body.append(details("执行时间、未执行命令与待验证事件", execution))
        if "maximum_arm_tracking_error_rad" in execution:
            body.append(f'<p>物理时长：{execution["elapsed_sim_s"]:.3f} s；'
                        f'最大关节跟踪误差：{execution["maximum_arm_tracking_error_rad"]:.4f} rad。</p>')
    if trial["evaluation"] is not None:
        if "stable_grasp_at_stop" in trial["evaluation"]:
            body.append(f'<p>停止时稳定抓持：<b>{escape(trial["evaluation"]["stable_grasp_at_stop"])}</b>。</p>')
        body.append(details("独立物理判定", trial["evaluation"]))
    for index, video in enumerate(report.get("videos", [])):
        relative = quote(os.path.relpath(trial["video_previews"][index], output), safe="/")
        body.append(f'<figure><video controls preload="metadata" src="{relative}"></video>'
                    f'<figcaption>真实控制试验 · {escape(video["view_id"])} · '
                    f'{escape(video["frame_count"])} 帧 / {escape(video["fps"])} fps</figcaption></figure>')
        body.append(link(video["path"], output, "原始试验视频"))
        if video.get("capture_times_path"):
            body.append(link(video["capture_times_path"], output, "原始仿真取帧时间")+
                        "<p>视频按固定帧率播放；最后一帧保留停止状态。取帧数组两列依次为 elapsed、sim_time，"
                        "精确物理时刻以上方数组为准。</p>")
    body.append(details("仿真源产物与模式", report)+"</div>")
    return "".join(body)


def render(manifest, audit, runs, unmatched, output, *, defer_untested=False, stage_reports=None):
    """Render every manifest segment, keeping independent snapshots separate."""
    workspace = Path(manifest["workspace_root"])
    output.mkdir(parents=True, exist_ok=False)
    (output / "assets").mkdir()
    trials = [trial for run in runs for curve in run["curves"] for trial in curve["trials"]]+unmatched
    for index, trial in enumerate(trials):
        prepare_video_previews(trial, output, index)
    by_segment = {segment["id"]: [] for segment in manifest["segments"]}
    for run in runs:
        by_segment[run["report"]["segment_id"]].append(run)
    # BEGIN bring real curves forward while retaining every failed setup record.
    for segment_runs in by_segment.values():
        segment_runs.sort(key=lambda run: (
            any(curve["metadata"]["validation_status"] == "completed" for curve in run["curves"]),
            any(curve["trials"] for curve in run["curves"]), bool(run["curves"])), reverse=True)
        for run in segment_runs:
            run["curves"].sort(key=lambda curve: (bool(curve["trials"]),
                                curve["metadata"]["validation_status"] == "completed"), reverse=True)
    untested = "未测 · 暂缓" if defer_untested else "未测"
    # END current-evidence ordering and explicit user-requested deferral.
    body = ["<h1>当前长程任务实测</h1><p>各段使用独立入口快照；此页记录分段测试，连续任务执行尚未验证。"
            "规划结果与物理判定分别列示。折线中的 FK 仅来自导出 q 节点，真实 TCP 来自物理步反馈。"
            "旧输入参考默认隐藏，保留原始源帧与时间。</p>",
            '<table><thead><tr><th>段 / 独立入口</th><th>当前规划</th><th>真实物理试验</th></tr></thead><tbody>']
    overview = []
    for segment in manifest["segments"]:
        selected = by_segment[segment["id"]]
        # BEGIN compact retry summaries; counts refer to runs, not physics trials.
        planned = [f"{label}（{count} 次运行）" for label, count in
                   Counter(planning_label(run) for run in selected).items()] if selected else [untested]
        physical = [f"{label}（{count} 次运行）" for label, count in
                    Counter(physics_label(run) for run in selected).items()] if selected else [untested]
        body.append(f'<tr><td><a href="#{escape(segment["id"])}">{escape(segment["id"])} · '
                    f'{escape(segment["name"])}</a><br>入口帧 {segment["snapshot"]["entry_frame"]}</td>'
                    f'<td>{"<br>".join(map(escape, planned))}</td>'
                    f'<td>{"<br>".join(map(escape, physical))}</td></tr>')
        overview.append([segment["id"]+"\n"+segment["name"], "\n".join(planned), "\n".join(physical)])
        # END compact retry summaries.
    body.append("</tbody></table>")
    # BEGIN add the requested planning diagnostics to only explicitly bound runs.
    stage_reports = {} if stage_reports is None else stage_reports
    planning_diagnostics = []
    # END other physical-trial pages keep their own established evidence scope.
    run_index, errors = 0, []
    for segment in manifest["segments"]:
        body.append(f'<section id="{escape(segment["id"])}"><h2>{escape(segment["id"])} · '
                    f'{escape(segment["name"])}</h2><p>独立入口帧：{segment["snapshot"]["entry_frame"]}；'
                    f'源程序帧范围：{escape(segment["source_frame_range"])}。</p>')
        if not by_segment[segment["id"]]:
            body.append(f"<p>{untested}；无当前规划曲线或物理执行结果。</p></section>")
            continue
        for run in by_segment[segment["id"]]:
            run_index += 1
            report = run["report"]
            incomplete_history = (any(curve["metadata"]["validation_status"] == "completed"
                for other in by_segment[segment["id"]] for curve in other["curves"])
                and bool(run["curves"]) and all(curve["metadata"]["validation_status"] == "not_completed"
                                               for curve in run["curves"]))
            history_label = " · 历史诊断（验证未完成）" if incomplete_history else ""
            body.append(f'<article><h3>{escape(report["candidate_id"])} · '
                        f'{escape(run["directory"].name)}{history_label}</h3><div class="outcomes"><p>规划：'
                        f'<b>{escape(planning_label(run))}</b></p><p>物理：'
                        f'<b>{escape(physics_label(run))}</b></p></div>')
            if report.get("error"):
                body.append(f'<p class="error">{escape(report["error"])}</p>')
            body.append(link(run["directory"] / "run-report.json", output, "当前运行报告"))
            if report.get("feedback_path"):
                body.append(" · "+link(report["feedback_path"], output, "命名机器人反馈"))
            if run["planning"] is not None:
                body.append(" · "+link(run["directory"] / "planning" / "planning-report.json", output, "规划报告"))
            try:
                body.append(observation_panel(segment, workspace, output, run, run_index))
            except (OSError, ValueError, KeyError) as error:
                run["errors"].append(f"RGB-D: {type(error).__name__}: {error}")
            historical = reference_for(segment["id"], report["candidate_id"], audit, workspace)
            # BEGIN retained native metrics and partial source evidence need no rerun.
            if run["directory"] in stage_reports:
                from recorded_planning_failure_visuals import render_failure_diagnostics
                panel, saved = render_failure_diagnostics(run, segment, manifest, output,
                    run_index, stage_path=stage_reports[run["directory"]])
                body.append(panel)
                planning_diagnostics.extend(saved)
            if audit is None:
                body.append("<p>独立的旧输入参考审计未提供；只显示本次保存产物。</p>")
            # END display failures remain visible rather than generating replacement evidence.
            if not run["curves"]:
                body.append("<p>本次运行未生成可展示的关节曲线。</p>")
            for curve in run["curves"]:
                label, _ = CLASSIFICATIONS[curve["classification"]]
                validation = ("最终验证未完成；仅保留原生曲线，没有最终约束判定"
                              if curve["metadata"]["validation_status"] == "not_completed"
                              else "最终验证已完成；是否通过以规划分类为准")
                body.append(f'<h4>{escape(curve["path"].name)} · {label}</h4>'
                            f'<p><b>{validation}</b></p>'
                            +link(curve["path"], output, "原始 q / dq / 时间数组")
                            +" · "+link(curve["path"].with_suffix(".json"), output, "事件与屏障")
                            +eef_plot(curve, segment["snapshot"]["reference_frame"], historical)
                            +joint_plot(curve)+details("当前候选检查", curve["attempt"])
                            +details("原导出事件与条件", curve["metadata"]))
                if curve.get("display_error"):
                    body.append(f'<p class="error">显示数据记录问题：{escape(curve["display_error"])}</p>')
                body.extend(trial_panel(trial, output) for trial in curve["trials"])
            body.append(details("完整运行记录", report))
            if run["planning"] is not None:
                body.append(details("完整规划记录", run["planning"]))
            if run["errors"]:
                body.append(details("可视化产物读取问题", run["errors"]))
                errors.extend({"run": str(run["directory"]), "error": error} for error in run["errors"])
            body.append("</article>")
        body.append("</section>")
    if unmatched:
        body.append("<section><h2>尚未关联到本页关节曲线的物理试验</h2>")
        body.extend(trial_panel(trial, output) for trial in unmatched)
        body.append("</section>")
    style = """body{margin:0 auto;padding:26px;max-width:1450px;color:#243447;background:#f5f7fa;
font:16px/1.5 system-ui,sans-serif}h1,h2,h3,h4{line-height:1.25}section,article{background:white;
padding:20px;margin:20px 0;border:1px solid #dce2ea;border-radius:8px}article{background:#fbfcfe}
table{border-collapse:collapse;width:100%;background:white}th,td{padding:12px;border:1px solid #dce2ea;
text-align:left}th{background:#e9eef5}.views,.outcomes{display:flex;gap:20px;flex-wrap:wrap}.view{flex:1;
min-width:280px}.view img{width:49%;height:auto;vertical-align:top}pre{white-space:pre-wrap;
overflow-wrap:anywhere;max-height:500px;overflow:auto;font-size:12px}details{margin:12px 0}
.error{color:#a63131}video{width:100%;max-width:960px}a{color:#2466a3}.trial{border-top:2px solid #bcc8d8}
@media(max-width:700px){body{padding:10px}section,article{padding:10px}.view img{width:100%}}
"""
    (output / "index.html").write_text("<!doctype html><html lang=\"zh-CN\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"><title>当前长程任务实测</title>"
        f"<style>{style}</style><script>{get_plotlyjs()}</script></head><body>"+"".join(body)+"</body></html>", encoding="utf-8")
    # BEGIN fit every grouped retry status without clipping the static overview.
    row_lines = [1]+[max(value.count("\n")+1 for value in row) for row in overview]
    figure, axis = plt.subplots(figsize=(14, 1.6+sum(row_lines)*.35))
    axis.set_axis_off()
    table = axis.table(cellText=overview, colLabels=["段", "当前规划", "真实物理试验"],
                       colWidths=[.38, .32, .3], cellLoc="left", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    for (row, _), cell in table.get_celld().items():
        cell.set_height(.9*row_lines[row]/sum(row_lines))
    # END fit every grouped retry status.
    axis.set_title("当前长程任务实测 · 独立入口分段测试\n规划接受与物理判定分列；连续执行尚未验证", pad=24)
    figure.tight_layout()
    figure.savefig(output / "overview.png", dpi=140, bbox_inches="tight")
    plt.close(figure)
    return {"segment_count": len(overview), "run_count": len(runs),
            "segments_tested": sum(bool(value) for value in by_segment.values()),
            "curve_count": sum(len(run["curves"]) for run in runs),
            "physical_trial_count": sum(len(curve["trials"]) for run in runs for curve in run["curves"]),
            "browser_video_count": sum(len(trial["video_previews"]) for trial in trials),
            "unmatched_trial_count": len(unmatched), "render_errors": errors,
            "untested_segments_deferred": defer_untested,
            "continuous_execution_tested": False, "gpu_used": False,
            "browser_interaction_tested": False,
            # BEGIN retain the provenance of each explicitly requested failure view.
            "planning_diagnostics": planning_diagnostics}
            # END query inputs and completed validation remain separate evidence.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    # BEGIN current planning diagnostics do not require an unrelated historical audit.
    parser.add_argument("--audit", type=Path,
                        help="Optional independent historical input-reference audit")
    parser.add_argument("--stage-report", nargs=2, action="append", default=[],
                        metavar=("RUN_DIRECTORY", "STAGES_JSON"),
                        help="Add planning failure/scene diagnostics for this exact run")
    # END the explicit directory binding prevents mixing timings from another run.
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--trial-dir", type=Path, action="append", default=[])
    parser.add_argument("--defer-untested", action="store_true",
                        help="Mark untested segments as explicitly deferred by the user")
    args = parser.parse_args()
    started = time.monotonic()
    configure_font()
    manifest = read_json(args.manifest)
    audit = None if args.audit is None else read_json(args.audit)
    if not manifest["snapshots_are_independent"]:
        raise ValueError("This report requires explicitly independent segment snapshots")
    runs, unmatched = load_runs(args.run_dir, args.trial_dir, manifest)
    # BEGIN refuse an unbound or repeated timing report before creating output.
    stage_reports = {Path(directory).resolve(): Path(path).resolve() for directory, path in args.stage_report}
    if (len(stage_reports) != len(args.stage_report)
            or not set(stage_reports) <= {run["directory"] for run in runs}):
        raise ValueError("Each stage report must name one distinct supplied run directory")
    # END no filename proximity supplies a timing identity.
    report = render(manifest, audit, runs, unmatched, args.output.resolve(),
                    defer_untested=args.defer_untested, stage_reports=stage_reports)
    report.update(manifest_path=str(args.manifest.resolve()),
                  audit_path=None if args.audit is None else str(args.audit.resolve()),
                  run_directories=[str(path.resolve()) for path in args.run_dir],
                  trial_directories=[str(path.resolve()) for path in args.trial_dir],
                  elapsed_seconds=time.monotonic()-started,
                  peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    (args.output / "visualization-report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
# END independent current-run visualization with explicit evidence identities.
