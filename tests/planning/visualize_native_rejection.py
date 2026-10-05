"""Show a matched archived native rejection and its original failure evidence.

The only geometry operation is CPU FK for display. Source witnesses come from
the recorded constraint replay; no collision query or physical step is run.
"""

# BEGIN explicit original-run/replay binding prevents mixing different candidates.
import argparse
import html
import json
from pathlib import Path
import resource
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from recorded_planning_failure_visuals import (
    _embedded_png, _static_scene, configure_font, failure_witnesses, robot_display,
    scene_figure, source_display)
from visualize_current_recorded_task import load_curve


def read(path):
    return json.loads(Path(path).read_text())


def matched_failures(replay, original, native_path, seed_index):
    """Require the explicit archived result and matching constraint maxima."""
    if (replay["status"] != "completed"
            or Path(replay["native_result"]).resolve() != native_path.resolve()):
        raise ValueError("Completed replay must name this original native result")
    original_seed, = [row for row in original["seeds"] if row["seed_index"] == seed_index]
    expected_labels = {name for name in ("rollout", "interpolated") if name in original_seed["metrics"]}
    if set(replay["labels"]) != expected_labels:
        raise ValueError("Replay must cover every archived native constraint scope")
    failures, witnesses, comparisons = [], [], {}
    for label, result in replay["labels"].items():
        seed, = [row for row in result["seeds"] if row["seed_index"] == seed_index]
        expected = original_seed["metrics"][label]["constraints"]
        if seed["constraints"] != expected:
            raise ValueError(f"{label}: replay constraints do not reproduce the archived rejection")
        trace = seed["source_trace"]
        if trace["status"] != "recorded":
            raise ValueError(f"{label}: native input identity has not been matched")
        comparisons[label] = {"original": expected, "replay": seed["constraints"], "equal": True}
        failures.extend({**row, "scope": label} for row in trace["component_failures"])
        for row in trace["component_failures"]:
            if row["component"] != "source_world_collision":
                continue
            for state in row["geometry_witnesses"]:
                # The recorder links each failed mode to its actual unique
                # geometry query. Rendering the worst state alone would lose
                # other failed times or other candidates' distinct evidence.
                for witness in failure_witnesses({"source_world_collision": {"worst_state": {
                        "trajectory_time_s": state["trajectory_time_s"],
                        "scene": {"violations": state["violations"]}}}}):
                    if witness not in witnesses:
                        witnesses.append(witness)
    return failures, witnesses, comparisons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--replay-report", type=Path, required=True)
    parser.add_argument("--candidate-index", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run = read(args.run/"run-report.json")
    planning = read(args.run/"planning/planning-report.json")
    record = planning["candidates"][args.candidate_index]
    directory = args.run/"planning"
    native = directory/record["native_result"]
    replay = read(args.replay_report)
    failures, witnesses, comparisons = matched_failures(replay,
        read(directory/record["native_metrics_path"]), native, record["attempt"]["seed_index"])
    if not failures:
        raise ValueError("No localized source failure is present in this matched replay")
    curve = load_curve(directory/record["path"], record["classification"])
    snapshot = read(directory/record["collision_snapshot"]["path"])
    if snapshot["curve_path"] != curve["path"].name:
        raise ValueError("The scene snapshot does not bind the displayed curve")
    snapshot.pop("scene_query_capture", None)
    partial = read(directory/record["export_validation_path"])
    # Keep each distinct native joint state. Repeated exact/left/right modes
    # remain separate in the evidence table but need one contextual pose.
    unique = {}
    for row in failures:
        key = (row["time_s"], tuple(row["joint_position"]))
        unique.setdefault(key, row)
    states_to_show = sorted(unique.values(), key=lambda row: row["time_s"])
    # A bounded set of displayed poses does not truncate the evidence archive.
    shown = states_to_show[:12]
    times = [row["time_s"] for row in shown]
    meshes, states = robot_display(snapshot, curve, read(run["feedback_path"]), times,
                                  joint_positions=[row["joint_position"] for row in shown])
    clouds, counts = source_display(snapshot)
    args.output.mkdir(parents=True, exist_ok=False)
    configure_font()
    figure = plt.figure(figsize=(17, 11), layout="constrained")
    grid = figure.add_gridspec(3, 2, height_ratios=[.6, 1.7, .55])
    axis = figure.add_subplot(grid[0, :]); axis.axis("off")
    messages = ["当前候选拒绝原因已定位 · 原生约束回放 · 未执行物理仿真",
        "原始与回放的全部约束最大值一致；输入为已保存的原生 q/dq。",
        f"失败物理状态 {len(unique)} 个；含模式记录 {len(failures)} 条；空间见证 {len(witnesses)} 对。"]
    for row in shown[:4]:
        messages.append(f"{row['scope']} | t={row['time_s']:.9f} s | {row['component']}={row['raw_residual']:g}"
                        f" | {row['sample_kind']}[{row['sample_index']}]")
    if witnesses:
        for witness in witnesses[:3]:
            messages.append(" ↔ ".join(witness["geometry_names"])+
                            f" | signed distance={witness['signed_distance_m']*1000:.6f} mm")
            evidence = witness["original_evidence"]
            if witness["reason"] == "independent_scene_measurement_margin":
                messages.append(f"几何仍分离；要求的测量留量 {evidence['measurement_allowance_m']*1000:.6f} mm，"
                                f"不足 {-evidence['measurement_adjusted_distance_m']*1000:.6f} mm。")
    axis.text(0, 1, "\n".join(messages), va="top", fontsize=12, linespacing=1.5)
    axis = figure.add_subplot(grid[1, 0], projection="3d")
    _static_scene(axis, curve, meshes, states[0], clouds)
    axis.set_title(f"失败状态 t={times[0]:.9f} s · 使用原生关节输入的 CPU 显示")
    if witnesses:
        points = np.asarray(witnesses[0]["points_world"])
        axis.scatter(*points[0], color="#b63e41", s=110, marker="x", depthshade=False)
        axis.text(*points[0], "  拒绝位置", color="#b63e41", fontsize=11)
    axis = figure.add_subplot(grid[1, 1])
    if witnesses and witnesses[0]["reason"] == "independent_scene_measurement_margin":
        witness = witnesses[0]
        evidence = witness["original_evidence"]
        distance, allowance = witness["signed_distance_m"]*1000, evidence["measurement_allowance_m"]*1000
        axis.barh([1., 0.], [distance, allowance], color=["#477d9d", "#92a89f"], height=.32)
        axis.barh([1.], [allowance-distance], left=distance, color="#f0c1c0", height=.32,
                  hatch="///", label="留量不足")
        axis.axvline(allowance, color="#b63e41", linestyle="--")
        axis.text(distance/2, 1., f"{distance:.6f} mm", ha="center", va="center", color="white")
        axis.text(allowance/2, 0., f"{allowance:.6f} mm", ha="center", va="center", color="white")
        axis.text((distance+allowance)/2, 1.3, f"差 {allowance-distance:.6f} mm", ha="center", color="#b63e41")
        axis.set(xlim=(0, allowance*1.18), ylim=(-.8, 1.8), xlabel="距离 / mm", yticks=[0., 1.],
                 yticklabels=["要求的测量留量", "模型中的实际间距"])
        axis.set_title("拒绝的是测量留量不足\n原始模型距离为正，没有实体穿透", fontsize=13)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(loc="lower right")
        axis.text(.02, .02, "距离来自原始查询点对；不是放宽容差后的结果。",
                  transform=axis.transAxes, fontsize=9)
    else:
        axis.axis("off")
        axis.text(0, 1, "原始失败证据\n"+json.dumps(witnesses, ensure_ascii=False, indent=2)[:2200],
                  va="top", fontsize=9, wrap=True)
    axis = figure.add_subplot(grid[2, :])
    duration = float(curve["times"][-1]); end = partial["checked_time_interval_s"][-1]
    axis.broken_barh([(0, duration)], (1.6, .5), facecolors="#647e97")
    axis.broken_barh([(0, end)], (.6, .5), facecolors="#477d9d", hatch="///")
    axis.broken_barh([(end, duration-end)], (.6, .5), facecolors="#c4c9ce")
    axis.scatter([row["time_s"] for row in states_to_show], [1.85]*len(states_to_show),
                 color="#b63e41", marker="x", s=80, zorder=3)
    for time_s in sorted(set(times))[:6]:
        axis.axvline(time_s, color="#b63e41", alpha=.65)
    axis.set(xlim=(0, duration), ylim=(0, 2.5), xlabel="轨迹时间 / s",
             yticks=[.85, 1.85], yticklabels=[f"此前独立复查：{partial['checked_points']}/{partial['planned_points']} 点",
                                           "原生离散检查：红叉为失败状态"])
    axis.set_title("覆盖条只表示离散采样范围；灰色为独立复查未完成部分")
    figure.savefig(args.output/"failure-summary.png", dpi=145); plt.close(figure)
    # All plotted witness points stay fixed at their recorded positions while
    # the selector changes only robot display poses.
    plot = scene_figure(curve, meshes, states, clouds, {}, witnesses)
    plot.update_layout(title=f"失败原生状态 t={times[0]:.9f} s · 未执行")
    rows = []
    for row in failures:
        rows.append("<tr>"+"".join("<td>"+html.escape(str(row[key]))+"</td>" for key in
            ("scope", "time_s", "progress", "mode", "sample_kind", "sample_index", "component", "raw_residual"))+"</tr>")
    body = ["<!doctype html><html lang='zh-CN'><meta charset='utf-8'><title>原生拒绝原因定位</title>",
        "<style>body{font-family:system-ui;max-width:1500px;margin:32px auto;padding:0 22px;color:#253347}"
        "table{border-collapse:collapse;width:100%}td,th{border:1px solid #ddd;padding:8px;text-align:left}"
        "pre{white-space:pre-wrap;word-break:break-word}img{max-width:100%}</style>",
        "<h1>原生拒绝原因与当前候选现象</h1><p>回放输入来自原始候选，全部命名约束最大值复现。"
        "以下是规划诊断，不是实际执行或完整曲线验收通过。</p>",
        _embedded_png(args.output/"failure-summary.png"),
        "<h2>具体失败记录</h2><table><tr>"+"".join(f"<th>{s}</th>" for s in
        ("检查", "时间 / s", "技能进度", "模式", "点类型", "点索引", "场景子项", "未加权残差"))+"</tr>",
        *rows, "</table><p>exact/left/right 是同一物理时刻的场景规则模式；模式重复不代表额外物理碰撞。"
        "anchor 是必须单独检查的事件位置，不能把折叠后的最后一列当作其物理时刻。</p>",
        "<p>source_world_collision 是来源实体与观测场景的检查结果，包含几何分离、测量留量和必需查询等要求。"
        "具体拒绝类型来自记录中的 violations/query_gaps。</p>",
        plot.to_html(full_html=False, include_plotlyjs=True, config={"displaylogo": False}),
        "<p>机器人显示使用失败记录的原生关节状态，经 CPU FK 绘制；红色点对是原始查询见证。"
        "灰/青点为冻结入口场景零值表面抽样，不能当作完整占据体积。未建模的未来手指实体隐藏。"
        "独立复查仍只完成原报告中的采样范围，物理执行未启动。</p>",
        "<details><summary>原始失败见证</summary><pre>"+
        html.escape(json.dumps(witnesses, ensure_ascii=False, indent=2))+"</pre></details>",
        "<details><summary>全部原始约束与场景失败记录</summary><pre>"+
        html.escape(json.dumps(failures, ensure_ascii=False, indent=2))+"</pre></details></html>"]
    (args.output/"index.html").write_text("\n".join(body))
    evidence = {"classification": "diagnostic_not_executable", "original_run": str(args.run.resolve()),
        "replay_report": str(args.replay_report.resolve()), "constraints_comparison": comparisons,
        "component_failures": failures, "original_failure_witnesses": witnesses,
        "unique_failure_states": len(unique), "displayed_native_states": shown,
        "previous_independent_validation": {key: partial[key] for key in
            ("checked_points", "planned_points", "checked_time_interval_s", "validation_complete")},
        "scene_display_counts": counts, "gpu_used": False, "physics_steps": 0,
        "geometry_queries": 0, "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024}
    (args.output/"evidence.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps({"output": str(args.output), "unique_failure_states": len(unique),
                      "witnesses": len(witnesses), "peak_rss_mib": evidence["peak_rss_mib"]}))


if __name__ == "__main__":
    main()
# END rendering cannot change either the original result or the replay record.
