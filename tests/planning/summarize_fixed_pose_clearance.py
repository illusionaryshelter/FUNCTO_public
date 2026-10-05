"""Export measured fixed-pose IK results as a table and a standalone figure."""

# BEGIN render archived diagnostic values without new geometry or acceptance.
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.report.read_text())
    if data["status"] != "completed":
        raise ValueError("A distribution summary requires a completed diagnostic")
    margins = sorted({pair["measurement_allowance_m"]*1000 for target in data["targets"]
        for row in [target["archived_q_at_this_target"], *target["solutions"]]
        for pair in row["source_scene"]["pairs"] if "measurement_allowance_m" in pair})
    args.output.mkdir(parents=True, exist_ok=False)
    fig, axes = plt.subplots(2, len(data["targets"]), figsize=(12, 8), squeeze=False,
                             gridspec_kw={"height_ratios": [1.3, 1]})
    rows, summary = [], {"source_report": str(args.report.resolve()), "targets": []}
    for column, target in enumerate(data["targets"]):
        selected = []
        for index, row in enumerate(target["solutions"]):
            minimum = row["source_environment_distances"]["minimum_recorded_unpermitted_pairs"]
            record = {"target": target["label"], "time_s": target["time_s"], "return_row": index,
                "batch": row["batch"], "return_rank": row["return_rank"],
                "native_ik_success": row["native_ik_success"],
                "fixed_pose_passed": row["static_gates"]["fixed_target_pose"],
                "modeled_static_passed": row["modeled_static_gates_passed"],
                "full_coverage_certified": row["full_robot_coverage_certified"],
                "minimum_recorded_unpermitted_distance_mm": None if minimum is None else minimum["distance_m"]*1000,
                "position_error_m": row["fixed_position_error_m"],
                "rotation_error_rad": row["fixed_rotation_error_rad"],
                "joint_limit_margin_rad": row["joint_limit_margin_rad"],
                "q0_distance_rad": row["distance_to_measured_q0_rad"],
                "archived_q_distance_rad": row["distance_to_archived_q_rad"]}
            rows.append(record)
            if record["fixed_pose_passed"] and minimum is not None:
                selected.append(record)
        selected.sort(key=lambda row: row["minimum_recorded_unpermitted_distance_mm"])
        passing = [row for row in selected if row["modeled_static_passed"]]
        summary["targets"].append({"label": target["label"], "time_s": target["time_s"],
            "returned": len(target["solutions"]), "fixed_pose_passed": len(selected),
            "modeled_static_passed": len(passing),
            "passing_distance_range_mm": None if not passing else [min(row["minimum_recorded_unpermitted_distance_mm"] for row in passing),
                max(row["minimum_recorded_unpermitted_distance_mm"] for row in passing)],
            "full_coverage_certified": sum(row["full_robot_coverage_certified"] for row in target["solutions"])})
        colors = ["#228168" if row["modeled_static_passed"] else "#b94747" for row in selected]
        values = [row["minimum_recorded_unpermitted_distance_mm"] for row in selected]
        axis = axes[0, column]
        axis.bar(np.arange(len(selected)), values, color=colors, width=.68)
        for margin in margins:
            axis.axhline(margin, color="#333333", linestyle="--", linewidth=1,
                         label=f"Recorded measurement margin: {margin:g} mm")
        axis.set_xticks(np.arange(len(selected)), [f"{row['batch']}:{row['return_rank']}" for row in selected])
        axis.set_xlabel("IK batch : return rank (sorted by measured distance)")
        axis.set_ylabel("Minimum recorded unpermitted distance [mm]")
        axis.set_title(f"{target['label']} | t = {target['time_s']:.3f} s\n"
                       f"{len(selected)}/{len(target['solutions'])} match fixed pose; {len(passing)} pass modeled static checks")
        axis.margins(y=.2)
        axis.grid(axis="y", alpha=.2)
        axis.legend(loc="upper left", fontsize=8)
        axis = axes[1, column]
        axis.scatter([row["q0_distance_rad"] for row in selected], values, c=colors, s=50)
        for row in selected:
            axis.annotate(f"{row['batch']}:{row['return_rank']}",
                (row["q0_distance_rad"], row["minimum_recorded_unpermitted_distance_mm"]),
                xytext=(5, 4), textcoords="offset points", fontsize=8)
        for margin in margins:
            axis.axhline(margin, color="#333333", linestyle="--", linewidth=1)
        axis.margins(x=.12, y=.2)
        axis.set_xlabel("Joint-space distance from measured entrance q0 [rad]")
        axis.set_ylabel("Recorded distance [mm]")
        axis.grid(alpha=.2)
    fig.suptitle("Fixed tool pose, different arm configurations", fontsize=17)
    fig.text(.5, .015, "Green: modeled static gates pass. Red: a modeled static gate fails.\n"
        "Same initializers per target. Future finger geometry remains unmodelled; no path or execution is certified.",
        ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .055, 1, .95))
    fig.savefig(args.output/"fixed-pose-clearance.png", dpi=180)
    fig.savefig(args.output/"fixed-pose-clearance.pdf")
    plt.close(fig)
    with (args.output/"solutions.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
# END figures retain the scope and classifications of the saved evidence.
