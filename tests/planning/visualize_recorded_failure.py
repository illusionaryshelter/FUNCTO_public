#!/usr/bin/env python3
"""Diagnose frozen recorded joint curves and physical trials on the CPU.

Each explicit simulation report identifies one exact saved native curve.
The original manifest/audit supplies its frozen ground_program reference and
source phase partition; the current planner/builder is never rerun. Saved
progress_u/progress_s connects the planning clock to source phase progress.
Physical progress remains the commanded progress recorded by the executor,
not an estimate of how far the real TCP actually moved.

Existing SciPy solves the exported Hermite polynomial and its extrema/limit
crossings. Existing yourdfpy evaluates that unchanged q at extra display
samples, using the recorded base/TCP calibration and serialized fixture.
No GPU, collision query, control command, simulation, IK, smoothing, or
angle wrapping is performed. Original references and execution artifacts
remain read-only. The output directory must not already exist.
"""

# BEGIN independent CPU failure analysis with explicit clocks and native units.
from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
import time

# The established renderer sets two CPU threads before numerical imports.
from visualize_current_recorded_task import (
    CLASSIFICATIONS, configure_font, details, equal_axis_ranges, escape,
    input_path, link, load_runs, plot_html, read_json, reference_for,
)
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs
from plotly.subplots import make_subplots
from scipy.integrate import trapezoid
from scipy.interpolate import CubicHermiteSpline, PPoly
from scipy.spatial.transform import Rotation, Slerp
from yourdfpy import URDF
import yaml


REFERENCE_COLOR = "#278751"
PLANNED_COLOR = "#2878b5"
ACTUAL_COLOR = "#9042a3"
LIMIT_COLOR = "#bf3e3e"


def finite_pose(value, name):
    """Require a calibrated rigid transform instead of guessing a frame."""
    pose = np.asarray(value, float)
    if (pose.shape != (4, 4) or not np.isfinite(pose).all()
            or not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-8, rtol=0)
            or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-8, rtol=0)
            or not np.isclose(np.linalg.det(pose[:3, :3]), 1, atol=1e-8, rtol=0)):
        raise ValueError(f"{name} must be a finite calibrated SE(3) matrix")
    return pose


class RecordedClocks:
    """Keep source, planning, and command-progress coordinates distinct.

    The frozen builder assigned equally spaced progress to source *rows*
    within each audited phase, independently of the source timestamps.
    An inverse progress map can be an interval during a planned hold. Its
    arrival/departure values must not be collapsed into an arbitrary time.
    """

    def __init__(self, curve, reference, phases, phase_count):
        self.duration = float(curve["times"][-1])
        self.u = np.asarray(curve["progress_u"], float)
        self.s = np.asarray(curve["progress_s"], float)
        self.frames, self.source_times = reference["frames"], reference["times"]
        self.reference_progress = np.full(len(self.frames), np.nan)
        self.phases = phases
        if (self.u.ndim != 1 or self.s.shape != self.u.shape or len(self.u) < 2
                or not np.isfinite(self.u).all() or not np.isfinite(self.s).all()
                or np.any(np.diff(self.u) <= 0) or np.any(np.diff(self.s) < 0)
                or not np.array_equal(self.u[[0, -1]], [0, 1])
                or not np.array_equal(self.s[[0, -1]], [0, phase_count])
                or len(phases) != phase_count):
            raise ValueError("Saved phase clocks need complete, monotone, explicitly matching audit provenance")
        for index, phase in enumerate(phases):
            if phase["id"] != index:
                raise ValueError("Frozen audit phases must retain contiguous original IDs")
            begins = np.flatnonzero(self.frames == phase["start_frame"])
            ends = np.flatnonzero(self.frames == phase["end_frame"])
            if len(begins) != 1 or len(ends) != 1 or begins[0] > ends[0]:
                raise ValueError("Each frozen phase boundary must identify an original source row")
            begin, end = int(begins[0]), int(ends[0])
            if begin == end:
                self.reference_progress[end] = index + 1.
            else:
                self.reference_progress[begin:end+1] = index + np.arange(end-begin+1)/(end-begin)
        if (not np.isfinite(self.reference_progress).all()
                or np.any(np.diff(self.reference_progress) <= 0)
                or self.reference_progress[0] != self.s[0]
                or self.reference_progress[-1] != self.s[-1]):
            raise ValueError("This audit must expose every source-connected phase; no missing clock is inferred")
        self.levels, first, counts = np.unique(self.s, return_index=True, return_counts=True)
        self.arrival_u, self.departure_u = self.u[first], self.u[first+counts-1]
        for event in curve["metadata"]["events"]:
            rows = np.flatnonzero(self.frames == event["source_frame"])
            if len(rows) != 1 or not np.isclose(
                    self.reference_progress[rows[0]], event["progress"], atol=1e-12, rtol=0):
                raise ValueError("Original event frame and saved phase progress disagree")
            if not np.isclose(self.progress(event["time"]), event["progress"], atol=1e-12, rtol=0):
                raise ValueError("Original event time and saved planning clock disagree")

    def progress(self, planned_time):
        values = np.asarray(planned_time, float)
        if np.any(values < 0) or np.any(values > self.duration):
            raise ValueError("A planning-clock query cannot leave the saved curve")
        return np.interp(values/self.duration, self.u, self.s)

    def source_time(self, progress):
        values = np.asarray(progress, float)
        if np.any(values < self.s[0]) or np.any(values > self.s[-1]):
            raise ValueError("A source-clock query cannot leave the explicit phase map")
        return np.interp(values, self.reference_progress, self.source_times)

    def planning_interval(self, progress):
        """Invert each monotone piece using departure -> next arrival.

        Interpolating only the first duplicate s would incorrectly include
        a preceding hold in moving time. Exact plateau levels retain both
        endpoints, while interior levels have a unique inverse.
        """
        values = np.atleast_1d(np.asarray(progress, float))
        if np.any(values < self.s[0]) or np.any(values > self.s[-1]):
            raise ValueError("Inverse progress must remain inside the saved map")
        upper = np.searchsorted(self.levels, values, side="left")
        arrival, departure = np.empty_like(values), np.empty_like(values)
        exact = self.levels[upper] == values
        arrival[exact], departure[exact] = self.arrival_u[upper[exact]], self.departure_u[upper[exact]]
        lower = upper[~exact]-1
        ratio = (values[~exact]-self.levels[lower])/(self.levels[upper[~exact]]-self.levels[lower])
        mapped = self.departure_u[lower]+ratio*(self.arrival_u[upper[~exact]]-self.departure_u[lower])
        arrival[~exact], departure[~exact] = mapped, mapped
        return np.column_stack((arrival, departure))*self.duration


class FixtureFK:
    """Independent mesh-free CPU FK and limits from the actual fixture files."""

    def __init__(self, fixture_path, feedback_path, names):
        self.fixture_path, self.feedback_path = fixture_path, feedback_path
        self.fixture, self.feedback = read_json(fixture_path), read_json(feedback_path)
        config = yaml.safe_load(Path(self.fixture["robot_config_path"]).read_text())["robot_cfg"]["kinematics"]
        self.config = config
        self.robot = URDF.load(self.fixture["urdf_path"], build_scene_graph=True, load_meshes=False,
                               build_collision_scene_graph=False, load_collision_meshes=False)
        self.base = finite_pose(self.feedback["base_pose"], "recorded world-from-base")
        self.tool = finite_pose(self.feedback["tool_transform"], "recorded link-from-TCP")
        self.base_link, self.tool_link = self.fixture["base_link"], self.feedback["tool_link"]
        self.current = dict(zip(self.feedback["joint_names"], self.feedback["start_q"]))
        if self.feedback["reference_frame"] != "world" or self.tool_link != self.fixture["tool_link"]:
            raise ValueError("Physical world TCP must share the exact recorded FK frame and tool link")
        cspace = config["cspace"]
        # These fixture tests used native defaults. Refuse to silently ignore
        # a future scaling/clipping configuration that changes physical rails.
        if any(cspace.get(key, default) != default for key, default in (
                ("velocity_scale", 1.), ("acceleration_scale", 1.), ("position_limit_clip", 0.))):
            raise ValueError("Non-default native limit scaling needs explicit diagnostic support")
        acceleration = np.asarray(cspace["max_acceleration"], float)
        if acceleration.ndim == 0:
            acceleration = np.full(len(cspace["joint_names"]), float(acceleration))
        if acceleration.shape != (len(cspace["joint_names"]),) or not np.isfinite(acceleration).all():
            raise ValueError("Configured acceleration limits must retain their named native layout")
        self.limits = []
        for name in names:
            joint = self.robot.joint_map[name]
            if joint.type not in {"revolute", "prismatic"} or joint.limit is None:
                raise ValueError("This analysis requires explicit bounded scalar joint limits")
            limit = joint.limit
            native = [limit.lower, limit.upper, limit.velocity,
                      acceleration[cspace["joint_names"].index(name)]]
            if not np.isfinite(native).all() or native[0] >= native[1] or min(native[2:]) <= 0:
                raise ValueError("Position, velocity, and acceleration limits must be finite and ordered")
            if not np.isclose(limit.velocity, self.fixture["velocity_limits"][name], atol=0, rtol=0):
                raise ValueError("Serialized URDF and fixture manifest disagree on native velocity")
            self.limits.append({"joint": name, "unit": "rad" if joint.type == "revolute" else "m",
                                "lower": float(native[0]), "upper": float(native[1]),
                                "velocity": float(native[2]), "acceleration": float(native[3])})

    def evaluate(self, positions, names):
        positions = np.asarray(positions, float)
        if positions.ndim != 2 or positions.shape[1] != len(names) or not np.isfinite(positions).all():
            raise ValueError("CPU FK needs finite named q without projection or angle wrapping")
        result = np.empty((len(positions), 4, 4))
        for index, row in enumerate(positions):
            self.robot.update_cfg({**self.current, **dict(zip(names, row))})
            result[index] = self.base @ self.robot.get_transform(
                self.tool_link, frame_from=self.base_link) @ self.tool
        return result


def finite_roots(polynomial, level, start, stop):
    """Return SciPy's real in-domain roots, dropping identical-piece sentinels."""
    roots = np.asarray(polynomial.solve(level, extrapolate=False), float)
    return roots[np.isfinite(roots) & (roots > start) & (roots < stop)]


def polynomial_metrics(polynomial, lower, upper, start, stop):
    """Find exact scalar extrema and all continuous open violation intervals.

    Every polynomial segment endpoint is evaluated from both sides. This
    matters for acceleration: a Hermite curve is C1, while rounded exported
    q/dq can produce an acceleration jump at a saved subdivision. Root solving
    finds within-segment crossings instead of relying on a display grid.
    """
    inside = polynomial.x[(polynomial.x > start) & (polynomial.x < stop)]
    boundaries = np.unique(np.r_[start, inside, stop])
    critical = finite_roots(polynomial.derivative(), 0., start, stop)
    candidate_times, candidate_values, candidate_sides = [], [], []
    for begin, end in zip(boundaries[:-1], boundaries[1:]):
        interval = min(int(np.searchsorted(polynomial.x, (begin+end)/2, side="right")-1),
                       polynomial.c.shape[1]-1)
        times = np.r_[begin, critical[(critical > begin) & (critical < end)], end]
        values = np.polyval(polynomial.c[:, interval], times-polynomial.x[interval])
        candidate_times.extend(times); candidate_values.extend(values)
        candidate_sides.extend(["right", *(["interior"]*(len(times)-2)), "left"])
    times, values = np.asarray(candidate_times), np.asarray(candidate_values)
    minimum, maximum, peak = int(values.argmin()), int(values.argmax()), int(np.abs(values).argmax())
    roots = [finite_roots(polynomial, level, start, stop) for level in (lower, upper)]
    cuts = np.unique(np.concatenate([boundaries, *roots]))
    intervals = []
    for begin, end in zip(cuts[:-1], cuts[1:]):
        value = float(polynomial((begin+end)/2))
        side = "below_lower" if value < lower else "above_upper" if value > upper else None
        if side is None:
            continue
        # Merge through ordinary knots only when the shared point itself is
        # outside. A tangent touch of the limit stays an explicit boundary.
        at_join = float(polynomial(begin))
        if (intervals and intervals[-1]["side"] == side and intervals[-1]["end_s"] == begin
                and (at_join < lower or at_join > upper)):
            intervals[-1]["end_s"] = float(end)
        else:
            intervals.append({"start_s": float(begin), "end_s": float(end), "side": side})
    return {"interval_s": [float(start), float(stop)], "minimum": float(values[minimum]),
            "minimum_at_s": float(times[minimum]), "maximum": float(values[maximum]),
            "maximum_at_s": float(times[maximum]), "peak_absolute": float(abs(values[peak])),
            "peak_signed": float(values[peak]), "peak_at_s": float(times[peak]),
            "peak_side": candidate_sides[peak], "limit": [float(lower), float(upper)],
            "maximum_excess": float(max(0., lower-values[minimum], values[maximum]-upper)),
            "first_limit_crossing_s": intervals[0]["start_s"] if intervals else None,
            "outside_duration_s": float(sum(row["end_s"]-row["start_s"] for row in intervals)),
            "outside_open_intervals": intervals}


def observed_metrics(times, values, lower, upper):
    """Describe sampled physical feedback without inventing between-step roots."""
    indices = np.flatnonzero((values < lower) | (values > upper))
    peak = int(np.abs(values).argmax())
    return {"sample_count": len(values), "minimum": float(values.min()), "maximum": float(values.max()),
            "peak_absolute": float(abs(values[peak])), "peak_at_elapsed_s": float(times[peak]),
            "maximum_excess": float(max(0., lower-values.min(), values.max()-upper)),
            "outside_sample_count": len(indices),
            "first_observed_outside_elapsed_s": float(times[indices[0]]) if len(indices) else None,
            "last_observed_outside_elapsed_s": float(times[indices[-1]]) if len(indices) else None}


def pose_errors(actual, expected):
    """Metres and shortest SO(3) radians; joint angles themselves stay unwrapped."""
    position = np.linalg.norm(actual[:, :3, 3]-expected[:, :3, 3], axis=1)
    rotation = (Rotation.from_matrix(expected[:, :3, :3]).inv()
                * Rotation.from_matrix(actual[:, :3, :3])).magnitude()
    return position, rotation


def error_summary(position, rotation, clock, selected):
    if not np.any(selected):
        return None
    p, r, t = position[selected], rotation[selected], clock[selected]
    duration = t[-1]-t[0]
    position_rms = np.sqrt(trapezoid(p*p, t)/duration) if duration > 0 else p[0]
    rotation_rms = np.sqrt(trapezoid(r*r, t)/duration) if duration > 0 else r[0]
    return {"sample_count": len(p), "position_max_m": float(p.max()),
            "position_max_at_clock_s": float(t[p.argmax()]), "position_time_rms_m": float(position_rms),
            "rotation_max_rad": float(r.max()), "rotation_max_at_clock_s": float(t[r.argmax()]),
            "rotation_time_rms_rad": float(rotation_rms)}


def analyze(run, curve, trial, manifest, audit):
    """Derive new diagnostic arrays while preserving all frozen source files."""
    segment = next(row for row in manifest["segments"] if row["id"] == run["report"]["segment_id"])
    historical = reference_for(segment["id"], run["report"]["candidate_id"], audit, Path(manifest["workspace_root"]))
    if historical is None or segment["snapshot"]["reference_frame"] != "world":
        raise ValueError("An exact frozen world-frame ground_program reference is required")
    reference, reference_record = historical
    reference_path = input_path(reference_record["reference_artifact"], Path(manifest["workspace_root"]))
    with np.load(reference_path, allow_pickle=False) as saved:
        reference_pose = np.asarray(saved["eef_poses"], float)
        if str(saved["reference_frame"].item()) != "world":
            raise ValueError("Reference poses cannot be relabelled into the physical world frame")
    clock = RecordedClocks(curve, reference, reference_record["functional_phases"], run["planning"]["phase_count"])
    physical = trial["physical"]
    if physical is None or trial["error"] is not None:
        raise ValueError("Failure comparison requires successfully decoded original physical feedback")
    if (physical["curve_time"].shape != physical["elapsed"].shape
            or physical["progress"].shape != physical["elapsed"].shape
            or not np.allclose(clock.progress(physical["curve_time"]), physical["progress"], atol=1e-12, rtol=0)):
        raise ValueError("Physical commanded progress must match its own saved curve sampling time")
    fixture = FixtureFK(Path(run["report"]["robot_fixture_path"]), Path(run["report"]["feedback_path"]), curve["names"])
    spline = CubicHermiteSpline(curve["times"], curve["position"], curve["velocity"], axis=0)
    barrier = float(curve["metadata"]["executable_until_seconds"])
    if barrier <= 0:
        raise ValueError("A measured moving-prefix analysis needs a positive explicit evidence boundary")
    joint_metrics = []
    for column, limit in enumerate(fixture.limits):
        scalar = PPoly(spline.c[:, :, column], spline.x, extrapolate=False)
        record = {**limit, "planned_full": {}, "planned_until_barrier": {}, "physical_sampled": {}}
        for derivative, key, bounds in ((0, "q", (limit["lower"], limit["upper"])),
                                        (1, "dq", (-limit["velocity"], limit["velocity"])),
                                        (2, "ddq", (-limit["acceleration"], limit["acceleration"]))):
            polynomial = scalar.derivative(derivative)
            record["planned_full"][key] = polynomial_metrics(polynomial, *bounds, 0., clock.duration)
            record["planned_until_barrier"][key] = polynomial_metrics(polynomial, *bounds, 0., barrier)
            if derivative < 2:
                actual_column = physical["joint_names"].tolist().index(limit["joint"])
                record["physical_sampled"][key] = observed_metrics(
                    physical["elapsed"], physical["actual_"+key][:, actual_column], *bounds)
        left_acceleration = 2*scalar.c[1]
        right_acceleration = left_acceleration+6*scalar.c[0]*np.diff(scalar.x)
        jumps = left_acceleration[1:]-right_acceleration[:-1]
        jump = int(np.abs(jumps).argmax()) if len(jumps) else None
        record["largest_acceleration_knot_jump"] = {
            "absolute": float(abs(jumps[jump])) if jump is not None else 0.,
            "at_s": float(scalar.x[jump+1]) if jump is not None else None}
        joint_metrics.append(record)
    # Include every native knot, every original source-row clock, phase/event
    # boundaries, and every analytic limit crossing/extremum in display q/FK.
    exact_times = [0., clock.duration, barrier]
    for record in joint_metrics:
        for metric in record["planned_full"].values():
            exact_times.extend([metric["minimum_at_s"], metric["maximum_at_s"], metric["peak_at_s"]])
            exact_times.extend(value for span in metric["outside_open_intervals"]
                               for value in (span["start_s"], span["end_s"]))
    intervals = clock.planning_interval(clock.reference_progress)
    planned_times = np.unique(np.r_[curve["times"], np.linspace(0, clock.duration, 2201),
                                    intervals.ravel(), exact_times])
    if len(planned_times) > 20000:
        raise ValueError("Exact diagnostic sample budget exceeded; no native samples are dropped")
    planned_q, planned_dq, planned_ddq = (spline(planned_times, derivative) for derivative in range(3))
    # Preserve both one-sided accelerations at every internal native knot.
    # Drawing only PPoly's right value can omit a true left-side extremum.
    knot_rows = np.searchsorted(planned_times, curve["times"][1:-1])
    left_ddq = 2*spline.c[1]+6*spline.c[0]*np.diff(curve["times"])[:, None]
    ddq_plot_times = np.insert(planned_times, knot_rows, curve["times"][1:-1])
    ddq_plot_values = np.insert(planned_ddq, knot_rows, left_ddq[:-1], axis=0)
    ddq_sides = np.full(len(planned_times), "sample", dtype="U6")
    ddq_sides[knot_rows] = "right"
    ddq_plot_sides = np.insert(ddq_sides, knot_rows, "left")
    planned_pose = fixture.evaluate(planned_q, curve["names"])
    native_rows = np.searchsorted(planned_times, curve["times"])
    saved_pose = np.tile(np.eye(4), (len(curve["times"]), 1, 1))
    saved_pose[:, :3, 3] = curve["eef_position"]
    saved_pose[:, :3, :3] = Rotation.from_quat(curve["eef_quaternion_wxyz"][:, [1, 2, 3, 0]]).as_matrix()
    fk_position, fk_rotation = pose_errors(planned_pose[native_rows], saved_pose)
    # Numerical display cross-check only, independent of planner feasibility.
    # These tolerances cover the saved float32 native FK and double CPU FK.
    if fk_position.max() > 1e-5 or fk_rotation.max() > 1e-5:
        raise ValueError("Independent CPU FK does not reproduce the saved native world TCP")
    actual_rows = np.unique(np.linspace(0, len(physical["elapsed"])-1, 65).astype(int))
    actual_check = fixture.evaluate(physical["actual_q"][actual_rows], physical["joint_names"].tolist())
    actual_p_error, actual_r_error = pose_errors(actual_check, physical["actual_tool_pose"][actual_rows])
    if actual_p_error.max() > 1e-8 or actual_r_error.max() > 1e-8:
        raise ValueError("Recorded physical q and actual world TCP fail the independent fixture check")
    planned_progress = clock.progress(planned_times)
    source_rotation = Slerp(clock.reference_progress, Rotation.from_matrix(reference_pose[:, :3, :3]))

    def reference_at(progress):
        poses = np.tile(np.eye(4), (len(progress), 1, 1))
        poses[:, :3, 3] = np.column_stack([np.interp(progress, clock.reference_progress, reference_pose[:, axis, 3])
                                         for axis in range(3)])
        poses[:, :3, :3] = source_rotation(progress).as_matrix()
        return poses

    planned_reference = reference_at(planned_progress)
    actual_reference = reference_at(physical["progress"])
    planned_errors = pose_errors(planned_pose, planned_reference)
    actual_errors = pose_errors(physical["actual_tool_pose"], actual_reference)
    phase_metrics = []
    for index, phase in enumerate(clock.phases):
        selected = (planned_progress >= index) & (planned_progress <= index+1)
        actual_selected = (physical["progress"] >= index) & (physical["progress"] <= index+1)
        phase_metrics.append({**phase,
            "interpretation": "deviation from frozen ground_program geometry; not a new hard-constraint verdict",
            "planned_full": error_summary(*planned_errors, planned_times, selected),
            "planned_until_barrier": error_summary(*planned_errors, planned_times, selected & (planned_times <= barrier)),
            "actual_command_progress_aligned": error_summary(*actual_errors, physical["elapsed"], actual_selected)})
    barrier_index = int(np.flatnonzero(planned_times == barrier)[0])
    events = []
    for event in sorted(curve["metadata"]["events"], key=lambda row: (row["time"], row["event_id"])):
        source_time = float(clock.source_time(event["progress"]))
        applied = [row for row in trial["execution"]["events"] if row["event_id"] == event["event_id"]]
        events.append({"event_id": event["event_id"], "kind": event["kind"],
            "source_frame": event["source_frame"], "progress": event["progress"],
            "source_time_s": source_time, "source_elapsed_s": source_time-float(reference["times"][0]),
            "planning_time_s": event["time"], "planning_hold_until_s": event["hold_until"],
            "physically_applied_at_elapsed_s": applied[0]["applied_at_s"] if applied else None,
            "requires_evidence": event["requires_evidence"]})
    report = {"segment_id": segment["id"], "candidate_id": run["report"]["candidate_id"],
        "run_directory": str(run["directory"]), "curve_path": str(curve["path"]),
        "trial_report_path": str(trial["path"]), "reference_path": str(reference_path),
        "fixture_path": str(fixture.fixture_path), "feedback_path": str(fixture.feedback_path),
        "classification": curve["classification"], "validation_status": curve["metadata"]["validation_status"],
        "physics_outcome": trial["report"]["physics_outcome"], "planner_attempt": curve["attempt"],
        "clock": {"planning_duration_s": clock.duration, "evidence_boundary_s": barrier,
                  "source_origin_s": float(reference["times"][0]),
                  "source_duration_s": float(reference["times"][-1]-reference["times"][0]),
                  "physical_elapsed_end_s": float(physical["elapsed"][-1]),
                  "physical_sim_time_end_s": float(physical["sim_time"][-1]),
                  "physical_command_progress_meaning": "sampled command progress; not achieved geometric progress"},
        "events": events, "phases": phase_metrics, "joints": joint_metrics,
        "eef_errors_at_evidence_boundary": {"planned_position_m": float(planned_errors[0][barrier_index]),
            "planned_rotation_rad": float(planned_errors[1][barrier_index]),
            "actual_position_m": float(actual_errors[0][-1]), "actual_rotation_rad": float(actual_errors[1][-1])},
        "cpu_fk_crosscheck": {"native_knots": len(native_rows),
            "maximum_native_translation_difference_m": float(fk_position.max()),
            "maximum_native_rotation_difference_rad": float(fk_rotation.max()),
            "physical_samples_checked": len(actual_rows),
            "maximum_physical_translation_difference_m": float(actual_p_error.max()),
            "maximum_physical_rotation_difference_rad": float(actual_r_error.max())},
        "sample_counts": {"source_reference": len(reference_pose), "native_knots": len(curve["times"]),
                          "dense_cpu_fk": len(planned_times), "physical": len(physical["elapsed"])},
        "eef_polyline_lengths_m": {
            "reference": float(np.linalg.norm(np.diff(reference_pose[:, :3, 3], axis=0), axis=1).sum()),
            "planned_full": float(np.linalg.norm(np.diff(planned_pose[:, :3, 3], axis=0), axis=1).sum()),
            "planned_until_barrier": float(np.linalg.norm(np.diff(planned_pose[planned_times <= barrier, :3, 3], axis=0), axis=1).sum()),
            "actual_physical": float(np.linalg.norm(np.diff(physical["actual_tool_pose"][:, :3, 3], axis=0), axis=1).sum())},
        "interpretation": ["The original native curve remains rejected and is never promoted by visualization",
            "The frozen ground_program reference already includes its historical free-prefix connection",
            "Free-phase reference deviation alone is not a hard planning-constraint failure",
            "Pose deviations are sampled diagnostic metrics; joint polynomial extrema/crossings are analytic",
            "Pose RMS uses trapezoidal time weighting; EEF path lengths are sampled polylines, not exact arc integrals",
            "Acceleration plots retain both one-sided values at every internal native Hermite knot",
            "SO(3) shortest-angle errors do not measure winding; joint q is shown without angle wrapping",
            "No actual acceleration is fabricated from differencing the physical feedback"]}
    arrays = {"planned_times": planned_times, "planned_progress": planned_progress,
        "planned_q": planned_q, "planned_dq": planned_dq, "planned_ddq": planned_ddq,
        "ddq_plot_times": ddq_plot_times, "ddq_plot_values": ddq_plot_values, "ddq_plot_sides": ddq_plot_sides,
        "planned_eef_pose_world": planned_pose, "planned_reference_pose_world": planned_reference,
        "planned_position_error_m": planned_errors[0], "planned_rotation_error_rad": planned_errors[1],
        "reference_progress": clock.reference_progress, "reference_pose_world": reference_pose,
        "source_frames": reference["frames"], "source_times": reference["times"],
        "physical_elapsed": physical["elapsed"], "physical_sim_time": physical["sim_time"],
        "physical_curve_time": physical["curve_time"], "physical_command_progress": physical["progress"],
        "actual_tool_pose_world": physical["actual_tool_pose"], "actual_q": physical["actual_q"],
        "actual_dq": physical["actual_dq"], "actual_joint_names": physical["joint_names"],
        "actual_position_error_m": actual_errors[0], "actual_rotation_error_rad": actual_errors[1],
        "planned_joint_names": np.asarray(curve["names"])}
    return {"run": run, "curve": curve, "trial": trial, "clock": clock, "report": report,
            "arrays": arrays, "limits": fixture.limits}


def display_rows(length, maximum=3000):
    """Display-only uniform thinning; every diagnostic metric uses full inputs."""
    return np.unique(np.linspace(0, length-1, min(length, maximum)).astype(int))


def eef_figure(case):
    data, clock = case["arrays"], case["clock"]
    figure = go.Figure()
    source_interval = clock.planning_interval(data["reference_progress"])
    records = [("冻结 ground_program EEF 参考", REFERENCE_COLOR, data["reference_pose_world"],
                np.column_stack((data["reference_progress"], data["source_frames"], data["source_times"], source_interval)),
                "s=%{customdata[0]:.4f}<br>源帧 %{customdata[1]:.0f} · 源钟 %{customdata[2]:.4f} s<br>"
                "对应规划钟 [%{customdata[3]:.4f}, %{customdata[4]:.4f}] s"),
               ("原生 q(t) 经 CPU FK", PLANNED_COLOR, data["planned_eef_pose_world"],
                np.column_stack((data["planned_progress"], data["planned_times"], clock.source_time(data["planned_progress"]))),
                "命令 s=%{customdata[0]:.4f}<br>规划钟 %{customdata[1]:.4f} s<br>源钟 %{customdata[2]:.4f} s"),
               ("原生物理实际 TCP", ACTUAL_COLOR, data["actual_tool_pose_world"],
                np.column_stack((data["physical_command_progress"], data["physical_elapsed"],
                                 data["physical_curve_time"], data["physical_sim_time"])),
                "命令 s=%{customdata[0]:.4f}<br>物理 elapsed %{customdata[1]:.4f} s<br>"
                "命令采样规划钟 %{customdata[2]:.4f} s · sim_time %{customdata[3]:.4f} s")]
    displayed = []
    for label, color, poses, custom, hover in records:
        rows = display_rows(len(poses)); xyz = poses[rows, :3, 3]; displayed.append(poses[:, :3, 3])
        figure.add_trace(go.Scatter3d(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode="lines",
            name=label, line={"color": color, "width": 5}, customdata=custom[rows],
            hovertemplate=hover+"<br>x=%{x:.4f}, y=%{y:.4f}, z=%{z:.4f} m<extra>"+label+"</extra>"))
    limits = equal_axis_ranges(np.concatenate(displayed))
    figure.update_layout(title="同一 world 坐标中的三类 EEF / TCP（默认全部显示）", height=560,
        scene={axis+"axis": {"title": axis+" (m)", "range": limits[index], "nticks": 5}
               for index, axis in enumerate("xyz")}, margin={"l": 0, "r": 0, "t": 50, "b": 20},
        legend={"orientation": "h", "y": -.03})
    figure.update_scenes(aspectmode="cube")
    return figure


def progress_figures(case):
    data, clock = case["arrays"], case["clock"]
    clocks = go.Figure()
    for name, color, progress, times in (
            ("原参考源钟（相对源入口）", REFERENCE_COLOR, data["reference_progress"], data["source_times"]-data["source_times"][0]),
            ("规划钟", PLANNED_COLOR, data["planned_progress"], data["planned_times"]),
            ("物理 elapsed", ACTUAL_COLOR, data["physical_command_progress"], data["physical_elapsed"])):
        rows = display_rows(len(progress))
        clocks.add_trace(go.Scatter(x=progress[rows], y=times[rows], name=name, line={"color": color}, mode="lines"))
    clocks.update_layout(title="事件对齐后的三个时钟；s 是命令进度", xaxis_title="命令 / 源 phase 进度 s",
                         yaxis_title="各自入口后的秒数", height=340, legend={"orientation": "h", "y": -.25})
    coordinates = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=.08)
    errors = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=.15)
    for label, color, progress, poses, time_values in (
            ("ground_program 参考", REFERENCE_COLOR, data["reference_progress"], data["reference_pose_world"], data["source_times"]),
            ("规划 CPU FK", PLANNED_COLOR, data["planned_progress"], data["planned_eef_pose_world"], data["planned_times"]),
            ("实际 TCP", ACTUAL_COLOR, data["physical_command_progress"], data["actual_tool_pose_world"], data["physical_elapsed"])):
        selection = display_rows(len(progress))
        for axis in range(3):
            coordinates.add_trace(go.Scatter(x=progress[selection], y=poses[selection, axis, 3], name=label,
                legendgroup=label, showlegend=axis == 0, mode="lines", line={"color": color},
                customdata=time_values[selection], hovertemplate="s=%{x:.4f}<br>位置 %{y:.5f} m<br>该轨迹钟 %{customdata:.5f} s<extra>"+label+"</extra>"), row=axis+1, col=1)
    for label, color, prefix, progress in (("规划 FK 对原参考", PLANNED_COLOR, "planned", data["planned_progress"]),
                                          ("实际 TCP 对原参考", ACTUAL_COLOR, "actual", data["physical_command_progress"])):
        selection = display_rows(len(progress))
        for row, quantity in enumerate(("position_error_m", "rotation_error_rad"), 1):
            errors.add_trace(go.Scatter(x=progress[selection], y=data[prefix+"_"+quantity][selection],
                name=label, legendgroup=label, showlegend=row == 1, mode="lines", line={"color": color}), row=row, col=1)
    for index in range(1, len(clock.phases)):
        for figure in (clocks, coordinates, errors):
            figure.add_vline(x=index, line_dash="dash", line_color="#777")
    for axis, label in enumerate("xyz", 1):
        coordinates.update_yaxes(title_text=label+" (m)", row=axis, col=1)
    coordinates.update_xaxes(title_text="命令 / 源 phase 进度 s", row=3, col=1)
    coordinates.update_layout(title="按命令进度对齐的 EEF 坐标；参考自由段偏差仅作诊断", height=600,
                              legend={"orientation": "h", "y": -.15})
    errors.update_yaxes(title_text="位置差 (m)", row=1, col=1)
    errors.update_yaxes(title_text="SO(3) 最短角 (rad)", row=2, col=1)
    errors.update_xaxes(title_text="命令 / 源 phase 进度 s", row=2, col=1)
    errors.update_layout(title="相对冻结参考的偏差（不代替原规划器约束判定）", height=440,
                         legend={"orientation": "h", "y": -.25})
    return clocks, coordinates, errors


def joint_figure(case):
    """Show raw named joint coordinates; no modulo, clipping or smoothing."""
    data, clock = case["arrays"], case["clock"]
    names = data["planned_joint_names"].tolist()
    figure = make_subplots(rows=len(names), cols=3, shared_xaxes=True, vertical_spacing=.025,
                           column_titles=("q：原生位置限制", "dq：原生速度限制", "ddq：Hermite 解析二阶导数"))
    stop = case["report"]["clock"]["evidence_boundary_s"]
    for row, limit in enumerate(case["limits"], 1):
        name, unit = limit["joint"], limit["unit"]
        for column, quantity in enumerate(("q", "dq", "ddq"), 1):
            planned_x = data["ddq_plot_times"] if quantity == "ddq" else data["planned_times"]
            planned_y = data["ddq_plot_values"] if quantity == "ddq" else data["planned_"+quantity]
            figure.add_trace(go.Scattergl(x=planned_x, y=planned_y[:, row-1],
                name="原生诊断曲线", legendgroup="planned", showlegend=row == column == 1,
                mode="lines", line={"color": PLANNED_COLOR, "width": 1.6}), row=row, col=column)
            if quantity != "ddq":
                actual_column = data["actual_joint_names"].tolist().index(name)
                figure.add_trace(go.Scattergl(x=data["physical_elapsed"], y=data["actual_"+quantity][:, actual_column],
                    name="原生物理反馈", legendgroup="actual", showlegend=row == column == 1,
                    mode="lines", line={"color": ACTUAL_COLOR, "width": 1.2, "dash": "dot"}), row=row, col=column)
            bounds = ((limit["lower"], limit["upper"]) if quantity == "q" else
                      (-limit["velocity"], limit["velocity"]) if quantity == "dq" else
                      (-limit["acceleration"], limit["acceleration"]))
            for bound in bounds:
                figure.add_trace(go.Scatter(x=[0, clock.duration], y=[bound, bound], mode="lines",
                    line={"color": LIMIT_COLOR, "dash": "dash", "width": 1}, name="fixture 原生限制",
                    legendgroup="limits", showlegend=row == column == 1 and bound == bounds[0]), row=row, col=column)
            figure.add_vrect(x0=stop, x1=clock.duration, fillcolor="#999", opacity=.12, line_width=0, row=row, col=column)
            for event_time in sorted({event["planning_time_s"] for event in case["report"]["events"] if event["planning_time_s"] > 0}):
                figure.add_vline(x=event_time, line_dash="dot", line_color="#777", row=row, col=column)
            figure.update_yaxes(title_text=f"{name}<br>{unit}"+("" if column == 1 else "/s" if column == 2 else "/s²"), row=row, col=column)
            if row == len(names):
                figure.update_xaxes(title_text="规划 t / 物理 elapsed (s)", row=row, col=column)
    figure.update_layout(title=f"{len(names)} 轴 q / dq / ddq；灰区为证据屏障后未执行的原规划尾段",
                          height=250*len(names), margin={"t": 90, "b": 65},
                          legend={"orientation": "h", "y": 1.035})
    return figure


def number(value, digits=4):
    return "—" if value is None else f"{value:.{digits}f}"


def tables(case):
    report = case["report"]
    event_rows = []
    for event in report["events"]:
        event_rows.append("<tr>"+"".join(f"<td>{escape(value)}</td>" for value in (
            event["kind"], event["source_frame"], number(event["progress"]), number(event["source_elapsed_s"]),
            number(event["planning_time_s"]), number(event["physically_applied_at_elapsed_s"])))+"</tr>")
    body = ["<h3>事件与时钟</h3><table><tr><th>事件</th><th>源帧</th><th>s</th><th>源钟相对入口 s</th>"
            "<th>规划钟 s</th><th>实际命令施加 elapsed s</th></tr>"+"".join(event_rows)+"</table>",
            "<p>verification 的实际命令栏为空：它仍是待证据确认的屏障。停止时间不是证据通过。</p>",
            "<h3>关节越界与峰值（原规划全时段）</h3><table><tr><th>关节</th><th>q 范围 / 原生范围</th>"
            "<th>q 首次越界 t</th><th>|dq| 峰值 / 限制</th><th>dq 首次越界 t</th>"
            "<th>|ddq| 峰值 / 限制</th><th>ddq 峰值 t</th></tr>"]
    for joint in report["joints"]:
        q, dq, ddq = (joint["planned_full"][key] for key in ("q", "dq", "ddq"))
        body.append("<tr>"+"".join(f"<td>{escape(value)}</td>" for value in (
            joint["joint"], f'{q["minimum"]:.4f}…{q["maximum"]:.4f} / {joint["lower"]:.4f}…{joint["upper"]:.4f}',
            number(q["first_limit_crossing_s"]), f'{dq["peak_absolute"]:.4f} / {joint["velocity"]:.4f}',
            number(dq["first_limit_crossing_s"]), f'{ddq["peak_absolute"]:.4f} / {joint["acceleration"]:.4f}',
            number(ddq["peak_at_s"])))+"</tr>")
    body.append("</table><p>位置与速度取自 fixture URDF，加速度取自实际 robot.yml cspace 配置。"
                "报告另列屏障前指标、全部解析越界区间和真实反馈的采样越界；不对 q 做角度取模。</p>")
    body.append("<h3>各原阶段相对 ground_program 参考的偏差</h3><table><tr><th>原阶段</th>"
                "<th>规划屏障前最大位置差 m</th><th>规划屏障前最大姿态差 rad</th>"
                "<th>实际最大位置差 m</th><th>实际最大姿态差 rad</th></tr>")
    for phase in report["phases"]:
        planned, actual = phase["planned_until_barrier"], phase["actual_command_progress_aligned"]
        body.append("<tr>"+"".join(f"<td>{escape(value)}</td>" for value in (
            f'{phase["id"]}: {phase["name"]} ({phase["start_frame"]}–{phase["end_frame"]})',
            number(planned["position_max_m"] if planned else None), number(planned["rotation_max_rad"] if planned else None),
            number(actual["position_max_m"] if actual else None), number(actual["rotation_max_rad"] if actual else None)))+"</tr>")
    body.append("</table><p>自由段参考允许改变，这一列不构成硬约束失败判定。"
                "末段数值是相对冻结参考的采样偏差；原约束残差保留在下面的规划检查中。"
                "姿态差为 SO(3) 最短角，不表示关节绕圈次数。</p>")
    return "".join(body)


def static_summary(cases, output):
    figure = plt.figure(figsize=(15, 5.5*len(cases)))
    for row, case in enumerate(cases):
        data = case["arrays"]
        axis = figure.add_subplot(len(cases), 2, row*2+1, projection="3d")
        xyz_all = []
        for label, color, key in (("ground_program reference", REFERENCE_COLOR, "reference_pose_world"),
                                   ("native q(t) CPU FK", PLANNED_COLOR, "planned_eef_pose_world"),
                                   ("actual physical TCP", ACTUAL_COLOR, "actual_tool_pose_world")):
            xyz = data[key][:, :3, 3]; xyz_all.append(xyz); indices = display_rows(len(xyz))
            axis.plot(*xyz[indices].T, color=color, linewidth=1.5, label=label)
        limits = equal_axis_ranges(np.concatenate(xyz_all))
        axis.set_xlim(limits[0]); axis.set_ylim(limits[1]); axis.set_zlim(limits[2]); axis.set_box_aspect((1,1,1))
        axis.set_xlabel("world x (m)"); axis.set_ylabel("y (m)"); axis.set_zlabel("z (m)")
        axis.set_title(case["report"]["candidate_id"]+" · EEF / TCP")
        axis.legend(fontsize=8, loc="upper left")
        bars = figure.add_subplot(len(cases), 2, row*2+2)
        joints = case["report"]["joints"]; x = np.arange(len(joints))
        speed = [joint["planned_full"]["dq"]["peak_absolute"]/joint["velocity"] for joint in joints]
        acceleration = [joint["planned_full"]["ddq"]["peak_absolute"]/joint["acceleration"] for joint in joints]
        bars.bar(x-.18, speed, .36, label="|dq| peak / native limit", color=PLANNED_COLOR)
        bars.bar(x+.18, acceleration, .36, label="|ddq| peak / native limit", color="#d48236")
        bars.axhline(1, color=LIMIT_COLOR, linestyle="--", label="native limit")
        bars.set_xticks(x, [str(index+1) for index in x]); bars.set_xlabel("named arm joint index (original order)")
        bars.set_ylabel("ratio"); bars.set_title("原生诊断曲线导数峰值与配置限制"); bars.legend(fontsize=9)
    figure.suptitle("冻结原生曲线失败分析 · 规划拒绝 / 物理判定保持原样", fontsize=16)
    figure.tight_layout(rect=(0,0,1,.97)); figure.savefig(output/"summary.png", dpi=140); plt.close(figure)


def static_joint_detail(case, output, index):
    """Provide a reviewable local PNG of the same q/dq/ddq and original rails."""
    data = case["arrays"]
    figure, axes = plt.subplots(len(case["limits"]), 3, figsize=(17, 2.1*len(case["limits"])),
                                sharex=True, squeeze=False)
    for row, limit in enumerate(case["limits"]):
        for column, quantity in enumerate(("q", "dq", "ddq")):
            axis = axes[row, column]
            x = data["ddq_plot_times"] if quantity == "ddq" else data["planned_times"]
            y = data["ddq_plot_values"] if quantity == "ddq" else data["planned_"+quantity]
            axis.plot(x, y[:, row], color=PLANNED_COLOR, linewidth=1, label="native diagnostic curve")
            if quantity != "ddq":
                actual_column = data["actual_joint_names"].tolist().index(limit["joint"])
                axis.plot(data["physical_elapsed"], data["actual_"+quantity][:, actual_column],
                          color=ACTUAL_COLOR, linewidth=.8, alpha=.8, label="actual physical feedback")
            bounds = ((limit["lower"], limit["upper"]) if quantity == "q" else
                      (-limit["velocity"], limit["velocity"]) if quantity == "dq" else
                      (-limit["acceleration"], limit["acceleration"]))
            for bound in bounds:
                axis.axhline(bound, color=LIMIT_COLOR, linestyle="--", linewidth=.8)
            axis.axvspan(case["report"]["clock"]["evidence_boundary_s"], case["clock"].duration, color="#888", alpha=.12)
            for event in case["report"]["events"]:
                if event["planning_time_s"] > 0:
                    axis.axvline(event["planning_time_s"], color="#777", linestyle=":", linewidth=.8)
            unit = limit["unit"]+("" if quantity == "q" else "/s" if quantity == "dq" else "/s²")
            axis.set_ylabel(f'{limit["joint"]}\n{quantity} ({unit})', fontsize=8)
            axis.grid(alpha=.15)
            if row == len(case["limits"])-1:
                axis.set_xlabel("planning t / physical elapsed (s)")
    axes[0,0].legend(fontsize=8)
    figure.suptitle(case["report"]["candidate_id"]+" · 旧失败曲线 q / dq / ddq\n"
                   "蓝：原生诊断曲线；紫：原物理反馈；红虚线：fixture 限制；灰区：屏障后未执行尾段", fontsize=14)
    figure.tight_layout(rect=(0,0,1,.96)); figure.savefig(output/f"case-{index:02d}-joints.png", dpi=140); plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--trial-report", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic(); configure_font()
    manifest, audit = read_json(args.manifest), read_json(args.audit)
    trial_paths = [path.resolve() for path in args.trial_report]
    if len(trial_paths) != len(set(trial_paths)) or len(trial_paths) > 8:
        raise ValueError("Supply one to eight distinct explicit trial reports within the local CPU budget")
    sources = [Path(read_json(path)["trajectory_path"]).resolve() for path in trial_paths]
    run_dirs = list(dict.fromkeys(path.parent.parent for path in sources))
    runs, unmatched = load_runs(run_dirs, [path.parent for path in trial_paths], manifest)
    cases = []
    for source, trial_path in zip(sources, trial_paths):
        matches = [(run, curve, trial) for run in runs for curve in run["curves"] for trial in curve["trials"]
                   if curve["path"] == source and trial["path"].resolve() == trial_path]
        if len(matches) != 1:
            raise ValueError("Each trial must bind exactly one validated current run artifact")
        run, curve, trial = matches[0]
        if Path(run["report"]["manifest_path"]).resolve() != args.manifest.resolve():
            raise ValueError("A same-named segment from another manifest cannot be substituted")
        cases.append(analyze(run, curve, trial, manifest, audit))
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=False)
    body = ["<h1>旧失败产物：EEF 与关节曲线分析</h1>"
            "<p>全部数据来自明确绑定的原规划/物理产物，未重新规划或模拟。绿色为冻结 ground_program 参考，"
            "蓝色为原生 Hermite q(t) 经独立 CPU FK，紫色为原物理 TCP / 关节反馈。"
            "参考中的自由前缀已经过当时的连接处理；它不是未变换的示教原轨迹。</p>"
            "<p><b>图中 s 是保存的命令进度，不代表机械臂实际完成的几何进度。</b>"
            "旧阶段边界来自冻结审计，不根据当前 builder 或新的 pre-grasp 契约重写。"
            "源钟、规划钟与物理钟分别保留；自由段偏差仅作诊断。</p>"]
    for index, case in enumerate(cases, 1):
        report = case["report"]
        npz_path, json_path = output/f"case-{index:02d}.npz", output/f"case-{index:02d}.json"
        np.savez_compressed(npz_path, **case["arrays"])
        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
        body.append(f'<section><h2>{escape(report["candidate_id"])} · {escape(case["run"]["directory"].name)} · '
                    f'{escape(case["curve"]["path"].name)}</h2><p><b>{CLASSIFICATIONS[report["classification"]][0]}</b>；'
                    f'validation={escape(report["validation_status"])}；physics={escape(report["physics_outcome"])}。</p>')
        body.append(" · ".join(link(path, output, label) for path, label in (
            (case["curve"]["path"], "冻结原生 q/dq"), (case["trial"]["path"], "冻结物理报告"),
            (report["reference_path"], "ground_program 参考"), (npz_path, "本页全部诊断数组"),
            (json_path, "解析越界 / 峰值 / 末段指标"),
            (output/f"case-{index:02d}-joints.png", "关节静态详图"))))
        body.append(tables(case)); body.append(plot_html(eef_figure(case)))
        body.extend(plot_html(figure) for figure in progress_figures(case))
        body.append(plot_html(joint_figure(case)))
        body.append(details("原规划器检查（不改接受状态）", report["planner_attempt"]))
        body.append(details("独立 CPU FK / 物理 q–TCP 交叉核对", report["cpu_fk_crosscheck"]))
        body.append(details("完整诊断指标与全部解析越界区间", report)+"</section>")
    style = """body{margin:auto;padding:24px;max-width:1550px;font:16px/1.5 system-ui,sans-serif;
background:#f4f6f9;color:#23354a}section{background:white;border:1px solid #d9e0e8;border-radius:8px;
padding:22px;margin:22px 0}table{border-collapse:collapse;width:100%;font-size:13px}th,td{border:1px solid
#d9e0e8;padding:8px;text-align:left}th{background:#eef2f7}pre{font-size:12px;white-space:pre-wrap;
overflow-wrap:anywhere;max-height:560px;overflow:auto}details{margin:14px 0}a{color:#2466a3}
@media(max-width:800px){body{padding:8px}section{padding:10px}table{font-size:11px}}
"""
    (output/"index.html").write_text('<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1"><title>EEF 与关节失败分析</title>'
        f'<style>{style}</style><script>{get_plotlyjs()}</script></head><body>'+"".join(body)+"</body></html>")
    static_summary(cases, output)
    for index, case in enumerate(cases, 1):
        static_joint_detail(case, output, index)
    summary = {"manifest_path": str(args.manifest.resolve()), "audit_path": str(args.audit.resolve()),
        "trial_reports": [str(path) for path in trial_paths], "case_reports": [f"case-{index:02d}.json" for index in range(1,len(cases)+1)],
        "case_count": len(cases), "planning_rerun": False, "simulation_rerun": False, "gpu_used": False,
        "browser_interaction_tested": False, "elapsed_seconds": time.monotonic()-started,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024}
    (output/"diagnostics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
# END independent CPU failure analysis with explicit clocks and native units.
