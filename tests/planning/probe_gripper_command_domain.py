"""Bounded robot-only calibration of inward native finger command endpoints.

This is a fresh device bench, not execution or rebinding of a task candidate.
The validated task archive supplies named reference motion and unchanged
physical protection limits. Only robot material-face declarations are read
from contact bindings. No object state, contact sensor, renderer or solver is
used to select or adjust controls. All choices precede the first native step.
"""

# BEGIN independent command-domain bench with one shared physical-step ledger.
from __future__ import annotations

import argparse
from copy import deepcopy
import gc
import json
import os
from pathlib import Path
import resource
import sys
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from manipulation.execution import ExecutionLimits, TrajectoryExecutor
from manipulation.execution_binding import check_result
from manipulation.gripper_trajectory import FingerTrajectory
from recorded_simulation import load_saved_trajectory


MAX_STEPS = 8000
NORMAL_STEPS = 1100
HOLD_STEPS = 200
MAX_SCAN_SAMPLES = 25001
MAX_RSS_MIB = 4096
MAX_CONTACT_ROWS_PER_SAMPLE = 5000
MAX_CONTACT_PAIRS_PER_CASE = 256
TORQUE_FIELDS = ("requested_arm_torque", "arm_position_feedback_torque",
                 "arm_velocity_feedback_torque", "arm_velocity_feedforward_torque",
                 "arm_acceleration_feedforward_torque", "arm_bias_torque")


def json_value(value):
    """Retain invalid numeric evidence explicitly instead of losing reports."""
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return {"nonfinite": str(value)}
    return value


def write_json(path, value):
    path.write_text(json.dumps(json_value(value), indent=2, allow_nan=False)+"\n")


def candidate_targets(original, ranges, fractions):
    """Map each named endpoint inward by a declared native-stroke fraction.

    Both source intentions must already be opposite native endpoints. An
    arbitrary intermediate command has no inferred open/close direction.
    Fractions are chosen before measurements, in increasing aperture loss.
    """
    ranges = np.asarray(ranges, float)
    values = np.asarray(fractions, float)
    if (ranges.ndim != 2 or ranges.shape[1] != 2 or not np.isfinite(ranges).all()
            or np.any(ranges[:, 0] >= ranges[:, 1]) or values.ndim != 1
            or not 1 <= values.size <= 2 or not np.isfinite(values).all()
            or np.any(values <= 0) or np.any(values >= .5) or np.any(np.diff(values) <= 0)):
        raise ValueError("At most two strictly increasing inward stroke fractions in (0,.5) are required")
    source = {kind: np.asarray(original[kind], float) for kind in ("open", "close")}
    for target in source.values():
        if (target.shape != (len(ranges),) or not np.isfinite(target).all()
                or np.any((target != ranges[:, 0]) & (target != ranges[:, 1]))):
            raise ValueError("Original named open/close commands must be exact native endpoints")
    if np.any(source["open"] == source["close"]):
        raise ValueError("Open and close must identify opposite endpoints for every finger")
    return [{"fraction": float(fraction), "inset_m": fraction*np.diff(ranges, axis=1)[:, 0],
             "targets": {kind: np.where(target == ranges[:, 0],
                 ranges[:, 0]+fraction*np.diff(ranges, axis=1)[:, 0],
                 ranges[:, 1]-fraction*np.diff(ranges, axis=1)[:, 0])
                 for kind, target in source.items()}} for fraction in values]


def robot_xml(source_text, *, base_link, joint_names, source_dir):
    """Keep the complete fixed-base robot, physics and its self constraints.

    External independent world objects are intentionally absent from this
    fresh unloaded bench. Cross-boundary constraints are rejected instead
    of quietly turning a coupled robot into a different mechanism. Assets
    and defaults remain untouched apart from resolving their file paths.
    """
    root = ET.fromstring(source_text)
    if root.findall(".//include"):
        raise ValueError("Bench extraction requires an already expanded source XML")
    world = root.find("worldbody")
    roots = [] if world is None else [body for body in world.findall("body") if body.get("name") == base_link]
    if len(roots) != 1:
        raise ValueError("The explicit fixed robot base must be one direct world child")
    robot = roots[0]
    if robot.findall(".//freejoint"):
        raise ValueError("This bench requires a fixed-base robot")
    names = {kind: {item.get("name") for item in robot.iter(kind) if item.get("name")}
             for kind in ("body", "joint", "geom", "site")}
    if names["joint"] != set(joint_names):
        raise ValueError("The extracted robot must contain exactly all named controlled joints")
    for item in list(world):
        if item is not robot:
            world.remove(item)
    # Cameras/lights and sensors have no role in this device bench.
    for parent in root.iter():
        for child in list(parent):
            if child.tag in ("camera", "light"):
                parent.remove(child)
    for tag in ("sensor", "keyframe"):
        for element in root.findall(tag):
            root.remove(element)
    for tag in ("actuator", "contact", "equality", "tendon"):
        section = root.find(tag)
        if section is None:
            continue
        for row in list(section):
            membership = []
            for element in row.iter():
                for key, value in element.attrib.items():
                    kind = next((kind for kind in names if key in (kind, kind+"1", kind+"2")), None)
                    if kind is not None:
                        membership.append(value in names[kind])
            if not membership:
                raise ValueError(f"Cannot establish robot membership of {tag}/{row.tag}")
            if any(membership) and not all(membership):
                raise ValueError(f"Cross-boundary {tag}/{row.tag} prevents an unloaded robot bench")
            if not any(membership):
                section.remove(row)
    compiler = root.find("compiler")
    if compiler is not None:
        for key in ("meshdir", "texturedir", "assetdir"):
            if compiler.get(key):
                compiler.set(key, str((Path(source_dir)/compiler.get(key)).resolve()))
    return ET.tostring(root, encoding="unicode"), names


def mechanical_equivalence(original, isolated, names, joint_names):
    """Compare native compiled robot mechanics by names, not shifted IDs."""
    fields = {
        "body": ("body_pos", "body_quat", "body_ipos", "body_iquat", "body_mass", "body_inertia", "body_gravcomp"),
        "joint": ("jnt_type", "jnt_pos", "jnt_axis", "jnt_limited", "jnt_range", "jnt_margin", "jnt_solref", "jnt_solimp", "jnt_stiffness", "jnt_actfrclimited", "jnt_actfrcrange", "jnt_actgravcomp"),
        "geom": ("geom_type", "geom_size", "geom_pos", "geom_quat", "geom_friction", "geom_solref", "geom_solimp", "geom_contype", "geom_conaffinity", "geom_condim", "geom_margin", "geom_gap", "geom_solmix", "geom_priority", "geom_fluid"),
        "site": ("site_pos", "site_quat", "site_type", "site_size")}
    comparisons = 0
    for kind, attributes in fields.items():
        for name in sorted(names[kind]):
            a, b = getattr(original, kind)(name).id, getattr(isolated, kind)(name).id
            for field in attributes:
                if not np.array_equal(getattr(original, field)[a], getattr(isolated, field)[b]):
                    raise ValueError(f"Robot extraction changed {field}/{name}")
                comparisons += 1
    actuator_names = []
    for name in joint_names:
        a, b = original.joint(name).id, isolated.joint(name).id
        da, db = original.jnt_dofadr[a], isolated.jnt_dofadr[b]
        for field in ("qpos0", "qpos_spring"):
            if not np.array_equal(getattr(original, field)[original.jnt_qposadr[a]],
                                  getattr(isolated, field)[isolated.jnt_qposadr[b]]):
                raise ValueError(f"Robot extraction changed {field}/{name}")
            comparisons += 1
        for field in ("dof_armature", "dof_damping", "dof_frictionloss", "dof_solref", "dof_solimp"):
            if not np.array_equal(getattr(original, field)[da], getattr(isolated, field)[db]):
                raise ValueError(f"Robot extraction changed {field}/{name}")
            comparisons += 1
        source_ids = np.flatnonzero(original.actuator_trnid[:, 0] == a)
        target_ids = np.flatnonzero(isolated.actuator_trnid[:, 0] == b)
        if len(source_ids) != 1 or len(target_ids) != 1:
            raise ValueError("Each named scalar joint needs its one preserved native actuator")
        ai, bi = int(source_ids[0]), int(target_ids[0])
        if original.actuator(ai).name != isolated.actuator(bi).name:
            raise ValueError("Robot actuator identity changed during extraction")
        actuator_names.append(isolated.actuator(bi).name)
        for field in ("actuator_trntype", "actuator_dyntype", "actuator_gaintype", "actuator_biastype", "actuator_dynprm", "actuator_gainprm", "actuator_biasprm", "actuator_gear", "actuator_ctrllimited", "actuator_ctrlrange", "actuator_forcelimited", "actuator_forcerange", "actuator_actlimited", "actuator_actrange", "actuator_lengthrange", "actuator_cranklength"):
            if not np.array_equal(getattr(original, field)[ai], getattr(isolated, field)[bi]):
                raise ValueError(f"Robot extraction changed {field}/{name}")
            comparisons += 1
    option_fields = ("timestep", "gravity", "wind", "magnetic", "density", "viscosity", "impratio", "integrator", "cone", "jacobian", "solver", "iterations", "ls_iterations", "noslip_iterations", "ccd_iterations", "tolerance", "ls_tolerance", "noslip_tolerance", "ccd_tolerance", "disableflags", "enableflags", "o_margin", "o_solref", "o_solimp", "o_friction", "sdf_initpoints", "sdf_iterations")
    for field in option_fields:
        if not np.array_equal(getattr(original.opt, field), getattr(isolated.opt, field)):
            raise ValueError(f"Robot extraction changed native option {field}")
        comparisons += 1
    if isolated.nq != len(joint_names) or isolated.nv != len(joint_names):
        raise ValueError("Only the named scalar robot degrees of freedom may enter the bench")
    return {"passed": True, "named_field_comparisons": comparisons,
            "robot_bodies": sorted(names["body"]), "robot_geometries": sorted(names["geom"]),
            "joint_names": list(joint_names), "actuator_names": actuator_names,
            "native_options_compared": list(option_fields), "nq": isolated.nq, "nv": isolated.nv,
            "scope": "Preserved compiled robot mechanics; independent external objects removed before fresh setup"}


def material_gaps(materials, finger_q):
    """Project full calibrated face centers, including their native offset."""
    rows = list(materials.values())
    if len(rows) != 2:
        raise ValueError("Gap evidence requires exactly two explicitly declared material faces")
    centers = [np.mean(np.asarray(row["feature_points"])+np.einsum("vcj,j->vc",
        np.asarray(row["feature_joint_translation"]), finger_q), axis=0) for row in rows]
    normals = [np.asarray(row["feature_normal"]) for row in rows]
    delta = centers[1]-centers[0]
    return np.asarray([delta @ normals[0], -delta @ normals[1]])


def native_self_contacts(backend, remaining):
    """Independent robot-only contact evidence; never a control observation."""
    import mujoco
    model, data = backend.native_model, backend.native_data
    count = int(data.ncon)
    rows = []
    for index in range(min(count, remaining)):
        contact = data.contact[index]
        force = np.zeros(6)
        mujoco.mj_contactForce(model, data, index, force)
        rows.append({"geom1": model.geom(int(contact.geom1)).name,
                     "geom2": model.geom(int(contact.geom2)).name,
                     "distance_m": float(contact.dist), "efc_address": int(contact.efc_address),
                     "contact_frame_force": force})
    return rows, count > remaining


class SelfContactEvidence:
    """Bound pair summaries, retaining the first active raw contact exactly.

    Per-state counts and maximum absolute force stay in the numeric archive.
    Contact vectors are native contact-frame values; their extrema are
    diagnostic evidence and are never invented acceptance force tolerances.
    """
    def __init__(self):
        self.pairs, self.rows, self.active = {}, 0, 0
        self.overflow, self.nonfinite = False, False
        self.sample_counts, self.active_counts, self.maximum_forces = [], [], []

    def append(self, rows, *, state_index, sim_time, phase, overflow=False):
        self.overflow |= overflow
        self.rows += len(rows)
        active = sum(row["efc_address"] >= 0 for row in rows)
        self.active += active
        self.sample_counts.append(len(rows))
        self.active_counts.append(active)
        self.maximum_forces.append(max((float(np.max(np.abs(row["contact_frame_force"]))) for row in rows), default=0.))
        visited = set()
        for row in rows:
            self.nonfinite |= not bool(np.isfinite(row["distance_m"]) and np.isfinite(row["contact_frame_force"]).all())
            key = (row["geom1"], row["geom2"])
            if key not in self.pairs:
                if len(self.pairs) >= MAX_CONTACT_PAIRS_PER_CASE:
                    self.overflow = True
                    continue
                self.pairs[key] = {"geom1": key[0], "geom2": key[1], "sample_count": 0,
                    "contact_rows": 0, "active_constraint_rows": 0,
                    "distance_min_m": row["distance_m"], "distance_max_m": row["distance_m"],
                    "contact_frame_force_min": np.asarray(row["contact_frame_force"]).copy(),
                    "contact_frame_force_max": np.asarray(row["contact_frame_force"]).copy(), "first_active": None}
            pair = self.pairs[key]
            pair["contact_rows"] += 1
            pair["sample_count"] += int(key not in visited)
            visited.add(key)
            pair["distance_min_m"] = min(pair["distance_min_m"], row["distance_m"])
            pair["distance_max_m"] = max(pair["distance_max_m"], row["distance_m"])
            pair["contact_frame_force_min"] = np.minimum(pair["contact_frame_force_min"], row["contact_frame_force"])
            pair["contact_frame_force_max"] = np.maximum(pair["contact_frame_force_max"], row["contact_frame_force"])
            if row["efc_address"] >= 0:
                pair["active_constraint_rows"] += 1
                if pair["first_active"] is None:
                    pair["first_active"] = {"state_index": state_index, "sim_time": sim_time, "phase": phase, **deepcopy(row)}

    def describe(self):
        return {"recorded_rows": self.rows, "active_constraint_rows": self.active,
            "storage_overflow": self.overflow, "nonfinite_evidence": self.nonfinite,
            "pair_cap": MAX_CONTACT_PAIRS_PER_CASE,
            "rows_per_sample_cap": MAX_CONTACT_ROWS_PER_SAMPLE, "pairs": list(self.pairs.values()),
            "scope": "Independent robot-only native contact evaluation; any active constraint rejects unloaded calibration scope without changing control"}


class StepLedger:
    """Charge every attempted native step, even when post-step code raises.

    Confirmed clock advances and conservative charges are separate facts.
    A failed call with no clock advance still consumes one reserved attempt;
    an ambiguous clock prevents claiming an exact physical count.
    """
    def __init__(self, limit=MAX_STEPS, journal=None):
        self.limit, self.charged, self.confirmed = limit, 0, 0
        self.clock_exact = True
        self.journal = journal
        self.context = {}

    def persist(self, event, **details):
        """Append and fsync the small charge record before native integration."""
        if self.journal is not None:
            with self.journal.open("a") as stream:
                stream.write(json.dumps(json_value({"event": event, **self.describe(),
                    "context": self.context, **details}), allow_nan=False)+"\n")
                stream.flush()
                os.fsync(stream.fileno())

    def reserve_case(self):
        if self.limit-self.charged < NORMAL_STEPS+HOLD_STEPS:
            raise RuntimeError("Shared calibration budget cannot reserve this case and its bounded stop")

    def step(self, backend, native_step, capture):
        if self.charged >= self.limit:
            raise RuntimeError("Shared calibration physical-step limit exhausted")
        before = backend.sim_time
        self.charged += 1
        # A failed journal write stops before the physical call. Its reserved
        # attempt remains conservatively charged in the in-memory final ledger.
        self.persist("reserved_before_native_step", before_sim_time=before)
        primary = None
        try:
            native_step()
        except Exception as error:
            primary = error
        after = backend.sim_time
        delta = (after-before)/backend.timestep
        advanced = np.isfinite(delta) and abs(delta-1.) <= 1e-6
        stationary = np.isfinite(delta) and abs(delta) <= 1e-6
        if advanced:
            self.confirmed += 1
        elif not stationary:
            self.clock_exact = False
        try:
            self.persist("native_step_outcome", before_sim_time=before, after_sim_time=after,
                         confirmed_one_step=bool(advanced), native_error=None if primary is None else str(primary))
        except Exception as error:
            if primary is None:
                primary = error
        # Record even a known completed step whose subsequent forward failed.
        # Capture errors must not erase the earlier native exception.
        try:
            capture(advanced, before, after)
        except Exception as error:
            if primary is None:
                primary = error
        if primary is not None:
            raise primary
        if not advanced:
            raise RuntimeError("Native bench step did not advance its one declared timestep")

    def describe(self):
        return {"limit": self.limit, "charged_steps": self.charged,
                "confirmed_physics_steps": self.confirmed, "exact_clock_accounting": self.clock_exact,
                "remaining_charged_steps": self.limit-self.charged}


def scan_reference(model, curve, fixture, initial_fingers):
    """Sample robot-only load proxies; these are not tracking-error bounds."""
    import mujoco
    dt, duration = float(model.opt.timestep), float(curve.duration)
    times = np.unique(np.r_[np.arange(int(np.floor(duration/dt))+1)*dt, duration])
    if len(times) > MAX_SCAN_SAMPLES or duration < NORMAL_STEPS*dt:
        raise ValueError("Reference must cover the bounded bench window and scan sample cap")
    names = tuple(fixture["arm_joint_names"])+tuple(fixture["finger_joint_names"])
    narm = len(fixture["arm_joint_names"])
    joints = np.asarray([model.joint(name).id for name in names])
    qids, vids = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
    finger_actuators = [int(np.flatnonzero(model.actuator_trnid[:, 0] == joint)[0]) for joint in joints[narm:]]
    gain = model.actuator_gainprm[finger_actuators, 0]
    if not np.isfinite(gain).all() or np.any(gain <= 0):
        raise ValueError("Load proxies require positive native position gains")
    q, dq, ddq, _ = curve.sample(times)
    static = np.empty((len(times), len(initial_fingers)))
    dynamic = np.empty_like(static)
    orientations = np.empty((len(times), 3, 3))
    data, mass = mujoco.MjData(model), np.empty((model.nv, model.nv))
    site = model.site(fixture["tcp_site"]).id
    for index in range(len(times)):
        data.qpos[qids] = np.r_[q[index], initial_fingers]
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)
        static[index] = data.qfrc_bias[vids[narm:]]/gain
        orientations[index] = data.site_xmat[site].reshape(3, 3)
        data.qvel[vids[:narm]] = dq[index]
        mujoco.mj_forward(model, data)
        mujoco.mj_fullM(model, mass, data.qM)
        dynamic[index] = (data.qfrc_bias[vids[narm:]]+
                         mass[np.ix_(vids[narm:], vids[:narm])] @ ddq[index])/gain
    high, low = np.unravel_index(np.argmax(static), static.shape)[0], np.unravel_index(np.argmin(static), static.shape)[0]
    peak = int(np.argmax(np.max(np.abs(dynamic), axis=1)))
    window = NORMAL_STEPS*dt
    begin = float(np.clip(times[peak]-window/2, 0., duration-window))
    cases = [{"name": "static_maximum_bias", "mode": "static", "source_time_s": float(times[high]), "order": ["open", "close"]},
             {"name": "static_minimum_bias", "mode": "static", "source_time_s": float(times[low]), "order": ["close", "open"]},
             {"name": "dynamic_maximum_load_window", "mode": "dynamic", "source_time_s": begin, "order": ["open", "close"]}]
    summary = {"sample_count": len(times), "native_steps": 0, "proxy_units": "m",
        "static_proxy": "native finger qfrc_bias / native position gain at zero velocity",
        "dynamic_proxy": "(finger qfrc_bias + M_fa @ analytic arm ddq) / native position gain; finger dq=0",
        "maximum_dynamic_proxy_source_time_s": float(times[peak]),
        "static_minimum_per_finger": static.min(axis=0), "static_maximum_per_finger": static.max(axis=0),
        "dynamic_minimum_per_finger": dynamic.min(axis=0), "dynamic_maximum_per_finger": dynamic.max(axis=0),
        "arm_q_min": q.min(axis=0), "arm_q_max": q.max(axis=0),
        "arm_dq_min": dq.min(axis=0), "arm_dq_max": dq.max(axis=0),
        "arm_ddq_min": ddq.min(axis=0), "arm_ddq_max": ddq.max(axis=0),
        "scope": "Finite reference-grid extrema and robot-only load proxies; no continuous maximum, error guarantee or contact load coverage"}
    arrays = {"source_time": times, "q": q, "dq": dq, "ddq": ddq,
              "static_bias_over_gain": static, "dynamic_load_over_gain": dynamic,
              "tcp_rotation": orientations}
    return cases, summary, arrays


def source_commands(curve, limits):
    """Resolve the archived named intent without choosing among duplicates."""
    fingers = curve.finger_trajectory
    if fingers is None:
        raise ValueError("The source archive needs its validated analytic finger reference")
    if tuple(limits.finger_joint_names) != tuple(fingers.joint_names):
        raise ValueError("Source physical limits and analytic reference names disagree")
    # The validated analytic finger archive is the sole command source. It
    # already contains every original event and its named physical target;
    # a later optional device rebinding is neither required nor consulted.
    targets = {(row["skill_id"], row["event_id"], row["occurrence"]): row
               for row in fingers.events}
    states = {(row["skill_id"], row["event_id"]): row for row in limits.events}
    chosen, protection = {}, {}
    for event in curve.events:
        kind = event["kind"]
        if kind not in ("open", "close"):
            continue
        identity = (event["skill_id"], event["event_id"])
        target = targets[(*identity, event["occurrence"])]
        if any(target[key] != event[key] for key in ("kind", "time", "progress")):
            raise ValueError("Finger archive changed an original command identity or clock")
        value = np.asarray(target["finger_positions"], float)
        state = {key: deepcopy(states[identity][key]) for key in ("position_bounds", "maximum_speed")}
        if kind in chosen and (not np.array_equal(chosen[kind], value) or state != protection[kind]):
            raise ValueError("Repeated source intentions have incompatible targets or actual protection intervals")
        chosen[kind], protection[kind] = value, state
    if set(chosen) != {"open", "close"}:
        raise ValueError("Both explicit original open and close intentions are required")
    return chosen, protection


def arm_reference(curve, case, elapsed):
    moment = case["source_time_s"]+(elapsed if case["mode"] == "dynamic" else 0.)
    q, dq, ddq, _ = curve.sample(min(moment, curve.duration))
    if case["mode"] == "static":
        dq, ddq = np.zeros_like(q), np.zeros_like(q)
    return q, dq, ddq


def make_finger_reference(source, case, candidate, initial_q, initial_dq, dt, timestamp):
    duration = NORMAL_STEPS*dt
    commands = [{"skill_id": "device_calibration", "event_id": str(index), "occurrence": 0,
                 "kind": kind, "time": index*duration/2, "progress": float(index),
                 "finger_positions": candidate["targets"][kind].tolist()}
                for index, kind in enumerate(case["order"])]
    return FingerTrajectory.build(joint_names=source.joint_names,
        initial_position=initial_q, initial_velocity=initial_dq, commands=commands,
        duration=duration, executable_until=duration, timestep=dt,
        position_limits=source.position_limits, velocity_limits=source.velocity_limits,
        acceleration_limits=source.acceleration_limits,
        source="Fresh unloaded device calibration; original named reference bounds retained",
        feedback_id="synthetic_bench_initialization_from_named_finger_feedback",
        timestamp_s=timestamp)


def bench_case(backend, curve, case, fingers, limits, protection, materials, ledger, output):
    """Run one frozen case and always retain its separately bounded hold."""
    ledger.reserve_case()
    narm = len(backend.arm_joint_names)
    ranges = backend.joint_ranges
    safety = {"joint_ranges": ranges, "maximum_speed": np.asarray([backend.velocity_limits[name] for name in backend.joint_names])}
    begin_time = backend.sim_time
    phase, current_control = "normal", None
    active_kind = case["order"][0]
    states, controls, capture_errors, hold_faults, endpoints = [], [], [], [], []
    contacts = SelfContactEvidence()
    primary = None
    physical_before = ledger.describe()
    initial = backend.proprioception()
    states.append({"phase": "initial", "feedback": initial, "gap": material_gaps(materials, initial["q"][narm:])})
    initial_contacts, overflow = native_self_contacts(backend, MAX_CONTACT_ROWS_PER_SAMPLE)
    contacts.append(initial_contacts, state_index=0, sim_time=initial["sim_time"], phase="initial", overflow=overflow)
    native_step, native_apply = backend.step, backend.apply_control

    def apply(q, dq, ddq, finger_q):
        nonlocal current_control
        value = native_apply(q, dq, ddq, finger_q)
        current_control = {"phase": phase, "sim_time": backend.sim_time,
                           "finger_reference_q": np.asarray(finger_q).copy(), **deepcopy(value)}
        controls.append(current_control)
        return value

    def guard(feedback, elapsed, *, hold=False):
        q, dq, _ = arm_reference(curve, case, min(elapsed, fingers.duration))
        index = 0 if elapsed <= fingers.duration/2 else 1
        failures = TrajectoryExecutor._execution_violations(SimpleNamespace(backend=backend),
            feedback, q, dq, backend.reference_tool_pose(q), limits=limits,
            binding=safety, finger_state=protection[active_kind if hold else case["order"][index]],
            expected_sim_time=begin_time+elapsed)
        if hold:
            # Stopping intentionally changes the arm reference. Native state,
            # clock, identity, speed and actual finger protections still apply.
            failures = [row for row in failures if row["kind"] not in
                        ("arm_position_tracking", "arm_velocity_tracking", "tcp_tracking")]
        return failures

    def capture(advanced, before, after):
        if not advanced:
            capture_errors.append({"phase": phase, "before": before, "after": after,
                                   "reason": "Native step not confirmed by its clock"})
            return
        try:
            actual = backend.proprioception()
            states.append({"phase": phase, "feedback": actual,
                           "gap": material_gaps(materials, actual["q"][narm:])})
            contact_rows, overflow = native_self_contacts(backend, MAX_CONTACT_ROWS_PER_SAMPLE)
            contacts.append(contact_rows, state_index=len(states)-1, sim_time=after, phase=phase, overflow=overflow)
            if phase == "hold":
                failures = guard(actual, after-begin_time, hold=True)
                if failures:
                    hold_faults.append({"sim_time": after, "violations": failures})
        except Exception as error:
            capture_errors.append({"phase": phase, "before": before, "after": after,
                                   "reason": f"{type(error).__name__}: {error}"})
            raise
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024 > MAX_RSS_MIB:
            raise RuntimeError("Device bench exceeded its 4 GiB RSS admission bound")

    backend.apply_control = apply
    backend.step = lambda: ledger.step(backend, native_step, capture)
    normal_steps = 0
    try:
        entry_faults = guard(initial, 0.)
        if entry_faults:
            raise ValueError(f"Bench entry feedback rejected: {json_value(entry_faults)}")
        for index in range(NORMAL_STEPS):
            elapsed = (index+1)*backend.timestep
            active_kind = case["order"][0 if elapsed <= fingers.duration/2 else 1]
            ledger.context = {"case": case["name"], "phase": phase, "normal_step": index+1}
            q, dq, ddq = arm_reference(curve, case, elapsed)
            fq, _ = fingers.sample(elapsed)
            backend.apply_control(q, dq, ddq, fq)
            backend.step()
            normal_steps += 1
            actual = states[-1]["feedback"]
            failures = guard(actual, elapsed)
            if contacts.overflow:
                primary = {"phase": "recording", "step": index+1,
                           "reason": "Bounded independent self-contact evidence storage exhausted"}
                break
            if failures:
                primary = {"phase": "normal", "step": index+1, "sim_time": actual["sim_time"], "violations": failures}
                break
            if index+1 in (NORMAL_STEPS//2, NORMAL_STEPS):
                speed = np.abs(actual["dq"][narm:])
                settled = bool(np.all(speed <= np.asarray(limits.stop_finger_velocity)))
                # A moving arm may induce legitimate finger motion inside
                # the original actual-speed interval. The stop criterion is
                # a static steady-state diagnostic and a true hold criterion,
                # never an extra dynamic tracking tolerance.
                passed = settled or case["mode"] == "dynamic"
                endpoints.append({"kind": case["order"][0 if index+1 == NORMAL_STEPS//2 else 1],
                    "step": index+1, "actual_q": actual["q"][narm:], "actual_dq": actual["dq"][narm:],
                    "actual_material_pad_gap_m": material_gaps(materials, actual["q"][narm:]),
                    "nominal_material_pad_gap_m": material_gaps(materials, fq),
                    "maximum_stop_speed": limits.stop_finger_velocity, "settled": settled,
                    "static_settling_required": case["mode"] == "static", "passed": passed})
                if not passed:
                    primary = {"phase": "endpoint", "step": index+1, "reason": "Finger speed did not meet original stop criteria"}
                    break
    except Exception as error:
        if primary is None:
            primary = {"phase": phase, "reason": f"{type(error).__name__}: {error}"}
    finally:
        before_hold = ledger.describe()
        phase = "hold"
        ledger.context = {"case": case["name"], "phase": phase, "completed_normal_steps": normal_steps}
        try:
            actual = backend.proprioception()
            hold = backend.abort_hold(actual, backend.finger_control.copy(), expected_sim_time=backend.sim_time,
                stop_velocity=np.r_[limits.stop_arm_velocity, limits.stop_finger_velocity], max_steps=HOLD_STEPS)
        except Exception as error:
            hold = {"physical_stop_confirmed": False, "reason": f"{type(error).__name__}: {error}"}
        backend.step, backend.apply_control = native_step, native_apply
    final_ledger = ledger.describe()
    confirmed = final_ledger["confirmed_physics_steps"]-physical_before["confirmed_physics_steps"]
    charged = final_ledger["charged_steps"]-physical_before["charged_steps"]
    normal_confirmed = before_hold["confirmed_physics_steps"]-physical_before["confirmed_physics_steps"]
    hold_confirmed = final_ledger["confirmed_physics_steps"]-before_hold["confirmed_physics_steps"]
    actual_q = np.asarray([row["feedback"]["q"] for row in states])
    actual_dq = np.asarray([row["feedback"]["dq"] for row in states])
    force = np.asarray([row["feedback"]["actuator_force"] for row in states])
    actual_control = np.asarray([row["feedback"]["actuator_ctrl"]["values"] for row in states])
    finger_error = actual_q[:, narm:]-actual_control[:, narm:]
    control_array = np.asarray([row["actuator_ctrl"] for row in controls])
    summary = {"case": case, "primary_failure": primary, "normal_steps": normal_confirmed,
        "normal_completed_loop_iterations": normal_steps, "hold_confirmed_physics_steps": hold_confirmed,
        "normal_charged_steps": before_hold["charged_steps"]-physical_before["charged_steps"],
        "hold_charged_steps": final_ledger["charged_steps"]-before_hold["charged_steps"],
        "charged_steps": charged, "confirmed_physics_steps": confirmed,
        "capture_errors": capture_errors, "hold_faults": hold_faults, "endpoints": endpoints,
        "hold": {key: value for key, value in hold.items() if key not in ("samples", "controls")},
        "finger_q_min": actual_q[:, narm:].min(axis=0), "finger_q_max": actual_q[:, narm:].max(axis=0),
        "finger_min_lower_margin_m": (actual_q[:, narm:]-ranges[narm:, 0]).min(axis=0),
        "finger_min_upper_margin_m": (ranges[narm:, 1]-actual_q[:, narm:]).min(axis=0),
        "finger_max_signed_upper_error_m": (actual_q[:, narm:]-ranges[narm:, 1]).max(axis=0),
        "finger_max_absolute_speed": np.abs(actual_dq[:, narm:]).max(axis=0),
        "finger_final_velocity": actual_dq[-1, narm:], "maximum_absolute_actuator_force": np.abs(force).max(axis=0),
        "finger_actual_minus_control_min_m": finger_error.min(axis=0),
        "finger_actual_minus_control_max_m": finger_error.max(axis=0),
        "arm_torque_saturated": any(bool(np.any(row["arm_torque_saturated"])) for row in controls),
        "measured_pad_gap_min_m": np.asarray([row["gap"] for row in states]).min(axis=0),
        "measured_pad_gap_max_m": np.asarray([row["gap"] for row in states]).max(axis=0),
        "joint_names": backend.joint_names, "actuator_names": backend.actuator_names,
        "self_contact": contacts.describe(),
        "bench_feasible": bool(primary is None and normal_steps == NORMAL_STEPS and len(endpoints) == 2
            and hold.get("physical_stop_confirmed") is True and not hold_faults and not capture_errors
            and not contacts.active and not contacts.overflow and not contacts.nonfinite
            and final_ledger["exact_clock_accounting"] and confirmed == len(states)-1)}
    arrays = {key: np.asarray([row["feedback"][key] for row in states]) for key in ("q", "dq", "qacc", "actuator_force", "tool_pose")}
    arrays.update(actual_phase=np.asarray([row["phase"] for row in states]),
        sim_time=np.asarray([row["feedback"]["sim_time"] for row in states]),
        measured_pad_gap=np.asarray([row["gap"] for row in states]),
        command_phase=np.asarray([row["phase"] for row in controls]),
        control_sim_time=np.asarray([row["sim_time"] for row in controls]), actuator_ctrl=control_array,
        actual_actuator_ctrl=actual_control,
        joint_names=np.asarray(backend.joint_names), actuator_names=np.asarray(backend.actuator_names))
    for key in (*TORQUE_FIELDS, "reference_arm_q", "reference_arm_dq", "reference_arm_ddq", "arm_torque_saturated", "finger_reference_q"):
        arrays[key] = np.asarray([row[key] for row in controls])
    arrays["nominal_pad_gap"] = np.asarray([material_gaps(materials, row["finger_reference_q"]) for row in controls])
    arrays.update(self_contact_count=np.asarray(contacts.sample_counts),
                  self_contact_active_count=np.asarray(contacts.active_counts),
                  self_contact_maximum_absolute_force=np.asarray(contacts.maximum_forces))
    persistence_errors = []
    # Artifact sinks are independent; one failure cannot hide the primary
    # physical failure or prevent attempting the other compact evidence file.
    try:
        np.savez_compressed(output.with_suffix(".npz"), **arrays)
    except Exception as error:
        persistence_errors.append(f"npz: {type(error).__name__}: {error}")
    summary["persistence_errors"] = persistence_errors
    if persistence_errors:
        summary["bench_feasible"] = False
    try:
        write_json(output.with_suffix(".json"), summary)
    except Exception as error:
        persistence_errors.append(f"json: {type(error).__name__}: {error}")
        summary["bench_feasible"] = False
    return summary


def run(args):
    import mujoco
    from robosuite.utils.binding_utils import MjSim
    from manipulation.contact_binding import calibrate_box_material_features
    from manipulation.robosuite_execution import RobosuiteExecutionBackend

    args.output.mkdir(parents=True, exist_ok=False)
    ledger = StepLedger(journal=args.output/"step-ledger.jsonl")
    report = {"scope": "Fresh unloaded robot device bench; no task, contact or Holding acceptance",
              "gpu_used": False, "renderer_used": False, "solver_calls": 0,
              "selected_candidate": None, "candidates": [], "primary_failure": None,
              "uncovered": ["external contact loads", "continuous motion extrema", "task geometric feasibility", "Holding", "unobserved motions"]}
    try:
        fixture = json.loads(args.fixture.read_text())
        feedback = json.loads(args.feedback.read_text())
        manifest = json.loads(args.controller_manifest.read_text())
        if args.controller_manifest.resolve().parent != Path(fixture["episode_path"]).resolve().parent:
            raise ValueError("Controller manifest must belong to the explicit fixture source")
        part = manifest["environment"]["controller_config"]["body_parts"][args.controller_part]
        result = load_saved_trajectory(args.trajectory)
        binding = check_result(result)
        curve = result.trajectory
        limits = ExecutionLimits.from_mapping(binding["execution_limits"])
        names = tuple(fixture["arm_joint_names"])+tuple(fixture["finger_joint_names"])
        narm = len(fixture["arm_joint_names"])
        if (tuple(feedback["joint_names"]) != names or tuple(curve.joint_names) != names[:narm]
                or tuple(limits.joint_names) != names or feedback["reference_frame"] != "world"):
            raise ValueError("Fixture, archive and named source feedback must agree exactly")
        initial_q, initial_dq = np.asarray(feedback["start_q"], float), np.asarray(feedback["start_dq"], float)
        if (initial_q.shape != (len(names),) or initial_dq.shape != initial_q.shape
                or not np.isfinite(initial_q).all() or not np.isfinite(initial_dq).all()):
            raise ValueError("Complete finite named initial q/dq are required")
        source_path = Path(fixture["source_scene_xml"]).resolve()
        xml, native_names = robot_xml(source_path.read_text(), base_link=fixture["base_link"], joint_names=names, source_dir=source_path.parent)
        (args.output/"robot-only.xml").write_text(xml)
        original = mujoco.MjModel.from_xml_path(str(source_path))
        model = mujoco.MjModel.from_xml_string(xml)
        mechanics = mechanical_equivalence(original, model, native_names, names)
        del original
        gc.collect()
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024 > MAX_RSS_MIB:
            raise RuntimeError("Model preparation exceeded the 4 GiB RSS admission bound")
        joints = np.asarray([model.joint(name).id for name in names])
        qids, vids = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
        actuator_ids = np.asarray([int(np.flatnonzero(model.actuator_trnid[:, 0] == joint)[0]) for joint in joints])
        ranges = model.jnt_range[joints]
        if (np.any(initial_q < ranges[:, 0]) or np.any(initial_q > ranges[:, 1])
                or np.any(initial_q[narm:] < model.actuator_ctrlrange[actuator_ids[narm:], 0])
                or np.any(initial_q[narm:] > model.actuator_ctrlrange[actuator_ids[narm:], 1])):
            raise ValueError("Named initial state must lie inside unchanged native domains")
        limits.bind(curve, joint_names=names, arm_joint_names=names[:narm], finger_joint_names=names[narm:],
            native_joint_ranges=ranges, named_velocity_limits=fixture["velocity_limits"])
        original_targets, protection = source_commands(curve, limits)
        candidates = candidate_targets(original_targets, ranges[narm:], args.candidate_fractions)
        # Read only robot material identities from this input. The contacts
        # declaration and its object geometry never enter this bench path.
        material_faces = json.loads(args.contact_bindings.read_text())["material_faces"]
        materials = calibrate_box_material_features(model, tcp_site=fixture["tcp_site"],
            joint_positions=dict(zip(names, initial_q)), finger_joint_names=names[narm:], material_faces=material_faces)
        cases, scan, scan_arrays = scan_reference(model, curve, fixture, initial_q[narm:])
        dt = float(model.opt.timestep)
        prepared = []
        for candidate in candidates:
            rows = []
            for case in cases:
                fingers = make_finger_reference(curve.finger_trajectory, case, candidate,
                    initial_q[narm:], initial_dq[narm:], dt, 0.)
                q, dq, ddq = arm_reference(curve, case, 0.)
                rows.append({"case": case, "fingers": fingers, "initial_q": np.r_[q, initial_q[narm:]],
                             "initial_dq": np.r_[dq, initial_dq[narm:]], "initial_arm_ddq": ddq})
            prepared.append(rows)
        setup = {"source_fixture": str(args.fixture.resolve()), "source_trajectory": str(args.trajectory.resolve()),
            "source_feedback": str(args.feedback.resolve()), "source_candidate_id": binding["candidate_id"],
            "mechanical_equivalence": mechanics, "source_limits": limits.describe(),
            "controller": {"kp": part["kp"], "kd": part["kd"], "part": args.controller_part},
            "candidates": candidates, "original_targets": original_targets,
            "materials": materials, "scan": scan, "timestep_s": dt,
            "normal_steps_per_case": NORMAL_STEPS, "hold_steps_per_case_maximum": HOLD_STEPS,
            "self_contact_evaluation": "Independent native robot-only contacts saved after each step; any active efc_address rejects unloaded no-self-contact scope after the case; no contacts feed control, target construction or adaptive target changes",
            "self_contact_pairs_per_case_cap": MAX_CONTACT_PAIRS_PER_CASE,
            "self_contact_rows_per_sample_cap": MAX_CONTACT_ROWS_PER_SAMPLE,
            "shared_step_limit": MAX_STEPS, "maximum_declared_case_reservations": len(candidates)*len(cases)*(NORMAL_STEPS+HOLD_STEPS),
            "selection_rule": "Ascending declared fractions; first candidate whose three cases pass; skip remaining cases after first failure; no adaptive target changes",
            "endpoint_acceptance": "Static endpoints require original finger stop speeds; dynamic endpoints retain original actual position/speed protection without an added stop criterion; every final abort_hold requires all original stop speeds",
            "fresh_setup": "New robot-only MjData per case; static arm dq=0 is synthetic; dynamic arm q/dq follow source; all finger q/dq retain named feedback; initial arm ctrl=0 and finger ctrl=actual initial finger q; no recorded world state or continuation claimed",
            "scope": report["scope"], "uncovered": report["uncovered"],
            "cases": [[{**{key: value for key, value in row.items() if key != "fingers"},
                        "finger_reference": row["fingers"].describe()} for row in rows] for rows in prepared]}
        for candidate in setup["candidates"]:
            candidate["nominal_pad_gap_m"] = {kind: material_gaps(materials, value) for kind, value in candidate["targets"].items()}
        setup["original_nominal_pad_gap_m"] = {kind: material_gaps(materials, value) for kind, value in original_targets.items()}
        np.savez_compressed(args.output/"reference-scan.npz", **scan_arrays)
        write_json(args.output/"setup.json", setup)
        # No native step occurs above this line. The written setup is the
        # immutable selection/initialization contract for every case below.
        for candidate_index, (candidate, rows) in enumerate(zip(candidates, prepared)):
            candidate_report = {"candidate": candidate, "cases": [], "bench_feasible": False}
            report["candidates"].append(candidate_report)
            for row in rows:
                ledger.reserve_case()
                sim = MjSim(model)
                data = sim.data._data
                data.qpos[qids], data.qvel[vids] = row["initial_q"], row["initial_dq"]
                data.ctrl[actuator_ids[:narm]] = 0.
                data.ctrl[actuator_ids[narm:]] = initial_q[narm:]
                mujoco.mj_forward(model, data)
                backend = RobosuiteExecutionBackend(sim, base_link=fixture["base_link"], tcp_site=fixture["tcp_site"],
                    arm_joint_names=names[:narm], finger_joint_names=names[narm:], arm_kp=part["kp"], arm_kd=part["kd"],
                    velocity_limits=fixture["velocity_limits"], source="Fresh isolated robot command-domain device bench")
                case_report = bench_case(backend, curve, row["case"], row["fingers"], limits, protection, materials,
                    ledger, args.output/f"candidate-{candidate_index+1}-{row['case']['name']}")
                candidate_report["cases"].append(case_report)
                report["ledger"] = ledger.describe()
                write_json(args.output/"report.json", report)
                del backend, sim, data
                gc.collect()
                if not case_report["bench_feasible"]:
                    break
            candidate_report["bench_feasible"] = (len(candidate_report["cases"]) == len(cases)
                and all(row["bench_feasible"] for row in candidate_report["cases"]))
            if candidate_report["bench_feasible"]:
                report["selected_candidate"] = candidate
                break
    except Exception as error:
        report["primary_failure"] = f"{type(error).__name__}: {error}"
    finally:
        report["ledger"] = ledger.describe()
        report["peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        report["status"] = "bench_candidate_found" if report["selected_candidate"] is not None else "no_bench_candidate"
        try:
            write_json(args.output/"report.json", report)
        except Exception as error:
            print(json.dumps({"primary_failure": report["primary_failure"], "report_sink_error": str(error), "ledger": ledger.describe()}), flush=True)
            raise
    print(json.dumps({"status": report["status"], "ledger": ledger.describe(), "primary_failure": report["primary_failure"]}), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("fixture", "trajectory", "feedback", "controller-manifest", "contact-bindings", "output"):
        parser.add_argument("--"+name, required=True, type=Path)
    parser.add_argument("--controller-part", required=True)
    parser.add_argument("--candidate-fractions", required=True, nargs="+", type=float)
    run(parser.parse_args())
# END bounded command-domain evidence; no formal task binding is modified.
