"""Explicit simulated two-pad tactile feedback for a bounded physical trial.

The adapter reads current native contact measurements and named robot feedback.
It never reads a target body's pose, joints, recording history, or evaluator
grade. Its positive evidence is limited to stable tactile pinching: it does
not establish an observed rigid attachment or certify a later loaded motion.
"""

# BEGIN declared simulated tactile sensing, independent of offline grading.
from copy import deepcopy
from collections import deque

import mujoco
import numpy as np


_FIELDS = {"source", "owner_id", "effect", "target_geometry_names", "contact_groups",
    "finger_joint_names", "stable_window_s", "maximum_sample_gap_s", "minimum_normal_force_n",
    "maximum_contact_point_span_m", "maximum_contact_normal_span_rad", "maximum_opposition_error_rad",
    "maximum_finger_position_span", "maximum_contacts_per_sample"}


def _names(value, name):
    if (not isinstance(value, (list, tuple)) or not value
            or any(not isinstance(item, str) or not item.strip() for item in value)
            or len(set(value)) != len(value)):
        raise ValueError(f"{name} needs explicit unique nonempty names")
    return tuple(value)


def _sampled_span_bounds(points):
    """Bound every measured pair and retain real counterexample distances.

    The Cartesian range diagonal upper-bounds the complete diameter. The six
    coordinate extrema are actual samples, so their fixed-size pair matrix
    supplies a lower bound with genuine witnesses. Linear work in the window
    length avoids an unbounded all-sample distance matrix. A gap between these
    bounds is uncertainty, not a positive certificate or a known violation.
    """
    upper = np.linalg.norm(np.ptp(points, axis=0), axis=-1)
    lower = []
    for group in range(points.shape[1]):
        samples = points[:, group]
        indices = np.r_[samples.argmin(axis=0), samples.argmax(axis=0)]
        extrema = samples[indices]
        lower.append(float(np.linalg.norm(extrema[:, None]-extrema[None], axis=-1).max()))
    return np.asarray(lower), upper


class SimulatedTactileObserver:
    """Measure a declared owner with two independently identified tactile pads.

    All thresholds and native identities are test inputs. Normal force,
    force-weighted contact centroids, and force-weighted inward normals are
    measured in the *actual* TCP frame. A complete current window must pass
    every sample's contact/force/opposition checks and the declared finger,
    centroid and normal stability bounds. This sensor cannot observe a full
    object-relative pose, so it supplies no owner pose or attachment fact.
    """

    def __init__(self, simulation, configuration, *, execution_binding, observation_policy, maximum_samples):
        if not isinstance(configuration, dict) or set(configuration) != _FIELDS:
            raise ValueError("The tactile sensor needs its complete explicit input contract")
        config = deepcopy(configuration)
        if any(not isinstance(config[key], str) or not config[key].strip() for key in ("source", "owner_id")):
            raise ValueError("The tactile sensor needs a current owner identity and source")
        if (type(maximum_samples) is not int or not 1 <= maximum_samples <= 100_001
                or type(config["maximum_contacts_per_sample"]) is not int
                or not 1 <= config["maximum_contacts_per_sample"] <= 4096):
            raise ValueError("Tactile retention/query budgets must be finite and explicitly bounded")
        scalars = ("stable_window_s", "maximum_sample_gap_s", "maximum_contact_point_span_m",
                   "maximum_contact_normal_span_rad", "maximum_opposition_error_rad")
        if any(isinstance(config[key], bool) or not isinstance(config[key], (int, float))
               or not np.isfinite(config[key]) or config[key] <= 0 for key in scalars):
            raise ValueError("Tactile windows and acceptance thresholds must be finite positive values")
        if (config["maximum_contact_normal_span_rad"] >= np.pi
                or config["maximum_opposition_error_rad"] >= np.pi/2):
            raise ValueError("Stable tactile normals must be bounded and opposing")
        if (observation_policy is None
                or observation_policy["period_s"] > config["maximum_sample_gap_s"]
                or config["maximum_sample_gap_s"] > config["stable_window_s"]):
            raise ValueError("The declared observation cadence must resolve the tactile stability window")
        effect = config["effect"]
        if (not isinstance(effect, dict) or set(effect) != {"condition_id", "skill_id", "event_id"}
                or any(not isinstance(value, str) or not value.strip() for value in effect.values())):
            raise ValueError("Tactile evidence must bind the exact original effect condition/event")
        if execution_binding["problem"].get("reference_frame") != "world":
            raise ValueError("Native tactile positions require the explicitly bound MuJoCo world reference frame")
        self._tool_link = execution_binding["problem"]["tool_link"]
        conditions = [row for row in execution_binding["problem"].get("conditions", ())
                      if all(row.get(key) == value for key, value in effect.items())]
        if (len(conditions) != 1 or conditions[0].get("carrier_id") != config["owner_id"]
                or conditions[0].get("effect_requirement") != "stable_grasp_from_current_synchronized_evidence"):
            raise ValueError("Tactile pinching must match one declared current stable-grasp effect")
        scene = execution_binding["scene"]
        if sum(row["owner_id"] == config["owner_id"] for row in scene["owners"]) != 1:
            raise ValueError("Tactile target identity must match the bound current collision owner")
        model = simulation.native_model
        target_names = _names(config["target_geometry_names"], "Target tactile geometry")
        self._target_ids = {int(model.geom(name).id) for name in target_names}
        target_bodies = {int(model.geom_bodyid[index]) for index in self._target_ids}
        if len(target_bodies) != 1:
            raise ValueError("This rigid-owner tactile sensor needs one explicit native target body")
        groups = config["contact_groups"]
        if not isinstance(groups, list) or len(groups) != 2:
            raise ValueError("This device adapter needs its two explicit opposing tactile-pad groups")
        self._group_ids, material_ids = [], []
        for group in groups:
            if (not isinstance(group, dict) or set(group) != {"material_id", "native_geometry_names"}
                    or not isinstance(group["material_id"], str) or not group["material_id"].strip()):
                raise ValueError("Each tactile group needs a material identity and native sensor geometries")
            names = _names(group["native_geometry_names"], "Tactile-pad geometry")
            if any(scene["native_geometry_materials"].get(name) != group["material_id"] for name in names):
                raise ValueError("Tactile pads and independent collision materials must use identical identities")
            if not any(row["actor"] == "robot" and row["material_id"] == group["material_id"]
                       and row["target_owner"] == config["owner_id"] for row in scene["contacts"]):
                raise ValueError("Each tactile pad must retain its separately declared finite owner-contact rule")
            ids = {int(model.geom(name).id) for name in names}
            if (ids & self._target_ids or any(ids & earlier for earlier in self._group_ids)
                    or any(int(model.geom_bodyid[index]) in target_bodies for index in ids)):
                raise ValueError("Target and independent tactile-pad identities must be disjoint")
            self._group_ids.append(ids)
            material_ids.append(group["material_id"])
        if len(set(material_ids)) != len(material_ids):
            raise ValueError("Independent tactile pads must have distinct material identities")
        self._finger_names = _names(config["finger_joint_names"], "Measured tactile finger coordinates")
        if self._finger_names != tuple(simulation.finger_joint_names):
            raise ValueError("Tactile finger names/order must match actual complete robot feedback")
        self._force_threshold = np.asarray(config["minimum_normal_force_n"], dtype=float)
        self._finger_threshold = np.asarray(config["maximum_finger_position_span"], dtype=float)
        if (self._force_threshold.shape != (2,) or self._finger_threshold.shape != (len(self._finger_names),)
                or not np.isfinite(self._force_threshold).all() or np.any(self._force_threshold <= 0)
                or not np.isfinite(self._finger_threshold).all() or np.any(self._finger_threshold <= 0)):
            raise ValueError("Each tactile group/finger needs its explicit positive force/stability threshold")
        self._simulation, self.configuration = simulation, config
        self._maximum_samples, self._samples = maximum_samples, []
        self._window = deque()
        self._latest = None
        self._policy = deepcopy(observation_policy)

    def describe(self):
        return {"kind": "explicit_simulated_two_pad_tactile_sensor", "configuration": deepcopy(self.configuration),
            "observation_policy": deepcopy(self._policy), "maximum_retained_samples": self._maximum_samples,
            "allowed_native_fields": ["contact.geom", "contact.exclude", "contact.efc_address", "contact.pos",
                "contact.frame", "mj_contactForce", "native acquisition time", "named robot q/TCP feedback"],
            "scope": "Current stable tactile pinching on the declared owner; no lift/load or full object-relative pose certificate",
            "owner_pose_read": False, "attachment_observation_provided": False,
            "offline_evaluator_consumed": False,
            "normal_convention": "MuJoCo frame normal geom0 to geom1, oriented from each actor pad toward target"}

    @property
    def observations(self):
        return deepcopy(self._samples)

    def __call__(self, feedback, curve_time):
        """Acquire only the current sensor state; never reuse a historical T."""
        data = self._simulation.native_data
        now = float(feedback["timestamp_s"])
        if (not np.isfinite(now) or abs(now-float(data.time)) > 1e-8
                or tuple(feedback["finger_joint_names"]) != self._finger_names
                or feedback.get("reference_frame") != "world" or feedback.get("tool_link") != self._tool_link):
            raise ValueError("Tactile acquisition and actual robot feedback must share one current timestamp and device")
        if self._samples and now < self._samples[-1]["timestamp_s"]-1e-8:
            raise ValueError("A tactile sensor cannot move its acquisition clock backward")
        if self._samples and now == self._samples[-1]["timestamp_s"]:
            return deepcopy(self._latest)
        if len(self._samples) >= self._maximum_samples or data.ncon > self.configuration["maximum_contacts_per_sample"]:
            raise ValueError("The declared tactile acquisition budget was exhausted; no contacts are truncated")
        names = tuple(feedback["joint_names"])
        fingers = np.asarray(feedback["q"], dtype=float)[[names.index(name) for name in self._finger_names]]
        tcp = np.asarray(feedback["tool_pose"], dtype=float)
        if tcp.shape != (4, 4) or not np.isfinite(tcp).all() or not np.isfinite(fingers).all():
            raise ValueError("Actual tactile robot coordinates must be finite")
        force = np.zeros(2)
        points, normals = np.zeros((2, 3)), np.zeros((2, 3))
        counts = np.zeros(2, dtype=int)
        for index in range(data.ncon):
            contact = data.contact[index]
            first, second = int(contact.geom1), int(contact.geom2)
            if contact.exclude != 0 or contact.efc_address < 0:
                continue
            for group, ids in enumerate(self._group_ids):
                orientation = (1. if first in ids and second in self._target_ids else
                               -1. if second in ids and first in self._target_ids else 0.)
                if orientation == 0:
                    continue
                measured = np.zeros(6)
                mujoco.mj_contactForce(self._simulation.native_model, data, index, measured)
                if not np.isfinite(measured).all() or not np.isfinite(contact.pos).all() or not np.isfinite(contact.frame).all():
                    raise ValueError("Actual tactile contact measurement became nonfinite")
                weight = max(0., float(measured[0]))
                force[group] += weight
                points[group] += weight*((np.asarray(contact.pos)-tcp[:3, 3]) @ tcp[:3, :3])
                normals[group] += weight*(orientation*np.asarray(contact.frame[:3]) @ tcp[:3, :3])
                counts[group] += 1
        directions = [None, None]
        centroids = [None, None]
        for group in range(2):
            if force[group] > 0:
                centroids[group] = (points[group]/force[group]).tolist()
                length = float(np.linalg.norm(normals[group]))
                if length > np.finfo(float).eps*force[group]:
                    directions[group] = (normals[group]/length).tolist()
        opposition = (None if any(row is None for row in directions) else
            float(np.arccos(np.clip(-np.dot(*np.asarray(directions)), -1., 1.))))
        row = {"timestamp_s": now, "curve_time_s": float(curve_time), "finger_positions": fingers.tolist(),
            "normal_force_n": force.tolist(), "contact_count": counts.tolist(),
            "contact_centroid_tcp_m": centroids, "inward_normal_tcp": directions,
            "opposition_error_rad": opposition, "feedback_id": feedback["feedback_id"]}
        self._samples.append(row)
        self._window.append(row)
        lower = now-self.configuration["stable_window_s"]
        # Retain the sample just before a non-grid window beginning. Checking
        # its complete measured interval is conservative; trimming it away
        # would make a 0.2 s window impossible on a 0.003 s sensor cadence.
        while len(self._window) > 1 and self._window[1]["timestamp_s"] <= lower+1e-9:
            self._window.popleft()
        window = list(self._window)
        times = np.asarray([sample["timestamp_s"] for sample in window])
        complete = times[-1]-times[0] >= self.configuration["stable_window_s"]-1e-9
        continuous = len(times) < 2 or np.max(np.diff(times)) <= self.configuration["maximum_sample_gap_s"]+1e-9
        false, unknown = [], []
        # Current loss is an actual counterexample. Missing observation history
        # remains U, regardless of how close the reference is to its endpoint.
        if np.any(force < self._force_threshold):
            false.append("Current measured normal force is below a declared pad threshold")
        if opposition is not None and opposition > self.configuration["maximum_opposition_error_rad"]:
            false.append("Current measured pad normals are not opposing within the declared bound")
        if not complete:
            unknown.append("No complete current tactile stability window")
        if not continuous:
            unknown.append("The current tactile window contains an acquisition gap")
        metrics = {"acquisition_interval_s": [float(times[0]), now], "sample_count": len(window),
            "maximum_sample_gap_s": 0. if len(times) < 2 else float(np.max(np.diff(times))),
            "complete_window": complete, "continuous_sampling": continuous}
        if complete and continuous:
            if any(np.any(np.asarray(sample["normal_force_n"]) < self._force_threshold) for sample in window):
                false.append("The current window contains an observed loss of bilateral force")
            if any(sample["opposition_error_rad"] is None for sample in window):
                unknown.append("A force-bearing tactile normal is unresolved in the current window")
            elif any(sample["opposition_error_rad"] > self.configuration["maximum_opposition_error_rad"] for sample in window):
                false.append("The current window contains non-opposing tactile normals")
            spans = np.ptp(np.asarray([sample["finger_positions"] for sample in window]), axis=0)
            metrics["finger_position_spans"] = spans.tolist()
            if np.any(spans > self._finger_threshold):
                false.append("Actual finger movement exceeds its declared stability threshold")
            if all(all(value is not None for value in sample["contact_centroid_tcp_m"]) for sample in window):
                centroids = np.asarray([sample["contact_centroid_tcp_m"] for sample in window])
                witness, bound = _sampled_span_bounds(centroids)
                metrics["contact_point_witness_spans_m"] = witness.tolist()
                metrics["contact_point_span_upper_bounds_m"] = bound.tolist()
                if np.any(witness > self.configuration["maximum_contact_point_span_m"]):
                    false.append("Actual TCP-frame tactile point movement exceeds its declared stability threshold")
                elif np.any(bound > self.configuration["maximum_contact_point_span_m"]):
                    unknown.append("The full tactile point-span bound does not establish the declared stability threshold")
            else:
                unknown.append("A current-window tactile contact point is unresolved")
            if all(all(value is not None for value in sample["inward_normal_tcp"]) for sample in window):
                normals = np.asarray([sample["inward_normal_tcp"] for sample in window])
                # The first sample alone can miss two later normals lying on
                # opposite sides of it. The complete Cartesian AABB diameter
                # bounds every measured unit-normal chord; its corresponding
                # spherical angle therefore certifies the whole window in
                # linear work. If that sufficient bound is loose, retain U
                # rather than claim T or allocate an unbounded pair matrix.
                chord_witness, chord_bound = _sampled_span_bounds(normals)
                span_witness = 2*np.arcsin(np.clip(chord_witness/2, 0., 1.))
                span_bound = 2*np.arcsin(np.clip(chord_bound/2, 0., 1.))
                metrics["contact_normal_witness_spans_rad"] = span_witness.tolist()
                metrics["contact_normal_span_upper_bounds_rad"] = span_bound.tolist()
                if np.any(span_witness > self.configuration["maximum_contact_normal_span_rad"]):
                    false.append("Actual TCP-frame tactile normal movement exceeds its declared stability threshold")
                elif np.any(span_bound > self.configuration["maximum_contact_normal_span_rad"]):
                    unknown.append("The full tactile normal-span bound does not establish the declared stability threshold")
            else:
                unknown.append("A current-window tactile normal is unresolved")
        value = "F" if false else "U" if unknown else "T"
        condition = self.configuration["effect"]["condition_id"]
        fact = {"value": value, "source": self.configuration["source"], "evidence_kind": "current_state",
            "owner_id": self.configuration["owner_id"], "sensor_kind": "simulated_two_pad_tactile",
            "scope": "Stable tactile pinching only; no full relative pose, rigid attachment, or loaded-motion proof",
            "window": metrics, "counterexamples": false, "unresolved": unknown}
        row["effect"] = deepcopy(fact)
        self._latest = {"source": self.configuration["source"], "timestamp_s": now, "effects": {condition: fact}}
        return deepcopy(self._latest)
# END simulated touch evidence never imports the independent task evaluator's truth.
