"""Named finger command references on the arm trajectory's physical clock.

The reference starts from actual measured q/dq and retains original command
identities. It predicts neither contact nor actual servo tracking. In
particular, a zero closing target is not a measured zero finger position.
"""

# BEGIN one saved C1 finger reference shared by planning and physical execution.
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import math

import numpy as np
from scipy.interpolate import CubicHermiteSpline, PPoly

# BEGIN one occurrence identity across planning, commands and saved polynomials.
from .planning_problem import event_occurrence_key
# END a repeated source event is not the same physical command occurrence.


_SCOPE = ("Named nominal command reference, not measured finger motion or a contact prediction; "
          "the full original duration retains every conditional command. The executable prefix "
          "limits current authorization; later commands require the original dependency evidence.")
_LIMITS_SCOPE = ("Continuous reference q/dq/ddq bounds; acceleration is the supplied native "
                 "planning bound, not a calibrated actual hardware acceleration allowance.")
_EVENT_FIELDS = {"activation_step", "applied_time_s", "application_delay_s", "in_executable_prefix", "superseded"}


def _text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must identify an explicit source")
    return value


def _array(value, shape, label):
    result = np.asarray(value, dtype=float)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{label} must have finite shape {shape}")
    return result.copy()


def _activation(time_s, timestep):
    """Select the first native control tick at or after the original event.

    The product is checked as well as the quotient: binary64 division must
    not round a slightly future event down to an earlier physical command.
    No physical or numerical early-application allowance is introduced.
    """
    step = int(math.ceil(time_s / timestep))
    # Division can round just above an integer even when that integer's
    # native tick product equals the supplied time exactly. Recover the
    # earliest valid product, then retain the strict no-early check below.
    while step > 0 and (step-1)*timestep >= time_s:
        step -= 1
    while step * timestep < time_s:
        step += 1
    return step, step * timestep


def _extrema(curve, names, limits):
    """Inspect all scalar polynomial extrema and both acceleration knot sides.

    SciPy supplies polynomial roots. Values at stored q/dq endpoints use the
    spline's endpoint constraints; ddq is affine on each cubic and needs both
    one-sided endpoints. Reported excess is never clipped by an allowance.
    """
    maxima = []
    rows = {}
    for column, name in enumerate(names):
        scalar = PPoly(curve.c[:, :, column], curve.x, extrapolate=False)
        joint = {}
        for order, label in enumerate(("position", "velocity", "acceleration")):
            if order < 2:
                roots = scalar.derivative(order + 1).roots(extrapolate=False)
                roots = roots[np.isfinite(roots) & (roots >= curve.x[0]) & (roots <= curve.x[-1])]
                times = np.unique(np.r_[curve.x, roots])
                values = scalar(times, order)
            else:
                width = np.diff(curve.x)
                values = np.stack((2 * scalar.c[1], 6 * scalar.c[0] * width + 2 * scalar.c[1]), axis=1).reshape(-1)
                times = np.stack((curve.x[:-1], curve.x[1:]), axis=1).reshape(-1)
            if not np.isfinite(values).all():
                raise ValueError("A finger reference has nonfinite analytic extrema")
            low, high = int(values.argmin()), int(values.argmax())
            lower, upper = limits[label][:, column]
            excess = max(0., float(lower - values[low]), float(values[high] - upper))
            joint[label] = {"minimum": float(values[low]), "maximum": float(values[high]),
                            "minimum_time_s": float(times[low]), "maximum_time_s": float(times[high]),
                            "lower_bound": float(lower), "upper_bound": float(upper),
                            "maximum_bound_excess": excess}
            maxima.append(excess)
        rows[name] = joint
    maximum = max(maxima, default=0.)
    return {"scope": _LIMITS_SCOPE, "joint_extrema": rows,
            "maximum_bound_excess": maximum, "bounds_satisfied": maximum == 0.}


def _segment(q0, dq0, goal, start, end, timestep, names, limits):
    """Find the first bounded Hermite ramp ending on the native control grid.

    This is a finite search over realizable control ticks, not a new dynamic
    model or a time-optimality claim. Every candidate uses SciPy's same cubic
    and exact scalar limits. The original deadline is also considered when
    it falls between ticks; failure never extends the source event/barrier.
    """
    if end <= start:
        raise ValueError("A finger command needs time before its original next event or evidence barrier")
    if np.array_equal(q0, goal) and np.all(dq0 == 0.):
        return start
    # A displacement and a change in velocity each need this much time even
    # before selecting a cubic. This lower bound only prunes impossible ticks.
    speeds = np.maximum(np.abs(limits["velocity"][0]), np.abs(limits["velocity"][1]))
    accelerations = np.maximum(np.abs(limits["acceleration"][0]), np.abs(limits["acceleration"][1]))
    required = max(float(np.max(np.abs(goal - q0) / speeds)),
                   float(np.max(np.abs(dq0) / accelerations)))
    # BEGIN admit the exact absolute endpoints later used by the saved spline.
    # Adding a relative duration to a nonzero start can change its binary64
    # width, so a locally admissible cubic need not remain admissible after
    # translation. Native endpoints come from the original integer tick
    # products; both admission and final construction use those same values.
    first, _ = _activation(start + required, timestep)
    # The lower-bound sum itself may round. Check represented differences
    # directly so this pruning cannot omit the earliest admissible tick.
    while first > 0 and (first-1)*timestep > start and (first-1)*timestep-start >= required:
        first -= 1
    while first*timestep <= start or first*timestep-start < required:
        first += 1
    available = end - start
    if int(math.floor(available / timestep)) > 100_000:
        raise ValueError("Finger reference search exceeds the bounded native control horizon")
    last, last_time = _activation(end, timestep)
    if last_time > end:
        last -= 1
    candidates = [step*timestep for step in range(first, last+1)]
    # A non-grid original deadline is still an eligible exact endpoint. It
    # is never reconstructed as start+(end-start), nor extended to a tick.
    if available >= required and (not candidates or candidates[-1] != end):
        candidates.append(end)
    for finish in candidates:
        ramp = CubicHermiteSpline([start, finish], np.stack((q0, goal)),
                                  np.stack((dq0, np.zeros_like(dq0))), axis=0)
        if _extrema(ramp, names, limits)["bounds_satisfied"]:
            return finish
    # END strict physical bounds and the original event/deadline are unchanged.
    raise ValueError(f"No C1 finger command satisfies the declared q/dq/ddq bounds before {end} s")


def _commands(values, names, duration, stop, timestep, position_limits):
    if not isinstance(values, (list, tuple)):
        raise ValueError("Finger command events must be an explicit ordered sequence")
    result, identities = [], set()
    # BEGIN require the occurrence that the Executor explicitly bound.
    required = {"skill_id", "event_id", "occurrence", "kind", "time", "progress", "finger_positions"}
    # END original event names remain provenance, not a unique attempt identity.
    for value in values:
        if not isinstance(value, dict) or not required <= value.keys():
            raise ValueError("Finger commands require original timed event identities and named position goals")
        row = {key: deepcopy(item) for key, item in value.items() if key not in _EVENT_FIELDS}
        for label in ("skill_id", "event_id"):
            _text(row[label], label)
        # BEGIN retries retain their source event and have a distinct occurrence.
        identity = event_occurrence_key(row)
        if identity in identities or row["kind"] not in ("open", "close", "hold"):
            raise ValueError("Finger command identities must be unique open/close/hold occurrences")
        # END a repeated occurrence is rejected before a polynomial is generated.
        identities.add(identity)
        if (not np.isfinite(row["time"]) or not 0. <= row["time"] <= duration
                or not np.isfinite(row["progress"]) or row["progress"] < 0.):
            raise ValueError("Finger commands must use the original finite physical clock and progress")
        goal = _array(row["finger_positions"], (len(names),), "Named finger event target")
        if np.any(goal < position_limits[0]) or np.any(goal > position_limits[1]):
            raise ValueError("Finger command targets must respect the declared native reference position bounds")
        # BEGIN a continuation subtracts native integer ticks, never re-ceils
        # a floating difference of two already-bound physical times.
        if "bound_activation_step" in row:
            step = row["bound_activation_step"]
            request_step = row.get("request_activation_step")
            origin_step = row.get("request_time_origin_step")
            if (type(step) is not int or type(request_step) is not int or type(origin_step) is not int
                    or step < 0 or origin_step < 0 or request_step-origin_step != step
                    or row.get("device_timestep_s") != timestep):
                raise ValueError("A bound finger command requires the unchanged original native tick identities")
            intended_step, requested_activation = _activation(float(row["request_intent_time_s"]), timestep)
            if intended_step != request_step or row.get("request_activation_time_s") != requested_activation:
                raise ValueError("A bound finger command changed its original intent/activation association")
            applied = step*timestep
        else:
            step, applied = _activation(float(row["time"]), timestep)
        # END the supplied original request clock remains the authority.
        # BEGIN timing quantization is checked against the frozen binding.
        if row.get("activation_time_s") is not None and applied != row["activation_time_s"]:
            raise ValueError("Finger activation differs from its frozen device clock")
        if applied >= duration:
            raise ValueError("A command needs activation/effect coverage within the valid trajectory")
        # END an event at T is not evidence that a command was actually applied.
        in_prefix = row["time"] < stop
        if in_prefix and applied >= stop:
            raise ValueError("An original finger command cannot activate before the evidence barrier on the native clock")
        row.update(finger_positions=goal.tolist(), activation_step=step, applied_time_s=applied,
                   application_delay_s=applied - float(row["time"]), in_executable_prefix=in_prefix,
                   superseded=False)
        result.append(row)
    result.sort(key=lambda row: row["time"])
    for index, row in enumerate(result[:-1]):
        # BEGIN only identical actuator goals may share an application tick.
        next_row = result[index + 1]
        same_tick = row["activation_step"] == next_row["activation_step"]
        if same_tick and (row["finger_positions"] != next_row["finger_positions"]
                          or row.get("requires_evidence") is not None):
            raise ValueError("Device quantization erases a required command sequence")
        row["superseded"] = same_tick
        # END source order is never replaced by silent last-command-wins.
    return tuple(result)


@dataclass(frozen=True)
class FingerTrajectory:
    """An owned, serializable reference; actual q/dq always come from feedback."""

    joint_names: tuple[str, ...]
    times: np.ndarray
    position: np.ndarray
    velocity: np.ndarray
    duration: float
    executable_until: float
    timestep: float
    position_limits: np.ndarray
    velocity_limits: np.ndarray
    acceleration_limits: np.ndarray
    events: tuple[dict, ...]
    segments: tuple[dict, ...]
    source: str
    feedback_id: str
    timestamp_s: float
    _curve: CubicHermiteSpline = field(init=False, repr=False)
    _bounds_report: dict = field(init=False, repr=False)

    def __post_init__(self):
        if isinstance(self.joint_names, str):
            raise ValueError("Finger reference joint names must be an ordered sequence")
        names = tuple(self.joint_names)
        if not names or len(set(names)) != len(names) or any(not isinstance(name, str) or not name.strip() for name in names):
            raise ValueError("Finger reference joint names must be complete and unique")
        object.__setattr__(self, "joint_names", names)
        _text(self.source, "Finger reference source")
        _text(self.feedback_id, "Initial measured feedback identity")
        if (not np.isfinite([self.duration, self.executable_until, self.timestep, self.timestamp_s]).all()
                or self.duration <= 0. or self.timestep <= 0.
                or not 0. <= self.executable_until <= self.duration
                or math.ceil(self.duration / self.timestep) > 100_000):
            raise ValueError("Finger references need one bounded native clock and original evidence barrier")
        for label in ("position", "velocity", "acceleration"):
            bound = _array(getattr(self, label + "_limits"), (2, len(names)), "Native reference " + label + " bounds")
            if np.any(bound[0] >= bound[1]) or (label != "position" and (np.any(bound[0] >= 0.) or np.any(bound[1] <= 0.))):
                raise ValueError("Native reference limits require ordered positions and signed velocity/acceleration intervals containing zero")
            object.__setattr__(self, label + "_limits", bound)
        times = np.asarray(self.times, float)
        if (times.ndim != 1 or not 2 <= len(times) <= 100_002 or not np.isfinite(times).all()
                or times[0] != 0. or times[-1] != self.duration or np.any(np.diff(times) <= 0.)):
            raise ValueError("Finger reference knots must cover the original duration exactly")
        position = _array(self.position, (len(times), len(names)), "Finger reference positions")
        velocity = _array(self.velocity, position.shape, "Finger reference velocities")
        object.__setattr__(self, "times", times.copy())
        object.__setattr__(self, "position", position)
        object.__setattr__(self, "velocity", velocity)
        events = _commands(self.events, names, self.duration, self.executable_until, self.timestep, self.position_limits)
        object.__setattr__(self, "events", events)
        object.__setattr__(self, "segments", tuple(deepcopy(self.segments)))
        curve = CubicHermiteSpline(times, position, velocity, axis=0)
        report = _extrema(curve, names, self._limits())
        if not report["bounds_satisfied"]:
            raise ValueError(f"Finger reference violates native q/dq/ddq bounds: {report}")
        object.__setattr__(self, "_curve", curve)
        object.__setattr__(self, "_bounds_report", report)
        self._validate_segments()
        for label in ("times", "position", "velocity", "position_limits", "velocity_limits", "acceleration_limits"):
            getattr(self, label).setflags(write=False)

    def _limits(self):
        return {label: getattr(self, label + "_limits") for label in ("position", "velocity", "acceleration")}

    @classmethod
    def build(cls, *, joint_names, initial_position, initial_velocity, commands, duration,
              executable_until, timestep, position_limits, velocity_limits, acceleration_limits,
              source, feedback_id, timestamp_s):
        """Compile explicit event goals without changing their original clock.

        Before a later first event, a bounded C1 takeover returns to the
        measured initial coordinate and zero reference velocity. This is an
        explicit position-hold reference, not fictitious stationary feedback.
        The initial measured velocity is always preserved as the first slope.
        """
        names = tuple(joint_names)
        count = len(names)
        q = _array(initial_position, (count,), "Actual initial finger q")
        dq = _array(initial_velocity, (count,), "Actual initial finger dq")
        bounds = {label: _array(value, (2, count), "Native " + label + " limits") for label, value in (
            ("position", position_limits), ("velocity", velocity_limits), ("acceleration", acceleration_limits))}
        if (not count or not np.isfinite([duration, executable_until, timestep]).all()
                or duration <= 0. or timestep <= 0. or not 0. <= executable_until <= duration):
            raise ValueError("A finger reference needs a finite named original trajectory clock")
        for label in ("velocity", "acceleration"):
            if np.any(bounds[label][0] >= 0.) or np.any(bounds[label][1] <= 0.):
                raise ValueError("Reference rate bounds must contain zero between finite signed limits")
        if (np.any(q < bounds["position"][0]) or np.any(q > bounds["position"][1])
                or np.any(dq < bounds["velocity"][0]) or np.any(dq > bounds["velocity"][1])):
            raise ValueError("Actual initial finger q/dq violate the declared native reference bounds")
        # BEGIN conditional authorization never truncates the nominal polynomial.
        events = _commands(commands, names, duration, executable_until, timestep, bounds["position"])
        active = [row for row in events if not row["superseded"]]
        windows = []
        if not active or active[0]["applied_time_s"] > 0.:
            windows.append((None, 0., active[0]["applied_time_s"] if active else duration, q.copy()))
        for index, event in enumerate(active):
            end = active[index + 1]["applied_time_s"] if index + 1 < len(active) else duration
            windows.append((event, event["applied_time_s"], end, np.asarray(event["finger_positions"], float)))
        # END the executor consumes only authorized parts of this same curve.
        knots, positions, velocities, segments = [0.], [q.copy()], [dq.copy()], []

        def append(time_s, value, rate):
            if time_s > knots[-1]:
                knots.append(float(time_s))
                positions.append(value.copy())
                velocities.append(rate.copy())
            elif time_s != knots[-1] or not np.array_equal(value, positions[-1]) or not np.array_equal(rate, velocities[-1]):
                raise ValueError("Finger command windows must have one continuous q/dq at each shared time")

        for event, begin, end, goal in windows:
            if end == begin:
                if np.any(dq != 0.):
                    raise ValueError("A zero-length command window cannot discard actual initial finger velocity")
                continue
            finish = _segment(q, dq, goal, begin, end, timestep, names, bounds)
            segments.append({"kind": "initial_hold" if event is None else "command",
                "skill_id": None if event is None else event["skill_id"],
                "event_id": None if event is None else event["event_id"],
                # BEGIN each ramp is bound to the exact command occurrence.
                "occurrence": None if event is None else event["occurrence"],
                # END an initial hold is not a retry of a source operation.
                "start_time_s": float(begin), "ramp_end_time_s": float(finish),
                "end_time_s": float(end), "target_positions": goal.tolist()})
            append(finish, goal, np.zeros(count))
            append(end, goal, np.zeros(count))
            q, dq = goal.copy(), np.zeros(count)
        append(duration, q, dq)
        return cls(names, np.asarray(knots), np.asarray(positions), np.asarray(velocities),
                   float(duration), float(executable_until), float(timestep), bounds["position"],
                   bounds["velocity"], bounds["acceleration"], events, tuple(segments),
                   source, feedback_id, float(timestamp_s))

    def _validate_segments(self):
        """Validate saved phase boundaries without regenerating or retiming.

        Reconstruct only the explicitly saved knots from event targets and
        ramp endpoints. A changed archive cannot hide motion before an event,
        replace the initial slope, or omit conditional commands in the suffix.
        """
        expected_times, expected_q, expected_dq = [0.], [self.position[0]], [self.velocity[0]]
        # BEGIN archive validation matches each ramp to its own occurrence.
        active = {event_occurrence_key(row): row for row in self.events
                  if not row["superseded"]}
        # END equal source names cannot cause a retry to overwrite an old ramp.
        used = set()
        end = 0.
        for segment in self.segments:
            if not isinstance(segment, dict) or set(segment) != {
                    "kind", "skill_id", "event_id", "occurrence", "start_time_s", "ramp_end_time_s", "end_time_s", "target_positions"}:
                raise ValueError("Every saved finger segment needs its exact source command and clock boundaries")
            begin, finish, stop = (float(segment[key]) for key in ("start_time_s", "ramp_end_time_s", "end_time_s"))
            if not np.isfinite([begin, finish, stop]).all() or begin != end or not begin <= finish <= stop <= self.duration or stop <= begin:
                raise ValueError("Saved finger segments must partition the full original duration")
            goal = _array(segment["target_positions"], (len(self.joint_names),), "Saved finger segment goal")
            # BEGIN initial holds have no source operation; commands use the strict key.
            identity = ((segment["skill_id"], segment["event_id"], segment["occurrence"])
                        if segment["kind"] == "initial_hold" else event_occurrence_key(segment))
            if segment["kind"] == "initial_hold":
                if begin != 0. or identity != (None, None, None) or not np.array_equal(goal, self.position[0]):
                    raise ValueError("The initial hold must retain actual initial finger coordinates")
            # END segment identities cannot silently revert to source-only matching.
            elif segment["kind"] == "command" and identity in active and identity not in used:
                event = active[identity]
                if begin != event["applied_time_s"] or not np.array_equal(goal, event["finger_positions"]):
                    raise ValueError("A saved finger ramp must begin at its original native command tick and retain its target")
                used.add(identity)
            else:
                raise ValueError("A saved finger segment cannot introduce or duplicate a command")
            for moment in (finish, stop):
                if moment > expected_times[-1]:
                    expected_times.append(moment)
                    expected_q.append(goal.copy())
                    expected_dq.append(np.zeros(len(self.joint_names)))
                elif not np.array_equal(expected_q[-1], goal) or np.any(expected_dq[-1] != 0.):
                    raise ValueError("A zero-time finger ramp cannot jump q or discard measured dq")
            end = stop
        if used != set(active) or end != self.duration:
            raise ValueError("Saved finger segments must preserve every original command over the complete duration")
        if (not np.array_equal(self.times, expected_times) or not np.array_equal(self.position, expected_q)
                or not np.array_equal(self.velocity, expected_dq)):
            raise ValueError("Saved finger arrays differ from their exact original event/reference contract")

    def sample(self, times):
        """Return commanded q/dq at the supplied shared physical times."""
        query = np.asarray(times, dtype=float)
        if not np.isfinite(query).all() or np.any(query < 0.) or np.any(query > self.duration):
            raise ValueError("Finger reference queries must stay inside the original trajectory duration")
        return self._curve(query), self._curve(query, 1)

    def sample_torch(self, times):
        """Evaluate the same saved polynomial without native FK or new timing."""
        import torch

        # Query tensors already belong to the planner's bounded device. This
        # module never creates a CUDA context or chooses a device itself.
        if (not isinstance(times, torch.Tensor) or not times.is_floating_point()
                or not bool(torch.isfinite(times).all()) or bool((times < 0.).any())
                or bool((times > self.duration).any())):
            raise ValueError("Finger reference queries must stay inside the original finite trajectory duration")
        knots = torch.as_tensor(self.times.copy(), dtype=times.dtype, device=times.device)
        coefficients = torch.as_tensor(self._curve.c.copy(), dtype=times.dtype, device=times.device)
        indices = torch.searchsorted(knots, times.contiguous(), right=True).clamp(1, len(self.times) - 1) - 1
        offset = (times - knots[indices])[..., None]
        selected = coefficients[:, indices, :]
        q = ((selected[0] * offset + selected[1]) * offset + selected[2]) * offset + selected[3]
        dq = (3 * selected[0] * offset + 2 * selected[1]) * offset + selected[2]
        return q, dq

    def describe(self):
        """Serialize owned values; a loader must not generate another curve."""
        return {"joint_names": list(self.joint_names), "source": self.source, "feedback_id": self.feedback_id,
                "timestamp_s": self.timestamp_s, "duration": self.duration, "executable_until": self.executable_until,
                "timestep": self.timestep, "times": self.times.tolist(), "position": self.position.tolist(),
                "velocity": self.velocity.tolist(), "position_limits": self.position_limits.tolist(),
                "velocity_limits": self.velocity_limits.tolist(), "acceleration_limits": self.acceleration_limits.tolist(),
                "events": deepcopy(list(self.events)), "segments": deepcopy(list(self.segments)),
                "scope": _SCOPE, "limits_scope": _LIMITS_SCOPE, "bounds_report": deepcopy(self._bounds_report)}

    @classmethod
    def from_mapping(cls, value):
        """Load the archived reference exactly, validating all saved claims."""
        fields = {name for name, definition in cls.__dataclass_fields__.items() if definition.init}
        if not isinstance(value, dict) or set(value) != fields | {"scope", "limits_scope", "bounds_report"}:
            raise ValueError("A saved finger reference requires its complete original polynomial and named contract")
        result = cls(**{key: deepcopy(value[key]) for key in fields})
        if result.describe() != value:
            raise ValueError("Saved finger reference metadata differs from the validated original polynomial")
        return result

    def validate_execution(self, trajectory, commands, *, initial_feedback, timestep,
                           position_limits, velocity_limits):
        """Bind a saved reference to current actual feedback and the arm clock."""
        if (trajectory.duration != self.duration or trajectory.executable_until != self.executable_until
                or timestep != self.timestep or tuple(initial_feedback["finger_joint_names"]) != self.joint_names
                or initial_feedback["feedback_id"] != self.feedback_id
                or initial_feedback["timestamp_s"] != self.timestamp_s):
            raise ValueError("Saved fingers, actual feedback and arm trajectory must share one exact original clock and identity")
        columns = [tuple(initial_feedback["joint_names"]).index(name) for name in self.joint_names]
        if (not np.array_equal(self.position[0], np.asarray(initial_feedback["q"])[columns])
                or not np.array_equal(self.velocity[0], np.asarray(initial_feedback["dq"])[columns])):
            raise ValueError("Finger reference must retain the actual measured initial q/dq without resetting velocity")
        native_position = _array(position_limits, self.position_limits.shape, "Native finger position bounds")
        native_velocity = _array(velocity_limits, self.velocity_limits.shape, "Hardware finger velocity bounds")
        if (np.any(self.position_limits[0] < native_position[0]) or np.any(self.position_limits[1] > native_position[1])
                or np.any(self.velocity_limits[0] < native_velocity[0]) or np.any(self.velocity_limits[1] > native_velocity[1])):
            raise ValueError("Saved reference bounds cannot exceed native finger position or hardware speed limits")
        expected = _commands(commands, self.joint_names, self.duration, self.executable_until, self.timestep, self.position_limits)
        if expected != self.events:
            raise ValueError("Saved finger reference must retain every exact original timed event and target")
        return deepcopy(self._bounds_report)
# END nominal finger references remain separate from measured motion and contact.
