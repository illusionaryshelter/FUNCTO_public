"""Native-time finger command curves with independent measured entrances.

The current active command supplies position, analytic velocity, target and
remaining ramp. Encoder q/dq are separate observations. New events retain their
original bound clock; nominal geometry has its own measured-state declaration.
"""

# BEGIN one saved C1 finger reference shared by planning and physical execution.
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, fields
import math

import numpy as np
from scipy.interpolate import CubicHermiteSpline, PPoly

# BEGIN one occurrence identity across planning, commands and saved polynomials.
from .planning_problem import event_occurrence_key
# END a repeated source event is not the same physical command occurrence.


_SCOPE = (
    "Named actuator command reference continuing the incoming active command. "
    "Measured entrance q/dq are separate observations; original event clocks "
    "and executable authorization prefixes are preserved.")
_LIMITS_SCOPE = ("Continuous reference q/dq/ddq bounds; acceleration is the supplied native "
                 "planning bound, not a calibrated actual hardware acceleration allowance.")
_EVENT_FIELDS = {"activation_step", "applied_time_s", "application_delay_s", "in_executable_prefix", "superseded"}


# BEGIN one command schema records actual command state and measured entrance.
_SCHEMA = "named-command-reference-with-measured-entrance-v1"
_INCOMING_SCHEMA = "active-finger-command-v1"
def _incoming_values(incoming, names):
    """Align command coordinates without changing the source event's ordering."""
    columns = [tuple(incoming["joint_names"]).index(name) for name in names]
    return tuple(np.asarray(incoming[key], dtype=float)[columns] for key in
                 ("position", "velocity", "target_position"))


def _validated_incoming(value, names, timestamp_s):
    """Own the explicit active-command state and its temporal provenance."""
    required = {"schema", "joint_names", "position", "velocity", "target_position",
                "remaining_ramp_s", "timestamp_s", "source", "source_feedback_id",
                "source_curve_time_s", "active_event"}
    if not isinstance(value, dict) or set(value) != required or value["schema"] != _INCOMING_SCHEMA:
        raise ValueError("An incoming command requires the complete active-command schema")
    result = deepcopy(value)
    incoming_names = tuple(result["joint_names"])
    if len(incoming_names) != len(names) or set(incoming_names) != set(names):
        raise ValueError("Incoming command coordinates must name exactly the current finger joints")
    for key in ("position", "velocity", "target_position"):
        result[key] = _array(result[key], (len(names),), "Incoming command " + key).tolist()
    for key in ("source", "source_feedback_id"):
        _text(result[key], "Incoming command " + key)
    timing = np.asarray([result["remaining_ramp_s"], result["timestamp_s"],
                         result["source_curve_time_s"]], dtype=float)
    if (not np.isfinite(timing).all() or timing[0] < 0. or timing[2] < 0.
            or timing[1] != timestamp_s):
        raise ValueError("Incoming command and measured entrance require the same finite boundary time")
    event = result["active_event"]
    if event is not None:
        event_occurrence_key(event)
        if event["kind"] not in ("open", "close", "hold"):
            raise ValueError("The inherited active event must be an original finger command")
        if not np.array_equal(result["target_position"], event["finger_positions"]):
            raise ValueError("The incoming target must remain the active source event's named target")
    q, dq, target = _incoming_values(result, names)
    if timing[0] == 0. and (not np.array_equal(q, target) or np.any(dq != 0.)):
        raise ValueError("A completed active ramp must already hold its target with zero command velocity")
    return result


def _incoming_sampler(incoming, names):
    """Return the unique remaining cubic followed by its existing target hold.

    A cubic is uniquely determined by its endpoint positions and velocities.
    Its initial endpoint is the preceding reference evaluated at the actual
    handover; its final endpoint is that active command's original zero-speed
    target. Queries after that ramp continue the same command, not a future
    event from the preceding skill.
    """
    q, dq, target = _incoming_values(incoming, names)
    finish = float(incoming["remaining_ramp_s"])
    curve = (CubicHermiteSpline([0., finish], np.stack((q, target)),
                               np.stack((dq, np.zeros_like(dq))), axis=0)
             if finish > 0. else None)

    def sample(time_s):
        if time_s == 0.:
            return q.copy(), dq.copy()
        if time_s >= finish:
            return target.copy(), np.zeros_like(dq)
        return curve(time_s), curve(time_s, 1)

    return sample

# END current source events retain their existing device clock.


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
    incoming_command: dict
    measured_initial_position: np.ndarray
    measured_initial_velocity: np.ndarray
    _curve: CubicHermiteSpline = field(init=False, repr=False)
    _bounds_report: dict = field(init=False, repr=False)
    # BEGIN reuse immutable polynomial tensors across repeated rollout queries.
    _torch_cache: dict = field(default_factory=dict, init=False, repr=False, compare=False)
    # END this runtime cache is excluded from the archived reference schema.

    def __post_init__(self):
        if isinstance(self.joint_names, str):
            raise ValueError("Finger reference joint names must be an ordered sequence")
        names = tuple(self.joint_names)
        if not names or len(set(names)) != len(names) or any(not isinstance(name, str) or not name.strip() for name in names):
            raise ValueError("Finger reference joint names must be complete and unique")
        object.__setattr__(self, "joint_names", names)
        # BEGIN the active command and encoder sample share a time, not values.
        object.__setattr__(self, "incoming_command",
            _validated_incoming(self.incoming_command, names, self.timestamp_s))
        for key in ("measured_initial_position", "measured_initial_velocity"):
            value = _array(getattr(self, key), (len(names),), key)
            value.setflags(write=False)
            object.__setattr__(self, key, value)
        # END bounds below qualify the command polynomial only.
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

    # BEGIN command and measured entrance use separate explicit records.
    @classmethod
    def build(cls, *, joint_names, incoming_command, measured_initial_position,
              measured_initial_velocity, commands, duration, executable_until,
              timestep, position_limits, velocity_limits, acceleration_limits,
              source, feedback_id, timestamp_s):
        """Compile original next-skill events after the current active command.

        If the first new event occurs during the inherited ramp, its q/dq are
        sampled at that event before constructing the new goal's bounded ramp.
        If it occurs later, the old target is held until that event. A new
        event at zero takes over from the incoming command q/dq immediately.
        """
        names = tuple(joint_names)
        incoming = _validated_incoming(incoming_command, names, timestamp_s)
        q, dq, active_target = _incoming_values(incoming, names)
        bounds = {label: _array(value, (2, len(names)), "Command " + label + " limits")
                  for label, value in (("position", position_limits),
                    ("velocity", velocity_limits), ("acceleration", acceleration_limits))}
        if (not names or not np.isfinite([duration, executable_until, timestep]).all()
                or duration <= 0. or timestep <= 0. or not 0. <= executable_until <= duration):
            raise ValueError("A command reference requires a finite original clock and authorization prefix")
        for label in ("velocity", "acceleration"):
            if np.any(bounds[label][0] >= 0.) or np.any(bounds[label][1] <= 0.):
                raise ValueError("Command rate bounds must contain zero between signed limits")
        if (np.any(q < bounds["position"][0]) or np.any(q > bounds["position"][1])
                or np.any(dq < bounds["velocity"][0]) or np.any(dq > bounds["velocity"][1])):
            raise ValueError("Incoming command q/dq violate the declared reference bounds")
        events = _commands(commands, names, duration, executable_until, timestep, bounds["position"])
        active = [event for event in events if not event["superseded"]]
        knots, positions, velocities, segments = [0.], [q.copy()], [dq.copy()], []

        def append(moment, value, rate):
            if moment > knots[-1]:
                knots.append(float(moment)); positions.append(value.copy()); velocities.append(rate.copy())
            elif (moment != knots[-1] or not np.array_equal(value, positions[-1])
                  or not np.array_equal(rate, velocities[-1])):
                raise ValueError("Adjacent command pieces require one continuous q/dq boundary")

        if not active or active[0]["applied_time_s"] > 0.:
            end = float(active[0]["applied_time_s"] if active else duration)
            finish = min(float(incoming["remaining_ramp_s"]), end)
            sample = _incoming_sampler(incoming, names)
            identity = (None, None, None) if incoming["active_event"] is None else event_occurrence_key(incoming["active_event"])
            segments.append(dict(kind="incoming_command", skill_id=identity[0], event_id=identity[1],
                occurrence=identity[2], start_time_s=0., ramp_end_time_s=finish,
                end_time_s=end, target_positions=active_target.tolist()))
            append(finish, *sample(finish))
            q, dq = sample(end)
            append(end, q, dq)
        for index, event in enumerate(active):
            begin = float(event["applied_time_s"])
            end = float(active[index + 1]["applied_time_s"] if index + 1 < len(active) else duration)
            goal = np.asarray(event["finger_positions"], dtype=float)
            finish = _segment(q, dq, goal, begin, end, timestep, names, bounds)
            identity = event_occurrence_key(event)
            segments.append(dict(kind="command", skill_id=identity[0], event_id=identity[1],
                occurrence=identity[2], start_time_s=begin, ramp_end_time_s=float(finish),
                end_time_s=end, target_positions=goal.tolist()))
            append(finish, goal, np.zeros(len(names)))
            append(end, goal, np.zeros(len(names)))
            q, dq = goal.copy(), np.zeros(len(names))
        return cls(joint_names=names, times=np.asarray(knots), position=np.asarray(positions),
            velocity=np.asarray(velocities), duration=float(duration), executable_until=float(executable_until),
            timestep=float(timestep), position_limits=bounds["position"], velocity_limits=bounds["velocity"],
            acceleration_limits=bounds["acceleration"], events=events, segments=tuple(segments),
            source=source, feedback_id=feedback_id, timestamp_s=float(timestamp_s),
            incoming_command=incoming, measured_initial_position=measured_initial_position,
            measured_initial_velocity=measured_initial_velocity)
    # END no measured derivative substitutes for an active command.

    # BEGIN command and measured entrance use separate explicit records.
    def _validate_segments(self):
        """Reconstruct saved knots from the active prefix and original events.

        Only the incoming prefix may end with nonzero command velocity: a new
        event can supersede an unfinished prior ramp. All newly generated goal
        ramps retain the existing zero terminal slope and original deadline.
        This audit never regenerates or retimes a saved trajectory.
        """
        q0, dq0, active_target = _incoming_values(self.incoming_command, self.joint_names)
        expected_times, expected_q, expected_dq = [0.], [q0], [dq0]
        active = {event_occurrence_key(row): row for row in self.events if not row["superseded"]}
        used, previous_end = set(), 0.

        def append(moment, value, rate):
            if moment > expected_times[-1]:
                expected_times.append(moment); expected_q.append(value); expected_dq.append(rate)
            elif (moment != expected_times[-1] or not np.array_equal(value, expected_q[-1])
                  or not np.array_equal(rate, expected_dq[-1])):
                raise ValueError("A saved zero-time boundary changed command position or velocity")

        for segment in self.segments:
            if not isinstance(segment, dict) or set(segment) != {
                    "kind", "skill_id", "event_id", "occurrence", "start_time_s",
                    "ramp_end_time_s", "end_time_s", "target_positions"}:
                raise ValueError("Saved command pieces require complete identities and clock boundaries")
            begin, finish, end = (float(segment[key]) for key in
                                 ("start_time_s", "ramp_end_time_s", "end_time_s"))
            if (not np.isfinite([begin, finish, end]).all() or begin != previous_end
                    or not begin <= finish <= end <= self.duration or end <= begin):
                raise ValueError("Saved command pieces must partition the original duration")
            goal = _array(segment["target_positions"], (len(self.joint_names),), "Saved command target")
            if segment["kind"] == "incoming_command":
                incoming_event = self.incoming_command["active_event"]
                identity = (None, None, None) if incoming_event is None else event_occurrence_key(incoming_event)
                first_event = min((row["applied_time_s"] for row in active.values()), default=self.duration)
                if (begin != 0. or end != first_event
                        or tuple(segment[key] for key in ("skill_id", "event_id", "occurrence")) != identity
                        or finish != min(float(self.incoming_command["remaining_ramp_s"]), end)
                        or not np.array_equal(goal, active_target)):
                    raise ValueError("The saved incoming prefix must retain its active command until the first new event")
                sample = _incoming_sampler(self.incoming_command, self.joint_names)
                append(finish, *sample(finish)); append(end, *sample(end))
            elif segment["kind"] == "command":
                identity = event_occurrence_key(segment)
                if identity not in active or identity in used:
                    raise ValueError("A saved command piece cannot introduce or repeat a new event")
                event = active[identity]
                if begin != event["applied_time_s"] or not np.array_equal(goal, event["finger_positions"]):
                    raise ValueError("Saved commands must retain the original event activation and target")
                used.add(identity)
                append(finish, goal, np.zeros(len(self.joint_names)))
                append(end, goal, np.zeros(len(self.joint_names)))
            else:
                raise ValueError("A command piece is an incoming active command or an original new event")
            previous_end = end
        if used != set(active) or previous_end != self.duration:
            raise ValueError("Saved command pieces must retain all new events and the complete duration")
        if (not np.array_equal(self.times, expected_times) or not np.array_equal(self.position, expected_q)
                or not np.array_equal(self.velocity, expected_dq)):
            raise ValueError("Saved arrays changed the incoming command state or original event pieces")
    # END no measured derivative substitutes for an active command.

    def sample(self, times):
        """Return commanded q/dq at the supplied shared physical times."""
        query = np.asarray(times, dtype=float)
        if not np.isfinite(query).all() or np.any(query < 0.) or np.any(query > self.duration):
            raise ValueError("Finger reference queries must stay inside the original trajectory duration")
        return self._curve(query), self._curve(query, 1)

    def sample_torch(self, times):
        """Evaluate the same saved polynomial without native FK or new timing."""
        import torch
        from .optimizer_device_checks import require_tensor

        # Query tensors already belong to the planner's bounded device. This
        # module never creates a CUDA context or chooses a device itself.
        # BEGIN preserve polynomial operations while removing per-call H2D copies.
        message = "Finger reference queries must stay inside the original finite trajectory duration"
        if not isinstance(times, torch.Tensor) or not times.is_floating_point():
            raise ValueError(message)
        require_tensor(torch.isfinite(times).all() & ~(times < 0.).any()
                       & ~(times > self.duration).any(), message)
        key = (times.device, times.dtype)
        if key not in self._torch_cache:
            self._torch_cache[key] = (
                torch.as_tensor(self.times.copy(), dtype=times.dtype, device=times.device),
                torch.as_tensor(self._curve.c.copy(), dtype=times.dtype, device=times.device))
        knots, coefficients = self._torch_cache[key]
        # END each dtype keeps the exact original conversion and operation order.
        indices = torch.searchsorted(knots, times.contiguous(), right=True).clamp(1, len(self.times) - 1) - 1
        offset = (times - knots[indices])[..., None]
        selected = coefficients[:, indices, :]
        q = ((selected[0] * offset + selected[1]) * offset + selected[2]) * offset + selected[3]
        dq = (3 * selected[0] * offset + 2 * selected[1]) * offset + selected[2]
        return q, dq

    # BEGIN budget the planner's float32 and float64 immutable device copies.
    @property
    def torch_storage_bytes(self):
        """Reserve both planning dtypes before lazy polynomial tensor creation."""
        return (self.times.size + self._curve.c.size) * (4 + 8)
    # END ordinary CPU serialization does not allocate these device copies.

    def describe(self):
        """Serialize owned values; a loader must not generate another curve."""
        return {"joint_names": list(self.joint_names), "source": self.source, "feedback_id": self.feedback_id,
                "timestamp_s": self.timestamp_s, "duration": self.duration, "executable_until": self.executable_until,
                "timestep": self.timestep, "times": self.times.tolist(), "position": self.position.tolist(),
                "velocity": self.velocity.tolist(), "position_limits": self.position_limits.tolist(),
                "velocity_limits": self.velocity_limits.tolist(), "acceleration_limits": self.acceleration_limits.tolist(),
                "events": deepcopy(list(self.events)), "segments": deepcopy(list(self.segments)),
                "schema": _SCHEMA, "incoming_command": deepcopy(self.incoming_command),
                "measured_initial_position": self.measured_initial_position.tolist(),
                "measured_initial_velocity": self.measured_initial_velocity.tolist(),
                "scope": _SCOPE, "limits_scope": _LIMITS_SCOPE, "bounds_report": deepcopy(self._bounds_report)}

    # BEGIN command and measured entrance use separate explicit records.
    @classmethod
    def from_mapping(cls, value):
        """Load only the current schema, without converting old archives."""
        arguments = {item.name for item in fields(cls) if item.init}
        if (not isinstance(value, dict) or set(value) != arguments | {"schema", "scope", "limits_scope", "bounds_report"}
                or value["schema"] != _SCHEMA):
            raise ValueError("A command reference requires its complete new measured-entrance schema")
        result = cls(**{key: deepcopy(value[key]) for key in arguments})
        if result.describe() != value:
            raise ValueError("Saved command-reference metadata differs from its reconstructed contract")
        return result
    # END no measured derivative substitutes for an active command.

    # BEGIN command and measured entrance use separate explicit records.
    def validate_execution(self, trajectory, commands, *, initial_feedback, timestep,
                           position_limits, velocity_limits):
        """Check actual entrance and freshly read command state independently.

        The executor supplies ``actuator_ctrl`` from
        ``executor.backend.proprioception()`` immediately before this call.
        It preserves the archived entrance identity and actual q/dq for the
        original executor's complete numeric-state comparison. A saved ctrl
        value cannot substitute for this live command observation. The device
        exposes no command-derivative sensor: that derivative is retained from
        the preceding saved active polynomial, independently of encoder dq.
        """
        if (trajectory.duration != self.duration or trajectory.executable_until != self.executable_until
                or timestep != self.timestep or tuple(initial_feedback["finger_joint_names"]) != self.joint_names
                or initial_feedback["feedback_id"] != self.feedback_id
                or initial_feedback["timestamp_s"] != self.timestamp_s):
            raise ValueError("Saved commands, actual entrance and arm trajectory require one original clock and identity")
        columns = [tuple(initial_feedback["joint_names"]).index(name) for name in self.joint_names]
        if (not np.array_equal(self.measured_initial_position, np.asarray(initial_feedback["q"])[columns])
                or not np.array_equal(self.measured_initial_velocity, np.asarray(initial_feedback["dq"])[columns])):
            raise ValueError("Measured finger entrance q/dq differ from the saved independent observations")
        observation = initial_feedback["actuator_ctrl"]
        command_columns = [tuple(observation["joint_names"]).index(name) for name in self.joint_names]
        live_command = np.asarray(observation["values"], dtype=float)[command_columns]
        if (observation["sim_time"] != initial_feedback["sim_time"]
                or observation["timestamp_s"] != initial_feedback["timestamp_s"]
                or not np.array_equal(live_command, self.position[0])):
            raise ValueError("The live incoming actuator command differs from the saved command entrance")
        native_position = _array(position_limits, self.position_limits.shape, "Native finger position bounds")
        native_velocity = _array(velocity_limits, self.velocity_limits.shape, "Hardware finger velocity bounds")
        if (np.any(self.position_limits[0] < native_position[0]) or np.any(self.position_limits[1] > native_position[1])
                or np.any(self.velocity_limits[0] < native_velocity[0]) or np.any(self.velocity_limits[1] > native_velocity[1])):
            raise ValueError("Saved command bounds cannot exceed native position or hardware speed limits")
        expected = _commands(commands, self.joint_names, self.duration, self.executable_until,
                             self.timestep, self.position_limits)
        if expected != self.events:
            raise ValueError("Saved commands must retain every original timed event and target")
        return deepcopy(self._bounds_report)
    # END no measured derivative substitutes for an active command.
# END nominal finger references remain separate from measured motion and contact.


# BEGIN physical-clock validity belongs to geometry, never command authorization.
_GEOMETRY_SCOPE = (
    "Nominal collision geometry only on [0, valid_until_s], including the exact "
    "typed endpoint. Later contact-dependent fingers remain unknown. This model "
    "does not predict measured tracking or change command/execution authorization.")


# BEGIN preserve measured-origin free-motion geometry independently of commands.
def _free_finger_geometry(reference):
    """Generate the established nominal free-hand model from measured q/dq.

    The model follows the original event targets using the existing bounded
    Hermite interpolation. Its entrance is the actual measured shape; the
    independent actuator curve continues its incoming command instead. This
    is a declared free-motion hypothesis ending at the first contact boundary.
    """
    q, dq = reference.measured_initial_position.copy(), reference.measured_initial_velocity.copy()
    active = [row for row in reference.events if not row["superseded"]]
    windows = []
    if not active or active[0]["applied_time_s"] > 0.:
        windows.append((0., active[0]["applied_time_s"] if active else reference.duration, q.copy()))
    for index, event in enumerate(active):
        end = active[index+1]["applied_time_s"] if index+1 < len(active) else reference.duration
        windows.append((event["applied_time_s"], end, np.asarray(event["finger_positions"], float)))
    times, positions, velocities = [0.], [q.copy()], [dq.copy()]
    for begin, end, goal in windows:
        finish = _segment(q, dq, goal, begin, end, reference.timestep,
                          reference.joint_names, reference._limits())
        for moment in (finish, end):
            if moment > times[-1]:
                times.append(float(moment))
                positions.append(goal.copy())
                velocities.append(np.zeros_like(dq))
        q, dq = goal.copy(), np.zeros_like(dq)
    return np.asarray(times), np.asarray(positions), np.asarray(velocities)
# END this model is consumed by geometry alone, never by physical actuation.


@dataclass(frozen=True)
class NominalFingerGeometry:
    """An owned finger polynomial with an explicit physical validity prefix.

    A free-motion model reuses the original command polynomial. A held model
    freezes measured entrance q; its zero derivative describes that geometric
    assumption, not the measured entrance dq or the executable command curve.
    Sampling evaluates the polynomial on its full stored clock. Collision
    consumers must use ``valid_mask`` before claiming geometry at a query.
    """

    joint_names: tuple[str, ...]
    times: np.ndarray
    position: np.ndarray
    velocity: np.ndarray
    duration: float
    valid_until_s: float
    source: str
    _curve: CubicHermiteSpline = field(init=False, repr=False)
    _torch_cache: dict = field(default_factory=dict, init=False, repr=False, compare=False)

    def __post_init__(self):
        names = tuple(self.joint_names)
        if (not names or len(set(names)) != len(names)
                or any(not isinstance(name, str) or not name.strip() for name in names)):
            raise ValueError("Nominal finger geometry needs unique named joints")
        _text(self.source, "Nominal finger geometry source")
        if (not np.isfinite([self.duration, self.valid_until_s]).all()
                or self.duration <= 0. or not 0. <= self.valid_until_s <= self.duration):
            raise ValueError("Nominal finger validity must be a prefix of its original physical clock")
        times = np.asarray(self.times, dtype=float)
        if (times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all()
                or times[0] != 0. or times[-1] != self.duration or np.any(np.diff(times) <= 0.)):
            raise ValueError("Nominal finger geometry requires the complete ordered polynomial clock")
        object.__setattr__(self, "joint_names", names)
        for name, value in (("times", times.copy()),
                            ("position", _array(self.position, (len(times), len(names)), "Nominal finger q")),
                            ("velocity", _array(self.velocity, (len(times), len(names)), "Nominal finger dq"))):
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        object.__setattr__(self, "_curve", CubicHermiteSpline(
            self.times, self.position, self.velocity, axis=0, extrapolate=False))

    def valid_mask(self, times):
        """Use the same exact typed boundary as the planner's Torch query.

        A float32 endpoint is its declared representation of the binary64
        source time, even when rounding lies above it. Its next representable
        value is outside. No epsilon or progress-based entrance is admitted.
        """
        query = np.asarray(times)
        terminal = np.asarray(self.duration, dtype=query.dtype)
        if (not np.isfinite(query).all() or np.any(query < 0.) or np.any(query > terminal)):
            raise ValueError("Finger reference queries must stay inside the original trajectory duration")
        endpoint = np.asarray(self.valid_until_s, dtype=query.dtype)
        return np.isfinite(query) & (query >= 0.) & (query <= endpoint)

    def valid_mask_torch(self, times):
        import torch
        from .optimizer_device_checks import require_tensor
        message = "Finger reference queries must stay inside the original finite trajectory duration"
        if not isinstance(times, torch.Tensor) or not times.is_floating_point():
            raise ValueError(message)
        admitted = torch.isfinite(times) & (times >= 0.) & (times <= self.duration)
        require_tensor(admitted.all(), message)
        return admitted & (times <= self.valid_until_s)

    def sample(self, times):
        return FingerTrajectory.sample(self, times)

    def sample_torch(self, times):
        # Reuse the original polynomial evaluator and stable tensor cache.
        return FingerTrajectory.sample_torch(self, times)

    @property
    def torch_storage_bytes(self):
        return FingerTrajectory.torch_storage_bytes.fget(self)

    def describe(self):
        return {"joint_names": list(self.joint_names), "times": self.times.tolist(),
                "position": self.position.tolist(), "velocity": self.velocity.tolist(),
                "duration": self.duration, "valid_until_s": self.valid_until_s,
                "source": self.source, "scope": _GEOMETRY_SCOPE}

    @classmethod
    def from_reference(cls, reference, *, valid_until_s, source):
        """Own the same polynomial under a separately declared geometry scope."""
        return cls(reference.joint_names, reference.times, reference.position,
                   reference.velocity, reference.duration, valid_until_s, source)

    @classmethod
    def from_mapping(cls, value):
        fields = {name for name, definition in cls.__dataclass_fields__.items() if definition.init}
        if not isinstance(value, dict) or set(value) != fields | {"scope"}:
            raise ValueError("Saved nominal geometry needs its polynomial, validity and source")
        result = cls(**{key: deepcopy(value[key]) for key in fields})
        if result.describe() != value:
            raise ValueError("Saved nominal geometry differs from its validated physical-clock contract")
        return result


def bind_nominal_finger_geometry(reference, conditions, event_clock, *, progress_interval):
    """Declare geometry from contact identities and the nominal held model.

    ``reference`` remains the original C1 command, including measured dq.
    ``event_clock`` is the already-bound original occurrence clock; no inverse
    progress map, replacement event, stop or execution barrier is introduced.
    Missing consumed contact identities cannot establish a free-motion suffix.
    """
    begin, end = np.asarray(progress_interval, dtype=float)
    if not np.isfinite([begin, end]).all() or begin < 0. or end < begin:
        raise ValueError("Nominal geometry needs the current bound progress interval")
    # BEGIN attachment conditions declare a model, not a measured predicate.
    holding = [row for row in conditions if row.get("kind") == "declared_attachment"]
    # END the accepted condition already owns its model provenance.
    boundaries = []
    if holding:
        # The first original open command or identified release effect ends
        # this held-shape assumption, regardless of source list ordering.
        boundaries.extend((row["applied_time_s"], event_occurrence_key(row))
                          for row in reference.events if row["kind"] == "open" and not row["superseded"])
        for row in conditions:
            if row.get("kind") == "predicted_release" and begin <= row["progress"] <= end:
                # Effect conditions name their source event; actual occurrences
                # come from the authoritative bound clock, including retries.
                matched = [(clock["time"], identity) for identity, clock in event_clock.items()
                           if identity[:2] == (row["skill_id"], row["event_id"])]
                if not matched:
                    if row["progress"] == begin:
                        # A continuation may retain the same progress while
                        # its already-consumed release is absent from this clock.
                        boundaries.append((0., (row["skill_id"], row["event_id"])))
                        continue
                    raise ValueError("A release inside this motion block requires its original bound event occurrence")
                boundaries.extend(matched)
        times = np.array([0., reference.duration])
        # BEGIN held geometry uses measured shape, independently of actuator target.
        positions = np.tile(reference.measured_initial_position, (2, 1))
        # END the same declared relation owns its physical validity interval.
        velocities = np.zeros_like(positions)
        origin = ("Declared attachment model; frozen measured entrance q, geometric dq=0 (not measured dq); "
                  f"feedback={reference.feedback_id}; conditions="
                  + repr([row["condition_id"] for row in holding]))
    else:
        for row in conditions:
            if row.get("kind") != "grasp_contact_activation":
                continue
            if row["interval"][0] > end:
                # A future block's close is outside this polynomial clock.
                continue
            identity = event_occurrence_key(row["activation_event"])
            if (row["interval"][0] < begin
                    or row["interval"][0] == begin and identity not in event_clock):
                boundaries.append((0., identity))
            elif identity not in event_clock:
                raise ValueError("Contact activation inside this motion block requires its original bound event occurrence")
            else:
                boundaries.append((event_clock[identity]["activation_time_s"], identity))
        # BEGIN free geometry retains its measured-origin interpolation model.
        times, positions, velocities = _free_finger_geometry(reference)
        origin = f"Free-motion measured-origin Hermite geometry; feedback={reference.feedback_id}"
        # END event targets and timing are shared; command entrance is independent.
    cutoff = min((time for time, _ in boundaries), default=reference.duration)
    source = origin + "; original invalidating occurrences=" + repr(
        [identity for time, identity in boundaries if time == cutoff])
    return NominalFingerGeometry(reference.joint_names, times, positions, velocities,
                                 reference.duration, cutoff, source)
# END nominal scope never shortens or edits the executable command curve.


# BEGIN explicit stationary device command at a new controller entrance.
def stationary_finger_command(*, joint_names, position, timestamp_s, source, feedback_id):
    """Declare an already issued stationary command in named joint coordinates.

    This producer is for a device holding a fixed target without an unfinished
    command ramp. Position comes from actuator command feedback, not encoders.
    A continuation of a moving reference uses ``capture_active_command``.
    """
    names = tuple(joint_names)
    q = _array(position, (len(names),), "Stationary command position")
    return _validated_incoming(dict(schema=_INCOMING_SCHEMA, joint_names=list(names),
        position=q.tolist(), velocity=np.zeros_like(q).tolist(), target_position=q.tolist(),
        remaining_ramp_s=0., timestamp_s=float(timestamp_s), source=source,
        source_feedback_id=feedback_id, source_curve_time_s=0., active_event=None),
        names, float(timestamp_s))
# END no encoder velocity supplies an unobserved command derivative.


def capture_active_command(reference, *, curve_time_s, last_dispatch_time_s,
                           boundary_timestamp_s, source):
    """Extract one already active command from an explicitly loaded reference.

    The caller loads the previous archive using that archive's own component.
    This function consumes a validated object, not an alternate input schema.
    Only the active segment's target and remaining cubic are returned; later
    events remain in the preceding archive. ``last_dispatch_time_s`` comes
    from the completed execution's last control_elapsed entry: an event at a
    step endpoint has not yet activated in that preceding step. The command
    q/dq are nevertheless the step-end reference sample at ``curve_time_s``.
    ``boundary_timestamp_s`` is the actual sample time, distinct from this
    possibly clamped reference clock. This interface describes a completed
    normal dispatch; an interrupted dispatch/hold needs its own command record.
    """
    moment = float(curve_time_s)
    dispatched = float(last_dispatch_time_s)
    if (not np.isfinite([moment, dispatched]).all()
            or not 0. <= dispatched <= moment <= reference.executable_until
            or dispatched >= reference.executable_until):
        raise ValueError("The command handover requires an actual dispatch inside the original executable prefix")
    segment = [row for row in reference.segments
               if row["start_time_s"] <= dispatched <= row["end_time_s"]][-1]
    q, dq = reference.sample(moment)
    command_names = reference.joint_names
    if segment["kind"] == "incoming_command":
        incoming = reference.incoming_command
        # Preserve the active source event's coordinate order through another
        # handover; the next builder independently aligns these named arrays.
        command_names = tuple(incoming["joint_names"])
        columns = [reference.joint_names.index(name) for name in command_names]
        q, dq = q[columns], dq[columns]
        target = np.asarray(incoming["target_position"], dtype=float)
        remaining = max(0., float(incoming["remaining_ramp_s"]) - moment)
        event = deepcopy(incoming["active_event"])
    else:
        target = np.asarray(segment["target_positions"], dtype=float)
        remaining = max(0., float(segment["ramp_end_time_s"]) - moment)
        event = deepcopy(next(
            row for row in reference.events if event_occurrence_key(row) == event_occurrence_key(segment)))
    result = dict(schema=_INCOMING_SCHEMA, joint_names=list(command_names),
        position=q.tolist(), velocity=dq.tolist(), target_position=target.tolist(),
        remaining_ramp_s=remaining, timestamp_s=float(boundary_timestamp_s),
        source=f"{source}; last actual control dispatch at reference time {dispatched!r} s",
        source_feedback_id=reference.feedback_id, source_curve_time_s=moment, active_event=event)
    return _validated_incoming(result, reference.joint_names, float(boundary_timestamp_s))
