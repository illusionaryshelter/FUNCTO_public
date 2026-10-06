"""Owned evidence for native acceptance, before modes and anchors are folded.

These records describe evaluated discrete states. They never authorize motion,
change a residual, or reconstruct a geometry query at a different precision.
"""

from copy import deepcopy

import numpy as np


# BEGIN permanent failure evidence and bounded current-candidate query scheduling.
class AttemptWitnesses:
    """Keep historical failures separate from the next optimizer's queries.

    The ledger records normalized physical time, not source progress: a pause
    can map several distinct trajectory times to one progress value. Exact
    repeated times share one query. Nearby but distinct times remain distinct
    because neither a numerical tolerance nor a previous passing query proves
    the intervening source geometry safe. Only the explicitly selected current
    candidate supplies the bounded optimizer query set. Historical parameters
    remain mandatory inputs to independent export validation after exchange;
    this scheduler neither discards evidence nor certifies motion.
    """

    def __init__(self):
        self._parameters = set()
        self._current = set()
        self._selection_counts = {}
        self._last_selection = None

    @staticmethod
    def _validated_parameters(values, *, source):
        parameters = np.asarray(values, dtype=float)
        if (parameters.ndim != 1 or not np.isfinite(parameters).all()
                or np.any(parameters < 0.) or np.any(parameters > 1.)):
            raise ValueError(f"{source} requires finite normalized physical times in [0, 1]")
        return parameters

    @property
    def parameters(self):
        """Return an owned sorted snapshot; callers cannot erase the ledger."""
        return np.asarray(sorted(self._parameters), dtype=float)

    @property
    def current_parameters(self):
        """Return failures belonging to the explicitly selected candidate."""
        return np.asarray(sorted(self._current), dtype=float)

    @property
    def last_selection(self):
        """Return owned scheduling evidence, not a constraint-slack verdict."""
        return deepcopy(self._last_selection)

    @classmethod
    def _native_parameters(cls, native_metrics, *, duration):
        """Read exact times from bound native source-constraint traces.

        Traces can carry the original normalized query parameter or their
        physical ``time_s`` and frozen duration; the latter ratio is used
        exactly, without clipping or epsilon merging. Unavailable/unmatched
        observer traces contain no attributed witness and are left to their
        existing diagnostic report.
        """
        duration = float(duration)
        if not np.isfinite(duration) or duration <= 0.:
            raise ValueError("Native failure witnesses require a finite positive physical duration")
        parameters = []
        for label in ("rollout", "interpolated"):
            metrics = native_metrics.get(label, {})
            # Unfolded native joint constraints can carry their own original
            # time. An index without that explicit time cannot identify an
            # anchor folded into the last cost column and is never guessed.
            parameters.extend(float(row["normalized_time"])
                              for row in metrics.get("violating_constraint_samples", ())
                              if "normalized_time" in row)
            trace = metrics.get("source_trace", {})
            if trace.get("status") != "recorded":
                continue
            for row in trace.get("component_failures", ()):
                # A recorded component failure is already selected using its
                # original hard residual, including nonfinite rejection.
                parameters.append(float(row["normalized_time"]) if "normalized_time" in row
                                  else float(row["time_s"])/duration)
        return cls._validated_parameters(parameters, source="Native failure witnesses")

    @classmethod
    def _validation_parameters(cls, validation):
        """Read failed times from a completed independent curve check."""
        parameters = cls._validated_parameters(validation["u"], source="Independent validation")
        violations = np.asarray(validation["violations"], dtype=float)
        if violations.shape != parameters.shape:
            raise ValueError("Independent validation needs one hard residual per physical query time")
        failed = (violations > 0.) | ~np.isfinite(violations)
        return parameters[failed]

    def add_native(self, native_metrics, *, duration):
        """Accumulate evidence from any visited seed without changing priority."""
        parameters = self._native_parameters(native_metrics, duration=duration)
        self._parameters.update(parameters.tolist())

    def add_validation(self, validation):
        """Accumulate completed validation failures without changing priority."""
        self._parameters.update(self._validation_parameters(validation).tolist())

    def set_current(self, native_metrics, *, validation=None, duration):
        """Select one continuation candidate's native and independent failures.

        The caller must invoke this only after candidate selection. Observers
        and unselected seeds use add_native/add_validation instead. Validate
        both inputs before mutation, so a malformed completed validation cannot
        partly replace current priority or add unbound native evidence.
        """
        native = self._native_parameters(native_metrics, duration=duration)
        independent = (np.empty(0) if validation is None else
                       self._validation_parameters(validation))
        current = set(np.r_[native, independent].tolist())
        self._parameters.update(current)
        self._current = current
        self._last_selection = None

    def select_parameters(self, *, fixed_parameters, regular_parameters, max_points):
        """Schedule bounded current failures fairly; retain all history.

        Fixed anchors and regular optimizer samples cover only their supplied
        exact representations. The caller supplies native float32 aliases for
        fixed anchors when those aliases represent actual existing queries.
        Among remaining current failures, choose the least frequently selected
        times first and break ties by time. Boolean collision failures and
        process errors in different units therefore never compete by magnitude.
        Pending failures remain explicit even at zero capacity. A changed query
        set is a gradient-work decision, not evidence of positive signed slack
        or a claim that an omitted historical failure cannot recur.
        """
        if type(max_points) is not int or max_points < 0:
            raise ValueError("The refinement witness capacity must be a nonnegative integer")
        fixed = self._validated_parameters(fixed_parameters, source="Fixed anchors")
        regular = self._validated_parameters(regular_parameters, source="Regular optimizer samples")
        covered = set(fixed.tolist()) | set(regular.tolist())
        eligible = self._current-covered
        ordered = sorted(eligible, key=lambda value: (self._selection_counts.get(value, 0), value))
        selected = set(ordered[:max_points])
        for parameter in selected:
            self._selection_counts[parameter] = self._selection_counts.get(parameter, 0)+1
        self._last_selection = {
            "scope": "optimizer query scheduling only; all history remains subject to independent validation",
            "policy": "current candidate failures; least selected count, then exact physical time",
            "slack_certified": False,
            "history_count": len(self._parameters),
            "current_failure_parameters": sorted(self._current),
            "covered_parameters": sorted(self._current & covered),
            "eligible_count": len(eligible), "capacity": max_points,
            "selected_parameters": sorted(selected),
            "pending_parameters": sorted(eligible-selected),
            "selection_counts": [{"parameter": value, "count": self._selection_counts.get(value, 0)}
                                 for value in sorted(eligible)]}
        return np.asarray(sorted(selected), dtype=float)
# END finite optimizer batches never delete evidence or relax final acceptance.


# BEGIN retain each native hard failure's time before constraint-column folding.
def capture_component_failures(components, times, progress, joints, position, quaternion, *,
                               normalized_time=None, mode="exact"):
    """Capture process/self residual times before anchor columns are folded.

    Inputs use the already ordered native query layout ``[batch, time, ...]``.
    Each component's trailing dimensions are reduced only within that same
    state. The resulting rows share the existing scene-trace schema, so the
    caller can attach them to the exact named joint-state trace without
    repeating FK, process evaluation or a source-geometry query.
    """
    cpu = lambda value: value.detach().cpu().numpy().copy()
    shape = tuple(times.shape)
    if len(shape) != 2:
        raise ValueError("Failure capture requires native [batch, time] query times")
    rows, state, parameters = [], None, None
    for name, tensor in components.items():
        values = cpu(tensor)
        if values.ndim < 2 or values.shape[:2] != shape:
            raise ValueError("Hard component samples must match their physical query times")
        values = values.reshape(*shape, -1).max(axis=-1)
        for batch, column in np.argwhere((values > 0.) | ~np.isfinite(values)):
            if state is None:
                state = [cpu(value) for value in (times, progress, joints, position, quaternion)]
                if any(value.shape[:2] != shape for value in state):
                    raise ValueError("Failure state samples must match their physical query times")
                if normalized_time is not None:
                    parameters = cpu(normalized_time)
                    if parameters.shape != shape:
                        raise ValueError("Normalized failure times must match the native query layout")
            time_s, phase, q, p, rotation = (value[batch, column] for value in state)
            row = {"component": name, "raw_residual": float(values[batch, column]),
                "batch_index": int(batch), "ordered_query_index": int(column),
                "expanded_query_index": int(column), "mode": mode,
                "time_s": float(time_s), "progress": float(phase),
                "permission_progress": float(phase), "joint_position": q.tolist(),
                "tool_position": p.tolist(), "tool_quaternion_wxyz": rotation.tolist()}
            if parameters is not None:
                row["normalized_time"] = float(parameters[batch, column])
            rows.append(row)
    return rows
# END all discovered physical failure times survive candidate and stage changes.


def scene_component_failures(components, times, progress, permission, joints,
                             position, quaternion, *, modes):
    """Retain every positive/nonfinite hard component in its original mode.

    Unknown observation coverage remains informational under the existing
    scene law. Missing release anchors are hard requirements under that same
    law. A mode index refers to the pre-fold query, never to its maximum.
    """
    cpu = lambda value: value.detach().cpu().numpy().copy()
    hard = [(name, value) for name, value in components.items()
            if not name.startswith("unknown/") or name.startswith("unknown/release_anchor/")]
    result = []
    state = None
    width = times.shape[1] // len(modes)
    for name, tensor in hard:
        values = cpu(tensor)
        for batch, column in np.argwhere((values > 0) | ~np.isfinite(values)):
            if state is None:
                state = [cpu(value) for value in (times, progress, permission, joints, position, quaternion)]
            time_s, phase, permitted, q, p, rotation = (value[batch, column] for value in state)
            result.append({"component": name, "raw_residual": float(values[batch, column]),
                "batch_index": int(batch), "ordered_query_index": int(column % width),
                "expanded_query_index": int(column),
                "mode": modes[column // width], "time_s": float(time_s),
                "progress": float(phase), "permission_progress": float(permitted),
                "joint_position": q.tolist(), "tool_position": p.tolist(),
                "tool_quaternion_wxyz": rotation.tolist()})
    return result


def bind_trace_to_native_seed(trace, state, seed_index, joint_names):
    """Bind by complete typed q/dq bytes, independently of native seed ranking.

    Native result sorting can permute seeds after metrics evaluation. Comparing
    the original named states avoids attributing another seed's scene witness.
    Identical q/dq rows share the same fixed-clock scene evaluation.
    """
    if trace is None:
        return {"status": "unavailable", "reason": "No source metrics trace was retained"}
    if "error" in trace:
        return {"status": "error", "error": trace["error"]}
    columns = [state.joint_names.index(name) for name in joint_names]
    arrays = [getattr(state, name)[0, seed_index].detach().cpu().numpy()[:, columns]
              for name in ("position", "velocity")]
    matches = [index for index in range(len(trace["input_position"]))
        if all(actual.dtype == stored[index].dtype and actual.shape == stored[index].shape
               and actual.tobytes() == stored[index].tobytes()
               for actual, stored in zip(arrays, (trace["input_position"], trace["input_velocity"])))]
    if not matches:
        return {"status": "unmatched", "reason": "No exact named q/dq input matches this result seed"}
    index = matches[0]
    failures = deepcopy([row for row in trace["component_failures"] if row["batch_index"] == index])
    witnesses = trace["world_geometry"].get("failure_witnesses", [])
    references = {(row["batch_index"], row["expanded_query_index"]): row["witness_index"]
                  for row in trace["world_geometry"].get("failure_witness_references", [])}
    for row in failures:
        if row["component"] == "source_world_collision":
            reference = references[(row["batch_index"], row["expanded_query_index"])]
            row["geometry_witnesses"] = [deepcopy(witnesses[reference])]
    return {"status": "recorded", "source_batch_index": index,
        "component_failures": failures,
        "world_geometry": deepcopy(trace["world_geometry"]),
        "world_geometry_scope": "Worst witness across this metrics call; joint/time identity is explicit",
        "scope": "Original source components before mode maximum, constraint weighting and anchor folding"}
