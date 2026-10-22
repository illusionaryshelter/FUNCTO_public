"""Generate one robot joint trajectory for a complete grounded skill window.

cuRobo V2 owns forward kinematics, inverse kinematics, joint constraints and
the final cubic trajectory.  Skill and scene violations enter its optimizer
through the same cost-manager extension used for result evaluation.  No
perception model, CUDA context or robot connection is initialized on import.

The public result contains measured numerical residuals and original event
identities.  It does not manufacture grasp evidence or execute robot commands.
"""

# BEGIN complete-window robot planning and generation-time process constraints.
from __future__ import annotations

from copy import copy, deepcopy
from dataclasses import asdict, dataclass, field, fields, replace
import gc
import time
from threading import Lock, get_ident
from uuid import uuid4
from types import SimpleNamespace

import numpy as np
from scipy.interpolate import BSpline, CubicHermiteSpline
from .planning_problem import event_occurrence_key
from .collision_records import encode_array
# BEGIN physical memory ownership is independent of optimizer workspace size.
from .resource_admission import (
    ResourceLimitError, require_host_memory, require_device_memory,
    DEFAULT_MAX_HOST_RSS_BYTES, MAX_HOST_RSS_BYTES,
    DEFAULT_MAX_DEVICE_USED_BYTES, MAX_DEVICE_USED_BYTES,
)
# END source geometry and native CUDA buffers use the same typed refusals.

# BEGIN final validation separates total finite work from live geometry storage.
_VALIDATION_CHUNK_SIZE = 256
_NATIVE_STAGE_POINT_LIMIT = 8192
# BEGIN byte admission is owned by actual record/scene consumers.
# No unused byte constants here may be mistaken for enforced protections.
# END finite-work limits above remain active.
# END increasing final work cannot enlarge native anchor/search batches.

# BEGIN reusable physical-domain contract for native numerical iterates.
from .physical_joint_domain import (CubicJointDomain, JointPositionDomain,
                                    JointDomainInitializationError, cubic_position_report)
# BEGIN one owned instantaneous-boundary representation for every native state.
from .cubic_trajectory import CubicBasis, cubic_transition_config_type
# END cuRobo still owns the optimizer, FK and original model constraints.
# END importing this CPU module does not allocate any CUDA workspace.


# BEGIN one exact output grid for the unchanged physical cubic clock.
def _fixed_spline_grid(*, intervals, native_dt, interpolation_steps, interpolation_dt):
    """Admit an output grid only when it preserves the native physical clock.

    Regular and dense samples must include every polynomial breakpoint on
    identical float32 and saved binary64 time axes. Rounding only proposes
    an integer subdivision count; it never authorizes retiming an input.
    """
    if (type(intervals) is not int or intervals < 1
            or type(interpolation_steps) is not int or interpolation_steps < 1):
        raise ValueError("Native spline intervals and subdivisions must be positive integers")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        raw_dt = np.float32(native_dt)
        output_dt = np.float32(interpolation_dt)
        knot_dt = np.float32(raw_dt * np.float32(interpolation_steps))
    if (not np.isfinite([raw_dt, output_dt, knot_dt]).all()
            or min(raw_dt, output_dt, knot_dt) <= 0):
        raise ValueError("Spline timing requires finite positive native float32 intervals")
    subdivisions = int(round(float(knot_dt) / float(output_dt)))
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        reconstructed = np.float32(output_dt * np.float32(subdivisions))
    if (subdivisions < 1 or reconstructed != knot_dt
            or float(output_dt) * subdivisions != float(knot_dt)
            or float(raw_dt) * interpolation_steps != float(knot_dt)):
        raise ValueError("Exact output grid must preserve the native spline knot interval; retiming is unsupported")
    count = intervals * subdivisions + 1
    duration = intervals * float(knot_dt)
    if (count - 1) * float(output_dt) != duration:
        raise ValueError("Exact output grid changed the native spline duration")
    return {"native_dt": float(raw_dt), "knot_dt": float(knot_dt),
            "interpolation_dt": float(output_dt), "subdivisions": subdivisions,
            "valid_count": count, "duration": duration}
# END incompatible grids never change dt, controls, or boundary derivatives.


# BEGIN a requested maximum spacing derives one exact native output grid.
def _derived_spline_grid(*, intervals, native_dt, interpolation_steps, interpolation_dt):
    """Choose exact subdivisions within the requested float32 spacing bound.

    ``interpolation_dt`` is a maximum spacing at request admission. A power
    of two subdivision preserves the represented knot duration exactly; the
    resulting grid is then checked by the same strict actual-clock validator
    used by native, event, finger and exported-curve consumers.
    """
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        limit = np.float32(interpolation_dt)
        knot_dt = np.float32(np.float32(native_dt) * np.float32(interpolation_steps))
    if not np.isfinite([limit, knot_dt]).all() or min(limit, knot_dt) <= 0:
        raise ValueError("Spline timing requires finite positive native float32 intervals")
    needed = max(1, int(np.ceil(float(knot_dt) / float(limit))))
    subdivisions = 1 << (needed - 1).bit_length()
    return _fixed_spline_grid(intervals=intervals, native_dt=native_dt,
        interpolation_steps=interpolation_steps, interpolation_dt=float(knot_dt) / subdivisions)
# END decimal request spacings are not themselves required to divide a knot.


@dataclass(frozen=True)
class PlanningBudget:
    """Explicit local resource and numerical-work limits, for a single request.

    These defaults bound the current 16 GB RAM / 12 GB GPU integration. More
    seeds or a larger batch must not be silently allocated by native defaults.
    The interpolation buffer covers the largest allowed native knot interval.
    Native solvers use eager execution for the dynamic process/scene tensors.
    ``max_task_anchors`` is the fixed semantic-anchor capacity. Native queries
    and bounded search share the checked scene/native allocation budgets.
    ``maximum_trajectory_dt`` fixes both native time bounds for this request:
    process feasibility and evidence stops take precedence over compression.
    ``interpolation_dt`` is the requested maximum sample spacing until time
    binding replaces it with the exact selected spacing for all consumers.
    """

    ik_seeds: int = 8
    trajectory_seeds: int = 2
    # BEGIN retain the interval budget while releasing two entrance equalities.
    # N keeps N+4 physical intervals; the actual action has N+2 coefficients.
    knots: int = 8
    # END this is a representation dimension change, not skill-specific tuning.
    interpolation_steps: int = 4
    ik_iterations: int = 40
    trajectory_iterations: int = 100
    interpolation_dt: float = 0.025
    maximum_trajectory_dt: float = 0.2
    interpolation_buffer_size: int = 512
    constraint_weight: float = 100000.0
    # BEGIN reserve optional refinement work separately from required evidence.
    max_task_anchors: int = 256
    # END reserve optional refinement work separately from required evidence.
    max_robot_spheres: int = 256
    max_robot_dof: int = 16
    max_robot_links: int = 64
    # BEGIN GPU scene workspaces exclude all resident CPU source geometry.
    max_scene_gpu_workspace_bytes: int = 1024**3
    max_host_rss_bytes: int = DEFAULT_MAX_HOST_RSS_BYTES
    max_device_used_bytes: int = DEFAULT_MAX_DEVICE_USED_BYTES
    # END larger physical limits require an explicit request, never a fallback.
    max_validation_points: int = 4096
    validation_joint_step: float = 0.025
    # BEGIN metric sampling also has meaning for analytic solids without voxels.
    validation_spatial_step_m: float = 0.001
    # END this relative-displacement bound is not a physical collision margin.
    search_seconds: float = 3.0
    maximum_trajectory_solves: int = 2
    # BEGIN wall work is cumulative across binding attempts, independently of physics.
    planning_seconds: float = 120.0
    # END a stage may finish over this limit, but its candidate cannot be accepted.
    random_seed: int = 123

    def __post_init__(self):
        if (not 1 <= self.trajectory_seeds <= 2 or not self.trajectory_seeds <= self.ik_seeds <= 8
                or not 6 <= self.knots <= 24 or not 1 <= self.interpolation_steps <= 8
                or not 1 <= self.ik_iterations <= 200 or not 1 <= self.trajectory_iterations <= 300
                # BEGIN admit bounded complete-volume robot cover refinement.
                # This is storage capacity, not a collision clearance or an
                # acceptance tolerance. Scene/native memory preflight remains
                # mandatory for every actual sphere count and query horizon.
                or not 1 <= self.max_robot_spheres <= 640
                # END geometric coverage and cumulative work limits are unchanged.
                or not 1 <= self.max_robot_dof <= 16
                or not 1 <= self.max_robot_links <= 64
                or not 32 <= self.max_validation_points <= 65536
                or not 1 <= self.maximum_trajectory_solves <= 2):
            raise ValueError("Planning budget exceeds the bounded local single-batch configuration")
        # BEGIN semantic anchors have one fixed resident capacity.
        if (type(self.max_task_anchors) is not int
                or not 2 <= self.max_task_anchors <= self.max_validation_points):
            raise ValueError("Task anchors must fit the finite native point budget")
        # END bound optional samples without capping a skill's source knots.
        # BEGIN physical host/device caps are separate from GPU scene capacity.
        if (type(self.max_scene_gpu_workspace_bytes) is not int
                or not 0 <= self.max_scene_gpu_workspace_bytes <= 6*1024**3):
            raise ValueError("GPU scene workspaces must remain within the local six-GiB allocation budget")
        if (type(self.max_host_rss_bytes) is not int
                or not 0 < self.max_host_rss_bytes <= MAX_HOST_RSS_BYTES):
            raise ValueError("Host RSS must remain within the explicit twelve-GiB maximum")
        if (type(self.max_device_used_bytes) is not int
                or not 0 < self.max_device_used_bytes <= MAX_DEVICE_USED_BYTES):
            raise ValueError("Device usage must remain within the explicit eight-GiB maximum")
        # END defaults retain the existing half-host/half-device resource limits.
        for name in ("interpolation_dt", "maximum_trajectory_dt", "constraint_weight", "validation_joint_step",
                     "validation_spatial_step_m", "planning_seconds"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not np.isfinite(self.search_seconds) or self.search_seconds < 0:
            raise ValueError("search_seconds must be finite and nonnegative")
        # BEGIN derive and admit the actual grid before native buffer allocation.
        # The selected cubic has degree+1 virtual intervals. Output sampling
        # must preserve its original physical clock, including nonzero dq0.
        grid = _derived_spline_grid(intervals=self.knots + 4,
            native_dt=self.maximum_trajectory_dt, interpolation_steps=self.interpolation_steps,
            interpolation_dt=self.interpolation_dt)
        maximum_count = grid["valid_count"]
        # END retain all original buffer, validation and workspace bounds.
        native_count = (self.knots + 4) * self.interpolation_steps + 1
        if (self.interpolation_buffer_size < maximum_count
                or self.interpolation_buffer_size <= native_count
                or self.interpolation_buffer_size > 2048):
            raise ValueError("Interpolation buffer must cover the bounded cubic trajectory and differ from its rollout horizon")


@dataclass
class JointTrajectory:
    """A candidate cubic joint curve and its exact skill-progress mapping.

    All arrays are CPU NumPy arrays in named active-joint order.  The cubic
    reconstruction preserves the supplied q/dq polynomial, whether provided
    by a caller or exported from native knot subdivisions. Sampling never
    performs another fit or smoothing pass.
    """

    joint_names: tuple[str, ...]
    times: np.ndarray
    position: np.ndarray
    velocity: np.ndarray
    progress_u: np.ndarray
    progress_s: np.ndarray
    events: tuple[dict, ...]
    conditions: tuple[dict, ...]
    residuals: dict[str, float]
    # BEGIN archive the exact independently controlled finger reference.
    finger_trajectory: object | None = None
    satisfied_dependencies: tuple[str, ...] = ()
    # END a missing hand curve is valid only for plans without hand events.
    _curve: CubicHermiteSpline = field(init=False, repr=False)

    def __post_init__(self):
        self._curve = CubicHermiteSpline(self.times, self.position, self.velocity, axis=0)

    @property
    def duration(self):
        return float(self.times[-1])

    @property
    def next_verification(self):
        """The first original evidence barrier, without claiming it was met."""
        barriers = [event for event in self.events if event["requires_evidence"] is not None
                    and not (event["requires_evidence"] in self.satisfied_dependencies
                             and event["progress"] <= self.progress_s[0])]
        return deepcopy(min(barriers, key=lambda event: event["time"])) if barriers else None

    @property
    def executable_until(self):
        """Limit of this conditional plan before new actual feedback is needed."""
        barrier = self.next_verification
        return self.duration if barrier is None else float(barrier["time"])

    def sample(self, times):
        # BEGIN one exact polynomial query owns all three control references.
        """Return q, dq, ddq and progress at the same finite curve time.

        SciPy evaluates an internal breakpoint on its right-hand piece; the
        final endpoint uses the last piece's left limit.  Thus acceleration
        is the analytic, one-sided second derivative of this saved C1 curve,
        not an estimated derivative, a new smoothing pass, or actual qacc.
        Independent acceleration validation still checks both knot sides.
        """
        query = np.asarray(times, dtype=float)
        if not np.isfinite(query).all() or np.any(query < 0) or np.any(query > self.duration):
            raise ValueError("Trajectory queries must lie within its finite time interval")
        return (self._curve(query), self._curve(query, 1), self._curve(query, 2),
                np.interp(query / self.duration, self.progress_u, self.progress_s))
        # END no q-only or derivative-free control sampling fallback.


# BEGIN continuous acceleration bounds on the exact exported Hermite curve.
def cubic_derivative_report(trajectory, acceleration_bounds, *, joint_names, until=None):
    """Check each scalar cubic's acceleration range, including both knot sides.

    Acceleration is affine within a cubic piece, so its extrema occur at that
    piece's endpoints. An internal knot has two acceleration limits: evaluating
    only ``curve(knot, 2)`` would lose the previous piece's left limit. A supplied
    ``until`` clips the last piece at the original execution barrier; future
    pieces do not enter that prefix report. Limits are bound by joint identity
    and compared in binary64 without adding a numerical or physical allowance.

    The input remains the original CubicHermiteSpline. Its C1 construction and
    raw floating-point knot jumps are reported; acceleration jumps do not add a
    C2 requirement. This checks a reference polynomial, not actual robot motion,
    continuous collision, jerk at jumps, or actuator forces.
    """
    names, curve_names = tuple(joint_names), tuple(trajectory.joint_names)
    bounds = np.asarray(acceleration_bounds, dtype=float)
    if (not names or len(set(names)) != len(names) or len(set(curve_names)) != len(curve_names)
            or set(names) != set(curve_names) or bounds.shape != (2, len(names))
            or not np.isfinite(bounds).all() or np.any(bounds[0] > bounds[1])):
        raise ValueError("Continuous acceleration checks require complete finite named lower/upper bounds")
    bounds = bounds[:, [names.index(name) for name in curve_names]]
    times = np.asarray(trajectory.times, dtype=float)
    position, velocity = np.asarray(trajectory.position, dtype=float), np.asarray(trajectory.velocity, dtype=float)
    if (times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all() or times[0] != 0.
            or np.any(np.diff(times) <= 0.) or position.shape != (len(times), len(names))
            or velocity.shape != position.shape or not np.isfinite(position).all()
            or not np.isfinite(velocity).all()):
        raise ValueError("Continuous acceleration checks require a complete finite ordered Hermite curve")
    curve = CubicHermiteSpline(times, position, velocity, axis=0)
    if (not np.array_equal(curve.x, trajectory._curve.x)
            or not np.array_equal(curve.c, trajectory._curve.c)):
        raise ValueError("Exported arrays differ from the executable Hermite polynomial")
    stop = float(times[-1] if until is None else until)
    if not np.isfinite(stop) or not 0. <= stop <= times[-1]:
        raise ValueError("Derivative report interval must lie inside the original curve")
    indices = np.flatnonzero(times[:-1] < stop)
    if not len(indices):
        # A zero-length prefix reports the initial right derivative only.
        indices = np.array([0])
    ends = np.minimum(times[indices+1], stop)
    widths = (ends-times[indices])[:, None]
    coefficients = curve.c[:, indices]
    acceleration = np.stack((2*coefficients[1],
                             6*coefficients[0]*widths+2*coefficients[1]), axis=1)
    values = acceleration.reshape(-1, len(names))
    endpoint_times = np.stack((times[indices], ends), axis=1).reshape(-1)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite analytic acceleration on the exported polynomial")
    columns = np.arange(len(names))
    low_indices, high_indices = values.argmin(axis=0), values.argmax(axis=0)
    minimum, maximum = values[low_indices, columns], values[high_indices, columns]
    lower_excess, upper_excess = np.maximum(bounds[0]-minimum, 0.), np.maximum(maximum-bounds[1], 0.)
    excess = np.stack((lower_excess, upper_excess))
    maximum_excess = float(excess.max())
    witness = None
    if maximum_excess > 0.:
        bound_index, joint_index = np.unravel_index(excess.argmax(), excess.shape)
        endpoint = (low_indices if bound_index == 0 else high_indices)[joint_index]
        witness = {"joint_name": curve_names[joint_index], "time_s": float(endpoint_times[endpoint]),
                   "segment_index": int(indices[endpoint//2]),
                   "side": "segment_start" if endpoint % 2 == 0 else "segment_end_left_limit",
                   "acceleration": float(values[endpoint, joint_index]),
                   "bound": float(bounds[bound_index, joint_index]), "excess": maximum_excess}

    # Only internal breakpoints of the requested interval have two executed
    # sides. In particular, a barrier exactly at a knot excludes future ddq.
    internal = np.flatnonzero(times[1:-1] < stop)
    c, h = curve.c, np.diff(times)[:, None]
    q_end = ((c[0]*h+c[1])*h+c[2])*h+c[3]
    dq_end = (3*c[0]*h+2*c[1])*h+c[2]
    ddq_end = 6*c[0]*h+2*c[1]
    jumps = {
        "position": c[3, 1:]-q_end[:-1],
        "velocity": c[2, 1:]-dq_end[:-1],
        "acceleration": 2*c[1, 1:]-ddq_end[:-1],
    }
    continuity = {"c1_by_hermite_construction": True, "internal_breakpoints": len(internal),
                  "acceleration_jump_is_rejection": False}
    for label, jump in jumps.items():
        selected = jump[internal]
        if not np.isfinite(selected).all():
            raise ValueError("Nonfinite derivative jump on the exported polynomial")
        continuity[f"{label}_maximum_absolute_jump"] = (
            np.abs(selected).max(axis=0) if len(selected) else np.zeros(len(names))).tolist()
    return {
        "scope": "binary64 polynomial acceleration ranges and knot continuity; no actual-motion or collision certification",
        "checked_interval_s": [0., stop], "full_curve_duration_s": float(times[-1]),
        "joint_names": list(curve_names), "checked_piece_count": len(indices),
        "acceleration_bounds": bounds.tolist(),
        "acceleration": {"minimum": minimum.tolist(), "maximum": maximum.tolist(),
                         "minimum_time_s": endpoint_times[low_indices].tolist(),
                         "maximum_time_s": endpoint_times[high_indices].tolist(),
                         "maximum_absolute": np.maximum(np.abs(minimum), np.abs(maximum)).tolist(),
                         "lower_excess": lower_excess.tolist(), "upper_excess": upper_excess.tolist(),
                         "maximum_bound_excess": maximum_excess, "bounds_satisfied": maximum_excess == 0.,
                         "maximum_violation": witness},
        "continuity": continuity,
    }
# END continuous acceleration bounds; no C2 requirement or curve modification.


@dataclass
class PlanningResult:
    success: bool
    trajectory: JointTrajectory | None
    diagnostics: dict
    # BEGIN bind accepted candidates to their executable input and clock.
    execution_binding: dict | None = None
    # BEGIN even a failed initialization returns the reusable same-request ledger.
    request_ledger: dict | None = None
    # END formal execution authorization remains independent of retained history.
    # END failed results never carry an execution authorization.


# BEGIN size required evidence and native anchor buffers from the actual request.
def _anchor_capacity(progress_map, anchors, budget):
    """Admit semantic anchors into a fixed native workspace; never add witnesses."""
    required = len(progress_map.anchor_parameters(anchors))
    if required > budget.max_task_anchors:
        raise MotionPlanningError("capacity_exceeded", "Task anchors exceed the resident backend capacity")
    return budget.max_task_anchors





# BEGIN one value carries independent host/device ownership through rebuilds.
@dataclass(frozen=True)
class _PlanningWorkspace:
    """Scene capacity plus one live query, with optional native graph pools."""

    cpu_shared_bytes: int
    cpu_live_bytes: int
    gpu_eager_bytes: int
    gpu_graph_bytes: int
    anchor_fk_bytes: int
    trajectory_graph: bool

    @property
    def cpu_bytes(self):
        return self.cpu_shared_bytes+self.cpu_live_bytes

    @property
    def gpu_bytes(self):
        return self.gpu_eager_bytes+(self.gpu_graph_bytes if self.trajectory_graph else 0)

    @property
    def total_bytes(self):
        """Diagnostic scene total; it is never a CUDA allocation request."""
        return self.cpu_bytes+self.gpu_bytes
# END graph admission changes only the two separately retained GPU pools.


def _planning_workspace_bytes(scene, robot_metadata, budget, anchor_capacity):
    """Return CPU/GPU scene capacity, anchor FK and graph eligibility.

    The main native rollout retains its original state-sampling horizon. Each trajectory
    manager evaluates the required anchors with a separate native FK model,
    then folds their penalties into that rollout's existing final cost column.
    """
    horizon = (budget.knots+4)*budget.interpolation_steps+1
    # Native LBFGS has two rollout instances and four line-search particles.
    # Include optimization/metrics, dense metrics and saved-curve verification.
    # BEGIN reference-only IK owns no independent scene query workspaces.
    # Reference IK retains native pose/self-collision and physical q-domain
    # costs. Scene/coverage queries belong to complete trajectory candidates.
    # The dense/native metrics and final CPU-solid validation never construct
    # a coverage FK or its backward graph. Their other scene queries remain.
    allocation_shapes = (
        (4, budget.trajectory_seeds, horizon+anchor_capacity, True),
        (4, budget.trajectory_seeds*4, horizon+anchor_capacity, True),
        (2, budget.trajectory_seeds, budget.interpolation_buffer_size+anchor_capacity, False),
        (1, 1, min(budget.max_validation_points, _VALIDATION_CHUNK_SIZE), False),
        (2, 1, anchor_capacity, False),
    )
    # END source reference IK no longer reserves unused scene/coverage buffers.
    # BEGIN use actual sequential native forward/backward memory lifetimes.
    # Scene switches still query every exact/left/right sample. Only native
    # query buffers persist across rollouts. Installed gradient_opt_core
    # executes evaluate_action then backward(retain_graph=False) before another
    # initial/exploration rollout; metrics consume its detached best action.
    # The solver factories keep CPU metrics outside capture. Eligible owned
    # optimizer cores additionally retain two private graph pools below.
    # The final two anchor-shaped reserves also cover the simultaneous OMPL
    # chart-scene stack: at most anchor_capacity one-state scene clones, whose
    # aggregate sphere buffers fit one such reserve. Scalar slots are counted
    # below per state as well, so this split cannot hide many small caches.
    scene_factor = 3 if scene.required_progress_anchors else 1
    # BEGIN the same observed-field workspace applies to each query layout.
    allocations = [(copies, scene.estimated_workspace_allocation(batch, scene_factor*length))
                   for copies, batch, length, _ in allocation_shapes]
    # END shared immutable fields and per-state witness storage are distinct.
    # BEGIN CPU residency cannot consume the GPU scene budget or graph pools.
    gpu_eager_bytes = (sum(copies*value["gpu_retained"] for copies, value in allocations)
                       + max(value["gpu_live"] for _, value in allocations)
                       + max(value["gpu_shared"] for _, value in allocations))
    cpu_shared_bytes = max(value["cpu_shared"] for _, value in allocations)
    cpu_live_bytes = max(value["cpu_live"] for _, value in allocations)
    # END geometry/query sets and the unchanged hard memory budget are preserved.
    # END all live optimization scratch remains under the unchanged hard budget.
    if gpu_eager_bytes > budget.max_scene_gpu_workspace_bytes:
        # BEGIN report the actual bound so a refused trial is reproducible.
        raise ResourceLimitError("scene_gpu_workspace", gpu_eager_bytes, budget.max_scene_gpu_workspace_bytes,
            description="Independent GPU scene query workspaces exceed the declared local allocation budget")
        # END reporting a shortfall does not alter the allocation contract.
    # BEGIN choose capture scope before allocation, preserving eager admission.
    # Each trajectory entry/inner executor retains a private pool in addition
    # to the ordinary live scratch used by warmup and metrics. A finite contact
    # scene needs CPU cost geometry. A scene whose two graph pools exceed the
    # declared capacity retains the unchanged native eager optimizer as well;
    # neither case changes required queries, arithmetic or iteration counts.
    # BEGIN CPU source validation has no allocations inside either graph pool.
    graph_live = max(value["gpu_graph_live"] for (_, value), (_, _, _, guidance)
                     in zip(allocations, allocation_shapes) if guidance)
    # END the ordinary live bound above still reserves all validation scratch.
    graph_scene_bytes = gpu_eager_bytes+2*graph_live
    trajectory_graph = False
    # END no independent captured pool is treated as reusable eager scratch.
    dof, links, spheres = (len(robot_metadata["active_joint_names"]),
                           robot_metadata["links"], robot_metadata["spheres"])
    # Installed KinematicsFusedFunction.create_buffers uses float32 and this
    # planner requests exactly one tool frame. Per [batch, anchor] it allocates
    # pose/COM values and gradients (22 floats), sphere values/gradients (8*S),
    # a 6*D Jacobian and two D joint gradients, and 12*L cumulative transforms.
    # The same conservative rollout counts above give 22*K anchor batches.
    # Main FK, general pose/sphere transform graphs, verification, model and
    # optimizer/trajectory storage retain the separate two-GiB allowance in
    # the device-memory preflight below. The scene budget is specifically for
    # dedicated geometry queries, not a claim about all planner allocations.
    # BEGIN reserve the independent self-collision buffers at exact anchors.
    # SelfCollisionCost stores one float output, four float gradients and one
    # uint8 sparse flag per sphere, plus two int16 indices/one float per block.
    # Its smallest native block covers 2048 pairs; all unordered sphere pairs
    # bound the blocks without constructing a dense pair matrix. Production
    # costs do not request the optional per-pair diagnostic distance tensor.
    self_blocks = (spheres*(spheres-1)//2+2047)//2048
    self_bytes_per_state = 4+17*spheres+8*self_blocks
    bytes_per_state = 4*(22+8*spheres+8*dof+12*links)+self_bytes_per_state
    # END anchor collision buffers share the existing finite admission limit.
    anchor_fk_bytes = 22*budget.trajectory_seeds*anchor_capacity*bytes_per_state
    return _PlanningWorkspace(int(cpu_shared_bytes), int(cpu_live_bytes), int(gpu_eager_bytes),
                              int(2*graph_live), int(anchor_fk_bytes), bool(trajectory_graph))
    # END current device usage is checked separately before any native allocation.


def _require_planning_host_memory(cpu_live_bytes, *, encoded_record_bytes_upper, max_host_rss_bytes):
    """Reserve new CPU work and candidate records.

    Three encoded payloads conservatively cover the new validation record,
    its candidate snapshot and serialization copy. Earlier candidates and
    resident source geometry already appear in measured RSS. Rechecking at
    each export therefore accounts for accumulated history without charging
    the same existing map or previous record as a fresh allocation.
    """
    # BEGIN record retention participates in the same total host admission.
    require_host_memory(cpu_live_bytes+3*encoded_record_bytes_upper,
        pending_existing_bytes=0,
        max_rss_bytes=max_host_rss_bytes)
    # END the independent encoded-record and process-tensor bounds still apply.


def _require_planning_memory(torch, device, workspace, *, max_device_used_bytes):
    """Admit Torch buffers without promising cached memory to native allocators.

    CollisionBuffer and native FK buffers use torch.zeros. Unused capacity in
    this process's Torch allocator can therefore serve those tensors after a
    schedule replacement. Existing Mapper fields are already allocated. The
    separate two-GiB allowance covers main FK, general pose/sphere transform
    graphs, solver/model and trajectory storage; it must still fit driver-
    reported free memory independently of the geometry-query tensor cache.

    CPU source geometry is already host-admitted and never enters this check.
    If the extra graph pools do not fit but the eager solver does, retain the
    existing eager path with the same geometry, samples and optimizer work.
    """
    # BEGIN the common guard accounts for whole-device usage at every boundary.
    def refusal(candidate):
        try:
            require_device_memory(torch, device, native_bytes=2*1024**3,
                torch_bytes=candidate.gpu_bytes+candidate.anchor_fk_bytes,
                max_device_used_bytes=max_device_used_bytes)
        except ResourceLimitError as error:
            return error
        return None

    error = refusal(workspace)
    if error is not None and workspace.trajectory_graph:
        eager = replace(workspace, trajectory_graph=False)
        eager_error = refusal(eager)
        if eager_error is None:
            return eager
        error = eager_error
    if error is not None:
        raise error
    return workspace
    # END CUDA API errors retain their original exception identity.
# END size required evidence and native anchor buffers from the actual request.


def _bounded_robot_configuration(robot, problem, scene, budget):
    """Admit a pure named-joint tree before native allocation.

    The native solvers own joint-domain constraints and tool FK. Registered
    observation fields own all entity collision geometry in the scene.
    """
    from pathlib import Path
    from curobo._src.util.config_io import load_yaml
    from .observed_robot import pure_kinematics
    data = deepcopy(robot if isinstance(robot, dict) else load_yaml(str(robot)))
    kinematics = data["robot_cfg"]["kinematics"]
    path = Path(kinematics["urdf_path"])
    if path.stat().st_size > 4*1024**2:
        raise ValueError("Kinematic description exceeds its metadata budget")
    tree = pure_kinematics(path)
    links = {row.attrib["name"] for row in tree.findall("link")}
    joints = {row.attrib["name"] for row in tree.findall("joint") if row.attrib["type"] != "fixed"}
    locks = kinematics.get("lock_joints") or {}
    if (len(links) > budget.max_robot_links or len(joints) > budget.max_robot_dof
            or not set(scene.required_robot_links) <= links or problem.tool_link not in links
            or joints-set(locks) != set(problem.joint_names)):
        raise ValueError("Pure robot tree must fit the named joint/link planning contract")
    if kinematics.get("extra_links") or kinematics.get("add_object_link", False):
        raise ValueError("Observed robot kinematics uses the supplied tree without extra links")
    # The maintained native configuration parser requires an explicit empty
    # sphere dictionary even when sphere loading is disabled. It allocates
    # no physical collision primitives; fields remain the sole scene model.
    kinematics.update(load_meshes=False, load_collision_spheres=False,
        collision_spheres={}, collision_link_names=[], extra_collision_spheres={},
        self_collision_ignore={}, self_collision_buffer={},
        tool_frames=[problem.tool_link])
    return data, {"active_joint_names": sorted(joints-set(locks)), "unlocked_dof": len(joints),
                  "links": len(links), "selected_link_names": sorted(links), "spheres": 0}


@dataclass(frozen=True)
class _ProgressMap:
    """A fixed, monotone phase schedule; repeated s values encode pauses."""

    u: np.ndarray
    s: np.ndarray
    # BEGIN original intent and device activation are distinct bound facts.
    event_clock: dict = field(default_factory=dict, compare=False)
    # END only initial binding creates this record.

    def __post_init__(self):
        if (self.u.ndim != 1 or self.s.shape != self.u.shape or len(self.u) < 2
                or not np.isfinite(self.u).all() or not np.isfinite(self.s).all()
                or self.u[0] != 0 or self.u[-1] != 1 or self.s[0] < 0
                or np.any(np.diff(self.u) <= 0) or np.any(np.diff(self.s) < 0)
                or np.any(np.diff(self.u.astype(np.float32)) <= 0)):
            raise ValueError("A progress map needs increasing u and nondecreasing s")

    def anchor_parameters(self, anchors):
        """Preserve both ends of an s plateau, including a paused event."""
        pairs = []
        for value in anchors:
            equal = np.flatnonzero(self.s == value)
            if len(equal):
                pairs.append((float(self.u[equal[0]]), float(value)))
                if equal[-1] != equal[0]:
                    pairs.append((float(self.u[equal[-1]]), float(value)))
            else:
                right = int(np.searchsorted(self.s, value, side="right"))
                if right == 0 or right == len(self.s):
                    raise ValueError("An event anchor is outside the planned progress interval")
                fraction = (value - self.s[right-1]) / (self.s[right] - self.s[right-1])
                pairs.append((float(self.u[right-1] + fraction * (self.u[right] - self.u[right-1])), float(value)))
        pairs = np.asarray(sorted(set(pairs)), dtype=float).reshape(-1, 2)
        # Required source knots and event boundaries must remain distinct in
        # the native float32 process tensors.  Reject an unrepresentable clock
        # instead of silently merging or moving a supplied hard anchor.
        if any(len(np.unique(pairs[:, column])) != len(np.unique(pairs[:, column].astype(np.float32)))
               for column in (0, 1)):
            raise ValueError("Required progress anchors exceed native float32 clock precision")
        return pairs


# BEGIN reference samples retain phase and motion within progress plateaus.
def _phase_aligned_parameters(source, target, parameters):
    """Map time parameters through equal progress, retaining a pause's phase.

    A progress value with a finite source plateau is not a single time: its
    interior motion must keep its relative position between both endpoints.
    An isolated source value uses the target's first arrival, so an added
    native verification hold cannot postpone its original arrival sample.
    """
    parameters = np.asarray(parameters, dtype=float)
    values = np.interp(parameters, source.u, source.s)
    result = []
    for parameter, value in zip(parameters, values):
        old = source.anchor_parameters([value])[:, 0]
        new = target.anchor_parameters([value])[:, 0]
        if len(old) == 2:
            if len(new) != 2:
                raise MotionPlanningError("unsupported_reference", "Target clock cannot retain reference plateau motion")
            phase = (parameter-old[0])/(old[-1]-old[0])
            result.append(new[0]+phase*(new[-1]-new[0]))
        else:
            result.append(new[0])
    return np.asarray(result)


def _candidate_sample_times(trajectory, budget):
    """Keep every original cubic piece and bound its query spacing/storage."""
    counts = np.maximum(1, np.ceil(np.diff(trajectory.times)/float(np.float32(budget.interpolation_dt))).astype(int))
    count = int(counts.sum())+1
    if count > min(budget.max_validation_points, budget.interpolation_buffer_size):
        raise MotionPlanningError("capacity_exceeded", "Candidate samples exceed the admitted native metric buffer")
    return np.r_[np.concatenate([np.linspace(left, right, count+1)[:-1]
        for left, right, count in zip(trajectory.times[:-1], trajectory.times[1:], counts)]), trajectory.duration]


def _reference_clock_issues(problem, trajectory, timestep):
    """Report required device occurrences outside the unchanged curve domain.

    The ordinary strict quantizer remains authoritative, including a source
    endpoint just beyond an exact tick in binary64. An unrepresented future
    occurrence is neither clipped, inferred from the endpoint, nor repaired
    by silently extending or retiming the supplied polynomial.
    """
    from .gripper_trajectory import _activation
    mapping = _ProgressMap(trajectory.progress_u, trajectory.progress_s)
    issues = []
    for event in problem.events:
        command = event.kind in {"open", "close", "hold"}
        if not (command or event.requires_evidence is not None or event.stop):
            continue
        if timestep is None:
            raise MotionPlanningError("unsupported_reference", "Reference commands require an explicit device timestep")
        intent = float(mapping.anchor_parameters([event.progress])[0, 0]*trajectory.duration)
        step, applied = _activation(intent, timestep)
        if applied > trajectory.duration or (command and applied == trajectory.duration):
            issues.append({"event_id": event.event_id, "skill_id": event.skill_id,
                "occurrence": event.occurrence, "intent_time_s": intent,
                "device_step": step, "device_time_s": applied, "curve_duration_s": trajectory.duration,
                "kind": "activation" if command else "consumption",
                "status": "outside_curve_unchecked"})
    return issues
# END neither time conversion nor metric sampling changes the source polynomial.


# BEGIN source rhythm proposes the initial clock before any solver is created.
def _initial_timing(problem, budget):
    """Quantize source duration, then sample its exact native clock densely.

    Source duration/rhythm are soft initialization information. The native
    interval cap bounds one request; an overlong demonstration may therefore
    propose a compressed clock. ``interpolation_dt`` bounds sample spacing at
    admission; the returned budget records the exact selected spacing. Hard
    event deadlines remain separate. Later refinement retains this clock.
    """
    source = problem.source_time_map
    seconds, progress = np.asarray(source["source_times"], float), np.asarray(source["progress"], float)
    if (seconds.ndim != 1 or seconds.shape != progress.shape or len(seconds) < 2
            or not np.isfinite(seconds).all() or np.any(np.diff(seconds) <= 0)):
        raise ValueError("Initial scheduling needs explicit increasing source seconds and their progress")
    source_duration = float(seconds[-1]-seconds[0])
    # BEGIN bind physical duration before choosing a representable output grid.
    # Fixing output_dt first admits only a few binary-exact knot durations
    # and can stretch a 9.85-second reference to 16 seconds. Instead round
    # native dt upward by at most a few float32 ulps, retaining suffix coverage
    # for continuations, and derive output_dt through exact power-of-two
    # division. The unchanged grid validator still checks every multiplication.
    maximum = _derived_spline_grid(intervals=budget.knots+4,
        native_dt=budget.maximum_trajectory_dt, interpolation_steps=budget.interpolation_steps,
        interpolation_dt=budget.interpolation_dt)
    proposed_dt = min(source_duration, maximum["duration"])/((budget.knots+4)*budget.interpolation_steps)
    native_dt = np.float32(proposed_dt)
    if float(native_dt) < proposed_dt:
        native_dt = np.nextafter(native_dt, np.float32(np.inf))
    # General admitted step counts include 3, 5, 6 and 7. At most the next
    # power of two float32 values contains a product exactly representable
    # at the larger exponent; the original admitted maximum is also exact.
    for _ in range(1 << (budget.interpolation_steps-1).bit_length()):
        knot_dt = np.float32(native_dt*np.float32(budget.interpolation_steps))
        if float(native_dt)*budget.interpolation_steps == float(knot_dt):
            break
        native_dt = np.nextafter(native_dt, np.float32(np.inf))
    if not 0 < float(native_dt) <= maximum["native_dt"]:
        raise ValueError("Source-time proposal has no exact representable device clock within the budget")
    # BEGIN source binding uses the same spacing derivation as budget admission.
    actual = _derived_spline_grid(intervals=budget.knots+4, native_dt=float(native_dt),
        interpolation_steps=budget.interpolation_steps, interpolation_dt=budget.interpolation_dt)
    selected = replace(budget, maximum_trajectory_dt=actual["native_dt"],
                       interpolation_dt=actual["interpolation_dt"])
    # END every later consumer receives the already selected actual spacing.
    # END capacity and strict native/dense clock equality remain mandatory.
    return selected, _ProgressMap((seconds-seconds[0])/source_duration, progress.copy())
# END no search path or candidate geometry can subsequently retime this binding.

# BEGIN quantize command-associated queries before freezing the binding.
def _bind_event_clock(problem, progress_map, duration, timestep=None):
    from .gripper_trajectory import _activation
    source = {}
    command_keys = []
    tick_progress = set()
    tick_entries = []
    for event in problem.events:
        identity = event_occurrence_key(event)
        parameters = progress_map.anchor_parameters([event.progress])[:, 0]
        intent = float(parameters[0]*duration)
        command = event.kind in {"open", "close", "hold"}
        needs_tick = command or event.requires_evidence is not None or event.stop
        if needs_tick and timestep is None:
            raise ValueError("Command timing requires the bound device control timestep")
        step, applied = _activation(intent, timestep) if needs_tick else (None, None)
        if needs_tick:
            if applied > duration or (command and applied == duration):
                raise ValueError("A bound command has no activation/effect coverage inside the valid trajectory")
            tick_entries.append((event, step, float(parameters[0]), applied/duration))
        source[identity] = {"event": asdict(event), "intent_time_s": intent, "activation_time_s": applied if command else None,
                            "consumption_time_s": applied if needs_tick and not command else None}
        if command:
            source[identity].update(bound_activation_step=step, request_activation_step=step,
                request_activation_time_s=applied, request_intent_time_s=intent,
                request_time_origin_step=0, device_timestep_s=float(timestep))
    # Multiple ordinary intentions on the same actuator tick share the
    # controller's existing last-intention policy. Preserve every identity;
    # supersession here is a schedule prediction, not an application fact.
    by_step = {}
    for entry in tick_entries:
        if entry[0].kind in {"open", "close", "hold"}:
            by_step.setdefault(entry[1], []).append(entry)
    superseded = set()
    true_goals = {goal.progress for name in ("pose_events", "coordinate_events", "supports")
                  for goal in getattr(problem, name)}
    for entries in by_step.values():
        ordered = sorted(entries, key=lambda item: (item[2], item[0].source_frame))
        for event, _, _, _ in ordered[:-1]:
            if (event.stop or event.requires_evidence is not None or event.progress in true_goals
                    or any(item.check_target == "activation" for item in event.timing)):
                raise ValueError("Device quantization supersedes a command with a required physical occurrence")
            key = event_occurrence_key(event)
            superseded.add(key)
            source[key]["scheduled_superseded_by"] = event_occurrence_key(ordered[-1][0])
    for event, _, before_u, after_u in tick_entries:
        if event_occurrence_key(event) not in superseded:
            command_keys.append((before_u, after_u))
            tick_progress.add(event.progress)
    # Independent functional events retain their own bound physical instant.
    # Only queries explicitly sharing a command's progress move to its tick;
    # ordinary reference samples between these anchors remain interpolated.
    independent = [event.progress for event in problem.events if event.progress not in tick_progress]
    independent += [goal.progress for name in ("pose_events", "coordinate_events", "supports")
                    for goal in getattr(problem, name) if goal.progress not in tick_progress]
    fixed = [(float(value), float(value)) for value in
             progress_map.anchor_parameters(independent)[:, 0]] if independent else []
    moves = dict([(0., 0.), (1., 1.)]+fixed+command_keys)
    before = np.asarray(sorted(moves))
    after = np.asarray([moves[value] for value in before])
    if after[0] != 0 or after[-1] != 1 or np.any(np.diff(after) <= 0):
        raise ValueError("Device quantization merges or reverses required command times")
    # A required event can lie inside a linear reference interval. Include
    # its old time explicitly before moving it, otherwise endpoint-only
    # interpolation could silently leave the event at the unquantized time.
    keys = np.union1d(progress_map.u, before)
    mapping = _ProgressMap(np.interp(keys, before, after), np.interp(keys, progress_map.u, progress_map.s))
    for event in problem.events:
        record = source[event_occurrence_key(event)]
        parameters = mapping.anchor_parameters([event.progress])[:, 0]
        record.update(time=(record["intent_time_s"] if record["activation_time_s"] is not None
                            else (record["consumption_time_s"] if record["consumption_time_s"] is not None
                                  else float(parameters[0]*duration))),
                      hold_until=float(parameters[-1]*duration))
    return _ProgressMap(mapping.u, mapping.s, source)
# END OMPL/refinement/export only read this record and never repeat quantization.


# BEGIN hard time requirements inspect the final tick-bound schedule.
def _validate_bound_event_times(problem, progress_map, duration, origin_s=0., history=(),
                                goal_identities=None, timing_events=()):
    """Check explicit windows against actual history or this tick-bound proposal.

    Consumed nodes may still own an unresolved time relation. Their recorded
    witness survives while a future origin is supplied by this candidate.
    Only the latest selected occurrence of an original event owns the current
    completion obligation; older attempts remain unchanged in the ledger.
    """
    from .planning_problem import PlanningEvent, TimeBindingError, resolve_event_window
    goal_identities = ({name: list(range(len(getattr(problem, name))))
                       for name in ("pose_events", "coordinate_events", "supports")}
                      if goal_identities is None else goal_identities)
    # BEGIN past target-pose timing cannot require unavailable online truth.
    # Future nominal anchors still obey their planned windows. Once an anchor
    # is outside the remaining plan, its actual arrival and any dependent time
    # origin belong only to simulation grading; they cannot reject a rebind.
    history = [row for row in history if not (row.get("check_target") == "arrival"
        and (row.get("goal_id") or [None])[0] in ("pose_events", "supports", "coordinate_events"))]

    def past_pose_timing(constraint):
        identities = [constraint.goal_id if constraint.check_target == "arrival" else None,
                      (constraint.origin_event or {}).get("goal_id")]
        return any(identity and identity[0] in ("pose_events", "supports", "coordinate_events")
                   and identity[1] not in goal_identities[identity[0]] for identity in identities)
    # END retained geometric-arrival labels never become actual planning witnesses.
    current = {event_occurrence_key(event): event for event in problem.events}
    all_events = {event_occurrence_key(row): PlanningEvent(**row) for row in timing_events}
    all_events.update(current)
    latest = {}
    for key in all_events:
        latest[key[:2]] = max(latest.get(key[:2], -1), key[2])
    predicted, planned, checks = [], {}, []

    def candidate_fact(key, target, value, **identity):
        predicted.append({"skill_id": key[0], "event_id": key[1], "occurrence": key[2],
                          "check_target": target, "time_s": origin_s+value, **identity})

    # A declared future origin can be another current command, goal arrival
    # or effect observation node even when the origin has no own time window.
    for key, event in current.items():
        record = progress_map.event_clock[key]
        if record.get("activation_time_s") is not None:
            candidate_fact(key, "activation", record["activation_time_s"])
        for collection in ("pose_events", "coordinate_events", "supports"):
            for index, goal in enumerate(getattr(problem, collection)):
                if abs(goal.progress-event.progress) <= 1e-7:
                    candidate_fact(key, "arrival", float(progress_map.anchor_parameters([goal.progress])[0, 0]*duration),
                                   goal_id=[collection, goal_identities[collection][index]])
        for condition in getattr(problem, "conditions", ()):
            if (condition.get("skill_id"), condition.get("event_id")) == key[:2] and condition.get("effect_requirement"):
                candidate_fact(key, "effect", record["time"], condition_id=condition["condition_id"])
        for constraint in event.timing:
            if past_pose_timing(constraint):
                continue
            if constraint.check_target == "activation":
                value = record.get("activation_time_s")
                if value is None:
                    raise TimeBindingError("An activation window requires a real device command activation")
            elif constraint.check_target == "arrival":
                collection, original_index = constraint.goal_id
                if original_index not in goal_identities[collection]:
                    raise TimeBindingError("An arrival requirement refers to a goal absent from this occurrence")
                goal = getattr(problem, collection)[goal_identities[collection].index(original_index)]
                value = float(progress_map.anchor_parameters([goal.progress])[0, 0]*duration)
            else:
                value = record["time"]
            identity = {"goal_id": None if constraint.goal_id is None else list(constraint.goal_id),
                        "condition_id": constraint.condition_id}
            candidate_fact(key, constraint.check_target, value, **identity)
            planned[(key, constraint.check_target, constraint.goal_id, constraint.condition_id)] = origin_s+value

    for key, event in all_events.items():
        if key[2] != latest[key[:2]]:
            continue
        for constraint in event.timing:
            if past_pose_timing(constraint):
                checks.append({"event": key, "check_target": constraint.check_target,
                               "status": "simulation_posthoc_only"})
                continue
            facts = [row for row in history if event_occurrence_key(row) == key and row.get("value") == "T"
                and row.get("check_target") == constraint.check_target
                and (constraint.goal_id is None or tuple(row.get("goal_id", ())) == constraint.goal_id)
                and (constraint.condition_id is None or row.get("condition_id") == constraint.condition_id)]
            value = min(row["time_s"] for row in facts) if facts else planned.get(
                (key, constraint.check_target, constraint.goal_id, constraint.condition_id))
            # A pending observed effect can be assessed again from fresh actual
            # data. An absent consumed activation/arrival witness cannot be
            # reconstructed from a demonstration or its former scheduled time.
            if value is None:
                if constraint.check_target != "effect":
                    raise TimeBindingError("A consumed timed event lacks its actual occurrence witness")
                value = origin_s
            earliest, latest_time = resolve_event_window(constraint, history, predicted_origins=predicted)
            if constraint.check_target == "effect":
                # An old true state observed before this effect window is
                # history, not a completion witness for the timed obligation.
                applicable = [row for row in facts if (earliest is None or row["time_s"] >= earliest-1e-9)
                              and (latest_time is None or row["time_s"] <= latest_time+1e-9)]
                facts = applicable
                value = (min(row["time_s"] for row in facts) if facts else max(origin_s,
                    planned.get((key, constraint.check_target, constraint.goal_id, constraint.condition_id), origin_s),
                    origin_s if earliest is None else earliest))
            if latest_time is not None and value > latest_time+1e-9:
                raise TimeBindingError(f"Event {key} {constraint.check_target} exceeds its original physical deadline")
            if constraint.check_target != "effect" and earliest is not None and value < earliest-1e-9:
                raise TimeBindingError(f"Event {key} {constraint.check_target} precedes its explicit physical window")
            checks.append({"event": key, "check_target": constraint.check_target,
                "scheduled_request_time_s": value, "window_s": [earliest, latest_time],
                "status": "requires_actual_effect_evidence" if constraint.check_target == "effect" and not facts
                          else "actual_time_inside_window" if facts else "scheduled_time_inside_window"})
    return checks

# END computation time is accounted separately and cannot shift these origins.



# BEGIN same-request continuation consumes facts and the original clock.
class ContinuationBindingError(ValueError):
    """Remaining work lacks facts, budget or a compatible original clock."""


# BEGIN one ledger accounts for successful, failed and rebound work alike.
_PLANNING_COUNTS = ("binding_attempts", "initializations", "ik_batches", "ik_queries", "search_queries",
                    "trajectory_solves", "validation_passes", "validation_queries")
_PLANNING_SECONDS = ("search_seconds", "planning_seconds")


def _obligation_contract(problem):
    """Keep original semantic identities and thresholds, allowing fresh geometry.

    Current poses, attachment transforms and observed surfaces are rebindable
    inputs. A new observation cannot erase a goal/process, reorder its original
    index, or relax a declared functional threshold under the same request ID.
    """
    from .execution_binding import plain
    columns = {
        # BEGIN retained obligations include their explicit orientation freedoms.
        "pose_events": ("progress", "source", "position_tolerance_m", "rotation_tolerance_rad",
                        "yaw_axis", "yaw_tolerance_rad"),
        # END continuation cannot silently replace a yaw interval with a full pose.
        "coordinate_events": ("progress", "source", "target_coordinate", "tolerance", "kind", "coordinate_mode", "coordinate_origin"),
        "supports": ("progress", "source", "margin", "gap_tolerance", "normal_tolerance", "hard_components"),
        "pose_relations": ("start", "end", "source", "pose_bounds"),
        "mechanisms": ("end", "source", "directions"),
        "interiors": ("start", "end", "source", "margin"),
        "couplings": ("joint_names", "matrix", "target", "source"),
    }
    contract = {}
    for collection, names in columns.items():
        contract[collection] = [{"type": type(item).__name__, **{name: plain(getattr(item, name)) for name in names},
            # BEGIN rebinding cannot turn an owner-relative goal into a world goal.
            **({"actual_relation": (None if item.actual_relation is None else
                {key: plain(value) for key, value in item.actual_relation.items()
                 if not key.startswith("T_")})} if collection == "pose_events" else {}),
            # END current local transforms remain rebindable geometric inputs.
            **({"kind": item.model.kind} if collection == "mechanisms" else {})}
            for item in getattr(problem, collection, ())]
    contract["conditions"] = [{name: plain(row[name]) for name in
        ("condition_id", "kind", "skill_id", "event_id", "effect_requirement", "requires_evidence") if name in row}
        for row in getattr(problem, "conditions", ())]
    contract["source_time_map"] = {name: plain(getattr(problem, name, None))
                                   for name in ("source_timestamps", "source_progress")}
    contract["source_phases"] = [{"source": plain(phase.reference.source),
                                  "skill_id": phase.metadata.get("skill_id")}
                                 for phase in getattr(problem, "phases", ())]
    return contract


def _restore_original_direction_origins(problem, contract):
    """Retain actual net-change baselines without freezing fresh mechanism poses.

    A compiler can derive an entrance baseline from its freshly observed
    initial coordinate. That is correct only for a new request. For the same
    request, recover the original evidenced baseline when every direction
    instruction is unchanged. An explicitly different baseline is rejected
    by the semantic contract instead of silently accepted as a new goal.
    """
    from .execution_binding import plain
    saved_mechanisms = contract["mechanisms"]
    if not saved_mechanisms or len(saved_mechanisms) != len(getattr(problem, "mechanisms", ())):
        return
    for mechanism, saved in zip(problem.mechanisms, saved_mechanisms):
        if len(mechanism.directions) != len(saved["directions"]):
            continue
        directions = []
        for direction, original in zip(mechanism.directions, saved["directions"]):
            current = plain(direction)
            semantics = lambda row: {name: value for name, value in row.items()
                                     if name not in {"origin_coordinate", "origin_source"}}
            automatic = (direction.start == mechanism.start
                and direction.origin_coordinate == float(mechanism.initial_coordinates[direction.coordinate_index])
                and plain(direction.origin_source) == plain(mechanism.source))
            if (direction.kind == "net" and original["origin_coordinate"] is not None
                    and semantics(current) == semantics(original)
                    and (direction.origin_coordinate is None or automatic)):
                direction = replace(direction, origin_coordinate=original["origin_coordinate"],
                                    origin_source=deepcopy(original["origin_source"]))
            directions.append(direction)
        mechanism.directions = tuple(directions)


class PlanningWorkExhausted(ValueError):
    """The original request has no remaining admitted computation budget."""


# BEGIN physical allocation limits do not replenish cumulative planning work.
_ALLOCATION_BUDGET_FIELDS = frozenset(("max_host_rss_bytes", "max_device_used_bytes",
                                     "max_scene_gpu_workspace_bytes"))


def _planning_work_budget(values):
    """Retain every work, geometry, timing and numerical field for comparison."""
    return {name: value for name, value in values.items() if name not in _ALLOCATION_BUDGET_FIELDS}
# END a resource change cannot alter samples, iterations, clocks or tolerances.


def _request_work_ledger(problem, budget, previous=None, continuation=None):
    """Copy an append-only request record, never mutate a caller's failed result."""
    from .execution_binding import plain
    ledger = ({"request_id": uuid4().hex, "planning_budget": asdict(budget),
        "requires_continuation": False,
        "consumed_planning": {**dict.fromkeys(_PLANNING_COUNTS, 0),
                              **dict.fromkeys(_PLANNING_SECONDS, 0.)},
        "original_events": plain(problem.events), "obligation_contract": _obligation_contract(problem),
        "history": [], "event_records": [], "timing_events": [], "timing_checks": [],
        "completed_goal_ids": [], "goal_evidence": [], "completed_direction_ids": [], "direction_evidence": [],
        "effects": {}, "effect_history": [], "binding_records": []}
        if previous is None else deepcopy(previous))
    requested_budget = asdict(budget)
    recorded_budget = ledger.get("planning_budget")
    if (not isinstance(ledger.get("request_id"), str) or not ledger["request_id"]
            or type(ledger.get("requires_continuation")) is not bool
            or not isinstance(recorded_budget, dict)
            or _planning_work_budget(recorded_budget) != _planning_work_budget(requested_budget)):
        raise ContinuationBindingError("A same-request ledger cannot reset its identity or planning budget")
    # A previously recorded allocation also has to satisfy the explicit
    # physical contract; malformed historical limits are not an override.
    PlanningBudget(**recorded_budget)
    if not isinstance(ledger.get("obligation_contract"), dict):
        raise ContinuationBindingError("The request ledger lacks its original goal/process identity contract")
    consumed = ledger.get("consumed_planning", {})
    for name in (*_PLANNING_COUNTS, *_PLANNING_SECONDS):
        value = consumed.get(name)
        if (not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0
                or (name in _PLANNING_COUNTS and type(value) is not int)):
            raise ContinuationBindingError("The request ledger needs every cumulative planning counter")
    for name in ("original_events", "history", "event_records", "timing_events", "timing_checks", "completed_goal_ids", "goal_evidence",
                 "completed_direction_ids", "direction_evidence", "effect_history", "binding_records"):
        if not isinstance(ledger.get(name), list):
            raise ContinuationBindingError(f"The request ledger lacks its append-only {name}")
    # BEGIN record a reviewed allocation change without rewriting work history.
    if recorded_budget != requested_budget:
        ledger["binding_records"].append({"mode": "resource_limits_changed",
            "source": "explicit_planning_budget",
            "previous": {name: recorded_budget[name] for name in sorted(_ALLOCATION_BUDGET_FIELDS)},
            "current": {name: requested_budget[name] for name in sorted(_ALLOCATION_BUDGET_FIELDS)},
            "consumed_planning": deepcopy(consumed)})
        ledger["planning_budget"] = requested_budget
    # END original identity, obligations, execution facts and counters survive.
    if continuation is not None:
        original = continuation["original_binding"]
        old = original["request_ledger"]
        if old["requires_continuation"] and not ledger["requires_continuation"]:
            raise ContinuationBindingError("A request cannot erase its actual execution continuation requirement")
        if old["request_id"] != ledger["request_id"]:
            raise ContinuationBindingError("Execution and planning ledgers belong to different requests")
        if (ledger["original_events"] != old["original_events"]
                or ledger["obligation_contract"] != old["obligation_contract"]):
            raise ContinuationBindingError("A request ledger cannot rewrite original source or goal/process identities")
        if not {tuple(value) for value in old["completed_direction_ids"]} <= {tuple(value) for value in ledger["completed_direction_ids"]}:
            raise ContinuationBindingError("A request ledger cannot erase historical completed directions")
        if any(consumed[name] < old["consumed_planning"][name] for name in consumed):
            raise ContinuationBindingError("Cumulative planning work must retain all prior attempts")
        for name in ("history", "event_records", "timing_events", "timing_checks", "direction_evidence", "effect_history", "binding_records"):
            if ledger[name][:len(old[name])] != old[name]:
                raise ContinuationBindingError(f"The request ledger rewrote prior {name}")
        # Append actual execution facts once; a historical T is retained here,
        # not promoted to currently applicable scene evidence in a new binding.
        # BEGIN geometric outcome labels are posthoc only, not request facts.
        for name in ("history", "timing_checks", "completed_direction_ids", "direction_evidence", "effect_history"):
            for row in continuation.get(name, ()):
                if row not in ledger[name]:
                    ledger[name].append(deepcopy(row))
        # END original geometric requirements remain in obligation_contract.
        for name in ("applied_events", "superseded_events", "evaluated_events", "stopped_events", "reached_events", "verified_events"):
            for row in continuation.get(name, ()):
                record = {"record_kind": name, **deepcopy(row)}
                if record not in ledger["event_records"]:
                    ledger["event_records"].append(record)
        ledger["effects"].update(deepcopy(continuation.get("effects", {})))
        ledger["requires_continuation"] = True
    return ledger


def _check_work_budget(budget, consumed, *, started, initial_seconds, stage=None):
    """Check at stage boundaries; report real elapsed overrun, never pretend preemption."""
    consumed["planning_seconds"] = initial_seconds+time.monotonic()-started
    cap = budget.maximum_trajectory_solves
    if consumed["planning_seconds"] >= budget.planning_seconds:
        raise PlanningWorkExhausted("The original cumulative planning wall-time budget is exhausted")
    stage_caps = {"binding_attempts": cap, "initializations": cap, "ik_batches": cap,
        "ik_queries": budget.max_validation_points*budget.ik_seeds*cap,
        "trajectory_solves": cap,
        "validation_passes": cap*budget.trajectory_seeds,
        "validation_queries": budget.max_validation_points*cap*budget.trajectory_seeds+cap,
        "search_queries": budget.max_validation_points*cap}
    for name, limit in stage_caps.items():
        if consumed[name] > limit or (stage == name and consumed[name] >= limit):
            raise PlanningWorkExhausted(f"The original cumulative {name} budget is exhausted")
# END derived capacities include every declared native candidate, without new services.


def _prepare_continuation(problem, scene, budget, continuation, *, rebind=None):
    """Validate a fresh actual/scene binding without replaying consumed events.

    A continuation is the executor's existing result record. It is not a
    request to restore the old actual state or guess past task success. The
    original absolute progress, event identities and cumulative solve budget
    survive. Nominal pose/support/coordinate anchors leave the suffix by
    progress; physical task effects remain separate independent observations.
    """
    from .execution_binding import plain
    from .planning_problem import event_occurrence_key
    if rebind is not None and (not isinstance(rebind, dict)
            or rebind.get("kind") not in {"remaining", "retry"} or not rebind.get("source")):
        raise ContinuationBindingError("An explicit rebind needs remaining/retry kind and its executor source")
    if not isinstance(continuation, dict) or not isinstance(continuation.get("original_binding"), dict):
        raise ContinuationBindingError("Continuation requires its original formal execution binding")
    original = continuation["original_binding"]
    if continuation.get("candidate_id") != original.get("candidate_id"):
        raise ContinuationBindingError("Continuation candidate identity differs from its original binding")
    # BEGIN physical window origins retain their original actual history.
    previous_history = original["request_ledger"]["history"]
    history = continuation.get("history", [])
    if (not isinstance(history, (list, tuple))
            or list(history[:len(previous_history)]) != previous_history):
        raise ContinuationBindingError("Continuation cannot erase or rewrite original physical timing history")
    # END a later repeated fact cannot move an already observed window origin.
    _restore_original_direction_origins(problem, original["request_ledger"]["obligation_contract"])
    if _obligation_contract(problem) != original["request_ledger"]["obligation_contract"]:
        raise ContinuationBindingError("A fresh binding cannot remove, reorder or change original goal/process obligations")
    if "planning_budget" not in original or "consumed_planning" not in original:
        raise ContinuationBindingError("Continuation lacks the original cumulative planning budget")
    request_budget = PlanningBudget(**original["planning_budget"])
    # BEGIN the frozen execution binding fixes work, not a later machine quota.
    if budget is not None:
        if _planning_work_budget(asdict(budget)) != _planning_work_budget(asdict(request_budget)):
            raise ContinuationBindingError("A continuation cannot reset or enlarge its original planning budget")
        request_budget = budget
    # END remaining solves/search time below still subtract all historical work.
    consumed = deepcopy(continuation.get("consumed_planning", original["consumed_planning"]))
    for key in (*_PLANNING_COUNTS, *_PLANNING_SECONDS):
        value = consumed.get(key)
        if (not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0
                or value < original["consumed_planning"].get(key, 0)):
            raise ContinuationBindingError("Cumulative planning work must retain all prior attempts")
    if type(consumed["trajectory_solves"]) is not int:
        raise ContinuationBindingError("Consumed trajectory solves must be an integer count")
    # Actual simulation/device steps, including holds, belong to the original
    # skill as well. A fresh binding cannot manufacture a fresh step budget.
    physics_cap = continuation.get("physics_budget")
    physics_consumed = continuation.get("cumulative_physics_steps")
    physics_remaining = continuation.get("remaining_physics_steps")
    if (type(physics_cap) is not int or physics_cap <= 0
            or type(physics_consumed) is not int or physics_consumed < 0
            or type(physics_remaining) is not int or physics_remaining < 0
            or physics_consumed < original.get("consumed_physics_steps", 0)
            or physics_cap != original.get("physics_budget")
            or physics_remaining > max(0, physics_cap-physics_consumed)):
        raise ContinuationBindingError("Continuation must retain its original cumulative physics-step budget")
    if physics_remaining == 0:
        raise ContinuationBindingError("The current skill's cumulative physics-step budget is exhausted")
    local_steps = continuation.get("consumed_physics_steps")
    # BEGIN physical time uses confirmed steps; attempts consume budget only.
    confirmed_steps = continuation.get("confirmed_physics_steps")
    accounting_exact = continuation.get("step_accounting_exact")
    # END no legacy charge-as-time interpretation is accepted implicitly.
    timestep = continuation.get("device_timestep_s")
    if (type(local_steps) is not int or local_steps < 0
            # BEGIN uncertain clock records cannot bind a new physical suffix.
            or type(confirmed_steps) is not int or not 0 <= confirmed_steps <= local_steps
            or accounting_exact is not True
            # END a known zero-step failed attempt may still carry one charge.
            or not isinstance(timestep, (float, int)) or not np.isfinite(timestep) or timestep <= 0
            or physics_consumed != original.get("consumed_physics_steps", 0)+local_steps):
        raise ContinuationBindingError("Continuation requires cumulative charges, exact confirmed steps and the unchanged native device timestep")
    remaining = request_budget
    feedback = continuation.get("actual_feedback", {})
    # BEGIN failed final capture cannot be replaced by stale or absent feedback.
    if not isinstance(feedback, dict):
        raise ContinuationBindingError("Continuation needs captured synchronized actual feedback")
    # END the failed attempt's conservative charge remains in the request.
    names = list(feedback.get("joint_names", ()))
    if not names or len(set(names)) != len(names) or not set(problem.joint_names) <= set(names):
        raise ContinuationBindingError("Continuation needs synchronized named actual q/dq feedback")
    try:
        columns = [names.index(name) for name in problem.joint_names]
        actual_q, actual_dq = np.asarray(feedback["q"], float)[columns], np.asarray(feedback["dq"], float)[columns]
    except (KeyError, IndexError, ValueError) as error:
        raise ContinuationBindingError("Continuation has incomplete actual joint feedback") from error
    if not np.array_equal(problem.start_q, actual_q) or not np.array_equal(problem.start_dq, actual_dq):
        raise ContinuationBindingError("Fresh planning actual differs from the executor's synchronized continuation feedback")
    hand = scene.gripper
    if hand is not None:
        if hand.control_timestep_s != timestep:
            raise ContinuationBindingError("Continuation cannot change the original native device control timestep")
        if not set(hand.joint_names) <= set(names) or not np.array_equal(hand.current_positions,
                np.asarray(feedback["q"], float)[[names.index(name) for name in hand.joint_names]]):
            raise ContinuationBindingError("Continuation hand model is not bound to the current actual fingers")
    curve = original.get("curve", {})
    duration, actual_elapsed = curve.get("duration"), continuation.get("elapsed_physical_s")
    # BEGIN advance the original clock only by verified native integration.
    elapsed = confirmed_steps*float(timestep)
    # END budget reservations never synthesize physical elapsed time.
    if (not isinstance(actual_elapsed, (int, float)) or not np.isfinite(actual_elapsed)
            or abs(actual_elapsed-elapsed) > 1e-8):
        raise ContinuationBindingError("Actual elapsed physics time differs from its native device step ledger")
    if (not isinstance(duration, (int, float)) or not isinstance(elapsed, (int, float))
            or not np.isfinite([duration, elapsed]).all() or elapsed < 0 or duration <= 0
            or (rebind is None and elapsed >= duration)):
        raise ContinuationBindingError("The remaining original physical clock is exhausted or unavailable")
    old_u, old_s = np.asarray(curve.get("progress_u"), float), np.asarray(curve.get("progress_s"), float)
    try:
        original_map = _ProgressMap(old_u, old_s)
    except ValueError as error:
        raise ContinuationBindingError("Continuation has no complete frozen progress map") from error
    curve_elapsed = elapsed if rebind is None else continuation.get("last_curve_time_s")
    if not isinstance(curve_elapsed, (int, float)) or not np.isfinite(curve_elapsed) or not 0 <= curve_elapsed <= duration:
        raise ContinuationBindingError("Rebinding requires the last actually executed curve time, separately from hold time")
    start_progress = float(np.interp(curve_elapsed/duration, original_map.u, original_map.s))
    if start_progress >= problem.end_progress and rebind is None:
        raise ContinuationBindingError("The original skill progress is finished; unresolved effects need assessment or an explicit same-goal retry")
    request_events = deepcopy(original["request_ledger"]["original_events"])
    source_events = {(row["skill_id"], row["event_id"]): row for row in request_events}
    current_events = {(event.skill_id, event.event_id): event for event in problem.events}
    if set(current_events) != set(source_events):
        raise ContinuationBindingError("Fresh continuation must retain the original skill and event identities before filtering")
    for identity, event in current_events.items():
        current, source = plain(event), deepcopy(source_events[identity])
        current.pop("occurrence")
        source.pop("occurrence")
        if current != source:
            raise ContinuationBindingError("Continuation changed an original event/source/goal association")
    remaining_rows = continuation.get("remaining_events")
    if not isinstance(remaining_rows, (list, tuple)):
        raise ContinuationBindingError("Continuation lacks its consumed/unconsumed event ledger")
    pending_ids = {event_occurrence_key(row) for row in remaining_rows}
    old_occurrences = {event_occurrence_key(row): row for row in curve.get("events", ())}
    old_rows = {(row["skill_id"], row["event_id"]): row for row in curve.get("events", ())}
    if not pending_ids <= set(old_occurrences):
        raise ContinuationBindingError("Remaining event ledger does not belong to the original candidate")
    applied = set(tuple(value) for value in original.get("consumed_events", ()))
    for row in (*continuation.get("applied_events", ()), *continuation.get("superseded_events", ())):
        applied.add(event_occurrence_key(row))
    if applied & pending_ids:
        raise ContinuationBindingError("An applied command cannot be replayed as an unconsumed event")
    evaluated = {event_occurrence_key(row) for row in continuation.get("evaluated_events", ())}
    stopped = {event_occurrence_key(row) for row in continuation.get("stopped_events", ())}
    reached = {event_occurrence_key(row) for row in continuation.get("reached_events", ())}
    witnessed = applied | evaluated | stopped | reached | {
        event_occurrence_key(row) for row in continuation.get("verified_events", ())}
    # Ordinary effect evaluation consumes its observation node even when
    # the value is F/U. It never claims a successful effect or a held object.
    for identity in evaluated:
        event = current_events.get(identity[:2])
        if (event is None or event.kind != "effect" or event.requires_evidence is not None
                or (event.stop and identity not in stopped)):
            raise ContinuationBindingError("An evaluation ledger entry is not an ordinary effect node")
        pending_ids.discard(identity)
    for identity in reached:
        event = current_events.get(identity[:2])
        if event is None or event.kind != "function" or event.requires_evidence is not None:
            raise ContinuationBindingError("A reached-node ledger entry is not an independent functional event")
        pending_ids.discard(identity)
    consumed_events = (set(old_occurrences)-pending_ids) | applied
    if not consumed_events <= witnessed:
        raise ContinuationBindingError("Consumed events need actual application or verification ledger entries")
    for identity in pending_ids:
        row = old_occurrences[identity]
        due = row.get("activation_time_s") if row.get("activation_time_s") is not None else row["time"]
        if due < elapsed and rebind is None:
            raise ContinuationBindingError(f"Unconsumed event {identity} is overdue on the frozen schedule; an explicit rebind is required")
    # BEGIN every geometric goal is nominal, independent of physical scoring.
    # Pose, support and mechanism-coordinate targets are all model constraints.
    # Missing/false/true result labels cannot change the remaining trajectory.
    all_goals = {(name, index) for name in ("pose_events", "coordinate_events", "supports")
                 for index in range(len(getattr(problem, name)))}
    # END explicit independent task effects keep their own event dependencies.
    # Actual direction completion is not implied by traversing source progress.
    all_directions = {(index, column) for index, mechanism in enumerate(problem.mechanisms)
                      for column in range(len(mechanism.directions))}
    completed_directions = {tuple(value) for value in original.get("completed_direction_ids", ())}
    claimed_directions = {tuple(value) for value in continuation.get("completed_direction_ids", ())}
    if not (completed_directions | claimed_directions) <= all_directions:
        raise ContinuationBindingError("A direction completion must retain its original mechanism/direction identity")
    for identity in claimed_directions-completed_directions:
        direction = problem.mechanisms[identity[0]].directions[identity[1]]
        rows = [row for row in continuation.get("direction_evidence", ())
                if tuple(row.get("identity", ())) == identity and row.get("candidate_id") == original["candidate_id"]
                and row.get("value") == "T" and row.get("scope_complete") is True
                and row.get("completion_verified") is True
                and isinstance(row.get("source"), str) and row["source"].strip()]
        # BEGIN completion consumes the executor's covered actual scope.
        # An observer's raw scope_complete label is retained as evidence but
        # cannot alone turn a short or stale interval into a completed guard.
        # Net displacement needs its original coordinate origin and endpoint;
        # monotone/rate obligations additionally cover the whole observed scope.
        declared_net_origin = (direction.kind == "net" and direction.origin_coordinate is not None
                               and bool(direction.origin_source))
        verified = False
        for row in rows:
            interval = np.asarray(row.get("acquisition_interval_s"), float)
            scope_start, scope_end = row.get("scope_started_at_s"), row.get("scope_ended_at_s")
            if (interval.shape != (2,) or not np.isfinite(interval).all() or interval[0] > interval[1]
                    or not isinstance(scope_end, (int, float)) or not np.isfinite(scope_end)
                    or interval[1] < scope_end-1e-8):
                continue
            scope_covered = (isinstance(scope_start, (int, float)) and np.isfinite(scope_start)
                             and scope_start <= scope_end and interval[0] <= scope_start+1e-8)
            if declared_net_origin or scope_covered:
                verified = True
                break
        if not verified:
            raise ContinuationBindingError("Direction completion needs actual scoped T evidence from this executed candidate")
        # END continuation cannot upgrade an observer's unverified completion label.
    completed_directions.update(claimed_directions)
    # BEGIN explicit rebinding preserves obligations but rebuilds the soft schedule.
    retry_ids, retry_goals, retry_directions = set(), set(), set()
    if rebind is not None and rebind["kind"] == "retry":
        retry_ids = {tuple(value) for value in rebind.get("retry_event_ids", ())}
        retry_goals = {tuple(value) for value in rebind.get("retry_goal_ids", ())}
        retry_directions = {tuple(value) for value in rebind.get("retry_direction_ids", ())}
        all_goals = {(name, index) for name in ("pose_events", "coordinate_events", "supports")
                     for index in range(len(getattr(problem, name)))}
        if (not retry_ids <= set(source_events) or not retry_goals <= all_goals or not retry_directions <= all_directions
                or not (retry_ids or retry_goals or retry_directions)):
            raise ContinuationBindingError("A same-goal retry must select original event/goal identities")
    selected_events = {}
    for identity in pending_ids:
        selected_events[identity[:2]] = replace(current_events[identity[:2]], occurrence=identity[2])
    # BEGIN completed motion blocks never consume events that were not planned.
    for event in problem.events:
        identity = event_occurrence_key(event)
        if (identity not in old_occurrences and identity not in applied
                and event.progress >= start_progress):
            selected_events[(event.skill_id, event.event_id)] = deepcopy(event)
    # END the original request ledger remains the owner of deferred obligations.
    for identity in retry_ids:
        if any(key[:2] == identity for key in pending_ids):
            raise ContinuationBindingError("Retry selects consumed occurrences; pending work uses remaining rebind")
        occurrences = [key[2] for key in (*old_occurrences, *applied) if key[:2] == identity]
        occurrences.extend(event_occurrence_key(row)[2] for row in continuation.get("history", ())
                           if event_occurrence_key(row)[:2] == identity)
        selected_events[identity] = replace(current_events[identity], occurrence=max(occurrences, default=0)+1)
    remaining_goal_ids = {(name, index) for name in ("pose_events", "coordinate_events", "supports")
        for index, goal in enumerate(getattr(problem, name))
        if goal.progress >= start_progress or (name, index) in retry_goals}
    if rebind is not None:
        required_progress = [event.progress for event in selected_events.values()]
        required_progress += [getattr(problem, name)[index].progress for name, index in remaining_goal_ids]
        required_progress += [problem.mechanisms[index].directions[column].start for index, column in retry_directions]
        if required_progress:
            start_progress = min(start_progress, min(required_progress))
        # A terminal unresolved effect/goal needs a nonempty new source window;
        # taking the preceding source interval proposes timing only, not q or a
        # re-execution of already consumed commands.
        if start_progress >= problem.end_progress:
            start_progress = float(problem.source_progress[-2])
    # END original frame/progress associations remain unchanged across occurrences.
    # BEGIN past nominal anchors are retained in the ledger, not re-proved.
    # Leaving a completed schedule prefix does not assert physical success.
    # END retries explicitly restore original goals through retry_goal_ids.
    for index, requirement in enumerate(problem.mechanisms):
        for column, direction in enumerate(requirement.directions):
            if (direction.end < start_progress and (index, column) not in completed_directions
                    and (index, column) not in retry_directions):
                raise ContinuationBindingError(f"Past direction {index}:{column} needs actual scoped completion evidence or an explicit retry")
        if any(direction.kind == "net" and direction.origin_coordinate is None
               and direction.start < start_progress <= direction.end for direction in requirement.directions):
            raise ContinuationBindingError(f"mechanism:{index} net direction needs its actually observed original scope-start coordinate")
        if (requirement.end >= start_progress and requirement.angular_column is not None
                and requirement.start != start_progress
                and not requirement.model.diagnostics.get("initial_coordinates_source")
                and requirement.model.diagnostics.get("initial_coordinate_convention")
                    != "unique actual angular branch within current physical limits"):
            raise ContinuationBindingError(f"mechanism:{index} needs current observed lifted coordinates and a fresh actual-start scope")
    # Goal ledger identities always use the original request's collection
    # indices. Filtering a remaining suffix must never renumber a proof.
    goal_identities = {collection: [index for index, item in enumerate(getattr(problem, collection))
                                  if item.progress >= start_progress and (collection, index) in remaining_goal_ids]
                       for collection in ("pose_events", "coordinate_events", "supports")}
    direction_identities = [[(index, column) for column in range(len(item.directions))]
                            for index, item in enumerate(problem.mechanisms) if item.end >= start_progress]
    rebound = deepcopy(problem)
    rebound.start_progress = start_progress
    rebound.events = tuple(deepcopy(event) for event in selected_events.values())
    # BEGIN a measured angular branch anchors this binding's actual suffix.
    # The compiler validates this explicit lifted coordinate against fresh
    # observed pose. Move only the mechanism's coordinate-history entrance;
    # original direction scopes, net origins and entry-delta targets remain.
    # Mutating the private object also preserves coordinate-goal references
    # to this same mechanism instead of creating two inconsistent lift scopes.
    for mechanism in rebound.mechanisms:
        if (mechanism.angular_column is not None and mechanism.start < start_progress <= mechanism.end
                and (mechanism.model.diagnostics.get("initial_coordinates_source")
                     or mechanism.model.diagnostics.get("initial_coordinate_convention")
                        == "unique actual angular branch within current physical limits")):
            mechanism.start = start_progress
    # END reference angles cannot supply this actual branch evidence.
    for collection in ("pose_events", "coordinate_events", "supports"):
        setattr(rebound, collection, tuple(item for index, item in enumerate(getattr(rebound, collection))
                                           if index in goal_identities[collection]))
    for collection in ("pose_relations", "mechanisms", "interiors"):
        setattr(rebound, collection, tuple(item for item in getattr(rebound, collection) if item.end >= start_progress))
    source_time = problem.progress_to_source_time(start_progress)
    selected = problem.source_progress > start_progress
    rebound.source_timestamps = np.r_[source_time, problem.source_timestamps[selected]]
    rebound.source_progress = np.r_[start_progress, problem.source_progress[selected]]
    rebound.__post_init__()
    # A hand factory consumes only this candidate's clock event records. Keep
    # the scene's declared command identities in that same selected set; its
    # original source policy is validated before changing occurrence numbers.
    if hand is not None:
        policies = {(row["skill_id"], row["event_id"]): row for row in hand.commands}
        commands = []
        for event in rebound.events:
            if event.kind not in {"open", "close", "hold"}:
                continue
            policy = policies.get((event.skill_id, event.event_id))
            if policy is None or policy["command"] != event.kind or policy["progress"] != event.progress:
                raise ContinuationBindingError("Rebinding cannot invent or change an original hand command policy")
            commands.append({**deepcopy(policy), "occurrence": event.occurrence})
        hand.commands = tuple(commands)
    # Only the private binding problem is filtered after validation succeeds.
    # Hand factories consume event metadata from the same frozen clock.
    for item in fields(problem):
        if item.init:
            setattr(problem, item.name, getattr(rebound, item.name))
    return remaining, {"original_duration": float(duration), "elapsed": float(elapsed),
        # BEGIN event-clock origins count only confirmed physical advancement.
        "actual_elapsed": float(actual_elapsed), "elapsed_device_steps": confirmed_steps,
        # END accumulated planning/execution budgets retain their separate charges.
        "device_timestep_s": float(timestep),
        "u": original_map.u.copy(), "s": original_map.s.copy(),
        "events": {identity: deepcopy(old_occurrences[identity]) for identity in pending_ids},
        "request_budget": asdict(request_budget), "consumed_planning": consumed,
        "physics_budget": physics_cap, "consumed_physics_steps": physics_consumed,
        "remaining_physics_steps": physics_remaining,
        "request_events": request_events, "consumed_events": sorted(consumed_events),
        "goal_identities": goal_identities,
        "retry_goal_ids": sorted(retry_goals),
        "direction_identities": direction_identities, "retry_direction_ids": sorted(retry_directions),
        "completed_direction_ids": sorted(completed_directions),
        "rebind": deepcopy(rebind),
        "scene_anchors": np.asarray(scene.required_progress_anchors, float).copy() if rebind is not None else None,
        "control_timestep_s": timestep if rebind is not None else None,
        "history": deepcopy(continuation.get("history", ())),
        "timing_events": deepcopy(continuation.get("request_ledger", original["request_ledger"])["timing_events"]),
        "device_policy": deepcopy(continuation.get("device_policy")),
        "active_finger_interval": deepcopy(continuation.get("active_finger_interval")),
        "completed_goal_ids": [], "effects": deepcopy(continuation.get("effects", {})),
        "previous_request_time_origin_s": float(original.get("request_time_origin_s", 0.)),
        "request_time_origin_s": float(original.get("request_time_origin_s", 0.))+float(elapsed),
        "request_time_origin_step": int(original.get("request_time_origin_step", 0))+local_steps}


def _continuation_timing(problem, budget, continuation):
    """Use the original suffix, or rebuild every clock consumer for an explicit rebind."""
    # BEGIN explicit rebind is a new synchronized binding with original time origins.
    if continuation.get("rebind") is not None:
        selected, mapping = _initial_timing(problem, budget)
        anchors = np.unique(np.r_[problem.anchors, continuation["scene_anchors"]])
        anchors = anchors[anchors >= problem.start_progress]
        mapping, _ = _progress_for_cubic_boundaries(mapping, selected.knots,
            stop_progress=problem.stop_progress, anchors=anchors)
        grid = _fixed_spline_grid(intervals=selected.knots+4, native_dt=selected.maximum_trajectory_dt,
            interpolation_steps=selected.interpolation_steps, interpolation_dt=selected.interpolation_dt)
        mapping = _bind_event_clock(problem, mapping, grid["duration"], continuation["control_timestep_s"])
        for record in mapping.event_clock.values():
            record["request_intent_time_s"] = continuation["request_time_origin_s"]+record["intent_time_s"]
            if record.get("activation_time_s") is not None:
                record["request_time_origin_step"] = continuation["request_time_origin_step"]
                record["request_activation_step"] = continuation["request_time_origin_step"]+record["bound_activation_step"]
                record["request_activation_time_s"] = continuation["request_time_origin_s"]+record["activation_time_s"]
        _validate_bound_event_times(problem, mapping, grid["duration"],
            continuation["request_time_origin_s"], continuation["history"], continuation["goal_identities"],
            continuation["timing_events"])
        return selected, mapping
    # END ordinary same-binding continuation retains its already frozen schedule.
    remaining = continuation["original_duration"]-continuation["elapsed"]
    proposal = SimpleNamespace(source_time_map={
        "source_times": np.array([0., remaining]),
        "progress": np.array([problem.start_progress, float(problem.end_progress)])})
    selected, _ = _initial_timing(proposal, budget)
    grid = _fixed_spline_grid(intervals=selected.knots+4,
        native_dt=selected.maximum_trajectory_dt, interpolation_steps=selected.interpolation_steps,
        interpolation_dt=selected.interpolation_dt)
    if grid["duration"] < remaining:
        raise ContinuationBindingError("The remaining frozen schedule exceeds the original native capacity")
    original_times = continuation["u"]*continuation["original_duration"]
    keep = original_times > continuation["elapsed"]
    seconds = np.r_[0., original_times[keep]-continuation["elapsed"]]
    progress = np.r_[problem.start_progress, continuation["s"][keep]]
    if seconds[-1] < grid["duration"]:
        seconds, progress = np.r_[seconds, grid["duration"]], np.r_[progress, progress[-1]]
    clock = {}
    for identity, event in continuation["events"].items():
        record = {name: deepcopy(event[name]) for name in ("time", "hold_until", "intent_time_s", "activation_time_s", "consumption_time_s") if name in event}
        for name in ("time", "hold_until", "intent_time_s", "activation_time_s", "consumption_time_s"):
            if record.get(name) is not None:
                record[name] = float(record[name])-continuation["elapsed"]
        record["request_intent_time_s"] = (event.get("request_intent_time_s")
            if event.get("request_intent_time_s") is not None else
            float(event.get("intent_time_s", event["time"]))+continuation["previous_request_time_origin_s"])
        if event.get("activation_time_s") is not None:
            timestep = continuation["device_timestep_s"]
            if event.get("device_timestep_s") != timestep:
                raise ContinuationBindingError("A pending command changed its bound device timestep")
            step = event["request_activation_step"]-continuation["request_time_origin_step"]
            if step < 0:
                raise ContinuationBindingError("A pending command missed its original device activation tick")
            record.update(activation_time_s=step*timestep, bound_activation_step=step,
                request_activation_step=event["request_activation_step"],
                request_activation_time_s=event["request_activation_time_s"],
                request_time_origin_step=continuation["request_time_origin_step"], device_timestep_s=timestep,
                time=max(0., record["time"]))
            if record["request_intent_time_s"] == record["request_activation_time_s"]:
                record["time"] = record["intent_time_s"] = record["activation_time_s"]
        elif record.get("time", 0.) < 0:
            raise ContinuationBindingError("A pending event missed its original physical time")
        clock[identity] = record
        record["event"] = asdict(next(event for event in problem.events if event_occurrence_key(event) == identity))
    return selected, _ProgressMap(seconds/grid["duration"], progress, clock)
# END a new actual binding retains past results and original pending clocks.


# BEGIN native-cubic scalar clock with bounded, source-preserving linearization.
_PROGRESS_PARAMETER_ERROR = 1e-6
_MAX_PROGRESS_KEYS = 16384


def _progress_for_cubic_boundaries(progress_map, knots, *, stop_progress, anchors):
    """Return the saved progress map and its shared original-path parameter.

    The cubic retains a stopped terminal interval and explicit verification
    stops. Build a monotone scalar reference clock for those boundaries:
    ``z(new_u)`` parameterizes the ORIGINAL
    path, and the returned skill progress is ``old_s(z(new_u))``. OMPL joint
    initializers must use that same z, including inside old progress pauses.

    Four equal coefficients create a complete constant cubic interval with
    zero first/second derivatives on either side. Only explicit verification
    stops add these intervals; ordinary path vertices remain ordinary points.
    Every original path vertex and mandatory progress anchor is retained.

    The saved representation stays piecewise linear. For a uniform cubic,
    ``|d2z/d(native_t)2| <= max|diff(coefficients, 2)|``. Splitting each native
    interval into L pieces bounds linearization error by D2/(8*L**2). The
    normalized-parameter bound 1e-6 is NOT a physical skill tolerance. Native
    and exported process checks both consume this same saved finite map.
    """
    if type(knots) is not int or not 1 <= knots <= 24:
        raise ValueError("The scalar progress clock requires the bounded native control horizon")
    stop_progress = np.unique(np.asarray(stop_progress, float))
    stop_pairs = progress_map.anchor_parameters(stop_progress)
    # First arrival owns the verification event. Any original s plateau's
    # other endpoint and all intervening q-path points are retained below;
    # adding the clock must not delete motion in a conditional search seed.
    stops = np.asarray([stop_pairs[stop_pairs[:, 1] == value, 0][0] for value in stop_progress])
    stops = np.unique(stops[(stops > 0) & (stops < 1)])
    boundaries = np.r_[0., stops, 1.]
    count = len(stops)
    # Each segment needs a positive increment, and each interior stop needs
    # three additional copies of its reached value: four equal coefficients.
    if knots < 4*count+1:
        raise ValueError("The native control horizon cannot represent every required verification stop")
    remaining = knots-4*count-1
    shares = remaining*np.diff(boundaries)
    increments = 1+np.floor(shares).astype(int)
    order = np.argsort(-(shares-np.floor(shares)), kind="stable")
    increments[order[:remaining-int(np.floor(shares).sum())]] += 1
    # BEGIN instantaneous q0/dq0 do not imply a complete waiting interval.
    # This scalar parameter belongs to the soft reference, not joint q(t).
    # Its positive rate does not prescribe the robot's measured velocity.
    initial_rate = boundaries[1]/(increments[0]+2)
    prefix = np.array([-initial_rate, 0., initial_rate, 2*initial_rate])
    free = []
    for index, (left, right, steps) in enumerate(zip(boundaries[:-1], boundaries[1:], increments)):
        segment = (np.arange(3, steps+3, dtype=float)*initial_rate
                   if index == 0 else np.linspace(left, right, steps+1)[1:])
        segment[-1] = right
        free.extend(segment)
        if index < count:
            free.extend([right]*3)
    # END only explicit stops/source plateaus and the retained end can pause progress.
    coefficients = np.r_[prefix, free, np.ones(3)]
    intervals = knots+4
    spline = BSpline(np.arange(-3., knots+8.), coefficients, 3)
    second_derivative_bound = float(np.abs(np.diff(coefficients, n=2)).max())
    # Reserve one subdivision beyond the ceiling so replacing a regular
    # sample by a nearby required float32 key has a strict error margin.
    subdivisions = max(1, int(np.ceil(np.sqrt(second_derivative_bound/(8*_PROGRESS_PARAMETER_ERROR))))+1)
    grid_count = intervals*subdivisions+1
    if grid_count > _MAX_PROGRESS_KEYS:
        raise ValueError("The scalar progress clock exceeds its 16384-key linearization budget")
    native_t = np.arange(grid_count, dtype=float)/subdivisions
    parameter = spline(native_t)
    # Constant spans have an exact mathematical value. Writing that value
    # avoids tiny numerical oscillations from summing equal B-spline terms.
    for interval in range(intervals):
        local = coefficients[interval:interval+4]
        if np.all(local == local[0]):
            parameter[interval*subdivisions:(interval+1)*subdivisions+1] = local[0]
    parameter[0], parameter[-1] = 0., 1.
    warp = _ProgressMap(native_t/intervals, parameter)

    source_pairs = progress_map.anchor_parameters(np.unique(np.r_[np.asarray(anchors, float), stop_progress]))
    retained_parameters = np.unique(np.r_[progress_map.u, source_pairs[:, 0]])
    inverse_pairs = warp.anchor_parameters(retained_parameters)
    # Required source keys own their native float32 value. Only an auxiliary
    # regular-grid key may be replaced when both become the same native key;
    # anchor_parameters already rejects aliases between two required keys.
    auxiliary = ~np.isin(warp.u.astype(np.float32), inverse_pairs[:, 0].astype(np.float32))
    new_u = np.unique(np.r_[warp.u[auxiliary], inverse_pairs[:, 0]])
    if len(new_u) > _MAX_PROGRESS_KEYS:
        raise ValueError("Source-preserving progress clock exceeds its 16384-key total budget")
    parameter = np.interp(new_u, warp.u, warp.s)
    parameter[np.searchsorted(new_u, inverse_pairs[:, 0])] = inverse_pairs[:, 1]
    # The difference of the old and new piecewise-linear maps is linear on
    # their common refinement, so its maximum occurs at one of their keys.
    # Add this exact finite-map perturbation to the cubic interpolation bound.
    perturbation = max(float(np.abs(np.interp(warp.u, new_u, parameter)-warp.s).max()),
                       float(np.abs(np.interp(new_u, warp.u, warp.s)-parameter).max()))
    if second_derivative_bound/(8*subdivisions**2)+perturbation > _PROGRESS_PARAMETER_ERROR:
        raise ValueError("Required progress keys exceed the scalar clock's linearization error bound")
    warp = _ProgressMap(new_u, parameter)
    progress = np.interp(warp.s, progress_map.u, progress_map.s)
    # Inverse/forward floating-point interpolation can differ by one final
    # digit. The original supplied progress values retain their exact identity.
    supplied = dict(zip(progress_map.u, progress_map.s))
    supplied.update(dict(source_pairs))
    progress = np.asarray([supplied.get(value, current) for value, current in zip(warp.s, progress)])
    return _ProgressMap(warp.u.copy(), progress), warp
# END native-cubic scalar clock with bounded, source-preserving linearization.


def _torch_interpolate(keys, values, query):
    """Small tensor interpolation for the fixed progress schedule only."""
    import torch
    right = torch.searchsorted(keys, query.contiguous(), right=True).clamp(1, len(keys)-1)
    alpha = ((query-keys[right-1]) / (keys[right]-keys[right-1])).clamp(0, 1)
    return values[right-1] + alpha * (values[right]-values[right-1])


def _hermite_at_parameters(position, velocity, dt, counts, parameters):
    """Evaluate q and dq at exact u values without interpolating FK poses.

    The native degree-three samples include every polynomial breakpoint.
    Adjacent q/dq Hermite evaluation is therefore exact up to floating-point
    arithmetic.  Counts are per seed and exclude padded final-state entries.
    """
    import torch
    batch, _, dof = position.shape
    # BEGIN physical anchors retain their original binary64 parameter.
    coordinates = parameters.to(torch.float64) * (counts[:, None]-1)
    left = torch.minimum(coordinates.floor().long(), counts[:, None]-2).clamp_min(0)
    fraction = (coordinates-left).unsqueeze(-1)
    gather = left.unsqueeze(-1).expand(batch, -1, dof)
    q0, q1 = position.gather(1, gather).to(torch.float64), position.gather(1, gather+1).to(torch.float64)
    v0, v1 = velocity.gather(1, gather).to(torch.float64), velocity.gather(1, gather+1).to(torch.float64)
    delta = dt.reshape(batch, 1, 1).to(torch.float64)
    a = fraction
    q = ((2*a**3-3*a**2+1)*q0 + (a**3-2*a**2+a)*delta*v0
         + (-2*a**3+3*a**2)*q1 + (a**3-a**2)*delta*v1)
    dq = (((6*a**2-6*a)*q0 + (-6*a**2+6*a)*q1)/delta
          + (3*a**2-4*a+1)*v0 + (3*a**2-2*a)*v1)
    return q.to(position.dtype), dq.to(velocity.dtype)
    # END only the completed physical state is rounded for the native FK input.


# BEGIN project more tightly than every existing mechanism acceptance band.
def _projection_tolerance(problem):
    """Set native projection accuracy without changing equalities or acceptance."""
    bounds = [problem.numerical.position_m, problem.numerical.rotation_rad, problem.numerical.joint]
    for relation in problem.mechanisms:
        model = relation.model
        bounds.extend((model.position_tolerance, model.rotation_tolerance))
    return min(bounds)*.1
# END project more tightly than every existing mechanism acceptance band.


# BEGIN combine completed OMPL term flags before the single host decision.
def _ompl_terms_accepted(terms):
    """Keep every existing FP comparison and synchronize one boolean only.

    OMPL constructs all term tensors before acceptance. They share the state
    device; collecting their finite/nonpositive flags changes no floating
    operation, sum, tolerance, or report value. Boolean AND is exact, unlike
    reassociating a floating-point reduction. The empty collection keeps
    Python ``all``'s identity, although production also supplies scene/task
    terms. Only the final scalar crosses the device/host boundary.
    """
    import torch
    flags = [torch.isfinite(term).all() & (term <= 0).all() for term in terms.values()]
    return bool(torch.stack(flags).all().item()) if flags else True
# END no floating-point term is recomputed, combined, or rounded here.


def _ompl_stage(start, goal, lower, upper, valid, equality, *, tolerance,
                step, seconds, sample_limit, query_parameters=()):
    """Run native directed RRTConnect in (q,u=t/T), retaining checked edge samples.

    ``equality(x, derivative)`` returns true equality residuals or their
    Jacobian.  An empty residual selects ordinary OMPL space; inequality
    bands are checked by ``valid`` and are never collapsed to center lines.
    The same exporter accepts edges during tree growth and reconstructs the
    solution, including the native geodesic's final delta-length tail.
    """
    from ompl import base as ob, geometric as og

    dimension = len(start)
    query_parameters = np.unique(np.asarray(query_parameters, float))
    if (not np.isfinite(query_parameters).all() or np.any(query_parameters < lower[-1])
            or np.any(query_parameters > upper[-1])):
        raise ValueError("OMPL query times must lie in the frozen binding interval")
    deadline = time.monotonic()+seconds
    residual = np.asarray(equality(start, False), dtype=float)
    count = len(residual)
    report = {"space": "projected" if count else "ordinary", "equalities": count,
              "motion_checks": 0, "validity_checks": 0, "rejections": {}}
    if count >= dimension:
        return None, {**report, "failure": "equality_codimension_not_below_state_dimension"}

    class ProcessConstraint(ob.Constraint):
        def __init__(self):
            super().__init__(dimension, count, tolerance)

        def function(self, values, output):
            output[:] = equality(np.asarray(values), False)

        def jacobian(self, values, output):
            output[:] = equality(np.asarray(values), True)

    constraint = ProcessConstraint() if count else None
    if count:
        # Newton projection requires locally independent equality rows.  A
        # degenerate system is a bounded seed-search failure, never permission
        # to silently remove a process equality.
        for endpoint in (start, goal):
            jacobian = np.asarray(equality(endpoint, True))
            if not np.isfinite(jacobian).all() or np.linalg.matrix_rank(jacobian, tol=1e-7) < count:
                return None, {**report, "failure": "singular_equality_jacobian"}

    ambient = ob.RealVectorStateSpace(dimension)
    bounds = ob.RealVectorBounds(dimension)
    for index, (lo, hi) in enumerate(zip(lower, upper)):
        bounds.setLow(index, float(lo))
        bounds.setHigh(index, float(hi))
    ambient.setBounds(bounds)
    space = ob.ProjectedStateSpace(ambient, constraint) if count else ambient
    if count:
        space.setDelta(step*2)
    information = ob.ConstrainedSpaceInformation(space) if count else ob.SpaceInformation(space)
    information.setStateValidityCheckingResolution(min(.05, step/np.linalg.norm(upper-lower)))

    def vector(state):
        return np.asarray([state[index] for index in range(dimension)], dtype=float)

    def valid_array(value):
        report["validity_checks"] += 1
        reason = None
        if time.monotonic() > deadline:
            reason = "time_budget"
        elif report["validity_checks"] > 2*sample_limit:
            reason = "sample_budget"
        elif not np.isfinite(value).all():
            reason = "nonfinite_state"
        elif np.any(value < lower) or np.any(value > upper):
            reason = "state_bounds"
        elif constraint is not None and not constraint.isSatisfied(value):
            reason = "equality"
            if "first_equality_rejection" not in report:
                remaining = np.asarray(equality(value, False), dtype=float)
                report["first_equality_rejection"] = {
                    "progress": float(value[-1]), "tolerance": float(tolerance),
                    "l2": float(np.linalg.norm(remaining)), "max_abs": float(np.max(np.abs(remaining)))}
        elif not valid(value):
            reason = "process_or_scene"
        if reason is not None:
            report["last_rejection"] = reason
            report["rejections"][reason] = report["rejections"].get(reason, 0)+1
            return False
        return True

    information.setStateValidityChecker(lambda state: valid_array(vector(state)))
    information.setup()
    native_validator = information.getMotionValidator()
    accepted_edges = {}

    def edge_key(first, last):
        return first.tobytes(), last.tobytes()

    def checked_edge(a, b):
        first, last = vector(a), vector(b)
        key = edge_key(first, last)
        if key in accepted_edges:
            return accepted_edges[key]
        if (last[-1] < first[-1] or (last[-1] == first[-1] and not np.array_equal(last, first))
                or not valid_array(first) or not valid_array(last)):
            return None
        if not native_validator.checkMotion(a, b):
            return None
        # Installed OMPL does not bind discreteGeodesic's pair return type.
        # Its native path container exposes exactly that geodesic instead.
        path = og.PathGeometric(information)
        path.append(a)
        path.append(b)
        path.interpolate()
        vertices = np.asarray([vector(path.getState(i)) for i in range(path.getStateCount())])
        if (len(vertices) > sample_limit or not np.array_equal(vertices[0], first)
                or not np.array_equal(vertices[-1], last)):
            return None
        checked = [first]
        for left, right in zip(vertices[:-1], vertices[1:]):
            subdivisions = max(1, int(np.ceil(np.linalg.norm(right-left)/step)))
            if len(checked)+subdivisions > sample_limit:
                return None
            # BEGIN every crossed event/plateau boundary owns an exact time query.
            if right[-1] < left[-1] or (right[-1] == left[-1] and not np.array_equal(right, left)):
                return None
            fractions = np.arange(1, subdivisions+1, dtype=float)/subdivisions
            required = query_parameters[(query_parameters > left[-1]) & (query_parameters < right[-1])]
            if len(required):
                fractions = np.unique(np.r_[fractions, (required-left[-1])/(right[-1]-left[-1])])
            if len(checked)+len(fractions) > sample_limit:
                return None
            for fraction in fractions:
                point = right.copy() if fraction == 1. else left+(right-left)*fraction
                # Projection is allowed to change q, never the crossed query's time.
                if fraction != 1. and constraint is not None:
                    point = _project_boundary(point, equality, tolerance, deadline=deadline)
                    if point is None:
                        return None
                if (point[-1] < checked[-1][-1]
                        or (point[-1] == checked[-1][-1] and not np.array_equal(point, checked[-1]))
                        or not valid_array(point)):
                    return None
                checked.append(point)
            # END monotone u also permits changing q during a frozen-s plateau.
        samples = np.asarray(checked)
        if not np.array_equal(samples[-1], last):
            return None
        # Bound retained edge geometry independently of RRT's native tree.
        if len(accepted_edges) >= 256:
            accepted_edges.pop(next(iter(accepted_edges)))
        accepted_edges[key] = samples
        return samples

    class ForwardValidator(ob.MotionValidator):
        def __init__(self):
            super().__init__(information)

        def checkMotion(self, a, b, *unused):
            report["motion_checks"] += 1
            return checked_edge(a, b) is not None

    information.setMotionValidator(ForwardValidator())
    try:
        a, b = space.allocState(), space.allocState()
        if constraint is not None:
            a.copy(start.tolist())
            b.copy(goal.tolist())
        else:
            for index in range(dimension):
                a[index], b[index] = float(start[index]), float(goal[index])
        report["initialization_seconds"] = seconds-max(0., deadline-time.monotonic())
        for label, value in (("start", start), ("goal", goal)):
            if not valid_array(value):
                failure = ("search_budget_exhausted_during_boundary_check"
                           if report["last_rejection"] in ("time_budget", "sample_budget")
                           else "boundary_candidate_not_feasible")
                return None, {**report, "failure": failure, "boundary": label}
        # Try the native connector before sampling unrelated configurations.
        # Projected space uses its checked manifold geodesic here; ordinary
        # space uses its checked linear edge. Both retain the exact caller
        # endpoints and every validity/progress/tail check used by RRT edges.
        report["motion_checks"] += 1
        direct = checked_edge(a, b)
        report["direct_connection_accepted"] = direct is not None
        if direct is not None:
            return direct, {**report, "samples": len(direct), "strategy": "native_direct_connection"}
        if time.monotonic() >= deadline:
            return None, {**report, "failure": "whole_search_time_budget_exhausted"}
        definition = ob.ProblemDefinition(information)
        definition.setStartAndGoalStates(a, b)
        search = og.RRTConnect(information)
        search.setIntermediateStates(False)
        search.setProblemDefinition(definition)
        search.setup()
        # Setup and native equality/Jacobian checks consume this same stage
        # budget; passing its original duration again would grant extra time.
        search.solve(max(0., deadline-time.monotonic()))
        if not definition.hasExactSolution():
            return None, {**report, "failure": "no_exact_path_within_search_budget"}
        path = definition.getSolutionPath()
        segments = [checked_edge(path.getState(i), path.getState(i+1))
                    for i in range(path.getStateCount()-1)]
        if not segments or any(segment is None for segment in segments):
            return None, {**report, "failure": "solution_edge_export_rejected"}
        samples = np.concatenate((segments[0], *(segment[1:] for segment in segments[1:])))
        if (len(samples) > sample_limit or np.any(np.diff(samples[:, -1]) < 0)
                or not np.array_equal(samples[0], start) or not np.array_equal(samples[-1], goal)):
            return None, {**report, "failure": "solution_samples_exceed_budget_or_change_endpoints"}
        return samples, {**report, "samples": len(samples), "strategy": "rrt_connect"}
    except TimeoutError:
        return None, {**report, "failure": "whole_search_time_budget_exhausted"}
    finally:
        # Native SI owns the Python callback, whose closure owns SI.  Break
        # that cycle explicitly so repeated requests release native objects.
        information.setMotionValidator(native_validator)


# BEGIN native boundary projection keeps allowed goals as sets of solutions.
def _independent_projection(equality, witness):
    """Select independent local equations without dropping their validity test.

    OMPL's fixed-codimension adapter cannot store inactive zero rows or
    duplicate equations. Pivoting selects coordinates for its Newton step;
    the complete original relations still decide state and edge acceptance.
    """
    from scipy.linalg import qr
    jacobian = np.asarray(equality(witness, True), float)
    if not len(jacobian):
        return equality, 0
    if not np.isfinite(jacobian).all():
        return None, 0
    _, triangular, order = qr(jacobian.T, pivoting=True, mode="economic")
    rank = int(np.linalg.matrix_rank(triangular, tol=1e-7))
    if not rank:
        # A singular real relation is a failed local chart, never an ordinary
        # unconstrained search substituted for the missing manifold.
        return None, 0
    rows = np.sort(order[:rank])
    return lambda value, derivative: equality(value, derivative)[rows], rank


def _project_allowed_boundary(value, residual, process_count, tolerance, lower, upper, *, deadline):
    """Use native OMPL Newton projection into true boundary allowed sets.

    The callback concatenates signed process equations and nonnegative goal
    violations. It adds no nominal target or equation for a free direction.
    Independent rows are refreshed between bounded native Newton iterations,
    so a satisfied band contributes no fictitious equality. Time is fixed.
    Physical bounds and the original goal thresholds are checked unchanged.

    A reflected native step is tested for strict membership when the native
    step remains outside. For an active smooth inequality, J*d=-g makes the
    linearized value at x+2*d equal -g rather than zero. This can cross a
    floating-point boundary plateau without changing any physical tolerance.
    Reflection is only a proposal: every original relation must accept it.
    """
    from ompl import base as ob
    from scipy.linalg import qr
    point = np.asarray(value, float).copy()
    dimension = len(point)
    if not np.isfinite(point).all() or np.any(point < lower) or np.any(point > upper):
        return None

    # BEGIN a single complete membership predicate owns every projected trial.
    def allowed(candidate):
        if (time.monotonic() >= deadline or not np.isfinite(candidate).all()
                or np.any(candidate < lower) or np.any(candidate > upper)):
            return False
        error = np.asarray(residual(candidate, False), float)
        return (time.monotonic() < deadline and np.isfinite(error).all()
                and np.all(np.abs(error[:process_count]) <= tolerance)
                and np.all(error[process_count:] <= 0))

    # END the original callback still bills all trial queries to the request.
    for _ in range(24):
        if time.monotonic() >= deadline:
            return None
        error = np.asarray(residual(point, False), float)
        # BEGIN an expensive initial query cannot publish an overdue point.
        if time.monotonic() >= deadline or not np.isfinite(error).all():
            return None
        # END subsequent native/reflected trials use the same post-query check.
        if (np.all(np.abs(error[:process_count]) <= tolerance)
                and np.all(error[process_count:] <= 0)):
            return point if np.all(point >= lower) and np.all(point <= upper) else None
        jacobian = np.asarray(residual(point, True), float)
        if not np.isfinite(jacobian).all():
            return None
        # The last coordinate is the frozen query time, not a Newton degree
        # of freedom. Choosing rows on the joint columns also permits a full
        # D-DOF pose goal in the D+1 dimensional time-state representation.
        _, triangular, order = qr(jacobian[:, :-1].T, pivoting=True, mode="economic")
        rank = int(np.linalg.matrix_rank(triangular, tol=1e-7))
        if not rank:
            return None
        rows = np.sort(order[:rank])

        class AllowedBoundary(ob.Constraint):
            def __init__(self):
                # The callback's goal entries already subtract the original
                # allowed error. A positive Newton convergence allowance here
                # would stop just outside that set (notably at a support
                # plane). One native step must act on every positive goal
                # violation; our outer test still uses the declared process
                # projection tolerance and exact original goal membership.
                super().__init__(dimension, rank, np.finfo(float).tiny)
                self.setMaxIterations(1)

            def function(self, values, output):
                output[:] = residual(np.asarray(values), False)[rows]

            def jacobian(self, values, output):
                result = residual(np.asarray(values), True)[rows].copy()
                result[:, -1] = 0.
                output[:] = result

        constraint = AllowedBoundary()
        # BEGIN verify the final native update and its reflected interior trial.
        # A one-iteration False is not convergence evidence either way. Keep
        # the native point for continuation unless a complete allowed-set
        # query certifies its reflection. No residual is rounded down to zero.
        previous = point.copy()
        constraint.project(point)
        point[-1] = value[-1]
        if not np.isfinite(point).all() or np.any(point < lower) or np.any(point > upper):
            return None
        if allowed(point):
            return point
        reflected = previous+2.*(point-previous)
        reflected[-1] = value[-1]
        if allowed(reflected):
            return reflected
        # END at most 24 native updates; a failed reflection is never retained.
    return None
# END boundary witnesses remain revisable proposals within the same goal.


def _project_boundary(value, equality, tolerance, *, deadline=None):
    """Refine an IK boundary using native projection while preserving its s.

    Alternating native manifold projection and the fixed progress plane is
    bounded.  Failure remains a rejected seed; measured q0 never enters this
    function.  No custom inverse-kinematics or projection solver is used.
    """
    from ompl import base as ob
    dimension, count = len(value), len(equality(value, False))
    if not count:
        return value.copy()
    if count >= dimension:
        return None

    class BoundaryConstraint(ob.Constraint):
        def __init__(self):
            super().__init__(dimension, count, tolerance)

        def function(self, values, output):
            output[:] = equality(np.asarray(values), False)

        def jacobian(self, values, output):
            output[:] = equality(np.asarray(values), True)

    constraint = BoundaryConstraint()
    point = value.copy()
    for _ in range(12):
        if deadline is not None and time.monotonic() >= deadline:
            return None
        if constraint.isSatisfied(point):
            return point
        if not constraint.project(point) or not np.isfinite(point).all():
            return None
        point[-1] = value[-1]
    return point if constraint.isSatisfied(point) else None


def _orientation_vjp_type():
    """Connect native quaternion values to native world geometric Jacobians.

    V2's native ToolPoseCost encodes angular gradients as ``q * omega`` and
    its FK backward consumes that matching body encoding.  An ordinary Torch
    scalar loss instead supplies a Euclidean quaternion gradient; feeding it
    to that native pair rotates its angular gradient into the wrong frame.
    The native world-frame geometric Jacobian supplies the correct derivative
    for custom process/scene losses without changing FK or native pose costs.

    The independent FK central-difference probe checks this chain, including
    orientations near 180 degrees where the original mismatch changes signs.
    """
    import torch

    class WorldQuaternionVJP(torch.autograd.Function):
        @staticmethod
        def forward(ctx, joint_position, quaternion, angular_jacobian):
            # Shapes are [B,H,D], [B,H,L,4] (wxyz), [B,H,L,3,D].
            # Native buffers stay valid until this rollout's backward pass.
            ctx.save_for_backward(quaternion.detach(), angular_jacobian.detach())
            return quaternion.clone()

        @staticmethod
        @torch.autograd.function.once_differentiable
        def backward(ctx, gradient):
            quaternion, jacobian = ctx.saved_tensors
            w, vector = quaternion[..., :1], quaternion[..., 1:]
            gw, gv = gradient[..., :1], gradient[..., 1:]
            # qdot = 1/2 * [0, omega_world] * q.  Its transpose maps a
            # quaternion covector to the world angular covector below.
            angular = .5*(-vector*gw + w*gv + torch.linalg.cross(vector, gv))
            joint_gradient = torch.einsum("...lkd,...lk->...d", jacobian, angular)
            # The quaternion argument deliberately gets no derivative:
            # otherwise native FK's distinct quaternion convention would
            # contribute a second, incorrect angular gradient.
            return joint_gradient, None, None

    return WorldQuaternionVJP


def _native_gradient_layout_type():
    """Meet native FK's contiguous-gradient contract after Torch view ops.

    Concatenating main and exact-anchor samples yields strided gradient slices
    for batch sizes above one.  Cloning forward values does not change that
    backward layout, so normalize it at the native output boundary itself.
    This identity changes neither FK values nor their mathematical gradient.
    """
    import torch

    class NativeGradientLayout(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value):
            return value

        @staticmethod
        def backward(ctx, gradient):
            return gradient.contiguous()

    return NativeGradientLayout


# BEGIN preserve a gradient toward the explicit verification-stop command.
def _verification_stop_preference(progress, joint_velocity, stop_progress, *, trajectory_time=None):
    """Squared speed at each declared stop's first physical occurrence.

    A verification stop commands zero velocity. Its numerical acceptance
    band still belongs exclusively to PlanningProblem.residuals(). Keeping
    this optimizer objective active inside that band avoids terminating on
    its floating-point edge; ordinary motion receives no added objective.
    """
    import torch
    from .planning_problem import _first_event_query
    stopped = torch.zeros_like(progress, dtype=torch.bool)
    for value in np.unique(stop_progress):
        # Use exactly the same progress support as the hard stop residual.
        stopped = stopped | _first_event_query(progress, value, trajectory_time)
    return torch.where(stopped, joint_velocity.square().sum(-1), torch.zeros_like(progress))
# END preserve a gradient toward the explicit verification-stop command.


# BEGIN one formal output clock and native named-body FK for timed hand geometry.
def _formal_native_clock(trajopt):
    """Bind hand, rollout and saved-curve export to one physical clock."""
    # BEGIN use the same admitted grid as the native interpolation adapter.
    from curobo._src.types.control_space import ControlSpace
    config = trajopt.config
    if config.minimum_trajectory_dt != config.maximum_trajectory_dt:
        raise ValueError("A separately timed hand requires one fixed request-wide native dt")
    control = trajopt.transition_model.control_space
    if control != ControlSpace.BSPLINE_3:
        raise ValueError("The formal clock must use the selected native cubic spline representation")
    grid = _fixed_spline_grid(
        # BEGIN M=N+2 controls have M+2=N+4 physical intervals.
        intervals=trajopt.action_horizon+2,
        # END cuRobo's four-fixed-prefix interval helper no longer applies.
        native_dt=config.maximum_trajectory_dt, interpolation_steps=trajopt.interpolation_steps,
        interpolation_dt=config.interpolation_dt)
    return {**grid, "source": "instantaneous-boundary cubic on unchanged exact float32 clock"}
    # END dense sampling never invents a new physical duration.


# BEGIN state time is the generating integer sample index times its true dt.
def _joint_state_sample_times(joint_state, counts):
    """Round physical time once, without a normalized-float32 roundtrip."""
    import torch
    position = joint_state.position
    indices = torch.minimum(torch.arange(position.shape[1], device=position.device)[None],
                            counts[:, None]-1)
    return (indices.to(torch.float64)*joint_state.dt.reshape(-1, 1).to(torch.float64)).to(position.dtype)
# END padding repeats the same final instant as the shared cubic sampler.





# BEGIN scene switches share q/t while querying distinct adjacent modes.
def _scene_boundary_evaluate(scene, problem, progress_map, spheres, position, quaternion,
                             progress, joint_position, joint_names, *, trajectory_time,
                             normalized_time, query_counts=None, component_diagnostics=None,
                             query_role="acceptance", scene_evidence_out=None,
                             optimizer_candidate_evidence=False, local_models=None,
                             geometry_evidence_out=None, geometry_scope="full"):
    """Check material/attachment mode limits without changing physical time.

    Float64 neighbors vanish when native inputs are float32. Construct the
    side in the actual progress dtype, retaining original samples first so
    release FK keeps its original exact witness. On a progress plateau only
    its entrance consumes the old side and only its exit consumes the new
    side; an action during that plateau never regains an expired old mode.
    """
    import torch
    from .optimizer_device_checks import constant_tensor
    pair_evidence = [] if geometry_evidence_out is not None else None
    def evaluate(*args, **kwargs):
        return scene.evaluate(*args, **kwargs, query_role=query_role, local_models=local_models,
                              geometry_evidence_out=pair_evidence, geometry_scope=geometry_scope)

    # BEGIN retain physical query identity through same-time mode expansion.
    def retain_pair_evidence(exact_count, left_columns=(), right_columns=()):
        if geometry_evidence_out is None:
            return
        for row in pair_evidence:
            batch, column = row["state_index"]
            if column < exact_count:
                source, side = column, 0
            elif column < exact_count+len(left_columns):
                source = int(left_columns[column-exact_count])
                side = -1 if bool(left[batch, source]) else 0
            else:
                source = int(right_columns[column-exact_count-len(left_columns)])
                side = 1 if bool(right[batch, source]) else 0
            geometry_evidence_out.append({**row, "query_index": [batch, source], "side": side})
    # END geometry proofs consume the same pair query as hard residuals.
    # BEGIN classify scene evidence without changing the hard scene sum.
    # The recovery consumer needs actual collision components, not the mixed
    # process_scene constraint (which also includes missing model support,
    # unknown space, contact requirements and controller violations). Only
    # requested metric/diagnostic queries retain collision, scene-update and
    # forbidden-coverage components from this same call, without conflation.
    # BEGIN diagnostics reuse the same observed-field component calculation.
    if component_diagnostics is not None or scene_evidence_out is not None:
        base_evaluate = evaluate
        def evaluate(*args, **kwargs):
            components = {}
            value = base_evaluate(*args, **kwargs, components_out=components)
            if scene_evidence_out is not None:
                # BEGIN retain independent same-query scene failure categories.
                # BEGIN explicit forbidden coverage can request a different path.
                for category, prefix in (("collision", "collision/"),
                                         ("scene_update", "scene_update/"),
                                         ("coverage", "unknown/forbidden/"),
                                         ("conditional_unknown", "conditional/unknown/")):
                    scene_evidence_out[category] = sum((term.detach() for name, term in components.items()
                        if name.startswith(prefix)), torch.zeros_like(value))
                # END conditional unknown and missing model data are not forbidden space.
                # END no category changes the original hard aggregate or gradient.
                # BEGIN original scene components determine hard candidate feasibility.
                if optimizer_candidate_evidence:
                    # Every observed-model component retains its own hard
                    # verdict before aggregation or optimizer preferences.
                    # BEGIN classification values remain independent of local cost.
                    hard = sum((term.detach() for name, term in components.items()
                                if not name.startswith("conditional/")), torch.zeros_like(value))
                    invalid = torch.zeros_like(value, dtype=torch.bool)
                    for name, term in components.items():
                        if name.startswith("conditional/"):
                            continue
                        term = term.detach()
                        invalid = invalid | ~torch.isfinite(term)
                        invalid = invalid | (term > 0)
                    if not components:
                        # An aggregate-only scene still supplies its hard value.
                        hard = value.detach()
                        invalid = invalid | (value.detach() > 0)
                        invalid = invalid | ~torch.isfinite(value.detach())
                    # Evaluate each component before reducing modes so small
                    # violations remain visible beside large residuals.
                    scene_evidence_out["optimizer_hard_invalid"] = invalid.to(value.dtype)
                    scene_evidence_out["optimizer_hard_value"] = hard
                    # END the positive softplus value cannot reject a clear state.
                # END hard evidence remains separate from the fixed local geometry objective.
            if component_diagnostics is not None:
                for name, residual in components.items():
                    flat = residual.detach().reshape(-1)
                    # A finite maximum must not hide another NaN or -inf.
                    # Use the first non-finite witness when present, retaining
                    # its original value and physical time in this summary.
                    index = int(torch.where(torch.isfinite(flat), flat,
                        torch.full_like(flat, float("inf"))).argmax().item())
                    maximum = float(flat[index].item())
                    if maximum > 0 or not np.isfinite(maximum):
                        component_diagnostics[name] = {"maximum": maximum,
                            "time_s": float(kwargs["trajectory_time"].detach().reshape(-1)[index].item()),
                            "progress": float(args[3].detach().reshape(-1)[index].item())}
            return value
    # END diagnostics never rerun, relax, or replace any scene constraint.
    anchors = [float(value) for value in scene.required_progress_anchors
               if problem.start_progress <= value <= problem.end_progress]
    # BEGIN compact side-query shapes are explicitly eager-only.
    # Production solvers already disable CUDA graphs. Reject direct capture
    # callers before nonzero or geometry allocation instead of silently using
    # the former three-times-expanded query as a compatibility fallback.
    if anchors and progress.is_cuda and torch.cuda.is_current_stream_capturing():
        raise MotionPlanningError("unsupported_model",
            "Compact scene mode-boundary queries require eager execution; CUDA graph capture is unsupported")
    # END the unrelated native-only optimizer graph API remains unchanged.
    if not anchors:
        if query_counts is not None:
            query_counts.update(exact_states=int(progress.numel()), left_states=0, right_states=0,
                                evaluated_states=int(progress.numel()))
        result = evaluate(spheres, position, quaternion, progress, joint_position,
                          joint_names, trajectory_time=trajectory_time)
        retain_pair_evidence(progress.shape[1])
        return result
    left = torch.zeros_like(progress, dtype=torch.bool)
    right = torch.zeros_like(progress, dtype=torch.bool)
    for anchor in anchors:
        parameters = progress_map.anchor_parameters([anchor])[:, 0]
        # BEGIN preserve exact native-dtype anchor constants during replay.
        matching = progress == constant_tensor(anchor, progress)
        if anchor > problem.start_progress:
            left |= matching & (normalized_time <= constant_tensor(parameters[0], normalized_time))
        if anchor < problem.end_progress:
            right |= matching & (normalized_time >= constant_tensor(parameters[-1], normalized_time))
        # END only constant storage changes, including each original FP32 cast.
    lower = torch.where(left, torch.nextafter(progress, torch.full_like(progress, -float("inf"))), progress)
    upper = torch.where(right, torch.nextafter(progress, torch.full_like(progress, float("inf"))), progress)
    # BEGIN exact arrival activates only declared finite material permission.
    # The same-q left-limit probe keeps attachment/controller laws separate
    # from the material permission granted at this exact required arrival.
    permission_lower = lower
    for rule in scene.contacts:
        if rule.actor != 'robot':
            continue
        arrival = constant_tensor(rule.interval[0], progress)
        permission_lower = torch.where((progress == arrival) & (lower < arrival), progress, permission_lower)
    # END equality is in the native query dtype; no tolerance advances arrival.
    # BEGIN evaluate exact witnesses once and only append real mode-side columns.
    # Keep the union of columns over the batch so native geometry retains its
    # rectangular [B,T,...] layout. A batch without a side at that column uses
    # its unchanged exact mode; no physical q/time or arrival permission moves.
    left_columns = left.any(0).nonzero().flatten()
    right_columns = right.any(0).nonzero().flatten()
    def expanded(value):
        return torch.cat((value, value.index_select(1, left_columns),
                          value.index_select(1, right_columns)), dim=1)
    expanded_progress = torch.cat((progress, lower.index_select(1, left_columns),
                                   upper.index_select(1, right_columns)), dim=1)
    permission_progress = torch.cat((progress, permission_lower.index_select(1, left_columns),
                                     upper.index_select(1, right_columns)), dim=1)
    if query_counts is not None:
        batch, exact = progress.shape
        query_counts.update(exact_states=batch*exact, left_states=batch*left_columns.numel(),
                            right_states=batch*right_columns.numel(),
                            evaluated_states=batch*expanded_progress.shape[1])
    # All exact states stay first in the same call: released owner poses need
    # their exact release witness, even when a different time probes a side.
    terms = evaluate(expanded(spheres), expanded(position), expanded(quaternion),
        expanded_progress, expanded(joint_position), joint_names,
        trajectory_time=expanded(trajectory_time),
        contact_permission_progress=permission_progress)
    retain_pair_evidence(progress.shape[1], left_columns, right_columns)
    exact_count, left_count = progress.shape[1], left_columns.numel()
    exact_terms = terms[:, :exact_count]
    left_terms = exact_terms.scatter(1, left_columns[None].expand(progress.shape[0], -1),
                                     terms[:, exact_count:exact_count+left_count])
    right_terms = exact_terms.scatter(1, right_columns[None].expand(progress.shape[0], -1),
                                      terms[:, exact_count+left_count:])
    # BEGIN each diagnostic category reduces independently of the aggregate.
    # A support gap may win the hard max while a different side has a real
    # collision. Preserve both categories without changing the hard winner.
    if scene_evidence_out is not None:
        for category, evidence in scene_evidence_out.items():
            exact_evidence = evidence[:, :exact_count]
            left_evidence = exact_evidence.scatter(1, left_columns[None].expand(progress.shape[0], -1),
                evidence[:, exact_count:exact_count+left_count])
            right_evidence = exact_evidence.scatter(1, right_columns[None].expand(progress.shape[0], -1),
                evidence[:, exact_count+left_count:])
            scene_evidence_out[category] = torch.stack((exact_evidence, left_evidence, right_evidence)).amax(0)
    # Reconstruct only these small scalar residuals before max. Its original
    # three-way tie weights matter for gradients: replacing this with a max
    # over the compact query would change derivatives at equal-cost modes.
    return torch.stack((exact_terms, left_terms, right_terms)).amax(0)
    # END duplicate scalar costs are cheap; duplicate geometry queries are gone.




def _process_manager_type(problem, scene, progress_map, anchors, budget,
                          *, bound_reference, clock=None, seed_parameter=None, robot_dict=None,
                          phase="feasibility", functional_constraints=None):
    """Build V2's documented manager extension after GPU use is requested.

    Configuration construction must return the derived type and class_type;
    registering a custom name alone is not dispatched by V2's base manager.
    The ordinary cost manager evaluates the transferred reference and native
    joint regularization. Constraint managers retain built-in joint limits,
    declared task constraints and the same frozen observed scene model.
    """
    import torch
    from curobo.kinematics import Kinematics
    from curobo.types import JointState
    from curobo._src.rollout.cost_manager.cost_manager_robot import RobotCostManager
    from curobo._src.rollout.cost_manager.cost_manager_robot_cfg import RobotCostManagerCfg
    # BEGIN exact-anchor collision checks use the same native implementation.
    from curobo._src.state.state_robot import RobotState
    # END no separate collision formula or policy is introduced.
    orientation_vjp = _orientation_vjp_type()
    native_layout = _native_gradient_layout_type()
    # BEGIN one fixed functional objective is shared by this solver's rollouts.
    from .functional_augmented_lagrangian import FunctionalAugmentedLagrangian
    if functional_constraints is None:
        functional_constraints = FunctionalAugmentedLagrangian(problem.functional_constraint_keys)
    # END a standalone manager uses the same zero-multiplier objective.

    managers = []
    # BEGIN all native optimizer rollouts consume one fixed local block model.
    from .local_geometry_model import LocalGeometryModels
    geometry_models = LocalGeometryModels()
    # END the core captures actual base states before initial cost/gradient.
    # BEGIN a short-lived event context selects the task allowed set for IK.
    # Reference IK leaves this empty and retains its original pose objective.
    ik_goal_ids = ()
    # END the same native IK instance owns both explicitly selected purposes.
    # BEGIN initializer and full solve share official robot self geometry.
    # The scope selects computation: robot self/hand state, or that same
    # calculation plus task fields, contact, support and observed coverage.
    optimizer_geometry_scope = "full"
    # END this scope changes computation, never a task declaration or tolerance.
    # BEGIN collision witnesses retain physical time, including plateau interiors.
    collision_witnesses = np.empty(0, dtype=float)

    def query_anchor_parameters():
        pairs = progress_map.anchor_parameters(anchors)
        if len(collision_witnesses):
            extra = np.column_stack((collision_witnesses,
                                     np.interp(collision_witnesses, progress_map.u, progress_map.s)))
            pairs = np.concatenate((pairs, extra))
            pairs = pairs[np.argsort(pairs[:, 0], kind="stable")]
        return pairs
    # END progress alone cannot identify a collision inside a hold.

    class ProcessCostManager(RobotCostManager):
        def initialize_from_config(self, config, transition_model, scene_collision_checker=None, **kwargs):
            # BEGIN give each native manager its own mutable cost configuration.
            # Native initialization divides self-collision weight by its
            # interpolation count and binds cspace limits/types in place. The
            # rollout template is shared by several managers and rebuilds;
            # preserve it so each instance receives the original parameters.
            config = deepcopy(config)
            # END give each native manager its own mutable cost configuration.
            super().initialize_from_config(config, transition_model, scene_collision_checker, **kwargs)
            self._process_mode = config.process_mode
            managers.append(self)
            self._dense_counts = None
            # BEGIN metric evidence is request-local and bound by tensor identity.
            self._scene_evidence = None
            # BEGIN optimizer evidence belongs to exactly one completed forward.
            self._optimizer_evidence = None
            # END it is not a new metric or an execution authorization.
            # END optimizer/IK managers never supply collision recovery evidence.
            # BEGIN retain only aggregate last-evaluation counts, never states.
            self._query_counts = {"process_mode": self._process_mode}
            # END request rebind clears this lightweight diagnostic in place.
            if not self._process_mode:
                return
            # BEGIN ordinary reference costs own no scene or event-anchor queries.
            if self._process_mode == "reference_objective":
                self._reference_pose_weight = config.tool_pose_cfg.weight.detach().cpu().tolist()
                self._cspace_template = deepcopy(self.get_cost("cspace").config)
                self.bind_request_data()
                return
            # END task and geometry constraints remain in their own manager.
            transition_model.robot_model.compute_jacobian = True
            self._scene = scene.clone_for_rollout()
            self._joint_names = tuple(transition_model.joint_names)
            self._tool_index = transition_model.robot_model.tool_frames.index(problem.tool_link)
            # BEGIN ordinary reference IK does not certify a task trajectory.
            # The retained private hand clock supplies native self-collision
            # geometry, but nominal pose samples need neither scene coverage
            # nor functional/attachment feasibility. Those checks remain on
            # every full trajectory and on independent search/verification.
            if self._process_mode.startswith("seed_"):
                return
            # END do not allocate coverage or anchor FK for reference IK.
            self.bind_request_data()
            # A separate native FK instance owns the anchor buffers.  Reusing
            # the main rollout FK would overwrite values needed by backward.
            self._anchor_fk = Kinematics(transition_model.robot_model.config, compute_jacobian=True)

        def bind_request_data(self):
            """Replace request values in fixed-capacity native manager buffers."""
            self._dense_counts = None
            # BEGIN rebind ordinary costs from the unscaled native configuration.
            if self._process_mode == "reference_objective":
                from .reference_tracking import NativeReferenceCost, FixedTimeStateCSpaceCost, freeze_cspace_time_weights
                self._reference_cost = NativeReferenceCost(problem, self.device_cfg,
                    pose_weight=self._reference_pose_weight)
                config = freeze_cspace_time_weights(deepcopy(self._cspace_template), clock["native_dt"])
                self.costs["cspace"] = FixedTimeStateCSpaceCost(config, clock["native_dt"])
                self._batch_size = self._horizon = None
                self._reference_grid_key = None
                return
            # END each request freezes weights once on its admitted physical clock.
            self._scene = scene.clone_for_rollout()
            # BEGIN a rebind invalidates the previous query's recovery evidence.
            self._scene_evidence = None
            # BEGIN request/clock/anchor rebinding invalidates optimizer evidence.
            self._optimizer_evidence = None
            # END no previous curve supplies hard-constraint evidence.
            # END no old batch or owner binding may enter the new request.
            self._query_counts.clear()
            self._query_counts.update(process_mode=self._process_mode)
            if not self._process_mode or self._process_mode.startswith("seed_"):
                return
            if len(progress_map.u) > _MAX_PROGRESS_KEYS:
                raise MotionPlanningError("capacity_exceeded", "Progress clock exceeds its fixed capacity")
            pad = _MAX_PROGRESS_KEYS-len(progress_map.u)
            keys = np.r_[progress_map.u, 1.+np.arange(1, pad+1)]
            values = np.r_[progress_map.s, np.full(pad, progress_map.s[-1])]
            pairs = query_anchor_parameters()
            _anchor_capacity(progress_map, anchors, budget)
            # BEGIN capacity admits requests; only actual witnesses are evaluated.
            anchor_count = self._anchor_count = len(pairs)
            self._query_counts.update(anchor_count=anchor_count, anchor_capacity=budget.max_task_anchors)
            # END fixed host/device input storage still supports later rebindings.
            pairs = np.concatenate((pairs, np.repeat(pairs[-1:], budget.max_task_anchors-anchor_count, axis=0)))
            self._anchor_valid = self.device_cfg.to_device(np.arange(budget.max_task_anchors) < anchor_count)[None]
            for name, value in (("_map_u", keys), ("_map_s", values),
                                ("_anchor_u", pairs[:, 0][None]), ("_anchor_s", pairs[:, 1][None])):
                tensor = self.device_cfg.to_device(value)
                previous = getattr(self, name, None)
                if previous is None:
                    setattr(self, name, tensor)
                elif previous.shape != tensor.shape:
                    raise MotionPlanningError("capacity_exceeded", "A request changed fixed manager capacity")
                else:
                    previous.copy_(tensor)
            # BEGIN preserve source anchor identity before native dtype rounding.
            query_u = torch.as_tensor(pairs[:, 0][None], dtype=torch.float64,
                                      device=self.device_cfg.device)
            if not hasattr(self, "_anchor_query_u"):
                self._anchor_query_u = query_u
            elif self._anchor_query_u.shape != query_u.shape:
                raise MotionPlanningError("capacity_exceeded", "A request changed fixed anchor capacity")
            else:
                self._anchor_query_u.copy_(query_u)
            # END q/dq sampling and physical time use this same original value.
            functional_constraints.bind(self._anchor_u)

        def set_dense_counts(self, counts):
            self._dense_counts = counts.reshape(-1).long()
            # BEGIN ordinary dense quadrature follows the newly supplied valid prefix.
            if self._process_mode == "reference_objective":
                self._reference_grid_key = None
            # END cached goals are rebuilt at the next ordinary-cost query.

        # BEGIN same-forward hard evidence and fixed objective for candidate retention.
        @property
        def optimization_constraints(self):
            return functional_constraints if self._process_mode == "trajectory_optimizer" else None

        @property
        def local_geometry_models(self):
            return (geometry_models if self._process_mode == "trajectory_optimizer"
                    and optimizer_geometry_scope == "full" else None)

        def optimizer_evidence(self, trajectories):
            """Read hard evidence and objective from this same rollout.

            Functional multipliers and centering/command preferences are
            objectives. Original task and observed-scene residuals determine
            sampled eligibility under the same nonpenetration rule as final
            metrics.
            """
            constraints = trajectories.costs_and_constraints
            collection = constraints.constraints
            if self._optimizer_evidence is None:
                raise RuntimeError("Optimizer has no same-query hard-constraint evidence")
            source, feasible, signed, scene_source = self._optimizer_evidence
            # BEGIN evidence describes exactly the constraints computed by this forward.
            scene_present = "process_scene" in collection.names
            if ("functional_augmented_lagrangian" not in collection.names
                    or collection.values[collection.names.index("functional_augmented_lagrangian")] is not source
                    or scene_present != (scene_source is not None)
                    or (scene_present and collection.values[collection.names.index("process_scene")] is not scene_source)):
                raise RuntimeError("Optimizer evidence belongs to a different native query")
            # END absent scene evidence remains absent, rather than a passed scene check.
            objective_names = {"functional_augmented_lagrangian", "skill_pose_equality_preference",
                               "gripper_command_preference", "verification_stop_preference",
                               # BEGIN raw scene feasibility was recorded before path substitution.
                               "process_scene"}
            # END optimizer path values never stand in for hard geometry evidence.
            fixed_objective = source.new_zeros(source.shape[0])
            for costs, hard in ((constraints.costs, False), (collection, True),
                                (constraints.hybrid_costs_constraints, True)):
                for name, value in zip(costs.names, costs.values):
                    detached = value.detach().flatten(1)
                    if hard and name not in objective_names:
                        finite = torch.isfinite(detached).all(-1)
                        valid = finite & (detached <= 0).all(-1)
                        feasible = feasible & valid
                    # BEGIN incumbent ranking is independent of changing block models.
                    if not hard:
                        fixed_objective = fixed_objective+detached.sum(-1)
                    # END actual geometry determines eligibility, not this score.
            feasible = feasible & torch.isfinite(fixed_objective)
            return feasible, signed, fixed_objective
        # END these tensors preserve query identity and never add a cost term.

        # BEGIN bind diagnostics only to the exact completed native metric query.
        def bind_scene_evidence(self, metrics):
            """Attach the last query's tensor only when its constraint is identical.

            Native CostCollection.merge retains the constraint tensor object.
            Shape equality would not distinguish another candidate/seed batch;
            object identity does, without storing or comparing complete states.
            RolloutMetrics.clone already clones a tensor-valued debug field.
            """
            constraints = metrics.costs_and_constraints.constraints
            if self._scene_evidence is None or "process_scene" not in constraints.names:
                raise RuntimeError("Native scene metrics have no same-query scene evidence")
            source, evidence = self._scene_evidence
            if constraints.values[constraints.names.index("process_scene")] is not source:
                raise RuntimeError("Scene evidence belongs to a different native metric query")
            # BEGIN fixed diagnostic channels survive native metric clone and seed slicing.
            metrics.debug = evidence.detach().clone()
            # END debug carries collision/scene_update/coverage, never a hard cost.
        # END diagnostics do not enter native costs, feasibility or seed ranking.

        def compute_costs(self, state, cost_collection=None, goal=None, **kwargs):
            # BEGIN one-shot explicit queries reuse the ordinary native metrics.
            if not self._process_mode:
                return super().compute_costs(state, cost_collection, goal, **kwargs)
            context = getattr(self, "_candidate_query", None)
            # BEGIN continuous reference and native regularization share time quadrature.
            if self._process_mode == "reference_objective":
                joint = state.joint_state
                batch, horizon = joint.position.shape[:2]
                counts = (torch.full((batch,), horizon, dtype=torch.long, device=joint.position.device)
                          if self._dense_counts is None or context is not None else self._dense_counts)
                reference = bound_reference if context is None else context["bound_reference"]
                # All rows of a native request use its one frozen grid; an
                # explicit candidate context supplies its own physical times.
                key = (horizon, id(self._dense_counts), id(context))
                if key != self._reference_grid_key:
                    times = _joint_state_sample_times(joint, counts) if context is None else context["times"]
                    self._reference_cost.bind(times[:1], counts[:1], reference)
                    self._reference_grid_key = key
                collection = super().compute_costs(state, cost_collection, None, **kwargs)
                weights = self._reference_cost.time_weights(batch)
                collection.values = [value*weights.reshape(*weights.shape, *([1]*(value.ndim-2)))
                                     for value in collection.values]
                collection.add(self._reference_cost.forward(state.tool_poses)[..., None], "continuous_reference")
                return collection
            # END exact event/witness costs do not change the ordinary time measure.
            active_scene = self._scene if context is None else context["scene"]
            active_map = progress_map if context is None else context["mapping"]
            active_clock = clock if context is None else {"duration": context["duration"]}
            # BEGIN both native grids use the actual state-generating dt.
            if not self._process_mode.startswith("seed_"):
                batch, horizon = state.joint_state.position.shape[:2]
                counts = (torch.full((batch,), horizon, dtype=torch.long, device=state.joint_state.position.device)
                          if self._dense_counts is None or context is not None else self._dense_counts)
                sample_times = (_joint_state_sample_times(state.joint_state, counts)
                                if context is None else context["times"])
            # END normalized progress remains separate from physical sample time.
            # END optimizer calls retain their original differentiable time grid.
            collection = super().compute_costs(state, cost_collection, goal, **kwargs)
            # END reference IK receives the same physical time as its optimizer.
            # BEGIN event IK minimizes violations of its continuous allowed set.
            # Native pose tracking is disabled by the event caller. The
            # ordinary reference query retains its original objective. Static
            # scene qualification is performed on the returned event states;
            # path history and dynamics belong to the completed trajectory.
            if self._process_mode.startswith("seed_"):
                if ik_goal_ids:
                    q = state.joint_state.position
                    tool = state.tool_poses
                    orientation = orientation_vjp.apply(q, tool.quaternion.detach(),
                        state.cuda_robot_model_state.tool_jacobians[..., 3:, :].detach())
                    p, r = problem.transform_tool_pose(
                        native_layout.apply(tool.position)[:, :, self._tool_index],
                        orientation[:, :, self._tool_index])
                    event_progress = getattr(problem, ik_goal_ids[0][0])[ik_goal_ids[0][1]].progress
                    progress = q.new_full(q.shape[:2], event_progress)
                    times = seed_parameter.to(q).expand(q.shape[:2])*clock["duration"]
                    terms = problem.residuals(p, r, progress, q, self._joint_names,
                        joint_velocity=torch.zeros_like(q), trajectory_time=times,
                        point_goal_ids=ik_goal_ids, direction_mode="point")
                    process = sum((value for name, value in terms.items()
                        if not name.startswith(("function_pose:", "function_coordinate:", "support:", "stop:"))),
                        torch.zeros_like(progress))
                    goals = problem.goal_residual(p, r, progress, goal_ids=ik_goal_ids,
                                                  normalized=self._process_mode == "seed_optimizer")
                    value = (process.square()+goals.square().sum(-1)
                             if self._process_mode == "seed_optimizer" else process+goals.sum(-1))
                    collection.add(value[..., None]*budget.constraint_weight, "event_allowed_set")
                return collection
            # END no nominal full-pose centre can replace the event declaration.
            q, dq = state.joint_state.position, state.joint_state.velocity
            batch, horizon, _ = q.shape
            tool = state.tool_poses
            quaternion = orientation_vjp.apply(q, tool.quaternion.detach(),
                                               state.cuda_robot_model_state.tool_jacobians[..., 3:, :].detach())
            position = native_layout.apply(tool.position)[:, :, self._tool_index]
            quaternion = quaternion[:, :, self._tool_index]
            # BEGIN local native-coordinate rates use differentiable physical J(q) dq.
            need_rates = any(direction.kind == "local_rate" for mechanism in problem.mechanisms
                             for direction in mechanism.directions)
            twist = (torch.einsum("...kd,...d->...k", native_layout.apply(
                state.cuda_robot_model_state.tool_jacobians)[:, :, self._tool_index], dq) if need_rates else None)
            # END Jacobian backward retains dJ/dq as well as the velocity derivative.
            spheres = native_layout.apply(state.robot_spheres)
            counts = (torch.full((batch,), horizon, dtype=torch.long, device=q.device)
                      if self._dense_counts is None or context is not None else self._dense_counts)
            u = ((torch.arange(horizon, device=q.device, dtype=q.dtype)[None, :]
                  / (counts[:, None]-1)).clamp(0, 1) if context is None else context["times"]/context["duration"])
            progress = (_torch_interpolate(self._map_u, self._map_s, u)
                        if context is None else context["progress"])
            # BEGIN no interpolation, FK or collision query for padded anchors.
            anchor_u = (self._anchor_u[:, :self._anchor_count].expand(batch, -1)
                        if context is None else context["anchor_u"])
            anchor_s = (self._anchor_s[:, :self._anchor_count].expand(batch, -1)
                        if context is None else context["anchor_s"])
            anchor_valid = (self._anchor_valid[:, :self._anchor_count] if context is None
                            else torch.ones_like(anchor_s, dtype=torch.bool))
            # BEGIN exact anchor time must not be reconstructed from rounded u.
            query_anchor_u = (self._anchor_query_u[:, :self._anchor_count].expand(batch, -1)
                              if context is None else None)
            anchor_times = ((query_anchor_u*active_clock["duration"]).to(q.dtype)
                            if context is None else context["anchor_times"])
            # END candidate context already sampled its original binary64 pairs.
            # END the admission capacity is unchanged; the work is request-sized.
            anchor_q, anchor_dq = (_hermite_at_parameters(q, dq, state.joint_state.dt, counts, query_anchor_u)
                                   if context is None else (context["anchor_q"], context["anchor_dq"]))
            # BEGIN retain the actual anchor state for native self collision.
            anchor_joints = JointState.from_position(anchor_q.contiguous(), joint_names=list(self._joint_names))
            anchor_state = self._anchor_fk.compute_kinematics(anchor_joints)
            anchor_quaternion = orientation_vjp.apply(anchor_q, anchor_state.tool_poses.quaternion.detach(),
                                                      anchor_state.tool_jacobians[..., 3:, :].detach())
            anchor_position = native_layout.apply(anchor_state.tool_poses.position)[:, :, self._tool_index]
            if need_rates:
                anchor_twist = torch.einsum("...kd,...d->...k", native_layout.apply(
                    anchor_state.tool_jacobians)[:, :, self._tool_index], anchor_dq)
                twist = torch.cat((twist, anchor_twist), 1)
            position = torch.cat((position, anchor_position), 1)
            quaternion = torch.cat((quaternion, anchor_quaternion[:, :, self._tool_index]), 1)
            spheres = torch.cat((spheres, native_layout.apply(anchor_state.robot_spheres)), 1)
            q, dq = torch.cat((q, anchor_q), 1), torch.cat((dq, anchor_dq), 1)
            progress = torch.cat((progress, anchor_s), 1)
            # End-point families consume a whole ordered trajectory and the
            # actual functional-event FK.  No nearest-grid event substitution.
            all_u = torch.cat((u, anchor_u), 1)
            order = torch.argsort(all_u, dim=1, stable=True)
            inverse = torch.argsort(order, dim=1)

            def ordered(value):
                index = order.reshape(*order.shape, *((1,)*(value.ndim-2))).expand_as(value)
                return value.gather(1, index)

            link_quaternion = ordered(quaternion)
            tool_linear, tool_angular = ((None, None) if twist is None else problem.transform_tool_velocity(
                ordered(twist)[..., :3], ordered(twist)[..., 3:], link_quaternion))
            position, quaternion = problem.transform_tool_pose(ordered(position), link_quaternion)
            spheres = problem.transform_robot_spheres(ordered(spheres))
            q, dq, progress = ordered(q), ordered(dq), ordered(progress)
            # BEGIN retain both real grids and exact anchor physical instants.
            trajectory_time = ordered(torch.cat((sample_times, anchor_times), 1))
            # END no double rounding manufactures another collision/event instant.
            # BEGIN every supported functional event has exactly one optimizer owner.
            process_terms = problem.residuals(position, quaternion, progress, q, self._joint_names,
                joint_velocity=dq, trajectory_time=trajectory_time, tool_linear_velocity=tool_linear,
                tool_angular_velocity=tool_angular)
            process_value = sum((value for name, value in process_terms.items()
                if self._process_mode != "trajectory_optimizer"
                or not name.startswith(("function_pose:", "function_coordinate:", "support:"))),
                torch.zeros_like(progress))
            # A support-only request still supplies a tensor. Its declared
            # hard rows belong to the same AL owner as pose/coordinate events.
            # END optimization owns all targets; metrics use delivery requirements.
            # BEGIN optimizer and success metrics consume the identical online scene.
            valid = torch.arange(horizon, device=q.device)[None, :] < counts[:, None]
            # BEGIN align validity with the time-sorted grid and exact anchors.
            # The trajectory tensors above contain both sample families in
            # ``order``. Hard-evidence reduction must use that same ordering;
            # padded grid entries do not become real trajectory states.
            active = ordered(torch.cat((valid, anchor_valid.bool().expand(batch, -1)), dim=1))
            # END this mask uses the existing sample counts and anchor validity.
            # BEGIN every optimizer scope evaluates its declared robot geometry.
            geometry_scope = optimizer_geometry_scope if self._process_mode == "trajectory_optimizer" else "full"
            values = [("skill_process", process_value)]
            # BEGIN candidate retention reads scalar evidence from this same scene query.
            scene_evidence_out = {} if self._process_mode in ("trajectory_metrics", "trajectory_optimizer") else None
            # END no extra FK, source query, or geometry history is created.
            # BEGIN omit columns with no real state in any batch row.
            # Keep the union over rows so every trajectory retains its ordered
            # exact release witnesses in the same rectangular scene query.
            # For unequal sample counts, a column needed by another row stays
            # present; the existing row-wise validity mask still owns reduction.
            scene_columns = active.any(0).nonzero().flatten()

            def scene_input(value):
                return value.index_select(1, scene_columns)

            scene_value = _scene_boundary_evaluate(active_scene, problem, active_map,
                scene_input(spheres), scene_input(position), scene_input(quaternion),
                scene_input(progress), scene_input(q), self._joint_names,
                trajectory_time=scene_input(trajectory_time), normalized_time=scene_input(ordered(all_u)),
                query_counts=self._query_counts,
                component_diagnostics=None if context is None else context["scene_components"],
                # BEGIN the caller selects the query path under one nonpenetration rule.
                query_role="optimization" if self._process_mode == "trajectory_optimizer" and context is None else "acceptance",
                scene_evidence_out=scene_evidence_out,
                optimizer_candidate_evidence=self._process_mode == "trajectory_optimizer",
                local_models=geometry_models if self._process_mode == "trajectory_optimizer" and context is None and geometry_scope == "full" else None,
                geometry_scope=geometry_scope)
            scene_indices = scene_columns[None].expand(batch, -1)
            scene_value = scene_value.new_zeros(progress.shape).scatter(1, scene_indices, scene_value)
            if scene_evidence_out is not None:
                for name, evidence in scene_evidence_out.items():
                    scene_evidence_out[name] = evidence.new_zeros(progress.shape).scatter(1, scene_indices, evidence)
            # END restore scalar layouts before the unchanged task/scene reduction.
            # END ordinary optimizer and proposal metric calls remain unchanged.
            # END original hard components and fixed local objectives are separate.
            values.append(("process_scene", scene_value))
            # END every present scene term retains its original complete computation.
            # BEGIN use the local scene's actual scalar and matching derivative.
            # Scene objectives include their own geometric scaling. Task
            # process violations retain the existing squared penalty.
            if self._process_mode == "trajectory_optimizer":
                values[0] = ("skill_process", process_value.square())
                # END the scene residual is not squared a second time.
                # BEGIN signed inequalities enter PHR directly, never squared twice.
                signed, _ = problem.functional_inequalities(
                    position, quaternion, progress, trajectory_time=trajectory_time)
                augmented = sum(functional_constraints.terms(signed).values(), q.new_zeros(batch))
                folded_augmented = torch.cat((q.new_zeros((batch, horizon-1, 1)),
                                               augmented[:, None, None]), dim=1)
                # BEGIN retain the original task zero set before objective substitution.
                augmented_cost = folded_augmented * budget.constraint_weight
                collection.add(augmented_cost, "functional_augmented_lagrangian")
                original_process = sum(process_terms.values(), torch.zeros_like(progress))
                original_process = torch.where(active, original_process, 0.).detach()
                feasible = torch.isfinite(original_process).all(-1) & (original_process <= 0).all(-1)
                for residual in signed.values():
                    feasible = feasible & torch.isfinite(residual.detach()) & (residual.detach() <= 0)
                # BEGIN candidate feasibility uses only evidence actually evaluated.
                # BEGIN preserve every original hard scene term.
                # Use where for inactive samples; multiplication by zero would
                # not suppress a nonfinite inactive value. Mode-side reduction
                # has already retained every active hard violation.
                hard_invalid = torch.where(active, scene_evidence_out["optimizer_hard_invalid"], 0.)
                feasible = feasible & (hard_invalid <= 0).all(-1)
                # BEGIN strict scene feasibility uses original unweighted hard residuals.
                raw_scene = torch.where(active, scene_evidence_out["optimizer_hard_value"], 0.)
                feasible = feasible & torch.isfinite(raw_scene).all(-1) & (raw_scene <= 0).all(-1)
                # END local objective values do not replace original scene feasibility.
                # END the full-scene solve still requires all original hard scene terms.
                self._optimizer_evidence = (augmented_cost, feasible,
                    {name: value.detach() for name, value in signed.items()}, None)
                # END the aggregate scene tensor identity is bound below, after folding.
                # END the core consumes this evidence before any other forward.
                # END original tolerance and first-arrival identity belong to the problem.
                # BEGIN task preferences follow the declared optimization phase.
                # Process relations remain active; quality work additionally
                # includes soft support and goal preferences. Hard rows are above.
                values.append(("skill_pose_equality_preference", problem.optimizer_preferences(
                    position, quaternion, progress, trajectory_time=trajectory_time,
                    tool_linear_velocity=tool_linear, tool_angular_velocity=tool_angular,
                    phase=phase)))
                # END a preference value cannot change task feasibility.
                # A position-tracking gripper has an exact physical command,
                # even inside its numerical acceptance band. Preserve a
                # gradient toward that command to avoid stopping on the
                # band's floating-point edge. Allowed pose/contact freedoms
                # receive no invented centering preference.
                values.append(("gripper_command_preference", active_scene.optimizer_preferences(
                    progress, q, self._joint_names)))
                # BEGIN optimize the actual zero-speed verification command.
                values.append(("verification_stop_preference", _verification_stop_preference(
                    progress, dq, problem.stop_progress, trajectory_time=trajectory_time)))
                # END optimize the actual zero-speed verification command.
            for name, value in values:
                value = value.gather(1, inverse)
                regular = value[:, :horizon] * valid
                # Keep CostCollection's native [B,H,*] shape.  Summing exact
                # anchor penalties into its final column retains their full
                # gradient and makes any anchor violation fail feasibility.
                folded = torch.cat((torch.zeros_like(regular[:, :-1]), (value[:, horizon:]*anchor_valid).sum(1, keepdim=True)), 1)
                # BEGIN retain the unchanged aggregate tensor as the query identity.
                weighted = (regular+folded).unsqueeze(-1) * budget.constraint_weight
                collection.add(weighted, name)
                if name == "process_scene" and self._process_mode == "trajectory_optimizer":
                    self._optimizer_evidence = (*self._optimizer_evidence[:-1], weighted)
                if name == "process_scene" and self._process_mode == "trajectory_metrics":
                    # BEGIN fold each same-query diagnostic category separately.
                    channels = []
                    # BEGIN all recovery categories retain identical sample accounting.
                    for category in ("collision", "scene_update", "coverage", "conditional_unknown"):
                        evidence = scene_evidence_out[category].gather(1, inverse)
                        regular_evidence = evidence[:, :horizon] * valid
                        folded_evidence = torch.cat((torch.zeros_like(regular_evidence[:, :-1]),
                            (evidence[:, horizon:]*anchor_valid).sum(1, keepdim=True)), 1)
                        channels.append(regular_evidence+folded_evidence)
                    self._scene_evidence = (weighted,
                        torch.stack(channels, dim=-1) * budget.constraint_weight)
                    # END channel order is collision, scene_update, forbidden and conditional coverage.
                # END recovery evidence never adds another cost or constraint.
            return collection

    @dataclass
    class ProcessCostManagerCfg(RobotCostManagerCfg):
        process_mode: str = ""

        def __post_init__(self):
            self.class_type = ProcessCostManager

        @staticmethod
        def create(data_dict, scene_collision_checker=None, device_cfg=None):
            base = RobotCostManagerCfg.create(data_dict, scene_collision_checker, device_cfg)
            values = {item.name: getattr(base, item.name) for item in fields(base) if item.name != "class_type"}
            return ProcessCostManagerCfg(**values, process_mode=data_dict.get("process_mode", ""))

    def bind_request(new_problem, new_scene, new_progress, new_anchors, new_budget, new_constraints,
                     new_bound_reference):
        nonlocal problem, scene, progress_map, anchors, budget, functional_constraints, collision_witnesses, ik_goal_ids, geometry_models, optimizer_geometry_scope
        nonlocal bound_reference
        problem, scene, progress_map, anchors, budget = new_problem, new_scene, new_progress, new_anchors, new_budget
        functional_constraints = new_constraints
        bound_reference = new_bound_reference
        # BEGIN a resident backend must never inherit another request's collision times.
        collision_witnesses = np.empty(0, dtype=float)
        ik_goal_ids = ()
        geometry_models = LocalGeometryModels()
        optimizer_geometry_scope = "full"
        # END semantic anchors are rebuilt from the newly bound request.
        for manager in managers:
            manager.bind_request_data()
    ProcessCostManagerCfg.bind_request = staticmethod(bind_request)
    # BEGIN one scope is shared by the initial and ordinary gradient rollouts.
    def set_geometry_scope(scope):
        """Bind the actual geometry computed by all optimizer rollouts.

        Robot self derivatives use the same query witnesses directly. Full
        observed-field optimization also captures its block-local models.
        A change invalidates same-forward evidence before reinitialization.
        """
        nonlocal optimizer_geometry_scope
        if scope not in {"robot", "full"}:
            raise ValueError("Optimizer geometry scope must be robot or full")
        optimizer_geometry_scope = scope
        for manager in managers:
            manager._optimizer_evidence = None
    ProcessCostManagerCfg.set_geometry_scope = staticmethod(set_geometry_scope)
    # END callers reinitialize the optimizer whenever its objective scope changes.
    # BEGIN event identities bind one point query without copying task data.
    def bind_ik_goals(goal_ids):
        nonlocal ik_goal_ids
        selected = tuple(tuple(identity) for identity in goal_ids)
        progress_values = {getattr(problem, name)[index].progress for name, index in selected}
        if len(progress_values) > 1:
            raise ValueError("One event IK query requires a single exact declared progress")
        ik_goal_ids = selected
    ProcessCostManagerCfg.bind_ik_goals = staticmethod(bind_ik_goals)
    # END clearing this context restores ordinary reference IK semantics.
    # BEGIN add bounded exact times only between completed eager native solves.
    def add_collision_witnesses(parameters):
        nonlocal collision_witnesses
        parameters = np.asarray(parameters, dtype=np.float32).astype(float).reshape(-1)
        if not np.isfinite(parameters).all() or np.any((parameters < 0) | (parameters > 1)):
            raise ValueError("Collision witnesses require finite normalized physical times")
        known = set(query_anchor_parameters()[:, 0].astype(np.float32).astype(float))
        extra = np.asarray(sorted(set(parameters)-known), dtype=float)
        if len(query_anchor_parameters())+len(extra) > budget.max_task_anchors:
            raise MotionPlanningError("capacity_exceeded", "Collision witnesses exceed the resident anchor capacity")
        if len(extra):
            collision_witnesses = np.sort(np.r_[collision_witnesses, extra])
            # Each manager owns independent FK/collision scratch storage and
            # resizes it on its next eager evaluation. AL parameters are keyed
            # by functional event, not query count: preserve the same object.
            for manager in managers:
                manager.bind_request_data()
        return extra
    ProcessCostManagerCfg.add_collision_witnesses = staticmethod(add_collision_witnesses)
    ProcessCostManagerCfg.query_anchor_parameters = staticmethod(query_anchor_parameters)
    # END request rebinding, rather than another candidate, clears these times.
    # BEGIN bounded aggregate evidence for real workload/performance reports.
    ProcessCostManagerCfg.query_diagnostics = staticmethod(
        lambda: [deepcopy(manager._query_counts) for manager in managers if manager._process_mode])
    # END no per-state geometry or query trace is retained in the optimization loop.
    return ProcessCostManagerCfg


# BEGIN exact native candidate extraction before independent verification.
def _trajectory_candidate(result, index):
    """Select one original seed through V2's own complete result gather.

    V2 has no arbitrary-index result accessor. Its top-k gather selects from
    ``seed_rank`` for actions, flags and valid lengths. JointState's native
    indexer separately handles the full joint states and their active-joint
    controls; these need not have the same DOF when some joints are locked.
    Never flatten seeds into one apparent temporal trajectory.
    """
    if result.batch_size != 1 or not 0 <= index < result.num_seeds:
        raise ValueError("Candidate extraction requires one batch and an existing seed index")
    # BEGIN even a single candidate must own buffers before formal metrics reuse them.
    ordered = copy(result)
    ordered.seed_rank = result.seed_rank.new_tensor(
        [[index, *(other for other in range(result.num_seeds) if other != index)]])
    # Native postprocessing augments state positions with locked joints but
    # intentionally leaves their action knots in active-joint coordinates.
    # Its result gather assumes both have position's DOF. The JointState
    # indexer instead slices each tensor on the seed axis without that false
    # reshape, preserving both coordinate layouts and every timing field.
    ordered.js_solution = ordered.interpolated_trajectory = None
    # BEGIN top-k discards metrics after cloning; avoid copying every seed first.
    ordered.metrics = ordered.interpolated_metrics = None
    # END selected dense constraint evidence is copied explicitly below.
    # Native top-k returns its input when all available seeds are requested.
    # Clone the selected result in both layouts to give it independent tensors.
    candidate = ordered.get_topk_seeds(1).clone()
    for name in ("js_solution", "interpolated_trajectory"):
        state = getattr(result, name)
        single = None if state is None else state[:, index:index+1].clone()
        if state is not None and state.dt is not None:
            # JointState's general indexer treats a leading singleton dt as
            # shared timing. Here dt is explicitly [batch,seed], so select
            # its seed column too, even though batch is exactly one.
            single.dt = state.dt[:, index:index+1].clone()
        setattr(candidate, name, single)
    # BEGIN native top-k drops metrics; retain both selected collision queries.
    # RolloutMetrics.__getitem__ is explicitly unimplemented in V2. Its
    # costs use a flattened [batch*seed, horizon, component] layout, while
    # the joint states above retain separate batch/seed axes. Copy only the
    # selected constraint tensors needed for collision routing and witnesses;
    # formal acceptance samples the saved curve; these values retain the
    # original native optimization diagnostics and collision witnesses.
    from curobo._src.rollout.metrics import CostCollection, CostsAndConstraints, RolloutMetrics
    for name in ("metrics", "interpolated_metrics"):
        metrics = getattr(result, name)
        if metrics is None:
            continue
        constraints = metrics.costs_and_constraints.constraints
        selected_metrics = RolloutMetrics(costs_and_constraints=CostsAndConstraints(
            constraints=CostCollection(names=list(constraints.names),
                values=[value[index:index+1].detach().clone() for value in constraints.values])),
            debug=None if metrics.debug is None else metrics.debug[index:index+1].detach().clone())
        setattr(candidate, name, selected_metrics)
    # END a different seed's dense times can never enter this continuation.
    return candidate
    # END single- and multiple-seed extraction have the same tensor ownership.
# END exact native candidate extraction before independent verification.


# BEGIN same-query evidence has one strict four-channel layout for all consumers.
def _scene_metric_channel(metrics, channel):
    """Read collision (0), support (1), forbidden (2), or conditional unknown (3)."""
    constraints = metrics.costs_and_constraints.constraints
    scene = constraints.values[constraints.names.index("process_scene")]
    if metrics.debug is None or metrics.debug.shape != (*scene.shape[:-1], 4):
        raise RuntimeError("Native scene metrics lack aligned collision/scene-update/coverage/conditional-unknown evidence")
    return metrics.debug[..., channel:channel+1]


def _scene_update_metric_residuals(metrics, seed_index=None):
    """Report support gaps for the evaluated candidate, never request infeasibility."""
    if "process_scene" not in metrics.costs_and_constraints.constraints.names:
        return {}
    evidence = _scene_metric_channel(metrics, 1)
    if seed_index is not None:
        evidence = evidence[seed_index]
    # BEGIN a finite zero cannot hide an invalid support result from another sample.
    flat = evidence.reshape(-1)
    index = flat.masked_fill(~flat.isfinite(), float("inf")).argmax()
    return {"scene_update": float(flat[index].item())}
    # END retain the first non-finite witness, otherwise the ordinary maximum.
# END these diagnostics cannot authorize execution or terminate candidate search.


# BEGIN geometric coverage violations retain a distinct search eligibility signal.
def _coverage_metric_residuals(metrics, seed_index=None):
    """Report forbidden-space violations, not collision or missing map data.

    A different path can stay within the same observed/explicitly permitted
    space. Reporting this category grants no permission to enter unknown space.
    """
    if "process_scene" not in metrics.costs_and_constraints.constraints.names:
        return {}
    evidence = _scene_metric_channel(metrics, 2)
    if seed_index is not None:
        evidence = evidence[seed_index]
    flat = evidence.reshape(-1)
    index = flat.masked_fill(~flat.isfinite(), float("inf")).argmax()
    return {"unknown_forbidden": float(flat[index].item())}
# END the original aggregate remains the sole hard scene constraint.


# BEGIN preserve policy-permitted uncertainty without adding a hard constraint.
def _conditional_unknown_metric_residuals(metrics, seed_index=None):
    """Retain the same query's unknown evidence even for native-success curves."""
    if "process_scene" not in metrics.costs_and_constraints.constraints.names:
        return {}
    evidence = _scene_metric_channel(metrics, 3)
    if seed_index is not None:
        evidence = evidence[seed_index]
    flat = evidence.reshape(-1)
    index = flat.masked_fill(~flat.isfinite(), float("inf")).argmax()
    return {"conditional_unknown": float(flat[index].item())}
# END this diagnostic channel contributes no cost and performs no extra query.


# BEGIN classify only a completed candidate whose sole hard failure is coverage.
def _unknown_policy_failure(report, policy):
    """Choose an explicit policy outcome without declaring all routes infeasible.

    Named scene components establish what caused the aggregate scene residual.
    Missing evidence, nonfinite results, or any other failed hard requirement
    preserve the ordinary failure/recovery path.
    """
    action = policy["outside_regions"]
    if action == "allow_attempt":
        return None
    components, residuals = report.get("scene_components", {}), report.get("residuals", {})
    if not components or not residuals:
        return None
    coverage_failed = False
    for name, entry in components.items():
        value = entry["maximum"]
        if not np.isfinite(value):
            return None
        if name.startswith("conditional/") or value <= 0:
            continue
        if not name.startswith("unknown/forbidden/"):
            return None
        coverage_failed = True
    scene_failed = False
    for name, value in residuals.items():
        if not np.isfinite(value):
            return None
        if value <= 0:
            continue
        if name != "scene" and not name.endswith(":process_scene"):
            return None
        scene_failed = True
    if not coverage_failed or not scene_failed:
        return None
    return "needs_observation" if action == "require_observation" else "unknown_space_forbidden"
# END the scope of this verdict is the evaluated state or selected curve only.


# BEGIN collision recovery consumes typed evidence, never aggregate name matching.
def _collision_metric_residuals(metrics, seed_index=None):
    """Summarize collision evidence from the same observed-scene query.

    The other diagnostic channels retain missing support and coverage as
    distinct results. The aggregate process_scene value is not reclassified.
    """
    constraints = metrics.costs_and_constraints.constraints
    def maximum(value):
        return float((value if seed_index is None else value[seed_index]).max().item())
    residuals = {}
    if "process_scene" in constraints.names:
        residuals["scene_collision"] = maximum(_scene_metric_channel(metrics, 0))
    return residuals
# END these summaries do not contribute to acceptance or candidate severity.


# BEGIN promote existing selected dense collisions without another geometry pass.
def _dense_collision_witnesses(result, *, regular_horizon, anchor_parameters, capacity):
    """Pick peaks only in intervals without an existing collision witness.

    Native dense metrics store real samples in a valid prefix. The final
    allocated column also contains folded anchor penalties, so it is never
    interpreted as a new physical-time observation. Work already performed
    for the selected candidate is the only source of additional queries.
    """
    import torch
    if result.batch_size != 1 or result.num_seeds != 1:
        raise ValueError("Collision witnesses require exactly one selected native candidate")
    if capacity < 0:
        raise ValueError("Collision witness capacity cannot be negative")
    report = {"parameters": [], "peak_count": 0, "pending_parameters": [],
              "source": "selected_native_dense_valid_prefix"}
    if result.interpolated_metrics is None:
        return report
    count = int(result.interpolated_last_tstep.reshape(-1)[0].item())
    like = result.interpolated_trajectory.position
    horizon = like.shape[-2]
    if not 2 <= count <= horizon or regular_horizon < 2:
        raise ValueError("Collision witness sampling requires valid native trajectory lengths")
    # Use exactly the native float32 divisions, without epsilon-based merging.
    dense_u = (torch.arange(count, device=like.device, dtype=like.dtype)/(count-1)).cpu().numpy()
    regular_u = (torch.arange(regular_horizon, device=like.device, dtype=like.dtype)
                 /(regular_horizon-1)).cpu().numpy()
    known = set(regular_u.astype(float)) | set(np.asarray(anchor_parameters, dtype=dense_u.dtype).astype(float))
    eligible = np.array([float(u) not in known and index != horizon-1 for index, u in enumerate(dense_u)])
    constraints = result.interpolated_metrics.costs_and_constraints.constraints
    # BEGIN validate collision evidence before interpreting any dense scene time.
    _collision_metric_residuals(result.interpolated_metrics)
    measurements = [(name, tensor) for name, tensor in zip(constraints.names, constraints.values)
                    if name == "self_collision"]
    if "process_scene" in constraints.names:
        measurements.insert(0, ("scene_collision", _scene_metric_channel(result.interpolated_metrics, 0)))
    # END anchor-only self constraints still route recovery but supply no time.
    # BEGIN ordinary positive evidence distinguishes missed from known collisions.
    # Both grids belong to this selected seed. A sampled positive interval
    # already represented by the same ordinary collision channel provides
    # no evidence that its collision was missed by the ordinary queries.
    ordinary = getattr(result, "metrics", None)
    if ordinary is None:
        raise ValueError("Collision recovery requires selected ordinary metric evidence")
    ordinary_constraints = ordinary.costs_and_constraints.constraints
    ordinary_measurements = dict(zip(ordinary_constraints.names, ordinary_constraints.values))
    if "process_scene" in ordinary_measurements:
        ordinary_measurements["scene_collision"] = _scene_metric_channel(ordinary, 0)
    positive_parameters = {}
    for name, _ in measurements:
        tensor = ordinary_measurements.get(name)
        if tensor is None or tensor.shape[:2] != (1, regular_horizon):
            raise ValueError("Selected ordinary collision metrics changed their sample layout")
        values = tensor[0].detach().reshape(regular_horizon, -1).amax(-1).cpu().numpy()
        positive = np.isfinite(values) & (values > 0)
        # The final ordinary column can fold exact-anchor penalties. Its
        # positive aggregate cannot identify a collision at physical u == 1.
        positive[-1] = False
        positive_parameters[name] = regular_u[positive]
    # END missing channels are integration errors, never zero-value fallbacks.
    component_peaks = []
    for name, tensor in measurements:
        if tensor.shape[:2] != (1, horizon):
            raise ValueError("Selected native collision metrics changed their sample layout")
        values = tensor[0, :count].detach().reshape(count, -1).amax(-1).cpu().numpy()
        positive = np.isfinite(values) & (values > 0)
        # BEGIN folded evidence cannot extend a physical positive interval.
        # When padding exists, the valid prefix's last sample is still real;
        # only the allocated final column can contain the folded anchor sum.
        if count == horizon:
            positive[-1] = False
        # END remove that column before segmentation, not just peak selection.
        edges = np.flatnonzero(np.diff(np.r_[False, positive, False]))
        peaks = []
        for left, right in edges.reshape(-1, 2):
            # BEGIN classify each interval independently, including mixed cases.
            # Real parameter comparisons also support non-nested grids. No
            # epsilon merges distinct times, and an unrelated earlier positive
            # cannot suppress a later interval or another collision channel.
            detected = positive_parameters[name]
            if np.any((detected >= dense_u[left]) & (detected <= dense_u[right-1])):
                continue
            # END this scheduling decision changes no hard acceptance evidence.
            candidates = np.flatnonzero(eligible[left:right])+left
            if len(candidates):
                # Existing exact queries cannot be added a second time.
                index = candidates[np.argmax(values[candidates])]
                peaks.append((float(values[index]), float(dense_u[index])))
        component_peaks.append([u for _, u in sorted(peaks, key=lambda row: (-row[0], row[1]))])
    # Alternate components without comparing native self-collision units to
    # scene units. Capacity is the existing anchor allowance, never a new one.
    candidates = []
    for index in range(max((len(peaks) for peaks in component_peaks), default=0)):
        for peaks in component_peaks:
            if index < len(peaks) and peaks[index] not in candidates:
                candidates.append(peaks[index])
    report.update(parameters=sorted(candidates[:capacity]), peak_count=len(candidates),
                  pending_parameters=sorted(candidates[capacity:]), valid_count=count)
    return report
# END these extra optimizer queries never certify trajectory feasibility.


# BEGIN bind the two request-private native rollout configurations together.
def _bind_optimizer_constraint_config(trajectory_optimizer, trajectory_metrics):
    """Share ordinary scoring and native joint constraints on their existing paths.

    Both arguments are newly loaded request configurations. Metric ordinary
    costs follow the optimizer objective. Optimizer constraints reuse the
    native metric joint-limit configuration, including its original weights
    and zero activation distance, so same-query retention sees those limits.
    Deep copies keep the subsequent native factories' configuration private.
    """
    trajectory_metrics["rollout"]["cost_cfg"] = deepcopy(trajectory_optimizer["rollout"]["cost_cfg"])
    trajectory_optimizer["rollout"]["constraint_cfg"]["cspace_cfg"] = deepcopy(
        trajectory_metrics["rollout"]["constraint_cfg"]["cspace_cfg"])
# END no second joint-residual implementation is needed.


# BEGIN physical reference entrance uses the fixed official robot and feedback.
def _reference_entrance(problem, scene, robot_validator):
    """Return calibrated TCP pose and world twist from named measured q/dq.

    Pinocchio's forward velocity equals its geometric Jacobian times dq.
    LOCAL_WORLD_ALIGNED expresses velocity at the link origin in robot-base
    axes. Base calibration rotates both vectors into the reference frame;
    the TCP lever arm adds omega cross offset to the linear velocity.
    Independently controlled fingers are already verified not to be TCP
    ancestors. Their absent velocity columns therefore have no TCP effect.
    The official fixed robot model is reused across tasks; this computation
    reads neither task-object geometry nor simulator environment state.
    """
    import pinocchio as pin
    checker = robot_validator.checker
    positions, _ = robot_validator._planning_positions(problem.start_q, problem.joint_names,
        time_s=0., gripper=scene.gripper)
    q, dq = pin.neutral(checker.model), np.zeros(checker.model.nv)
    measured_rates = dict(zip(problem.joint_names, problem.start_dq))
    for name, position in zip(robot_validator.joint_names, positions):
        joint = checker.model.joints[checker.model.getJointId(name)]
        q[joint.idx_q] = position
        if name in measured_rates:
            dq[joint.idx_v] = measured_rates[name]
    data = checker.model.createData()
    pin.forwardKinematics(checker.model, data, q, dq)
    pin.updateFramePlacements(checker.model, data)
    frame = checker._frame_id(robot_validator.tool_frame)
    velocity = pin.getFrameVelocity(checker.model, data, frame, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
    world_link = robot_validator.T_reference_base @ data.oMf[frame].homogeneous
    world_tool = world_link @ robot_validator.T_frame_tcp
    rotation = robot_validator.T_reference_base[:3, :3]
    angular = rotation @ velocity.angular
    offset = world_link[:3, :3] @ robot_validator.T_frame_tcp[:3, 3]
    linear = rotation @ velocity.linear + np.cross(angular, offset)
    return world_tool, np.r_[linear, angular]
# END the returned velocity is measured robot kinematics, not a target error.


class _CuroboPlanner:
    """Resident native solvers with request-local online constraint inputs."""

    # BEGIN every planner binds one explicit source geometry and pair policy.
    def __init__(self, problem, scene, robot, budget, device, *, robot_validator, continuation=None, goal_identities=None):
        self.ik = self.trajopt = None
        self.check_request = lambda: None
        # BEGIN keep full float32 geometry through the complete native request.
        # TF32 truncates even an identity matrix product, so its coordinate
        # quantization can exceed the physical collision tolerance. Precision
        # must cover backward as well as forward; a cost-only context is not
        # sufficient. Save caller settings before cuRobo's import side effects.
        import torch
        self._precision_torch = torch
        self._caller_precision = (torch.backends.cuda.matmul.allow_tf32,
                                  torch.backends.cudnn.allow_tf32)
        torch.backends.cuda.matmul.allow_tf32 = False
        # END close() restores both settings on success and initialization error.
        try:
            self._initialize(problem, scene, robot, budget, device,
                             robot_validator=robot_validator, continuation=continuation, goal_identities=goal_identities)
        except BaseException:
            # A later native configuration failure must release an IK solver
            # already created in this process before propagating the error.
            self.close()
            raise

    def _initialize(self, problem, scene, robot, budget, device, *, robot_validator, continuation=None, goal_identities=None):
        # Validate grounded evidence and bound independent scene workspaces
        # before importing native GPU solvers or creating any of their buffers.
        self.robot_validator = robot_validator
        # BEGIN native planning and measured execution share calibrated FK identity.
        robot_validator.validate_planning_binding(problem, scene.gripper)
        # END this checks named inputs only and performs no source geometry query.
        # BEGIN continuation keeps original goal identities after local filtering.
        scene.validate_problem(problem, goal_identities=goal_identities)
        # END finite arrival binding never treats an original ID as a local index.
        # END binding precedes native allocation and applies to direct planner construction.
        budget, initial_progress = (_initial_timing(problem, budget) if continuation is None
                                    else _continuation_timing(problem, budget, continuation))
        anchors = np.unique(np.concatenate((problem.anchors, np.asarray(scene.required_progress_anchors, float))))
        anchors = anchors[(anchors >= problem.start_progress) & (anchors <= problem.end_progress)]
        # BEGIN preserve all required source points within the total query budget.
        if (len(anchors) > budget.max_validation_points or not np.isfinite(anchors).all()
                or np.any(anchors < 0) or np.any(anchors > problem.end_progress)):
            raise ValueError("All required source knots and scene anchors must fit the finite planning horizon and budget")
        if problem.reference_frame != scene.reference_frame:
            raise ValueError("Problem and planning scene must use the same explicitly calibrated reference frame")
        self.progress_map = initial_progress
        # BEGIN retain the original path clock for phase-aligned joint seeds.
        self.source_progress_map = initial_progress
        self.reference_time_map = None
        if continuation is None:
            self.progress_map, self.reference_time_map = _progress_for_cubic_boundaries(initial_progress,
                budget.knots, stop_progress=problem.stop_progress, anchors=anchors)
        initial_clock = _fixed_spline_grid(intervals=budget.knots+4,
            native_dt=budget.maximum_trajectory_dt, interpolation_steps=budget.interpolation_steps,
            interpolation_dt=budget.interpolation_dt)
        if continuation is None:
            before_tick = self.progress_map
            self.progress_map = _bind_event_clock(problem, self.progress_map, initial_clock["duration"],
                None if scene.gripper is None else scene.gripper.control_timestep_s)
            self.reference_time_map = _ProgressMap(_phase_aligned_parameters(
                before_tick, self.progress_map, self.reference_time_map.u), self.reference_time_map.s)
        # END native boundaries and device ticks move the reference together.
        # BEGIN freeze synchronized hand inputs before any rollout owns a clone.
        # IK, trajectory managers and independent verification each deep-copy
        # the scene. Binding only the parent after their construction leaves
        # those copies without a command/nominal polynomial. The admitted grid
        # already defines the exact native duration; verify it against the
        # constructed solver below without regenerating these commands.
        hand = scene.gripper
        if hand is not None:
            hand.bind_clock(initial_clock["duration"], self.progress_map)
        self.finger_trajectory = None if hand is None else hand.finger_trajectory
        # END every private scene receives the same already frozen hand clock.
        self.anchor_capacity = _anchor_capacity(self.progress_map, anchors, budget)
        robot_dict, self.robot_metadata = _bounded_robot_configuration(robot, problem, scene, budget)
        # BEGIN independent and differentiable FK share the same pure kinematic tree.
        # Compare the parsed XML with the validator's consumed snapshot, not
        # with its mutable source file. Runtime native tree overrides have no
        # counterpart in the bound source model and cannot change this tree.
        from curobo.content import get_assets_path
        from pathlib import Path
        import xml.etree.ElementTree as ET
        kinematics = robot_dict["robot_cfg"]["kinematics"]
        native_urdf = Path(kinematics.get("external_asset_path") or get_assets_path()) / kinematics["urdf_path"]
        checker = robot_validator.checker
        if (kinematics["base_link"] != robot_validator.checker.base_frame
                or kinematics.get("extra_links") or kinematics.get("add_object_link", False)
                or ET.tostring(ET.parse(native_urdf).getroot(), encoding="unicode")
                != checker.urdf_xml):
            raise ValueError("Native planning requires the consumed source URDF/base without tree overrides")
        locks = kinematics.get("lock_joints") or {}
        independent_hand = hand is not None and hand.controller in {"timed_reference", "binary"}
        allowed_locks = dict(zip(hand.joint_names, hand.current_positions)) if independent_hand else {}
        if (not set(locks) <= set(allowed_locks) or any(
                value != allowed_locks[name] and value != float(np.float32(allowed_locks[name]))
                for name, value in locks.items())):
            raise ValueError("Native joint locks may only retain the measured independent finger coordinates")
        if independent_hand:
            joint = int(checker.model.frames[checker._frame_id(robot_validator.tool_frame)].parentJoint)
            while joint:
                if str(checker.model.names[joint]) in hand.joint_names:
                    raise ValueError("An independently controlled finger cannot be an ancestor of the native TCP")
                joint = int(checker.model.parents[joint])
        # END source identity is established before allocating GPU solvers.
        # BEGIN bind the tool reference after the actual event/hold clock.
        from .timed_reference import BoundToolReference
        entrance_pose, entrance_twist = _reference_entrance(problem, scene, robot_validator)
        self.bound_reference = BoundToolReference(problem, self.progress_map.u, self.progress_map.s,
            initial_clock["duration"], entrance_pose, entrance_twist)
        # END the same physical-time reference serves IK and ordinary costs.
        self.validation_storage = {"encoded_record_bytes_upper": 0}
        workspace = _planning_workspace_bytes(scene, self.robot_metadata, budget, self.anchor_capacity)
        _require_planning_host_memory(workspace.cpu_live_bytes,
            encoded_record_bytes_upper=self.validation_storage["encoded_record_bytes_upper"],
            max_host_rss_bytes=budget.max_host_rss_bytes)
        # END preserve all required source points within the total query budget.
        import torch
        from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
        from curobo.trajectory_optimizer import TrajectoryOptimizer, TrajectoryOptimizerCfg
        from curobo.kinematics import Kinematics
        from curobo.types import JointState, GoalToolPose
        from curobo._src.types.device_cfg import DeviceCfg
        from curobo._src.util.config_io import load_yaml, join_path
        from curobo.content import get_task_configs_path

        # BEGIN undo the native seed-IK module's global TF32 import assignment.
        # Apply after native imports and before any native solver/FK buffers or
        # geometry calculations. cuDNN is not used here; preserve its caller
        # setting rather than leaking the dependency's unrelated assignment.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = self._caller_precision[1]
        # END the single-request lifecycle owns forward and backward precision.
        self.problem, self.scene, self.budget = problem, scene, budget
        self.torch, self.JointState, self.GoalToolPose = torch, JointState, GoalToolPose
        self.device_cfg = DeviceCfg(device)
        # BEGIN include request-sized anchor FK in native allocation admission.
        workspace = _require_planning_memory(torch, self.device_cfg.device, workspace,
                                             max_device_used_bytes=budget.max_device_used_bytes)
        self.scene_workspace_bytes = workspace.total_bytes
        self.scene_cpu_workspace_bytes, self.scene_gpu_workspace_bytes = workspace.cpu_bytes, workspace.gpu_bytes
        # BEGIN export needs only new CPU scratch; shared geometry is in RSS.
        self.scene_cpu_live_bytes = workspace.cpu_live_bytes
        # END subsequent record admission does not re-charge resident maps.
        self.anchor_fk_workspace_bytes = workspace.anchor_fk_bytes
        self._trajectory_optimizer_graph_admitted = False
        # END include request-sized anchor FK in native allocation admission.
        self.ik = self.trajopt = None
        self.anchors = anchors
        self.robot_dict = robot_dict
        self.seed_parameter = self.device_cfg.to_device([[0.]])
        # BEGIN native callbacks inherit the admitted clock during construction.
        self.formal_clock = dict(initial_clock)
        # END construction cannot observe a missing or independently chosen clock.
        # BEGIN each native solve owns independent per-seed block continuation.
        from .functional_augmented_lagrangian import FunctionalAugmentedLagrangian
        self.functional_constraints = FunctionalAugmentedLagrangian(problem.functional_constraint_keys)
        # END the core binds/reset parameters before evaluating an independent seed layout.
        self._manager_type = _process_manager_type(
            problem, scene, self.progress_map, self.anchors, budget,
            bound_reference=self.bound_reference,
            clock=self.formal_clock, seed_parameter=self.seed_parameter, robot_dict=self.robot_dict,
            functional_constraints=self.functional_constraints)
        self.phase = "feasibility"
        self.goal_ik_candidates = {}
        self.last_optimizer_diagnostics = []
        self.delivery_candidate = None
        self.initialization_candidate = None
        self.reference_initialization = {"status": "not_started", "geometry_scope": "robot", "task_scene_constraints": "not_evaluated"}
        self.reference_ik_queries, self.reference_ik_chains, self.reference_seed_fits = [], [], []
        self.initial_seed_strategies = []

        def configuration(name, mode):
            data = deepcopy(load_yaml(join_path(get_task_configs_path(), name)))
            constraint = data["rollout"].setdefault("constraint_cfg", {})
            # All environment geometry is owned by PlanningScene in reference
            # coordinates. Spheres remain an optimizer gradient only.
            constraint.pop("scene_collision_cfg", None)
            constraint.pop("self_collision_cfg", None)
            constraint["process_mode"] = mode
            return data

        ik_optimizer = configuration("ik/lbfgs_ik.yml", "seed_optimizer")
        # BEGIN consistent full-axis IK energy and native joint derivatives.
        # This installed Lie kernel uses a half-angle vector. Multiplying its
        # rotation weight by four preserves the original quadratic energy.
        # Metrics below retain axis-angle so convergence still uses true rad.
        pose_cost = ik_optimizer["rollout"]["cost_cfg"]["tool_pose_cfg"]
        self.reference_ik_pose_weights = tuple(pose_cost["weight"])
        pose_cost.update(use_lie_group=True, _terminal_pose_axes_weight_factor=[1.]*6,
                         _non_terminal_pose_axes_weight_factor=[1.]*6)
        pose_cost["weight"][1] *= 4.
        # END only the optimizer cost changes; task/metric tolerances are unchanged.
        trajectory_optimizer = configuration("trajopt/lbfgs_bspline_trajopt.yml", "trajectory_optimizer")
        # BEGIN both optimization and explicit scoring use the same ordinary costs.
        trajectory_optimizer["rollout"]["cost_cfg"]["process_mode"] = "reference_objective"
        trajectory_metrics = configuration("metrics_base.yml", "trajectory_metrics")
        _bind_optimizer_constraint_config(trajectory_optimizer, trajectory_metrics)
        # END native optimizer retention and final metrics share physical limits.
        for name, config, iterations in (("IK", ik_optimizer, budget.ik_iterations),
                                         ("trajectory", trajectory_optimizer, budget.trajectory_iterations)):
            inner_iterations = config["optimizer"]["inner_iters"]
            if iterations % inner_iterations:
                raise ValueError(f"{name} iterations must be a multiple of native inner_iters={inner_iterations}")
        ik_cfg = InverseKinematicsCfg.create(
            robot=deepcopy(robot_dict), optimizer_configs=[ik_optimizer],
            metrics_rollout=configuration("metrics_base.yml", "seed_metrics"),
            device_cfg=self.device_cfg, self_collision_check=False, num_seeds=budget.ik_seeds, seed_solver_num_seeds=budget.ik_seeds,
            # BEGIN one pose buffer serves reference IK and allowed-set event IK.
            max_batch_size=1, max_goalset=1,
            use_cuda_graph=False,
            # END the native seed budget counts alternative joint configurations.
            position_tolerance=problem.numerical.position_m,
            orientation_tolerance=problem.numerical.rotation_rad,
            cost_manager_config_instance_type=self._manager_type,
            override_optimizer_num_iters={"lbfgs": budget.ik_iterations}, random_seed=budget.random_seed)
        # BEGIN configure the checked physical domain through the native API.
        # The factory does not expose this dataclass field. Set it before
        # constructing the solver so the optional LM seed stage does not use
        # its separate inward-offset position interval. Halton seeds and native L-BFGS still
        # supply the declared branches through the domain-restricted core.
        ik_cfg.use_lm_seed = False
        # END no independent optimizer may precede the checked native core.
        # BEGIN evaluate the declared alternative IK branches before returning.
        # Native early exit only requires one successful seed per batch.
        # Even one requested trajectory can benefit from choosing among the
        # several IK postures already admitted by its separate seed budget.
        if budget.ik_seeds > 1:
            ik_cfg.exit_early = False
        # END evaluate the declared alternative IK branches before returning.
        self.ik = InverseKinematics(ik_cfg)
        self.joint_names = tuple(self.ik.joint_names)
        if (set(self.joint_names) != set(self.robot_metadata["active_joint_names"])
                or self.ik.kinematics.total_spheres != self.robot_metadata["spheres"]):
            raise RuntimeError("Native robot layout differs from its checked metadata")
        columns = [problem.joint_names.index(name) for name in self.joint_names]
        self.start = JointState.from_numpy(list(self.joint_names), problem.start_q[columns][None],
                                          velocity=problem.start_dq[columns][None], device_cfg=self.device_cfg)
        self._bind_online_robot()
        self._verification_fk = Kinematics(self.ik.kinematics.config, compute_jacobian=True)
        self._orientation_vjp = _orientation_vjp_type()
        self._native_layout = _native_gradient_layout_type()
        self._verification_scene = scene.clone_for_rollout()
        transition = deepcopy(load_yaml(join_path(get_task_configs_path(), "trajopt/transition_bspline_trajopt.yml")))
        # BEGIN bind the same original duration with two released entrance controls.
        transition["transition_model_cfg"].update(n_knots=budget.knots+2,
            interpolation_steps=budget.interpolation_steps,
            dt_traj_params={"base_dt": budget.maximum_trajectory_dt,
                            "max_dt": budget.maximum_trajectory_dt, "base_ratio": 1.})
        # END the owned config uses M+2 intervals; no native horizon is lengthened.
        self._trajectory_configuration = {
            "robot": deepcopy(robot_dict), "self_collision_check": False,
            "optimizer_configs": [trajectory_optimizer],
            "metrics_rollout": trajectory_metrics,
            "transition_model": transition, "device_cfg": self.device_cfg,
            "num_seeds": budget.trajectory_seeds, "max_batch_size": 1, "max_goalset": 1,
            "use_cuda_graph": False, "random_seed": budget.random_seed,
            # BEGIN inert native pose storage has no trajectory acceptance role.
            "position_tolerance": problem.numerical.position_m,
            "orientation_tolerance": problem.numerical.rotation_rad,
            # END typed functional requirements own their original tolerances.
            "cost_manager_config_instance_type": self._manager_type,
            # BEGIN only action integration is replaced through the native API.
            "transition_model_config_instance_type": cubic_transition_config_type(),
            # END optimizer, robot FK, task costs and limits remain native.
            "override_optimizer_num_iters": {"lbfgs": budget.trajectory_iterations},
            # Native retiming runs even with zero finetuning attempts. A
            # different dt changes derivative constraints and, for nonzero
            # measured dq0, the cubic's entrance coefficient relation. Keep one
            # request-wide physical clock through optimization and metrics.
            "minimum_trajectory_dt": budget.maximum_trajectory_dt,
            "maximum_trajectory_dt": budget.maximum_trajectory_dt,
            "interpolation_dt": budget.interpolation_dt,
            "interpolation_buffer_size": budget.interpolation_buffer_size,
        }

        class ProcessTrajectoryOptimizer(TrajectoryOptimizer):
            # BEGIN every native entry point binds the explicit problem batch.
            def _prepare_goal_buffer(self, solve_state, goal_tool_poses, current_state=None,
                                     use_implicit_goal=False, seed_goal_state=None, goal_state=None):
                """Preserve native buffers while completing goal-free metadata.

                The installed registry infers its batch only from pose or joint
                tracking goals. Our process constraints own those requirements,
                so seed-only registries retain -1 unless the request supplies
                this metadata. Bind it here for direct solves, candidate metrics
                and resident reuse before any rollout allocates state storage.
                """
                buffer, update_reference = super()._prepare_goal_buffer(
                    solve_state, goal_tool_poses, current_state, use_implicit_goal,
                    seed_goal_state, goal_state)
                buffer.batch_size = solve_state.batch_size
                return buffer, update_reference
            # END no synthetic tracking goal or third-party mutation is needed.

            # BEGIN bind both completed query identities before native cloning/ranking.
            def _get_result(self, opt_result, metrics_result, interpolated_metrics,
                            interpolated_trajectory, last_tstep, num_seeds, return_seeds):
                self.metrics_rollout.metrics_constraint_manager.bind_scene_evidence(metrics_result)
                self.additional_metrics_rollouts["interpolated_rollout"].metrics_constraint_manager.bind_scene_evidence(
                    interpolated_metrics)
                return super()._get_result(opt_result, metrics_result, interpolated_metrics,
                    interpolated_trajectory, last_tstep, num_seeds, return_seeds)
            # END regular-only collisions cannot disappear behind a clear dense grid.

            def get_interpolated_trajectory(self, js_optimized):
                # BEGIN preserve raw controls and boundary derivatives on one clock.
                from curobo._src.util.trajectory import TrajInterpolationType

                grid = _formal_native_clock(self)
                if self.config.interpolation_type != TrajInterpolationType.BSPLINE_KNOTS_CUDA:
                    raise ValueError("Exact native cubic export requires the configured B-spline interpolation")
                batch = js_optimized.position.shape[0]
                if (js_optimized.position.ndim != 3 or not 1 <= batch <= self.config.num_seeds
                        or js_optimized.position.shape[1:] != (self.horizon, self.action_dim)
                        or js_optimized.knot is None
                        or js_optimized.knot.shape != (batch, self.action_horizon, self.action_dim)):
                    raise ValueError("Native interpolation state does not match the bounded active-joint layout")
                # Only a few timing scalars synchronize here. No trajectory
                # array is copied to CPU, and no action or derivative is changed.
                for name, expected in (("dt", grid["native_dt"]), ("knot_dt", grid["knot_dt"])):
                    value = getattr(js_optimized, name)
                    if (value is None or value.numel() != batch
                            or value.dtype != torch.float32 or not bool((value == expected).all())):
                        raise ValueError("Native result changed the fixed request-wide spline clock")
                capacity = self.config.interpolation_buffer_size
                if (grid["valid_count"] > capacity or capacity <= self.horizon or capacity > 2048):
                    raise ValueError("Native interpolation exceeds its declared fixed buffer capacity")
                shape = (batch, capacity, self.action_dim)
                if self._interpolated_traj_buffer is None or self._interpolated_traj_buffer.shape[0] != batch:
                    self._interpolated_traj_buffer = JointState.zeros(shape, self.device_cfg,
                                                                     joint_names=self.joint_names)
                elif self._interpolated_traj_buffer.shape != shape:
                    raise ValueError("Native interpolation buffer changed its declared shape")
                if self._interpolation_dt_buffer is None or self._interpolation_dt_buffer.shape != (batch,):
                    self._interpolation_dt_buffer = torch.full((batch,), grid["interpolation_dt"],
                        dtype=torch.float32, device=self.device_cfg.device)
                elif not bool((self._interpolation_dt_buffer == grid["interpolation_dt"]).all()):
                    raise ValueError("Native interpolation buffer changed its exact output dt")
                counts = torch.full((batch,), grid["valid_count"], dtype=torch.int32,
                                    device=self.device_cfg.device)
                goal = self.core._goal_buffer
                js_optimized.control_space = self.transition_model.control_space
                # BEGIN dense output uses the exact same map as regular rollout.
                step = self.transition_model._rollout_step_fn
                if (step.action_horizon != self.action_horizon or step.knot_dt != grid["knot_dt"]):
                    raise ValueError("Cubic export changed the bound coefficient map or clock")
                state = step.sample(js_optimized.knot, goal.current_js, goal.idxs_current_js,
                    self._interpolated_traj_buffer, valid_count=grid["valid_count"],
                    dense=True, output_dt=grid["interpolation_dt"])
                # END no second boundary kernel or independent curve is retained.
                rollout = self.additional_metrics_rollouts["interpolated_rollout"]
                # Ordinary costs and both constraint managers retain the valid
                # prefix before dense metrics and their original success flag.
                for manager in rollout._cost_manager_list:
                    if getattr(manager, "_process_mode", ""):
                        manager.set_dense_counts(counts)
                return state, counts, False
                # END no output resize, retiming, independent interpolation or metric bypass.

        self._optimizer_class = ProcessTrajectoryOptimizer
        self._optimizer_config_class = TrajectoryOptimizerCfg
        self.trajopt = ProcessTrajectoryOptimizer(TrajectoryOptimizerCfg.create(**deepcopy(self._trajectory_configuration)))
        self._bind_formal_clock()
        self._initialize_verification_constraints()
        self._bind_physical_domains()
        self._manager_type.bind_request(problem, scene, self.progress_map, self.anchors, budget,
                                       self.functional_constraints, self.bound_reference)

    # BEGIN resident native solvers rebind only request values and optimizer state.
    def _bind_online_robot(self):
        self.scene.bind_online_robot(configuration=self.robot_dict,
            validator=self.robot_validator, device=self.device_cfg.device,
            check_request=lambda: self.check_request())

    def rebind(self, problem, scene, budget, *, continuation=None, goal_identities=None):
        """Reuse IK/TrajOpt while clearing all prior candidate and request state."""
        # BEGIN a new request cannot change the resident robot calibration.
        self.robot_validator.validate_planning_binding(problem, scene.gripper)
        # END measured entrance and hand coordinates keep their named model.
        scene.validate_problem(problem, goal_identities=goal_identities)
        budget, mapping = (_initial_timing(problem, budget) if continuation is None else
                           _continuation_timing(problem, budget, continuation))
        anchors = np.unique(np.r_[problem.anchors, scene.required_progress_anchors])
        anchors = anchors[(anchors >= problem.start_progress) & (anchors <= problem.end_progress)]
        # BEGIN resident requests replace their complete reference time map.
        self.source_progress_map, self.reference_time_map = mapping, None
        if continuation is None:
            mapping, self.reference_time_map = _progress_for_cubic_boundaries(mapping, budget.knots,
                stop_progress=problem.stop_progress, anchors=anchors)
        grid = _fixed_spline_grid(intervals=budget.knots+4, native_dt=budget.maximum_trajectory_dt,
            interpolation_steps=budget.interpolation_steps, interpolation_dt=budget.interpolation_dt)
        if continuation is None:
            before_tick = mapping
            mapping = _bind_event_clock(problem, mapping, grid["duration"],
                None if scene.gripper is None else scene.gripper.control_timestep_s)
            self.reference_time_map = _ProgressMap(_phase_aligned_parameters(
                before_tick, mapping, self.reference_time_map.u), self.reference_time_map.s)
        # END a previous request's timing cannot enter this candidate's seed.
        _anchor_capacity(mapping, anchors, budget)
        workspace = _planning_workspace_bytes(scene, self.robot_metadata, budget, self.anchor_capacity)
        _require_planning_host_memory(workspace.cpu_live_bytes, encoded_record_bytes_upper=0,
            max_host_rss_bytes=budget.max_host_rss_bytes)
        _require_planning_memory(self.torch, self.device_cfg.device, workspace,
            max_device_used_bytes=budget.max_device_used_bytes)
        if scene.gripper is not None:
            scene.gripper.bind_clock(grid["duration"], mapping)
        self.problem, self.scene, self.budget, self.progress_map, self.anchors = problem, scene, budget, mapping, anchors
        self.formal_clock.clear()
        self.formal_clock.update(grid)
        self.finger_trajectory = None if scene.gripper is None else scene.gripper.finger_trajectory
        columns = [problem.joint_names.index(name) for name in self.joint_names]
        self.start = self.JointState.from_numpy(list(self.joint_names), problem.start_q[columns][None],
            velocity=problem.start_dq[columns][None], device_cfg=self.device_cfg)
        self._bind_online_robot()
        from .functional_augmented_lagrangian import FunctionalAugmentedLagrangian
        # BEGIN resident requests bind their own entrance twist and physical time.
        from .timed_reference import BoundToolReference
        entrance_pose, entrance_twist = _reference_entrance(problem, scene, self.robot_validator)
        self.bound_reference = BoundToolReference(problem, mapping.u, mapping.s,
            grid["duration"], entrance_pose, entrance_twist)
        # END a prior reference cannot survive a new robot feedback binding.
        self.functional_constraints = FunctionalAugmentedLagrangian(problem.functional_constraint_keys)
        self._manager_type.bind_request(problem, scene, mapping, anchors, budget,
                                       self.functional_constraints, self.bound_reference)
        self._verification_scene = scene.clone_for_rollout()
        self.trajopt.config.minimum_trajectory_dt = budget.maximum_trajectory_dt
        self.trajopt.config.maximum_trajectory_dt = budget.maximum_trajectory_dt
        # BEGIN every resident transition receives this request's explicit clock.
        transitions = {id(rollout.transition_model): rollout.transition_model
            for rollout in [*self.trajopt.core.get_all_rollout_instances(),
                            *self.trajopt.additional_metrics_rollouts.values()]}
        for rollout in [*self.trajopt.core.get_all_rollout_instances(),
                        *self.trajopt.additional_metrics_rollouts.values()]:
            transitions[id(rollout.metrics_transition_model)] = rollout.metrics_transition_model
        for transition in transitions.values():
            transition.update_traj_dt(budget.maximum_trajectory_dt)
        # END no old basis, q0 or dq0 survives an explicit request rebind.
        # BEGIN resident rebindings replace every value in the selected clock.
        self.trajopt.config.interpolation_dt = budget.interpolation_dt
        if self.trajopt._interpolation_dt_buffer is not None:
            self.trajopt._interpolation_dt_buffer.fill_(budget.interpolation_dt)
        # END reuse storage; never carry an earlier request's output spacing.
        self._bind_formal_clock()
        self.seed_parameter.zero_()
        self.last_validation, self.last_optimizer_diagnostics, self.goal_ik_candidates = None, [], {}
        self.delivery_candidate = None
        self.initialization_candidate = None
        self.reference_initialization = {"status": "not_started", "geometry_scope": "robot", "task_scene_constraints": "not_evaluated"}
        self.reference_ik_queries, self.reference_ik_chains, self.reference_seed_fits = [], [], []
        self.initial_seed_strategies = []
        for solver in (self.ik, self.trajopt):
            solver.reset_seed()
            # A missing goal must never reuse the previous request's pose.
            manager = solver.core.goal_registry_manager
            if manager._goal_buffer is not None:
                manager._goal_buffer.link_goal_poses = None
            rollouts = solver.core.get_all_rollout_instances()
            if solver is self.trajopt:
                rollouts += list(self.trajopt.additional_metrics_rollouts.values())
            for rollout in rollouts:
                rollout.reset()
                for name in ("_num_particles_goal", "_metrics_goal"):
                    goal = getattr(rollout, name, None)
                    if goal is not None:
                        goal.link_goal_poses = None
                        goal.goal_js = None
        self._bind_physical_domains()
        return self
    # END native solver objects and their fixed tensor capacities remain resident.

    # BEGIN bind the independently controlled fingers to native formal output.
    def _bind_formal_clock(self):
        # BEGIN confirm native construction preserved the already bound inputs.
        native = _formal_native_clock(self.trajopt)
        if any(native.get(name) != value for name, value in self.formal_clock.items()):
            raise ValueError("Native solver changed the admitted synchronized clock")
        self.formal_clock.update(native)
        # END do not regenerate hand commands after private rollouts were cloned.

    def _seed_u(self, progress):
        """Choose the earliest plateau point for initializer-only state queries."""
        values, indices = np.unique(self.progress_map.s, return_index=True)
        return float(np.interp(progress, values, self.progress_map.u[indices]))
    # END formal export always evaluates its actual time, not this seed inverse.

    # BEGIN bind occupancy evidence to the actual bounded current robot model.

    # END bind occupancy evidence to the actual bounded current robot model.

    def _initialize_verification_constraints(self):
        from curobo._src.rollout.cost_manager.cost_manager_robot import RobotCostManager
        from curobo._src.state.state_robot import RobotState
        self._RobotState = RobotState
        # The base manager supplies native joint/derivative limits at extra
        # control-curve samples. The same native self-collision policy
        # remains active in trajectory metrics, IK and the planning entrance.
        self._verification_native = RobotCostManager(self.device_cfg)
        self._verification_native.initialize_from_config(
            deepcopy(self.trajopt.config.metrics_rollout_config.constraint_cfg),
            self.trajopt.transition_model)

    # BEGIN constrain the actual cubic throughout native initialization/updates.
    def _bind_physical_domains(self):
        # BEGIN FUNCTO owns the optimizer/goal extensions on its solver instances.
        from .curobo_solver_components import bind_solver_components

        limits = self._verification_native.get_cost("cspace").config.joint_limits
        columns = [limits.joint_names.index(name) for name in self.joint_names]
        # BEGIN the cubic and its QP share all four physical derivative bounds.
        bounds = self.torch.stack([getattr(limits, name)[:, columns]
            for name in ("position", "velocity", "acceleration", "jerk")])
        self.position_domain = JointPositionDomain(bounds[0, 0], bounds[0, 1])
        self.physical_domain = CubicJointDomain(bounds,
            self.start.position, self.start.velocity, self.formal_clock["knot_dt"],
            self.trajopt.action_horizon)
        # END single-state IK retains its separate position-only domain.
        # BEGIN solver metrics remain eager; only admitted CUDA cores capture.
        cuda = False
        bind_solver_components(self.ik, self.position_domain, use_optimizer_graph=cuda)
        bind_solver_components(self.trajopt, self.physical_domain,
            use_optimizer_graph=cuda and self._trajectory_optimizer_graph_admitted)
        # END native fixed-iteration/kernel requirements remain explicit errors.
        # END the installed cuRobo library needs no physical-domain/cache patch.
    # END the model supplies all continuous bounds; IK uses its position interval.

    # BEGIN reject an immutable infeasible entrance before running native IK.
    def initial_state_diagnostics(self):
        """Test measured q/dq and online collision geometry before proposing motion."""
        limits = self._verification_native.get_cost("cspace").config.joint_limits
        columns = [limits.joint_names.index(name) for name in self.joint_names]
        q, dq = self.start.position[:, None], self.start.velocity[:, None]
        with self.torch.no_grad():
            components = {}
            for name, value in (("position", q), ("velocity", dq)):
                bounds = getattr(limits, name)[:, columns]
                components[name] = self.torch.relu(bounds[0]-value)+self.torch.relu(value-bounds[1])
            js = self.JointState(position=q.contiguous(), velocity=dq.contiguous(), joint_names=list(self.joint_names))
            state = self._verification_fk.compute_kinematics(js)
            native = self._RobotState(joint_state=js, cuda_robot_model_state=state)
            index = state.tool_poses.tool_frames.index(self.problem.tool_link)
            position, quaternion = self.problem.transform_tool_pose(
                state.tool_poses.position[:, :, index], state.tool_poses.quaternion[:, :, index])
            progress = q.new_full(q.shape[:2], self.problem.start_progress)
            # BEGIN retain the same entrance query's scene and native evidence.
            # The hard scene residual remains authoritative. Capture named
            # without rerunning geometry or changing any acceptance threshold.
            scene_components = {}
            # END diagnostic collection owns no additional collision query.
            components["scene"] = _scene_boundary_evaluate(self.scene, self.problem, self.progress_map,
                self.problem.transform_robot_spheres(native.robot_spheres), position, quaternion,
                progress, q, self.joint_names, trajectory_time=q.new_zeros(q.shape[:2]),
                normalized_time=q.new_zeros(q.shape[:2]),
                # BEGIN measured entrance uses the same registered-entity scene query.
                component_diagnostics=scene_components)
            residuals = {name: float(value.max().item()) for name, value in components.items()}
        failed = [name for name, value in residuals.items() if not np.isfinite(value) or value > 0]
        return {"passed": not failed, "residuals": residuals, "failed_components": failed,
                # BEGIN expose final same-query evidence beside the unchanged verdict.
                "scene_components": scene_components,
                "observed_geometry": deepcopy(self.scene.last_query_report),
                # END callers can identify a failed owner/pair without another solve.
                "scope": "measured joint limits and selected online collision model"}
    # END reject an immutable infeasible entrance before running native IK.

    def close(self):
        # BEGIN release native callback cycles at the request lifecycle boundary.
        # Native destroy() resets CUDA graphs. LBFGS and GradientOptCore also
        # retain each other through bound callbacks, so drop every solver
        # reference before collecting their now-unreachable tensor workspaces.
        # BEGIN restore caller precision even if native destruction raises.
        try:
            for name in ("ik", "trajopt"):
                solver = getattr(self, name)
                if solver is not None:
                    solver.destroy()
                setattr(self, name, None)
            del solver
            gc.collect()
        finally:
            prior = getattr(self, "_caller_precision", None)
            if prior is not None:
                self._precision_torch.backends.cuda.matmul.allow_tf32 = prior[0]
                self._precision_torch.backends.cudnn.allow_tf32 = prior[1]
                self._caller_precision = None
        # END repeated close() cannot overwrite a later caller setting.
        # END release native callback cycles at the request lifecycle boundary.

    def _goal(self, time_s):
        """Create native link goals from the same physical-time TCP reference."""
        times = self.device_cfg.to_device([[float(time_s)]])
        position, quaternion = self.problem.reference_to_link_pose(*self.bound_reference.sample(times))
        return self.GoalToolPose(tool_frames=[self.problem.tool_link],
                                 position=position[:, :, None, None].contiguous(),
                                 quaternion=quaternion[:, :, None, None].contiguous())

    # BEGIN whole-chain selection and native-basis soft reference fitting.
    def _seed_velocity_bounds(self):
        """Read existing physical speed bounds in the action's named joint order."""
        limits = self._verification_native.get_cost("cspace").config.joint_limits
        columns = [limits.joint_names.index(name) for name in self.joint_names]
        return limits.velocity.detach().cpu().numpy()[:, columns]

    # BEGIN every native IK submission consumes the original request ledger.
    def _solve_ik(self, goal, seeds, *, optional=False):
        """Submit a bounded IK batch and retain work even if native code fails.

        ``seeds`` is [batch, supplied_seed, named_joint]. Native SeedManager
        fills each batch to the request's fixed ``ik_seeds``. The ledger unit
        is one submitted candidate solve, including unsuccessful candidates.
        Optional reference enrichment yields no result when its count quota
        is spent; required reference/event calls retain the request failure.
        ``request_work`` is the public request's cumulative ledger, bound
        before either reference initialization or event search is entered.
        """
        self.check_request()
        amount = int(seeds.shape[0])*self.budget.ik_seeds
        limit = (self.budget.max_validation_points*self.budget.ik_seeds
                 * self.budget.maximum_trajectory_solves)
        if self.request_work["ik_queries"] + amount > limit:
            if optional:
                return None
            raise PlanningWorkExhausted("The original cumulative ik_queries budget is exhausted")
        self.request_work["ik_queries"] += amount
        return self.ik.solve_pose(goal, current_state=self.start,
                                  seed_config=seeds, return_seeds=self.budget.ik_seeds)
    # END query purpose does not reset the cumulative candidate count.

    def reference_seeds(self, joint_reference=None):
        """Fit and refine bounded candidates against the complete tool reference.

        Finite position-domain IK states provide numerical chain proposals;
        only native-success rows supply joint observations to the curve fit.
        One tracking state leaves the remaining fixed IK slots for exploration.
        Direct curve refinement owns the returned coefficients, including all
        original process/event requirements on the bound physical clock.
        """
        # BEGIN completed initialization records also survive an interrupted request.
        self.joint_reference_report = None
        self.reference_ik_queries, self.reference_ik_distinct_counts, self.reference_ik_chains = [], [], []
        self.reference_seed_fits, self.initial_seed_strategies = [], []
        self.reference_initialization = {"status": "collecting_proposals", "geometry_scope": "robot", "task_scene_constraints": "not_evaluated"}
        # END records contain only completed work on this request's inputs.
        if joint_reference is not None:
            if not isinstance(joint_reference, JointTrajectory) or set(joint_reference.joint_names) != set(self.joint_names):
                raise MotionPlanningError("unsupported_reference", "Joint reference must name the backend's active joints")
            columns = [joint_reference.joint_names.index(name) for name in self.joint_names]
            q = joint_reference.position[:, columns]
            dq = joint_reference.velocity[:, columns]
            if (not np.isfinite(q).all() or not np.isfinite(dq).all()
                    or joint_reference.duration <= 0 or joint_reference.times[0] != 0):
                raise MotionPlanningError("unsupported_reference", "Joint reference must have finite q/dq and a zero-origin clock")
            # Fitting is optional proposal generation. It never rewrites the
            # measured entrance or replaces current grounded functional goals.
            # BEGIN joint samples follow the task's complete phase/hold clock.
            source_map = _ProgressMap(joint_reference.progress_u, joint_reference.progress_s)
            original_u = _phase_aligned_parameters(source_map, self.source_progress_map,
                                                   joint_reference.times/joint_reference.duration)
            if self.reference_time_map is None:
                parameters = _phase_aligned_parameters(source_map, self.progress_map,
                    joint_reference.times/joint_reference.duration)
            else:
                parameters = np.asarray([self.reference_time_map.anchor_parameters([value])[0, 0]
                                         for value in original_u])
            controls, fit = self.physical_domain.fit_reference(parameters,
                self.device_cfg.to_device(q))
            # END fitting changes shape only after the unmodified candidate failed.
            self.joint_reference_report = {"fit": fit, "source_curve_preserved": False,
                "source_duration_s": joint_reference.duration,
                "bound_duration_s": self.formal_clock["duration"],
                "scope": "Native-basis proposal; complete trajectory qualification follows refinement"}
            if controls is not None:
                count = self.budget.trajectory_seeds
                seeds = controls[None, None].expand(1, count, -1, -1).contiguous()
                self.reference_ik_queries, self.reference_ik_distinct_counts, self.reference_ik_chains = [], [], []
                self.reference_seed_fits = [deepcopy(fit) for _ in range(count)]
                self.initial_seed_strategies = self.seed_strategies = ["recorded_joint_reference"]*count
                return self._refine_reference_seeds(seeds), []
        # BEGIN successful IK rows alone supply the coarse joint observations.
        previous = self.start.position[:, None].clone()
        source = {"query_index": None, "candidate_index": None, "kind": "measured_entrance"}
        layers = {}
        for u in self.physical_domain._curve_map.reference_parameters:
            layer = self._reference_ik_query(float(u), previous,
                warm_start_source=source, kind="original")
            layers[float(u)] = layer
            tracking = layer["tracking"]
            if tracking is not None:
                previous = self.device_cfg.to_device(layer["values"][tracking:tracking+1][None]).contiguous()
                source = {"query_index": layer["query_index"], "candidate_index": tracking,
                          "kind": "previous_tracking_state"}
        chains = self._select_reference_chains(layers)
        self.reference_ik_chains = chains
        # q0 repeats are free-coefficient proposals. Domain initialization uses
        # its bound q0/dq0 and the same clock to construct a legal entry curve.
        proposals = self.start.position[:, None, None].expand(1, self.budget.trajectory_seeds,
            self.trajopt.action_horizon, -1).contiguous()
        entry_controls = self.physical_domain.initialize(proposals)
        seeds, self.reference_seed_fits, self.initial_seed_strategies = self._fit_reference_chains(
            chains, entry_controls)
        refined = self._refine_reference_seeds(seeds, enrichment=(layers, chains, entry_controls))
        return refined, [row["native_success_count"] for row in self.reference_ik_queries]
        # END failed proposals remain reported but carry no joint fitting weight.

    def _reference_ik_query(self, u, previous, *, warm_start_source, kind, optional=False):
        """Keep one native query's original candidate and static evidence identity.

        ``u`` is normalized physical time; ``previous[1,1,D]`` is the declared
        warm start in active-joint order. Original and additional queries use
        the same full tool reference, static checks and fixed native seed cap.
        The execution log is append-only; time-ordered pool views are separate.
        """
        from .planning_ik import select_ik_tracking_candidate
        position_limits = np.stack((self.position_domain.lower.detach().cpu().numpy(),
                                    self.position_domain.upper.detach().cpu().numpy()))
        query_index = len(self.reference_ik_queries)
        self.seed_parameter.fill_(float(u))
        result = self._solve_ik(self._goal(u*self.formal_clock["duration"]), previous, optional=optional)
        if result is None:
            return None
        values = self.ik.get_active_js(result.js_solution).position[0].detach().cpu().numpy()
        flags = result.success[0].detach().cpu().numpy()
        success_count = int((flags & np.isfinite(values).all(-1)).sum())
        ranked, reports = self._rank_ik_candidates(values, flags,
            previous[0].detach().cpu().numpy(), float(u))
        # BEGIN proposal admission must not confuse missing observations with failed IK.
        # The diagnostic name refers to native pose-IK solver
        # success and native joint constraints. Task and online
        # scene results remain independent evidence for ranking and for
        # the final complete-curve checks; neither authorizes this seed.
        legal = (np.isfinite(values).all(-1) & (values >= position_limits[0]).all(-1)
                 & (values <= position_limits[1]).all(-1))
        qualified = [int(index) for index in ranked if legal[index] and reports[index]["native_pose_ik_success"]]
        self.reference_ik_distinct_counts.append(len(qualified))
        for index, report in enumerate(reports):
            report["reference_status"] = ("SOLVED" if np.isfinite(values[index]).all()
                and report["native_pose_ik_success"]
                else "FAILED_REFERENCE")
        # Native result errors describe convergence to the nominal pose;
        # static task and geometry evidence above remain independent.
        residuals = {}
        for attribute, name in (("position_error", "nominal_position_error_m"),
                                ("rotation_error", "nominal_rotation_error_rad")):
            errors = getattr(result, attribute)[0].detach().cpu().numpy().reshape(-1)
            if errors.shape != (len(reports),):
                raise ValueError("Native IK convergence errors must align with returned candidates")
            residuals[attribute] = errors
            for report, error in zip(reports, errors):
                report[name] = float(error) if np.isfinite(error) else None
        # BEGIN original convergence separates solved and approximate tracking states.
        # True-angle metrics reconstruct the untruncated energy; the cost
        # kernel's tiny energy dead zone does not change convergence flags.
        wp, wr = self.reference_ik_pose_weights
        with np.errstate(over="ignore", invalid="ignore"):
            energy = .5*wp*residuals["position_error"]**2+wr*residuals["rotation_error"]**2
        proposals = [int(index) for index in ranked if legal[index] and np.isfinite(energy[index])]
        tracking = select_ik_tracking_candidate(values, proposals, energy,
            previous[0].detach().cpu().numpy(), reports=reports)
        for index, report in enumerate(reports):
            report["position_domain_valid"] = bool(legal[index])
            report["quadratic_pose_energy"] = float(energy[index]) if np.isfinite(energy[index]) else None
        # END approximate states keep their original failed-pose evidence.
        for index, report in enumerate(reports):
            report.update(source_query_index=query_index, source_candidate_index=index)
        self.reference_ik_queries.append({"u": float(u),
            "time_s": float(u*self.formal_clock["duration"]),
            "reference_status": "SOLVED" if qualified else "FAILED_REFERENCE",
            "seed_status": "SOLVED" if qualified else "APPROXIMATE_PROPOSAL" if proposals else "NO_LEGAL_PROPOSAL",
            "qualified_candidate_indices": qualified,
            "native_success_count": success_count,
            "proposal_candidate_indices": proposals,
            "query_index": query_index, "kind": kind,
            "warm_start_from_query": warm_start_source["query_index"],
            "warm_start_source": warm_start_source,
            "warm_start_count": int(previous.shape[1]),
            "generated_seed_count": self.budget.ik_seeds-int(previous.shape[1]),
            "tracking_candidate_index": tracking,
            "candidates": [{"joint_position": [float(value) if np.isfinite(value) else None
                                                for value in row], "report": report}
                           for row, report in zip(values, reports)]})

        return {"values": values, "reports": reports, "energies": energy,
                "tracking": tracking, "success_count": success_count, "query_index": query_index}

    def _select_reference_chains(self, layers):
        """Select complete numerical chains from the time-ordered pool view."""
        from .planning_ik import select_ik_chains
        parameters = sorted(layers)
        rows = [layers[u] for u in parameters]
        chains = select_ik_chains([row["values"] for row in rows], [row["reports"] for row in rows],
            self.start.position.detach().cpu().numpy(),
            np.asarray(parameters)*self.formal_clock["duration"], self._seed_velocity_bounds(),
            count=self.budget.trajectory_seeds,
            position_limits=np.stack((self.position_domain.lower.detach().cpu().numpy(),
                                      self.position_domain.upper.detach().cpu().numpy())),
            pose_costs_by_query=[row["energies"] for row in rows],
            joint_tolerance=self.problem.numerical.joint)
        for chain in chains:
            chain["query_parameters"] = parameters
            chain["candidate_sources"] = [None if index is None else {
                "query_index": row["reports"][index]["source_query_index"],
                "candidate_index": row["reports"][index]["source_candidate_index"]}
                for row, index in zip(rows, chain["candidate_indices"])]
        return chains

    def _fit_reference_chains(self, chains, entry_controls, *, previous_seeds=None):
        """Fit only successful observations with a fixed physical entry prior.

        The basis, four derivative bounds and measured q0/dq0 are owned by
        ``physical_domain``. Missing observations do not create equalities or
        fabricated samples. Its existing tiny regularizer selects an under-
        determined solution near the legal entry curve. A failed new fit keeps
        the corresponding original coarse curve for the later comparison.
        """
        controls, fits, strategies = [], [], []
        for branch in range(self.budget.trajectory_seeds):
            chain = chains[branch] if branch < len(chains) else None
            mask = np.asarray([] if chain is None else chain["observation_mask"], dtype=bool)
            prior = entry_controls[0, branch]
            if mask.any():
                parameters = np.asarray(chain["query_parameters"])[mask]
                targets = self.device_cfg.to_device(np.asarray(chain["joint_positions"])[mask])
                control, fit = self.physical_domain.fit_reference(parameters, targets, prior_controls=prior)
                strategy = "successful_reference_ik_fit"
                if control is None:
                    control = prior if previous_seeds is None else previous_seeds[0, branch]
                    strategy = "entry_curve_after_fit_failure" if previous_seeds is None else "original_curve_after_fit_failure"
            else:
                control, fit, strategy = prior, {"success": True}, "physical_entry_curve"
            fits.append({**fit, "observation_count": int(mask.sum()),
                "observation_query_indices": np.flatnonzero(mask).tolist(),
                "source": strategy})
            controls.append(control)
            strategies.append(strategy)
        return self.torch.stack(controls)[None], fits, strategies
    # END q0/dq0, native clock, hard task checks and admitted seed count are fixed.

    # BEGIN one bounded query pass precedes the full-curve optimizer lifecycle.
    def _enrich_reference_seeds(self, seeds, coarse, layers, chains, entry_controls,
                                *, initialization, total):
        """Spend at most two native queries per original coarse curve.

        The ordinary reference cost already includes physical-time weights.
        Its largest unqueried contribution chooses one new time. A successful
        new query may warm-start its right original neighbor once. Pool views
        keep at most the original S candidates and retain producer identities.
        All fits use the same fixed basis and are compared under one reset AL
        objective before L-BFGS or local geometry histories are initialized.
        """
        from .planning_ik import rank_ik_candidates
        started = time.monotonic()
        stop = started+self.remaining_request_seconds()*initialization/total
        first_query, first_work = len(self.reference_ik_queries), self.request_work["ik_queries"]
        record = {"status": "collecting", "maximum_new_calls": 2*len(seeds[0]),
                  "calls_started": 0, "candidate_solves": 0, "selected_sources": []}
        self.reference_initialization["enrichment"] = record
        original_parameters = np.asarray(sorted(layers))
        new_observations = 0
        duration = self.formal_clock["duration"]
        # Original query and grid identities share the native physical-time
        # dtype. This is exact representation equality, not a new tolerance.
        time_dtype = self.start.position.detach().cpu().numpy().dtype
        def time_key(u):
            return float(np.asarray(u*duration, dtype=time_dtype))
        queried = {time_key(u) for u in original_parameters}
        queried.add(time_key(0.))

        def submit(u, previous, source, kind):
            if not initialization or time.monotonic() >= stop:
                return None
            before = self.request_work["ik_queries"]
            try:
                return self._reference_ik_query(u, previous, warm_start_source=source,
                                                kind=kind, optional=True)
            finally:
                # Native exceptions retain actual submitted work as well.
                increment = self.request_work["ik_queries"]-before
                record["calls_started"] += int(increment > 0)
                record["candidate_solves"] += increment

        def successful_indices(layer):
            return [index for index, row in enumerate(layer["reports"])
                    if row["native_pose_ik_success"] and row["position_domain_valid"]
                    and np.isfinite(layer["energies"][index])]

        try:
            for branch, snapshot in enumerate(coarse["seeds"]):
                if not initialization or time.monotonic() >= stop:
                    break
                times = np.asarray(snapshot["reference_times_s"])
                contributions = np.asarray(snapshot["reference_point_costs"])
                available = [index for index, t in enumerate(times) if time_key(t/duration) not in queried]
                if not available:
                    break
                index = max(available, key=lambda item: (contributions[item], -times[item]))
                u = float(times[index]/duration)
                previous = self.start.position[:, None].clone()
                source = {"query_index": None, "candidate_index": None, "kind": "measured_entrance"}
                if branch < len(chains):
                    chain = chains[branch]
                    left = [i for i, parameter in enumerate(chain["query_parameters"])
                            if parameter < u and chain["observation_mask"][i]]
                    if left:
                        i = left[-1]
                        previous = self.device_cfg.to_device([[chain["joint_positions"][i]]])
                        source = {**chain["candidate_sources"][i], "kind": "left_successful_reference"}
                added = submit(u, previous, source, "additional")
                if added is None:
                    break
                queried.add(time_key(u))
                layers[u] = added
                successful = successful_indices(added)
                new_observations += len(successful)
                right = original_parameters[original_parameters > u]
                if successful and len(right):
                    # A successful state is selected with the common evidence
                    # policy; a failed tracking proposal cannot warm this call.
                    order = rank_ik_candidates(added["values"][successful],
                        [added["reports"][i] for i in successful], previous[0].detach().cpu().numpy(),
                        count=1, joint_tolerance=self.problem.numerical.joint, query_kind="reference")
                    chosen = successful[int(order[0])]
                    warm = self.device_cfg.to_device(added["values"][chosen:chosen+1][None])
                    right_u = float(right[0])
                    repeated = submit(right_u, warm, {"query_index": added["query_index"],
                        "candidate_index": chosen, "kind": "additional_successful_reference"}, "right_neighbor")
                    if repeated is not None:
                        old = layers[right_u]
                        values = np.concatenate((old["values"], repeated["values"]))
                        reports = old["reports"]+repeated["reports"]
                        energies = np.concatenate((old["energies"], repeated["energies"]))
                        legal = [i for i, row in enumerate(reports)
                                 if row["position_domain_valid"] and np.isfinite(energies[i])]
                        indices = rank_ik_candidates(values[legal], [reports[i] for i in legal],
                            warm[0].detach().cpu().numpy(), count=self.budget.ik_seeds,
                            joint_tolerance=self.problem.numerical.joint, query_kind="reference")
                        retained = np.asarray(legal, dtype=int)[indices]
                        layers[right_u] = {"values": values[retained], "reports": [reports[i] for i in retained],
                                           "energies": energies[retained]}
            if not new_observations:
                record.update(status=("no_additional_query" if len(self.reference_ik_queries) == first_query
                                      else "no_new_successful_observation"),
                              selected_sources=["original"]*len(seeds[0]))
                return seeds, coarse, [self._controls_curve(row) for row in seeds[0]]
            updated_chains = self._select_reference_chains(layers)
            proposed, fits, strategies = self._fit_reference_chains(updated_chains, entry_controls,
                                                                    previous_seeds=seeds)
            proposal, proposed_curves = self._initialization_snapshot(proposed[0])
            record.update(proposed=proposal, chains=updated_chains, fits=fits)
            selected, curves, selected_records = [], [], []
            for branch, (old, new) in enumerate(zip(coarse["seeds"], proposal["seeds"])):
                def key(row):
                    feasible = row["sampled_scope_feasible"]
                    objective = row["fixed_objective"] if feasible else row["optimization_objective"]
                    return (not feasible, objective if np.isfinite(objective) else float("inf"))
                improved = key(new) < key(old)
                selected.append(proposed[0, branch] if improved else seeds[0, branch])
                curves.append(proposed_curves[branch] if improved else self._controls_curve(seeds[0, branch]))
                selected_records.append(new if improved else old)
                record["selected_sources"].append("enriched" if improved else "original")
                if improved:
                    self.initial_seed_strategies[branch] = strategies[branch]+"+additional_reference"
                    self.reference_seed_fits[branch] = fits[branch]
            self.reference_ik_chains = updated_chains
            record["status"] = "completed"
            return self.torch.stack(selected)[None], {**coarse, "seeds": selected_records}, curves
        finally:
            record.update(elapsed_seconds=time.monotonic()-started,
                          candidate_solves=self.request_work["ik_queries"]-first_work)
    # END original curves compete with complete refits under one shared objective.

    # BEGIN complete tool-curve initialization shares the native task optimizer.
    def _initialization_snapshot(self, actions):
        """Evaluate the actual saved coefficients and export their physical curve.

        This forward uses the optimizer's robot geometry scope. Its same-query
        evidence includes original process/event sets, native joint terms and
        the shared official robot self-collision computation.
        The optimizer already runs inside the shared four-derivative domain.
        These reports describe its saved polynomial and task evidence before
        the later full-scene candidate qualification.
        """
        from .execution_binding import curve_snapshot
        self.check_request()
        rollout = self.trajopt.optimizer_rollouts[-1]
        # This optimizer rollout owns two copies of the same cost/constraint
        # configuration. Its metrics entry uses one entrance index per seed;
        # evaluate_action uses the line-search-particle-expanded goal indices.
        # Both copies retain the robot geometry scope and identical physical clock.
        manager = rollout.metrics_constraint_manager
        with self.torch.no_grad():
            evaluated = rollout.compute_metrics_from_action(actions)
            # Native aggregated feasibility includes AL objective terms. The
            # same-query owner supplies original task/joint evidence instead.
            feasible, signed, objective = manager.optimizer_evidence(evaluated)
            costs = evaluated.costs_and_constraints
            components = {name: value.detach().flatten(1).sum(-1).cpu().tolist()
                          for name, value in zip(costs.costs.names, costs.costs.values)}
            constraints = {name: value.detach().flatten(1).amax(-1).cpu().tolist()
                           for name, value in zip(costs.constraints.names, costs.constraints.values)}
            optimization_objective = costs.get_sum_cost_and_constraint(sum_horizon=True).reshape(-1).cpu().tolist()
            # BEGIN ordinary contributions retain native physical quadrature.
            reference_points, reference_times = None, None
            if "continuous_reference" in costs.costs.names:
                values = costs.costs.values[costs.costs.names.index("continuous_reference")]
                joint = evaluated.state.joint_state
                counts = (self.torch.full((len(actions),), joint.position.shape[1],
                          dtype=self.torch.long, device=joint.position.device)
                          if manager._dense_counts is None else manager._dense_counts)
                reference_times = _joint_state_sample_times(joint, counts).cpu().numpy()
                reference_points = values.detach().reshape(len(actions), values.shape[1], -1).sum(-1).cpu().numpy()
                counts = counts.cpu().tolist()
            # END padding contributes neither a new query time nor a sample.
            signed = {name: value.cpu().tolist() for name, value in signed.items()}
            feasible, objective = feasible.cpu().tolist(), objective.cpu().tolist()
            # BEGIN snapshots export the actual coefficients at their original breaks.
            records, curves = [], []
            for index in range(len(actions)):
                curve = self._controls_curve(actions[index])
                domain = self._candidate_domain_validation(curve)
                records.append({"seed_index": index, "sampled_scope_feasible": feasible[index],
                    "fixed_objective": objective[index],
                    "optimization_objective": optimization_objective[index],
                    "objective_components": {key: value[index] for key, value in components.items()},
                    "optimizer_constraints": {key: value[index] for key, value in constraints.items()},
                    "functional_inequalities": {key: value[index] for key, value in signed.items()},
                    "continuous_domain": domain, "continuous_residuals": deepcopy(curve.residuals),
                    "controls": actions[index].detach().cpu().tolist(),
                    "trajectory": curve_snapshot(curve)})
                if reference_points is not None:
                    records[-1].update(reference_times_s=reference_times[index, :counts[index]].tolist(),
                        reference_point_costs=reference_points[index, :counts[index]].tolist())
                curves.append(curve)
            # END same-forward sampled evidence and saved polynomial keep their scope.
        self.check_request()
        return {"geometry_scope": "robot", "task_scene_constraints": "not_evaluated", "seeds": records}, curves

    def _refine_reference_seeds(self, seeds, *, enrichment=None):
        """Optimize complete tool paths within the first solve's fixed quota.

        The original native iteration blocks are split between this task
        objective and full-scene optimization. Both use the same physical
        clock, bounded cubic domain and request deadline. Native reinitialize
        resets the objective-dependent history, multipliers and candidates.
        """
        self.physical_domain.validate(seeds)
        optimizer = self.trajopt.optimizer
        inner = optimizer.optimizers[-1].config.inner_iters
        total = self.budget.trajectory_iterations
        initialization = (total//inner//2)*inner
        self.main_trajectory_iterations = total-initialization
        record = self.reference_initialization
        record.update(status="preparing_curve_optimization", iteration_allocation={
            "total": total, "inner_block": inner, "initialization": initialization,
            "main": self.main_trajectory_iterations})
        # Native graph reset also clears the regular rollout's goal registry.
        # Reset before binding so both gradient entries retain the request's
        # measured entrance, explicit clock and free-terminal declaration.
        for stage in optimizer.optimizers:
            stage.reset_cuda_graph()
        state, goal = self._native_request(seeds)
        endpoint = self.JointState.from_position(seeds[:, :, -1].contiguous())
        endpoint.dt = self.torch.full((1, seeds.shape[1]), self.budget.maximum_trajectory_dt,
                                      dtype=seeds.dtype, device=seeds.device)
        buffer, _ = self.trajopt._prepare_goal_buffer(state, goal, self.start, False, endpoint)
        self.trajopt._update_rollout_params(buffer)
        actions = seeds.reshape(-1, self.trajopt.action_horizon, self.trajopt.action_dim).detach().clone()
        self.functional_constraints.bind(actions, num_seeds=len(actions))
        self.functional_constraints.reset()
        self._manager_type.set_geometry_scope("robot")
        try:
            coarse, curves = self._initialization_snapshot(actions)
            record.update(status="optimizing_curve", coarse=coarse)
            self.initialization_candidate = curves[0]
            # BEGIN enrichment compares complete curves before optimizer history.
            if enrichment is not None:
                layers, chains, entry_controls = enrichment
                record["coarse_chains"] = deepcopy(chains)
                seeds, selected, curves = self._enrich_reference_seeds(seeds, coarse,
                    layers, chains, entry_controls, initialization=initialization, total=total)
                actions = seeds[0].detach().clone()
                record["selected"] = selected
                self.initialization_candidate = curves[0]
            else:
                selected = coarse
            # END one subsequent reinitialize owns all AL and L-BFGS history.
            if initialization:
                optimizer.update_niters(initialization)
                self.check_request()
                optimizer.reinitialize(actions)
                actions = optimizer.optimize(actions).detach().clone()
                final, curves = self._initialization_snapshot(actions)
            else:
                final = selected
            result = actions.reshape(seeds.shape).contiguous()
            record.update(status="completed" if initialization else "no_initialization_iteration_block",
                refined=final, optimizer=self._optimizer_reports(result, seeds) if initialization else [])
            self.initialization_candidate = curves[0]
            self.seed_strategies = [name+"+tool_curve_refinement" if initialization else name
                                    for name in self.initial_seed_strategies]
            return result
        except Exception as error:
            record.update(status="interrupted", reason=str(error), error_type=type(error).__name__)
            raise
        finally:
            self._manager_type.set_geometry_scope("full")
            optimizer.update_niters(total)
            for stage in optimizer.optimizers:
                stage.reset_cuda_graph()
    # END only completed coefficient snapshots are retained across interruptions.

    # BEGIN scoped spatial search on one already bound physical clock.
    def ompl_seed(self, initial_seed, candidate_trajectory):
        """Return a spatial initializer and evidence; never produce a new clock.

        Only real relation switches, goal times and scene switches divide
        local OMPL charts. Their boundary configurations are sampled and
        projected into the original allowed sets, then revised on failure.
        Reference controls initialize these proposals without fixing them.
        A selected prefix supplies its actual release FK to its child scene.
        """
        torch = self.torch
        # BEGIN discrete search uses the existing registered-solid acceptance.
        # END differentiable projection equations retain their original costs.
        # BEGIN every required connection consumes the same request deadline.
        search_started = time.monotonic()
        deadline = search_started+self.budget.search_seconds
        # END later charts and alternate proposals never replenish search time.
        report = {"branches": [], "success": False, "state": "q,u=t/T",
                  "budget_seconds": self.budget.search_seconds, "clock_changed": False}
        lower = self.trajopt.transition_model.action_bound_lows.detach().cpu().numpy().reshape(-1)
        upper = self.trajopt.transition_model.action_bound_highs.detach().cpu().numpy().reshape(-1)
        tolerance = _projection_tolerance(self.problem)
        map_u, map_s = self.device_cfg.to_device(self.progress_map.u), self.device_cfg.to_device(self.progress_map.s)
        anchor_pairs = self.progress_map.anchor_parameters(self.anchors)
        query_u = anchor_pairs[:, 0]
        anchor_progress = dict(anchor_pairs)
        if self.finger_trajectory is not None:
            query_u = np.unique(np.r_[query_u,
                [event["applied_time_s"]/self.formal_clock["duration"] for event in self.finger_trajectory.events]])
        # The scene evaluator checks adjacent modes at this same q/u. Moving
        # float64 times by one ULP would vanish in the native float32 inputs.
        query_u = np.unique(query_u)
        start = np.r_[self.start.position[0].detach().cpu().numpy(), 0.]
        # Ordinary source phases and reference sample density never create
        # search boundaries. Plateaus retain both physical-time endpoints.
        boundary_times = [0., 1.]
        for name in ("pose_relations", "mechanisms", "interiors"):
            for requirement in getattr(self.problem, name):
                if requirement.start >= self.problem.start_progress:
                    boundary_times.append(self.progress_map.anchor_parameters([requirement.start])[0, 0])
                if requirement.end >= self.problem.start_progress:
                    boundary_times.append(self.progress_map.anchor_parameters([requirement.end])[-1, 0])
        goals_by_time = {}
        for name in ("pose_events", "coordinate_events", "supports"):
            for index, requirement in enumerate(getattr(self.problem, name)):
                parameter = self.progress_map.anchor_parameters([requirement.progress])[0, 0]
                boundary_times.append(parameter)
                goals_by_time.setdefault(float(parameter), []).append((name, index))
        scene_anchors = [value for value in self.scene.required_progress_anchors
                         if value >= self.problem.start_progress]
        if scene_anchors:
            boundary_times.extend(self.progress_map.anchor_parameters(scene_anchors)[:, 0])
        boundaries = np.unique(boundary_times)
        report["chart_times"] = boundaries.tolist()
        work = {"state_queries": 0}
        rng = np.random.default_rng(self.budget.random_seed)
        source_controls = initial_seed[0].detach().cpu().numpy()
        invalid_states = []

        def goals_at_time(parameter):
            return tuple(goals_by_time.get(float(parameter), ()))

        # BEGIN entity geometry queries consume the same search/request deadlines.
        def check_search():
            self.check_request()
            if time.monotonic() >= deadline:
                raise TimeoutError("The shared OMPL time budget was exhausted")
        # END source geometry queries cannot replenish the search allowance.

        def state_parts(values):
            check_search()
            count = int(np.prod(values.shape[:-1]))
            if work["state_queries"]+count > min(self.budget.max_validation_points, _NATIVE_STAGE_POINT_LIMIT):
                raise TimeoutError("The shared OMPL state-query budget was exhausted")
            work["state_queries"] += count
            joints = values[..., :-1].contiguous()
            zero = torch.zeros_like(joints)
            js = self.JointState(position=joints, velocity=zero, acceleration=zero, jerk=zero,
                dt=self.device_cfg.to_device([self.budget.interpolation_dt]), joint_names=list(self.joint_names))
            state = self._verification_fk.compute_kinematics(js)
            tools = state.tool_poses
            index = tools.tool_frames.index(self.problem.tool_link)
            quaternion = self._orientation_vjp.apply(joints, tools.quaternion.detach(),
                state.tool_jacobians[..., 3:, :].detach())
            position = self._native_layout.apply(tools.position)
            p, r = self.problem.transform_tool_pose(position[:, :, index], quaternion[:, :, index])
            return js, state, p, r, _torch_interpolate(map_u, map_s, values[..., -1])

        def equations(scene, active, goals=()):
            cache = {}
            def residual_function(values):
                js, _, p, r, progress = state_parts(values.reshape(1, 1, -1))
                process = self.problem.project_residual(p, r, progress,
                    joint_position=js.position, joint_names=self.joint_names, active_requirements=active)
                drive = scene.project_residual(progress, js.position, self.joint_names)
                goal = self.problem.goal_residual(p, r, progress, goal_ids=goals)
                return torch.cat((process, drive, goal), -1).reshape(-1)

            def evaluate(values, derivative):
                # Native setup, Newton projection and geodesic export often
                # repeat an identical query. Cache only immutable CPU numeric
                # answers for this exact chart/scene; no tensor graph or
                # result is shared with a different candidate branch.
                key = (np.asarray(values, dtype=float).tobytes(), derivative)
                if key in cache:
                    return cache[key]
                values_t = self.device_cfg.to_device(np.asarray(values))
                if derivative:
                    with torch.enable_grad():
                        if residual_function(values_t).numel() == 0:
                            return np.empty((0, len(values)))
                        answer = torch.autograd.functional.jacobian(residual_function, values_t,
                            create_graph=False, vectorize=False)
                else:
                    with torch.no_grad():
                        answer = residual_function(values_t)
                result = answer.detach().cpu().numpy().astype(float)
                if len(cache) < min(self.budget.max_validation_points, _NATIVE_STAGE_POINT_LIMIT):
                    cache[key] = result
                return result
            return evaluate

        def valid(values, scene):
            with torch.no_grad():
                values_t = self.device_cfg.to_device(values[None, None])
                js, state, p, r, progress = state_parts(values_t)
                clock_time = values_t[..., -1]*self.formal_clock["duration"]
                terms = {"process": self.problem.evaluate(p, r, progress, js.position,
                    self.joint_names, joint_velocity=js.velocity,
                    point_goal_ids=goals_at_time(float(values[-1])), direction_mode="point"),
                    "scene": _scene_boundary_evaluate(scene, self.problem, self.progress_map,
                        self.problem.transform_robot_spheres(state.robot_spheres), p, r, progress,
                        js.position, self.joint_names, trajectory_time=clock_time,
                        normalized_time=values_t[..., -1])}
                # BEGIN self collision consumes the original timed hand model.
                # Scene evaluation above transforms its own unpadded spheres;
                # only the native manager receives this separate timed state.
                native_state = self._RobotState(joint_state=js, cuda_robot_model_state=state)
                # END never apply native self padding to the scene spheres.
                # BEGIN OMPL shares the native registered-sphere self-collision contract.
                native = self._verification_native.compute_costs(native_state)
                terms.update({f"native:{name}": value for name, value in zip(native.names, native.values)})
                # BEGIN geometry queries and setup consume the same request deadline.
                check_search()
                # END all task, unknown, contact and unresolved terms remain hard.
                # BEGIN all existing tensors precede one exact boolean host read.
                accepted = _ompl_terms_accepted(terms)
                # END rejected-state diagnostics retain their original order/values.
                if not accepted and len(invalid_states) < 3:
                    invalid_states.append({"u": float(values[-1]), "progress": float(progress.item()),
                        "terms": {name: float(term.max().item()) for name, term in terms.items()}})
                return accepted

        # BEGIN a two-attempt boundary budget includes a non-reference posture.
        def proposals(begin, parameter):
            # BEGIN choose alternatives only within the statically qualified IK pool.
            entry = self.goal_ik_candidates.get(float(parameter))
            qualified = ([] if entry is None else [(index, row) for index, row in enumerate(entry["candidates"])
                if row["report"]["task_satisfied"] and row["report"]["model_valid"]
                and np.isfinite(row["joint_position"]).all()])
            if qualified:
                # The first ranked qualified event state is this search's
                # initial proposal; later attempts retain projected samples.
                index, row = qualified[0]
                yield "ranked_functional_ik", np.r_[row["joint_position"], parameter], {
                    "functional_ik_candidate_index": index, "functional_ik_report": deepcopy(row["report"])}
            else:
                yield "candidate_curve", np.r_[candidate_trajectory._curve(parameter*candidate_trajectory.duration), parameter], {}
            # END actual prefix geometry, projection and state checks recheck every proposal.
            # Reference hints must not consume every small attempt budget.
            # This proposal still uses the same allowed-set projection and
            # complete state/edge validity checks as every other member.
            yield "joint_box_random", np.r_[rng.uniform(lower, upper), parameter], {}
            # BEGIN proposal states come from the represented cubic, not coefficients.
            for controls in source_controls:
                yield "reference_curve", np.r_[self.physical_domain.evaluate(
                    controls, np.asarray([parameter]))[0], parameter], {}
            # END the original scene/projection checks still qualify each proposal.
            yield "previous_boundary", np.r_[begin[:-1], parameter], {}
            # Sampling the actual joint box permits different support tangents,
            # yaw and redundant IK branches. There is no reference corridor.
            while True:
                yield "joint_box_random", np.r_[rng.uniform(lower, upper), parameter], {}
        # END proposal ordering changes neither attempts nor time/query limits.

        # An explicit stack permits bounded backtracking without recursive
        # depth or copying all prior path samples into every branch frame.
        # BEGIN each backtracking frame owns one reusable scene-bound confirmer.
        # An explicit empty inventory means the scene has no native-confirmable
        # components; it never grants permission to skip its ordinary checks.
        def search_frame(begin, branch_scene, parameter):
            branch_scene.check_request = check_search
            return {"start": begin, "scene": branch_scene,
                    "proposals": proposals(begin, parameter), "attempts": 0, "path": None}

        scene = self.scene.clone_for_rollout()
        frames = [search_frame(start, scene, boundaries[1])]
        # END siblings cannot inherit a failed branch's release geometry or engine.
        try:
            # BEGIN only search consumes event allowed-set proposals and their time.
            self._build_goal_ik_candidates(initial_seed, check_search=check_search)
            # END curve initialization retains the original task-space event sets.
            while frames and time.monotonic() < deadline:
                stage = len(frames)-1
                frame = frames[-1]
                lo, hi = boundaries[stage:stage+2]
                # Every visit receives finite alternative members; exhausting
                # them returns to the preceding allowed set, not a new clock.
                if frame["attempts"] >= self.budget.ik_seeds:
                    frames.pop()
                    continue
                frame["attempts"] += 1
                begin, scene = frame["start"], frame["scene"]
                # BEGIN retain the initializer identity through projection.
                proposal_source, goal, proposal_evidence = next(frame["proposals"])
                # END the source labels a proposal, not a validity certificate.
                goal_s = float(anchor_progress.get(hi, np.interp(hi, self.progress_map.u, self.progress_map.s)))
                middle_s = float(np.interp((lo+hi)/2, self.progress_map.u, self.progress_map.s))
                active = self.problem.projection_requirements(middle_s)
                boundary_active = self.problem.projection_requirements(goal_s)
                goal_ids = goals_at_time(float(hi))
                boundary_equations = equations(scene, boundary_active)
                process_count = len(boundary_equations(goal, False))
                goal = _project_allowed_boundary(goal, equations(scene, boundary_active, goal_ids),
                    process_count, tolerance, np.r_[lower, lo], np.r_[upper, hi], deadline=deadline)
                # BEGIN rejected projections retain their proposal source too.
                evidence = {"chart": stage, "time_interval": [float(lo), float(hi)],
                            "goals": list(goal_ids), "attempt": frame["attempts"],
                            "proposal_source": proposal_source, **proposal_evidence}
                # END provenance does not alter projection or validity results.
                if goal is None:
                    report["branches"].append({**evidence, "failure": "allowed_boundary_projection_failed"})
                    continue
                evidence["boundary_state"] = goal.tolist()
                constraint, rank = _independent_projection(equations(scene, active), goal)
                if constraint is None:
                    report["branches"].append({**evidence, "failure": "singular_active_relation_chart"})
                    continue
                invalid_states.clear()
                # BEGIN a necessary connection may use all remaining request time.
                # Dividing time by future charts can reject this prerequisite
                # before they are reachable and leave the global budget idle.
                # Existing state-query and proposal caps still bound the work.
                stage_started = time.monotonic()
                seconds = max(0., deadline-stage_started)
                validity_cache = {}
                def stage_valid(value):
                    key = np.asarray(value, dtype=float).tobytes()
                    if key not in validity_cache:
                        validity_cache[key] = valid(value, scene)
                    return validity_cache[key]
                path, stage_report = _ompl_stage(begin, goal, np.r_[lower, lo], np.r_[upper, hi],
                    stage_valid, constraint, tolerance=tolerance,
                    step=self.budget.validation_joint_step, seconds=seconds,
                    sample_limit=min(self.budget.max_validation_points, _NATIVE_STAGE_POINT_LIMIT)-work["state_queries"],
                    query_parameters=query_u[(query_u >= lo) & (query_u <= hi)])
                evidence.update(stage_report, independent_equalities=rank,
                                allocated_seconds=seconds,
                                elapsed_seconds=time.monotonic()-stage_started,
                                first_invalid_states=deepcopy(invalid_states))
                # END projection, connection and later charts share one deadline.
                report["branches"].append(evidence)
                if path is None:
                    continue
                path = path[np.r_[True, np.any(path[1:] != path[:-1], axis=1)]]
                if np.any(np.diff(path[:, -1]) <= 0):
                    evidence["failure"] = "nonmonotone_or_instantaneous_joint_motion"
                    continue
                frame["path"] = path
                if sum(len(item["path"])-1 for item in frames)+1 > min(self.budget.max_validation_points, _NATIVE_STAGE_POINT_LIMIT):
                    evidence["failure"] = "whole_path_sample_budget"
                    continue
                if stage+2 < len(boundaries):
                    # This exact endpoint is the selected prefix's release
                    # FK. It is never taken from a demonstration or failed
                    # whole-curve witness, and sibling scenes remain private.
                    with torch.no_grad():
                        _, _, p, r, progress = state_parts(self.device_cfg.to_device(goal[None, None]))
                        child_scene = scene.with_release_history(p, r, progress)
                    frames.append(search_frame(goal, child_scene, boundaries[stage+2]))
                    continue
                whole_path = np.concatenate((frames[0]["path"], *(item["path"][1:] for item in frames[1:])))
                # Local angular goal screening cannot prove winding. Recheck
                # the ordered full path with its actual start and all goal
                # queries before using it as a complete spatial initializer.
                with torch.no_grad():
                    positions, rotations, progress_values = [], [], []
                    for chunk in np.array_split(whole_path, max(1, int(np.ceil(len(whole_path)/32)))):
                        _, _, p, r, progress = state_parts(self.device_cfg.to_device(chunk[None]))
                        positions.append(p.clone()); rotations.append(r.clone()); progress_values.append(progress.clone())
                    full_progress = torch.cat(progress_values, 1)
                    q = self.device_cfg.to_device(whole_path[None, :, :-1])
                    process = self.problem.evaluate(torch.cat(positions, 1), torch.cat(rotations, 1),
                        full_progress, q, self.joint_names, joint_velocity=torch.zeros_like(q),
                        trajectory_time=self.device_cfg.to_device(whole_path[None, :, -1])*self.formal_clock["duration"],
                        direction_mode="path")
                    if (not bool(torch.isfinite(process).all()) or not bool((process <= 0).all())
                            or self.problem.verification_issues(full_progress)):
                        evidence["failure"] = "complete_path_process_or_history_rejected"
                        continue
                # BEGIN OMPL point samples also require the native representation map.
                controls, fit = self.physical_domain.fit_reference(whole_path[:, -1],
                    self.device_cfg.to_device(whole_path[:, :-1]), prior_controls=initial_seed[0, 0])
                if controls is None:
                    evidence.update(failure="native_cubic_fit_failed", fit=fit)
                    continue
                report.update(success=True, samples=len(whole_path), state_queries=work["state_queries"],
                              cubic_fit=fit,
                              # BEGIN coarse wall time includes boundary and path work.
                              elapsed_seconds=time.monotonic()-search_started)
                # END timing is diagnostic and does not grant additional work.
                return controls[None, None].contiguous(), report
                # END fitted curves still need full trajectory optimization/verification.
        except TimeoutError:
            report["budget_exhausted"] = True
        report["state_queries"] = work["state_queries"]
        # BEGIN unsuccessful searches also expose actual shared-budget consumption.
        report["elapsed_seconds"] = time.monotonic()-search_started
        # END an exhausted attempt cap may still leave unused wall time.
        report["failure"] = "no_complete_time_state_seed_within_budget"
        return None, report
    # END OMPL has no schedule replacement return value or side effect.

    # BEGIN bounded true-goal proposals retain static evidence before continuity.
    def _functional_goal(self, target):
        """Express the declared nominal material pose in native link coordinates.

        Native requests require a pose buffer. The constrained route evaluates
        the event's continuous allowed set in its cost manager; this nominal
        pose only supplies the native request's buffer and frame layout.
        """
        from scipy.spatial.transform import Rotation
        # BEGIN apply the complete material-to-TCP and calibration transforms.
        # Native layout is [batch,horizon,tool,goalset,component].
        poses = target.target[None] @ np.linalg.inv(target.tool_from_function)
        position = self.device_cfg.to_device(poses[:, :3, 3][None])
        quaternion = self.device_cfg.to_device(
            Rotation.from_matrix(poses[:, :3, :3]).as_quat()[:, [3, 0, 1, 2]][None])
        position, quaternion = self.problem.reference_to_link_pose(position, quaternion)
        return self.GoalToolPose(tool_frames=[self.problem.tool_link],
            position=position[:, None, None].contiguous(),
            quaternion=quaternion[:, None, None].contiguous())
        # END the buffer preserves the declaration's complete frame relation.

    def _rank_ik_candidates(self, candidates, success, previous, u, goal_ids=()):
        """Rank bounded IK proposals with the same online geometry as trajectory metrics."""
        from .planning_ik import rank_ik_candidates
        values, flags = np.asarray(candidates), np.asarray(success, bool)
        phase = float(np.interp(u, self.progress_map.u, self.progress_map.s))
        reports = []
        # BEGIN static IK qualification uses the same discrete collision model.
        # END this local scope retains no evidence across a different query/scene.
        for index, row in enumerate(values):
            report = {"native_pose_ik_success": bool(flags[index]), "candidate_source": "native_ik",
                      "task_satisfied": False, "model_valid": False}
            reports.append(report)
            if not np.isfinite(row).all():
                # BEGIN an invalid proposal reports its actual missing input.
                report["invalid_joint_indices"] = np.flatnonzero(~np.isfinite(row)).tolist()
                # END non-finite joints never enter FK or acquire scene evidence.
                continue
            # BEGIN failed reference IK rows retain numerical proposal status.
            # Their nominal failure skips this static query. Legal finite
            # rows can still initialize a curve; event allowed-set proposals
            # receive their own task/model query before entering search.
            if not goal_ids and not flags[index]:
                report["static_model_query"] = "not_run_native_reference_ik_failed"
                continue
            report["static_model_query"] = "evaluated"
            # END source reference failures remain in the diagnostic inventory.
            with self.torch.no_grad():
                q = self.device_cfg.to_device(row[None, None]).contiguous()
                zero = self.torch.zeros_like(q)
                js = self.JointState(position=q, velocity=zero, acceleration=zero, jerk=zero,
                    dt=self.device_cfg.to_device([self.budget.interpolation_dt]), joint_names=list(self.joint_names))
                state = self._verification_fk.compute_kinematics(js)
                index = state.tool_poses.tool_frames.index(self.problem.tool_link)
                p, r = self.problem.transform_tool_pose(state.tool_poses.position[:, :, index],
                                                       state.tool_poses.quaternion[:, :, index])
                progress, times = q.new_full(q.shape[:2], phase), q.new_full(q.shape[:2], u*self.formal_clock["duration"])
                task = self.problem.residuals(p, r, progress, q, self.joint_names,
                    # BEGIN seed qualification uses the same task set as curve delivery.
                    joint_velocity=zero, trajectory_time=times, point_goal_ids=goal_ids,
                    direction_mode="point")
                    # END nominal support preferences cannot become static rejection rules.
                task = {name: value for name, value in task.items() if not name.startswith("stop:")}
                # BEGIN reuse the existing static query for bounded rejection reasons.
                scene_components = {}
                # END no second geometry evaluation or new ranking criterion is added.
                online = _scene_boundary_evaluate(self.scene, self.problem, self.progress_map,
                    self.problem.transform_robot_spheres(state.robot_spheres), p, r, progress, q,
                    self.joint_names, trajectory_time=times, normalized_time=q.new_full(q.shape[:2], u),
                    component_diagnostics=scene_components)
                native = self._RobotState(joint_state=js, cuda_robot_model_state=state)
                constraints = self._verification_native.compute_costs(native)
                model = {"scene": online, **dict(zip(constraints.names, constraints.values))}
                report.update(task_satisfied=_ompl_terms_accepted(task), model_valid=_ompl_terms_accepted(model))
                # BEGIN preserve positive/non-finite residuals in their original families.
                def failures(terms):
                    maxima = {}
                    for name, value in terms.items():
                        flat = value.detach().reshape(-1)
                        index = self.torch.where(self.torch.isfinite(flat), flat,
                            self.torch.full_like(flat, float("inf"))).argmax()
                        maxima[name] = float(flat[index].item())
                    return {name: value for name, value in maxima.items() if value > 0 or not np.isfinite(value)}
                report.update(task_violations=failures(task),
                    native_violations=failures(dict(zip(constraints.names, constraints.values))),
                    scene_components=scene_components)
                # END conditional scene entries retain their names and have no veto.
        ranked = rank_ik_candidates(values, reports, np.asarray(previous), count=self.budget.ik_seeds,
            joint_tolerance=self.problem.numerical.joint, query_kind="functional" if goal_ids else "reference")
        return ranked, reports

    def _build_goal_ik_candidates(self, reference_seeds, *, check_search):
        """Solve each actual event's allowed set with the existing native IK.

        All exact same-time pose, coordinate and support declarations enter
        the shared point objective. A nominal pose only fills the native goal
        buffer while its tracking cost is disabled. The returned states retain
        independent task and scene qualification before curve construction.
        """
        self.goal_ik_candidates = {}
        controls = reference_seeds[0].detach().cpu().numpy()
        # BEGIN enumerate event data rather than a particular target geometry.
        events = {}
        for name in ("pose_events", "coordinate_events", "supports"):
            for index, target in enumerate(getattr(self.problem, name)):
                if self.problem.start_progress <= target.progress <= self.problem.end_progress:
                    events.setdefault(target.progress, []).append((name, index))
        for progress, identities in sorted(events.items()):
            check_search()
            goal_ids = tuple(identities)
            u = float(self.progress_map.anchor_parameters([progress])[0, 0])
            previous = self.physical_domain.evaluate(controls, np.asarray([u]))[:, 0]
            self.seed_parameter.fill_(u)
            pose = next((self.problem.pose_events[index] for name, index in goal_ids
                         if name == "pose_events"), None)
            native_goal = self._goal(u*self.formal_clock["duration"]) if pose is None else self._functional_goal(pose)
            self._manager_type.bind_ik_goals(goal_ids)
            self.ik.disable_tool_pose_tracking()
            try:
                result = self._solve_ik(native_goal,
                    self.device_cfg.to_device(previous[None]).contiguous())
            finally:
                self._manager_type.bind_ik_goals(())
                self.ik.enable_tool_pose_tracking()
            check_search()
            candidates = self.ik.get_active_js(result.js_solution).position[0].detach().cpu().numpy()
            success = result.success[0].detach().cpu().numpy()
            # An allowed-set solve does not claim convergence to a pose centre.
            selected, reports = self._rank_ik_candidates(candidates, np.zeros_like(success), previous, u, goal_ids)
            for index, report in enumerate(reports):
                report.update(candidate_source="event_allowed_set_ik", allowed_set_ik_success=bool(success[index]))
            self.goal_ik_candidates[u] = {"progress": float(progress), "u": u,
                "time_s": u*self.formal_clock["duration"], "goal_ids": [list(value) for value in goal_ids],
                "candidates": [{"joint_position": candidates[index].tolist(), "report": reports[index]}
                               for index in selected],
                "scope": "Task allowed-set IK states; scene qualification and complete curve validation remain explicit"}
        # END a finite candidate pool reports its coverage without changing goals.
        check_search()
        return self.goal_ik_candidates
    # END static candidates cannot manufacture release history or trajectory feasibility.

    # BEGIN independently qualify bounded seed and exploration observations.


    def _observed_native_result(self, actions):
        """Evaluate a complete reference seed through native success metrics once."""
        count = actions.shape[0]
        with self.torch.no_grad():
            metrics = self.trajopt.metrics_rollout.compute_metrics_from_action(actions)
            joint = metrics.state.joint_state
            joint.joint_names = self.trajopt.joint_names
            joint.knot, joint.knot_dt = metrics.actions, joint.dt*self.trajopt.interpolation_steps
            interpolated, trajectory, last = self.trajopt._interpolate_and_compute_metrics(joint)
            result = self.trajopt._get_result(actions, metrics, interpolated, trajectory, last, count, count)
            return self.trajopt._get_best_result(result, count)
            # END normal dense completion keeps only the two bounded batch slots.



    def _optimizer_reports(self, solution, initial):
        core_reports = [optimizer._core.solve_diagnostics for optimizer in self.trajopt.core.optimizers]
        reports = []
        for index in range(solution.shape[1]):
            displacement = (solution[0, index]-initial[0, index]).abs()
            reports.append({"seed_index": index,
                "unchanged_from_seed": bool((displacement == 0).all().item()),
                "maximum_control_update": float(displacement.max().item()),
                "cores": [rows[index] for rows in core_reports]})
        return reports
    # END functional screening cannot itself authorize execution.

    # BEGIN collision recovery changes sample coverage within its existing solve budget.
    def promote_collision_witnesses(self, result):
        """Rebind every manager's exact queries before the one recovery solve."""
        if self._trajectory_configuration["use_cuda_graph"]:
            raise MotionPlanningError("unsupported_model", "Collision witness promotion requires eager native rollouts")
        pairs = self._manager_type.query_anchor_parameters()
        report = _dense_collision_witnesses(result,
            regular_horizon=result.js_solution.position.shape[-2], anchor_parameters=pairs[:, 0],
            capacity=self.budget.max_task_anchors-len(pairs))
        added = self._manager_type.add_collision_witnesses(report["parameters"])
        report.update(changed=bool(len(added)), added_count=len(added),
                      anchor_count=len(self._manager_type.query_anchor_parameters()))
        return report

    # END functional multipliers are owned by each native solve's inner blocks.

    def _native_request(self, seed):
        """Bind the typed task constraints and continuous ordinary reference cost."""
        from curobo._src.solver.solve_mode import SolveMode
        from curobo._src.solver.solve_state import SolveState
        # BEGIN the transferred path supplies soft tracking; typed events own hard goals.
        self.route = "constrained"
        self.trajopt.disable_tool_pose_tracking()
        self.trajopt.disable_joint_position_tracking()
        # END no second terminal pose objective changes the reference definition.
        state = SolveState(solve_type=SolveMode.BATCH, num_trajopt_seeds=seed.shape[1],
                          batch_size=1, num_envs=1, num_goalset=1, tool_frames=[])
        return state, None

    def evaluate_reference(self, seed):
        """Qualify initial controls before the request's main reference improvement."""
        self.physical_domain.validate(seed)
        state, goal = self._native_request(seed)
        endpoint = self.JointState.from_position(seed[:, :, -1].contiguous())
        endpoint.dt = self.torch.full((1, seed.shape[1]), self.budget.maximum_trajectory_dt,
                                      dtype=seed.dtype, device=seed.device)
        buffer, _ = self.trajopt._prepare_goal_buffer(state, goal, self.start, False, endpoint)
        self.trajopt._update_rollout_params(buffer)
        self.last_optimizer_diagnostics = [{"unchanged_from_seed": True, "maximum_control_update": 0.,
                                           "source": "reference_evaluation"} for _ in range(seed.shape[1])]
        return self._observed_native_result(seed.reshape(-1, self.trajopt.action_horizon, self.trajopt.action_dim))

    # BEGIN caller and solver curves share one metric/domain delivery boundary.
    def bind_reference_candidate(self, reference):
        """Bind current obligations and hand commands without changing q(t)."""
        if (not isinstance(reference, JointTrajectory)
                or set(reference.joint_names) != set(self.joint_names)
                or len(set(reference.joint_names)) != len(reference.joint_names)
                or reference.times[0] != 0 or reference.duration <= 0):
            raise MotionPlanningError("unsupported_reference", "Reference needs the complete named joints and a zero-origin clock")
        mapping = _ProgressMap(reference.progress_u.copy(), reference.progress_s.copy())
        if mapping.s[0] != self.problem.start_progress or mapping.s[-1] != self.problem.end_progress:
            raise MotionPlanningError("unsupported_reference", "Reference progress must cover this complete requested skill window")
        candidate_scene = self.scene.clone_for_rollout()
        hand = candidate_scene.gripper
        issues = _reference_clock_issues(self.problem, reference, None if hand is None else hand.control_timestep_s)
        if issues:
            raise MotionPlanningError("reference_clock_outside_curve", str(issues))
        mapping = _bind_event_clock(self.problem, mapping, reference.duration,
                                   None if hand is None else hand.control_timestep_s)
        if hand is not None:
            hand.bind_clock(reference.duration, mapping)
        events = tuple({**deepcopy(record["event"]),
                        **{key: deepcopy(value) for key, value in record.items() if key != "event"}}
                       for record in mapping.event_clock.values())
        trajectory = JointTrajectory(reference.joint_names, reference.times.copy(), reference.position.copy(),
            reference.velocity.copy(), mapping.u.copy(), mapping.s.copy(), events,
            deepcopy(self.problem.conditions), {}, None if hand is None else hand.finger_trajectory,
            tuple(getattr(self, "satisfied_dependencies", ())))
        return trajectory, candidate_scene, mapping

    # BEGIN all delivered candidates are ranked on one request-wide quadrature grid.
    def _reference_for_candidate(self, trajectory):
        """Bind the original reference inputs to this candidate's actual clock.

        A supplied joint trajectory retains its own time/progress map. Its
        reference is constructed from the same nominal skill and measured
        entrance, without retiming the candidate or carrying another clock's
        correction coefficients into its ordinary objective.
        """
        from .timed_reference import BoundToolReference
        return BoundToolReference(self.problem, trajectory.progress_u, trajectory.progress_s,
            trajectory.duration, self.bound_reference.initial_pose, self.bound_reference.initial_twist)

    def _candidate_objective(self, trajectory):
        """Evaluate fixed ordinary costs using this curve's physical derivatives.

        Normalized nodes are fixed by the admitted native rollout, while each
        curve retains its own duration and progress map. FK is the only model
        query; scene constraints and event anchors have already defined their
        independent qualification roles.
        """
        mapping = _ProgressMap(trajectory.progress_u, trajectory.progress_s)
        times = np.linspace(0., trajectory.duration, (self.budget.knots+4)*self.budget.interpolation_steps+1)
        q, dq, ddq, _ = trajectory.sample(times)
        columns = [trajectory.joint_names.index(name) for name in self.joint_names]
        tensor = self.device_cfg.to_device
        js = self.JointState(position=tensor(q[:, columns])[None].contiguous(),
            velocity=tensor(dq[:, columns])[None].contiguous(),
            acceleration=tensor(ddq[:, columns])[None].contiguous(),
            jerk=tensor(trajectory._curve(times, 3)[:, columns])[None].contiguous(),
            dt=tensor([self.formal_clock["native_dt"]]), joint_names=list(self.joint_names))
        rollout = self.trajopt.additional_metrics_rollouts["interpolated_rollout"]
        manager = next(item for item in rollout._cost_manager_list
                       if getattr(item, "_process_mode", "") == "reference_objective")
        manager._candidate_query = {"times": tensor(times)[None], "mapping": mapping,
                                    "duration": trajectory.duration,
                                    "bound_reference": self._reference_for_candidate(trajectory)}
        manager._reference_grid_key = None
        try:
            self.check_request()
            with self.torch.no_grad():
                state = rollout.transition_model.compute_augmented_state(js)
                costs = manager.compute_costs(state)
                components = {name: float(value.sum().item()) for name, value in zip(costs.names, costs.values)}
            self.check_request()
        finally:
            manager._candidate_query = None
            manager._reference_grid_key = None
        return {"fixed_objective": sum(components.values()), "objective_components": components}
    # END finite quadrature supplies one comparable definition, not exact integration.

    def _candidate_metrics(self, trajectory, candidate_scene, *, record_work):
        """Run existing native metrics on exact, possibly nonuniform queries.

        The installed rollout does not forward query kwargs to its managers.
        A short-lived manager context therefore carries actual times and exact
        source-curve anchors; finally clears it even when native evaluation
        raises. All states query the same observed fields at the acceptance
        spacing; task and observation evidence remain tied to that query.
        """
        from curobo._src.solver.solve_mode import SolveMode
        from curobo._src.solver.solve_state import SolveState
        # BEGIN use the candidate clock and its explicitly bound relation model.
        candidate_scene = self.scene if candidate_scene is None else candidate_scene
        # END observed fields remain shared with the bound planning scene.
        mapping = _ProgressMap(trajectory.progress_u, trajectory.progress_s)
        pairs = mapping.anchor_parameters(self.anchors)
        # BEGIN a supplied curve uses its own physical grid and task anchors.
        times = _candidate_sample_times(trajectory, self.budget)
        # END every candidate is queried from the saved polynomial.
        if (len(pairs) > self.budget.max_task_anchors
                or len(times)+len(pairs) > min(self.budget.max_validation_points, self.budget.interpolation_buffer_size)):
            raise MotionPlanningError("capacity_exceeded", "Candidate samples and exact anchors exceed the admitted query budget")
        columns = [trajectory.joint_names.index(name) for name in self.joint_names]
        q, dq, ddq, progress = trajectory.sample(times)
        anchor_q, anchor_dq, _, _ = trajectory.sample(pairs[:, 0]*trajectory.duration)
        tensor = self.device_cfg.to_device
        context = {"scene": candidate_scene, "mapping": mapping, "duration": trajectory.duration,
            "bound_reference": self._reference_for_candidate(trajectory),
            "times": tensor(times)[None], "progress": tensor(progress)[None],
            "anchor_u": tensor(pairs[:, 0])[None], "anchor_s": tensor(pairs[:, 1])[None],
            # BEGIN actual anchor q/dq and time share the original double pair.
            "anchor_times": tensor(pairs[:, 0]*trajectory.duration)[None],
            # END normalized float32 keys never reconstruct physical time.
            "anchor_q": tensor(anchor_q[:, columns])[None].contiguous(),
            "anchor_dq": tensor(anchor_dq[:, columns])[None].contiguous(),
            "scene_components": {}}
        # The task manager owns all real pose requirements. These inert goal
        # slots only initialize the same resident native rollout buffers.
        self.trajopt.disable_tool_pose_tracking()
        self.trajopt.disable_joint_position_tracking()
        state = SolveState(solve_type=SolveMode.BATCH, num_trajopt_seeds=1,
                           batch_size=1, num_envs=1, num_goalset=1, tool_frames=[])
        # BEGIN native seed goals retain both problem and candidate dimensions.
        endpoint = self.JointState.from_position(tensor(q[-1:, columns])[None].contiguous())
        endpoint.dt = tensor([[self.budget.maximum_trajectory_dt]])
        # END GoalRegistry requires [batch, seeds, joints], even for one curve.
        buffer, _ = self.trajopt._prepare_goal_buffer(state, None, self.start, False, endpoint)
        self.trajopt._update_rollout_params(buffer)
        rollout = self.trajopt.additional_metrics_rollouts["interpolated_rollout"]
        managers = [manager for manager in rollout._cost_manager_list
                    if getattr(manager, "_process_mode", "")]
        if rollout.metrics_constraint_manager not in managers:
            raise MotionPlanningError("unsupported_model", "Candidate metrics require the bound process/scene constraint manager")
        if any(getattr(manager, "_candidate_query", None) is not None for manager in managers):
            raise RuntimeError("Candidate metric context cannot overlap another query")
        # BEGIN column reordering must not send Fortran-strided arrays to CUDA/Warp.
        js = self.JointState(position=tensor(q[:, columns])[None].contiguous(),
            velocity=tensor(dq[:, columns])[None].contiguous(),
            acceleration=tensor(ddq[:, columns])[None].contiguous(),
            jerk=tensor(trajectory._curve(times, 3)[:, columns])[None].contiguous(),
            dt=tensor([self.budget.interpolation_dt]), joint_names=list(self.joint_names))
        # END native kernels flatten physical q/dq/ddq/jerk without copying.
        try:
            for manager in managers:
                manager._candidate_query = context
                if manager._process_mode == "reference_objective":
                    manager._reference_grid_key = None
            self.check_request()
            with self.torch.no_grad():
                native = rollout.transition_model.compute_augmented_state(js)
                metrics = rollout.compute_metrics_from_state(native)
            # BEGIN charge completed native queries before any deadline can raise.
            record_work(points=len(times)+len(pairs))
            self.check_request()
            # END this completed work survives an interrupted candidate verdict.
            constraints = metrics.costs_and_constraints.constraints
            # BEGIN exact-curve metrics retain the same query's collision evidence.
            rollout.metrics_constraint_manager.bind_scene_evidence(metrics)
            collision_residuals = _collision_metric_residuals(metrics)
            # BEGIN confirmed metrics retain support gaps from that same query.
            scene_update_residuals = _scene_update_metric_residuals(metrics)
            # BEGIN entity collisions and observation coverage retain separate evidence.
            coverage_residuals = _coverage_metric_residuals(metrics)
            conditional_unknown_residuals = _conditional_unknown_metric_residuals(metrics)
            # END all categories belong to these same completed candidate queries.
            # BEGIN a nonfinite sample remains visible beside finite neighbours.
            def maximum(value):
                flat = value.detach().reshape(-1)
                index = self.torch.where(self.torch.isfinite(flat), flat,
                    self.torch.full_like(flat, float('inf'))).argmax()
                return float(flat[index].item())
            residuals = {f"interpolated:{name}": maximum(value)
                         for name, value in zip(constraints.names, constraints.values)}
            # END neither another channel's clear nor a maximum hides NaN/-Inf.
            feasible = bool(metrics.feasible) if isinstance(metrics.feasible, bool) else bool(metrics.feasible.all().item())
            # BEGIN native feasibility flags cannot hide an invalid raw constraint.
            feasible = feasible and all(np.isfinite(value) and value <= 0 for value in residuals.values())
        finally:
            for manager in managers:
                manager._candidate_query = None
                if manager._process_mode == "reference_objective":
                    manager._reference_grid_key = None
        report = {"native_success": feasible, "accepted": feasible,
            "residuals": residuals, "seed_provenance": "supplied_joint_curve",
            # BEGIN classification is diagnostic-only, outside hard residuals.
            "collision_residuals": {f"interpolated:{name}": value for name, value in collision_residuals.items()},
            # BEGIN expose candidate-scoped support evidence beside collision evidence.
            "scene_update_residuals": {f"interpolated:{name}": value for name, value in scene_update_residuals.items()},
            # BEGIN retain observation categories for the existing bounded search.
            "coverage_residuals": {f"interpolated:{name}": value for name, value in coverage_residuals.items()},
            "conditional_unknown_residuals": {f"interpolated:{name}": value
                for name, value in conditional_unknown_residuals.items()},
            # END coverage is never interpreted as an entity intersection.
            # END support evidence does not claim every route lacks support.
            # END the original metrics determine feasibility and severity.
            "native_metrics": {"success": feasible, "interpolated": {"constraints": residuals}},
            "scene_components": context["scene_components"],
            # BEGIN retain the actual registered-entity query classification.
            "observed_geometry": deepcopy(candidate_scene.last_query_report),
            # END the report distinguishes intersections, separation and unresolved pairs.
            "optimizer": {"unchanged_from_seed": True, "maximum_control_update": 0.}}
        return report, times/trajectory.duration, len(times)+len(pairs)

    # BEGIN one exact-curve scene callback supplies interval proof and hard evidence.
    def _candidate_geometry(self, trajectory, candidate_scene, *, remaining_points, record_work):
        """Check continuous entity separation on the candidate's physical clock.

        Scene queries evaluate the original polynomial and its release anchors.
        The motion bound only determines where further geometry is needed;
        collision, contact and observation residuals retain scene semantics.
        """
        from .interval_geometry import SceneMotionBounds, check_interval_geometry
        mapping = _ProgressMap(trajectory.progress_u, trajectory.progress_s)
        motion = SceneMotionBounds.from_scene(candidate_scene, tool_link=self.problem.tool_link,
                                              T_link_tool=self.problem.tool_transform)
        columns = [trajectory.joint_names.index(name) for name in self.joint_names]
        release_progress = [relation.interval[1] for owner in candidate_scene.owners
                            for relation in owner.attachments if relation.release_condition_id is not None
                            and self.problem.start_progress <= relation.interval[1] <= self.problem.end_progress]
        release_times = (mapping.anchor_parameters(release_progress)[:, 0]*trajectory.duration
                         if release_progress else np.empty(0))
        hard, categories, details = [], {}, {}
        # BEGIN one immutable candidate owns one set of modeled release anchors.
        released_scene = candidate_scene if not len(release_times) else None

        def query(times_s, *, point_budget):
            nonlocal released_scene
            times = np.unique(np.r_[np.asarray(times_s, float),
                release_times if released_scene is None else np.empty(0)])
            if len(times) > point_budget:
                return {"states": [], "query_points": 0, "capacity_exhausted": True}
            self.check_request()
            q, dq, _, progress = trajectory.sample(times)
            tensor = self.device_cfg.to_device
            with self.torch.no_grad():
                js = self.JointState(position=tensor(q[:, columns])[None].contiguous(),
                    velocity=tensor(dq[:, columns])[None].contiguous(), joint_names=list(self.joint_names))
                state = self._verification_fk.compute_kinematics(js)
                native = self._RobotState(joint_state=js, cuda_robot_model_state=state)
                tool = state.tool_poses.tool_frames.index(self.problem.tool_link)
                position, quaternion = self.problem.transform_tool_pose(
                    state.tool_poses.position[:, :, tool], state.tool_poses.quaternion[:, :, tool])
                evidence, scene_channels, query_details = [], {}, {}
                active_scene = candidate_scene if released_scene is None else released_scene
                values = _scene_boundary_evaluate(active_scene, self.problem, mapping,
                    self.problem.transform_robot_spheres(native.robot_spheres), position, quaternion,
                    tensor(progress)[None], js.position, self.joint_names,
                    trajectory_time=tensor(times)[None], normalized_time=tensor(times/trajectory.duration)[None],
                    component_diagnostics=query_details, scene_evidence_out=scene_channels,
                    geometry_evidence_out=evidence)
                # BEGIN count each completed scene query exactly once.
                record_work(points=len(times))
                # END counts remain committed if a later batch is interrupted.
                if released_scene is None:
                    released_scene = candidate_scene.with_release_history(position, quaternion, tensor(progress)[None])
                # END subsequent samples reuse anchors from this same exact curve.
                # Keep the most severe original component and its own time;
                # a later smaller violation must not replace that witness.
                for name, entry in query_details.items():
                    previous = details.get(name)
                    if previous is None or (np.isfinite(previous["maximum"]) and (
                            not np.isfinite(entry["maximum"]) or entry["maximum"] > previous["maximum"])):
                        details[name] = entry
                values = values.detach().cpu().numpy()[0]
                hard.extend(values.tolist())
                for name, value in scene_channels.items():
                    numbers = value.detach().cpu().numpy().ravel()
                    previous = categories.get(name, 0.)
                    categories[name] = (float("inf") if not np.isfinite(numbers).all()
                                        else max(previous, float(numbers.max())))
            states = {}
            for row in evidence:
                index = row["query_index"][1]
                key = (float(times[index]), row["side"])
                pair = {name: value for name, value in row.items()
                        if name not in {"state_index", "query_index", "time_s", "side"}}
                state = states.setdefault(key, {"time_s": key[0], "side": key[1],
                    "progress": float(progress[index]), "residuals": {"scene": float(values[index])}, "pairs": []})
                state["pairs"].append(pair)
            self.check_request()
            return {"states": list(states.values()), "query_points": len(times), "capacity_exhausted": False}

        report = check_interval_geometry(trajectory, motion, query,
            max_query_points=max(0, remaining_points), spatial_step_m=self.budget.validation_spatial_step_m)
        report["hard_scene_residual"] = (float("inf") if not np.isfinite(hard).all()
                                         else max(hard, default=0.))
        report["scene_categories"] = categories
        report["scene_components"] = details
        return report
    # END unresolved intervals report their scope separately from actual scene violations.

    # BEGIN one saved polynomial determines every candidate qualification.
    def evaluate_candidate(self, trajectory, *, record_work, candidate_scene=None, objective=None):
        """Query the saved curve, its exact joint domain and interval geometry.

        ``record_work`` commits started validation attempts and completed query
        points to the request ledger. ``objective`` may contain the fixed score
        already computed for this same curve during native candidate ranking.
        """
        record_work(passes=1)
        self.delivery_candidate, self.last_validation = trajectory, None
        scene = self.scene if candidate_scene is None else candidate_scene
        attempt, parameters, points = self._candidate_metrics(trajectory, scene, record_work=record_work)
        validation = self._candidate_domain_validation(trajectory)
        residuals = {**attempt["residuals"], **trajectory.residuals}
        accepted = bool(attempt["accepted"]) and all(
            np.isfinite(value) and value <= 0 for value in residuals.values())
        if accepted:
            geometry = self._candidate_geometry(trajectory, scene,
                remaining_points=self.budget.max_validation_points-points, record_work=record_work)
            points += geometry["query_points"]
            attempt["continuous_geometry"] = geometry
            validation["continuous_geometry"] = geometry
            residuals["interval:process_scene"] = geometry["hard_scene_residual"]
            categories = geometry["scene_categories"]
            for target, category, component in (
                    ("collision_residuals", "collision", "scene_collision"),
                    ("scene_update_residuals", "scene_update", "scene_update"),
                    ("coverage_residuals", "coverage", "unknown_forbidden"),
                    ("conditional_unknown_residuals", "conditional_unknown", "conditional_unknown")):
                attempt.setdefault(target, {})[f"interval:{component}"] = categories.get(category, 0.)
            attempt.setdefault("scene_components", {}).update(geometry["scene_components"])
            accepted = all(np.isfinite(value) and value <= 0 for value in residuals.values())
        attempt.update(self._candidate_objective(trajectory) if objective is None else objective)
        accepted = accepted and bool(np.isfinite(attempt["fixed_objective"]))
        attempt.update(residuals=residuals, accepted=accepted)
        validation.update(u=parameters, violations=np.full(parameters.shape,
            max(residuals.values(), default=0.)), scope="saved polynomial task metrics, exact joint domain and reported geometry intervals")
        self.last_validation = validation if accepted else None
        return trajectory, attempt, points
    # END optimizer sampling grids supply provenance, not a separate verdict.

    def optimize(self, seed, *, iterations=None):
        """One native solve with the current physical domain and unchanged clock."""
        self.physical_domain.validate(seed)
        state, goal = self._native_request(seed)
        initial = seed.detach().clone()
        try:
            result = self.trajopt._solve_impl(current_state=self.start, solve_state=state,
                goal_tool_poses=goal, seed_traj=seed, num_seeds=seed.shape[1], return_seeds=seed.shape[1],
                dt=self.torch.full((1, seed.shape[1]), self.budget.maximum_trajectory_dt,
                                   dtype=seed.dtype, device=seed.device),
                use_implicit_goal=False, goal_state=None, finetune_attempts=0, finetune_dt_scale=1.,
                initial_iters=self.budget.trajectory_iterations if iterations is None else iterations)
        finally:
            # Every request/recovery starts from its original immutable quota.
            self.trajopt.optimizer.update_niters(self.budget.trajectory_iterations)
        if result.solution.shape != initial.shape:
            raise ValueError("Native solve changed the admitted seed layout")
        self.last_optimizer_diagnostics = self._optimizer_reports(result.solution, initial)
        return result
        # END unchanged output is separate from movement of native exploration states.
    # END intermediate and terminal goals share the same typed requirement owner.

    def native_diagnostics(self, result, seed_index=None):
        """Report the native success flag and original online-model residuals."""
        success = result.success if seed_index is None else result.success[:, seed_index]
        report = {"success": bool(success.all().item())}
        for label, metrics in (("rollout", result.metrics), ("interpolated", result.interpolated_metrics)):
            if metrics is None:
                continue
            def maximum(value):
                # BEGIN every native seed retains its own nonfinite hard samples.
                selected = value if seed_index is None else value[seed_index]
                flat = selected.detach().reshape(-1)
                import torch
                # BEGIN native convergence also carries integer IDs/bool flags.
                # Those dtypes have no NaN/Inf and cannot store an infinity
                # sentinel. Reduce in their original dtype before reporting;
                # keep the existing nonfinite witness rule for floating costs.
                if not flat.is_floating_point():
                    return float(flat.max().item())
                # END no float sentinel is constructed for discrete metrics.
                index = torch.where(torch.isfinite(flat), flat,
                                    torch.full_like(flat, float('inf'))).argmax()
                return float(flat[index].item())
            constraints = metrics.costs_and_constraints.constraints
            # BEGIN report completed ordinary costs without another rollout query.
            ordinary = metrics.costs_and_constraints.costs
            ordinary_costs = {name: float((value if seed_index is None else value[seed_index]).detach().sum().item())
                              for name, value in zip(ordinary.names, ordinary.values)}
            # END these per-grid integrals are diagnostics; canonical scores rank candidates.
            report[label] = {"ordinary_costs": ordinary_costs,
                "constraints": {name: maximum(value) for name, value in
                zip(constraints.names, constraints.values)}, "convergence": {name: maximum(value)
                for name, value in zip(metrics.convergence.names, metrics.convergence.values)},
                # BEGIN typed collision evidence is separate from native constraints.
                "collision_residuals": _collision_metric_residuals(metrics, seed_index),
                # BEGIN both native sampling grids retain the same seed's support gaps.
                "scene_update_residuals": _scene_update_metric_residuals(metrics, seed_index),
                # BEGIN both native grids retain the selected seed's coverage evidence.
                "coverage_residuals": _coverage_metric_residuals(metrics, seed_index),
                "conditional_unknown_residuals": _conditional_unknown_metric_residuals(metrics, seed_index)}
                # END this adds no geometry query or acceptance condition.
                # END no extra geometry query or request-level verdict is produced.
                # END diagnostics never affect the original severity calculation.
        return report

    # BEGIN fixed scores order saved curves; formal curve metrics determine eligibility.


    def select_candidate(self, result, *, record_work):
        """Rank finite saved curves, then qualify them on one common boundary."""
        # BEGIN capture every proposal before any exact-curve metric query.
        # Native metric evaluation reuses resident buffers. Candidate actions,
        # collision witnesses and diagnostic scalars must therefore be owned
        # before the first call to evaluate_candidate can overwrite them.
        captured, proposals, reports = [], [], []
        for index in range(result.num_seeds):
            metrics = self.native_diagnostics(result, index)
            owned = _trajectory_candidate(result, index)
            provenance = {"seed_index": index, "optimization_diagnostics": metrics,
                "seed_provenance": self.seed_strategies[index],
                "optimizer": deepcopy(self.last_optimizer_diagnostics[index]) if self.last_optimizer_diagnostics else {}}
            if not bool(owned.solution.isfinite().all().item()):
                reports.append({**provenance, "accepted": False, "residuals": {"finite_controls": float("inf")}})
                continue
            captured.append((index, owned, provenance))
        # Fixed scoring also uses resident FK/reference buffers. Start it only
        # after every seed's original state and evidence have been captured.
        for index, owned, provenance in captured:
            trajectory = self._native_curve(owned)
            objective = self._candidate_objective(trajectory)
            proposals.append((index, owned, trajectory, objective, provenance))
        if not proposals:
            raise MotionPlanningError("nonfinite_candidate", "Native solve returned no finite cubic controls")
        proposals.sort(key=lambda item: (
            item[3]["fixed_objective"] if np.isfinite(item[3]["fixed_objective"]) else float("inf"), item[0]))
        selected_record, selected_key, query_points, evaluated = None, None, 0, 0
        for index, owned, trajectory, objective, provenance in proposals:
            trajectory, report, points = self.evaluate_candidate(
                trajectory, record_work=record_work, objective=objective)
            report.update(provenance)
            query_points += points
            evaluated += 1
            reports.append(report)
            values = list(report["residuals"].values())
            severity = sum(max(0., value) for value in values) if np.isfinite(values).all() else float("inf")
            key = (not report["accepted"], report["fixed_objective"] if report["accepted"] else severity, index)
            if selected_key is None or key < selected_key:
                selected_record, selected_key = (index, owned, trajectory, report), key
            if report["accepted"]:
                break
        selected, owned, trajectory, report = selected_record
        self.delivery_candidate = trajectory
        return owned, trajectory, {**report, "selected_seed_index": selected,
            "candidates": reports, "validation_query_points": query_points,
            "evaluated_candidate_count": evaluated}
        # END the first qualified curve has the best finite fixed score.

    # BEGIN selected native coefficients own one binary64 saved polynomial.
    def _native_curve(self, result):
        """Copy the bound native cubic without independent geometry queries."""
        if result.batch_size != 1 or result.num_seeds != 1:
            raise ValueError("Export requires exactly one native batch and one selected seed")
        return self._controls_curve(result.solution[0, 0])

    def _controls_curve(self, actions):
        """Export one active-joint coefficient array with its original event clock."""
        basis = CubicBasis(self.trajopt.action_horizon, self.formal_clock["knot_dt"])
        times, position, velocity = basis.hermite_nodes(actions.detach().cpu().numpy(),
            self.start.position.detach().cpu().numpy()[0], self.start.velocity.detach().cpu().numpy()[0])
        if position.shape[1] != len(self.joint_names) or times[-1] != self.formal_clock["duration"]:
            raise ValueError("Native candidate changed the physical clock bound to the finger reference")
        event_records = []
        for event in self.problem.events:
            # BEGIN export the same flattened command record as the bound hand.
            clock = self.progress_map.event_clock[event_occurrence_key(event)]
            if clock["event"] != asdict(event):
                raise ValueError("Exported event differs from its frozen occurrence binding")
            event_records.append({**deepcopy(clock["event"]),
                                  **{key: deepcopy(value) for key, value in clock.items() if key != "event"}})
            # END the internal event payload is provenance, not a second command field.
        trajectory = JointTrajectory(self.joint_names, times, position, velocity,
                                     self.progress_map.u.copy(), self.progress_map.s.copy(),
                                     tuple(event_records), deepcopy(self.problem.conditions), {})
        trajectory.finger_trajectory = self.finger_trajectory
        trajectory.satisfied_dependencies = tuple(getattr(self, "satisfied_dependencies", ()))
        return trajectory
    # END native metric buffers are not a second saved-curve representation.

    def _candidate_domain_validation(self, trajectory):
        """Check the actual saved polynomial, independent of its proposal basis."""
        limits = self._verification_native.get_cost("cspace").config.joint_limits
        columns = [limits.joint_names.index(name) for name in trajectory.joint_names]
        bounds = limits.position[:, columns].detach().cpu().numpy()
        position = cubic_position_report(trajectory._curve, *bounds)
        velocity_bounds = limits.velocity[:, columns].detach().cpu().numpy()
        velocity = cubic_position_report(trajectory._curve.derivative(), *velocity_bounds)
        if velocity["maximum_violation"] is not None:
            velocity["maximum_violation"]["velocity"] = velocity["maximum_violation"].pop("position")
        derivatives = cubic_derivative_report(trajectory, limits.acceleration.detach().cpu().numpy(),
                                              joint_names=limits.joint_names)
        # BEGIN jerk is constant on each original polynomial segment.
        jerk = cubic_position_report(trajectory._curve.derivative(3),
            *limits.jerk[:, columns].detach().cpu().numpy())
        if jerk["maximum_violation"] is not None:
            jerk["maximum_violation"]["jerk"] = jerk["maximum_violation"].pop("position")
        jerk["scope"] = "piecewise third derivative; both adjacent segments at knots"
        # END continuous-domain validation uses the same four physical bounds.
        entrance_columns = [self.joint_names.index(name) for name in trajectory.joint_names]
        trajectory.residuals = {
            "continuous_position": position["maximum_bound_excess"],
            "continuous_velocity": velocity["maximum_bound_excess"],
            "continuous_acceleration": derivatives["acceleration"]["maximum_bound_excess"],
            "continuous_jerk": jerk["maximum_bound_excess"],
            "initial_position": max(0., float(np.max(np.abs(trajectory.position[0]-self.start.position[0].detach().cpu().numpy()[entrance_columns])))-self.problem.numerical.joint),
            "initial_velocity": max(0., float(np.max(np.abs(trajectory.velocity[0]-self.start.velocity[0].detach().cpu().numpy()[entrance_columns])))-self.problem.numerical.velocity)}
        return {
            "continuous_position": position, "continuous_velocity": velocity, "continuous_jerk": jerk, "cubic_derivatives": derivatives,
            "numerical_tolerances": asdict(self.problem.numerical)}

    def export(self, result, native_report):
        """Deliver the exact curve already qualified during candidate selection."""
        # BEGIN selection completes every query before the recovery decision.
        if not native_report["accepted"] or self.last_validation is None:
            raise MotionPlanningError("output_inconsistent", "Delivery requires a qualified selected candidate")
        return self.delivery_candidate, 0
        # END export performs no repeated geometry or polynomial computation.


# BEGIN one resident native backend with typed refusal and explicit ownership.
def _verified_dependency_outcomes(continuation):
    """Name only dependencies with the executed occurrence's fresh T record."""
    if continuation is None:
        return set()
    verified = {event_occurrence_key(row) for row in continuation.get("verified_events", ())}
    original = continuation["original_binding"]
    events = {event_occurrence_key(row): row for row in original["curve"]["events"]}
    satisfied = set()
    for row in continuation.get("dependency_outcomes", ()):
        event, evidence = row.get("event", {}), row.get("evidence", {})
        identity = event_occurrence_key(event)
        elapsed = row.get("physical_elapsed_s")
        if (identity in verified and identity in events and event == events[identity]
                and evidence.get("value") == "T" and evidence.get("source") and row.get("feedback_id")
                and isinstance(elapsed, (int, float)) and np.isfinite(elapsed)
                and 0 <= elapsed <= continuation["elapsed_physical_s"]):
            dependency = event.get("requires_evidence")
            if dependency is not None:
                satisfied.add(dependency)
    return satisfied


def _bind_motion_work_ledger(ledger, block, continuation, entrance_consumed):
    """Start another two-solve allowance only after a verified block boundary."""
    previous = ledger.get("motion_request")
    new_request = previous is None
    if continuation is not None and previous is not None:
        old = continuation["original_binding"].get("motion_block", {})
        event = old.get("boundary_event")
        verified = _verified_dependency_outcomes(continuation)
        new_request = bool(old.get("requires_rebind") and event is not None
            and event.get("requires_evidence") in verified
            and block["start_progress"] >= old["end_progress"]
            and block["end_progress"] > old["end_progress"])
    if new_request:
        ledger["motion_request"] = {"index": 0 if previous is None else previous["index"]+1,
            "start_progress": block["start_progress"], "end_progress": block["end_progress"],
            "planning_baseline": (dict.fromkeys(entrance_consumed, 0) if previous is None else deepcopy(entrance_consumed))}
    return ledger["motion_request"]["planning_baseline"]


class MotionPlanningError(ValueError):
    """A failed bounded request, never a proof that the mathematical problem is infeasible."""
    def __init__(self, reason, message):
        super().__init__(message)
        self.reason = reason


class MotionBackend:
    """Own one robot's reusable native IK and trajectory optimizer instances.

    The robot, device and tensor capacities are constructor inputs. Requests
    replace goals, measured q/dq, scene values and clocks; they cannot resize
    the robot or silently buy another solve. close() is the owner's lifecycle.
    """
    def __init__(self, robot, *, robot_validator, budget=None, device="cuda:0"):
        self.robot, self.robot_validator, self.device = deepcopy(robot), robot_validator, device
        self.budget = PlanningBudget() if budget is None else budget
        self._planner = None
        self._closed = False
        self._busy = Lock()
        self._request_owner = None

    def _claim(self):
        if not self._busy.acquire(blocking=False):
            raise MotionPlanningError("backend_busy", "A native request still owns this backend's buffers")
        self._request_owner = get_ident()

    def _release(self):
        if self._request_owner != get_ident():
            raise RuntimeError("Only the active motion request can release native buffers")
        self._request_owner = None
        self._busy.release()

    def bind_request(self, problem, scene, *, budget, continuation=None, goal_identities=None):
        if self._request_owner != get_ident():
            raise MotionPlanningError("backend_busy", "Request binding requires ownership of the native buffers")
        if self._closed:
            raise MotionPlanningError("backend_closed", "MotionBackend is closed")
        for name in ("knots", "interpolation_steps", "interpolation_dt", "interpolation_buffer_size",
                     "ik_seeds", "trajectory_seeds", "max_task_anchors", "ik_iterations", "trajectory_iterations"):
            if getattr(budget, name) != getattr(self.budget, name):
                raise MotionPlanningError("capacity_exceeded", f"Resident backend cannot change {name}")
        if self._planner is None:
            self._planner = _CuroboPlanner(problem, scene, self.robot, budget, self.device,
                robot_validator=self.robot_validator, continuation=continuation, goal_identities=goal_identities)
        else:
            self._planner.rebind(problem, scene, budget, continuation=continuation, goal_identities=goal_identities)
        return self._planner

    def plan(self, problem, scene, *, continuation=None, request_ledger=None, rebind=None,
             cancelled=None, deadline=None, joint_reference=None):
        return plan_trajectory(problem, scene, self.robot, robot_validator=self.robot_validator,
            budget=self.budget, device=self.device, continuation=continuation,
            request_ledger=request_ledger, rebind=rebind, backend=self,
            cancelled=cancelled, deadline=deadline, joint_reference=joint_reference)

    def close(self):
        self._claim()
        try:
            if self._planner is not None:
                self._planner.close()
                self._planner = None
            self._closed = True
        finally:
            self._release()
# END a failed request leaves the resident solvers available for the next binding.


# BEGIN request-wide execution, failure and rebind accounting.
def plan_trajectory(problem, scene, robot, *, robot_validator, budget=None, device="cuda:0", continuation=None,
                    request_ledger=None, rebind=None, backend=None, cancelled=None, deadline=None, joint_reference=None):
    """Generate one fully checked candidate within the original request budget.

    ``request_ledger`` continues failed planning attempts. ``continuation``
    carries actual execution facts. An explicit ``rebind`` rebuilds the soft
    schedule of remaining work or creates selected retry occurrences; ordinary
    continuation preserves the original frozen clock. Every request admitted
    including initialization failure carries the cumulative ledger. No GPU stage is
    claimed to be preemptible. A candidate qualifies only after all required
    work completes within budget; later improvement exhaustion preserves that
    independently stored delivery record.
    """
    from .execution_binding import bind_candidate, plain
    from .planning_problem import TimeBindingError
    started = time.monotonic()
    prior = planner = ledger = trajectory = None
    incumbent = None
    initial_seconds = 0.
    motion_ready, motion_baseline = False, {}
    owned_backend = backend is None
    backend_claimed = False
    diagnostics = {"attempts": []}
    # BEGIN passive request milestones distinguish lazy native initialization from solving.
    diagnostics["wallclock"] = {"scope": "One planning request; Unix timestamps join processes, elapsed seconds use the request monotonic clock", "events": {}}
    def mark_wallclock(name):
        diagnostics["wallclock"]["events"][name] = {
            "unix_time_ns": time.time_ns(), "elapsed_seconds": time.monotonic()-started}
    mark_wallclock("planning_request_started")
    # END timing never changes the cumulative work ledger or triggers GPU synchronization.
    outcome = PlanningResult(False, None, diagnostics)
    stage = "request_binding"
    try:
        # BEGIN every request owns its mutable online scene and cumulative ledger.
        scene = scene.clone_for_rollout()
        # END clones share the immutable model policy, not candidate hand clocks or query evidence.
        previous = request_ledger
        if previous is None and isinstance(continuation, dict) and isinstance(continuation.get("original_binding"), dict):
            previous = continuation.get("request_ledger", continuation["original_binding"].get("request_ledger"))
        if previous is not None:
            original_budget = PlanningBudget(**previous["planning_budget"])
        else:
            original_budget = PlanningBudget() if budget is None else budget
        # BEGIN execution facts enter a trusted work ledger only after validation.
        # An executor may already have appended its new observations to the
        # supplied ledger. Keep their computation history, but restore the
        # last frozen binding's facts until this continuation passes. A failed
        # request must never return unverified completion IDs as trusted facts.
        execution_fields = ("history", "event_records", "timing_events", "timing_checks",
            "completed_goal_ids", "goal_evidence", "completed_direction_ids", "direction_evidence",
            "effects", "effect_history")
        ledger_input = previous
        if isinstance(continuation, dict) and isinstance(continuation.get("original_binding"), dict):
            original_ledger = continuation["original_binding"]["request_ledger"]
            ledger_input = deepcopy(previous)
            for name in execution_fields:
                ledger_input[name] = deepcopy(original_ledger[name])
            diagnostics["continuation_facts_pending_validation"] = {
                name: deepcopy(continuation[name]) for name in (*execution_fields, "actual_feedback")
                if name in continuation}
        ledger = _request_work_ledger(problem, original_budget, ledger_input)
        if continuation is not None:
            ledger["requires_continuation"] = True
        # END pending facts remain diagnostics, not completed obligations.
        consumed = ledger["consumed_planning"]
        initial_seconds = consumed["planning_seconds"]
        entrance_consumed = deepcopy(consumed)
        request_budget = asdict(original_budget)
        diagnostics.update(planning_budget=request_budget, consumed_planning=consumed)
        def check(next_stage=None):
            if cancelled is not None and cancelled():
                raise MotionPlanningError("cancelled", "Motion request was cancelled")
            if deadline is not None and time.monotonic() >= deadline:
                raise MotionPlanningError("deadline_exceeded", "Motion request deadline elapsed")
            consumed["planning_seconds"] = initial_seconds+time.monotonic()-started
            if motion_ready:
                relative = {name: value-motion_baseline.get(name, 0) for name, value in consumed.items()}
                _check_work_budget(original_budget, relative, started=started,
                    initial_seconds=initial_seconds-motion_baseline.get("planning_seconds", 0.), stage=next_stage)
        # BEGIN the request ledger owns started attempts and completed queries.
        def record_validation_work(*, passes=0, points=0):
            if passes:
                check("validation_passes")
                consumed["validation_passes"] += passes
            consumed["validation_queries"] += points
            check()
        # END query accounting precedes the post-query deadline/budget check.
        check("binding_attempts")
        consumed["binding_attempts"] += 1
        # BEGIN explicit allocation changes preserve the existing request ledger.
        # Validate and retain the old ledger before considering the new limits,
        # so an invalid change still returns this attempt's cumulative work.
        if previous is not None and budget is not None and asdict(budget) != asdict(original_budget):
            ledger = _request_work_ledger(problem, budget, ledger)
            original_budget = budget
            consumed = ledger["consumed_planning"]
            request_budget = asdict(original_budget)
            diagnostics.update(planning_budget=request_budget, consumed_planning=consumed)
        # END allocation admission never buys a new solve, second or obligation.
        # BEGIN each admitted binding owns mutable problem and device state.
        # Initial clock construction can bind a hand before native allocation
        # fails. A private problem is therefore required for both first and
        # subsequent attempts, just as for remaining-obligation filtering.
        problem = deepcopy(problem)
        if previous is not None:
            _restore_original_direction_origins(problem, ledger["obligation_contract"])
        # END each admitted binding owns mutable problem and device state.
        if _obligation_contract(problem) != ledger["obligation_contract"]:
            raise ContinuationBindingError("A same request cannot remove, reorder or change original goal/process obligations")
        if continuation is not None:
            if not isinstance(continuation, dict) or not isinstance(continuation.get("original_binding"), dict):
                raise ContinuationBindingError("Continuation requires its original formal execution binding")
            if previous is None:
                raise ContinuationBindingError("Continuation lacks the original cumulative request ledger")
            # Binding owns its filtered obligations and hand occurrences. The
            # caller retains the full source problem after any failed attempt;
            # the scene clone shares only immutable geometry and the existing
            # cumulative robot oracle, never a reset query allowance.
            current = deepcopy(continuation)
            current["consumed_planning"] = deepcopy(consumed)
            budget, prior = _prepare_continuation(problem, scene, original_budget, current, rebind=rebind)
            # BEGIN commit actual facts only after their identities, clock and proofs pass.
            ledger = _request_work_ledger(problem, original_budget, ledger, continuation)
            consumed = ledger["consumed_planning"]
            diagnostics["consumed_planning"] = consumed
            diagnostics.pop("continuation_facts_pending_validation", None)
            # END a rejected continuation still retains its real planning work counters.
            prior["history"] = deepcopy(ledger["history"])
            prior["timing_events"] = deepcopy(ledger["timing_events"])
        else:
            if ledger["requires_continuation"] or any(ledger[name] for name in execution_fields if name != "timing_events"):
                raise ContinuationBindingError("An executed request needs its actual continuation facts before another binding")
            if plain(problem.events) != ledger["original_events"]:
                raise ContinuationBindingError("A failed-request retry changed the original event/source requirements")
            if rebind is not None and (not rebind.get("source") or rebind.get("kind") != "remaining"):
                raise ContinuationBindingError("Creating retry occurrences requires an actual execution continuation")
            budget = original_budget
        stage = "initialization"
        satisfied_dependencies = _verified_dependency_outcomes(continuation)
        try:
            problem, block_record = problem.bound_motion_block(
                goal_identities=None if prior is None else prior["goal_identities"],
                satisfied_dependencies=satisfied_dependencies)
        except ValueError as error:
            if "requires fresh dependency evidence" in str(error):
                raise MotionPlanningError("needs_rebind", str(error)) from error
            raise
        motion_baseline = _bind_motion_work_ledger(ledger, block_record, continuation, entrance_consumed)
        motion_ready = True
        check()
        budget = replace(budget, search_seconds=max(0., original_budget.search_seconds-
            (consumed["search_seconds"]-motion_baseline.get("search_seconds", 0.))))
        diagnostics["motion_block"] = block_record
        # BEGIN reject an unrepresented source occurrence before native allocation.
        if isinstance(joint_reference, JointTrajectory) and prior is None:
            stage = "reference_clock_binding"
            issues = _reference_clock_issues(problem, joint_reference,
                None if scene.gripper is None else scene.gripper.control_timestep_s)
            if issues:
                diagnostics.update(failure="reference_clock_outside_curve", failed_stage=stage,
                    reference_clock_issues=issues, diagnostic_trajectory=joint_reference,
                    diagnostic_candidate_scope="Supplied curve has no complete event clock and no execution authorization",
                    original_reference_candidate={"accepted": False, "clock_issues": issues,
                        "scope": "Required occurrence lies outside supplied curve; online geometry was not evaluated"},
                    joint_reference={"source_curve_preserved": True, "candidate_kind": "original_joint_curve",
                        "source_duration_s": joint_reference.duration, "bound_duration_s": None})
                return outcome
        # END this refusal does not invent a hold, source fit, or collision verdict.
        stage = "initialization"
        if backend is None:
            backend = MotionBackend(robot, robot_validator=robot_validator, budget=original_budget, device=device)
        check("initializations")
        consumed["initializations"] += 1
        backend._claim()
        backend_claimed = True
        if scene.gripper is not None:
            occurrences = {event_occurrence_key(event) for event in problem.events}
            scene.gripper.commands = tuple(row for row in scene.gripper.commands
                                           if event_occurrence_key(row) in occurrences)
        # BEGIN bind_request creates native solvers on first use or rebinds resident ones.
        mark_wallclock("backend_binding_started")
        planner = backend.bind_request(problem, scene, budget=budget, continuation=prior,
                                       goal_identities=block_record["goal_identities"])
        mark_wallclock("backend_binding_completed")
        # END failed initialization retains only the start milestone.
        planner.check_request = check
        # BEGIN native reference, enrichment and event IK share actual work.
        planner.request_work = consumed
        def remaining_request_seconds():
            check()
            remaining = original_budget.planning_seconds-(consumed["planning_seconds"]
                - motion_baseline.get("planning_seconds", 0.))
            return max(0., remaining if deadline is None else min(remaining, deadline-time.monotonic()))
        planner.remaining_request_seconds = remaining_request_seconds
        # END this is the same dictionary retained by request continuation.
        planner.satisfied_dependencies = tuple(sorted(satisfied_dependencies))
        for solver in (planner.ik, planner.trajopt):
            for optimizer in solver.core.optimizers:
                optimizer._core.request_guard = check
        check()
        diagnostics.update(backend="curobo_v2", batch_size=1, ik_seeds=budget.ik_seeds,
            trajectory_seeds=budget.trajectory_seeds, used_ompl=False,
            timing_policy="explicit_rebind_then_frozen" if rebind is not None else "source_proposal_then_frozen_binding",
            native_trajectory_dt=planner.budget.maximum_trajectory_dt,
            progress_parameter_error_bound=_PROGRESS_PARAMETER_ERROR, progress_clock_max_keys=_MAX_PROGRESS_KEYS,
            scene=scene.describe(), scene_workspace_bytes=planner.scene_workspace_bytes,
            scene_cpu_workspace_bytes=planner.scene_cpu_workspace_bytes,
            scene_gpu_workspace_bytes=planner.scene_gpu_workspace_bytes,
            anchor_fk_workspace_bytes=planner.anchor_fk_workspace_bytes, exact_anchor_capacity=planner.anchor_capacity,
            validation_storage=deepcopy(getattr(planner, "validation_storage", {})))
        # BEGIN describe the actual native state computation, including effort.
        diagnostics["joint_state_computation"] = {
            "kinematic_derivative_orders": [0, 1, 2, 3],
            "effort": "inverse_dynamics" if planner.trajopt.transition_model.compute_inverse_dynamics else "not_computed"}
        # END zero effort buffers in a kinematic rollout are not torque evidence.
        limits = planner._verification_native.get_cost("cspace").config.joint_limits
        diagnostics["verification_joint_limits"] = {"joint_names": list(limits.joint_names),
            **{name: getattr(limits, name).detach().cpu().numpy().tolist()
               for name in ("position", "velocity", "acceleration", "jerk")}}
        stage = "initial_state_verification"
        consumed["validation_queries"] += 1
        diagnostics["initial_state"] = planner.initial_state_diagnostics()
        check()
        if not diagnostics["initial_state"]["passed"]:
            # BEGIN distinguish a policy decision from a measured physical violation.
            diagnostics["failure"] = (_unknown_policy_failure(diagnostics["initial_state"],
                scene.unknown_policy) or "initial_state_infeasible")
            # END no optimization can change the measured entrance.
            return outcome
        # BEGIN seed, block and final candidates share one complete binding boundary.
        def qualify_candidate(trajectory, validation, attempt, *, binding_scene=None):
            nonlocal incumbent
            if not attempt["accepted"] or validation is None:
                raise ValueError("Only a completely validated candidate may enter execution binding")
            if incumbent is not None and attempt["fixed_objective"] >= incumbent["report"]["fixed_objective"]:
                return False
            bound_scene = scene if binding_scene is None else binding_scene
            binding = bind_candidate(trajectory, problem=problem, scene=scene if binding_scene is None else binding_scene,
                validation=deepcopy(validation), native_report=attempt)
            for event in problem.events:
                if event.timing and not any(event_occurrence_key(row) == event_occurrence_key(event)
                                            for row in ledger["timing_events"]):
                    ledger["timing_events"].append(plain(event))
            retry_goal_ids = set() if prior is None else set(prior["retry_goal_ids"])
            retry_direction_ids = set() if prior is None else set(prior["retry_direction_ids"])
            binding.update(motion_block=deepcopy(block_record), planning_budget=deepcopy(request_budget),
                request_events=deepcopy(ledger["original_events"]),
                consumed_events=[] if prior is None else prior["consumed_events"],
                goal_identities=deepcopy(block_record["goal_identities"]),
                # BEGIN no physical geometry score enters an execution binding.
                completed_goal_ids=[], goal_evidence=[],
                # END independent effect and direction histories stay separate.
                direction_identities=([[(index, column) for column in range(len(mechanism.directions))]
                    for index, mechanism in enumerate(getattr(problem, "mechanisms", ()))]
                    if prior is None else deepcopy(prior["direction_identities"])),
                completed_direction_ids=[deepcopy(value) for value in
                    (ledger["completed_direction_ids"] if prior is None else prior["completed_direction_ids"])
                    if tuple(value) not in retry_direction_ids],
                retry_direction_ids=sorted(retry_direction_ids),
                direction_evidence=deepcopy(ledger["direction_evidence"]),
                history=deepcopy(ledger["history"]), effect_history=deepcopy(ledger["effect_history"]),
                timing_events=deepcopy(ledger["timing_events"]), timing_checks=deepcopy(ledger["timing_checks"]),
                request_time_origin_s=0. if prior is None else prior["request_time_origin_s"],
                request_time_origin_step=0 if prior is None else prior["request_time_origin_step"])
            if prior is not None:
                binding.update({name: deepcopy(prior[name]) for name in (
                    "physics_budget", "consumed_physics_steps", "remaining_physics_steps",
                    "device_policy", "active_finger_interval")})
            # All work required to qualify and bind this candidate must
            # finish within budget. Cleanup after that fact is recorded,
            # but cannot erase an already available qualified candidate.
            # BEGIN freeze all delivery data before a later query replaces native buffers.
            binding["verification_joint_limits"] = deepcopy(diagnostics["verification_joint_limits"])
            record = {"trajectory": deepcopy(trajectory), "binding": binding,
                "report": deepcopy(attempt), "validation": deepcopy(validation),
                "scene": bound_scene.describe(), "event_time_checks": deepcopy(diagnostics["event_time_checks"]),
                "joint_reference": deepcopy(diagnostics.get("joint_reference")),
                "reference_clock": deepcopy(diagnostics.get("reference_clock")),
                "qualified_elapsed_seconds": None}
            check()
            record["qualified_elapsed_seconds"] = time.monotonic()-started
            incumbent = record
            # END a completed record does not depend on the planner's last candidate.

            # BEGIN only a fully qualified candidate receives this readiness milestone.
            mark_wallclock("qualified_candidate_ready")
            # END report persistence and execution preflight are later stages.
            return True

        def deliver_incumbent():
            """Restore the selected curve's clock, evidence and binding as one record."""
            outcome.success = True
            outcome.trajectory = incumbent["trajectory"]
            outcome.execution_binding = incumbent["binding"]
            diagnostics.update(incumbent["report"], returned_candidate=deepcopy(incumbent["report"]),
                scene=deepcopy(incumbent["scene"]), event_time_checks=deepcopy(incumbent["event_time_checks"]),
                joint_reference=deepcopy(incumbent["joint_reference"]),
                reference_clock=deepcopy(incumbent["reference_clock"]),
                qualified_candidate_elapsed_seconds=incumbent["qualified_elapsed_seconds"],
                executable_until=incumbent["trajectory"].executable_until,
                next_verification=incumbent["trajectory"].next_verification)
            return outcome

        def qualify_native(result, attempt):
            """Export immediately while the selected native validation still belongs to this curve."""
            if not attempt["accepted"] or (incumbent is not None
                    and attempt["fixed_objective"] >= incumbent["report"]["fixed_objective"]):
                return
            check()
            curve, _ = planner.export(result, attempt)
            qualify_candidate(curve, planner.last_validation, attempt)

        # BEGIN preserve a supplied polynomial before creating native proposals.
        if joint_reference is not None:
            if prior is not None:
                raise MotionPlanningError("unsupported_reference", "A supplied complete curve cannot replace an actual continuation's frozen suffix")
            stage = "reference_curve_verification"
            candidate, candidate_scene, candidate_map = planner.bind_reference_candidate(joint_reference)
            diagnostics["event_time_checks"] = _validate_bound_event_times(problem, candidate_map, candidate.duration,
                0., ledger["history"], None, ledger["timing_events"])
            candidate, reference_attempt, _ = planner.evaluate_candidate(candidate,
                candidate_scene=candidate_scene, record_work=record_validation_work)
            diagnostics["original_reference_candidate"] = deepcopy(reference_attempt)
            diagnostics["joint_reference"] = {"source_curve_preserved": True,
                "candidate_kind": "original_joint_curve", "source_duration_s": joint_reference.duration,
                "bound_duration_s": candidate.duration}
            diagnostics["reference_clock"] = {"duration": candidate.duration,
                "sample_times_s": candidate.times.tolist(),
                "scope": "Supplied polynomial retains its physical clock and bound progress map"}
            check()
            if reference_attempt["accepted"]:
                qualify_candidate(candidate, planner.last_validation, reference_attempt, binding_scene=candidate_scene)
                diagnostics.update(reference_candidate=deepcopy(reference_attempt), route="candidate_metrics")
            # BEGIN explicit observation requests stop at the completed reference verdict.
            policy_failure = _unknown_policy_failure(reference_attempt, candidate_scene.unknown_policy)
            if policy_failure == "needs_observation":
                diagnostics.update(failure=policy_failure, reference_candidate=deepcopy(reference_attempt),
                    diagnostic_trajectory=candidate,
                    failure_scope="Supplied curve requires observation under the selected policy; other routes were not evaluated")
                return outcome
            # END forbidden coverage may still use the existing bounded repair path.
            missing = {name: value for name, value in reference_attempt.get("scene_components", {}).items()
                if name.startswith("scene_update/") and (not np.isfinite(value["maximum"]) or value["maximum"] > 0)}
            if missing:
                diagnostics.update(failure="reference_outside_observation_scope", reference_scope_gaps=missing,
                    diagnostic_trajectory=candidate,
                    diagnostic_candidate_scope="Supplied joint curve; no execution authorization",
                    failure_scope="Supplied curve lacks current model support; other routes were not evaluated")
                return outcome
        # END the supplied curve remains eligible while native proposals improve reference fit.
        grid = _fixed_spline_grid(intervals=planner.budget.knots+4,
            native_dt=planner.budget.maximum_trajectory_dt, interpolation_steps=planner.budget.interpolation_steps,
            interpolation_dt=planner.budget.interpolation_dt)
        # BEGIN report the admitted clock without claiming unchanged source timestamps.
        source_seconds = problem.source_time_map["source_times"]
        source_duration = float(source_seconds[-1]-source_seconds[0])
        diagnostics["reference_clock"] = {**grid, "source_duration_s": source_duration,
            "duration_change_s": grid["duration"]-source_duration,
            "scope": "Source duration is a soft proposal; native, dense, event and finger consumers share the bound clock"}
        # END hard event deadlines are checked separately on this exact binding.
        diagnostics["event_time_checks"] = _validate_bound_event_times(problem, planner.progress_map, grid["duration"],
            0. if prior is None else prior["request_time_origin_s"], ledger["history"],
            None if prior is None else prior["goal_identities"], ledger["timing_events"])

        check("ik_batches")
        # BEGIN task initialization and main optimization share the first solve slot.
        check("trajectory_solves")
        consumed["ik_batches"] += 1
        consumed["trajectory_solves"] += 1
        stage = "reference_initialization"
        seeds, ik_success = planner.reference_seeds(joint_reference=joint_reference)
        check()
        diagnostics.update(reference_ik_success_counts=ik_success,
            ik_exit_early=bool(planner.ik.config.exit_early),
            reference_ik_distinct_counts=planner.reference_ik_distinct_counts,
            initial_seed_strategies=planner.initial_seed_strategies,
            reference_ik_queries=deepcopy(planner.reference_ik_queries),
            reference_ik_chains=deepcopy(planner.reference_ik_chains),
            reference_seed_fits=deepcopy(planner.reference_seed_fits))
        diagnostics["functional_ik_candidates"] = deepcopy(getattr(planner, "goal_ik_candidates", {}))
        diagnostics["joint_reference"] = deepcopy(planner.joint_reference_report)
        # BEGIN one reference verdict, one main solve and one mutually exclusive recovery.
        def select_candidate(result):
            return planner.select_candidate(result, record_work=record_validation_work)

        reference_result = planner.evaluate_reference(seeds)
        native_result, trajectory, attempt = select_candidate(reference_result)
        if joint_reference is not None and planner.joint_reference_report is not None:
            same = (trajectory.joint_names == joint_reference.joint_names
                    and np.array_equal(trajectory.times, joint_reference.times)
                    and np.array_equal(trajectory.position, joint_reference.position)
                    and np.array_equal(trajectory.velocity, joint_reference.velocity))
            diagnostics["joint_reference"]["source_curve_preserved"] = same
            diagnostics["joint_reference"]["candidate_kind"] = "original_joint_curve" if same else "native_fitted_joint_reference"
        diagnostics["reference_candidate"] = deepcopy(attempt)
        qualify_native(native_result, attempt)
        recovery = None
        # BEGIN observation policy stops only when completed evidence isolates coverage.
        observation_required = _unknown_policy_failure(attempt, scene.unknown_policy) == "needs_observation"
        if not observation_required:
            check()
            stage = "main_solve"
            native_result, trajectory, attempt = select_candidate(planner.optimize(
                seeds, iterations=planner.main_trajectory_iterations))
            # END the first slot owns both allocated parts; recovery keeps its quota.
            diagnostics["attempts"].append(deepcopy(attempt))
            qualify_native(native_result, attempt)
            check()
            observation_required = _unknown_policy_failure(attempt, scene.unknown_policy) == "needs_observation"
            if (incumbent is None and not attempt["accepted"] and not observation_required
                    and consumed["trajectory_solves"]-motion_baseline.get("trajectory_solves", 0) < original_budget.maximum_trajectory_solves):
                # BEGIN mixed scene support/contact failures cannot consume collision recovery.
                collision_failed = any(value > 0 for value in attempt["collision_residuals"].values())
                # END all original hard residuals still reject the candidate above.
                # BEGIN forbidden space can be avoided without changing observation policy.
                coverage_failed = any(np.isfinite(value) and value > 0
                    for value in attempt["coverage_residuals"].values())
                # END missing map support/contact/controller terms do not trigger this route.
                # BEGIN reuse dense collision evidence inside the existing second solve.
                promoted = (planner.promote_collision_witnesses(native_result) if collision_failed else None)
                if promoted is not None:
                    diagnostics["collision_witness_update"] = promoted
                if promoted is not None and promoted["changed"]:
                    recovery, stage = "collision_witnesses", "collision_witness_recovery"
                    seeds = native_result.solution.detach().clone().expand(1, budget.trajectory_seeds, -1, -1).contiguous()
                    planner.seed_strategies = [attempt["seed_provenance"]]*budget.trajectory_seeds
                # END only collision evidence may promote collision witnesses.
                # BEGIN reuse the same bounded search for either geometric obstruction.
                elif (collision_failed or coverage_failed) and budget.search_seconds > 0:
                    recovery, stage = "ompl", "bounded_seed_search"
                    search_started = time.monotonic()
                    recovered, report = planner.ompl_seed(seeds, trajectory)
                    consumed["search_seconds"] += time.monotonic()-search_started
                    consumed["search_queries"] += report.get("state_queries", 0)
                    diagnostics["ompl"] = report
                    if recovered is not None:
                        seeds = recovered
                        planner.seed_strategies = ["ompl"]*seeds.shape[1]
                    else:
                        recovery = "ompl_failed"
                # END no new observations, permissions or request time are granted.
                # BEGIN functional AL updates consume only the main solve's native blocks.
                if recovery in {"collision_witnesses", "ompl"}:
                    # END no extra full solve is authorized by task residuals alone.
                    check("trajectory_solves")
                    consumed["trajectory_solves"] += 1
                    native_result, trajectory, attempt = select_candidate(planner.optimize(seeds))
                    diagnostics["attempts"].append(deepcopy(attempt))
                    qualify_native(native_result, attempt)
                    check()
        # END mixed failures retain ordinary optimization and recovery.
        diagnostics.update(attempt, route=planner.route, recovery=recovery, used_ompl=recovery in {"ompl", "ompl_failed"})
        if incumbent is not None:
            return deliver_incumbent()
        # BEGIN completed finite search reports its selected failure evidence.
        diagnostics.update(failure=(_unknown_policy_failure(attempt, scene.unknown_policy)
            or "no_feasible_solution_found"), diagnostic_trajectory=trajectory,
            failure_scope="The selected candidate failed the declared requirements or unknown-space policy; "
                "its component evidence does not establish that every route to the fixed target fails")
        return outcome
        # END an executable result is delivered only by the completed incumbent record.
    except Exception as error:
        # BEGIN improvement exhaustion preserves only work completed within budget.
        if isinstance(error, PlanningWorkExhausted) and incumbent is not None:
            diagnostics["improvement_end"] = {"reason": "cumulative_planning_budget_exhausted",
                "stage": stage, "detail": str(error)}
            return deliver_incumbent()
        # END cancellation, deadlines and unexpected errors keep their own failure semantics.
        # BEGIN preserve the original failure stack without repeating any query.
        import traceback
        diagnostics.update(failure=(error.reason if isinstance(error, MotionPlanningError)
            else "cumulative_planning_budget_exhausted" if isinstance(error, PlanningWorkExhausted)
            else "continuation_binding_gap" if isinstance(error, ContinuationBindingError)
            else "event_time_requirement_unsatisfied" if isinstance(error, TimeBindingError)
            else "physical_domain_initializer_not_constructed" if isinstance(error, JointDomainInitializationError)
            else "planning_stage_failed"), failed_stage=stage, reason=str(error), error_type=type(error).__name__,
            failure_traceback=traceback.format_exc())
        # END stage context and native API failures remain directly diagnosable.
        candidate = (getattr(planner, "delivery_candidate", None) if planner is not None else None)
        candidate = trajectory if candidate is None else candidate
        if candidate is None:
            candidate = getattr(planner, "initialization_candidate", None)
        if isinstance(candidate, JointTrajectory):
            diagnostics["diagnostic_trajectory"] = candidate
            diagnostics["diagnostic_candidate_scope"] = "last completed native candidate; no execution authorization"
        outcome.success, outcome.trajectory, outcome.execution_binding = False, None, None
        return outcome
    finally:
        # BEGIN interrupted initialization retains its completed actual coefficients.
        if planner is not None:
            for name in ("reference_initialization", "reference_ik_queries", "reference_ik_chains",
                         "reference_seed_fits", "initial_seed_strategies"):
                if hasattr(planner, name):
                    diagnostics[name] = deepcopy(getattr(planner, name))
            diagnostics["functional_ik_candidates"] = deepcopy(planner.goal_ik_candidates)
        # END diagnostics do not qualify a partial candidate for execution.
        # BEGIN report bounded aggregate query counts for success and timeout.
        manager_type = getattr(planner, "_manager_type", None)
        if manager_type is not None:
            diagnostics["rollout_query_counts"] = manager_type.query_diagnostics()
        # END report collection does not retain or repeat individual states.
        if backend_claimed:
            # Native kernels/iteration blocks have returned before this point.
            # A timeout cannot release storage while their call is in flight.
            backend._release()
        if owned_backend and backend is not None:
            backend.close()
        diagnostics["elapsed_seconds"] = time.monotonic()-started
        if ledger is not None:
            ledger["consumed_planning"]["planning_seconds"] = initial_seconds+diagnostics["elapsed_seconds"]
            request_seconds = ledger["consumed_planning"]["planning_seconds"]-motion_baseline.get("planning_seconds", 0.)
            if request_seconds >= ledger["planning_budget"]["planning_seconds"]:
                diagnostics["wall_time_overrun_s"] = request_seconds-ledger["planning_budget"]["planning_seconds"]
                diagnostics["planning_budget_exhausted"] = True
            # BEGIN distinguish planning-only retries from actual-world continuation.
            # A previous work ledger carries cumulative computation, but does
            # not by itself supply executed state, events or a physical clock.
            # Append the accurate mode without relabelling historical records.
            ledger["binding_records"].append({
                "candidate_id": None if outcome.execution_binding is None else outcome.execution_binding.get("candidate_id"),
                "mode": "initial" if previous is None else
                        "planning_retry" if continuation is None and rebind is None else
                        "continuation" if rebind is None else
                        rebind.get("kind") if isinstance(rebind, dict) else "invalid_rebind",
                "source": deepcopy(rebind.get("source")) if isinstance(rebind, dict) else None,
                "request_time_origin_s": (None if continuation is not None and prior is None else
                                           0. if prior is None else prior["request_time_origin_s"]),
                "accepted": outcome.success, "failed_stage": diagnostics.get("failed_stage"),
                "failure": diagnostics.get("failure"), "reason": diagnostics.get("reason"),
                "consumed_planning": deepcopy(ledger["consumed_planning"])})
            # END the mode describes this attempt, not all earlier request work.
            outcome.request_ledger = deepcopy(ledger)
            if outcome.execution_binding is not None:
                outcome.execution_binding["consumed_planning"] = deepcopy(ledger["consumed_planning"])
                outcome.execution_binding["request_ledger"] = deepcopy(ledger)
# END every admitted attempt retains its work even when no curve was produced.


# END complete-window robot planning and generation-time process constraints.
