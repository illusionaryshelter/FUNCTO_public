"""Generate one robot joint trajectory for a complete grounded skill window.

cuRobo V2 owns forward kinematics, inverse kinematics and numerical search.
Nominal task errors enter ordinary costs; known model collisions enter
constraints. The delivered cubic preserves measured q0/dq0 and receives one
final polynomial bound check. Import initializes no perception, CUDA or robot.

The public result contains measured numerical residuals and original event
identities.  It does not manufacture grasp evidence or execute robot commands.
"""

# BEGIN native attempt-planner rebuild: nominal objectives and basic model bounds.
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field, fields, replace
import gc
import time
from threading import Lock, get_ident
from uuid import uuid4
from types import SimpleNamespace

import numpy as np
from scipy.interpolate import BSpline, CubicHermiteSpline
from .planning_problem import event_occurrence_key
from .resource_admission import (
    ResourceLimitError, require_host_memory, require_device_memory,
    DEFAULT_MAX_HOST_RSS_BYTES, MAX_HOST_RSS_BYTES,
    DEFAULT_MAX_DEVICE_USED_BYTES, MAX_DEVICE_USED_BYTES,
)

# No unused byte constants here may be mistaken for enforced protections.

from .cubic_seed import CubicSeed, cubic_position_report
from .cubic_trajectory import cubic_transition_config_type


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


@dataclass(frozen=True)
class PlanningBudget:
    """Explicit local resource and numerical-work limits, for a single request.

    These defaults bound the current 16 GB RAM / 12 GB GPU integration. More
    seeds or a larger batch must not be silently allocated by native defaults.
    The interpolation buffer covers the largest allowed native knot interval.
    Native solvers use eager execution for the dynamic process/scene tensors.
    ``max_task_anchors`` bounds the required semantic anchors. Native queries
    and bounded search share the checked scene/native allocation budgets.
    ``maximum_trajectory_dt`` fixes both native time bounds for this request:
    command timing and evidence stops take precedence over compression.
    """

    ik_seeds: int = 8
    trajectory_seeds: int = 2
    # N keeps N+4 physical intervals; the actual action has N+2 coefficients.
    knots: int = 8
    interpolation_steps: int = 4
    ik_iterations: int = 40
    trajectory_iterations: int = 100
    interpolation_dt: float = 0.025
    maximum_trajectory_dt: float = 0.2
    interpolation_buffer_size: int = 512
    constraint_weight: float = 100000.0
    max_task_anchors: int = 256
    max_robot_spheres: int = 256
    max_robot_dof: int = 16
    max_robot_links: int = 64
    max_scene_gpu_workspace_bytes: int = 1024**3
    max_host_rss_bytes: int = DEFAULT_MAX_HOST_RSS_BYTES
    max_device_used_bytes: int = DEFAULT_MAX_DEVICE_USED_BYTES
    max_validation_points: int = 4096
    validation_joint_step: float = 0.025
    search_seconds: float = 3.0
    maximum_trajectory_solves: int = 2
    planning_seconds: float = 120.0
    random_seed: int = 123

    def __post_init__(self):
        if (not 1 <= self.trajectory_seeds <= 2 or not self.trajectory_seeds <= self.ik_seeds <= 8
                or not 6 <= self.knots <= 24 or not 1 <= self.interpolation_steps <= 8
                or not 1 <= self.ik_iterations <= 200 or not 1 <= self.trajectory_iterations <= 300
                # Storage limits do not change collision clearance. Allocation
                # admission uses the actual sphere count and query horizon.
                or not 1 <= self.max_robot_spheres <= 640
                or not 1 <= self.max_robot_dof <= 16
                or not 1 <= self.max_robot_links <= 64
                or not 32 <= self.max_validation_points <= 65536
                or not 1 <= self.maximum_trajectory_solves <= 2):
            raise ValueError("Planning budget exceeds the bounded local single-batch configuration")
        if (type(self.max_task_anchors) is not int
                or not 2 <= self.max_task_anchors <= self.max_validation_points):
            raise ValueError("Task anchors must fit the finite native point budget")
        if (type(self.max_scene_gpu_workspace_bytes) is not int
                or not 0 <= self.max_scene_gpu_workspace_bytes <= 6*1024**3):
            raise ValueError("GPU scene workspaces must remain within the local six-GiB allocation budget")
        if (type(self.max_host_rss_bytes) is not int
                or not 0 < self.max_host_rss_bytes <= MAX_HOST_RSS_BYTES):
            raise ValueError("Host RSS must remain within the explicit twelve-GiB maximum")
        if (type(self.max_device_used_bytes) is not int
                or not 0 < self.max_device_used_bytes <= MAX_DEVICE_USED_BYTES):
            raise ValueError("Device usage must remain within the explicit eight-GiB maximum")
        for name in ("interpolation_dt", "maximum_trajectory_dt", "constraint_weight", "validation_joint_step",
                     "planning_seconds"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not np.isfinite(self.search_seconds) or self.search_seconds < 0:
            raise ValueError("search_seconds must be finite and nonnegative")
        # The selected cubic has degree+1 virtual intervals. Output sampling
        # must preserve its original physical clock, including nonzero dq0.
        grid = _fixed_spline_grid(intervals=self.knots + 4,
            native_dt=self.maximum_trajectory_dt, interpolation_steps=self.interpolation_steps,
            interpolation_dt=self.interpolation_dt)
        maximum_count = grid["valid_count"]
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
    finger_trajectory: object | None = None
    satisfied_dependencies: tuple[str, ...] = ()
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


@dataclass
class PlanningResult:
    success: bool
    trajectory: JointTrajectory | None
    diagnostics: dict
    execution_binding: dict | None = None
    request_ledger: dict | None = None


def _anchor_capacity(progress_map, anchors, budget):
    """Admit semantic anchors into a fixed native workspace; never add witnesses."""
    required = len(progress_map.anchor_parameters(anchors))
    if required > budget.max_task_anchors:
        raise MotionPlanningError("capacity_exceeded", "Task anchors exceed the resident backend capacity")
    return required


@dataclass(frozen=True)
class _PlanningWorkspace:
    """Actual ordinary scene storage plus bounded native anchor buffers."""
    cpu_shared_bytes: int
    cpu_live_bytes: int
    gpu_bytes: int
    anchor_fk_bytes: int
    @property
    def cpu_bytes(self):
        return self.cpu_shared_bytes+self.cpu_live_bytes
    @property
    def total_bytes(self):
        return self.cpu_bytes+self.gpu_bytes


def _planning_workspace_bytes(scene, robot_metadata, budget, anchor_capacity):
    """Account for ordinary field queries; no source certificates or graph pools."""
    horizon = (budget.knots+4)*budget.interpolation_steps+1
    shapes = ((4, budget.trajectory_seeds, horizon+anchor_capacity),
              (4, budget.trajectory_seeds*4, horizon+anchor_capacity),
              (2, budget.trajectory_seeds, budget.interpolation_buffer_size+anchor_capacity))
    allocations = [(copies, scene.estimated_workspace_allocation(batch, length))
                   for copies, batch, length in shapes]
    gpu = (sum(copies*value["gpu_retained"] for copies, value in allocations)
           + max(value["gpu_live"] for _, value in allocations)
           + max(value["gpu_shared"] for _, value in allocations))
    if gpu > budget.max_scene_gpu_workspace_bytes:
        raise ResourceLimitError("scene_gpu_workspace", gpu, budget.max_scene_gpu_workspace_bytes,
            description="Ordinary scene query storage exceeds the request allocation")
    dof, links, spheres = len(robot_metadata["active_joint_names"]), robot_metadata["links"], robot_metadata["spheres"]
    # Each cost and constraint manager has independent FK scratch so its
    # gradient buffers are not overwritten by the other manager's query.
    anchor_bytes = 44*budget.trajectory_seeds*anchor_capacity*4*(22+8*spheres+8*dof+12*links)
    return _PlanningWorkspace(max(value["cpu_shared"] for _, value in allocations),
        max(value["cpu_live"] for _, value in allocations), int(gpu), int(anchor_bytes))


def _require_planning_host_memory(cpu_live_bytes, *, max_host_rss_bytes):
    """Bound actual new host scratch; no full-source geometry is prepared."""
    require_host_memory(cpu_live_bytes,
        pending_existing_bytes=0, max_rss_bytes=max_host_rss_bytes)


def _require_planning_memory(torch, device, workspace, *, max_device_used_bytes):
    """Bound ordinary query/FK buffers and the existing native solver allowance."""
    require_device_memory(torch, device, native_bytes=2*1024**3,
        torch_bytes=workspace.gpu_bytes+workspace.anchor_fk_bytes,
        max_device_used_bytes=max_device_used_bytes)
    return workspace


def _bounded_robot_configuration(robot, problem, scene, budget):
    """Check the actual robot layout before native tensor/solver allocation.

    The installed native full configuration loader is not CPU-only: locked
    joints and even URDF inertial metadata invoke CUDA internally.  Reuse its
    existing ``yourdfpy`` metadata parser without meshes or FK instead.  Only
    selected link/joint names and sphere counts are inspected here; native
    cuRobo still constructs all transforms and performs every FK/IK operation.
    """
    from pathlib import Path
    import yourdfpy
    from curobo.content import get_assets_path, get_robot_configs_path
    from curobo._src.util.config_io import load_yaml, join_path

    def bounded_file(path):
        # A small robot description must not cause an unbounded CPU parse
        # before its declared joint/link/sphere sizes can be inspected.
        if Path(path).stat().st_size > 4*1024**2:
            raise ValueError("Robot description files must fit the four-MiB metadata budget")
        return str(path)

    data = robot if isinstance(robot, dict) else load_yaml(
        bounded_file(join_path(get_robot_configs_path(), robot)))
    kinematics = data["robot_cfg"]["kinematics"]
    if (kinematics.get("num_envs", 1) != 1 or kinematics.get("load_meshes", False)
            or kinematics.get("load_tool_frames_with_mesh", False)):
        raise ValueError("Planning requires one robot environment and its requested tool frame without mesh allocation")
    if not kinematics.get("load_collision_spheres", True):
        raise ValueError("Complete robot collision spheres are required for planning")
    collision_links = kinematics.get("collision_link_names", [])
    extra_links = kinematics.get("extra_links") or {}
    if (len(collision_links) > budget.max_robot_links or len(extra_links) > budget.max_robot_links
            or len(set(collision_links)) != len(collision_links)):
        raise ValueError("Declared robot links exceed the bounded model or contain duplicate collision links")
    if any(not isinstance(value, dict) for value in extra_links.values()):
        raise ValueError("Robot configuration must contain YAML-style extra-link dictionaries")
    cspace = kinematics.get("cspace") or {}
    if not isinstance(cspace, dict):
        raise ValueError("Robot configuration must contain a YAML-style cspace dictionary")
    for value in cspace.values():
        if isinstance(value, (list, tuple, np.ndarray)) and (
                len(value) > budget.max_robot_dof or np.asarray(value).ndim != 1):
            raise ValueError("Robot cspace arrays exceed the bounded independent-joint layout")

    spheres = kinematics.get("collision_spheres")
    if isinstance(spheres, str):
        spheres = load_yaml(bounded_file(join_path(get_robot_configs_path(), spheres)))["collision_spheres"]
    if not isinstance(spheres, dict):
        raise ValueError("Planning requires an explicit native robot sphere dictionary")
    extra_spheres = kinematics.get("extra_collision_spheres") or {}
    if (sum(len(value) for value in spheres.values()) > budget.max_robot_spheres
            or any(type(count) is not int or count < 0 or count > budget.max_robot_spheres
                   for count in extra_spheres.values())):
        raise ValueError("Robot collision geometry exceeds the declared sphere budget before native allocation")
    # Native extra_collision_spheres replaces that link's list, including
    # inactive attachment slots.  Count its real allocation, not positive
    # radii only, before native self-collision constructs an S-by-S table.
    counts = {name: len(value) for name, value in spheres.items()}
    counts.update(extra_spheres)
    if sum(counts.values()) > budget.max_robot_spheres:
        raise ValueError("Robot attachment sphere slots exceed the declared sphere budget")
    if any(name not in counts for name in collision_links):
        raise ValueError("Every native collision link requires an explicit sphere model")
    sphere_count = sum(counts[name] for name in collision_links)
    if not 0 < sphere_count <= budget.max_robot_spheres:
        raise ValueError("Robot collision model must fit the declared nonempty sphere budget")
    if len(scene.robot_sphere_materials) != sphere_count:
        raise ValueError("Scene robot-sphere material identities must match the complete native sphere layout")

    asset_root = kinematics.get("external_asset_path") or get_assets_path()
    urdf_path = bounded_file(join_path(asset_root, kinematics["urdf_path"]))
    parsed = yourdfpy.URDF.load(urdf_path, load_meshes=False, load_collision_meshes=False,
                                build_scene_graph=False, build_collision_scene_graph=False)
    parents = {joint.child: joint.parent for joint in parsed.joint_map.values()}
    driven = {}
    for joint in parsed.joint_map.values():
        if joint.type not in ("fixed", "prismatic", "revolute", "continuous"):
            raise ValueError("Robot joints must use the native one-coordinate kinematic types")
        if joint.type != "fixed":
            driven[joint.child] = joint.name if joint.mimic is None else joint.mimic.joint
    for name, link in extra_links.items():
        parents[name] = link["parent_link_name"]
        if link.get("child_link_name") is not None:
            parents[link["child_link_name"]] = name
        if link["joint_type"] != "FIXED":
            driven[name] = link["joint_name"]

    base = kinematics["base_link"]
    if base not in parsed.link_map and base not in extra_links:
        raise ValueError("The declared robot base is absent from its URDF and extra links")
    selected = {base}
    # Match the native model's selected tree: requested tool, collision links,
    # and all explicit extra links. This checks metadata connectivity only.
    for target in [problem.tool_link, *collision_links, *extra_links]:
        link, chain = target, set()
        while link != base:
            if link in chain or link not in parents:
                raise ValueError("A selected robot link has no acyclic chain to its declared base")
            chain.add(link)
            selected.add(link)
            if len(selected) > budget.max_robot_links:
                raise ValueError("Native robot link workspaces exceed the declared link budget")
            link = parents[link]
    unlocked = {driven[link] for link in selected if link != base and link in driven}
    # Current occupied-volume evidence may request extra link poses, but it
    # may not enlarge the already bounded planning model to obtain them.
    required_links = scene.required_robot_links
    if (len(required_links) > budget.max_robot_links or len(set(required_links)) != len(required_links)
            or not set(required_links) <= selected):
        raise ValueError("Occupied robot evidence links must belong to the bounded planning model")
    if len(unlocked) > budget.max_robot_dof:
        # Native locking evaluates the complete unlocked tree once, so bound
        # that intermediate model as well as the final active joint set.
        raise ValueError("Native unlocked-joint workspaces exceed the declared DOF budget")
    locked = kinematics.get("lock_joints") or {}
    active = unlocked-set(locked)
    if active != set(problem.joint_names):
        raise ValueError("Robot active joints must equal the provided feedback joints; configure locked or driven gripper joints explicitly")

    configuration = deepcopy(data)
    configuration["robot_cfg"]["kinematics"]["tool_frames"] = [problem.tool_link]
    # Resolve the already checked geometry once, preserving native extra-slot
    # semantics and avoiding a second unbounded file read inside the solver.
    configuration["robot_cfg"]["kinematics"]["collision_spheres"] = deepcopy(spheres)
    metadata = {"active_joint_names": sorted(active), "unlocked_dof": len(unlocked),
                "links": len(selected), "selected_link_names": sorted(selected), "spheres": sphere_count}
    return configuration, metadata


@dataclass(frozen=True)
class _ProgressMap:
    """A fixed, monotone phase schedule; repeated s values encode pauses."""

    u: np.ndarray
    s: np.ndarray
    event_clock: dict = field(default_factory=dict, compare=False)

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
    # Fixing output_dt first admits only a few binary-exact knot durations
    # and can stretch a 9.85-second reference to 16 seconds. Instead round
    # native dt upward by at most a few float32 ulps, retaining suffix coverage
    # for continuations, and derive output_dt through exact power-of-two
    # division. The unchanged grid validator still checks every multiplication.
    output_limit = float(np.float32(budget.interpolation_dt))
    maximum = _fixed_spline_grid(intervals=budget.knots+4,
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
    needed = max(1, int(np.ceil(float(knot_dt)/output_limit)))
    subdivisions = 1 << (needed-1).bit_length()
    output_dt = float(knot_dt)/subdivisions
    selected = replace(budget, maximum_trajectory_dt=float(native_dt), interpolation_dt=output_dt)
    return selected, _ProgressMap((seconds-seconds[0])/source_duration, progress.copy())

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
    # Future nominal anchors still obey their planned windows. Once an anchor
    # is outside the remaining plan, its actual arrival and any dependent time
    # origin belong only to simulation grading; they cannot reject a rebind.
    history = [row for row in history if not (row.get("check_target") == "arrival"
        and (row.get("goal_id") or [None])[0] == "pose_events")]

    def past_pose_timing(constraint):
        identities = [constraint.goal_id if constraint.check_target == "arrival" else None,
                      (constraint.origin_event or {}).get("goal_id")]
        return any(identity and identity[0] == "pose_events"
                   and identity[1] not in goal_identities["pose_events"] for identity in identities)
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


class ContinuationBindingError(ValueError):
    """Remaining work lacks facts, budget or a compatible original clock."""


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
        "pose_events": ("progress", "source", "position_tolerance_m", "rotation_tolerance_rad"),
        "coordinate_events": ("progress", "source", "target_coordinate", "tolerance", "kind", "coordinate_mode", "coordinate_origin"),
        "supports": ("progress", "source", "margin", "gap_tolerance", "normal_tolerance"),
        "pose_relations": ("start", "end", "source", "pose_bounds"),
        "mechanisms": ("end", "source", "directions"),
        "interiors": ("start", "end", "source", "margin"),
        "couplings": ("joint_names", "matrix", "target", "source"),
    }
    contract = {}
    for collection, names in columns.items():
        contract[collection] = [{"type": type(item).__name__, **{name: plain(getattr(item, name)) for name in names},
            **({"actual_relation": (None if item.actual_relation is None else
                {key: plain(value) for key, value in item.actual_relation.items()
                 if not key.startswith("T_")})} if collection == "pose_events" else {}),
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


_ALLOCATION_BUDGET_FIELDS = frozenset(("max_host_rss_bytes", "max_device_used_bytes",
                                     "max_scene_gpu_workspace_bytes"))


def _planning_work_budget(values):
    """Retain every work, geometry, timing and numerical field for comparison."""
    return {name: value for name, value in values.items() if name not in _ALLOCATION_BUDGET_FIELDS}


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
    if recorded_budget != requested_budget:
        ledger["binding_records"].append({"mode": "resource_limits_changed",
            "source": "explicit_planning_budget",
            "previous": {name: recorded_budget[name] for name in sorted(_ALLOCATION_BUDGET_FIELDS)},
            "current": {name: requested_budget[name] for name in sorted(_ALLOCATION_BUDGET_FIELDS)},
            "consumed_planning": deepcopy(consumed)})
        ledger["planning_budget"] = requested_budget
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
        if not {tuple(value) for value in old["completed_goal_ids"]} <= {tuple(value) for value in ledger["completed_goal_ids"]}:
            raise ContinuationBindingError("A request ledger cannot erase historical completed goals")
        if not {tuple(value) for value in old["completed_direction_ids"]} <= {tuple(value) for value in ledger["completed_direction_ids"]}:
            raise ContinuationBindingError("A request ledger cannot erase historical completed directions")
        if any(consumed[name] < old["consumed_planning"][name] for name in consumed):
            raise ContinuationBindingError("Cumulative planning work must retain all prior attempts")
        for name in ("history", "event_records", "timing_events", "timing_checks", "goal_evidence", "direction_evidence", "effect_history", "binding_records"):
            if ledger[name][:len(old[name])] != old[name]:
                raise ContinuationBindingError(f"The request ledger rewrote prior {name}")
        # Append actual execution facts once; a historical T is retained here,
        # not promoted to currently applicable scene evidence in a new binding.
        for name in ("history", "timing_checks", "goal_evidence", "completed_goal_ids", "completed_direction_ids", "direction_evidence", "effect_history"):
            for row in continuation.get(name, ()):
                if ((name == "completed_goal_ids" and row[0] == "pose_events")
                        or (name == "goal_evidence" and row["identity"][0] == "pose_events")):
                    continue
                if row not in ledger[name]:
                    ledger[name].append(deepcopy(row))
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


def _verified_completed_goals(continuation, original, all_goals, *, elapsed,
                              curve_elapsed):
    """Verify non-pose goal evidence without importing simulation pose metrics.

    Target-pose accuracy is exclusively posthoc. Neither a previous pose label
    nor its error may gate continuation or certify physical task completion.
    Past nominal pose anchors leave the remaining plan by progress, retaining
    their original requirements in the request ledger for simulation grading.
    """
    def identities(rows):
        if (not isinstance(rows, (tuple, list)) or any(
                not isinstance(row, (tuple, list)) or len(row) != 2
                or not isinstance(row[0], str) or type(row[1]) is not int
                for row in rows)):
            raise ContinuationBindingError("Completed goals need original collection/index identities")
        values = {tuple(row) for row in rows}
        if not values <= all_goals:
            raise ContinuationBindingError("Completed goal identity does not belong to the original request")
        return {identity for identity in values if identity[0] != "pose_events"}

    previous = identities(original.get("completed_goal_ids", []))
    claimed = identities(continuation.get("completed_goal_ids", []))
    ledger = original["request_ledger"]
    if not previous <= identities(ledger["completed_goal_ids"]):
        raise ContinuationBindingError("Prior completed goals lack their preserved request ledger")
    for identity in previous:
        if not any(tuple(row.get("identity", ())) == identity
                   and row.get("value") == "T" and row.get("completion_verified") is True
                   and isinstance(row.get("candidate_id"), str) and row["candidate_id"]
                   for row in ledger["goal_evidence"]):
            raise ContinuationBindingError("Prior completed goals lack their preserved actual T evidence")
    new = claimed-previous
    if not new:
        return previous | claimed
    feedback = continuation["actual_feedback"]
    timestamp = feedback.get("timestamp_s")
    if (isinstance(timestamp, bool) or not isinstance(timestamp, (float, int))
            or not np.isfinite(timestamp)):
        raise ContinuationBindingError("Goal completion needs synchronized actual continuation time")
    origin = float(timestamp)-elapsed
    for identity in new:
        proofs = [row for row in continuation.get("goal_evidence", ())
                  if isinstance(row, dict) and tuple(row.get("identity", ())) == identity]
        verified = False
        for row in proofs:
            if (row.get("candidate_id") != original["candidate_id"] or row.get("value") != "T"
                    or row.get("completion_verified") is not True
                    or any(not isinstance(row.get(key), str) or not row[key].strip()
                           for key in ("source", "feedback_id"))):
                continue
            times = [row.get(key) for key in ("timestamp_s", "physical_elapsed_s", "evaluated_curve_time_s")]
            if (any(isinstance(value, bool) or not isinstance(value, (float, int))
                    or not np.isfinite(value) for value in times)):
                continue
            acquired, physical, curve_time = times
            if (not 0 <= physical <= elapsed+1e-8 or not 0 <= curve_time <= curve_elapsed+1e-8
                    or abs(acquired-origin-physical) > 1e-8):
                continue
            goals = original["problem"][identity[0]]
            indices = original.get("goal_identities", {}).get(identity[0], list(range(len(goals))))
            if identity[1] not in indices:
                continue
            requirement = goals[indices.index(identity[1])]
            if row.get("requirement") != requirement or not row.get("measurement_sources"):
                continue
            verified = True
            break
        if not verified:
            raise ContinuationBindingError("Goal completion needs actual allowed-set T evidence from this executed candidate")
    return previous | claimed


def _prepare_continuation(problem, scene, budget, continuation, *, rebind=None):
    """Validate a fresh actual/scene binding without replaying consumed events.

    A continuation is the executor's existing result record. It is not a
    request to restore the old actual state or guess past task success. The
    original absolute progress, event identities and cumulative solve budget
    survive; angular history and past goal effects require supplied facts.
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
    previous_history = original["request_ledger"]["history"]
    history = continuation.get("history", [])
    if (not isinstance(history, (list, tuple))
            or list(history[:len(previous_history)]) != previous_history):
        raise ContinuationBindingError("Continuation cannot erase or rewrite original physical timing history")
    _restore_original_direction_origins(problem, original["request_ledger"]["obligation_contract"])
    if _obligation_contract(problem) != original["request_ledger"]["obligation_contract"]:
        raise ContinuationBindingError("A fresh binding cannot remove, reorder or change original goal/process obligations")
    if "planning_budget" not in original or "consumed_planning" not in original:
        raise ContinuationBindingError("Continuation lacks the original cumulative planning budget")
    request_budget = PlanningBudget(**original["planning_budget"])
    if budget is not None:
        if _planning_work_budget(asdict(budget)) != _planning_work_budget(asdict(request_budget)):
            raise ContinuationBindingError("A continuation cannot reset or enlarge its original planning budget")
        request_budget = budget
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
    confirmed_steps = continuation.get("confirmed_physics_steps")
    accounting_exact = continuation.get("step_accounting_exact")
    timestep = continuation.get("device_timestep_s")
    if (type(local_steps) is not int or local_steps < 0
            or type(confirmed_steps) is not int or not 0 <= confirmed_steps <= local_steps
            or accounting_exact is not True
            or not isinstance(timestep, (float, int)) or not np.isfinite(timestep) or timestep <= 0
            or physics_consumed != original.get("consumed_physics_steps", 0)+local_steps):
        raise ContinuationBindingError("Continuation requires cumulative charges, exact confirmed steps and the unchanged native device timestep")
    remaining = request_budget
    feedback = continuation.get("actual_feedback", {})
    if not isinstance(feedback, dict):
        raise ContinuationBindingError("Continuation needs captured synchronized actual feedback")
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
    elapsed = confirmed_steps*float(timestep)
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
    all_goals = {(name, index) for name in ("pose_events", "coordinate_events", "supports")
                 for index in range(len(getattr(problem, name)))}
    completed_goals = _verified_completed_goals(continuation, original, all_goals,
        elapsed=elapsed, curve_elapsed=curve_elapsed)
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
    completed_directions.update(claimed_directions)
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
    for event in problem.events:
        identity = event_occurrence_key(event)
        if (identity not in old_occurrences and identity not in applied
                and event.progress >= start_progress):
            selected_events[(event.skill_id, event.event_id)] = deepcopy(event)
    for identity in retry_ids:
        if any(key[:2] == identity for key in pending_ids):
            raise ContinuationBindingError("Retry selects consumed occurrences; pending work uses remaining rebind")
        occurrences = [key[2] for key in (*old_occurrences, *applied) if key[:2] == identity]
        occurrences.extend(event_occurrence_key(row)[2] for row in continuation.get("history", ())
                           if event_occurrence_key(row)[:2] == identity)
        selected_events[identity] = replace(current_events[identity], occurrence=max(occurrences, default=0)+1)
    remaining_goal_ids = {(name, index) for name in ("pose_events", "coordinate_events", "supports")
        for index, goal in enumerate(getattr(problem, name))
        if ((name, index) not in completed_goals and (name != "pose_events" or goal.progress >= start_progress))
        or (name, index) in retry_goals}
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
    for collection in ("pose_events", "coordinate_events", "supports"):
        for index, requirement in enumerate(getattr(problem, collection)):
            if (collection != "pose_events" and rebind is None
                    and requirement.progress < start_progress and (collection, index) not in completed_goals):
                raise ContinuationBindingError(f"Past goal {collection}:{index} needs actual completion evidence")
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
        "actual_elapsed": float(actual_elapsed), "elapsed_device_steps": confirmed_steps,
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
        "completed_goal_ids": sorted(completed_goals), "effects": deepcopy(continuation.get("effects", {})),
        "previous_request_time_origin_s": float(original.get("request_time_origin_s", 0.)),
        "request_time_origin_s": float(original.get("request_time_origin_s", 0.))+float(elapsed),
        "request_time_origin_step": int(original.get("request_time_origin_step", 0))+local_steps}


def _continuation_timing(problem, budget, continuation):
    """Use the original suffix, or rebuild every clock consumer for an explicit rebind."""
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


def _ompl_stage(start, goal, lower, upper, valid, *, step, seconds, sample_limit):
    """One ordinary native RRTConnect query; this only proposes a seed path."""
    from ompl import base as ob, geometric as og
    deadline = time.monotonic()+seconds
    dimension = len(start)
    report = {"space": "joint", "validity_checks": 0}
    space = ob.RealVectorStateSpace(dimension)
    bounds = ob.RealVectorBounds(dimension)
    for index, (lo, hi) in enumerate(zip(lower, upper)):
        bounds.setLow(index, float(lo)); bounds.setHigh(index, float(hi))
    space.setBounds(bounds)
    information = ob.SpaceInformation(space)
    information.setStateValidityCheckingResolution(min(.05, step/np.linalg.norm(upper-lower)))
    def check(state):
        report["validity_checks"] += 1
        if time.monotonic() >= deadline or report["validity_checks"] > sample_limit:
            return False
        value = np.asarray([state[index] for index in range(dimension)])
        return bool(np.isfinite(value).all() and np.all(value >= lower)
                    and np.all(value <= upper) and valid(value))
    information.setStateValidityChecker(check)
    information.setup()
    first, last = space.allocState(), space.allocState()
    for index in range(dimension):
        first[index], last[index] = float(start[index]), float(goal[index])
    if not check(first) or not check(last):
        return None, {**report, "failure": "model_collision_at_connection_boundary"}
    definition = ob.ProblemDefinition(information)
    definition.setStartAndGoalStates(first, last)
    search = og.RRTConnect(information)
    search.setProblemDefinition(definition)
    search.setup()
    search.solve(max(0., deadline-time.monotonic()))
    if not definition.hasExactSolution():
        return None, {**report, "failure": "no_connection_within_search_budget"}
    path = definition.getSolutionPath()
    path.interpolate(min(sample_limit, max(path.getStateCount(), int(np.ceil(path.length()/step))+1)))
    values = np.asarray([[path.getState(i)[j] for j in range(dimension)]
                         for i in range(path.getStateCount())])
    return values, {**report, "samples": len(values), "strategy": "native_rrt_connect"}


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


def _verification_stop_preference(progress, joint_velocity, stop_progress, *, trajectory_time=None):
    """Squared speed at each declared stop's first physical occurrence.

    A verification stop commands zero velocity. This objective approaches
    that command even inside the nominal residual's numerical band; actual
    stopped-state evidence remains the execution adapter's responsibility.
    """
    import torch
    from .planning_problem import _first_event_query
    stopped = torch.zeros_like(progress, dtype=torch.bool)
    for value in np.unique(stop_progress):
        # Use exactly the same progress support as the hard stop residual.
        stopped = stopped | _first_event_query(progress, value, trajectory_time)
    return torch.where(stopped, joint_velocity.square().sum(-1), torch.zeros_like(progress))


def _formal_native_clock(trajopt):
    """Bind hand, rollout and dense export to one unchanged physical clock."""
    from curobo._src.types.control_space import ControlSpace
    config = trajopt.config
    if config.minimum_trajectory_dt != config.maximum_trajectory_dt:
        raise ValueError("A separately timed hand requires one fixed request-wide native dt")
    control = trajopt.transition_model.control_space
    if control != ControlSpace.BSPLINE_3:
        raise ValueError("The formal clock must use the selected native cubic spline representation")
    grid = _fixed_spline_grid(
        intervals=trajopt.action_horizon+2,
        native_dt=config.maximum_trajectory_dt, interpolation_steps=trajopt.interpolation_steps,
        interpolation_dt=config.interpolation_dt)
    return {**grid, "source": "instantaneous-boundary cubic on unchanged exact float32 clock"}


def _joint_state_sample_times(joint_state, counts):
    """Round physical time once, without a normalized-float32 roundtrip."""
    import torch
    position = joint_state.position
    indices = torch.minimum(torch.arange(position.shape[1], device=position.device)[None],
                            counts[:, None]-1)
    return (indices.to(torch.float64)*joint_state.dt.reshape(-1, 1).to(torch.float64)).to(position.dtype)


def _known_scene_cost(scene, spheres, position, quaternion,
                             progress, joint_position, joint_names, *, trajectory_time,
                             component_diagnostics=None):
    """Query the frozen ordinary model once; unknown data is diagnostic only."""
    values = scene.components(spheres, position, quaternion, progress, joint_position,
        joint_names, trajectory_time=trajectory_time)
    if component_diagnostics is not None:
        component_diagnostics.update({name: {"maximum": float(value.detach().max().item())}
                                      for name, value in values.items()})
    return sum(values.values(), progress.new_zeros(progress.shape))


def _nominal_task_cost(problem, position, quaternion, progress, joint_position,
                       joint_names, joint_velocity, *, trajectory_time,
                       tool_linear_velocity=None, tool_angular_velocity=None):
    """Return finite-reference objectives without a task feasibility decision."""
    terms = problem.residuals(position, quaternion, progress, joint_position,
        joint_names, joint_velocity, trajectory_time=trajectory_time,
        tool_linear_velocity=tool_linear_velocity, tool_angular_velocity=tool_angular_velocity)
    cost = sum((value.square() for value in terms.values()), progress.new_zeros(progress.shape))
    cost = cost+_verification_stop_preference(progress, joint_velocity, problem.stop_progress,
                                             trajectory_time=trajectory_time)
    return cost+problem.optimizer_preferences(position, quaternion, progress,
                                               trajectory_time=trajectory_time)


def _finite_maximum(value):
    """Keep nonfinite model evidence visible instead of hiding it in a max."""
    import torch
    flat = value.detach().reshape(-1)
    return float(flat.max().item()) if bool(torch.isfinite(flat).all()) else float("inf")


def _minimum_cost_candidate(objectives, physical_acceptance):
    """Use one scalar objective inside the finite, physically valid seed set.

    When no candidate passes, choose the finite minimum only for diagnostics
    and bounded spatial search. The caller retains its rejected status.
    """
    values = np.asarray(objectives, dtype=float)
    finite = np.isfinite(values)
    eligible = finite & np.asarray(physical_acceptance, dtype=bool)
    if not finite.any():
        raise MotionPlanningError("no_finite_candidate", "Native optimization returned no finite objective")
    mask = eligible if eligible.any() else finite
    return int(np.argmin(np.where(mask, values, np.inf)))


def _process_manager_type(problem, scene, progress_map, anchors, budget, *, clock=None,
                          seed_parameter=None):
    """Compose native costs and constraints without replacing its optimizer.

    Modes ending in ``cost`` contribute nominal motion objectives only. Modes
    ending in ``constraint`` retain each named scene constraint. Native
    cspace terms retain joint limits; the scene owns all solid collisions. Exact event
    samples are finite semantic inputs, never adaptive feasibility witnesses.
    """
    import torch
    from curobo.kinematics import Kinematics
    from curobo.types import JointState
    from curobo._src.rollout.cost_manager.cost_manager_robot import RobotCostManager
    from curobo._src.rollout.cost_manager.cost_manager_robot_cfg import RobotCostManagerCfg
    orientation_vjp, native_layout = _orientation_vjp_type(), _native_gradient_layout_type()
    managers = []

    class ProcessCostManager(RobotCostManager):
        def initialize_from_config(self, config, transition_model, scene_collision_checker=None, **kwargs):
            # Native initialization mutates component weights. Each manager
            # therefore receives its own config, including repeated metrics.
            config = deepcopy(config)
            super().initialize_from_config(config, transition_model, scene_collision_checker, **kwargs)
            self._process_mode = config.process_mode
            self._dense_counts = None
            self._query_counts = {"mode": self._process_mode, "calls": 0, "states": 0}
            managers.append(self)
            if not self._process_mode:
                return
            self._joint_names = tuple(transition_model.joint_names)
            transition_model.robot_model.compute_jacobian = True
            self._tool_index = transition_model.robot_model.tool_frames.index(problem.tool_link)
            if not self._process_mode.startswith("seed_"):
                self._anchor_fk = Kinematics(transition_model.robot_model.config, compute_jacobian=True)
            self.bind_request_data()

        def bind_request_data(self):
            self._scene = scene.clone_for_rollout()
            self._dense_counts = None
            if not self._process_mode or self._process_mode.startswith("seed_"):
                return
            pairs = progress_map.anchor_parameters(anchors)
            _anchor_capacity(progress_map, anchors, budget)
            self._map_u = self.device_cfg.to_device(progress_map.u)
            self._map_s = self.device_cfg.to_device(progress_map.s)
            self._anchor_u = self.device_cfg.to_device(pairs[:, 0])[None]
            self._anchor_s = self.device_cfg.to_device(pairs[:, 1])[None]
            self._anchor_query_u = torch.as_tensor(pairs[:, 0][None], device=self.device_cfg.device,
                                                   dtype=torch.float64)

        def set_dense_counts(self, counts):
            self._dense_counts = counts.reshape(-1).long()

        def compute_costs(self, state, cost_collection=None, goal=None, **kwargs):
            if not self._process_mode:
                return super().compute_costs(state, cost_collection, goal, **kwargs)
            q, dq = state.joint_state.position, state.joint_state.velocity
            batch, horizon = q.shape[:2]
            seed_mode = self._process_mode.startswith("seed_")
            counts = (torch.full((batch,), horizon, device=q.device, dtype=torch.long)
                      if self._dense_counts is None else self._dense_counts)
            times = ((seed_parameter*clock["duration"]).expand(batch, horizon) if seed_mode else
                     _joint_state_sample_times(state.joint_state, counts))
            # BEGIN scene queries own timed fingers and all solid collisions.
            # IK only constructs a nominal seed; full trajectory constraints
            # evaluate the same registered solids as entrance and export.
            # END no native sphere cost may duplicate that physical criterion.
            collection = super().compute_costs(state, cost_collection, goal, **kwargs)
            if seed_mode:
                return collection
            self._query_counts["calls"] += 1
            self._query_counts["states"] += batch*horizon
            u = (times/clock["duration"]).clamp(0, 1)
            progress = _torch_interpolate(self._map_u, self._map_s, u)
            valid = torch.arange(horizon, device=q.device)[None] < counts[:, None]
            tool = state.tool_poses
            quaternion = orientation_vjp.apply(q, tool.quaternion.detach(),
                state.cuda_robot_model_state.tool_jacobians[..., 3:, :].detach())[:, :, self._tool_index]
            position = native_layout.apply(tool.position)[:, :, self._tool_index]
            spheres = native_layout.apply(state.robot_spheres)
            need_rates = (self._process_mode.endswith("cost") and any(
                direction.kind == "local_rate" for mechanism in problem.mechanisms for direction in mechanism.directions))
            twist = (torch.einsum("...kd,...d->...k", native_layout.apply(
                state.cuda_robot_model_state.tool_jacobians)[:, :, self._tool_index], dq) if need_rates else None)
            anchor_u = self._anchor_u.expand(batch, -1)
            anchor_s = self._anchor_s.expand(batch, -1)
            query_u = self._anchor_query_u.expand(batch, -1)
            anchor_times = (query_u*clock["duration"]).to(q.dtype)
            anchor_q, anchor_dq = _hermite_at_parameters(q, dq, state.joint_state.dt, counts, query_u)
            joints = JointState.from_position(anchor_q.contiguous(), joint_names=list(self._joint_names))
            anchor_state = self._anchor_fk.compute_kinematics(joints)
            anchor_quaternion = orientation_vjp.apply(anchor_q, anchor_state.tool_poses.quaternion.detach(),
                anchor_state.tool_jacobians[..., 3:, :].detach())[:, :, self._tool_index]
            anchor_position = native_layout.apply(anchor_state.tool_poses.position)[:, :, self._tool_index]
            if need_rates:
                anchor_twist = torch.einsum("...kd,...d->...k", native_layout.apply(
                    anchor_state.tool_jacobians)[:, :, self._tool_index], anchor_dq)
                twist = torch.cat((twist, anchor_twist), 1)
            position = torch.cat((position, anchor_position), 1)
            quaternion = torch.cat((quaternion, anchor_quaternion), 1)
            spheres = torch.cat((spheres, native_layout.apply(anchor_state.robot_spheres)), 1)
            q, dq = torch.cat((q, anchor_q), 1), torch.cat((dq, anchor_dq), 1)
            progress, times = torch.cat((progress, anchor_s), 1), torch.cat((times, anchor_times), 1)
            order = torch.argsort(torch.cat((u, anchor_u), 1), dim=1, stable=True)
            inverse = torch.argsort(order, dim=1)
            def ordered(value):
                indices = order.reshape(*order.shape, *((1,)*(value.ndim-2))).expand_as(value)
                return value.gather(1, indices)
            link_quaternion = ordered(quaternion)
            linear, angular = ((None, None) if twist is None else problem.transform_tool_velocity(
                ordered(twist)[..., :3], ordered(twist)[..., 3:], link_quaternion))
            position, quaternion = problem.transform_tool_pose(ordered(position), link_quaternion)
            spheres = problem.transform_robot_spheres(ordered(spheres))
            q, dq, progress, times = ordered(q), ordered(dq), ordered(progress), ordered(times)
            if self._process_mode.endswith("cost"):
                value = _nominal_task_cost(problem, position, quaternion, progress, q,
                    self._joint_names, dq, trajectory_time=times,
                    tool_linear_velocity=linear, tool_angular_velocity=angular)
                value = value+self._scene.optimizer_preferences(progress, q, self._joint_names)
                terms = {"nominal_task": value}
            else:
                # BEGIN preserve component identity in one scene evaluation.
                # Only collision/* denotes geometry that a spatial seed search
                # can address. Controller and missing-model terms retain their
                # own names; splitting the sum changes neither weights nor
                # the constraint gradient, and makes no second scene query.
                terms = self._scene.components(spheres, position, quaternion, progress, q,
                    self._joint_names, trajectory_time=times)
            for name, value in terms.items():
                value = value.gather(1, inverse)
                # Invalid padded samples are excluded with where, which also
                # prevents a padded NaN from being revived by multiplication by 0.
                regular = torch.where(valid, value[:, :horizon], 0.)
                extra = value[:, horizon:].sum(1, keepdim=True)
                folded = torch.cat((torch.zeros_like(regular[:, :-1]), extra), 1)
                collection.add((regular+folded)[..., None]*budget.constraint_weight, name)
            # END named terms preserve the original scalar objective.
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

    def bind_request(new_problem, new_scene, new_progress, new_anchors, new_budget):
        nonlocal problem, scene, progress_map, anchors, budget
        problem, scene, progress_map, anchors, budget = new_problem, new_scene, new_progress, new_anchors, new_budget
        for manager in managers:
            manager._query_counts.update(calls=0, states=0)
            manager.bind_request_data()
    ProcessCostManagerCfg.bind_request = staticmethod(bind_request)
    ProcessCostManagerCfg.query_diagnostics = staticmethod(lambda: [dict(manager._query_counts)
        for manager in managers if manager._process_mode])
    return ProcessCostManagerCfg


def _trajectory_candidate(result, index):
    """Construct a single-seed native result without inventing pose metrics.

    V2's top-k method requires a pose goalset and assumes that active control
    knots and full joint states have equal DOF. Goal-free motion has no
    goalset, and locked fingers violate that DOF assumption. Slice explicit
    batch/seed tensors and use JointState's native indexer instead. Acceptance
    and objective diagnostics have already been read from the original result.
    """
    if result.batch_size != 1 or not 0 <= index < result.num_seeds:
        raise ValueError("Candidate extraction requires one batch and an existing seed index")
    if result.num_seeds == 1:
        return result
    candidate = replace(result, num_seeds=1, metrics=None, interpolated_metrics=None,
                        solution_state=None, debug_info={})
    for name in ("success", "solution", "optimized_seeds", "position_error", "rotation_error",
                 "cspace_error", "goalset_index", "seed_cost", "total_cost_reshaped", "feasible"):
        value = getattr(result, name)
        setattr(candidate, name, None if value is None else value[:, index:index+1].clone())
    candidate.seed_rank = None if result.seed_rank is None else result.seed_rank.new_zeros((1, 1))
    if result.interpolated_last_tstep is not None:
        candidate.interpolated_last_tstep = result.interpolated_last_tstep.reshape(1, -1)[:, index:index+1].clone()
    for name in ("js_solution", "interpolated_trajectory"):
        state = getattr(result, name)
        single = None if state is None else state[:, index:index+1].clone()
        if state is not None and state.dt is not None:
            # JointState's general indexer treats a leading singleton dt as
            # shared timing. Here dt is explicitly [batch,seed], so select
            # its seed column too, even though batch is exactly one.
            single.dt = state.dt[:, index:index+1].clone()
        setattr(candidate, name, single)
    return candidate


class _CuroboPlanner:
    """Resident native solvers with request-local online constraint inputs."""

    def __init__(self, problem, scene, robot, budget, device, *, robot_validator, continuation=None, goal_identities=None):
        self.ik = self.trajopt = None
        self.check_request = lambda: None
        # TF32 truncates even an identity matrix product, so its coordinate
        # quantization can exceed the physical collision tolerance. Precision
        # must cover backward as well as forward; a cost-only context is not
        # sufficient. Save caller settings before cuRobo's import side effects.
        import torch
        self._precision_torch = torch
        self._caller_precision = (torch.backends.cuda.matmul.allow_tf32,
                                  torch.backends.cudnn.allow_tf32)
        torch.backends.cuda.matmul.allow_tf32 = False
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
        robot_validator.validate_planning_binding(problem, scene.gripper)
        scene.validate_problem(problem, goal_identities=goal_identities)
        budget, initial_progress = (_initial_timing(problem, budget) if continuation is None
                                    else _continuation_timing(problem, budget, continuation))
        anchors = np.unique(np.concatenate((problem.anchors, np.asarray(scene.required_progress_anchors, float))))
        anchors = anchors[(anchors >= problem.start_progress) & (anchors <= problem.end_progress)]
        if (len(anchors) > budget.max_validation_points or not np.isfinite(anchors).all()
                or np.any(anchors < 0) or np.any(anchors > problem.end_progress)):
            raise ValueError("All required source knots and scene anchors must fit the finite planning horizon and budget")
        if problem.reference_frame != scene.reference_frame:
            raise ValueError("Problem and planning scene must use the same explicitly calibrated reference frame")
        self.progress_map = initial_progress
        if continuation is None:
            self.progress_map, _ = _progress_for_cubic_boundaries(initial_progress,
                budget.knots, stop_progress=problem.stop_progress, anchors=anchors)
        initial_clock = _fixed_spline_grid(intervals=budget.knots+4,
            native_dt=budget.maximum_trajectory_dt, interpolation_steps=budget.interpolation_steps,
            interpolation_dt=budget.interpolation_dt)
        if continuation is None:
            self.progress_map = _bind_event_clock(problem, self.progress_map, initial_clock["duration"],
                None if scene.gripper is None else scene.gripper.control_timestep_s)
        # IK, trajectory managers and independent verification each deep-copy
        # the scene. Binding only the parent after their construction leaves
        # those copies without a command/nominal polynomial. The admitted grid
        # already defines the exact native duration; verify it against the
        # constructed solver below without regenerating these commands.
        hand = scene.gripper
        if hand is not None:
            hand.bind_clock(initial_clock["duration"], self.progress_map)
        self.finger_trajectory = None if hand is None else hand.finger_trajectory
        self.anchor_capacity = _anchor_capacity(self.progress_map, anchors, budget)
        robot_dict, self.robot_metadata = _bounded_robot_configuration(robot, problem, scene, budget)
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
                != checker.model_record()["urdf_xml"]):
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
        workspace = _planning_workspace_bytes(scene, self.robot_metadata, budget, self.anchor_capacity)
        _require_planning_host_memory(workspace.cpu_live_bytes,
            max_host_rss_bytes=budget.max_host_rss_bytes)
        import torch
        from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
        from curobo.trajectory_optimizer import TrajectoryOptimizer, TrajectoryOptimizerCfg
        from curobo.kinematics import Kinematics
        from curobo.types import JointState, GoalToolPose
        from curobo._src.types.device_cfg import DeviceCfg
        from curobo._src.util.config_io import load_yaml, join_path
        from curobo.content import get_task_configs_path

        # Apply after native imports and before any native solver/FK buffers or
        # geometry calculations. cuDNN is not used here; preserve its caller
        # setting rather than leaking the dependency's unrelated assignment.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = self._caller_precision[1]
        self.problem, self.scene, self.budget = problem, scene, budget
        self.torch, self.JointState, self.GoalToolPose = torch, JointState, GoalToolPose
        self.device_cfg = DeviceCfg(device)
        workspace = _require_planning_memory(torch, self.device_cfg.device, workspace,
                                             max_device_used_bytes=budget.max_device_used_bytes)
        self.scene_workspace_bytes = workspace.total_bytes
        self.scene_cpu_workspace_bytes, self.scene_gpu_workspace_bytes = workspace.cpu_bytes, workspace.gpu_bytes
        self.anchor_fk_workspace_bytes = workspace.anchor_fk_bytes
        self.ik = self.trajopt = None
        self.anchors = anchors
        self.seed_parameter = self.device_cfg.to_device([[0.]])
        self.formal_clock = dict(initial_clock)
        self._manager_type = _process_manager_type(
            problem, scene, self.progress_map, self.anchors, budget,
            clock=self.formal_clock, seed_parameter=self.seed_parameter,
            )
        self.last_optimizer_diagnostics = []
        self.delivery_candidate = None

        def configuration(name, mode):
            data = deepcopy(load_yaml(join_path(get_task_configs_path(), name)))
            constraint = data["rollout"].setdefault("constraint_cfg", {})
            constraint.pop("scene_collision_cfg", None)
            # BEGIN remove proxy self costs from optimization and metrics alike.
            # NativeCollisionModel supplies self pairs inside the scene term.
            for manager in ("constraint_cfg", "cost_cfg", "convergence_cfg"):
                data["rollout"].get(manager, {}).pop("self_collision_cfg", None)
            # END cspace costs remain owned by the native solver.
            if mode.startswith("trajectory"):
                constraint["process_mode"] = "trajectory_constraint"
                data["rollout"].setdefault("cost_cfg", {})["process_mode"] = "trajectory_cost"
                # No nominal target accuracy participates in native success.
                data["rollout"]["convergence_cfg"] = {}
            else:
                constraint["process_mode"] = mode
            return data

        ik_optimizer = configuration("ik/lbfgs_ik.yml", "seed_optimizer")
        trajectory_optimizer = configuration("trajopt/lbfgs_bspline_trajopt.yml", "trajectory_optimizer")
        for name, config, iterations in (("IK", ik_optimizer, budget.ik_iterations),
                                         ("trajectory", trajectory_optimizer, budget.trajectory_iterations)):
            inner_iterations = config["optimizer"]["inner_iters"]
            if iterations % inner_iterations:
                raise ValueError(f"{name} iterations must be a multiple of native inner_iters={inner_iterations}")
        ik_cfg = InverseKinematicsCfg.create(
            robot=deepcopy(robot_dict), optimizer_configs=[ik_optimizer],
            metrics_rollout=configuration("metrics_base.yml", "seed_metrics"),
            device_cfg=self.device_cfg, num_seeds=budget.ik_seeds, seed_solver_num_seeds=budget.ik_seeds,
            max_batch_size=1, max_goalset=1, use_cuda_graph=False,
            position_tolerance=problem.numerical.position_m,
            orientation_tolerance=problem.numerical.rotation_rad,
            cost_manager_config_instance_type=self._manager_type,
            override_optimizer_num_iters={"lbfgs": budget.ik_iterations}, random_seed=budget.random_seed)
        # Keep the declared IK seed count and one native L-BFGS search. The
        # optional LM pre-solver has its own seed expansion and is unnecessary
        # for this finite reference initializer.
        ik_cfg.use_lm_seed = False
        # Native early exit only requires one successful seed per batch.
        # Even one requested trajectory can benefit from choosing among the
        # several IK postures already admitted by its separate seed budget.
        if budget.ik_seeds > 1:
            ik_cfg.exit_early = False
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
        transition = deepcopy(load_yaml(join_path(get_task_configs_path(), "trajopt/transition_bspline_trajopt.yml")))
        transition["transition_model_cfg"].update(n_knots=budget.knots+2,
            interpolation_steps=budget.interpolation_steps,
            dt_traj_params={"base_dt": budget.maximum_trajectory_dt,
                            "max_dt": budget.maximum_trajectory_dt, "base_ratio": 1.})
        self._trajectory_configuration = {
            "robot": deepcopy(robot_dict),
            "optimizer_configs": [trajectory_optimizer],
            "metrics_rollout": configuration("metrics_base.yml", "trajectory_metrics"),
            "transition_model": transition, "device_cfg": self.device_cfg,
            "num_seeds": budget.trajectory_seeds, "max_batch_size": 1, "max_goalset": 1,
            "use_cuda_graph": False, "random_seed": budget.random_seed,
            "position_tolerance": problem.numerical.position_m,
            "orientation_tolerance": problem.numerical.rotation_rad,
            "cost_manager_config_instance_type": self._manager_type,
            "transition_model_config_instance_type": cubic_transition_config_type(),
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


            def get_interpolated_trajectory(self, js_optimized):
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
                step = self.transition_model._rollout_step_fn
                if (step.action_horizon != self.action_horizon or step.knot_dt != grid["knot_dt"]):
                    raise ValueError("Cubic export changed the bound coefficient map or clock")
                state = step.sample(js_optimized.knot, goal.current_js, goal.idxs_current_js,
                    self._interpolated_traj_buffer, valid_count=grid["valid_count"],
                    dense=True, output_dt=grid["interpolation_dt"])
                rollout = self.additional_metrics_rollouts["interpolated_rollout"]
                # Both managers retain valid-prefix masking before the native
                # caller computes dense metrics and its original success flag.
                for manager in (rollout.cost_manager, rollout.metrics_cost_manager,
                                rollout.constraint_manager, rollout.metrics_constraint_manager):
                    manager.set_dense_counts(counts)
                return state, counts, False

        self._optimizer_class = ProcessTrajectoryOptimizer
        self._optimizer_config_class = TrajectoryOptimizerCfg
        self.trajopt = ProcessTrajectoryOptimizer(TrajectoryOptimizerCfg.create(**deepcopy(self._trajectory_configuration)))
        self._bind_formal_clock()
        self._initialize_verification_constraints()
        self._bind_seed_basis()
        self._manager_type.bind_request(problem, scene, self.progress_map, self.anchors, budget)

    def _bind_online_robot(self):
        params = self.ik.kinematics.config.kinematics_config
        spheres = params.link_spheres.detach().cpu().numpy().reshape(-1, 4)
        links = params.link_sphere_idx_map.detach().cpu().numpy().reshape(-1)
        names = {int(index): name for name, index in params.link_name_to_idx_map.items()}
        geometry = [{"link_name": names[int(link)], "sphere_indices": np.flatnonzero(links == link).tolist(),
                     "centers_local": spheres[links == link, :3].tolist(), "radii": spheres[links == link, 3].tolist()}
                    for link in np.unique(links)]
        validator = self.robot_validator
        # BEGIN share the registered solids and pair policy with scene queries.
        # This is Coal's existing self-pair numerical boundary, not a world
        # clearance or a task-specific allowance. World pairs use zero margin.
        margin = max(validator.checker.numerics["gjk_tolerance"],
                     validator.checker.numerics["epa_tolerance"])
        self.scene.bind_online_robot(robot_sphere_geometry=geometry,
            native_robot_model=validator.checker.model_record(),
            native_joint_names=validator.joint_names,
            native_self_pairs=[{"first": pair.first, "second": pair.second, "margin": margin}
                               for pair in validator.required_pairs],
            robot_geometry_source=self.scene.robot_geometry_source,
            robot_model={"urdf_path": str(validator.checker.urdf_path), "base_frame": validator.checker.base_frame,
                         "tool_frame": validator.tool_frame, "T_reference_base": validator.T_reference_base.tolist(),
                         "T_frame_tcp": validator.T_frame_tcp.tolist()})
        # END native and observed geometry share one saved scene binding.
        # BEGIN admit the entity derivative buffers before the first query.
        # The initial check precedes cuRobo sphere indexing. This binding now
        # knows the exact solid groups, joint count and owner transforms; only
        # their newly declared device storage is additional to that admission.
        workspace = _planning_workspace_bytes(self.scene, self.robot_metadata,
                                               self.budget, self.anchor_capacity)
        _require_planning_host_memory(workspace.cpu_live_bytes,
                                     max_host_rss_bytes=self.budget.max_host_rss_bytes)
        require_device_memory(self.torch, self.device_cfg.device, native_bytes=0,
            torch_bytes=max(0, workspace.gpu_bytes-self.scene_gpu_workspace_bytes),
            max_device_used_bytes=self.budget.max_device_used_bytes)
        self.scene_workspace_bytes = workspace.total_bytes
        self.scene_cpu_workspace_bytes = workspace.cpu_bytes
        self.scene_gpu_workspace_bytes = workspace.gpu_bytes
        # END no sphere-pair worst case substitutes for the actual entity model.


    def rebind(self, problem, scene, budget, *, continuation=None, goal_identities=None):
        """Reuse IK/TrajOpt while clearing all prior candidate and request state."""
        self.robot_validator.validate_planning_binding(problem, scene.gripper)
        scene.validate_problem(problem, goal_identities=goal_identities)
        budget, mapping = (_initial_timing(problem, budget) if continuation is None else
                           _continuation_timing(problem, budget, continuation))
        anchors = np.unique(np.r_[problem.anchors, scene.required_progress_anchors])
        anchors = anchors[(anchors >= problem.start_progress) & (anchors <= problem.end_progress)]
        if continuation is None:
            mapping, _ = _progress_for_cubic_boundaries(mapping, budget.knots,
                stop_progress=problem.stop_progress, anchors=anchors)
        grid = _fixed_spline_grid(intervals=budget.knots+4, native_dt=budget.maximum_trajectory_dt,
            interpolation_steps=budget.interpolation_steps, interpolation_dt=budget.interpolation_dt)
        if continuation is None:
            mapping = _bind_event_clock(problem, mapping, grid["duration"],
                None if scene.gripper is None else scene.gripper.control_timestep_s)
        anchor_capacity = _anchor_capacity(mapping, anchors, budget)
        workspace = _planning_workspace_bytes(scene, self.robot_metadata, budget, anchor_capacity)
        _require_planning_host_memory(workspace.cpu_live_bytes,
            max_host_rss_bytes=budget.max_host_rss_bytes)
        _require_planning_memory(self.torch, self.device_cfg.device, workspace,
            max_device_used_bytes=budget.max_device_used_bytes)
        if scene.gripper is not None:
            scene.gripper.bind_clock(grid["duration"], mapping)
        self.problem, self.scene, self.budget, self.progress_map, self.anchors = problem, scene, budget, mapping, anchors
        self.anchor_capacity = anchor_capacity
        self.scene_workspace_bytes = workspace.total_bytes
        self.scene_cpu_workspace_bytes, self.scene_gpu_workspace_bytes = workspace.cpu_bytes, workspace.gpu_bytes
        self.anchor_fk_workspace_bytes = workspace.anchor_fk_bytes
        self.formal_clock.clear()
        self.formal_clock.update(grid)
        self.finger_trajectory = None if scene.gripper is None else scene.gripper.finger_trajectory
        columns = [problem.joint_names.index(name) for name in self.joint_names]
        self.start = self.JointState.from_numpy(list(self.joint_names), problem.start_q[columns][None],
            velocity=problem.start_dq[columns][None], device_cfg=self.device_cfg)
        self._bind_online_robot()
        self._manager_type.bind_request(problem, scene, mapping, anchors, budget)
        self.trajopt.config.minimum_trajectory_dt = budget.maximum_trajectory_dt
        self.trajopt.config.maximum_trajectory_dt = budget.maximum_trajectory_dt
        transitions = {id(rollout.transition_model): rollout.transition_model
            for rollout in [*self.trajopt.core.get_all_rollout_instances(),
                            *self.trajopt.additional_metrics_rollouts.values()]}
        for rollout in [*self.trajopt.core.get_all_rollout_instances(),
                        *self.trajopt.additional_metrics_rollouts.values()]:
            transitions[id(rollout.metrics_transition_model)] = rollout.metrics_transition_model
        for transition in transitions.values():
            transition.update_traj_dt(budget.maximum_trajectory_dt)
        self.trajopt.config.interpolation_dt = budget.interpolation_dt
        if self.trajopt._interpolation_dt_buffer is not None:
            self.trajopt._interpolation_dt_buffer.fill_(budget.interpolation_dt)
        self._bind_formal_clock()
        self.seed_parameter.zero_()
        self.last_validation, self.last_optimizer_diagnostics = None, []
        self.delivery_candidate = None
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
        self._bind_seed_basis()
        return self

    def _bind_formal_clock(self):
        native = _formal_native_clock(self.trajopt)
        if any(native.get(name) != value for name, value in self.formal_clock.items()):
            raise ValueError("Native solver changed the admitted synchronized clock")
        self.formal_clock.update(native)


    def _initialize_verification_constraints(self):
        from curobo._src.rollout.cost_manager.cost_manager_robot import RobotCostManager
        # The base manager supplies joint limits for the measured
        # entrance and ordinary OMPL states. Trajectory metrics own their
        # sampled motion checks; final export checks polynomial joint bounds.
        self._verification_native = RobotCostManager(self.device_cfg)
        self._verification_native.initialize_from_config(
            deepcopy(self.trajopt.config.metrics_rollout_config.constraint_cfg),
            self.trajopt.transition_model)

    def _bind_seed_basis(self):
        """Only the shared cubic basis is owned here; cuRobo owns its search."""
        self.seed_curve = CubicSeed(self.start.position, self.start.velocity,
            self.formal_clock["knot_dt"], self.trajopt.action_horizon)


    def initial_state_diagnostics(self):
        """Check the measured entrance against basic joint and model bounds."""
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
            # BEGIN one scene call checks native self, owners and observed world.
            index = state.tool_poses.tool_frames.index(self.problem.tool_link)
            position, quaternion = self.problem.transform_tool_pose(
                state.tool_poses.position[:, :, index], state.tool_poses.quaternion[:, :, index])
            progress = q.new_full(q.shape[:2], self.problem.start_progress)
            detail = {}
            # BEGIN this aggregate also includes controller/model conditions.
            components["scene_constraints"] = _known_scene_cost(self.scene,
                self.problem.transform_robot_spheres(state.robot_spheres),
                position, quaternion, progress, q, self.joint_names,
                trajectory_time=q.new_zeros(q.shape[:2]), component_diagnostics=detail)
            # END scene_components below retains each exact diagnostic name.
            # END no second proxy-self rejection follows the entity query.
            residuals = {name: float(value.max().item()) for name, value in components.items()}
        failed = [name for name, value in residuals.items() if not np.isfinite(value) or value > 0]
        return {"passed": not failed, "residuals": residuals, "failed_components": failed,
                "scene_components": detail, "scope": "measured joint bounds and named scene constraints"}


    def close(self):
        # Native destroy() resets CUDA graphs. LBFGS and GradientOptCore also
        # retain each other through bound callbacks, so drop every solver
        # reference before collecting their now-unreachable tensor workspaces.
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

    def _goal(self, progress):
        s = self.device_cfg.to_device([[float(progress)]])
        position, quaternion = self.problem.reference_to_link_pose(*self.problem.sample_reference(s))
        return self.GoalToolPose(tool_frames=[self.problem.tool_link],
                                 position=position[:, :, None, None].contiguous(),
                                 quaternion=quaternion[:, :, None, None].contiguous())


    def reference_seeds(self, joint_reference=None):
        """Fit finite native IK samples as soft initial guesses on the fixed clock.

        Native IK orders its own alternatives. Its pose success is recorded,
        never used to reject the complete motion request. The subsequent native
        trajectory solve must optimize the full nominal path and events.
        """
        torch, count = self.torch, self.budget.trajectory_seeds
        self.reference_ik_queries, success_counts, self.reference_seed_fits = [], [], []
        self.joint_reference_report = None
        if joint_reference is not None:
            if not isinstance(joint_reference, JointTrajectory) or set(joint_reference.joint_names) != set(self.joint_names):
                raise MotionPlanningError("unsupported_reference", "Joint reference must name the robot's active joints")
            columns = [joint_reference.joint_names.index(name) for name in self.joint_names]
            source_map = _ProgressMap(joint_reference.progress_u, joint_reference.progress_s)
            parameters = _phase_aligned_parameters(source_map, self.progress_map,
                joint_reference.times/joint_reference.duration)
            controls, report = self.seed_curve.fit(parameters, joint_reference.position[:, columns])
            seeds = controls[None, None].expand(1, count, -1, -1).contiguous()
            self.reference_seed_fits = [report]*count
            self.initial_seed_strategies = ["joint_reference_least_squares"]*count
            self.joint_reference_report = {"fit": report, "source_curve_preserved": False,
                "source_duration_s": joint_reference.duration, "bound_duration_s": self.formal_clock["duration"]}
        else:
            parameters = self.seed_curve.basis.reference_parameters
            previous = self.start.position[:, None].expand(-1, self.budget.ik_seeds, -1).contiguous()
            targets = []
            for u in parameters:
                self.check_request()
                self.seed_parameter.fill_(float(u))
                progress = float(np.interp(u, self.progress_map.u, self.progress_map.s))
                result = self.ik.solve_pose(self._goal(progress), current_state=self.start,
                    seed_config=previous, return_seeds=self.budget.ik_seeds)
                values = self.ik.get_active_js(result.js_solution).position.detach()
                if values.shape != previous.shape or not bool(torch.isfinite(values).all()):
                    raise MotionPlanningError("invalid_seed", "Native IK returned an invalid finite joint layout")
                previous = values.contiguous()
                targets.append(values[0, :count])
                solved = int(result.success.sum().item())
                success_counts.append(solved)
                self.reference_ik_queries.append({"u": float(u), "native_success_count": solved,
                    "scope": "nominal initial guess, not a task acceptance test"})
                self.check_request()
            observations = torch.stack(targets, dim=1)
            fitted = [self.seed_curve.fit(parameters, row) for row in observations]
            seeds = torch.stack([row[0] for row in fitted])[None]
            self.reference_seed_fits = [row[1] for row in fitted]
            self.initial_seed_strategies = ["native_ik_reference"]*count
        self.seed_strategies = list(self.initial_seed_strategies)
        return seeds.contiguous(), success_counts


    def ompl_seed(self, candidate_trajectory):
        """Use one bounded native spatial connection as another numerical seed.

        Known collision geometry is evaluated at the connection's phase. The
        whole timed motion is checked by the following native trajectory solve;
        this spatial initializer never proves task or contact success.
        """
        deadline = time.monotonic()+self.budget.search_seconds
        limits = self._verification_native.get_cost("cspace").config.joint_limits
        columns = [limits.joint_names.index(name) for name in self.joint_names]
        lower, upper = limits.position[:, columns].detach().cpu().numpy()
        boundaries = np.unique(np.r_[0., self.progress_map.anchor_parameters(
            self.scene.required_progress_anchors)[:, 0], 1.])
        q, _, _, _ = candidate_trajectory.sample(boundaries*candidate_trajectory.duration)
        # A single spatial query after release has no history from which to
        # locate the placed owner. The candidate's boundary FK supplies that
        # history for each segment; full optimization keeps releases variable.
        boundary_q = self.device_cfg.to_device(q)[None].contiguous()
        boundary_js = self.JointState.from_position(boundary_q, joint_names=list(self.joint_names))
        with self.torch.no_grad():
            boundary_state = self._verification_fk.compute_kinematics(boundary_js)
            tool_index = boundary_state.tool_poses.tool_frames.index(self.problem.tool_link)
            boundary_p, boundary_r = self.problem.transform_tool_pose(
                boundary_state.tool_poses.position[:, :, tool_index],
                boundary_state.tool_poses.quaternion[:, :, tool_index])
        boundary_s = self.device_cfg.to_device(np.interp(boundaries, self.progress_map.u,
                                                         self.progress_map.s))[None]
        query_count, paths, stages = 0, [], []
        for part, (left, right) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            self.check_request()
            stage_scene = self.scene.with_release_history(boundary_p[:, :part+1],
                boundary_r[:, :part+1], boundary_s[:, :part+1])
            query_time = .5*(left+right)*self.formal_clock["duration"]
            query_progress = float(np.interp(.5*(left+right), self.progress_map.u, self.progress_map.s))
            def valid(value):
                nonlocal query_count
                query_count += 1
                self.check_request()
                tensor = self.device_cfg.to_device(value)[None, None].contiguous()
                js = self.JointState.from_position(tensor, joint_names=list(self.joint_names))
                with self.torch.no_grad():
                    state = self._verification_fk.compute_kinematics(js)
                    # BEGIN ordinary search uses the same scene collision terms.
                    times = tensor.new_full((1, 1), query_time)
                    index = state.tool_poses.tool_frames.index(self.problem.tool_link)
                    position, rotation = self.problem.transform_tool_pose(state.tool_poses.position[:, :, index],
                        state.tool_poses.quaternion[:, :, index])
                    progress = tensor.new_full((1, 1), query_progress)
                    value = stage_scene.evaluate(self.problem.transform_robot_spheres(state.robot_spheres),
                        position, rotation, progress, tensor, self.joint_names, trajectory_time=times)
                    # END no separate sphere self predicate changes acceptance.
                    return bool(self.torch.isfinite(value).all() & (value <= 0).all())
            remaining = self.budget.max_validation_points-query_count
            if remaining < 2 or time.monotonic() >= deadline:
                return None, {"failure": "bounded_search_exhausted", "state_queries": query_count, "stages": stages}
            path, report = _ompl_stage(q[part], q[part+1], lower, upper, valid,
                step=self.budget.validation_joint_step, seconds=max(0., deadline-time.monotonic()),
                sample_limit=remaining)
            stages.append(report)
            if path is None:
                return None, {"failure": report["failure"], "state_queries": query_count, "stages": stages}
            parameter = np.linspace(left, right, len(path))
            paths.append((parameter if part == 0 else parameter[1:], path if part == 0 else path[1:]))
        controls, fit = self.seed_curve.fit(np.concatenate([row[0] for row in paths]),
                                            np.concatenate([row[1] for row in paths]))
        seeds = controls[None, None].expand(1, self.budget.trajectory_seeds, -1, -1).contiguous()
        return seeds, {"state_queries": query_count, "stages": stages, "fit": fit}


    def _native_request(self, seed):
        """Bind a goal-free native buffer; nominal objectives live in cost_cfg."""
        from curobo._src.solver.solve_mode import SolveMode
        from curobo._src.solver.solve_state import SolveState
        self.route = "native_nominal_cost"
        self.trajopt.disable_tool_pose_tracking()
        self.trajopt.disable_joint_position_tracking()
        return SolveState(solve_type=SolveMode.BATCH, num_trajopt_seeds=seed.shape[1],
            batch_size=1, num_envs=1, num_goalset=1, tool_frames=[]), None


    def evaluate_candidate(self, trajectory, *, native_report):
        """Complete one seed's actual polynomial checks before seed selection."""
        attempt = deepcopy(native_report)
        validation = self._candidate_domain_validation(trajectory)
        residuals = {**attempt["residuals"], **trajectory.residuals}
        attempt.update(residuals=residuals, accepted=bool(attempt["accepted"])
            and all(np.isfinite(value) and value <= 0 for value in residuals.values()))
        validation.update(u=np.array([0., 1.]), violations=np.full(2, max(residuals.values(), default=0.)),
            scope="native known-model checks and actual exported polynomial; task costs are diagnostic")
        self.delivery_candidate = trajectory
        self.last_validation = validation if attempt["accepted"] else None
        return trajectory, attempt, 0


    def optimize(self, seed):
        """Run one native solve on admitted seeds with the unchanged clock."""
        state, goal = self._native_request(seed)
        initial = seed.detach().clone()
        result = self.trajopt._solve_impl(current_state=self.start, solve_state=state,
            goal_tool_poses=goal, seed_traj=seed, num_seeds=seed.shape[1], return_seeds=seed.shape[1],
            dt=self.torch.full((1, seed.shape[1]), self.budget.maximum_trajectory_dt,
                               dtype=seed.dtype, device=seed.device),
            use_implicit_goal=False, goal_state=None, finetune_attempts=0, finetune_dt_scale=1.)
        if result.solution.shape != initial.shape:
            raise ValueError("Native solve changed the admitted seed layout")
        self.last_optimizer_diagnostics = [{"seed_index": i,
            "maximum_control_update": float((result.solution[0, i]-initial[0, i]).abs().max().item())}
            for i in range(seed.shape[1])]
        return result

    def native_diagnostics(self, result, seed_index):
        """Report physical constraints separately from nominal objective values."""
        report = {"success": bool(result.success[0, seed_index].item())}
        for label, metrics in (("rollout", result.metrics), ("interpolated", result.interpolated_metrics)):
            if metrics is None:
                continue
            collection = metrics.costs_and_constraints
            constraints = {name: _finite_maximum(value[seed_index]) for name, value in
                zip(collection.constraints.names, collection.constraints.values)}
            costs = {name: float(value[seed_index].detach().sum().item()) for name, value in
                zip(collection.costs.names, collection.costs.values)}
            report[label] = {"constraints": constraints, "costs": costs}
        return report


    def select_candidate(self, result):
        """Compare the native reporting score after each actual curve is checked.

        The score retains the existing nominal-task preference among feasible
        results. It is a selection score, not the L-BFGS optimization objective,
        which additionally contains native joint-state regularization costs.
        """
        total = result.metrics.costs_and_constraints.get_sum_cost_and_constraint(sum_horizon=True)
        objectives = total.detach().reshape(-1).cpu().numpy()
        # BEGIN finish the same physical checks for every returned seed.
        # A low reporting score cannot select a curve whose between-sample
        # extrema fail while another returned curve already satisfies them.
        reports, trajectories, validations = [], [], []
        ranking_scores = objectives.copy()
        self.delivery_candidate, self.last_validation = None, None
        for index in range(result.num_seeds):
            metrics = self.native_diagnostics(result, index)
            residuals = {f"{label}:{name}": value for label in ("rollout", "interpolated")
                for name, value in metrics.get(label, {}).get("constraints", {}).items()}
            finite = np.isfinite(objectives[index]) and bool(self.torch.isfinite(result.solution[:, index]).all())
            accepted = (finite and metrics["success"]
                        and all(np.isfinite(v) and v <= 0 for v in residuals.values()))
            report = {"seed_index": index, "accepted": bool(accepted),
                "objective": float(objectives[index]), "native_success": metrics["success"],
                "native_metrics": metrics, "residuals": residuals,
                "collision_residuals": {name: value for name, value in residuals.items()
                    if name.split(":", 1)[1].startswith("collision/")},
                "seed_provenance": self.seed_strategies[index],
                "optimizer": self.last_optimizer_diagnostics[index] if self.last_optimizer_diagnostics else {}}
            candidate = _trajectory_candidate(result, index)
            native = self.trajopt.get_active_js(candidate.get_interpolated_plan())
            finite_curve = bool(self.torch.isfinite(native.position).all()
                                and self.torch.isfinite(native.velocity).all())
            trajectory, validation = None, None
            if finite_curve:
                trajectory, report, _ = self.evaluate_candidate(
                    self._native_curve(candidate), native_report=report)
                validation = self.last_validation
            else:
                # A nonfinite state has no polynomial domain to evaluate. It
                # excludes this seed without swallowing clock/layout errors
                # or preventing another finite returned seed from selection.
                report["accepted"] = False
                report["residuals"]["finite_joint_curve"] = float("inf")
            if not finite or not finite_curve:
                ranking_scores[index] = np.nan
            reports.append(report)
            trajectories.append(trajectory)
            validations.append(validation)
        selected = _minimum_cost_candidate(ranking_scores, [row["accepted"] for row in reports])
        self.delivery_candidate = trajectories[selected]
        self.last_validation = validations[selected]
        # END selected diagnostics and delivery reuse this exact checked curve.
        return trajectories[selected], {**reports[selected],
            "selected_seed_index": selected, "candidates": reports}


    def _native_curve(self, result):
        """Copy the bound native cubic without independent geometry queries."""
        if result.batch_size != 1 or result.num_seeds != 1:
            raise ValueError("Export requires exactly one native batch and one selected seed")
        native = self.trajopt.get_active_js(result.get_interpolated_plan())
        position = native.position.detach().cpu().numpy().reshape(-1, len(self.joint_names))
        velocity = native.velocity.detach().cpu().numpy().reshape(position.shape)
        dt = float(native.dt.detach().cpu().reshape(-1)[0])
        if (len(position) < 2 or not np.isfinite(position).all() or not np.isfinite(velocity).all()
                or not np.isfinite(dt) or dt <= 0):
            raise ValueError("Native interpolation produced an invalid finite joint curve")
        times = np.arange(len(position), dtype=float) * dt
        if (len(times) != self.formal_clock["valid_count"] or dt != self.formal_clock["interpolation_dt"]
                or times[-1] != self.formal_clock["duration"]):
            raise ValueError("Native candidate changed the physical clock bound to the finger reference")
        event_records = []
        for event in self.problem.events:
            clock = self.progress_map.event_clock[event_occurrence_key(event)]
            if clock["event"] != asdict(event):
                raise ValueError("Exported event differs from its frozen occurrence binding")
            event_records.append({**deepcopy(clock["event"]),
                                  **{key: deepcopy(value) for key, value in clock.items() if key != "event"}})
        trajectory = JointTrajectory(self.joint_names, times, position, velocity,
                                     self.progress_map.u.copy(), self.progress_map.s.copy(),
                                     tuple(event_records), deepcopy(self.problem.conditions), {})
        trajectory.finger_trajectory = self.finger_trajectory
        trajectory.satisfied_dependencies = tuple(getattr(self, "satisfied_dependencies", ()))
        return trajectory

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
        entrance_columns = [self.joint_names.index(name) for name in trajectory.joint_names]
        trajectory.residuals = {
            "continuous_position": position["maximum_bound_excess"],
            "continuous_velocity": velocity["maximum_bound_excess"],
            "continuous_acceleration": derivatives["acceleration"]["maximum_bound_excess"],
            "initial_position": max(0., float(np.max(np.abs(trajectory.position[0]-self.start.position[0].detach().cpu().numpy()[entrance_columns])))-self.problem.numerical.joint),
            "initial_velocity": max(0., float(np.max(np.abs(trajectory.velocity[0]-self.start.velocity[0].detach().cpu().numpy()[entrance_columns])))-self.problem.numerical.velocity)}
        return {
            "continuous_position": position, "continuous_velocity": velocity, "cubic_derivatives": derivatives,
            "numerical_tolerances": asdict(self.problem.numerical)}

    def export(self, trajectory, native_report):
        """Deliver the selected curve with its already completed domain checks."""
        # BEGIN delivery does not reconstruct or re-evaluate a selected curve.
        if not native_report["accepted"]:
            raise MotionPlanningError("output_inconsistent", "An unqualified candidate cannot be delivered")
        return trajectory, 0
        # END last_validation remains the selected seed's completed report.


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


def plan_trajectory(problem, scene, robot, *, robot_validator, budget=None, device="cuda:0", continuation=None,
                    request_ledger=None, rebind=None, backend=None, cancelled=None, deadline=None, joint_reference=None):
    """Generate one fully checked candidate within the original request budget.

    ``request_ledger`` continues failed planning attempts. ``continuation``
    carries actual execution facts. An explicit ``rebind`` rebuilds the soft
    schedule of remaining work or creates selected retry occurrences; ordinary
    continuation preserves the original frozen clock. Every request admitted
    including initialization failure carries the cumulative ledger. No GPU stage is
    claimed to be preemptible: elapsed overrun rejects its candidate.
    """
    from .execution_binding import bind_candidate, plain
    from .planning_problem import TimeBindingError
    started = time.monotonic()
    prior = planner = ledger = trajectory = None
    initial_seconds = 0.
    motion_ready, motion_baseline = False, {}
    owned_backend = backend is None
    backend_claimed = False
    diagnostics = {"attempts": []}
    diagnostics["wallclock"] = {"scope": "One planning request; Unix timestamps join processes, elapsed seconds use the request monotonic clock", "events": {}}
    def mark_wallclock(name):
        diagnostics["wallclock"]["events"][name] = {
            "unix_time_ns": time.time_ns(), "elapsed_seconds": time.monotonic()-started}
    mark_wallclock("planning_request_started")
    outcome = PlanningResult(False, None, diagnostics)
    stage = "request_binding"
    try:
        scene = scene.clone_for_rollout()
        previous = request_ledger
        if previous is None and isinstance(continuation, dict) and isinstance(continuation.get("original_binding"), dict):
            previous = continuation.get("request_ledger", continuation["original_binding"].get("request_ledger"))
        if previous is not None:
            original_budget = PlanningBudget(**previous["planning_budget"])
        else:
            original_budget = PlanningBudget() if budget is None else budget
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
        check("binding_attempts")
        consumed["binding_attempts"] += 1
        # Validate and retain the old ledger before considering the new limits,
        # so an invalid change still returns this attempt's cumulative work.
        if previous is not None and budget is not None and asdict(budget) != asdict(original_budget):
            ledger = _request_work_ledger(problem, budget, ledger)
            original_budget = budget
            consumed = ledger["consumed_planning"]
            request_budget = asdict(original_budget)
            diagnostics.update(planning_budget=request_budget, consumed_planning=consumed)
        # Initial clock construction can bind a hand before native allocation
        # fails. A private problem is therefore required for both first and
        # subsequent attempts, just as for remaining-obligation filtering.
        problem = deepcopy(problem)
        if previous is not None:
            _restore_original_direction_origins(problem, ledger["obligation_contract"])
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
            ledger = _request_work_ledger(problem, original_budget, ledger, continuation)
            consumed = ledger["consumed_planning"]
            diagnostics["consumed_planning"] = consumed
            diagnostics.pop("continuation_facts_pending_validation", None)
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
        mark_wallclock("backend_binding_started")
        planner = backend.bind_request(problem, scene, budget=budget, continuation=prior,
                                       goal_identities=block_record["goal_identities"])
        mark_wallclock("backend_binding_completed")
        planner.check_request = check
        planner.satisfied_dependencies = tuple(sorted(satisfied_dependencies))
        check()
        diagnostics.update(backend="curobo_v2", batch_size=1, ik_seeds=budget.ik_seeds,
            trajectory_seeds=budget.trajectory_seeds, used_ompl=False,
            timing_policy="explicit_rebind_then_frozen" if rebind is not None else "source_proposal_then_frozen_binding",
            native_trajectory_dt=planner.budget.maximum_trajectory_dt,
            progress_parameter_error_bound=_PROGRESS_PARAMETER_ERROR, progress_clock_max_keys=_MAX_PROGRESS_KEYS,
            scene=scene.describe(), scene_workspace_bytes=planner.scene_workspace_bytes,
            scene_cpu_workspace_bytes=planner.scene_cpu_workspace_bytes,
            scene_gpu_workspace_bytes=planner.scene_gpu_workspace_bytes,
            anchor_fk_workspace_bytes=planner.anchor_fk_workspace_bytes, exact_anchor_capacity=planner.anchor_capacity)
        diagnostics["optimizer_core_types"] = {
            name: [f"{type(optimizer._core).__module__}.{type(optimizer._core).__name__}"
                   for optimizer in solver.core.optimizers]
            for name, solver in (("ik", planner.ik), ("trajectory", planner.trajopt))}
        limits = planner._verification_native.get_cost("cspace").config.joint_limits
        diagnostics["verification_joint_limits"] = {"joint_names": list(limits.joint_names),
            **{name: getattr(limits, name).detach().cpu().numpy().tolist()
               for name in ("position", "velocity", "acceleration")}}
        stage = "initial_state_verification"
        consumed["validation_queries"] += 1
        diagnostics["initial_state"] = planner.initial_state_diagnostics()
        check()
        if not diagnostics["initial_state"]["passed"]:
            diagnostics["failure"] = "initial_state_infeasible"
            return outcome
        def qualify_candidate(trajectory, validation, attempt, *, binding_scene=None):
            if not attempt["accepted"] or validation is None:
                raise ValueError("Only a completely validated candidate may enter execution binding")
            diagnostics.update(executable_until=trajectory.executable_until,
                               next_verification=trajectory.next_verification)
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
                completed_goal_ids=[deepcopy(value) for value in
                                    (ledger["completed_goal_ids"] if prior is None else prior["completed_goal_ids"])
                                    if tuple(value) not in retry_goal_ids],
                goal_evidence=deepcopy(ledger["goal_evidence"]),
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
            check()
            diagnostics["qualified_candidate_elapsed_seconds"] = time.monotonic()-started
            outcome.success, outcome.trajectory, outcome.execution_binding = True, trajectory, binding
            mark_wallclock("qualified_candidate_ready")
            return True

        if joint_reference is not None and prior is not None:
            raise MotionPlanningError("unsupported_reference", "A complete reference cannot replace an actual continuation suffix")
        grid = _fixed_spline_grid(intervals=planner.budget.knots+4,
            native_dt=planner.budget.maximum_trajectory_dt, interpolation_steps=planner.budget.interpolation_steps,
            interpolation_dt=planner.budget.interpolation_dt)
        source_seconds = problem.source_time_map["source_times"]
        source_duration = float(source_seconds[-1]-source_seconds[0])
        diagnostics["reference_clock"] = {**grid, "source_duration_s": source_duration,
            "duration_change_s": grid["duration"]-source_duration,
            "scope": "Source duration is a soft proposal; native, dense, event and finger consumers share the bound clock"}
        diagnostics["event_time_checks"] = _validate_bound_event_times(problem, planner.progress_map, grid["duration"],
            0. if prior is None else prior["request_time_origin_s"], ledger["history"],
            None if prior is None else prior["goal_identities"], ledger["timing_events"])

        check("ik_batches")
        consumed["ik_batches"] += 1
        stage = "reference_ik"
        mark_wallclock('reference_seed_started')
        seeds, ik_success = planner.reference_seeds(joint_reference=joint_reference)
        mark_wallclock('reference_seed_completed')
        consumed['ik_queries'] += len(planner.reference_ik_queries)
        check()
        diagnostics.update(reference_ik_success_counts=ik_success,
            ik_exit_early=bool(planner.ik.config.exit_early),
            initial_seed_strategies=planner.initial_seed_strategies,
            reference_ik_queries=deepcopy(planner.reference_ik_queries),
            reference_seed_fits=deepcopy(planner.reference_seed_fits))
        diagnostics["joint_reference"] = deepcopy(planner.joint_reference_report)
        check("trajectory_solves")
        consumed["trajectory_solves"] += 1
        stage = "main_solve"
        mark_wallclock("main_solve_started")
        trajectory, attempt = planner.select_candidate(planner.optimize(seeds))
        mark_wallclock("main_solve_completed")
        diagnostics["attempts"].append(deepcopy(attempt))
        check()
        recovery = None
        collision_failed = any(value > 0 for value in attempt["collision_residuals"].values())
        if (not attempt["accepted"] and collision_failed and budget.search_seconds > 0
                and consumed["trajectory_solves"]-motion_baseline.get("trajectory_solves", 0)
                < original_budget.maximum_trajectory_solves):
            stage, recovery = "bounded_seed_search", "ompl"
            search_started = time.monotonic()
            mark_wallclock("search_started")
            recovered, report = planner.ompl_seed(trajectory)
            mark_wallclock("search_completed")
            consumed["search_seconds"] += time.monotonic()-search_started
            consumed["search_queries"] += report.get("state_queries", 0)
            diagnostics["ompl"] = report
            check()
            if recovered is None:
                recovery = "ompl_failed"
            else:
                check("trajectory_solves")
                consumed["trajectory_solves"] += 1
                stage = "connection_solve"
                planner.seed_strategies = ["native_ompl_connection"]*recovered.shape[1]
                mark_wallclock("connection_solve_started")
                trajectory, attempt = planner.select_candidate(planner.optimize(recovered))
                mark_wallclock("connection_solve_completed")
                diagnostics["attempts"].append(deepcopy(attempt))
                check()
        diagnostics.update(attempt, route=planner.route, recovery=recovery,
                           used_ompl=recovery in {"ompl", "ompl_failed"})
        if not attempt["accepted"]:
            diagnostics.update(failure="no_model_feasible_candidate_found", diagnostic_trajectory=trajectory,
                failure_scope="The finite native candidates failed basic motion/model checks; this is not proof that the task has no solution")
            return outcome
        stage = "delivery"
        check()
        mark_wallclock("validation_started")
        trajectory, points = planner.export(trajectory, attempt)
        mark_wallclock("validation_completed")
        # Every returned seed completed its polynomial checks before selection.
        # Delivery reuses the selected curve and native geometry verdict, with
        # zero new queries. Candidate binding below still checks wall/work.
        if points:
            consumed["validation_passes"] += 1
            consumed["validation_queries"] += points
        qualify_candidate(trajectory, planner.last_validation, attempt)
        diagnostics["returned_candidate"] = deepcopy(attempt)
        limits = planner._verification_native.get_cost("cspace").config.joint_limits
        outcome.execution_binding["verification_joint_limits"] = deepcopy(diagnostics["verification_joint_limits"])
        return outcome
    except Exception as error:
        import traceback
        diagnostics.update(failure=(error.reason if isinstance(error, MotionPlanningError)
            else "cumulative_planning_budget_exhausted" if isinstance(error, PlanningWorkExhausted)
            else "continuation_binding_gap" if isinstance(error, ContinuationBindingError)
            else "event_time_requirement_unsatisfied" if isinstance(error, TimeBindingError)
            else "planning_stage_failed"), failed_stage=stage, reason=str(error), error_type=type(error).__name__,
            failure_traceback=traceback.format_exc())
        candidate = (getattr(planner, "delivery_candidate", None) if planner is not None else None)
        candidate = trajectory if candidate is None else candidate
        if isinstance(candidate, JointTrajectory):
            diagnostics["diagnostic_trajectory"] = candidate
            diagnostics["diagnostic_candidate_scope"] = "last completed native candidate; no execution authorization"
        outcome.success, outcome.trajectory, outcome.execution_binding = False, None, None
        return outcome
    finally:
        manager_type = getattr(planner, "_manager_type", None)
        if manager_type is not None:
            diagnostics["rollout_query_counts"] = manager_type.query_diagnostics()
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
            outcome.request_ledger = deepcopy(ledger)
            if outcome.execution_binding is not None:
                outcome.execution_binding["consumed_planning"] = deepcopy(ledger["consumed_planning"])
                outcome.execution_binding["request_ledger"] = deepcopy(ledger)
# END native attempt-planner rebuild.
