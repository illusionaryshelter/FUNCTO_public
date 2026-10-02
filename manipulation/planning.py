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
from types import SimpleNamespace

import numpy as np
from scipy.interpolate import BSpline, CubicHermiteSpline
from scipy.optimize import linear_sum_assignment

# BEGIN reusable physical-domain contract for native numerical iterates.
from .physical_joint_domain import (CubicJointDomain, JointPositionDomain,
                                    JointDomainInitializationError, cubic_position_report)
# END importing this CPU module does not allocate any CUDA workspace.


# BEGIN one exact output grid for the unchanged native cubic clock.
def _fixed_spline_grid(*, intervals, native_dt, interpolation_steps, interpolation_dt):
    """Admit an output grid only when it preserves the native physical clock.

    The installed dense kernel reconstructs its knot interval as float32
    ``output_dt * subdivisions``. Check that product AND the saved binary64
    time axes before using it. Rounding only proposes an integer subdivision
    count; it never authorizes retiming an incompatible input.
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


@dataclass(frozen=True)
class PlanningBudget:
    """Explicit local resource and numerical-work limits, for a single request.

    These defaults bound the current 16 GB RAM / 12 GB GPU integration. More
    seeds or a larger batch must not be silently allocated by native defaults.
    The interpolation buffer covers the largest allowed native knot interval.
    Native solvers use eager execution for the dynamic process/scene tensors.
    Required source vertices use their actual count; ``max_refinement_points``
    bounds only additional queries. Both share ``max_validation_points`` and
    the checked scene/native allocation budgets.
    ``maximum_trajectory_dt`` fixes both native time bounds for this request:
    process feasibility and evidence stops take precedence over compression.
    """

    ik_seeds: int = 8
    trajectory_seeds: int = 2
    knots: int = 8
    interpolation_steps: int = 4
    ik_iterations: int = 40
    trajectory_iterations: int = 100
    interpolation_dt: float = 0.025
    maximum_trajectory_dt: float = 0.2
    interpolation_buffer_size: int = 512
    constraint_weight: float = 100000.0
    # BEGIN reserve optional refinement work separately from required evidence.
    max_refinement_points: int = 64
    # END reserve optional refinement work separately from required evidence.
    max_robot_spheres: int = 256
    max_robot_dof: int = 16
    max_robot_links: int = 64
    max_scene_workspace_bytes: int = 1024**3
    max_validation_points: int = 4096
    validation_joint_step: float = 0.025
    search_seconds: float = 3.0
    maximum_trajectory_solves: int = 3
    random_seed: int = 123

    def __post_init__(self):
        if (not 1 <= self.trajectory_seeds <= 2 or not self.trajectory_seeds <= self.ik_seeds <= 8
                or not 6 <= self.knots <= 24 or not 1 <= self.interpolation_steps <= 8
                or not 1 <= self.ik_iterations <= 200 or not 1 <= self.trajectory_iterations <= 300
                or not 1 <= self.max_robot_spheres <= 512
                or not 1 <= self.max_robot_dof <= 16
                or not 1 <= self.max_robot_links <= 64
                or not 32 <= self.max_validation_points <= 8192
                or not 1 <= self.maximum_trajectory_solves <= 4):
            raise ValueError("Planning budget exceeds the bounded local single-batch configuration")
        # BEGIN bound optional samples without capping a skill's source knots.
        if (type(self.max_refinement_points) is not int
                or not 0 <= self.max_refinement_points <= self.max_validation_points):
            raise ValueError("Refinement points must fit the finite validation-point budget")
        # END bound optional samples without capping a skill's source knots.
        # BEGIN allow complete observed robots within the existing 12-GiB GPU.
        # The default stays one GiB. Larger explicit requests still pass the
        # actual-device check with separate anchor-FK and two-GiB native
        # allowances; current observation fields already consume free memory.
        if type(self.max_scene_workspace_bytes) is not int or not 0 <= self.max_scene_workspace_bytes <= 2*1024**3:
            raise ValueError("Scene workspaces must remain within the local two-GiB allocation budget")
        # END allow complete observed robots within the existing 12-GiB GPU.
        for name in ("interpolation_dt", "maximum_trajectory_dt", "constraint_weight", "validation_joint_step"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not np.isfinite(self.search_seconds) or self.search_seconds < 0:
            raise ValueError("search_seconds must be finite and nonnegative")
        # BEGIN admit the exact grid before any native buffer allocation.
        # The selected cubic has degree+1 virtual intervals. Output sampling
        # must preserve its original physical clock, including nonzero dq0.
        grid = _fixed_spline_grid(intervals=self.knots + 4,
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
    """An exported cubic joint curve and its exact skill-progress mapping.

    All arrays are CPU NumPy arrays in named active-joint order.  The cubic
    reconstruction uses every native knot subdivision and its derivative;
    it therefore represents the exported cuRobo cubic, rather than an EEF
    interpolation or a new smoothing pass.
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
        barriers = [event for event in self.events if event["requires_evidence"] is not None]
        return deepcopy(min(barriers, key=lambda event: event["time"])) if barriers else None

    @property
    def executable_until(self):
        """Limit of this conditional plan before new actual feedback is needed."""
        barrier = self.next_verification
        return self.duration if barrier is None else float(barrier["time"])

    def sample(self, times):
        """Return q, dq and progress on the actual exported joint curve."""
        query = np.asarray(times, dtype=float)
        if not np.isfinite(query).all() or np.any(query < 0) or np.any(query > self.duration):
            raise ValueError("Trajectory queries must lie within its finite time interval")
        return (self._curve(query), self._curve(query, 1),
                np.interp(query / self.duration, self.progress_u, self.progress_s))


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
    # END failed results never carry an execution authorization.


# BEGIN size required evidence and native anchor buffers from the actual request.
def _anchor_capacity(progress_map, anchors, budget, refinement_parameters=()):
    """Keep every required clock anchor; bound only additional refinement.

    A stopped source vertex can have two clock parameters. Count those actual
    pairs before reserving optional samples, so no source point or pause is
    removed to fit the refinement allowance.
    """
    required = len(progress_map.anchor_parameters(anchors))
    if required > budget.max_validation_points:
        raise ValueError("Required source anchors exceed the finite validation-point budget")
    additional = np.unique(np.asarray(refinement_parameters, float))
    refinement_limit = min(budget.max_refinement_points, budget.max_validation_points-required)
    if (not np.isfinite(additional).all() or np.any(additional < 0) or np.any(additional > 1)
            or len(additional) > refinement_limit):
        raise ValueError("Refinement samples exceed the finite refinement-point budget")
    return required+refinement_limit


def _planning_workspace_bytes(scene, robot_metadata, budget, anchor_capacity):
    """Estimate the independent scene queries and all native anchor FK buffers.

    The main native rollout retains its original control horizon. Each custom
    manager evaluates the required anchors with a separate native FK model,
    then folds their penalties into that rollout's existing final cost column.
    """
    horizon = (budget.knots+4)*budget.interpolation_steps+1
    # Native LBFGS has two rollout instances and four line-search particles.
    # Include optimization/metrics, dense export and independent verification.
    # BEGIN account for body-point guidance only in its actual optimizer roles.
    # The dense/native metrics and final CPU-solid validation never construct
    # a coverage FK or its backward graph. Their other scene queries remain.
    allocation_shapes = (
        (4, budget.ik_seeds, 1, True), (8, budget.ik_seeds*4, 1, True),
        (4, budget.trajectory_seeds, horizon+anchor_capacity, True),
        (4, budget.trajectory_seeds*4, horizon+anchor_capacity, True),
        (2, budget.trajectory_seeds, budget.interpolation_buffer_size+anchor_capacity, False),
        (1, 1, budget.max_validation_points, False), (2, 1, anchor_capacity, False),
    )
    # Scene mode switches reuse FK but query exact, left and right material
    # permissions. Account for their bounded three-way query tensor here.
    scene_factor = 3 if scene.required_progress_anchors else 1
    scene_bytes = sum(copies*scene.estimated_workspace_bytes(batch, scene_factor*length,
                        robot_coverage_guidance=guidance)
                      for copies, batch, length, guidance in allocation_shapes)
    # END all live optimization scratch remains under the unchanged hard budget.
    if scene_bytes > budget.max_scene_workspace_bytes:
        raise ValueError("Independent scene query workspaces exceed the declared local allocation budget")
    dof, links, spheres = (len(robot_metadata["active_joint_names"]),
                           robot_metadata["links"], robot_metadata["spheres"])
    # Installed KinematicsFusedFunction.create_buffers uses float32 and this
    # planner requests exactly one tool frame. Per [batch, anchor] it allocates
    # pose/COM values and gradients (22 floats), sphere values/gradients (8*S),
    # a 6*D Jacobian and two D joint gradients, and 12*L cumulative transforms.
    # The same conservative rollout counts above give 22*K anchor batches.
    # Main FK, verification, model storage and other native allocations retain
    # the separate two-GiB allowance in the device-memory preflight below.
    bytes_per_state = 4*(22+8*spheres+8*dof+12*links)
    anchor_fk_bytes = 22*budget.trajectory_seeds*anchor_capacity*bytes_per_state
    return int(scene_bytes), int(anchor_fk_bytes)


def _require_planning_memory(torch, device, scene_bytes, anchor_fk_bytes):
    """Admit Torch buffers without promising cached memory to native allocators.

    CollisionBuffer and native FK buffers use torch.zeros. Unused capacity in
    this process's Torch allocator can therefore serve those tensors after a
    schedule replacement. Existing Mapper fields are already allocated. The
    separate native allowance must still fit driver-reported free memory.
    """
    free_bytes, _ = torch.cuda.mem_get_info(device)
    reusable_torch_bytes = max(0, torch.cuda.memory_reserved(device)-torch.cuda.memory_allocated(device))
    other_native_bytes = 2*1024**3
    if (free_bytes < other_native_bytes
            or free_bytes+reusable_torch_bytes < other_native_bytes+scene_bytes+anchor_fk_bytes):
        raise ValueError("Available GPU memory does not cover the bounded planner, anchor FK and scene workspaces")
# END size required evidence and native anchor buffers from the actual request.


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


# BEGIN source rhythm proposes the initial clock before any solver is created.
def _initial_timing(problem, budget):
    """Quantize a source-time proposal within this device's finite capacity.

    Source duration/rhythm are soft initialization information. The native
    interval cap bounds one request; an overlong demonstration may therefore
    propose a compressed clock. Hard event deadlines remain separate inputs.
    All later seed and refinement operations retain the returned clock.
    """
    source = problem.source_time_map
    seconds, progress = np.asarray(source["source_times"], float), np.asarray(source["progress"], float)
    if (seconds.ndim != 1 or seconds.shape != progress.shape or len(seconds) < 2
            or not np.isfinite(seconds).all() or np.any(np.diff(seconds) <= 0)):
        raise ValueError("Initial scheduling needs explicit increasing source seconds and their progress")
    source_duration = float(seconds[-1]-seconds[0])
    output_dt = float(np.float32(budget.interpolation_dt))
    maximum = _fixed_spline_grid(intervals=budget.knots+4,
        native_dt=budget.maximum_trajectory_dt, interpolation_steps=budget.interpolation_steps,
        interpolation_dt=budget.interpolation_dt)
    subdivisions = min(maximum["subdivisions"], max(1,
        int(np.ceil(source_duration/((budget.knots+4)*output_dt)))))
    # A power-of-two divisor preserves the existing binary32 grid contract;
    # enumerate exact realizable intervals when the subdivision count does not.
    selected = None
    for count in range(subdivisions, maximum["subdivisions"]+1):
        native_dt = float(np.float32(output_dt*count/budget.interpolation_steps))
        try:
            _fixed_spline_grid(intervals=budget.knots+4, native_dt=native_dt,
                interpolation_steps=budget.interpolation_steps, interpolation_dt=output_dt)
        except ValueError:
            continue
        selected = replace(budget, maximum_trajectory_dt=native_dt)
        break
    if selected is None:
        raise ValueError("Source-time proposal has no exact representable device clock within the budget")
    return selected, _ProgressMap((seconds-seconds[0])/source_duration, progress.copy())
# END no search path or candidate geometry can subsequently retime this binding.

# BEGIN quantize command-associated queries before freezing the binding.
def _bind_event_clock(problem, progress_map, duration, timestep=None):
    from .gripper_trajectory import _activation
    source = {}
    command_keys = []
    for event in problem.events:
        identity = (event.skill_id, event.event_id)
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
            command_keys.append((float(parameters[0]), applied/duration))
        source[identity] = {"intent_time_s": intent, "activation_time_s": applied if command else None,
                            "consumption_time_s": applied if needs_tick and not command else None}
        if command:
            source[identity].update(bound_activation_step=step, request_activation_step=step,
                request_activation_time_s=applied, request_intent_time_s=intent,
                request_time_origin_step=0, device_timestep_s=float(timestep))
    moves = dict([(0., 0.), (1., 1.)]+command_keys)
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
        record = source[(event.skill_id, event.event_id)]
        parameters = mapping.anchor_parameters([event.progress])[:, 0]
        record.update(time=(record["intent_time_s"] if record["activation_time_s"] is not None
                            else (record["consumption_time_s"] if record["consumption_time_s"] is not None
                                  else float(parameters[0]*duration))),
                      hold_until=float(parameters[-1]*duration))
    return _ProgressMap(mapping.u, mapping.s, source)
# END OMPL/refinement/export only read this record and never repeat quantization.



# BEGIN same-request continuation consumes facts and the original clock.
class ContinuationBindingError(ValueError):
    """Remaining work lacks facts, budget or a compatible original clock."""


def _prepare_continuation(problem, scene, budget, continuation):
    """Validate a fresh actual/scene binding without replaying consumed events.

    A continuation is the executor's existing result record. It is not a
    request to restore the old actual state or guess past task success. The
    original absolute progress, event identities and cumulative solve budget
    survive; angular history and past goal effects require supplied facts.
    """
    from .execution_binding import plain
    if not isinstance(continuation, dict) or not isinstance(continuation.get("original_binding"), dict):
        raise ContinuationBindingError("Continuation requires its original formal execution binding")
    original = continuation["original_binding"]
    if continuation.get("candidate_id") != original.get("candidate_id"):
        raise ContinuationBindingError("Continuation candidate identity differs from its original binding")
    if "planning_budget" not in original or "consumed_planning" not in original:
        raise ContinuationBindingError("Continuation lacks the original cumulative planning budget")
    request_budget = PlanningBudget(**original["planning_budget"])
    if budget is not None and asdict(budget) != asdict(request_budget):
        raise ContinuationBindingError("A continuation cannot reset or enlarge its original planning budget")
    consumed = deepcopy(continuation.get("consumed_planning", original["consumed_planning"]))
    for key in ("trajectory_solves", "search_seconds"):
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
    timestep = continuation.get("device_timestep_s")
    if (type(local_steps) is not int or local_steps < 0
            or not isinstance(timestep, (float, int)) or not np.isfinite(timestep) or timestep <= 0
            or physics_consumed != original.get("consumed_physics_steps", 0)+local_steps):
        raise ContinuationBindingError("Continuation requires cumulative steps and the unchanged native device timestep")
    remaining_solves = request_budget.maximum_trajectory_solves-consumed["trajectory_solves"]
    if remaining_solves <= 0:
        raise ContinuationBindingError("The current skill's cumulative trajectory-solve budget is exhausted")
    remaining = replace(request_budget, maximum_trajectory_solves=remaining_solves,
                        search_seconds=max(0., request_budget.search_seconds-consumed["search_seconds"]))
    feedback = continuation.get("actual_feedback", {})
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
    elapsed = local_steps*float(timestep)
    if (not isinstance(actual_elapsed, (int, float)) or not np.isfinite(actual_elapsed)
            or abs(actual_elapsed-elapsed) > 1e-8):
        raise ContinuationBindingError("Actual elapsed physics time differs from its native device step ledger")
    if (not isinstance(duration, (int, float)) or not isinstance(elapsed, (int, float))
            or not np.isfinite([duration, elapsed]).all() or not 0 <= elapsed < duration):
        raise ContinuationBindingError("The remaining original physical clock is exhausted or unavailable")
    old_u, old_s = np.asarray(curve.get("progress_u"), float), np.asarray(curve.get("progress_s"), float)
    try:
        original_map = _ProgressMap(old_u, old_s)
    except ValueError as error:
        raise ContinuationBindingError("Continuation has no complete frozen progress map") from error
    start_progress = float(np.interp(elapsed/duration, original_map.u, original_map.s))
    if start_progress >= problem.phase_count:
        raise ContinuationBindingError("The original skill progress is finished; unresolved effects need assessment or an explicit same-goal retry")
    request_events = deepcopy(original.get("request_events", original["problem"]["events"]))
    source_events = {(row["skill_id"], row["event_id"]): row for row in request_events}
    current_events = {(event.skill_id, event.event_id): event for event in problem.events}
    if set(current_events) != set(source_events):
        raise ContinuationBindingError("Fresh continuation must retain the original skill and event identities before filtering")
    for identity, event in current_events.items():
        if plain(event) != source_events[identity]:
            raise ContinuationBindingError("Continuation changed an original event/source/goal association")
    remaining_rows = continuation.get("remaining_events")
    if not isinstance(remaining_rows, (list, tuple)):
        raise ContinuationBindingError("Continuation lacks its consumed/unconsumed event ledger")
    pending_ids = {(row["skill_id"], row["event_id"]) for row in remaining_rows}
    old_rows = {(row["skill_id"], row["event_id"]): row for row in curve.get("events", ())}
    if not pending_ids <= set(old_rows):
        raise ContinuationBindingError("Remaining event ledger does not belong to the original candidate")
    applied = set(tuple(value) for value in original.get("consumed_events", ()))
    for row in (*continuation.get("applied_events", ()), *continuation.get("superseded_events", ())):
        applied.add((row["skill_id"], row["event_id"]))
    if applied & pending_ids:
        raise ContinuationBindingError("An applied command cannot be replayed as an unconsumed event")
    evaluated = {(row["skill_id"], row["event_id"]) for row in continuation.get("evaluated_events", ())}
    stopped = {(row["skill_id"], row["event_id"]) for row in continuation.get("stopped_events", ())}
    reached = {(row["skill_id"], row["event_id"]) for row in continuation.get("reached_events", ())}
    witnessed = applied | evaluated | stopped | reached | {
        (row["skill_id"], row["event_id"]) for row in continuation.get("verified_events", ())}
    # Ordinary effect evaluation consumes its observation node even when
    # the value is F/U. It never claims a successful effect or a held object.
    for identity in evaluated:
        event = current_events.get(identity)
        if (event is None or event.kind != "effect" or event.requires_evidence is not None
                or (event.stop and identity not in stopped)):
            raise ContinuationBindingError("An evaluation ledger entry is not an ordinary effect node")
        pending_ids.discard(identity)
    for identity in reached:
        event = current_events.get(identity)
        if event is None or event.kind != "function" or event.requires_evidence is not None:
            raise ContinuationBindingError("A reached-node ledger entry is not an independent functional event")
        pending_ids.discard(identity)
    consumed_events = set(source_events)-pending_ids
    if not consumed_events <= witnessed:
        raise ContinuationBindingError("Consumed events need actual application or verification ledger entries")
    for identity in pending_ids:
        row = old_rows[identity]
        due = row.get("activation_time_s") if row.get("activation_time_s") is not None else row["time"]
        if due < elapsed:
            raise ContinuationBindingError(f"Unconsumed event {identity} is overdue on the original hard clock")
    completed_goals = {tuple(value) for value in original.get("completed_goal_ids", ())}
    completed_goals.update(tuple(value) for value in continuation.get("completed_goal_ids", ()))
    for collection in ("pose_events", "coordinate_events", "supports"):
        for index, requirement in enumerate(getattr(problem, collection)):
            if requirement.progress < start_progress and (collection, index) not in completed_goals:
                raise ContinuationBindingError(f"Past goal {collection}:{index} needs actual completion evidence")
    for index, requirement in enumerate(problem.mechanisms):
        if (requirement.end >= start_progress and requirement.angular_column is not None
                and requirement.start != start_progress):
            raise ContinuationBindingError(f"mechanism:{index} needs current observed lifted coordinates and a fresh actual-start scope")
    # Goal ledger identities always use the original request's collection
    # indices. Filtering a remaining suffix must never renumber a proof.
    goal_identities = {collection: [index for index, item in enumerate(getattr(problem, collection))
                                  if item.progress >= start_progress]
                       for collection in ("pose_events", "coordinate_events", "supports")}
    rebound = deepcopy(problem)
    rebound.start_progress = start_progress
    rebound.events = tuple(event for event in rebound.events if (event.skill_id, event.event_id) in pending_ids)
    for collection in ("pose_events", "coordinate_events", "supports"):
        setattr(rebound, collection, tuple(item for item in getattr(rebound, collection) if item.progress >= start_progress))
    for collection in ("pose_relations", "mechanisms", "interiors"):
        setattr(rebound, collection, tuple(item for item in getattr(rebound, collection) if item.end >= start_progress))
    source_time = problem.progress_to_source_time(start_progress)
    selected = problem.source_progress > start_progress
    rebound.source_timestamps = np.r_[source_time, problem.source_timestamps[selected]]
    rebound.source_progress = np.r_[start_progress, problem.source_progress[selected]]
    rebound.__post_init__()
    # The existing hand factory closes over this fresh problem object. Update
    # its declared fields only after all validation succeeds so it sees the
    # same remaining events; no independent command list is manufactured.
    for item in fields(problem):
        if item.init:
            setattr(problem, item.name, getattr(rebound, item.name))
    return remaining, {"original_duration": float(duration), "elapsed": float(elapsed),
        "actual_elapsed": float(actual_elapsed), "elapsed_device_steps": local_steps,
        "device_timestep_s": float(timestep),
        "u": original_map.u.copy(), "s": original_map.s.copy(),
        "events": {identity: deepcopy(old_rows[identity]) for identity in pending_ids},
        "request_budget": asdict(request_budget), "consumed_planning": consumed,
        "physics_budget": physics_cap, "consumed_physics_steps": physics_consumed,
        "remaining_physics_steps": physics_remaining,
        "request_events": request_events, "consumed_events": sorted(consumed_events),
        "goal_identities": goal_identities,
        "completed_goal_ids": sorted(completed_goals), "effects": deepcopy(continuation.get("effects", {})),
        "previous_request_time_origin_s": float(original.get("request_time_origin_s", 0.)),
        "request_time_origin_s": float(original.get("request_time_origin_s", 0.))+float(elapsed),
        "request_time_origin_step": int(original.get("request_time_origin_step", 0))+local_steps}


def _continuation_timing(problem, budget, continuation):
    """Keep s(elapsed+t), extending only a final plateau to the native grid."""
    remaining = continuation["original_duration"]-continuation["elapsed"]
    proposal = SimpleNamespace(source_time_map={
        "source_times": np.array([0., remaining]),
        "progress": np.array([problem.start_progress, float(problem.phase_count)])})
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
    return selected, _ProgressMap(seconds/grid["duration"], progress, clock)
# END a new actual binding retains past results and original pending clocks.


# BEGIN native-cubic scalar clock with bounded, source-preserving linearization.
_PROGRESS_PARAMETER_ERROR = 1e-6
_MAX_PROGRESS_KEYS = 16384


def _progress_for_cubic_boundaries(progress_map, knots, start_velocity, *, stop_progress, anchors):
    """Return the saved progress map and its shared original-path parameter.

    Native cubic motion has fixed boundary derivatives. A linear reference
    clock demanding nonzero progress speed right up to a stopped boundary
    cannot generally fit that basis. Build a monotone scalar clock in the
    same uniform cubic basis instead: ``z(new_u)`` parameterizes the ORIGINAL
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
    moving_start = not np.all(np.asarray(start_velocity) == 0)
    # V2's actual cubic boundary coefficients for p=0, a=0 are
    # [-v,0,v,2v]. A positive scalar v starts progress immediately while the
    # actual robot still receives its own original measured joint velocity.
    initial_rate = boundaries[1]/(increments[0]+2) if moving_start else 0.
    prefix = np.array([-initial_rate, 0., initial_rate, 2*initial_rate])
    free = []
    for index, (left, right, steps) in enumerate(zip(boundaries[:-1], boundaries[1:], increments)):
        segment = (np.arange(3, steps+3, dtype=float)*initial_rate
                   if index == 0 and moving_start else np.linspace(left, right, steps+1)[1:])
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
    coordinates = parameters * (counts[:, None]-1)
    left = torch.minimum(coordinates.floor().long(), counts[:, None]-2).clamp_min(0)
    fraction = (coordinates-left).unsqueeze(-1)
    gather = left.unsqueeze(-1).expand(batch, -1, dof)
    q0, q1 = position.gather(1, gather), position.gather(1, gather+1)
    v0, v1 = velocity.gather(1, gather), velocity.gather(1, gather+1)
    delta = dt.reshape(batch, 1, 1)
    a = fraction
    q = ((2*a**3-3*a**2+1)*q0 + (a**3-2*a**2+a)*delta*v0
         + (-2*a**3+3*a**2)*q1 + (a**3-a**2)*delta*v1)
    dq = (((6*a**2-6*a)*q0 + (-6*a**2+6*a)*q1)/delta
          + (3*a**2-4*a+1)*v0 + (3*a**2-2*a)*v1)
    return q, dq


# BEGIN project more tightly than every existing mechanism acceptance band.
def _projection_tolerance(problem):
    """Set native projection accuracy without changing equalities or acceptance."""
    bounds = [problem.numerical.position_m, problem.numerical.rotation_rad, problem.numerical.joint]
    for relation in problem.mechanisms:
        model = relation.model
        bounds.extend((model.position_tolerance, model.rotation_tolerance))
    return min(bounds)*.1
# END project more tightly than every existing mechanism acceptance band.


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
def _verification_stop_preference(progress, joint_velocity, stop_progress):
    """Squared speed at declared stops, including every sample of a pause.

    A verification stop commands zero velocity. Its numerical acceptance
    band still belongs exclusively to PlanningProblem.residuals(). Keeping
    this optimizer objective active inside that band avoids terminating on
    its floating-point edge; ordinary motion receives no added objective.
    """
    import torch
    stopped = torch.zeros_like(progress, dtype=torch.bool)
    for value in np.unique(stop_progress):
        # Use exactly the same progress support as the hard stop residual.
        stopped = stopped | ((progress-value).abs() <= 1e-7)
    return torch.where(stopped, joint_velocity.square().sum(-1), torch.zeros_like(progress))
# END preserve a gradient toward the explicit verification-stop command.


# BEGIN one formal output clock and native named-body FK for timed hand geometry.
def _formal_native_clock(trajopt):
    """Bind hand, native rollout and dense export to one unchanged cubic clock."""
    # BEGIN use the same admitted grid as the native interpolation adapter.
    from curobo._src.types.control_space import ControlSpace
    config = trajopt.config
    if config.minimum_trajectory_dt != config.maximum_trajectory_dt:
        raise ValueError("A separately timed hand requires one fixed request-wide native dt")
    control = trajopt.transition_model.control_space
    if control != ControlSpace.BSPLINE_3:
        raise ValueError("The formal clock must use the selected native cubic spline representation")
    grid = _fixed_spline_grid(
        intervals=ControlSpace.spline_total_knots(control, trajopt.action_horizon),
        native_dt=config.maximum_trajectory_dt, interpolation_steps=trajopt.interpolation_steps,
        interpolation_dt=config.interpolation_dt)
    return {**grid, "source": "unchanged native cubic clock with exact float32 output grid"}
    # END dense sampling never invents a new physical duration.


class _RobotCoverageFK:
    """An independent same-model FK buffer for native solid body frames."""

    def __init__(self, problem, scene, robot_dict, device_cfg):
        self.problem, self.scene = problem, scene
        self.fk = None
        if scene.robot_coverage is None:
            return
        from curobo.kinematics import Kinematics, KinematicsCfg
        from curobo.types import JointState
        from curobo.content import get_assets_path
        from curobo._src.util.config_io import join_path
        from pathlib import Path
        self.JointState = JointState
        configuration = deepcopy(robot_dict)
        kinematics = configuration["robot_cfg"]["kinematics"]
        urdf = join_path(kinematics.get("external_asset_path") or get_assets_path(), kinematics["urdf_path"])
        if Path(urdf).resolve() != Path(scene._coverage_geometry["urdf_path"]).resolve():
            raise ValueError("Native coverage FK must use the exact independently checked URDF")
        locks = kinematics.get("lock_joints", {})
        if (scene.gripper is not None
                and scene.gripper.controller in {"timed_reference", "binary"}
                and scene.gripper.nominal_finger_trajectory is not None):
            for name in scene.gripper.joint_names:
                if name not in locks:
                    raise ValueError("Coverage may unlock only the explicitly independent finger joints")
                del locks[name]
        self.fk = Kinematics(KinematicsCfg.from_robot_yaml_file(
            configuration, tool_frames=list(scene.robot_coverage.required_link_names), device_cfg=device_cfg),
            compute_jacobian=True, compute_spheres=False, compute_com=False)
        if set(self.fk.joint_names) != set(scene.robot_coverage.baseline_joint_names):
            raise ValueError("Coverage FK and the native solid oracle must use exactly the same full joint set")
        self._quaternion = _orientation_vjp_type()
        self._layout = _native_gradient_layout_type()

    def poses(self, joint_position, joint_names, trajectory_time):
        if self.fk is None:
            return None
        import torch
        from scipy.spatial.transform import Rotation
        from .planning_problem import quaternion_multiply
        names, full = self.scene.full_robot_positions(joint_position, joint_names, trajectory_time)
        columns = [names.index(name) for name in self.fk.joint_names]
        q = full[..., columns].contiguous()
        state = self.fk.compute_kinematics(self.JointState.from_position(q, joint_names=self.fk.joint_names))
        p = self._layout.apply(state.tool_poses.position)
        r = self._quaternion.apply(q, state.tool_poses.quaternion.detach(), state.tool_jacobians[..., 3:, :].detach())
        base = torch.as_tensor(self.problem.base_pose, dtype=q.dtype, device=q.device)
        base_quaternion = torch.as_tensor(Rotation.from_matrix(self.problem.base_pose[:3, :3]).as_quat()[[3, 0, 1, 2]],
                                          dtype=q.dtype, device=q.device)
        p = p@base[:3, :3].T+base[:3, 3]
        r = quaternion_multiply(base_quaternion, r)
        return {name: (p[..., index, :], r[..., index, :]) for index, name in enumerate(self.fk.tool_frames)}


def _timed_native_state(state, problem, scene, joint_names, trajectory_time, *, sphere_padding=0.):
    """Replace only timed finger spheres before the unchanged native costs.

    cuRobo still owns the arm FK and all native collision pairs. A shallow
    state copy keeps its buffers immutable while the ordinary Torch graph
    carries finger geometry derivatives through the same actual TCP pose.
    """
    hand = scene.gripper
    if hand is None or hand.controller not in {"timed_reference", "binary"}:
        return state
    import torch
    q = state.joint_state.position
    tool = state.tool_poses
    index = tool.tool_frames.index(problem.tool_link)
    layout = _native_gradient_layout_type()
    r = _orientation_vjp_type().apply(q, tool.quaternion.detach(),
                                     state.cuda_robot_model_state.tool_jacobians[..., 3:, :].detach())
    p, r = problem.transform_tool_pose(layout.apply(tool.position)[:, :, index], r[:, :, index])
    spheres = problem.transform_robot_spheres(layout.apply(state.robot_spheres))
    progress = torch.zeros_like(trajectory_time)  # Timed state depends exclusively on elapsed time.
    fingers, half, _ = hand.state(progress, q, joint_names, trajectory_time=trajectory_time)
    spheres = hand.spheres(spheres, p, r, fingers, half)
    # BEGIN native self collision shares the scene's scoped uncertainty policy.
    # The backend explicitly ignores pairs containing a negative padded radius.
    # Only moving finger proxies after t=0 lack a physical model; fixed palm,
    # arm and every measured entrance sphere retain all native collision pairs.
    if hand.nominal_finger_trajectory is None:
        moving = np.zeros(spheres.shape[-2], dtype=bool)
        for sphere_index, translation in zip(hand.sphere_indices, hand.sphere_joint_translation):
            moving[sphere_index] = bool(np.any(translation != 0))
        uncertain = ((trajectory_time > 0)[..., None]
                     & torch.as_tensor(moving, device=spheres.device)[None, None])
        padding = torch.as_tensor(sphere_padding, dtype=spheres.dtype, device=spheres.device)
        radius = torch.where(uncertain, -1.-padding.abs(), spheres[..., 3])
        spheres = torch.cat((spheres[..., :3], radius[..., None]), -1)
    # END command completion is never treated as actual finger penetration.
    base = torch.as_tensor(problem.base_pose, dtype=q.dtype, device=q.device)
    spheres = torch.cat(((spheres[..., :3]-base[:3, 3])@base[:3, :3], spheres[..., 3:4]), -1)
    result = copy(state)
    result.cuda_robot_model_state = copy(state.cuda_robot_model_state)
    result.cuda_robot_model_state.robot_spheres = spheres.contiguous()
    return result
# END extra FK changes neither the optimizer variables nor its single IK target.


# BEGIN use each native manager's own effective sphere padding.
def _native_collision_padding(manager):
    cost = manager.get_cost("self_collision")
    return 0. if cost is None or not cost.enabled else cost.config.self_collision_kin_config.sphere_padding
# END a disabled moving proxy remains negative after native padding is added.


# BEGIN scene switches share q/t while querying distinct adjacent modes.
def _scene_boundary_evaluate(scene, problem, progress_map, spheres, position, quaternion,
                             progress, joint_position, joint_names, *, trajectory_time,
                             normalized_time):
    """Check material/attachment mode limits without changing physical time.

    Float64 neighbors vanish when native inputs are float32. Construct the
    side in the actual progress dtype, retaining original samples first so
    release FK keeps its original exact witness. On a progress plateau only
    its entrance consumes the old side and only its exit consumes the new
    side; an action during that plateau never regains an expired old mode.
    """
    import torch
    anchors = [float(value) for value in scene.required_progress_anchors
               if problem.start_progress <= value <= problem.phase_count]
    if not anchors:
        return scene.evaluate(spheres, position, quaternion, progress, joint_position,
                              joint_names, trajectory_time=trajectory_time)
    left = torch.zeros_like(progress, dtype=torch.bool)
    right = torch.zeros_like(progress, dtype=torch.bool)
    for anchor in anchors:
        parameters = progress_map.anchor_parameters([anchor])[:, 0]
        matching = progress == progress.new_tensor(anchor)
        if anchor > problem.start_progress:
            left |= matching & (normalized_time <= normalized_time.new_tensor(parameters[0]))
        if anchor < problem.phase_count:
            right |= matching & (normalized_time >= normalized_time.new_tensor(parameters[-1]))
    lower = torch.where(left, torch.nextafter(progress, torch.full_like(progress, -float("inf"))), progress)
    upper = torch.where(right, torch.nextafter(progress, torch.full_like(progress, float("inf"))), progress)
    repeated = lambda value: torch.cat((value, value, value), dim=1)
    terms = scene.evaluate(repeated(spheres), repeated(position), repeated(quaternion),
        torch.cat((progress, lower, upper), dim=1), repeated(joint_position), joint_names,
        trajectory_time=repeated(trajectory_time))
    return torch.stack(terms.split(progress.shape[1], dim=1)).amax(0)
# END query-side selection never changes the command or nominal hand clock.


def _process_manager_type(problem, scene, progress_map, anchors, budget, seed_progress,
                          refinement_parameters=(), *, clock=None, seed_parameter=None, robot_dict=None):
    """Build V2's documented manager extension after GPU use is requested.

    Configuration construction must return the derived type and class_type;
    registering a custom name alone is not dispatched by V2's base manager.
    Only constraint_cfg enables this extension.  Built-in joint and self-
    collision terms remain intact in every manager and every evaluation.
    """
    import torch
    from curobo.kinematics import Kinematics
    from curobo.types import JointState
    from curobo._src.rollout.cost_manager.cost_manager_robot import RobotCostManager
    from curobo._src.rollout.cost_manager.cost_manager_robot_cfg import RobotCostManagerCfg
    orientation_vjp = _orientation_vjp_type()
    native_layout = _native_gradient_layout_type()

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
            self._dense_counts = None
            if not self._process_mode:
                return
            transition_model.robot_model.compute_jacobian = True
            self._scene = scene.clone_for_rollout()
            self._coverage_fk = (_RobotCoverageFK(problem, self._scene, robot_dict, self.device_cfg)
                                 if scene.robot_coverage is not None
                                 and (scene.gripper is None or scene.gripper.controller == "position_tracking"
                                      or scene.gripper.nominal_finger_trajectory is not None)
                                 and self._process_mode in {"seed_optimizer", "trajectory_optimizer"} else None)
            self._joint_names = tuple(transition_model.joint_names)
            self._tool_index = transition_model.robot_model.tool_frames.index(problem.tool_link)
            self._map_u = self.device_cfg.to_device(progress_map.u)
            self._map_s = self.device_cfg.to_device(progress_map.s)
            anchor_pairs = progress_map.anchor_parameters(anchors)
            if len(refinement_parameters):
                additional = np.asarray(refinement_parameters, float)
                anchor_pairs = np.unique(np.concatenate((anchor_pairs, np.column_stack(
                    (additional, np.interp(additional, progress_map.u, progress_map.s))))), axis=0)
            self._anchor_u = self.device_cfg.to_device(anchor_pairs[:, 0]).unsqueeze(0)
            self._anchor_s = self.device_cfg.to_device(anchor_pairs[:, 1]).unsqueeze(0)
            # A separate native FK instance owns the anchor buffers.  Reusing
            # the main rollout FK would overwrite values needed by backward.
            self._anchor_fk = Kinematics(transition_model.robot_model.config, compute_jacobian=True)

        def set_dense_counts(self, counts):
            self._dense_counts = counts.reshape(-1).long()

        def compute_costs(self, state, cost_collection=None, goal=None, **kwargs):
            if self._process_mode and self._scene.gripper is not None and self._scene.gripper.controller in {"timed_reference", "binary"}:
                batch, horizon = state.joint_state.position.shape[:2]
                if self._process_mode.startswith("seed_"):
                    query_time = (seed_parameter*clock["duration"]).expand(batch, horizon)
                else:
                    counts = (torch.full((batch,), horizon, dtype=torch.long, device=state.joint_state.position.device)
                              if self._dense_counts is None else self._dense_counts)
                    query_time = (torch.arange(horizon, device=counts.device, dtype=state.joint_state.position.dtype)[None]
                                  /(counts[:, None]-1)).clamp(0, 1)*clock["duration"]
                state = _timed_native_state(state, problem, self._scene, self._joint_names, query_time,
                                            sphere_padding=_native_collision_padding(self))
            collection = super().compute_costs(state, cost_collection, goal, **kwargs)
            if not self._process_mode:
                return collection
            q, dq = state.joint_state.position, state.joint_state.velocity
            batch, horizon, _ = q.shape
            tool = state.tool_poses
            quaternion = orientation_vjp.apply(q, tool.quaternion.detach(),
                                               state.cuda_robot_model_state.tool_jacobians[..., 3:, :].detach())
            position = native_layout.apply(tool.position)[:, :, self._tool_index]
            quaternion = quaternion[:, :, self._tool_index]
            spheres = native_layout.apply(state.robot_spheres)
            if self._process_mode.startswith("seed_"):
                # IK is only an initializer: use its target phase for collision
                # permissions, while the complete process belongs to TrajOpt.
                progress = seed_progress.expand(batch, horizon)
                position, quaternion = problem.transform_tool_pose(position, quaternion)
                scene_value = _scene_boundary_evaluate(self._scene, problem, progress_map,
                    problem.transform_robot_spheres(spheres), position, quaternion, progress,
                    q, self._joint_names,
                    trajectory_time=(seed_parameter*clock["duration"]).expand(batch, horizon),
                    normalized_time=seed_parameter.expand(batch, horizon))
                if self._process_mode == "seed_optimizer":
                    scene_value = scene_value.square()
                    if self._coverage_fk is not None:
                        query_time = (seed_parameter*clock["duration"]).expand(batch, horizon)
                        scene_value = scene_value+self._scene.robot_coverage_guidance(
                            self._coverage_fk.poses(q, self._joint_names, query_time), progress).square()
                collection.add(scene_value.unsqueeze(-1) * budget.constraint_weight, "process_scene")
                return collection

            counts = (torch.full((batch,), horizon, dtype=torch.long, device=q.device)
                      if self._dense_counts is None else self._dense_counts)
            u = (torch.arange(horizon, device=q.device, dtype=q.dtype)[None, :]
                 / (counts[:, None]-1)).clamp(0, 1)
            progress = _torch_interpolate(self._map_u, self._map_s, u)
            anchor_u = self._anchor_u.expand(batch, -1)
            anchor_s = self._anchor_s.expand(batch, -1)
            anchor_q, anchor_dq = _hermite_at_parameters(q, dq, state.joint_state.dt, counts, anchor_u)
            anchor_state = self._anchor_fk.compute_kinematics(
                JointState.from_position(anchor_q.contiguous(), joint_names=list(self._joint_names)))
            anchor_quaternion = orientation_vjp.apply(anchor_q, anchor_state.tool_poses.quaternion.detach(),
                                                      anchor_state.tool_jacobians[..., 3:, :].detach())
            anchor_position = native_layout.apply(anchor_state.tool_poses.position)[:, :, self._tool_index]
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

            position, quaternion = problem.transform_tool_pose(ordered(position), ordered(quaternion))
            spheres = problem.transform_robot_spheres(ordered(spheres))
            q, dq, progress = ordered(q), ordered(dq), ordered(progress)
            trajectory_time = ordered(all_u)*clock["duration"]
            process_value = problem.evaluate(position, quaternion, progress, q, self._joint_names,
                                             joint_velocity=dq)
            scene_value = _scene_boundary_evaluate(self._scene, problem, progress_map,
                spheres, position, quaternion, progress, q, self._joint_names,
                trajectory_time=trajectory_time, normalized_time=ordered(all_u))
            valid = torch.arange(horizon, device=q.device)[None, :] < counts[:, None]
            values = [("skill_process", process_value), ("process_scene", scene_value)]
            # A squared optimizer penalty has a gradient proportional to
            # the remaining violation. Metrics retain the original raw
            # nonnegative values and therefore their original zero sets.
            if self._process_mode == "trajectory_optimizer":
                values = [(name, value.square()) for name, value in values]
                if self._coverage_fk is not None:
                    values.append(("robot_new_volume_guidance", self._scene.robot_coverage_guidance(
                        self._coverage_fk.poses(q, self._joint_names, trajectory_time), progress).square()))
                # BEGIN retain a gradient toward source-declared pose equalities.
                # The problem owns equality-axis selection and its complete
                # endpoint-family reference; free pose axes gain no target.
                values.append(("skill_pose_equality_preference", problem.optimizer_preferences(
                    position, quaternion, progress)))
                # END retain a gradient toward source-declared pose equalities.
                # A position-tracking gripper has an exact physical command,
                # even inside its numerical acceptance band. Preserve a
                # gradient toward that command to avoid stopping on the
                # band's floating-point edge. Allowed pose/contact freedoms
                # receive no invented centering preference.
                values.append(("gripper_command_preference", self._scene.optimizer_preferences(
                    progress, q, self._joint_names)))
                # BEGIN optimize the actual zero-speed verification command.
                values.append(("verification_stop_preference", _verification_stop_preference(
                    progress, dq, problem.stop_progress)))
                # END optimize the actual zero-speed verification command.
            for name, value in values:
                value = value.gather(1, inverse)
                regular = value[:, :horizon] * valid
                # Keep CostCollection's native [B,H,*] shape.  Summing exact
                # anchor penalties into its final column retains their full
                # gradient and makes any anchor violation fail feasibility.
                folded = torch.cat((torch.zeros_like(regular[:, :-1]), value[:, horizon:].sum(1, keepdim=True)), 1)
                collection.add((regular+folded).unsqueeze(-1) * budget.constraint_weight, name)
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
    if result.num_seeds == 1:
        return result
    ordered = copy(result)
    ordered.seed_rank = result.seed_rank.new_tensor(
        [[index, *(other for other in range(result.num_seeds) if other != index)]])
    # Native postprocessing augments state positions with locked joints but
    # intentionally leaves their action knots in active-joint coordinates.
    # Its result gather assumes both have position's DOF. The JointState
    # indexer instead slices each tensor on the seed axis without that false
    # reshape, preserving both coordinate layouts and every timing field.
    ordered.js_solution = ordered.interpolated_trajectory = None
    candidate = ordered.get_topk_seeds(1)
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
# END exact native candidate extraction before independent verification.


# BEGIN bounded, numerically distinct native IK continuation selection.
def _select_ik_candidates(candidates, success, previous, count, joint_tolerance):
    """Choose up to two continuations from one existing native IK result.

    Feasible candidates always take precedence over finite failed IK seeds.
    Distinct result indices need not represent distinct configurations, so
    merge only candidates inside the already declared joint numerical band.
    A single available posture may join two chains temporarily; when they
    separate again, retain the nearest continuation and the most separated
    available alternative. Otherwise Hungarian matching preserves continuity.
    This selects initializers, never certifies a complete robot trajectory.
    """
    # BEGIN isolate malformed numerical seeds before candidate matching.
    finite = np.isfinite(candidates).all(axis=-1)
    feasible = np.flatnonzero(np.asarray(success, bool) & finite)
    available = feasible if len(feasible) else np.flatnonzero(finite)
    if not len(available):
        return np.empty(0, dtype=int), 0
    # END finite failed IK remains an initializer, never a feasibility claim.
    # A duplicate group's representative retains the nearest actual primary
    # continuation, rather than whichever native result happened to come first.
    available = available[np.argsort(np.linalg.norm(candidates[available]-previous[0], axis=-1), kind="stable")]
    distinct = []
    for index in available:
        if (not distinct or np.all(np.max(np.abs(candidates[distinct]-candidates[index]), axis=-1)
                                   > joint_tolerance)):
            distinct.append(int(index))
    distinct = np.asarray(distinct, dtype=int)
    distinct_success_count = len(distinct) if len(feasible) else 0
    distances = np.linalg.norm(previous[:, None]-candidates[distinct][None], axis=-1)
    nearest = int(np.argmin(distances[0]))
    if count == 1:
        # One branch keeps the nearest actual candidate, even if another
        # representative lies in the same numerical deduplication band.
        selected = available[[int(np.argmin(np.linalg.norm(previous[0]-candidates[available], axis=-1)))]]
    elif len(distinct) == 1:
        selected = np.repeat(distinct, 2)
    elif len(previous) == 1 or np.max(np.abs(previous[0]-previous[1])) <= joint_tolerance:
        separation = np.linalg.norm(candidates[distinct]-candidates[distinct[nearest]], axis=-1)
        selected = distinct[[nearest, int(np.argmax(separation))]]
    else:
        _, columns = linear_sum_assignment(distances)
        selected = distinct[columns]
    return selected, distinct_success_count
# END bounded, numerically distinct native IK continuation selection.


class _CuroboPlanner:
    """One request's native solvers and independent verification workspaces."""

    def __init__(self, problem, scene, robot, budget, device, *, continuation=None):
        self.ik = self.trajopt = None
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
            self._initialize(problem, scene, robot, budget, device, continuation=continuation)
        except BaseException:
            # A later native configuration failure must release an IK solver
            # already created in this process before propagating the error.
            self.close()
            raise

    def _initialize(self, problem, scene, robot, budget, device, *, continuation=None):
        # Validate grounded evidence and bound independent scene workspaces
        # before importing native GPU solvers or creating any of their buffers.
        scene.validate_problem(problem)
        budget, initial_progress = (_initial_timing(problem, budget) if continuation is None
                                    else _continuation_timing(problem, budget, continuation))
        anchors = np.unique(np.concatenate((problem.anchors, np.asarray(scene.required_progress_anchors, float))))
        # BEGIN preserve all required source points within the total query budget.
        if (len(anchors) > budget.max_validation_points or not np.isfinite(anchors).all()
                or np.any(anchors < 0) or np.any(anchors > problem.phase_count)):
            raise ValueError("All required source knots and scene anchors must fit the finite planning horizon and budget")
        if problem.reference_frame != scene.reference_frame:
            raise ValueError("Problem and planning scene must use the same explicitly calibrated reference frame")
        self.progress_map = initial_progress
        if continuation is None:
            self.progress_map, _ = _progress_for_cubic_boundaries(initial_progress,
                budget.knots, problem.start_dq, stop_progress=problem.stop_progress, anchors=anchors)
        initial_clock = _fixed_spline_grid(intervals=budget.knots+4,
            native_dt=budget.maximum_trajectory_dt, interpolation_steps=budget.interpolation_steps,
            interpolation_dt=budget.interpolation_dt)
        if continuation is None:
            self.progress_map = _bind_event_clock(problem, self.progress_map, initial_clock["duration"],
                None if scene.gripper is None else scene.gripper.control_timestep_s)
        self.anchor_capacity = _anchor_capacity(self.progress_map, anchors, budget)
        robot_dict, self.robot_metadata = _bounded_robot_configuration(robot, problem, scene, budget)
        self.scene_workspace_bytes, self.anchor_fk_workspace_bytes = _planning_workspace_bytes(
            scene, self.robot_metadata, budget, self.anchor_capacity)
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
        _require_planning_memory(torch, self.device_cfg.device,
                                 self.scene_workspace_bytes, self.anchor_fk_workspace_bytes)
        # END include request-sized anchor FK in native allocation admission.
        self.ik = self.trajopt = None
        self.anchors = anchors
        self.robot_dict = robot_dict
        self.seed_progress = self.device_cfg.to_device([[problem.start_progress]])
        self.seed_parameter = self.device_cfg.to_device([[0.]])
        self.formal_clock = {}
        self._manager_type = _process_manager_type(
            problem, scene, self.progress_map, self.anchors, budget, self.seed_progress,
            clock=self.formal_clock, seed_parameter=self.seed_parameter, robot_dict=self.robot_dict)
        self.refinement_parameters = np.empty(0)

        def configuration(name, mode):
            data = deepcopy(load_yaml(join_path(get_task_configs_path(), name)))
            constraint = data["rollout"].setdefault("constraint_cfg", {})
            # All environment geometry is owned by PlanningScene in reference
            # coordinates.  Native robot self-collision remains enabled.
            constraint.pop("scene_collision_cfg", None)
            constraint["process_mode"] = mode
            return data

        ik_optimizer = configuration("ik/lbfgs_ik.yml", "seed_optimizer")
        # BEGIN retain native trajectory smoothness without changing acceptance.
        # Native velocity/acceleration/jerk preferences choose among curves;
        # independent hard constraints still decide whether a curve is usable.
        trajectory_optimizer = configuration("trajopt/lbfgs_bspline_trajopt.yml", "trajectory_optimizer")
        # END retain native trajectory smoothness without changing acceptance.
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
            # BEGIN every optimized IK action uses the same full physical domain.
            # The optional LM pre-solver has its own narrowed/clamped domain.
            # Existing Halton initialization and native L-BFGS already supply
            # all budgeted branches through the domain-restricted core.
            use_lm_seed=False,
            # END no independent optimizer may precede the checked native core.
            position_tolerance=problem.numerical.position_m,
            orientation_tolerance=problem.numerical.rotation_rad,
            cost_manager_config_instance_type=self._manager_type,
            override_optimizer_num_iters={"lbfgs": budget.ik_iterations}, random_seed=budget.random_seed)
        # BEGIN evaluate the declared alternative IK branches before returning.
        # Native early exit only requires one successful seed per batch. Two
        # requested trajectory branches must let all budgeted IK seeds run
        # through their configured optimizer; a one-branch request is unchanged.
        if budget.trajectory_seeds > 1:
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
        self._verification_fk = Kinematics(self.ik.kinematics.config, compute_jacobian=True)
        self._validate_occupied_robot_geometry()
        self._orientation_vjp = _orientation_vjp_type()
        self._native_layout = _native_gradient_layout_type()
        self._verification_scene = scene.clone_for_rollout()
        transition = deepcopy(load_yaml(join_path(get_task_configs_path(), "trajopt/transition_bspline_trajopt.yml")))
        transition["transition_model_cfg"].update(n_knots=budget.knots, interpolation_steps=budget.interpolation_steps)
        terminal_position_tolerance, terminal_rotation_tolerance = self._terminal_tolerances()
        self._trajectory_configuration = {
            "robot": deepcopy(robot_dict),
            "optimizer_configs": [trajectory_optimizer],
            "metrics_rollout": configuration("metrics_base.yml", "trajectory_metrics"),
            "transition_model": transition, "device_cfg": self.device_cfg,
            "num_seeds": budget.trajectory_seeds, "max_batch_size": 1, "max_goalset": 1,
            "use_cuda_graph": False, "random_seed": budget.random_seed,
            # BEGIN native terminal acceptance uses the declared goal accuracy.
            "position_tolerance": terminal_position_tolerance,
            "orientation_tolerance": terminal_rotation_tolerance,
            # END a default native threshold cannot shrink an allowed target.
            "cost_manager_config_instance_type": self._manager_type,
            "override_optimizer_num_iters": {"lbfgs": budget.trajectory_iterations},
            # Native retiming runs even with zero finetuning attempts. A
            # different dt changes derivative constraints and, for nonzero
            # measured dq0, the cubic's fixed boundary controls. Keep one
            # request-wide physical clock through optimization and metrics.
            "minimum_trajectory_dt": budget.maximum_trajectory_dt,
            "maximum_trajectory_dt": budget.maximum_trajectory_dt,
            "interpolation_dt": budget.interpolation_dt,
            "interpolation_buffer_size": budget.interpolation_buffer_size,
        }

        class ProcessTrajectoryOptimizer(TrajectoryOptimizer):
            def get_interpolated_trajectory(self, js_optimized):
                # BEGIN preserve raw controls and boundary derivatives on one clock.
                from curobo._src.curobolib.cuda_ops.trajectory import get_bspline_interpolation
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
                state = get_bspline_interpolation(js_optimized, self._interpolated_traj_buffer,
                    self._interpolation_dt_buffer, current_state=goal.current_js,
                    goal_state=goal.seed_goal_js, start_idx=goal.idxs_current_js,
                    goal_idx=goal.idxs_seed_goal_js,
                    use_implicit_goal_state=goal.seed_enable_implicit_goal_js,
                    interpolated_horizon=counts, bspline_degree=3)
                rollout = self.additional_metrics_rollouts["interpolated_rollout"]
                # Both managers retain valid-prefix masking before the native
                # caller computes dense metrics and its original success flag.
                rollout.constraint_manager.set_dense_counts(counts)
                rollout.metrics_constraint_manager.set_dense_counts(counts)
                return state, counts, False
                # END no native resize, retiming, interpolation switch or metric bypass.

        self._optimizer_class = ProcessTrajectoryOptimizer
        self._optimizer_config_class = TrajectoryOptimizerCfg
        self.trajopt = ProcessTrajectoryOptimizer(TrajectoryOptimizerCfg.create(**deepcopy(self._trajectory_configuration)))
        self._bind_formal_clock()
        self._initialize_verification_constraints()
        self._bind_physical_domains()

    # BEGIN bind the independently controlled fingers to native formal output.
    def _bind_formal_clock(self):
        self.formal_clock.clear()
        self.formal_clock.update(_formal_native_clock(self.trajopt))
        hand = self.scene.gripper
        if hand is not None:
            hand.bind_clock(self.formal_clock["duration"], self.progress_map)
        self.finger_trajectory = None if hand is None else hand.finger_trajectory

    def _seed_u(self, progress):
        """Choose the earliest plateau point for initializer-only state queries."""
        values, indices = np.unique(self.progress_map.s, return_index=True)
        return float(np.interp(progress, values, self.progress_map.u[indices]))
    # END formal export always evaluates its actual time, not this seed inverse.

    # BEGIN bind occupancy evidence to the actual bounded current robot model.
    def _validate_occupied_robot_geometry(self):
        """Verify q0 spheres and requested link poses before trajectory setup.

        Collision spheres come from the actual planner's FK configuration.
        Additional pose frames use the same already checked robot dictionary
        and cannot add links, joints, meshes or optimizer tool targets. CPU
        float64 frame transforms match the occupancy adapter's evidence path.
        """
        from scipy.spatial.transform import Rotation
        from curobo.kinematics import Kinematics, KinematicsCfg

        required = self.scene.required_robot_links
        with self.torch.no_grad():
            state = self._verification_fk.compute_kinematics(self.JointState.from_position(
                self.start.position[:, None].contiguous(), joint_names=list(self.joint_names)))
            spheres = state.robot_spheres[0, 0].detach().cpu().numpy().astype(float)
            poses = {}
            if required:
                # The preflight already proved every requested link belongs
                # to this selected tree. Only this single-state FK exposes
                # more frames; IK retains its one-tool seed allocation.
                pose_fk = Kinematics(KinematicsCfg.from_robot_yaml_file(
                    deepcopy(self.robot_dict), tool_frames=list(required), device_cfg=self.device_cfg),
                    compute_spheres=False, compute_com=False)
                if (set(pose_fk.joint_names) != set(self.joint_names)
                        or not set(pose_fk.config.kinematics_config.link_name_to_idx_map)
                        <= set(self.robot_metadata["selected_link_names"])):
                    raise ValueError("Occupied-link FK changed the bounded planning robot tree")
                columns = [self.joint_names.index(name) for name in pose_fk.joint_names]
                posed = pose_fk.compute_kinematics(self.JointState.from_position(
                    self.start.position[:, None, columns].contiguous(), joint_names=pose_fk.joint_names))
                positions = posed.tool_poses.position[0, 0].detach().cpu().numpy().astype(float)
                quaternions = posed.tool_poses.quaternion[0, 0].detach().cpu().numpy().astype(float)
                for name in required:
                    index = pose_fk.tool_frames.index(name)
                    pose = np.eye(4)
                    pose[:3, 3] = positions[index]
                    pose[:3, :3] = Rotation.from_quat(quaternions[index, [1, 2, 3, 0]]).as_matrix()
                    poses[name] = self.problem.base_pose@pose
        spheres[:, :3] = spheres[:, :3]@self.problem.base_pose[:3, :3].T+self.problem.base_pose[:3, 3]
        self.scene.validate_robot_geometry(spheres, poses)
    # END bind occupancy evidence to the actual bounded current robot model.

    def _initialize_verification_constraints(self):
        from curobo._src.rollout.cost_manager.cost_manager_robot import RobotCostManager
        from curobo._src.state.state_robot import RobotState
        self._RobotState = RobotState
        # The base manager alone supplies the same native self-collision and
        # joint/derivative limits at extra control-curve samples.  Its buffers
        # are independent of the optimization rollouts.
        self._verification_native = RobotCostManager(self.device_cfg)
        self._verification_native.initialize_from_config(
            deepcopy(self.trajopt.config.metrics_rollout_config.constraint_cfg),
            self.trajopt.transition_model)

    # BEGIN constrain the actual cubic throughout native initialization/updates.
    def _bind_physical_domains(self):
        limits = self._verification_native.get_cost("cspace").config.joint_limits
        columns = [limits.joint_names.index(name) for name in self.joint_names]
        bounds = limits.position[:, columns]
        self.position_domain = JointPositionDomain(bounds[0], bounds[1])
        self.physical_domain = CubicJointDomain(bounds[0], bounds[1],
            self.start.position, self.start.velocity, self.formal_clock["knot_dt"],
            self.trajopt.action_horizon)
        for solver, domain in ((self.ik, self.position_domain), (self.trajopt, self.physical_domain)):
            for optimizer in solver.core.optimizers:
                if not hasattr(optimizer._core, "set_action_domain"):
                    raise RuntimeError("The active cuRobo source lacks the physical action-domain hook; "
                                       "run with the reviewed local curobo source on PYTHONPATH")
                optimizer._core.set_action_domain(domain)
    # END limits are the model's full physical q interval, never a seed box.

    # BEGIN reject an immutable infeasible entrance before running native IK.
    def initial_state_diagnostics(self):
        """Check only necessary conditions known at the measured entrance.

        The complete native STATE cost requires acceleration, jerk and torque
        inputs. Those measurements are absent, so inspect its effective q/dq
        limits directly and call its sphere-only self-collision term alone.
        Scene terms retain their normal conservative meaning, including
        unknown space and a binary gripper's reachable envelope; a positive
        term need not mean that the actual robot is physically in contact.
        """
        torch = self.torch
        cspace = self._verification_native.get_cost("cspace")
        if cspace is None or not cspace.enabled:
            raise RuntimeError("Entrance checks require the enabled native cspace constraints")
        limits = cspace.config.joint_limits
        if (len(limits.joint_names) != len(self.joint_names)
                or set(limits.joint_names) != set(self.joint_names)):
            raise RuntimeError("Native entrance limits must match the active feedback joints")
        columns = [limits.joint_names.index(name) for name in self.joint_names]
        q, dq = self.start.position[:, None], self.start.velocity[:, None]
        joint_bounds = {}
        with torch.no_grad():
            components = {}
            for name, values in (("position", q), ("velocity", dq)):
                bounds = getattr(limits, name)[:, columns]
                violations = torch.relu(bounds[0]-values) + torch.relu(values-bounds[1])
                components[f"native:{name}"] = violations
                joint_bounds[name] = {
                    "measured": values[0, 0].detach().cpu().tolist(),
                    "lower": bounds[0].detach().cpu().tolist(),
                    "upper": bounds[1].detach().cpu().tolist(),
                    "violation": violations[0, 0].detach().cpu().tolist(),
                }
            # Construct only the measured fields. Native from_position/from_numpy
            # helpers otherwise synthesize zero acceleration and jerk tensors.
            measured = self.JointState(position=q.contiguous(), velocity=dq.contiguous(),
                                       joint_names=list(self.joint_names))
            state = self._verification_fk.compute_kinematics(measured)
            native_self = self._verification_native.get_cost("self_collision")
            self_collision_checked = native_self is not None and native_self.enabled
            if self_collision_checked:
                # Calling this component directly also owns its one-state
                # workspace; the full manager normally performs that setup.
                native_self.setup_batch_tensors(*q.shape[:2])
                actual_state = _timed_native_state(self._RobotState(joint_state=measured, cuda_robot_model_state=state),
                    self.problem, self._verification_scene, self.joint_names, q.new_zeros(q.shape[:2]),
                    sphere_padding=_native_collision_padding(self._verification_native))
                components["native:self_collision"] = native_self.forward(actual_state.cuda_robot_model_state.robot_spheres)
            tool = state.tool_poses
            tool_index = tool.tool_frames.index(self.problem.tool_link)
            position, quaternion = self.problem.transform_tool_pose(
                tool.position[:, :, tool_index], tool.quaternion[:, :, tool_index])
            progress = q.new_full(q.shape[:2], self.problem.start_progress)
            unresolved_scene = {}
            for name, value in self._verification_scene.components(
                    self.problem.transform_robot_spheres(state.robot_spheres),
                    position, quaternion, progress, q, self.joint_names,
                    trajectory_time=torch.zeros_like(progress)).items():
                if name.startswith("unknown/") and not name.startswith("unknown/release_anchor/"):
                    unresolved_scene[name] = float(value.max().item())
                else:
                    components[f"scene:{name}"] = value
            if self._verification_scene.required_progress_anchors:
                components["scene:mode_boundary"] = _scene_boundary_evaluate(
                    self._verification_scene, self.problem, self.progress_map,
                    self.problem.transform_robot_spheres(state.robot_spheres), position,
                    quaternion, progress, q, self.joint_names,
                    trajectory_time=torch.zeros_like(progress), normalized_time=torch.zeros_like(progress))
            residuals = {name: float(value.max().item()) for name, value in components.items()}
            coverage = self._verification_scene.check_robot_coverage(
                q, self.joint_names, torch.zeros_like(progress), entrance=True)
            if coverage:
                residuals["scene:unknown/new_robot_volume"] = float(any(not item["attempt_allowed"] for item in coverage))
        failed = [name for name, value in residuals.items() if not np.isfinite(value) or value > 0]
        # BEGIN preserve nonfinite failures in strict JSON diagnostics.
        # JSON null plus the explicit component list retains the rejection
        # without emitting NaN/Infinity or losing the report during saving.
        nonfinite = [name for name, value in residuals.items() if not np.isfinite(value)]
        for name in nonfinite:
            residuals[name] = None
        # END preserve nonfinite failures in strict JSON diagnostics.
        return {
            "passed": not failed, "residuals": residuals, "failed_components": failed,
            "nonfinite_components": nonfinite,
            "joint_names": list(self.joint_names), "joint_bounds": joint_bounds,
            "robot_coverage": coverage, "unresolved_scene": unresolved_scene,
            "scope": {
                "kind": "necessary_conditions_at_fixed_measured_entrance",
                "progress": self.problem.start_progress, "active_joint_position_velocity": "native_effective_limits",
                "native_self_collision_checked": bool(self_collision_checked),
                "scene": "actual native FK with the declared conservative scene and gripper model",
                "not_checked": ["acceleration", "jerk", "torque", "unprovided_locked_joint_velocity",
                                "future_trajectory_feasibility", "physical_task_success"],
            },
        }
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

    def _goal(self, progress):
        s = self.device_cfg.to_device([[float(progress)]])
        position, quaternion = self.problem.reference_to_link_pose(*self.problem.sample_reference(s))
        return self.GoalToolPose(tool_frames=[self.problem.tool_link],
                                 position=position[:, :, None, None].contiguous(),
                                 quaternion=quaternion[:, :, None, None].contiguous())

    def reference_seeds(self):
        """Return continuous native IK chains within the fixed seed budget.

        Nominal poses may be infeasible even when their allowed functional
        family contains a solution. If any nominal control pose has no native
        successful IK candidate, or both complete chains coincide numerically,
        use the primary chain and a measured-state initializer instead. Every
        initializer still requires the complete-process optimizer and checks.
        """
        torch, count = self.torch, self.budget.trajectory_seeds
        # BEGIN bounded distinct IK chains and measured-state initialization.
        previous = self.start.position[:, None].clone()
        sequence = []
        success_counts = []
        finite_reference_samples = 0
        self.reference_ik_distinct_counts = []
        # Control points are initialization, not a promise of exact reference
        # interpolation.  With four leading fixed cubic controls, free action
        # j has Greville time (j+3)/(N+4).  The first free action is not q0;
        # native boundary controls already enforce the measured start state.
        # All internal boundaries remain variables in full TrajOpt.
        control_u = (np.arange(self.trajopt.action_horizon)+3.)/(self.trajopt.action_horizon+4.)
        for u in control_u:
            progress = float(np.interp(u, self.progress_map.u, self.progress_map.s))
            self.seed_progress.fill_(progress)
            self.seed_parameter.fill_(float(u))
            result = self.ik.solve_pose(self._goal(progress), current_state=self.start,
                                       seed_config=previous, return_seeds=self.budget.ik_seeds)
            candidates = self.ik.get_active_js(result.js_solution).position
            selected, distinct_count = _select_ik_candidates(
                candidates[0].detach().cpu().numpy(), result.success[0].detach().cpu().numpy(),
                previous[0].detach().cpu().numpy(), count, self.problem.numerical.joint)
            # BEGIN an entirely broken IK query only loses this reference sample.
            if len(selected):
                finite_reference_samples += 1
                previous = candidates[:, selected].clone()
            else:
                previous = previous[:, :1].expand(1, count, -1).clone()
            # END actual and remaining finite chains stay intact.
            sequence.append(previous)
            success_counts.append(int((result.success & torch.isfinite(candidates).all(dim=-1)).sum().item()))
            self.reference_ik_distinct_counts.append(distinct_count)
        seeds = torch.stack(sequence, dim=2)
        self.initial_seed_strategies = ["reference_ik" if finite_reference_samples else "measured_state_constant"]
        if count == 2:
            distinct_chains = bool(torch.any((seeds[:, 0]-seeds[:, 1]).abs() > self.problem.numerical.joint))
            if all(success_counts) and distinct_chains:
                self.initial_seed_strategies.append("reference_ik_alternative")
            else:
                # Only free controls are constant. The original measured q0
                # and dq0 still enter the native fixed boundary via self.start.
                constant = self.trajopt._trajectory_seed_generator.generate_constant_seeds(
                    self.start.position, 1)
                seeds = torch.cat((seeds[:, :1], constant), dim=1)
                self.initial_seed_strategies.append("measured_state_constant")
        seeds = self.physical_domain.initialize(seeds)
        self.seed_strategies = list(self.initial_seed_strategies)
        return seeds.contiguous(), success_counts
        # END bounded distinct IK chains and measured-state initialization.

    def refine_validation(self, refinement_parameters):
        """Rebuild only validation workspaces in the already frozen binding.

        Release old solver workspaces before constructing new rollouts. IK and
        immutable robot/scene configuration are retained; no second planner
        and no duplicate native roadmap is allocated.
        """
        # BEGIN apply the same point and memory admission to every new clock.
        progress_map = self.progress_map
        parameters = np.unique(np.asarray(refinement_parameters, float))
        capacity = _anchor_capacity(progress_map, self.anchors, self.budget, parameters)
        scene_bytes, anchor_fk_bytes = _planning_workspace_bytes(
            self.scene, self.robot_metadata, self.budget, capacity)
        self.trajopt.destroy()
        self.trajopt = None
        # Native destroy() releases graphs, while optimizer callbacks can keep
        # rollout tensors in a reference cycle. Collect only at this explicit
        # replacement boundary before querying live allocator statistics.
        gc.collect()
        _require_planning_memory(self.torch, self.device_cfg.device, scene_bytes, anchor_fk_bytes)
        self.refinement_parameters = parameters
        self.anchor_capacity = capacity
        self.scene_workspace_bytes, self.anchor_fk_workspace_bytes = scene_bytes, anchor_fk_bytes
        manager_type = _process_manager_type(self.problem, self.scene, progress_map, self.anchors,
                                             self.budget, self.seed_progress, parameters,
                                             clock=self.formal_clock, seed_parameter=self.seed_parameter,
                                             robot_dict=self.robot_dict)
        self._trajectory_configuration["cost_manager_config_instance_type"] = manager_type
        self.trajopt = self._optimizer_class(
            self._optimizer_config_class.create(**deepcopy(self._trajectory_configuration)))
        # BEGIN refinement may add queries but cannot regenerate hand commands.
        if _formal_native_clock(self.trajopt) != self.formal_clock:
            raise ValueError("Validation refinement changed the frozen native clock")
        self._initialize_verification_constraints()
        self._bind_physical_domains()
        # END schedule changes require a separately bound planning request.

    # BEGIN whole-window spatial search on the already bound physical time.
    def ompl_seed(self, initial_seed, candidate_trajectory):
        """Return a spatial initializer and evidence; never produce a new clock.

        A failed complete curve and the finite reference controls provide
        endpoint witnesses. Intermediate reference/phase vertices do not
        become mandatory IK configurations. Variable-scope equalities stay
        in the exact validity relation; a global projection manifold is used
        only when its independent equality rows exist over the whole window.
        """
        torch = self.torch
        deadline = time.monotonic()+self.budget.search_seconds
        report = {"branches": [], "success": False, "state": "q,u=t/T",
                  "budget_seconds": self.budget.search_seconds, "clock_changed": False}
        lower = self.trajopt.transition_model.action_bound_lows.detach().cpu().numpy().reshape(-1)
        upper = self.trajopt.transition_model.action_bound_highs.detach().cpu().numpy().reshape(-1)
        tolerance = _projection_tolerance(self.problem)
        fixed_scene = self.scene.clone_for_rollout()
        map_u, map_s = self.device_cfg.to_device(self.progress_map.u), self.device_cfg.to_device(self.progress_map.s)
        query_u = self.progress_map.anchor_parameters(self.anchors)[:, 0]
        if self.finger_trajectory is not None:
            query_u = np.unique(np.r_[query_u,
                [event["applied_time_s"]/self.formal_clock["duration"] for event in self.finger_trajectory.events]])
        # The scene evaluator checks adjacent modes at this same q/u. Moving
        # float64 times by one ULP would vanish in the native float32 inputs.
        query_u = np.unique(query_u)
        start = np.r_[self.start.position[0].detach().cpu().numpy(), 0.]
        endpoints = [candidate_trajectory._curve(candidate_trajectory.duration)]
        endpoints.extend(initial_seed[0, :, -1].detach().cpu().numpy())
        invalid_states = []

        def state_parts(values):
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

        def residual_function(values):
            js, _, p, r, progress = state_parts(values.reshape(1, 1, -1))
            phase = min(self.problem.phase_count-1, int(progress.detach().item()))
            process = self.problem.project_residual(p, r, progress, phase, js.position, self.joint_names)
            drive = fixed_scene.project_residual(progress, js.position, self.joint_names)
            return torch.cat((process, drive), -1).reshape(-1)

        def equality(values, derivative):
            if time.monotonic() >= deadline:
                raise TimeoutError("The whole time-state search budget was exhausted")
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
            return answer.detach().cpu().numpy().astype(float)

        def valid(values):
            with torch.no_grad():
                values_t = self.device_cfg.to_device(values[None, None])
                js, state, p, r, progress = state_parts(values_t)
                clock_time = values_t[..., -1]*self.formal_clock["duration"]
                terms = {"process": self.problem.evaluate(p, r, progress, js.position,
                    self.joint_names, joint_velocity=js.velocity),
                    "scene": _scene_boundary_evaluate(fixed_scene, self.problem, self.progress_map,
                        self.problem.transform_robot_spheres(state.robot_spheres), p, r, progress,
                        js.position, self.joint_names, trajectory_time=clock_time,
                        normalized_time=values_t[..., -1])}
                native = self._verification_native.compute_costs(_timed_native_state(
                    self._RobotState(joint_state=js, cuda_robot_model_state=state), self.problem,
                    fixed_scene, self.joint_names, clock_time,
                    sphere_padding=_native_collision_padding(self._verification_native)))
                terms.update({f"native:{name}": value for name, value in zip(native.names, native.values)})
                coverage = fixed_scene.check_robot_coverage(js.position, self.joint_names, clock_time,
                    entrance=bool(values[-1] == 0 and np.array_equal(values, start)))
                if coverage:
                    terms["robot_query_incomplete"] = progress.new_tensor([[float(any(
                        not row["attempt_allowed"] for row in coverage))]])
                accepted = all(bool(torch.isfinite(term).all()) and bool((term <= 0).all())
                               for term in terms.values())
                if not accepted and len(invalid_states) < 3:
                    invalid_states.append({"u": float(values[-1]), "progress": float(progress.item()),
                        "terms": {name: float(term.max().item()) for name, term in terms.items()}})
                return accepted

        for branch, endpoint in enumerate(endpoints):
            if time.monotonic() >= deadline:
                break
            goal = np.r_[endpoint, 1.]
            invalid_states.clear()
            try:
                # OMPL requires constant codimension. Scoped constraints are
                # still checked by valid() at every edge/query sample.
                probes = np.unique(np.r_[0., query_u, (query_u[:-1]+query_u[1:])/2., 1.])
                ranks = []
                for parameter in probes:
                    witness = start+(goal-start)*parameter
                    jac = equality(witness, True)
                    ranks.append((len(jac), np.linalg.matrix_rank(jac, tol=1e-7) if len(jac) else 0))
                projected = bool(ranks and ranks[0][0] and len(set(ranks)) == 1 and ranks[0][0] == ranks[0][1])
                constraint = equality if projected else lambda value, derivative: (
                    np.empty((0, len(value))) if derivative else np.empty(0))
                if projected:
                    goal = _project_boundary(goal, equality, tolerance, deadline=deadline)
                    if goal is None:
                        report["branches"].append({"failure": "terminal_witness_projection_failed"})
                        continue
                path, evidence = _ompl_stage(start, goal, np.r_[lower, 0.], np.r_[upper, 1.],
                    valid, constraint, tolerance=tolerance, step=self.budget.validation_joint_step,
                    seconds=max(0., deadline-time.monotonic()), sample_limit=self.budget.max_validation_points,
                    query_parameters=query_u)
            except TimeoutError:
                path, evidence = None, {"failure": "whole_search_time_budget_exhausted"}
            evidence["first_invalid_states"] = deepcopy(invalid_states)
            report["branches"].append(evidence)
            if path is None:
                continue
            keep = np.r_[True, np.any(path[1:] != path[:-1], axis=1)]
            path = path[keep]
            if np.any(np.diff(path[:, -1]) <= 0):
                report["branches"][-1]["failure"] = "nonmonotone_or_instantaneous_joint_motion"
                continue
            control_u = (np.arange(self.trajopt.action_horizon)+3.)/(self.trajopt.action_horizon+4.)
            controls = np.stack([np.interp(control_u, path[:, -1], path[:, column])
                for column in range(len(self.joint_names))], axis=-1)
            report.update(success=True, samples=len(path), selected_branch=branch)
            return self.physical_domain.initialize(self.device_cfg.to_device(controls[None, None])), report
        report["failure"] = "no_complete_time_state_seed_within_budget"
        return None, report
    # END OMPL has no schedule replacement return value or side effect.

    # BEGIN only a true complete terminal pose enables native H=1 tracking.
    def _terminal_tolerances(self):
        """Enclose the functional goal in native link-error coordinates.

        A rotated tool with a lever arm can meet a functional position goal
        while its link origin moves. The native spherical position envelope
        must include that entire allowed set; the original typed requirement
        still evaluates the exact functional-frame position and orientation.
        """
        target = self.problem.terminal_pose_target
        if target is None:
            return self.problem.numerical.position_m, self.problem.numerical.rotation_rad
        link_from_function = self.problem.tool_transform @ target.tool_from_function
        lever = float(np.linalg.norm(link_from_function[:3, 3]))
        # Native convergence uses strict '<' on independently computed link
        # errors. Its envelope includes the request's already declared
        # numerical allowances; the typed functional goal does not. This
        # prevents that secondary convergence test narrowing the true goal.
        return (target.position_tolerance_m+2*lever*np.sin(target.rotation_tolerance_rad/2)
                +self.problem.numerical.position_m,
                target.rotation_tolerance_rad+self.problem.numerical.rotation_rad)

    def _terminal_goal(self):
        target = self.problem.terminal_pose_target
        if target is None:
            return None
        from scipy.spatial.transform import Rotation
        pose = target.target @ np.linalg.inv(target.tool_from_function)
        position = self.device_cfg.to_device(pose[:3, 3][None, None])
        quaternion = self.device_cfg.to_device(Rotation.from_matrix(pose[:3, :3]).as_quat()[[3, 0, 1, 2]][None, None])
        position, quaternion = self.problem.reference_to_link_pose(position, quaternion)
        return self.GoalToolPose(tool_frames=[self.problem.tool_link],
            position=position[:, :, None, None].contiguous(),
            quaternion=quaternion[:, :, None, None].contiguous())

    def optimize(self, seed):
        self.physical_domain.validate(seed)
        goal = self._terminal_goal()
        if goal is None:
            # Native storage still requires an H=1 pose even when every real
            # goal is intermediate or an allowed set. All such goals enter
            # ProcessCostManager; this inert metadata supplies no residual.
            self.trajopt.disable_tool_pose_tracking()
            goal = self._goal(self.problem.phase_count)
        else:
            self.trajopt.enable_tool_pose_tracking()
        return self.trajopt.solve_pose(
            goal_tool_poses=goal, current_state=self.start,
            seed_traj=seed, num_seeds=seed.shape[1], return_seeds=seed.shape[1],
            dt=self.torch.full((1, seed.shape[1]), self.budget.maximum_trajectory_dt,
                               dtype=seed.dtype, device=seed.device),
            use_implicit_goal=False, goal_state=None,
            finetune_attempts=0, finetune_dt_scale=1.)
    # END intermediate interaction goals never become whole-window endpoints.

    def native_diagnostics(self, result, seed_index=None):
        """Expose the native reason for rejection, including dense metrics.

        Native H=1 tracking only represents a complete terminal pose, with
        a link-frame envelope enclosing its functional-frame allowed set.
        Every actual skill error is independently checked on the exported
        curve in :meth:`export`, including intermediate and allowed-set goals.
        """
        success = result.success if seed_index is None else result.success[:, seed_index]
        report = {"success": bool(success.all().item())}

        def maximum(value):
            # Native rollout batch is [batch*seeds,H,...]. This planner's
            # batch is explicitly one, so this is an exact seed slice.
            if seed_index is not None:
                if result.batch_size != 1 or value.shape[0] != result.num_seeds:
                    raise ValueError("Native metrics do not match the single-batch seed layout")
                value = value[seed_index]
            return float(value.max().item())

        for label, metrics in (("rollout", result.metrics),
                               ("interpolated", result.interpolated_metrics)):
            if metrics is None:
                continue
            constraints = metrics.costs_and_constraints.constraints
            report[label] = {
                "constraints": {name: maximum(value)
                                for name, value in zip(constraints.names, constraints.values)},
                "convergence": {name: maximum(value)
                                for name, value in zip(metrics.convergence.names, metrics.convergence.values)},
            }
        return report

    # BEGIN all-candidate hard verification and continuation selection.
    def select_candidate(self, result):
        """Verify all budgeted curves before selecting a result or continuation.

        A curve is accepted only when its native constraints and independent
        exported-curve constraints both pass. Native smoothness/time ranking
        applies only within that accepted set. If none passes, compare the
        actual nonnegative constraint residuals to retain useful progress for
        bounded re-optimization; this score never changes an acceptance test.
        """
        if result.batch_size != 1 or not 1 <= result.num_seeds <= self.budget.trajectory_seeds:
            raise ValueError("Native results exceed the declared single-batch candidate budget")
        reports, selected, selected_key = [], None, None

        def violation_sum(values):
            values = tuple(values)
            return (sum(max(0., value) for value in values)
                    if all(np.isfinite(value) for value in values) else float("inf"))

        for index in range(result.num_seeds):
            # Read native metrics before native gather intentionally drops its
            # all-seed metric buffers. Keep original seed identity in reports.
            native_metrics = self.native_diagnostics(result, seed_index=index)
            native_success = native_metrics["success"]
            native_cost = float(result.seed_cost[0, index].item())
            candidate = _trajectory_candidate(result, index)
            report = {"seed_index": index, "native_success": native_success,
                      "native_metrics": native_metrics, "native_seed_cost": native_cost}
            try:
                trajectory, points = self.export(candidate)
            except ValueError as error:
                # A bounded export rejection is local to this seed. Another
                # already allocated seed can still provide a valid curve.
                reports.append({**report, "accepted": False, "export_error": str(error)})
                continue
            final_success = all(np.isfinite(value) and value <= 0
                                for value in trajectory.residuals.values())
            accepted = native_success and final_success
            exported_violation = violation_sum(trajectory.residuals.values())
            native_violation = violation_sum(
                value for label in ("rollout", "interpolated") if label in native_metrics
                for value in native_metrics[label]["constraints"].values())
            continuation_violation = exported_violation+native_violation
            report.update(accepted=accepted, validation_points=points,
                          residuals=trajectory.residuals,
                          measurements=self.last_validation["measurements"],
                          numerical_tolerances=self.last_validation["numerical_tolerances"],
                          continuation_violation=continuation_violation)
            reports.append(report)
            quality = native_cost if np.isfinite(native_cost) else float("inf")
            provenance = self.seed_strategies[index]
            report["seed_provenance"] = provenance
            demo_priority = 0 if provenance.startswith("reference_ik") else 1
            key = (0, demo_priority, quality, index) if accepted else (1, continuation_violation, index)
            if selected_key is None or key < selected_key:
                selected_key = key
                selected = candidate, trajectory, self.last_validation, report
        if selected is None:
            errors = "; ".join(f"seed {item['seed_index']}: {item['export_error']}" for item in reports)
            raise ValueError(f"No native candidate has an exportable bounded curve: {errors}")
        candidate, trajectory, validation, report = selected
        # The last evaluated seed need not be the selected seed. Refinement
        # must use witnesses from the exact candidate whose controls continue.
        self.last_validation = validation
        attempt = {**report, "selected_seed_index": report["seed_index"], "candidates": reports}
        return candidate, trajectory, attempt
    # END all-candidate hard verification and continuation selection.

    def refinement_samples(self):
        """Select uncovered failures within the separate refinement allowance."""
        violations = self.last_validation["violations"]
        if not np.isfinite(violations).all():
            return None
        fixed = self.progress_map.anchor_parameters(self.anchors)[:, 0]
        existing = np.r_[fixed, self.refinement_parameters,
                         np.linspace(0, 1, self.trajopt.horizon)]
        # BEGIN required evidence never consumes the optional refinement limit.
        remaining = min(self.budget.max_refinement_points,
                        self.budget.max_validation_points-len(fixed))-len(self.refinement_parameters)
        # END required evidence never consumes the optional refinement limit.
        selected = []
        for index in np.argsort(-violations, kind="stable"):
            if violations[index] <= 0 or len(selected) >= remaining:
                break
            parameter = float(self.last_validation["u"][index])
            if np.min(np.abs(existing-parameter)) <= 1e-7:
                continue
            if selected and np.min(np.abs(np.asarray(selected)-parameter)) <= 1e-7:
                continue
            selected.append(parameter)
        return np.unique(np.r_[self.refinement_parameters, selected]) if selected else None

    def export(self, result):
        """Build the same cubic on CPU, then check exact anchors and midpoints."""
        if result.batch_size != 1 or result.num_seeds != 1:
            raise ValueError("Export requires exactly one native batch and one selected seed")
        torch = self.torch
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
        anchor_pairs = self.progress_map.anchor_parameters(self.anchors)
        event_records = []
        for event in self.problem.events:
            event_records.append({**asdict(event),
                                  **self.progress_map.event_clock[(event.skill_id, event.event_id)]})
        trajectory = JointTrajectory(self.joint_names, times, position, velocity,
                                     self.progress_map.u.copy(), self.progress_map.s.copy(),
                                     tuple(event_records), deepcopy(self.problem.conditions), {})
        trajectory.finger_trajectory = self.finger_trajectory
        # BEGIN complete the existing native acceleration bound between samples.
        # These are the actual verification manager's named limits, after
        # native model construction. Input YAML scalars and sampled right-side
        # accelerations cannot substitute for the complete exported curve.
        cspace = self._verification_native.get_cost("cspace")
        if cspace is None or not cspace.enabled:
            raise ValueError("Export requires enabled native joint-state limits")
        limits = cspace.config.joint_limits
        derivatives = cubic_derivative_report(
            trajectory, limits.acceleration.detach().cpu().numpy(), joint_names=limits.joint_names)
        # END no new bound, tolerance, C2 requirement or retiming is introduced.
        # Each native interval is one cubic piece.  Use its derivative extrema
        # to bound joint displacement between validation queries; endpoints
        # alone could hide a reversing excursion.
        dense_times = [0.]
        coefficients = trajectory._curve.c
        for index, width in enumerate(np.diff(times)):
            c0, c1, c2 = coefficients[0, index], coefficients[1, index], coefficients[2, index]
            vertex = np.divide(-c1, 3*c0, out=np.zeros_like(c0), where=np.abs(c0) > 1e-15)
            vertex = np.clip(vertex, 0, width)
            speeds = np.maximum.reduce((np.abs(c2), np.abs(3*c0*width**2 + 2*c1*width + c2),
                                        np.abs(3*c0*vertex**2 + 2*c1*vertex + c2)))
            required = width * np.linalg.norm(speeds) / self.budget.validation_joint_step
            if (not np.isfinite(required)
                    or required > self.budget.max_validation_points-len(dense_times)):
                raise ValueError("Final curve validation exceeds its declared finite sample budget")
            subdivisions = max(2, int(np.ceil(required)))
            if len(dense_times)+subdivisions > self.budget.max_validation_points:
                raise ValueError("Final curve validation exceeds its declared finite sample budget")
            dense_times.extend(np.linspace(times[index], times[index+1], subdivisions+1)[1:])
        queries = np.unique(np.concatenate((dense_times, anchor_pairs[:, 0]*times[-1])))
        if len(queries) > self.budget.max_validation_points:
            raise ValueError("Final curve validation exceeds its declared finite sample budget")
        q, dq, progress = trajectory.sample(queries)
        q_t = self.device_cfg.to_device(q[None])
        dq_t = self.device_cfg.to_device(dq[None])
        s_t = self.device_cfg.to_device(progress[None])
        with torch.no_grad():
            joint_state = self.JointState(
                position=q_t, velocity=dq_t,
                acceleration=self.device_cfg.to_device(trajectory._curve(queries, 2)[None]),
                jerk=self.device_cfg.to_device(trajectory._curve(queries, 3)[None]),
                dt=self.device_cfg.to_device([dt]), joint_names=list(self.joint_names))
            state = self._verification_fk.compute_kinematics(joint_state)
            tool = state.tool_poses
            tool_index = tool.tool_frames.index(self.problem.tool_link)
            p, r = self.problem.transform_tool_pose(tool.position[:, :, tool_index], tool.quaternion[:, :, tool_index])
            components = self.problem.residuals(p, r, s_t, q_t, self.joint_names, joint_velocity=dq_t)
            components["scene"] = _scene_boundary_evaluate(self._verification_scene,
                self.problem, self.progress_map, self.problem.transform_robot_spheres(state.robot_spheres),
                p, r, s_t, q_t, self.joint_names,
                trajectory_time=self.device_cfg.to_device(queries[None]),
                normalized_time=self.device_cfg.to_device((queries/times[-1])[None]))
            native_constraints = self._verification_native.compute_costs(
                _timed_native_state(self._RobotState(joint_state=joint_state, cuda_robot_model_state=state),
                    self.problem, self._verification_scene, self.joint_names, self.device_cfg.to_device(queries[None]),
                    sphere_padding=_native_collision_padding(self._verification_native)))
            for name, value in zip(native_constraints.names, native_constraints.values):
                components[f"native:{name}"] = value
            residuals = {name: float(value.max().item()) for name, value in components.items()}
            measurements = {name: float(value.max().item()) for name, value in
                            self.problem.measurements(p, r, s_t).items()}
            # Retain failing curve coordinates for bounded mesh refinement.
            # Nonfinite terms always fail acceptance and are never clipped.
            violation = torch.zeros_like(s_t)
            for name, value in components.items():
                reduced = value.reshape(value.shape[0], value.shape[1], -1).amax(-1)
                violation = torch.maximum(violation, reduced)
                if residuals[name] > 0:
                    index = int(reduced.reshape(-1).argmax().item())
                    measurements[f"{name}:maximum_violation_u"] = float(queries[index]/times[-1])
                    measurements[f"{name}:maximum_violation_progress"] = float(progress[index])
                    measurements[f"{name}:maximum_violation_time"] = float(queries[index])
            self.last_validation = {
                "measurements": measurements,
                "numerical_tolerances": asdict(self.problem.numerical),
                "u": queries/times[-1],
                "violations": violation.detach().cpu().numpy().reshape(-1),
            }
            # Coverage remains evidence; only an actual query/binding failure
            # contributes a rejection. Missing observed free space is allowed.
            coverage = self._verification_scene.check_robot_coverage(
                torch.as_tensor(q[None, :1], dtype=torch.float64), self.joint_names,
                torch.as_tensor(queries[None, :1], dtype=torch.float64), entrance=True)
            coverage += self._verification_scene.check_robot_coverage(
                torch.as_tensor(q[None, 1:], dtype=torch.float64), self.joint_names,
                torch.as_tensor(queries[None, 1:], dtype=torch.float64))
            if coverage:
                unknown = np.asarray([0. if row["attempt_allowed"] else 1. for row in coverage])
                residuals["unknown/new_robot_volume"] = float(unknown.max())
                self.last_validation["violations"] = np.maximum(self.last_validation["violations"], unknown)
                self.last_validation["robot_coverage"] = {
                    "queries": len(coverage), "attempt_allowed": bool(not unknown.any()),
                    "passed": all(row["passed"] for row in coverage),
                    "failed": [{"time_s": float(queries[index]), "result": row}
                               for index, row in enumerate(coverage) if not row["passed"]][:16],
                    "scope": "native solid volume at every exported validation sample; not continuous collision proof"}
        # BEGIN final scope and physical-domain checks refer to this exact curve.
        scope_issues = self.problem.verification_issues(progress)
        self.last_validation["scope_issues"] = scope_issues
        residuals["required_relation_query_incomplete"] = float(bool(scope_issues))
        columns = [limits.joint_names.index(name) for name in self.joint_names]
        position_limits = limits.position[:, columns].detach().cpu().numpy()
        position_report = cubic_position_report(trajectory._curve, *position_limits)
        residuals["continuous_position"] = position_report["maximum_bound_excess"]
        self.last_validation["continuous_position"] = position_report
        # END sampled native success does not replace polynomial or history checks.
        residuals["initial_position"] = max(0., float(np.max(np.abs(position[0]-self.start.position[0].detach().cpu().numpy())))
                                              - self.problem.numerical.joint)
        residuals["initial_velocity"] = max(0., float(np.max(np.abs(velocity[0]-self.start.velocity[0].detach().cpu().numpy())))
                                              - self.problem.numerical.velocity)
        # BEGIN require the existing acceleration limits on every cubic piece.
        # This residual is raw joint-coordinate acceleration excess, not a
        # weighted native cost. The original candidate selector already rejects
        # every positive residual. Jump magnitudes remain diagnostic only.
        residuals["continuous_acceleration"] = derivatives["acceleration"]["maximum_bound_excess"]
        self.last_validation["cubic_derivatives"] = derivatives
        witness = derivatives["acceleration"]["maximum_violation"]
        if witness is not None:
            parameter = witness["time_s"]/times[-1]
            self.last_validation["measurements"].update({
                "continuous_acceleration:maximum_violation_time": witness["time_s"],
                "continuous_acceleration:maximum_violation_u": parameter,
                "continuous_acceleration:maximum_violation_progress": float(np.interp(
                    parameter, self.progress_map.u, self.progress_map.s)),
            })
            # The existing bounded refinement consumes these coordinates and
            # residuals, not measurements. Retain the exact analytic witness
            # even if that time already has a sampled right-side acceleration;
            # cubic_derivatives above identifies the failing piece and side.
            self.last_validation["u"] = np.r_[self.last_validation["u"], parameter]
            self.last_validation["violations"] = np.r_[
                self.last_validation["violations"], witness["excess"]]
        # END continuous acceleration acceptance; C1/C2 reports do not add gates.
        trajectory.residuals = residuals
        return trajectory, len(queries)


def plan_trajectory(problem, scene, robot, *, budget=None, device="cuda:0", continuation=None):
    """Generate and verify a complete-window robot trajectory with cuRobo V2.

    ``robot`` is an existing V2 robot YAML path or configuration dictionary.
    Scene geometry, calibration, measured joints and skill tolerances are
    explicit inputs; a robot name never supplies missing perception evidence.
    GPU work starts only when this function is called.
    """
    prior = None
    if continuation is not None:
        try:
            budget, prior = _prepare_continuation(problem, scene, budget, continuation)
        except (ContinuationBindingError, ValueError) as error:
            return PlanningResult(False, None, {"failure": "continuation_binding_gap", "reason": str(error)})
    budget = PlanningBudget() if budget is None else budget
    request_budget = asdict(budget) if prior is None else prior["request_budget"]
    consumed = {"trajectory_solves": 0, "search_seconds": 0.} if prior is None else deepcopy(prior["consumed_planning"])
    started = time.monotonic()
    try:
        planner = _CuroboPlanner(problem, scene, robot, budget, device, continuation=prior)
    except ContinuationBindingError as error:
        return PlanningResult(False, None, {"failure": "continuation_binding_gap", "reason": str(error),
            "planning_budget": request_budget, "consumed_planning": consumed})
    # BEGIN cleanup also covers diagnostic construction failures.
    diagnostics = {}
    try:
        diagnostics = {"backend": "curobo_v2", "batch_size": 1, "ik_seeds": budget.ik_seeds,
                       "trajectory_seeds": budget.trajectory_seeds, "used_ompl": False,
                       "timing_policy": "source_proposal_then_frozen_binding",
                       "native_trajectory_dt": planner.budget.maximum_trajectory_dt,
                       "progress_parameter_error_bound": _PROGRESS_PARAMETER_ERROR,
                       "progress_clock_max_keys": _MAX_PROGRESS_KEYS,
                       "scene": scene.describe(), "scene_workspace_bytes": planner.scene_workspace_bytes,
                       # BEGIN expose the request-sized native anchor allocation.
                       "anchor_fk_workspace_bytes": planner.anchor_fk_workspace_bytes,
                       "exact_anchor_capacity": planner.anchor_capacity,
                       # END expose the request-sized native anchor allocation.
                       "attempts": [], "planning_budget": request_budget, "consumed_planning": consumed}
        # END the planner always reaches its precision-restoring close().
        # BEGIN fixed entrance failures cannot be repaired by later IK or search.
        diagnostics["initial_state"] = planner.initial_state_diagnostics()
        if not diagnostics["initial_state"]["passed"]:
            diagnostics["failure"] = "initial_state_infeasible"
            return PlanningResult(False, None, diagnostics)
        # END fixed entrance failures cannot be repaired by later IK or search.
        seeds, ik_success = planner.reference_seeds()
        diagnostics["reference_ik_success_counts"] = ik_success
        # BEGIN report the actual native execution and selected seed policy.
        diagnostics["ik_exit_early"] = bool(planner.ik.config.exit_early)
        diagnostics["reference_ik_distinct_counts"] = planner.reference_ik_distinct_counts
        diagnostics["initial_seed_strategies"] = planner.initial_seed_strategies
        # END report the actual native execution and selected seed policy.
        reference_seeds = seeds
        searched = False
        for iteration in range(budget.maximum_trajectory_solves):
            consumed["trajectory_solves"] += 1
            result, trajectory, attempt = planner.select_candidate(planner.optimize(seeds))
            native_success = attempt["native_success"]
            diagnostics["attempts"].append(attempt)
            diagnostics.update(attempt)
            if attempt["accepted"]:
                diagnostics["executable_until"] = trajectory.executable_until
                diagnostics["next_verification"] = trajectory.next_verification
                # BEGIN formal execution handoff uses this exact verified candidate.
                from .execution_binding import bind_candidate
                binding = bind_candidate(trajectory, problem=problem, scene=scene,
                    validation=planner.last_validation, native_report=attempt)
                binding.update(planning_budget=deepcopy(request_budget), consumed_planning=deepcopy(consumed),
                    request_events=([asdict(event) for event in problem.events] if prior is None else prior["request_events"]),
                    consumed_events=[] if prior is None else prior["consumed_events"],
                    goal_identities=({collection: list(range(len(getattr(problem, collection))))
                                      for collection in ("pose_events", "coordinate_events", "supports")}
                                     if prior is None else prior["goal_identities"]),
                    completed_goal_ids=[] if prior is None else prior["completed_goal_ids"],
                    prior_effects={} if prior is None else prior["effects"],
                    request_time_origin_s=0. if prior is None else prior["request_time_origin_s"])
                binding["request_time_origin_step"] = 0 if prior is None else prior["request_time_origin_step"]
                if prior is not None:
                    binding.update({name: prior[name] for name in (
                        "physics_budget", "consumed_physics_steps", "remaining_physics_steps")})
                return PlanningResult(True, trajectory, diagnostics, binding)
                # END a diagnostic label alone never authorizes execution.
            if iteration+1 >= budget.maximum_trajectory_solves:
                break

            additional = planner.refinement_samples()
            # A native success invalidated between its rollout samples calls
            # for exact witness constraints on this same cubic first.
            if native_success and additional is not None:
                seeds = result.solution.detach().clone()
                planner.seed_strategies = [attempt["seed_provenance"]]
                planner.refine_validation(additional)
                attempt["next_action"] = "refine_uncovered_curve_samples"
                attempt["refinement_parameters"] = additional.tolist()
                continue

            if not searched and budget.search_seconds > 0:
                searched = True
                search_started = time.monotonic()
                path_seed, search_report = planner.ompl_seed(reference_seeds, trajectory)
                consumed["search_seconds"] += time.monotonic()-search_started
                diagnostics["ompl"] = search_report
                if path_seed is not None:
                    seeds = path_seed
                    planner.seed_strategies = ["ompl"]
                    diagnostics["used_ompl"] = True
                    attempt["next_action"] = "optimize_complete_ompl_seed"
                    continue

            if additional is not None:
                seeds = result.solution.detach().clone()
                planner.seed_strategies = [attempt["seed_provenance"]]
                planner.refine_validation(additional)
                attempt["next_action"] = "refine_uncovered_curve_samples"
                attempt["refinement_parameters"] = additional.tolist()
                continue
            break

        diagnostics["failure"] = ("exported_curve_constraint_violation" if diagnostics["native_success"]
                                  else "complete_process_optimization_not_feasible")
        return PlanningResult(False, None, diagnostics)
    except JointDomainInitializationError as error:
        diagnostics["failure"] = "physical_domain_initializer_not_constructed"
        diagnostics["reason"] = str(error)
        return PlanningResult(False, None, diagnostics)
    finally:
        # BEGIN report the final admitted schedule's actual capacity estimates.
        diagnostics["scene_workspace_bytes"] = planner.scene_workspace_bytes
        diagnostics["anchor_fk_workspace_bytes"] = planner.anchor_fk_workspace_bytes
        diagnostics["exact_anchor_capacity"] = planner.anchor_capacity
        # END report the final admitted schedule's actual capacity estimates.
        diagnostics["elapsed_seconds"] = time.monotonic()-started
        planner.close()


# END complete-window robot planning and generation-time process constraints.
