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
from dataclasses import asdict, dataclass, field, fields
import time

import numpy as np
from scipy.interpolate import BSpline, CubicHermiteSpline
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class PlanningBudget:
    """Explicit local resource and numerical-work limits, for a single request.

    These defaults bound the current 16 GB RAM / 12 GB GPU integration.  More
    seeds or a larger batch must not be silently allocated by native defaults.
    The interpolation buffer covers the largest allowed native knot interval.
    Native solvers use eager execution for the dynamic process/scene tensors.
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
    max_anchors: int = 64
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
                or not 1 <= self.max_anchors <= 128
                or not 1 <= self.max_robot_spheres <= 512
                or not 1 <= self.max_robot_dof <= 16
                or not 1 <= self.max_robot_links <= 64
                or not 32 <= self.max_validation_points <= 8192
                or not 1 <= self.maximum_trajectory_solves <= 4):
            raise ValueError("Planning budget exceeds the bounded local single-batch configuration")
        if type(self.max_scene_workspace_bytes) is not int or not 0 <= self.max_scene_workspace_bytes <= 1024**3:
            raise ValueError("Scene workspaces must remain within the local one-GiB allocation budget")
        for name in ("interpolation_dt", "maximum_trajectory_dt", "constraint_weight", "validation_joint_step"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not np.isfinite(self.search_seconds) or self.search_seconds < 0:
            raise ValueError("search_seconds must be finite and nonnegative")
        # V2 uses degree+1 virtual knots and floor(knot_dt / output_dt)+1
        # samples per knot interval.  Validate before native buffers exist.
        maximum_count = ((self.knots + 4)
                         * int(self.maximum_trajectory_dt * self.interpolation_steps / self.interpolation_dt + 1)
                         + 1)
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


@dataclass
class PlanningResult:
    success: bool
    trajectory: JointTrajectory | None
    diagnostics: dict


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

    def __post_init__(self):
        if (self.u.ndim != 1 or self.s.shape != self.u.shape or len(self.u) < 2
                or not np.isfinite(self.u).all() or not np.isfinite(self.s).all()
                or self.u[0] != 0 or self.u[-1] != 1 or self.s[0] != 0
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


def _ompl_stage(start, goal, lower, upper, valid, equality, *, tolerance,
                step, seconds, sample_limit):
    """Run native directed RRTConnect in (q,s), retaining checked edge samples.

    ``equality(x, derivative)`` returns true equality residuals or their
    Jacobian.  An empty residual selects ordinary OMPL space; inequality
    bands are checked by ``valid`` and are never collapsed to center lines.
    The same exporter accepts edges during tree growth and reconstructs the
    solution, including the native geodesic's final delta-length tail.
    """
    from ompl import base as ob, geometric as og

    dimension = len(start)
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
        if last[-1] < first[-1] or not valid_array(first) or not valid_array(last):
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
            for index in range(1, subdivisions+1):
                # The caller's true endpoints must stay exact; projecting a
                # nearby replacement endpoint would accept a different edge.
                point = right.copy() if index == subdivisions else left+(right-left)*(index/subdivisions)
                if index != subdivisions and constraint is not None and not constraint.project(point):
                    return None
                if point[-1] < checked[-1][-1] or not valid_array(point):
                    return None
                checked.append(point)
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


def _process_manager_type(problem, scene, progress_map, anchors, budget, seed_progress,
                          refinement_parameters=()):
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
            super().initialize_from_config(config, transition_model, scene_collision_checker, **kwargs)
            self._process_mode = config.process_mode
            self._dense_counts = None
            if not self._process_mode:
                return
            transition_model.robot_model.compute_jacobian = True
            self._scene = scene.clone_for_rollout()
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
                scene_value = self._scene.evaluate(problem.transform_robot_spheres(spheres), position, quaternion,
                                                   progress, q, self._joint_names)
                if self._process_mode == "seed_optimizer":
                    scene_value = scene_value.square()
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
            order = torch.argsort(torch.cat((u, anchor_u), 1), dim=1, stable=True)
            inverse = torch.argsort(order, dim=1)

            def ordered(value):
                index = order.reshape(*order.shape, *((1,)*(value.ndim-2))).expand_as(value)
                return value.gather(1, index)

            position, quaternion = problem.transform_tool_pose(ordered(position), ordered(quaternion))
            spheres = problem.transform_robot_spheres(ordered(spheres))
            q, dq, progress = ordered(q), ordered(dq), ordered(progress)
            process_value = problem.evaluate(position, quaternion, progress, q, self._joint_names,
                                             joint_velocity=dq)
            scene_value = self._scene.evaluate(spheres, position, quaternion, progress, q, self._joint_names)
            valid = torch.arange(horizon, device=q.device)[None, :] < counts[:, None]
            values = [("skill_process", process_value), ("process_scene", scene_value)]
            # A squared optimizer penalty has a gradient proportional to
            # the remaining violation. Metrics retain the original raw
            # nonnegative values and therefore their original zero sets.
            if self._process_mode == "trajectory_optimizer":
                values = [(name, value.square()) for name, value in values]
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
    feasible = np.flatnonzero(success)
    available = feasible if len(feasible) else np.arange(len(candidates))
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

    def __init__(self, problem, scene, robot, budget, device):
        self.ik = self.trajopt = None
        try:
            self._initialize(problem, scene, robot, budget, device)
        except BaseException:
            # A later native configuration failure must release an IK solver
            # already created in this process before propagating the error.
            self.close()
            raise

    def _initialize(self, problem, scene, robot, budget, device):
        # Validate grounded evidence and bound independent scene workspaces
        # before importing native GPU solvers or creating any of their buffers.
        scene.validate_problem(problem)
        anchors = np.unique(np.concatenate((problem.anchors, np.asarray(scene.required_progress_anchors, float))))
        if (len(anchors) > budget.max_anchors or not np.isfinite(anchors).all()
                or np.any(anchors < 0) or np.any(anchors > problem.phase_count)):
            raise ValueError("All required source knots and scene anchors must fit the finite planning horizon and budget")
        if problem.reference_frame != scene.reference_frame:
            raise ValueError("Problem and planning scene must use the same explicitly calibrated reference frame")
        self.progress_map, _ = _progress_for_cubic_boundaries(
            _ProgressMap(np.array([0., 1.]), np.array([0., float(problem.phase_count)])),
            budget.knots, problem.start_dq, stop_progress=problem.stop_progress, anchors=anchors)
        if len(self.progress_map.anchor_parameters(anchors)) > budget.max_anchors:
            raise ValueError("Exact anchors including native boundary pauses exceed the declared anchor budget")
        horizon = (budget.knots+4)*budget.interpolation_steps+1
        # Native LBFGS has two rollout instances and four line-search particles.
        # Count both optimization/metrics managers conservatively, including
        # the seed initializer, dense export, and separate verification scene.
        allocation_shapes = (
            (4, budget.ik_seeds, 1), (8, budget.ik_seeds*4, 1),
            (4, budget.trajectory_seeds, horizon+budget.max_anchors),
            (4, budget.trajectory_seeds*4, horizon+budget.max_anchors),
            (2, budget.trajectory_seeds, budget.interpolation_buffer_size+budget.max_anchors),
            (1, 1, budget.max_validation_points), (2, 1, budget.max_anchors),
        )
        self.scene_workspace_bytes = sum(copies*scene.estimated_workspace_bytes(batch, length)
                                         for copies, batch, length in allocation_shapes)
        if self.scene_workspace_bytes > budget.max_scene_workspace_bytes:
            raise ValueError("Independent scene query workspaces exceed the declared local allocation budget")
        robot_dict, self.robot_metadata = _bounded_robot_configuration(robot, problem, scene, budget)
        import torch
        from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
        from curobo.trajectory_optimizer import TrajectoryOptimizer, TrajectoryOptimizerCfg
        from curobo.kinematics import Kinematics
        from curobo.types import JointState, GoalToolPose
        from curobo._src.types.device_cfg import DeviceCfg
        from curobo._src.util.config_io import load_yaml, join_path
        from curobo.content import get_task_configs_path

        self.problem, self.scene, self.budget = problem, scene, budget
        self.torch, self.JointState, self.GoalToolPose = torch, JointState, GoalToolPose
        self.device_cfg = DeviceCfg(device)
        free_device_bytes, _ = torch.cuda.mem_get_info(self.device_cfg.device)
        # Reserve two GiB beyond geometry for the bounded native model,
        # optimizer, FK/Jacobian and library workspaces measured by the small
        # integration probe. Existing perception allocations remain visible
        # through the actual device free-memory query.
        if free_device_bytes < self.scene_workspace_bytes+2*1024**3:
            raise ValueError("Available GPU memory does not cover the bounded planner and scene workspaces")
        self.ik = self.trajopt = None
        self.anchors = anchors
        self.robot_dict = robot_dict
        self.seed_progress = self.device_cfg.to_device([[0.]])
        self._manager_type = _process_manager_type(problem, scene, self.progress_map, self.anchors, budget, self.seed_progress)
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
        trajectory_optimizer = configuration("trajopt/lbfgs_bspline_trajopt.yml", "trajectory_optimizer")
        # First find a feasible process.  Native unconditional smoothness and
        # energy preferences must not purchase a process violation.  Bounds
        # retain their independent weights and final feasibility checks.
        trajectory_optimizer["rollout"]["cost_cfg"]["cspace_cfg"]["squared_l2_regularization_weight"] = [0.]*5
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
        self._trajectory_configuration = {
            "robot": deepcopy(robot_dict),
            "optimizer_configs": [trajectory_optimizer],
            "metrics_rollout": configuration("metrics_base.yml", "trajectory_metrics"),
            "transition_model": transition, "device_cfg": self.device_cfg,
            "num_seeds": budget.trajectory_seeds, "max_batch_size": 1, "max_goalset": 1,
            "use_cuda_graph": False, "random_seed": budget.random_seed,
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
                state, counts, updated = super().get_interpolated_trajectory(js_optimized)
                rollout = self.additional_metrics_rollouts["interpolated_rollout"]
                # Native interpolation has different valid lengths per seed.
                # Padded final states must not stretch the progress schedule.
                rollout.constraint_manager.set_dense_counts(counts)
                rollout.metrics_constraint_manager.set_dense_counts(counts)
                return state, counts, updated

        self._optimizer_class = ProcessTrajectoryOptimizer
        self._optimizer_config_class = TrajectoryOptimizerCfg
        self.trajopt = ProcessTrajectoryOptimizer(TrajectoryOptimizerCfg.create(**deepcopy(self._trajectory_configuration)))
        self._initialize_verification_constraints()

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

    def close(self):
        for solver in (self.ik, self.trajopt):
            if solver is not None:
                solver.destroy()

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
            result = self.ik.solve_pose(self._goal(progress), current_state=self.start,
                                       seed_config=previous, return_seeds=self.budget.ik_seeds)
            candidates = self.ik.get_active_js(result.js_solution).position
            if not bool(torch.isfinite(candidates).all()):
                raise ValueError("IK produced nonfinite initialization candidates")
            selected, distinct_count = _select_ik_candidates(
                candidates[0].detach().cpu().numpy(), result.success[0].detach().cpu().numpy(),
                previous[0].detach().cpu().numpy(), count, self.problem.numerical.joint)
            previous = candidates[:, selected].clone()
            sequence.append(previous)
            success_counts.append(int(result.success.sum().item()))
            self.reference_ik_distinct_counts.append(distinct_count)
        seeds = torch.stack(sequence, dim=2)
        self.initial_seed_strategies = ["reference_ik"]
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
        return seeds.contiguous(), success_counts
        # END bounded distinct IK chains and measured-state initialization.

    def replace_schedule(self, progress_map, refinement_parameters=()):
        """Rebuild only TrajOpt when its schedule or constraint samples change.

        Release old solver workspaces before constructing new rollouts. IK and
        immutable robot/scene configuration are retained; no second planner
        and no duplicate native roadmap is allocated.
        """
        parameters = np.unique(np.asarray(refinement_parameters, float))
        if (not np.isfinite(parameters).all() or np.any(parameters < 0) or np.any(parameters > 1)
                or len(parameters)+len(progress_map.anchor_parameters(self.anchors)) > self.budget.max_anchors):
            raise ValueError("Refinement samples exceed the finite exact-anchor budget")
        self.progress_map, self.refinement_parameters = progress_map, parameters
        manager_type = _process_manager_type(self.problem, self.scene, progress_map, self.anchors,
                                             self.budget, self.seed_progress, parameters)
        self._trajectory_configuration["cost_manager_config_instance_type"] = manager_type
        self.trajopt.destroy()
        self.trajopt = None
        self.trajopt = self._optimizer_class(
            self._optimizer_config_class.create(**deepcopy(self._trajectory_configuration)))
        self._initialize_verification_constraints()

    def _boundary_candidates(self, seed, candidate_trajectory, *, deadline):
        """Create a complete candidate for every shared boundary before search."""
        torch, count = self.torch, seed.shape[1]
        previous = self.start.position[:, None].expand(1, count, -1).clone()
        candidates = [previous]
        for progress in self.anchors[1:]:
            if time.monotonic() >= deadline:
                return None
            self.seed_progress.fill_(float(progress))
            result = self.ik.solve_pose(self._goal(progress), current_state=self.start,
                                       seed_config=previous, return_seeds=self.budget.ik_seeds)
            options = self.ik.get_active_js(result.js_solution).position
            if not bool(torch.isfinite(options).all()):
                raise ValueError("IK produced nonfinite boundary candidates")
            # BEGIN use the same distinct native-IK rule for shared boundaries.
            indices, _ = _select_ik_candidates(
                options[0].detach().cpu().numpy(), result.success[0].detach().cpu().numpy(),
                previous[0].detach().cpu().numpy(), count, self.problem.numerical.joint)
            # END use the same distinct native-IK rule for shared boundaries.
            previous = options[:, indices].clone()
            candidates.append(previous)
        candidates = torch.stack(candidates, dim=2)
        # Retain the optimizer's best complete proposal as the first branch.
        # In particular, a valid relocated support endpoint must not disappear
        # merely because the preceding path needs a different homotopy.
        parameters = self.progress_map.anchor_parameters(self.anchors)
        first_u = np.asarray([parameters[parameters[:, 1] == value, 0][0] for value in self.anchors])
        candidate_q = candidate_trajectory._curve(first_u*candidate_trajectory.duration)
        candidates[:, 0] = self.device_cfg.to_device(candidate_q[None])
        candidates[:, :, 0] = self.start.position[:, None]
        return candidates

    def ompl_seed(self, initial_seed, candidate_trajectory):
        """Search whole boundary branches, then return one full-window seed.

        Both free and constrained stages include progress in their state.
        Only a complete branch is returned; failure later in the process
        retries another whole branch rather than freezing early decisions.
        Final optimization always returns to the original unfrozen problem.
        """
        torch = self.torch
        deadline = time.monotonic()+self.budget.search_seconds
        report = {"branches": [], "success": False, "budget_seconds": self.budget.search_seconds}
        candidates = self._boundary_candidates(initial_seed, candidate_trajectory, deadline=deadline)
        if candidates is None or time.monotonic() >= deadline:
            report["failure"] = "whole_search_time_budget_exhausted_during_boundary_initialization"
            return None, None, report
        lower = self.trajopt.transition_model.action_bound_lows.detach().cpu().numpy().reshape(-1)
        upper = self.trajopt.transition_model.action_bound_highs.detach().cpu().numpy().reshape(-1)
        tolerance = min(self.problem.numerical.position_m, self.problem.numerical.rotation_rad,
                        self.problem.numerical.joint)*.1

        for branch in range(candidates.shape[1]):
            if time.monotonic() >= deadline:
                report["branches"].append([{"failure": "whole_search_time_budget_exhausted"}])
                break
            joint_proposal = candidates[:, branch]
            with torch.no_grad():
                proposal_state = self._verification_fk.compute_kinematics(
                    self.JointState.from_position(joint_proposal.contiguous(), joint_names=list(self.joint_names)))
                tools = proposal_state.tool_poses
                index = tools.tool_frames.index(self.problem.tool_link)
                p, r = self.problem.transform_tool_pose(tools.position[:, :, index], tools.quaternion[:, :, index])
                proposal_progress = self.device_cfg.to_device(self.anchors[None])
                fixed_problem = self.problem.with_fixed_endpoints(p, r, proposal_progress)
                fixed_scene = self.scene.with_fixed_endpoints(p, r, proposal_progress).clone_for_rollout()
            q = joint_proposal[0].detach().cpu().numpy()
            branch_report, paths = [], []

            def state_parts(values):
                joint_position = values[..., :-1].contiguous()
                zero = torch.zeros_like(joint_position)
                joint_state = self.JointState(position=joint_position, velocity=zero,
                                              acceleration=zero, jerk=zero,
                                              dt=self.device_cfg.to_device([self.budget.interpolation_dt]),
                                              joint_names=list(self.joint_names))
                state = self._verification_fk.compute_kinematics(joint_state)
                tools = state.tool_poses
                index = tools.tool_frames.index(self.problem.tool_link)
                quaternion = self._orientation_vjp.apply(joint_position, tools.quaternion.detach(),
                                                         state.tool_jacobians[..., 3:, :].detach())
                position = self._native_layout.apply(tools.position)
                p, r = fixed_problem.transform_tool_pose(position[:, :, index], quaternion[:, :, index])
                return joint_state, state, p, r

            def equality_for(phase_index):
                def residual_function(values_t):
                    js, _, p, r = state_parts(values_t.reshape(1, 1, -1))
                    progress = values_t[-1:].reshape(1, 1)
                    process = fixed_problem.project_residual(p, r, progress, phase_index,
                                                              js.position, self.joint_names)
                    # The scene contributes only independent commanded drive
                    # coordinates. Its preflight verifies that the original
                    # actuator coupling rows already occur in the problem.
                    drive = fixed_scene.project_residual(progress, js.position, self.joint_names)
                    return torch.cat((process, drive), -1).reshape(-1)

                def equality(values, derivative):
                    # Native Newton/geodesic loops call back here repeatedly.
                    # Stop them explicitly when the shared search budget is
                    # spent, instead of fabricating an equality violation.
                    if time.monotonic() >= deadline:
                        raise TimeoutError("The whole boundary-search budget was exhausted")
                    values_t = self.device_cfg.to_device(np.asarray(values))
                    if derivative:
                        with torch.enable_grad():
                            answer = torch.autograd.functional.jacobian(residual_function, values_t,
                                                                         create_graph=False, vectorize=False)
                    else:
                        with torch.no_grad():
                            answer = residual_function(values_t)
                    return answer.detach().cpu().numpy().astype(float)
                return equality

            # IK outputs satisfy its numerical target tolerance but need not
            # lie on OMPL's tighter equality manifold.  Refine proposals at
            # fixed s, then freeze the entire updated proposal together.
            # The actual current joint state at index zero is never projected.
            projection_failed = False
            for anchor_index in range(1, len(self.anchors)):
                progress = self.anchors[anchor_index]
                adjacent = [index for index in range(self.problem.phase_count)
                            if index-1e-7 <= progress <= index+1+1e-7]
                for phase_index in adjacent:
                    try:
                        value = _project_boundary(np.r_[q[anchor_index], progress],
                                                  equality_for(phase_index), tolerance, deadline=deadline)
                    except TimeoutError:
                        value = None
                    if value is None:
                        projection_failed = True
                        break
                    q[anchor_index] = value[:-1]
                if projection_failed:
                    break
            if projection_failed:
                report["branches"].append([{"failure": ("whole_search_time_budget_exhausted"
                                                        if time.monotonic() >= deadline
                                                        else "boundary_projection_failed")}])
                continue
            with torch.no_grad():
                _, _, p, r = state_parts(self.device_cfg.to_device(np.column_stack((q, self.anchors))[None]))
                fixed_problem = self.problem.with_fixed_endpoints(p, r, proposal_progress)
                fixed_scene = self.scene.with_fixed_endpoints(p, r, proposal_progress).clone_for_rollout()

            invalid_states = []

            def valid(values):
                with torch.no_grad():
                    values_t = self.device_cfg.to_device(values[None, None])
                    js, state, p, r = state_parts(values_t)
                    s = values_t[..., -1]
                    process = fixed_problem.evaluate(p, r, s, js.position, self.joint_names,
                                                     joint_velocity=js.velocity)
                    scene = fixed_scene.evaluate(fixed_problem.transform_robot_spheres(state.robot_spheres),
                                                 p, r, s, js.position, self.joint_names)
                    native = self._verification_native.compute_costs(
                        self._RobotState(joint_state=js, cuda_robot_model_state=state))
                    terms = {"process": process, "scene": scene,
                             **{f"native:{name}": value for name, value in zip(native.names, native.values)}}
                    accepted = all(bool(torch.isfinite(term).all()) and bool((term <= 0).all())
                                   for term in terms.values())
                    if not accepted and len(invalid_states) < 3:
                        # Retain actual rejection evidence, rather than
                        # attributing a spent callback budget to projection.
                        invalid_states.append({"progress": float(values[-1]),
                                               "terms": {name: float(term.max().item())
                                                         for name, term in terms.items()}})
                    return accepted

            for interval, (start_s, end_s) in enumerate(zip(self.anchors[:-1], self.anchors[1:])):
                search_remaining = deadline-time.monotonic()
                if search_remaining <= 0:
                    branch_report.append({"failure": "whole_search_time_budget_exhausted"})
                    break
                phase_index = min(self.problem.phase_count-1, int(np.floor((start_s+end_s)/2)))
                start = np.concatenate((q[interval], [start_s]))
                goal = np.concatenate((q[interval+1], [end_s]))
                invalid_states.clear()
                try:
                    path, stage_report = _ompl_stage(
                        start, goal, np.concatenate((lower, [start_s])), np.concatenate((upper, [end_s])),
                        valid, equality_for(phase_index), tolerance=tolerance, step=self.budget.validation_joint_step,
                        seconds=search_remaining, sample_limit=self.budget.max_validation_points)
                except TimeoutError:
                    # Equality rank checks precede native SI construction;
                    # they still consume the same whole-search deadline.
                    path, stage_report = None, {"failure": "whole_search_time_budget_exhausted"}
                if invalid_states:
                    stage_report["first_invalid_states"] = deepcopy(invalid_states)
                branch_report.append({"progress": [float(start_s), float(end_s)], **stage_report})
                if path is None:
                    break
                paths.append(path)
            report["branches"].append(branch_report)
            if len(paths) != len(self.anchors)-1:
                continue
            path = np.concatenate((paths[0], *(item[1:] for item in paths[1:])))
            if len(path) > self.budget.max_validation_points:
                report["branches"][-1].append({"failure": "whole_path_sample_budget_exhausted"})
                continue
            # Remove only exact duplicates. A short but distinct final edge
            # still owns the caller's true goal and must never be discarded.
            keep = np.r_[True, np.any(path[1:] != path[:-1], axis=1)]
            path = path[keep]
            u = np.r_[0., np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=-1))]
            u /= u[-1]
            try:
                progress_map, parameter_warp = _progress_for_cubic_boundaries(
                    _ProgressMap(u, path[:, -1]), self.budget.knots, self.problem.start_dq,
                    stop_progress=self.problem.stop_progress, anchors=self.anchors)
                exact_anchor_count = len(progress_map.anchor_parameters(self.anchors))
            except ValueError as error:
                report["branches"][-1].append({"failure": "progress_schedule_rejected", "reason": str(error)})
                continue
            if exact_anchor_count > self.budget.max_anchors:
                report["branches"][-1].append({"failure": "progress_schedule_exceeds_anchor_budget"})
                continue
            # These are control initializers only.  Native whole-window
            # optimization subsequently changes every internal boundary q.
            control_u = (np.arange(self.trajopt.action_horizon)+3.)/(self.trajopt.action_horizon+4.)
            # Skill progress and q-path controls use the identical old-path
            # parameter. An old progress pause may still contain a change in
            # q; keeping its original vertices preserves that search seed.
            source_u = np.interp(control_u, parameter_warp.u, parameter_warp.s)
            controls = np.stack([np.interp(source_u, u, path[:, column])
                                 for column in range(len(self.joint_names))], axis=-1)
            seed = self.device_cfg.to_device(controls[None, None])
            report.update(success=True, samples=len(path), selected_branch=branch)
            return seed, progress_map, report
        return None, None, report

    def optimize(self, seed):
        # This is the public supported V2 control: all pose axis factors are
        # zero, while goalset metadata remains valid for native seed ranking.
        self.trajopt.disable_tool_pose_tracking()
        return self.trajopt.solve_pose(
            goal_tool_poses=self._goal(self.problem.phase_count), current_state=self.start,
            # Every budgeted seed must survive native ranking. Its failed
            # seeds share a float32 1e16 penalty, which loses their different
            # process violations before this planner can select a continuation.
            seed_traj=seed, num_seeds=seed.shape[1], return_seeds=seed.shape[1],
            # The seed uses the same fixed request clock as both native dt
            # bounds. No post-solve time compression may amplify a stop's
            # numerical velocity band or alter measured boundary derivatives.
            dt=self.torch.full((1, seed.shape[1]), self.budget.maximum_trajectory_dt,
                               dtype=seed.dtype, device=seed.device),
            use_implicit_goal=False, goal_state=None,
            finetune_attempts=0, finetune_dt_scale=1.)

    def native_diagnostics(self, result, seed_index=None):
        """Expose the native reason for rejection, including dense metrics.

        Pose tracking is deliberately disabled; its native error fields do
        not measure process accuracy.  Actual skill errors are evaluated
        independently on the exported curve in :meth:`export`.
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
            key = (0, quality, index) if accepted else (1, continuation_violation, index)
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
        """Select uncovered failing curve samples within the exact-anchor cap."""
        violations = self.last_validation["violations"]
        if not np.isfinite(violations).all():
            return None
        fixed = self.progress_map.anchor_parameters(self.anchors)[:, 0]
        existing = np.r_[fixed, self.refinement_parameters,
                         np.linspace(0, 1, self.trajopt.horizon)]
        remaining = self.budget.max_anchors-len(fixed)-len(self.refinement_parameters)
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
        anchor_pairs = self.progress_map.anchor_parameters(self.anchors)
        event_records = []
        for event in self.problem.events:
            event_u = self.progress_map.anchor_parameters([event.progress])[:, 0]
            event_records.append({**asdict(event), "time": float(event_u[0]*times[-1]),
                                  "hold_until": float(event_u[-1]*times[-1])})
        trajectory = JointTrajectory(self.joint_names, times, position, velocity,
                                     self.progress_map.u.copy(), self.progress_map.s.copy(),
                                     tuple(event_records), deepcopy(self.problem.conditions), {})
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
            components["scene"] = self._verification_scene.evaluate(
                self.problem.transform_robot_spheres(state.robot_spheres), p, r, s_t, q_t, self.joint_names)
            native_constraints = self._verification_native.compute_costs(
                self._RobotState(joint_state=joint_state, cuda_robot_model_state=state))
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
        residuals["initial_position"] = max(0., float(np.max(np.abs(position[0]-self.start.position[0].detach().cpu().numpy())))
                                              - self.problem.numerical.joint)
        residuals["initial_velocity"] = max(0., float(np.max(np.abs(velocity[0]-self.start.velocity[0].detach().cpu().numpy())))
                                              - self.problem.numerical.velocity)
        trajectory.residuals = residuals
        return trajectory, len(queries)


def plan_trajectory(problem, scene, robot, *, budget=None, device="cuda:0"):
    """Generate and verify a complete-window robot trajectory with cuRobo V2.

    ``robot`` is an existing V2 robot YAML path or configuration dictionary.
    Scene geometry, calibration, measured joints and skill tolerances are
    explicit inputs; a robot name never supplies missing perception evidence.
    GPU work starts only when this function is called.
    """
    budget = PlanningBudget() if budget is None else budget
    started = time.monotonic()
    planner = _CuroboPlanner(problem, scene, robot, budget, device)
    diagnostics = {"backend": "curobo_v2", "batch_size": 1, "ik_seeds": budget.ik_seeds,
                   "trajectory_seeds": budget.trajectory_seeds, "used_ompl": False,
                   "timing_policy": "fixed_request_dt", "native_trajectory_dt": budget.maximum_trajectory_dt,
                   "progress_parameter_error_bound": _PROGRESS_PARAMETER_ERROR,
                   "progress_clock_max_keys": _MAX_PROGRESS_KEYS,
                   "scene": scene.describe(), "scene_workspace_bytes": planner.scene_workspace_bytes,
                   "attempts": []}
    try:
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
            result, trajectory, attempt = planner.select_candidate(planner.optimize(seeds))
            native_success = attempt["native_success"]
            diagnostics["attempts"].append(attempt)
            diagnostics.update(attempt)
            if attempt["accepted"]:
                diagnostics["executable_until"] = trajectory.executable_until
                diagnostics["next_verification"] = trajectory.next_verification
                return PlanningResult(True, trajectory, diagnostics)
            if iteration+1 >= budget.maximum_trajectory_solves:
                break

            additional = planner.refinement_samples()
            # A native success invalidated between its rollout samples calls
            # for exact witness constraints on this same cubic first.
            if native_success and additional is not None:
                seeds = result.solution.detach().clone()
                planner.replace_schedule(planner.progress_map, additional)
                attempt["next_action"] = "refine_uncovered_curve_samples"
                attempt["refinement_parameters"] = additional.tolist()
                continue

            if not searched and budget.search_seconds > 0:
                searched = True
                path_seed, progress_map, search_report = planner.ompl_seed(reference_seeds, trajectory)
                diagnostics["ompl"] = search_report
                if path_seed is not None:
                    planner.replace_schedule(progress_map)
                    seeds = path_seed
                    diagnostics["used_ompl"] = True
                    attempt["next_action"] = "optimize_complete_ompl_seed"
                    continue

            if additional is not None:
                seeds = result.solution.detach().clone()
                planner.replace_schedule(planner.progress_map, additional)
                attempt["next_action"] = "refine_uncovered_curve_samples"
                attempt["refinement_parameters"] = additional.tolist()
                continue
            break

        diagnostics["failure"] = ("exported_curve_constraint_violation" if diagnostics["native_success"]
                                  else "complete_process_optimization_not_feasible")
        return PlanningResult(False, None, diagnostics)
    finally:
        diagnostics["elapsed_seconds"] = time.monotonic()-started
        planner.close()


# END complete-window robot planning and generation-time process constraints.
