"""Confirm online sphere candidates against native robot solids on the CPU.

The frozen online field still performs broad-phase queries and owns every
unknown-space, numerical-support and finite-contact constraint. This module
only decides whether a positive robot/target sphere term also intersects the
registered native robot solid. It prepares occupied-cell indices once, then
queries only the native geometry/target pairs named by positive spheres.

An ESDF target is represented by the same complete occupied cells used when
preparing finite contact fields. A primitive target retains its analytic Coal
shape outside finite contact mode. This is a discrete observed-geometry check,
not a continuous-trajectory, sensor-error or unobserved-space certificate.
"""

# BEGIN bounded native confirmation owned only by planning acceptance.
from itertools import product
import math
from time import perf_counter

import coal
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .online_geometry import FrozenESDF, FrozenPrimitives, OnlinePlanningScene
from .resource_admission import DEFAULT_MAX_HOST_RSS_BYTES, require_host_memory
from .robot_geometry_check import RobotGeometryChecker, _rigid_pose, _solid_axis_bounds


_CHUNK = 65_536


def _pose7(row):
    """Convert the frozen primitive's already-validated xyz/wxyz pose."""
    value = np.asarray(row, dtype=float)
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_quat(value[[4, 5, 6, 3]]).as_matrix()
    pose[:3, 3] = value[:3]
    return pose


def _primitive(row):
    """Keep exact primitive dimensions; capsules/cylinders store half height."""
    kind, dimensions = row['kind'], np.asarray(row['dimensions'], dtype=float)
    vertices = None
    if kind in {'box', 'cuboid'}:
        lengths = dimensions * 2 if kind == 'box' else dimensions
        shape = coal.Box(*lengths)
        vertices = np.asarray(list(product((-.5, .5), repeat=3))) * lengths
    elif kind == 'sphere':
        shape = coal.Sphere(dimensions[0])
    elif kind == 'cylinder':
        shape = coal.Cylinder(dimensions[0], dimensions[1] * 2)
    elif kind == 'capsule':
        shape = coal.Capsule(dimensions[0], dimensions[1] * 2)
    else:
        raise ValueError('Unsupported frozen primitive in native confirmation')
    return shape, vertices, _pose7(row['pose'])


def _bounds(vertices, shape, pose):
    """Reuse complete native support bounds, adding the capsule end spheres."""
    if isinstance(shape, coal.Capsule):
        extent = shape.radius + shape.halfLength * np.abs(pose[:3, 2])
        return pose[:3, 3] - extent, pose[:3, 3] + extent
    return _solid_axis_bounds(vertices, shape, pose)


class _OccupiedCells:
    """One immutable spatial index and rule-membership bitset per target.

    Cell deletion uses the exact whole-cell containment test from
    PreparedContactFields.prepare. The index remains the full owner; active
    rules filter nearby cells after the spatial query, never a whole owner.
    """

    def __init__(self, query, rules):
        if isinstance(query, FrozenPrimitives):
            prepared = query._contact_model
            if prepared is None:
                raise ValueError('Primitive finite contact requires its prepared field')
            origin, h, grid_shape = prepared.origin, prepared.voxel_size, prepared.shape
            count = math.prod(grid_shape)
            require_host_memory(count * 8 + _CHUNK * 128, pending_existing_bytes=0,
                                max_rss_bytes=DEFAULT_MAX_HOST_RSS_BYTES)
            # Match FrozenPrimitives.prepare_contacts exactly. No simulator
            # geometry or separately chosen contact resolution is admitted.
            import torch
            occupied = np.empty(count, dtype=bool)
            with torch.no_grad():
                for start in range(0, count, _CHUNK):
                    flat = np.arange(start, min(count, start + _CHUNK))
                    indices = np.column_stack(np.unravel_index(flat, grid_shape))
                    points = origin + indices * h
                    distance = query.signed_distance(torch.as_tensor(points, dtype=torch.float64)).numpy()
                    occupied[start:start + len(flat)] = distance <= math.sqrt(3.) * h / 2
            occupied = occupied.reshape(grid_shape)
        else:
            origin, h = query.origin, query.voxel_size
            # These are the source-valid/sign-supported occupied cells of the
            # SAME frozen field, including negative observed interior samples.
            count = query.distance.size
            require_host_memory(count * 8, pending_existing_bytes=0,
                                max_rss_bytes=DEFAULT_MAX_HOST_RSS_BYTES)
            occupied = query.site_valid & ((np.abs(query.distance) <= h * .51)
                                           | ((query.distance < 0) & query.observed_sign))
        # Admission now uses the ACTUAL occupied count. The allowance covers
        # simultaneous integer argwhere coordinates, floating arithmetic and
        # retained centres, cKDTree indices/nodes, rule bits and chunk scratch;
        # sparse recorded scenes need not reserve a dense Nx3 world inventory.
        occupied_count = int(np.count_nonzero(occupied))
        require_host_memory(occupied_count * 128 + _CHUNK * 96,
                            pending_existing_bytes=0, max_rss_bytes=DEFAULT_MAX_HOST_RSS_BYTES)
        self.centers = np.ascontiguousarray(origin + np.argwhere(occupied) * h)
        self.centers.flags.writeable = False
        self.h = float(h)
        self.shape = coal.Box(h, h, h)
        self.tree = cKDTree(self.centers, copy_data=False) if len(self.centers) else None
        self.rule_bits = {rule.rule_id: 1 << index for index, rule in enumerate(rules)}
        if len(self.rule_bits) > 16:
            raise ValueError('Finite contact rule bitset exceeds the prepared online field limit')
        self.removed_by_rule = np.zeros(len(self.centers), dtype=np.uint16)
        for rule in rules:
            axes = np.asarray(rule.patch_axes)
            half_support = np.abs(axes).sum(axis=0) * h / 2
            widths = np.r_[rule.patch_half_size, rule.modeling_allowance_m]
            for start in range(0, len(self.centers), _CHUNK):
                centers = self.centers[start:start + _CHUNK]
                local = (centers - rule.patch_origin) @ axes
                contained = (np.abs(local) + half_support
                             <= widths + rule.numerical_position_tolerance_m).all(-1)
                self.removed_by_rule[start:start + len(centers)] |= (
                    contained.astype(np.uint16) * self.rule_bits[rule.rule_id])
        self.removed_by_rule.flags.writeable = False

    def nearby(self, bounds, active_rules, boundary):
        """Conservatively select intersecting cubes with a Chebyshev KD query."""
        if self.tree is None:
            return np.empty(0, dtype=int), 0
        lower, upper = bounds
        half = (upper - lower) / 2 + self.h / 2 + boundary
        middle = (lower + upper) / 2
        ids = np.asarray(self.tree.query_ball_point(middle, float(half.max()), p=np.inf), dtype=int)
        if len(ids):
            ids = ids[(np.abs(self.centers[ids] - middle) <= half).all(-1)]
        found = len(ids)
        bits = sum(self.rule_bits[name] for name in active_rules)
        if bits and len(ids):
            ids = ids[(self.removed_by_rule[ids] & bits) == 0]
        # Stable source order avoids KD tree traversal history affecting the
        # retained first collision witness or regression diagnostics.
        return np.sort(ids), found - len(ids)


class NativeCollisionConfirmation:
    """Request-owned native confirmation of positive robot sphere terms.

    ``checker`` owns source-verified robot shapes. ``scene`` owns the exact
    frozen fields and native geometry/sphere/material identities. Construct
    once per scene, then call ``confirm`` with the same state's existing FK,
    positive sphere indices, owner-pose snapshot, and EFFECTIVE policy IDs.
    Raw executor latch IDs are insufficient: interval, finger uncertainty and
    arrival activation must already be resolved by OnlinePlanningScene.

    A cleared target only authorizes clearing its robot collision component.
    It cannot clear contact conditions, scene-update obligations, unknown
    space, self collision, or collisions of moving non-robot actors.
    """

    def __init__(self, checker, scene):
        if not isinstance(checker, RobotGeometryChecker) or not isinstance(scene, OnlinePlanningScene):
            raise TypeError('Native confirmation needs the bound robot checker and online scene')
        start = perf_counter()
        self.checker, self.scene = checker, scene
        self.boundary = max(checker.numerics['gjk_tolerance'], checker.numerics['epa_tolerance'])
        self.sphere_geometry = {}
        geometry_spheres = {}
        for group in scene.robot_collision_bodies:
            names = tuple(group['geometry_names'])
            if not names or any(name not in checker._geometries for name in names):
                raise ValueError('Native confirmation requires every body source geometry')
            for index in group['sphere_indices']:
                if index in self.sphere_geometry:
                    raise ValueError('A robot sphere must belong to one native collision body')
                self.sphere_geometry[index] = names
            for name in names:
                geometry_spheres.setdefault(name, set()).update(group['sphere_indices'])
        if set(self.sphere_geometry) != set(range(len(scene.robot_sphere_materials))):
            raise ValueError('Native collision bodies must cover the complete robot sphere layout')
        self.rules = {rule.rule_id: rule for rule in scene.contacts if rule.actor == 'robot'}
        self.geometry_rules = {}
        for rule in self.rules.values():
            name = rule.online_material['native_geometry']
            # A partial set of a native body's spheres must not grant its
            # whole physical solid a permission intended for another subset.
            if (name not in geometry_spheres or set(rule.sphere_indices) != geometry_spheres[name]
                    or any(self.sphere_geometry[index] != (name,) for index in rule.sphere_indices)
                    or scene.native_geometry_materials.get(name) != rule.material_id):
                raise ValueError('Finite native permission needs exact complete material/sphere ownership')
            self.geometry_rules.setdefault((name, rule.target_owner), set()).add(rule.rule_id)
        self.targets = {}
        queries = [('background', scene.background), *((owner.owner_id, owner.query) for owner in scene.owners)]
        for name, query in queries:
            rules = tuple(rule for rule in self.rules.values() if rule.target_owner == name)
            primitives = tuple(_primitive(row) for row in query.primitives) if isinstance(query, FrozenPrimitives) else ()
            cells = _OccupiedCells(query, rules) if not primitives or rules else None
            self.targets[name] = {'query': query, 'primitives': primitives, 'cells': cells}
        self.preparation = {'elapsed_seconds': perf_counter() - start,
            'occupied_cells': {name: 0 if row['cells'] is None else len(row['cells'].centers)
                               for name, row in self.targets.items()},
            'scope': 'One frozen target index per request; no current simulator object truth'}

    def confirm(self, *, robot_state, candidates, owner_poses, effective_contact_rule_ids,
                check_request=None):
        """Confirm only supplied positive sphere/native-geometry target pairs.

        ``candidates`` maps target owner IDs (or ``background``) to positive
        robot sphere indices. ``owner_poses`` maps the SAME queried owners to
        world-from-owner transforms. Background already uses world axes.
        ``robot_state`` is RobotGeometryChecker.query output; no new FK runs.
        Unsupported/numerical-boundary results remain unresolved and cannot
        erase the original candidate. Finite collisions stop their pair early.
        ``check_request`` may reuse the caller's existing deadline/cancellation
        check. Its exception aborts this call without publishing partial clears.
        """
        start = perf_counter()
        if check_request is not None and not callable(check_request):
            raise TypeError('A supplied request guard must be callable')
        active = frozenset(effective_contact_rule_ids)
        if not active <= self.rules.keys():
            raise ValueError('Effective contact policies must identify registered robot rules')
        pairs, stats = [], {'native_pairs': 0, 'coal_queries': 0, 'nearby_cells': 0, 'permitted_cells': 0}
        for target_name, indices in candidates.items():
            if target_name not in self.targets:
                raise ValueError('Candidate target is absent from this frozen request')
            indices = tuple(sorted(set(indices)))
            if any(type(index) not in (int, np.int64, np.int32) or index not in self.sphere_geometry for index in indices):
                raise ValueError('Candidate must name a registered robot sphere index')
            if not indices:
                continue
            pose = np.eye(4) if target_name == 'background' else _rigid_pose(owner_poses[target_name], 'world from current owner')
            owner_world = np.linalg.inv(pose)
            geometries = sorted({name for index in indices for name in self.sphere_geometry[index]})
            target = self.targets[target_name]
            for name in geometries:
                if check_request is not None:
                    check_request()
                stats['native_pairs'] += 1
                actor_pose = owner_world @ _rigid_pose(robot_state['geometry_poses'][name], 'world from native robot solid')
                _, shape, _, _ = self.checker._geometries[name]
                vertices = self.checker._solid_vertices[name]
                bounds = _bounds(vertices, shape, actor_pose)
                rules = sorted(active & self.geometry_rules.get((name, target_name), set()))
                record = {'target_owner': target_name, 'actor_geometry': name,
                    'sphere_indices': [int(index) for index in indices if name in self.sphere_geometry[index]],
                    'effective_contact_rule_ids': rules, 'status': 'clear', 'coal_queries': 0,
                    'minimum_queried_distance_m': None}
                pairs.append(record)
                if bounds is None:
                    record.update(status='unresolved', reason='native_solid_support_bounds_unavailable')
                    continue
                actor_transform = coal.Transform3s(actor_pose[:3, :3], actor_pose[:3, 3])
                if target['primitives'] and not rules:
                    obstacles = []
                    for index, (other, other_vertices, other_pose) in enumerate(target['primitives']):
                        other_bounds = _bounds(other_vertices, other, other_pose)
                        if (other_bounds is None or not (np.any(bounds[1] + self.boundary < other_bounds[0])
                                                       or np.any(other_bounds[1] + self.boundary < bounds[0]))):
                            obstacles.append((other, other_pose, {'primitive_index': index}))
                    record['target_representation'] = 'analytic_frozen_primitive'
                else:
                    cells = target['cells']
                    ids, removed = cells.nearby(bounds, rules, self.boundary)
                    stats['nearby_cells'] += len(ids) + removed
                    stats['permitted_cells'] += removed
                    record.update(target_representation='frozen_observed_occupied_cells',
                                  nearby_cell_count=len(ids) + removed, permitted_cell_count=removed)
                    # The list is local to this native AABB, not a world-sized
                    # per-state geometry inventory; one Coal box is reused.
                    obstacles = ((cells.shape, center, {'cell_index': int(index)})
                                 for index, center in zip(ids, cells.centers[ids]))
                for other, other_pose, identity in obstacles:
                    if check_request is not None and record['coal_queries'] % 64 == 0:
                        check_request()
                    if other_pose.shape == (3,):
                        other_transform = coal.Transform3s(np.eye(3), other_pose)
                    else:
                        other_transform = coal.Transform3s(other_pose[:3, :3], other_pose[:3, 3])
                    # Fresh requests/results avoid carrying simplex or witness
                    # state between unrelated source cells and robot solids.
                    request, distance = coal.DistanceRequest(), coal.DistanceResult()
                    request.enable_signed_distance = True
                    coal.distance(shape, actor_transform, other, other_transform, request, distance)
                    value = float(distance.min_distance)
                    stats['coal_queries'] += 1
                    record['coal_queries'] += 1
                    if not np.isfinite(value):
                        record.update(status='unresolved', reason='nonfinite_native_distance', **identity)
                        break
                    if record['minimum_queried_distance_m'] is None or value < record['minimum_queried_distance_m']:
                        record['minimum_queried_distance_m'] = value
                    if value <= self.boundary:
                        record.update(status='collision' if value < -self.boundary else 'unresolved',
                            reason='native_solid_intersection' if value < -self.boundary else 'coal_numerical_boundary',
                            witness_native=np.asarray(distance.getNearestPoint1()).tolist(),
                            witness_target=np.asarray(distance.getNearestPoint2()).tolist(),
                            witness_frame=target_name, **identity)
                        break
        statuses = {name: [row['status'] for row in pairs if row['target_owner'] == name] for name in candidates}
        stats['elapsed_seconds'] = perf_counter() - start
        return {'pairs': pairs,
            'cleared_targets': [name for name, values in statuses.items() if values and all(value == 'clear' for value in values)],
            'collision_targets': [name for name, values in statuses.items() if 'collision' in values],
            'unresolved_targets': [name for name, values in statuses.items() if 'unresolved' in values],
            'statistics': stats,
            'scope': 'Native source solids against the same frozen occupied cells/analytic primitives at one state; effective finite-cell permissions only; other scene constraints remain owned by the caller'}
# END candidates retain their original residual and every non-collision obligation.


# BEGIN shared native confirmation of robot/world sphere candidates.
class OnlineCollisionConfirmation:
    """Confirm only positive sphere candidates; preserve every other constraint.

    The differentiable rollout never calls this object. Final candidate
    metrics use this adapter and the frozen observed cells. Physical execution
    never constructs this planning-only adapter. Native FK is performed only for states needing confirmation.
    """

    def __init__(self, validator, scene, check_request=None):
        self.validator, self.scene, self.engine = validator, scene, None
        self.check_request = check_request
        self.last_report = {}

    def __call__(self, costs, evidence):
        # All candidates were masked for unresolved hand geometry and native
        # exclusions before reaching this boundary. Non-finite costs remain
        # failures; a finite positive candidate is the only confirmable input.
        candidates = {}
        for name, value in evidence['targets'].items():
            if not bool((value > 0).any().item()):
                continue
            raw = value.detach().cpu().numpy()
            # BEGIN incomplete/non-finite broad-phase evidence cannot become clear.
            finite = np.isfinite(raw).all(axis=-1) & np.isfinite(costs[name].detach().cpu().numpy())
            retained = np.where(finite[..., None], raw, 0.)
            if np.any(retained > 0):
                candidates[name] = retained
            # END a NaN beside a positive sphere keeps the original failed term.
        self.last_report = {'scope': 'native robot solids against the same frozen observed geometry',
                            'candidate_states': 0, 'states': []}
        if not candidates:
            return costs
        if self.engine is None:
            self.engine = NativeCollisionConfirmation(self.validator.checker, self.scene)
        if evidence['joint_position'] is None or not evidence['joint_names']:
            raise ValueError('Native confirmation requires this query\'s complete named robot coordinates')
        joints = evidence['joint_position'].detach().cpu().numpy()
        fingers = (None if evidence['finger_positions'] is None else
                   evidence['finger_positions'].detach().cpu().numpy())
        poses = {name: (p.detach().cpu().numpy(), q.detach().cpu().numpy())
                 for name, (p, q) in evidence['owner_poses'].items()}
        effective = {name: value.detach().cpu().numpy()
                     for name, value in evidence['effective_rules'].items()}
        locations = np.argwhere(np.logical_or.reduce([np.any(value > 0, axis=-1)
                                                       for value in candidates.values()]))
        result = dict(costs)
        for batch, index in locations:
            named = dict(zip(evidence['joint_names'], joints[batch, index]))
            if self.scene.gripper is not None:
                named.update(zip(self.scene.gripper.joint_names, fingers[batch, index]))
            if set(named) != set(self.validator.joint_names):
                raise ValueError('Native confirmation cannot fill absent joints with a command or default')
            robot = self.validator.checker.query(joint_names=self.validator.joint_names,
                joint_positions=[named[name] for name in self.validator.joint_names],
                T_reference_base=self.validator.T_reference_base, pairs=(), frames=[])
            owner_poses = {}
            for owner, (p, q) in poses.items():
                pose = np.eye(4)
                pose[:3, :3] = Rotation.from_quat(q[batch, index][[1, 2, 3, 0]]).as_matrix()
                pose[:3, 3] = p[batch, index]
                owner_poses[owner] = pose
            by_owner = {name.removeprefix('collision/robot/'): np.flatnonzero(value[batch, index] > 0).tolist()
                        for name, value in candidates.items() if np.any(value[batch, index] > 0)}
            record = self.engine.confirm(robot_state=robot, candidates=by_owner,
                owner_poses=owner_poses,
                effective_contact_rule_ids=frozenset(name for name, active in effective.items()
                                                     if bool(active[batch, index])),
                check_request=self.check_request)
            record['query_index'] = [int(batch), int(index)]
            record['sphere_residuals'] = {name: float(costs[name][batch, index].item())
                                         for name in candidates if name.removeprefix('collision/robot/') in by_owner}
            self.last_report['states'].append(record)
            for owner in record['cleared_targets']:
                name = 'collision/robot/'+owner
                if owner not in by_owner or name not in result:
                    raise ValueError('Native confirmation returned an unqueried collision target')
                if result[name] is costs[name]:
                    result[name] = costs[name].clone()
                result[name][batch, index] = 0.
        self.last_report['candidate_states'] = len(self.last_report['states'])
        return result
# END no contact, coverage, unknown, controller or self-collision term is relaxed.
