"""Batch original occupied-point queries by target geometry.

Official fixed robot volumes retain their registered convex or analytic
shape. All other entities retain their RGB-D TSDF and observed interpolation
support. The arrays here are numerical query coordinates, not proxy bodies.
Fixed robot and moving-owner coordinates query the resident target fields.
Static environmental material does not become a per-iterate query source.
"""

# BEGIN concentrated geometry arrays replace per-row material proof objects.
from copy import copy
from time import perf_counter

import coal
import numpy as np
from scipy.spatial import ConvexHull
import warp as wp
import torch
from .gpu_field import DeviceField, submit_pair_maxima

from curobo._src.geom.sphere_fit.wp_mesh_query import _sdf_forward_kernel
from manipulation.implicit_distance import CORNERS, ObservedField
from manipulation.sampled_geometry_objective import MaterialSamples


class PointInventory:
    """Own unique original occupied coordinates for a fixed query source.

    Removing duplicate coordinates preserves a maximum over the original
    finite set. A declared original-cell exclusion constructs its own subset
    before deduplication; adjacent retained cells keep their shared points.
    Zero-only cells with no occupied volume do not supply collision material.
    """

    def __init__(self, samples, excluded=(), *, native_points=None):
        self.geometry = samples.geometry
        if isinstance(self.geometry, ObservedField):
            selected = ~self.geometry._boundary_only_cells[samples.cells]
            if excluded:
                selected &= ~np.isin(samples.cells, np.asarray(tuple(excluded), dtype=np.int64))
            points = np.unique(samples.points[selected], axis=0)
        else:
            points = np.vstack((samples.points, native_points))
            if isinstance(self.geometry, coal.Convex):
                points = np.vstack((points, np.asarray(self.geometry.points())))
        # BEGIN a native lattice is already unique. Its few original vertices
        # may coincide with lattice points; repeated coordinates preserve the
        # same finite maximum and need no full-array sorting at preparation.
        self.points = points
        # END observed cell coordinates are deduplicated before this assignment.
        self.bounds = samples.negative_bounds
        self.scale = samples.length_scale


# BEGIN native volume queries use the observation's spatial sampling resolution.
def native_volume_points(geometry, spacing, *, device='cpu'):
    """Sample a registered robot solid on a fixed local Cartesian lattice.

    The spacing is the smallest observed-field voxel size of this request.
    Interior samples detect an observed occupied patch inside a robot volume,
    where surface vertices alone provide no query. These are numerical points
    in the same official solid; they do not define a replacement collision body.
    """
    geometry.computeLocalAABB()
    lower, upper = np.asarray(geometry.aabb_local.min_), np.asarray(geometry.aabb_local.max_)
    axes = [np.linspace(a, b, int(np.ceil((b-a)/spacing))+1) for a, b in zip(lower, upper)]
    points = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, 3)
    if isinstance(geometry, coal.Box):
        return points
    if isinstance(geometry, coal.Cylinder):
        return points[np.sum(points[:, :2]**2, axis=1) <= geometry.radius**2]
    # BEGIN membership in the registered convex hull is a fixed array query.
    # Double precision retains the original half-space inequalities; bounded
    # chunks keep preparation storage independent of the full lattice size.
    planes = torch.as_tensor(ConvexHull(np.asarray(geometry.points())).equations,
                             dtype=torch.float64, device=device)
    # BEGIN bounded half-space tiles share one final membership transfer.
    accepted = torch.empty(len(points), dtype=torch.bool, device=device)
    for first in range(0, len(points), 8192):
        batch = torch.as_tensor(points[first:first+8192], dtype=torch.float64).to(device, non_blocking=True)
        accepted[first:first+len(batch)] = ((batch @ planes[:, :3].T+planes[:, 3]) <= 0.).all(dim=1)
    accepted = accepted.cpu().numpy()
    # END the original point order and inequalities are unchanged.
    # END the accepted coordinates remain fixed in the original local frame.
    return points[accepted]
# END the finite lattice is retained across all trajectory iterates.



class NativePointQuery:
    """Query the registered official robot volume without task geometry files.

    A registered convex body's own vertices define the same closed boundary
    passed to Warp. Its BVH is reusable across tasks. Registered analytic box
    and cylinder volumes use their exact signed-distance formulas, retaining
    curved cylinder sides instead of a tessellated substitute.
    """

    def __init__(self, geometry, *, device):
        self.geometry, self.device = geometry, device
        # BEGIN target bounds belong to the registered solid, independently
        # of whether the solid supplies moving query points (fixed mounts do not).
        geometry.computeLocalAABB()
        self.bounds = np.stack((np.asarray(geometry.aabb_local.min_),
                                np.asarray(geometry.aabb_local.max_)))
        # END target registration and source sampling have distinct ownership.
        self.mesh = None
        if isinstance(geometry, coal.Convex):
            points = np.asarray(geometry.points())
            hull = ConvexHull(points)
            faces = hull.simplices.copy()
            normals = np.cross(points[faces[:, 1]]-points[faces[:, 0]],
                               points[faces[:, 2]]-points[faces[:, 0]])
            flip = np.einsum('ij,ij->i', normals, hull.equations[:, :3]) < 0
            faces[flip] = faces[flip][:, [0, 2, 1]]
            self.mesh = wp.Mesh(points=wp.array(points.astype(np.float32), dtype=wp.vec3, device=device),
                indices=wp.array(faces.astype(np.int32).ravel(), dtype=wp.int32, device=device))
            self.search_distance = float(2*np.linalg.norm(np.ptp(points, axis=0))+1.)
        elif not isinstance(geometry, (coal.Box, coal.Cylinder)):
            raise TypeError('Registered robot geometry requires convex, box or cylinder volume')

    def sample(self, points):
        """Return signed distance and its local-coordinate first derivative."""
        if self.mesh is not None:
            query = wp.array(np.asarray(points, np.float32), dtype=wp.vec3, device=self.device)
            distances = wp.empty(len(points), dtype=wp.float32, device=self.device)
            gradients = wp.empty(len(points), dtype=wp.vec3, device=self.device)
            wp.launch(_sdf_forward_kernel, dim=len(points), inputs=[self.mesh.id, query,
                distances, gradients, self.search_distance], device=self.device)
            return distances.numpy().astype(float), gradients.numpy().astype(float)
        if isinstance(self.geometry, coal.Box):
            offset = np.abs(points)-np.asarray(self.geometry.halfSide)
            outside = np.maximum(offset, 0.)
            length = np.linalg.norm(outside, axis=1)
            axis = offset.argmax(1)
            distance = length+np.minimum(offset[np.arange(len(points)), axis], 0.)
            gradient = np.zeros_like(points)
            exterior = length > 0.
            gradient[exterior] = np.sign(points[exterior])*outside[exterior]/length[exterior, None]
            inside_rows = np.flatnonzero(~exterior)
            gradient[inside_rows, axis[inside_rows]] = np.where(
                points[inside_rows, axis[inside_rows]] >= 0., 1., -1.)
            return distance, gradient
        radius = np.linalg.norm(points[:, :2], axis=1)
        offset = np.column_stack((radius-self.geometry.radius,
                                  np.abs(points[:, 2])-self.geometry.halfLength))
        outside = np.maximum(offset, 0.)
        length = np.linalg.norm(outside, axis=1)
        axis = offset.argmax(1)
        distance = length+np.minimum(offset[np.arange(len(points)), axis], 0.)
        direction = np.zeros_like(offset)
        exterior = length > 0.
        direction[exterior] = outside[exterior]/length[exterior, None]
        inside_rows = np.flatnonzero(~exterior)
        direction[inside_rows, axis[inside_rows]] = 1.
        radial = np.zeros((len(points), 2))
        nonzero = radius > 0.
        radial[nonzero] = points[nonzero, :2]/radius[nonzero, None]
        radial[~nonzero, 0] = 1.
        gradient = np.column_stack((radial*direction[:, :1],
                                   np.where(points[:, 2] >= 0., 1., -1.)*direction[:, 1]))
        return distance, gradient


class BatchDistance:
    """Fixed moving samples query the original resident distance fields.

    ``query_sources`` is derived from the robot registration and declared
    owner motion. Target TSDFs keep every original supported scalar sample.
    Only winners of finite maxima return to CPU for the existing rigid-pose
    derivative. Registered native/native pairs remain in NativePairTable.
    """

    def __init__(self, geometries, *, query_sources, device='cuda:0'):
        started = perf_counter()
        # BEGIN library initialization uses Warp's caller-owned cache setting.
        wp.init()
        # END numerical kernels do not select an experiment-specific directory.
        self.geometries, self.query_sources, self.device = geometries, frozenset(query_sources), device
        self.samples = {name: MaterialSamples(geometries[name]) for name in query_sources}
        self.sampling_spacing = min(value.voxel_size for value in geometries.values()
                                    if isinstance(value, ObservedField))
        self.native_points = {name: native_volume_points(geometries[name], self.sampling_spacing, device=device)
                              for name in query_sources if not isinstance(geometries[name], ObservedField)}
        self.inventories = {(name, ()): PointInventory(samples,
            native_points=None if isinstance(geometries[name], ObservedField) else self.native_points[name])
            for name, samples in self.samples.items()}
        self.device_points = {key: torch.as_tensor(value.points.copy(), dtype=torch.float64, device=device)
                              for key, value in self.inventories.items()}
        self.fields = {(name, ()): geometry for name, geometry in geometries.items()
                       if isinstance(geometry, ObservedField)}
        self.device_fields = {key: DeviceField(value, device) for key, value in self.fields.items()}
        # A target band indexes only cells that can produce a value above
        # the row's -1 tail. For a TSDF this follows its scalar samples,
        # rather than interpreting a voxel value as Euclidean distance.
        self.field_bands = {}
        for name, geometry in geometries.items():
            if not isinstance(geometry, ObservedField):
                continue
            minimum = np.full(tuple(geometry.cell_shape), np.inf, dtype=np.float32)
            for corner in CORNERS.astype(int):
                slices = tuple(slice(int(k), int(k+n)) for k, n in zip(corner, geometry.cell_shape))
                np.minimum(minimum, geometry.values[slices], out=minimum)
            cells = np.argwhere(geometry.valid_cells & (minimum < geometry.voxel_size))
            self.field_bands[name] = (None if not len(cells) else geometry.origin+
                geometry.voxel_size*np.stack((cells.min(0), cells.max(0)+1)))
        self.native = {name: NativePointQuery(geometry, device=device)
                       for name, geometry in geometries.items() if not isinstance(geometry, ObservedField)}
        self.preparation_seconds = perf_counter()-started

    def inventory(self, name, excluded):
        """Prepare a declared original-cell source subset before its first use."""
        key = name, tuple(sorted(excluded))
        if key not in self.inventories:
            self.inventories[key] = PointInventory(self.samples[name], key[1],
                native_points=None if isinstance(self.geometries[name], ObservedField) else self.native_points[name])
            self.device_points[key] = torch.as_tensor(self.inventories[key].points.copy(),
                dtype=torch.float64, device=self.device)
        return self.inventories[key]

    def field(self, name, excluded):
        """Bind an original target's explicit cell exclusion on both devices."""
        key = name, tuple(sorted(excluded))
        if key not in self.fields:
            original = self.geometries[name]
            field = copy(original)
            field.valid_cells = original.valid_cells.copy()
            field.valid_cells[tuple(original.cells[np.asarray(key[1], dtype=np.int64)].T)] = False
            self.fields[key] = field
            self.device_fields[key] = DeviceField(field, self.device)
        return self.fields[key]

    # BEGIN GPU submission and host consumption have separate dependency points.
    def submit(self, *, records, columns, matrices, permissions, active, check_request):
        """Enqueue independent field queries and return their pending result."""
        started = perf_counter()
        pending = PendingDistance(self, len(records), started)
        groups = {}
        for ordinal in np.flatnonzero(active):
            first, second = records[ordinal]['first'], records[ordinal]['second']
            if first in self.native and second in self.native:
                continue
            permission = permissions[ordinal]
            excluded = () if permission is None else tuple(sorted(permission.excluded_second_cells))
            groups.setdefault((first, second, excluded), []).append(ordinal)
        skipped_states = 0
        for (first, second, excluded), group_rows in groups.items():
            group_rows = np.asarray(group_rows, dtype=np.int64)
            for reverse in (False, True):
                rows = group_rows
                source, target = (second, first) if reverse else (first, second)
                if source not in self.query_sources:
                    continue
                check_request()
                source_excluded, target_excluded = (excluded, ()) if reverse else ((), excluded)
                inventory = self.inventory(source, source_excluded)
                if not len(inventory.points):
                    continue
                source_pose = matrices[source][0, columns[rows]]
                target_pose = matrices[target][0, columns[rows]]
                scale = (self.geometries[target].voxel_size if target in self.fields_by_name
                         else self.geometries[source].voxel_size)
                target_bounds = (self.field_bands[target] if target in self.fields_by_name else
                    self.native[target].bounds+np.array([[-scale], [scale]]))
                # Rigid AABBs skip only disjoint source-coordinate and target
                # scalar-band ranges. They never define occupied geometry.
                def world_box(bounds, poses):
                    center = np.einsum('qij,j->qi', poses[:, :3, :3], bounds.mean(0))+poses[:, :3, 3]
                    half = np.einsum('qij,j->qi', np.abs(poses[:, :3, :3]), (bounds[1]-bounds[0])/2)
                    return center-half, center+half
                possible = np.ones(len(rows), dtype=bool)
                if target_bounds is not None:
                    source_lo, source_hi = world_box(inventory.bounds, source_pose)
                    target_lo, target_hi = world_box(target_bounds, target_pose)
                    possible = np.all((source_hi >= target_lo) & (target_hi >= source_lo), axis=1)
                skipped_states += int((~possible).sum())
                if not possible.any():
                    continue
                rows, source_pose, target_pose = rows[possible], source_pose[possible], target_pose[possible]
                # BEGIN one query kernel owns rigid coordinates, finite-region
                # membership and original scalar/support lookup for every point.
                local_source = self.device_points[source, tuple(sorted(source_excluded))]
                if target in self.native:
                    query = self.native[target]
                else:
                    self.field(target, target_excluded)
                    query = self.device_fields[target, tuple(sorted(target_excluded))]
                packed = submit_pair_maxima(query, local_source, source_pose, target_pose,
                    [permissions[index] for index in rows], scale)
                pending.groups.append((rows, reverse, inventory, source_pose, target_pose,
                                       target, target_excluded, scale))
                pending.device_results.append(packed)
        pending.skipped_states = skipped_states
        pending.submit_seconds = perf_counter()-started
        return pending
    # END native CPU geometry may run before PendingDistance.finish waits.

    @property
    def fields_by_name(self):
        return {name for name, geometry in self.geometries.items() if isinstance(geometry, ObservedField)}
# END source inventories are independent of optimization iterates and static environment size.


# BEGIN a numerical point owns one pending device-to-host geometry transfer.
class PendingDistance:
    """Own group outputs until CPU geometry reaches the shared result boundary.

    Native/native rows never occur in these groups. The caller can therefore
    fill those rows in values/covectors/witnesses while device queries run.
    Consumption follows original group and direction order, so strict greater
    comparisons preserve the original first-winner rule, including ties.
    """

    def __init__(self, owner, count, started):
        self.owner, self.started = owner, started
        self.values = np.full(count, -1.)
        self.covectors = np.zeros((count, 2, 6))
        self.witnesses = np.full((count, 3), np.nan)
        self.unknown = np.zeros(count, dtype=np.int64)
        self.queried = np.zeros(count, dtype=np.int64)
        self.groups, self.device_results = [], []
        self.submit_seconds, self.skipped_states = 0., 0

    def finish(self):
        """Wait once, then compute derivatives at the original winning points."""
        started = perf_counter()
        # A scene with no field groups has an empty numerical query domain.
        packed = (torch.cat(self.device_results).cpu().numpy() if self.device_results
                  else np.empty((0, 4)))
        waited = perf_counter()-started
        offset = 0
        derivatives_started = perf_counter()
        for rows, reverse, inventory, source_pose, target_pose, target, excluded, scale in self.groups:
            result = packed[offset:offset+len(rows)]
            offset += len(rows)
            maximum, winners = result[:, 0], result[:, 1].astype(np.int64)
            self.queried[rows] += result[:, 2].astype(np.int64)
            self.unknown[rows] += result[:, 3].astype(np.int64)
            changed = np.flatnonzero(maximum > self.values[rows])
            if not len(changed):
                continue
            ids, indices = rows[changed], winners[changed]
            chosen = inventory.points[indices]
            world = np.einsum('qij,qj->qi', source_pose[changed, :3, :3], chosen)
            world += source_pose[changed, :3, 3]
            local = np.einsum('qji,qj->qi', target_pose[changed, :3, :3],
                world-target_pose[changed, :3, 3])
            if target in self.owner.native:
                _, gradient = self.owner.native[target].sample(local)
            else:
                _, _, gradient, _ = self.owner.field(target, excluded).sample(local)
            self.values[ids], self.witnesses[ids] = maximum[changed], world
            direction = np.einsum('qij,qj->qi', target_pose[changed, :3, :3], gradient)/scale
            self.covectors[ids, int(reverse), :3] = -direction
            self.covectors[ids, int(reverse), 3:] = -np.cross(world-source_pose[changed, :3, 3], direction)
            self.covectors[ids, int(not reverse), :3] = direction
            self.covectors[ids, int(not reverse), 3:] = np.cross(world-target_pose[changed, :3, 3], direction)
        seconds = dict(host_submission=self.submit_seconds, host_result_wait=waited,
                       winning_derivatives=perf_counter()-derivatives_started)
        return self.values, self.covectors, dict(query_count=int(self.queried.sum()),
            unknown_query_count=int(self.unknown.sum()), query_counts=self.queried,
            unknown_counts=self.unknown, witness_world=self.witnesses,
            disjoint_source_band_states=self.skipped_states, submitted_groups=len(self.groups),
            result_transfers=int(bool(self.groups)), seconds=seconds,
            elapsed_seconds=self.submit_seconds+perf_counter()-started,
            submission_to_consumption_seconds=perf_counter()-self.started,
            scope='Fixed original samples; host timings exclude intervening CPU native work')
# END device completion and original CPU gradient lookup share one result owner.
