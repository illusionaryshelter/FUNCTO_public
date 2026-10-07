"""Finite contact intersections over the already bound target solid model.

The ordinary scene query remains responsible for every nonmaterial pair.
These helpers localize material permission to observed TSDF components or
explicit cuboids; no task identity or live object truth enters this adapter.
"""

# BEGIN bounded finite-cell contact witnesses and exact box distance gradients.
from copy import deepcopy
import math

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull, cKDTree
from scipy.spatial.transform import Rotation
import torch


# BEGIN complete finite-component contact proofs, including points and surfaces.
_LP_OPTIONS = {"primal_feasibility_tolerance": 1e-9, "dual_feasibility_tolerance": 1e-9}


def convex_component_intersects_region(vertices, region):
    """Test the entire convex component against ``A @ x <= b``.

    Barycentric coordinates represent its complete convex hull, including a
    single observed point, an edge, or a zero-level polygon. No 3D hull or
    sampled contact witness substitutes for these lower-dimensional sets.
    """
    vertices = np.asarray(vertices, dtype=float)
    normals, bounds = map(np.asarray, region)
    result = linprog(np.zeros(len(vertices)), A_ub=normals @ vertices.T,
        b_ub=bounds, A_eq=np.ones((1, len(vertices))), b_eq=[1.],
        bounds=(0., None), method="highs", options=_LP_OPTIONS)
    if result.status == 2:
        return {"completed": True, "intersects": False, "numerical_certificate": False}
    if not result.success:
        return {"completed": False, "reason": "component_region_query_incomplete",
                "solver_status": int(result.status)}
    return {"completed": True, "intersects": True,
            "witness": (result.x @ vertices).tolist(), "numerical_certificate": False}


def _component_intersection_support(vertices, region, *, A_ub=None, b_ub=None,
                                    A_eq=None, b_eq=None, variable_bounds=(0., None)):
    """Maximize every region plane over one complete barycentric feasible set."""
    normals, bounds = map(np.asarray, region)
    maximum = -np.inf
    # A zero objective also distinguishes an empty intersection when the
    # caller asks only for intersection, without any region inequalities.
    queries = list(zip(normals, bounds)) or [(np.zeros(3), 0.)]
    for axis, bound in queries:
        objective = vertices @ axis
        result = linprog(-objective, A_ub=A_ub, b_ub=b_ub,
            A_eq=A_eq, b_eq=b_eq, bounds=variable_bounds, method="highs", options=_LP_OPTIONS)
        if result.status == 2:
            return {"completed": True, "contained": True, "intersection_empty": True,
                    "numerical_certificate": False, "method": "complete_component_intersection_support"}
        if not result.success:
            return {"completed": False, "contained": False,
                    "reason": "component_intersection_support_incomplete", "solver_status": int(result.status)}
        point = result.x @ vertices
        excess = float(axis @ point-bound)
        # BEGIN an explicit identical constraint is already a support bound.
        # At a clipping plane, reconstructing a barycentric point may put it
        # one floating-point step outside that same plane. An exactly equal
        # input row proves the upper bound directly; this needs no tolerance
        # and does not enlarge the permitted region or waive a real gap.
        if A_ub is not None:
            identical = np.all(np.asarray(A_ub) == objective, axis=1)
            if np.any(identical):
                excess = min(excess, float(np.min(np.asarray(b_ub)[identical])-bound))
        # END only an actual input inequality can override point round-off.
        maximum = max(maximum, excess)
        if excess > 0.:
            return {"completed": True, "contained": False, "intersection_empty": False,
                    "outside_witness": point.tolist(), "maximum_constraint_excess_m": excess,
                    "numerical_certificate": False, "method": "complete_component_intersection_support"}
    return {"completed": True, "contained": True, "intersection_empty": False,
            "maximum_constraint_excess_m": maximum, "numerical_certificate": False,
            "method": "complete_component_intersection_support"}


def convex_component_overlap_within_region(actor_halfspaces, component_vertices, region, actor_margin=0.):
    """Prove containment of a source convex solid's entire component overlap.

    ``actor_halfspaces`` and ``region`` are ``(A, b)`` for ``A @ x <= b``.
    The existing measured-body allowance shifts source support planes only;
    it does not enlarge the material permission or fill a TSDF surface.
    """
    vertices = np.asarray(component_vertices, dtype=float)
    normals, bounds = map(np.asarray, actor_halfspaces)
    shifted = bounds + float(actor_margin)*np.linalg.norm(normals, axis=1)
    return _component_intersection_support(vertices, region,
        A_ub=normals @ vertices.T, b_ub=shifted,
        A_eq=np.ones((1, len(vertices))), b_eq=np.ones(1))


def _component_support_planes(vertices):
    """Finite supporting planes for a convex component of any affine rank.

    Zero-thickness directions remain equalities represented by two planes.
    Shifting every normalized plane by a measured radius contains the whole
    component plus that radius ball, as for the existing native-solid model.
    """
    vertices = np.asarray(vertices, float)
    origin = vertices[0]
    _, singular, axes = np.linalg.svd(vertices-origin, full_matrices=True)
    # Coordinates result from finite-precision rigid transforms. Classification
    # uses their representable spatial precision, not a contact allowance.
    tolerance = np.finfo(float).eps*max(vertices.shape)*max(1., float(np.abs(vertices).max()))
    rank = int(np.count_nonzero(singular > tolerance))
    if rank == 3:
        planes = ConvexHull(vertices).equations
        return planes[:, :3], -planes[:, 3], rank
    if rank == 2:
        basis = axes[:2].T
        polygon = ConvexHull((vertices-origin) @ basis)
        normals = polygon.equations[:, :2] @ basis.T
        bounds = normals @ origin-polygon.equations[:, 2]
    else:
        normals = np.empty((0, 3))
        bounds = np.empty(0)
        if rank == 1:
            coordinate = (vertices-origin) @ axes[0]
            normals = np.array([axes[0], -axes[0]])
            bounds = np.array([coordinate.max()+axes[0] @ origin,
                               -coordinate.min()-axes[0] @ origin])
    fixed = axes[rank:]
    normals = np.concatenate((normals, fixed, -fixed))
    bounds = np.r_[bounds, fixed @ origin, -fixed @ origin]
    return normals, bounds, rank


def component_component_overlap_within_region(first_vertices, second_vertices, region,
                                              *, first_margin_m=0., second_margin_m=0.):
    """Prove the full intersection of two finite components by two simplexes.

    Neither component needs nonzero volume. Measured component bands use the
    same conservative supporting-plane outer polytope as native solid bands;
    proving containment of that full outer intersection covers the real band.
    """
    first, second = np.asarray(first_vertices, float), np.asarray(second_vertices, float)
    if first_margin_m or second_margin_m:
        first_a, first_b, _ = _component_support_planes(first)
        second_a, second_b, _ = _component_support_planes(second)
        evidence = _component_intersection_support(np.eye(3), region,
            A_ub=np.concatenate((first_a, second_a)),
            b_ub=np.r_[first_b+first_margin_m*np.linalg.norm(first_a, axis=1),
                       second_b+second_margin_m*np.linalg.norm(second_a, axis=1)],
            variable_bounds=(None, None))
        evidence["margin_geometry"] = "component_support_plane_outer_polytope"
        return evidence
    n, m = len(first), len(second)
    equalities = np.zeros((5, n+m))
    equalities[0, :n], equalities[1, n:] = 1., 1.
    equalities[2:, :n], equalities[2:, n:] = first.T, -second.T
    # Objective and returned point use the first component; the equality
    # constraints require exactly the same point in the second component.
    objective_vertices = np.concatenate((first, np.zeros_like(second)), axis=0)
    return _component_intersection_support(objective_vertices, region,
        A_eq=equalities, b_eq=np.array([1., 1., 0., 0., 0.]))


def _component_region_slack(vertices, normals, bounds):
    """Differentiable required-contact slack on an unchanged finite component."""
    vertices = np.asarray(vertices, float)
    matrix = normals.detach().cpu().numpy()
    rhs = bounds.detach().cpu().numpy()
    result = linprog(np.r_[np.zeros(len(vertices)), 1.],
        A_ub=np.c_[matrix @ vertices.T, -np.ones(len(matrix))], b_ub=rhs,
        A_eq=np.r_[np.ones(len(vertices)), 0.][None], b_eq=[1.],
        bounds=[(0., None)]*(len(vertices)+1), method="highs", options=_LP_OPTIONS)
    if not result.success:
        raise ValueError(f"Required component contact did not complete: {result.message}")
    point = result.x[:-1] @ vertices
    dual = np.asarray(result.ineqlin.marginals)
    return (bounds.new_tensor(result.fun)
            + ((normals-normals.detach())*normals.new_tensor(-dual[:, None]*point)).sum()
            + ((bounds-bounds.detach())*bounds.new_tensor(dual)).sum())
# END each permission covers the complete intersection, never a witness subset.


def region_slack(normals, bounds):
    """Minimum nonnegative common violation of a finite convex region.

    HiGHS supplies both a complete feasibility solve and its dual. The
    envelope derivative gives the value gradient with respect to the moving
    halfspaces; no sampled feature vertex replaces existential overlap.
    This is used only for explicitly REQUIRED contact, not every permission.
    """
    shape = bounds.shape[:-1]
    matrices = normals.detach().cpu().numpy().reshape(-1, normals.shape[-2], 3)
    rhs = bounds.detach().cpu().numpy().reshape(-1, bounds.shape[-1])
    if len(matrices) > 4096:
        raise ValueError("Required finite contact exceeds the 4096-state feasibility budget")
    values, gradient_a, gradient_b = [], [], []
    for matrix, bound in zip(matrices, rhs):
        result = linprog([0., 0., 0., 1.], A_ub=np.c_[matrix, -np.ones(len(matrix))],
                         b_ub=bound, bounds=[(None, None)]*3+[(0., None)], method="highs",
                         options={"primal_feasibility_tolerance": 1e-9, "dual_feasibility_tolerance": 1e-9})
        if not result.success:
            raise ValueError(f"Required contact region feasibility did not complete: {result.message}")
        dual = np.asarray(result.ineqlin.marginals)
        values.append(result.fun)
        gradient_a.append(-dual[:, None]*result.x[:3])
        gradient_b.append(dual)
    value = bounds.new_tensor(values).reshape(shape)
    return (value+((normals-normals.detach())*normals.new_tensor(np.asarray(gradient_a)).reshape(normals.shape)).sum((-1, -2))
            + ((bounds-bounds.detach())*bounds.new_tensor(np.asarray(gradient_b)).reshape(bounds.shape)).sum(-1))


def query_geometry(query):
    """Own the same native solid input later consumed by independent Coal."""
    explicit = getattr(query, "independent_geometry", None)
    if explicit is not None:
        return deepcopy(explicit)
    native = getattr(query, "scene_collision", None)
    model = getattr(native, "scene_model", None)
    if model is None or isinstance(model, list):
        return {"solids": [], "query_gaps": ["No single independent native scene model"]}
    solids, gaps = [], []
    for kind in ("cuboid", "cylinder", "sphere", "capsule", "mesh", "voxel"):
        for shape in getattr(model, kind, ()) or ():
            if len(solids) >= 4096:
                gaps.append("Independent scene solid budget exceeded")
                break
            row = {"name": shape.name, "kind": kind, "pose": list(shape.pose), "source": query.source}
            if kind == "cuboid":
                row["dimensions"] = list(shape.dims)
            elif kind in {"cylinder", "sphere"}:
                row["dimensions"] = [shape.radius]+([shape.height] if kind == "cylinder" else [])
            elif kind == "mesh":
                mesh = shape.get_trimesh_mesh(process=False, transform_with_pose=False)
                if len(mesh.vertices) > 65536 or not mesh.is_watertight or not mesh.is_convex:
                    gaps.append(f"{shape.name}: requires complete convex solid adaptation")
                    continue
                row["vertices"] = np.asarray(mesh.vertices).tolist()
            else:
                gaps.append(f"{shape.name}: {kind} requires an independent solid adapter")
                continue
            solids.append(row)
    return {"solids": solids, "query_gaps": gaps}


# BEGIN required contact consumes actual target surfaces, never negative caps.
_BOX_SIGNS = np.array([[x, y, z] for x in (-1., 1.) for y in (-1., 1.) for z in (-1., 1.)])


def _bind_required_surfaces(target, surfaces):
    target.surface_components = tuple(surfaces)
    target.surface_centres = np.asarray([points.mean(0) for points in surfaces], float).reshape(-1, 3)
    target.surface_radii = np.asarray([
        np.linalg.norm(points-centre, axis=1).max() for points, centre in zip(surfaces, target.surface_centres)])
    target.surface_tree = cKDTree(target.surface_centres)
    target.maximum_surface_radius = float(target.surface_radii.max(initial=0.))


def _required_surface_contact(target, policy):
    """Existential LP contact on all candidate zero surfaces or native faces."""
    active = policy["active"].reshape(-1)
    shape = policy["active"].shape
    selected = torch.nonzero(active, as_tuple=False).flatten()
    points = policy["feature_points"].reshape(len(active), -1, 3)
    centre = points.mean(-2)
    radius = torch.linalg.vector_norm(points-centre[:, None], dim=-1).amax(-1)+policy["allowance"]
    output = radius*0
    if not len(selected):
        return output.reshape(shape)
    if not len(target.surface_centres):
        target.last_required_report = {"active_states": len(selected), "candidate_components": 0,
            "target_surface_inventory_empty": True, "negative_components_do_not_establish_contact": True}
        return output.index_fill(0, selected, 1.).reshape(shape)
    rows = selected.detach().cpu().numpy()
    centres = centre[selected].detach().cpu().numpy()
    radii = np.nextafter(radius[selected].detach().cpu().numpy()+target.maximum_surface_radius, np.inf)
    lengths = target.surface_tree.query_ball_point(centres, radii, return_length=True, workers=1)
    if int(lengths.sum()) > 4096:
        raise ValueError(f"Required surface contact needs {int(lengths.sum())} component LPs, budget 4096")
    neighbors = target.surface_tree.query_ball_point(centres, radii, workers=1)
    normals = policy["normals"].reshape(len(active), -1, 3)
    bounds = policy["bounds"].reshape(len(active), -1)
    values = []
    for local_index, state_index in enumerate(rows):
        candidates = neighbors[local_index]
        if len(candidates):
            slack = [_component_region_slack(target.surface_components[index],
                normals[state_index], bounds[state_index]) for index in candidates]
            values.append(torch.stack(slack).amin())
        else:
            _, nearest = target.surface_tree.query(centres[local_index], workers=1)
            # The complete bounding-ball search excluded every possible
            # intersection. This positive bound supplies a useful approach
            # gradient without pretending the nearest surface is contact.
            distance = torch.linalg.vector_norm(centre[state_index]-centre.new_tensor(target.surface_centres[nearest]))
            values.append(torch.relu(distance-target.maximum_surface_radius-radius[state_index]))
    output = output.index_copy(0, selected, torch.stack(values))
    target.last_required_report = {"active_states": len(selected), "candidate_components": int(lengths.sum()),
        "maximum_component_lps": 4096, "target_surface_inventory_empty": False,
        "negative_components_do_not_establish_contact": True,
        "scope": "some actual target zero surface or native face intersects the finite material/patch/gap region"}
    return output.reshape(shape)
# END an observed negative volume may block motion without witnessing contact.


class ContactCellGeometry:
    """A complete finite box union with a bounded exact spatial broad phase.

    A KD-tree removes only cells proved too far from the queried sphere by
    the enclosing cell ball. Candidate count is obtained BEFORE allocating
    neighbor lists. At most two million candidates and 32768 live pairs enter
    a query; exceeding either supported model/query budget raises explicitly.
    These boxes are explicitly supplied native solids. Mapper observations
    use ObservedContactGeometry and never enter an ESDF-to-box conversion.
    """

    max_cells = 250_000
    max_candidate_pairs = 2_000_000
    max_candidate_plane_products = 32_000_000
    pair_chunk = 32_768

    def __init__(self, geometry):
        centres, halves, rotations = [], [], []
        self.gaps = list(geometry.get("query_gaps", ()))
        self.cell_half_diagonal_m = 0.
        for record in geometry["solids"]:
            pose = np.asarray(record["pose"], float)
            if (pose.shape != (7,) or not np.isfinite(pose).all()
                    or not np.isclose(np.linalg.norm(pose[3:]), 1., atol=1e-6)):
                raise ValueError("Contact target cells need a finite normalized native pose")
            rotation = Rotation.from_quat(pose[[4, 5, 6, 3]]).as_matrix()
            if record["kind"] == "cuboid":
                dimensions = np.asarray(record["dimensions"], float)
                if dimensions.shape != (3,) or not np.isfinite(dimensions).all() or np.any(dimensions <= 0):
                    raise ValueError("Contact target cuboids need positive native dimensions")
                local = np.zeros((1, 3))
                half = dimensions[None]/2
            else:
                self.gaps.append(f"{record['name']}: finite native contact requires an explicit supported solid")
                continue
            centres.append(local @ rotation.T+pose[:3])
            halves.append(half)
            rotations.append(np.broadcast_to(rotation, (len(local), 3, 3)).copy())
        self.centres = np.concatenate(centres) if centres else np.empty((0, 3))
        self.halves = np.concatenate(halves) if halves else np.empty((0, 3))
        self.rotations = np.concatenate(rotations) if rotations else np.empty((0, 3, 3))
        if len(self.centres) > self.max_cells:
            raise ValueError("Complete contact target exceeds its 250000-cell model budget")
        self.tree = cKDTree(self.centres)
        self.maximum_cell_radius = float(np.linalg.norm(self.halves, axis=1).max(initial=0.))
        self.last_report = None
        self.last_required_report = None
        surfaces = []
        for centre, half, rotation in zip(self.centres, self.halves, self.rotations):
            vertices = (_BOX_SIGNS*half) @ rotation.T+centre
            surfaces.extend(vertices[_BOX_SIGNS[:, axis] == sign]
                            for axis in range(3) for sign in (-1., 1.))
        _bind_required_surfaces(self, surfaces)

    @property
    def complete(self):
        return not self.gaps

    def workspace_bytes(self):
        return int(self.pair_chunk*1600)

    @property
    def storage_bytes(self):
        return int(self.centres.nbytes+self.halves.nbytes+self.rotations.nbytes
                   + self.surface_centres.nbytes+self.surface_radii.nbytes
                   + sum(vertices.nbytes for vertices in self.surface_components))

    def required_contact(self, policy):
        """Require a supplied native box face, not merely interior occupancy."""
        return _required_surface_contact(self, policy)

    def query(self, spheres, policies):
        """Return complete-cell collision for the enabled material spheres.

        ``policies`` gives sphere indices, per-state active flags and owner-
        local halfspaces of the complete finite permitted intersection. For
        each cell/sphere pair, min(support(cell), support(sphere)) is an upper
        bound on support of their intersection. Passing EVERY region plane
        therefore cannot lend permission to a side/back/out-of-patch overlap.
        Failure of this sufficient bound keeps the unmodified collision.

        Search of the maximum pair cost is detached. Its chosen exact box
        distance is recomputed with Torch, preserving the max's gradient
        (one valid subgradient at ties) without saving all candidate pairs.
        """
        if not self.complete:
            raise ValueError("A partial target-cell inventory cannot grant contact permission")
        states, count = math.prod(spheres.shape[:-2]), spheres.shape[-2]
        flat = spheres.reshape(states*count, 4)
        enabled = torch.zeros((states, count), dtype=torch.bool, device=spheres.device)
        for policy in policies:
            enabled[:, list(policy["sphere_indices"])] |= policy["active"].reshape(-1, 1)
        chosen = torch.nonzero(enabled.reshape(-1), as_tuple=False).flatten()
        base = flat[chosen]
        supplied = base.detach().cpu().numpy().astype(float, copy=False)
        if len(supplied) and (not np.isfinite(supplied).all() or np.any(supplied[:, 3] <= 0)):
            raise ValueError("Finite contact queries need positive finite material spheres")
        radius = supplied[:, 3]+self.maximum_cell_radius
        # The Euclidean cell enclosing ball supplies a complete broad phase.
        # One representational ULP protects equality at a touching boundary.
        radius = np.nextafter(radius, np.inf)
        lengths = self.tree.query_ball_point(supplied[:, :3], radius, return_length=True, workers=1)
        candidates = int(np.sum(lengths))
        raw_pairs = int(len(supplied)*len(self.centres))
        plane_count = max((policy["normals"].shape[-2] for policy in policies), default=1)
        if candidates > self.max_candidate_pairs:
            raise ValueError(f"Complete contact cell query requires {candidates} candidate pairs, budget {self.max_candidate_pairs}; raw product {raw_pairs}")
        if candidates*plane_count*len(policies) > self.max_candidate_plane_products:
            raise ValueError("Complete contact region checks exceed their 32000000 candidate-plane budget")
        live_chunk = min(self.pair_chunk, max(1, 1_048_576//plane_count))
        neighbors = self.tree.query_ball_point(supplied[:, :3], radius, workers=1)
        ball_indices = np.repeat(np.arange(len(supplied)), lengths)
        cell_indices = np.concatenate(neighbors).astype(np.int64) if candidates else np.empty(0, np.int64)
        best_values = np.zeros(len(supplied))
        best_cells = np.full(len(supplied), -1, np.int64)
        best_allowances = np.zeros(len(supplied))
        permitted_pairs = 0
        selected_flat = chosen.detach().cpu().numpy()
        with torch.no_grad():
            for begin in range(0, candidates, live_chunk):
                ib = ball_indices[begin:begin+live_chunk]
                ic = cell_indices[begin:begin+live_chunk]
                balls = base[torch.as_tensor(ib, device=spheres.device)]
                centre = spheres.new_tensor(self.centres[ic])
                half = spheres.new_tensor(self.halves[ic])
                rotation = spheres.new_tensor(self.rotations[ic])
                local = torch.einsum("ni,nij->nj", balls[:, :3]-centre, rotation)
                delta = local.abs()-half
                sdf = torch.linalg.vector_norm(torch.relu(delta), dim=-1)+delta.amax(-1).clamp_max(0)
                raw = torch.relu(balls[:, 3]-sdf)
                allowance = torch.zeros_like(raw)
                indices = selected_flat[ib]
                state_index = torch.as_tensor(indices//count, device=spheres.device)
                for policy in policies:
                    member = np.isin(indices % count, policy["sphere_indices"])
                    active = (policy["active"].reshape(-1)[state_index]
                              & policy["normal_valid"].reshape(-1)[state_index]
                              & torch.as_tensor(member, device=spheres.device))
                    normals = policy["normals"].reshape(states, -1, 3)[state_index]
                    bounds = policy["bounds"].reshape(states, -1)[state_index]
                    sphere_support = torch.einsum("nki,ni->nk", normals, balls[:, :3])+balls[:, 3:4]*torch.linalg.vector_norm(normals, dim=-1)
                    box_support = (torch.einsum("nki,ni->nk", normals, centre)
                        + (torch.einsum("nki,nij->nkj", normals, rotation).abs()*half[:, None]).sum(-1))
                    inside = (torch.minimum(sphere_support, box_support) <= bounds).all(-1)
                    allowed = active & inside
                    allowance = torch.maximum(allowance, allowed.to(raw.dtype)*policy["allowance"])
                cost = torch.relu(raw-allowance).cpu().numpy()
                selected_allowance = allowance.cpu().numpy()
                permitted_pairs += int(np.count_nonzero(selected_allowance))
                # Grouped numpy maximum chooses a real maximizer, never an
                # average contact witness. Only that pair's gradient is live.
                order = np.lexsort((np.arange(len(ib)), -cost, ib))
                ordered = ib[order]
                first = np.r_[True, ordered[1:] != ordered[:-1]]
                winner = order[first]
                improve = cost[winner] > best_values[ib[winner]]
                winner = winner[improve]
                best_values[ib[winner]] = cost[winner]
                best_cells[ib[winner]] = ic[winner]
                best_allowances[ib[winner]] = selected_allowance[winner]
        output = flat[:, 3]*0
        hit = np.flatnonzero(best_cells >= 0)
        if len(hit):
            selected = torch.as_tensor(hit, device=spheres.device)
            balls = base[selected]
            cells = best_cells[hit]
            local = torch.einsum("ni,nij->nj", balls[:, :3]-spheres.new_tensor(self.centres[cells]),
                                 spheres.new_tensor(self.rotations[cells]))
            delta = local.abs()-spheres.new_tensor(self.halves[cells])
            sdf = torch.linalg.vector_norm(torch.relu(delta), dim=-1)+delta.amax(-1).clamp_max(0)
            value = torch.relu(balls[:, 3]-sdf-spheres.new_tensor(best_allowances[hit]))
            output = output.index_copy(0, chosen[selected], value)
        self.last_report = {"complete_cells": len(self.centres), "enabled_spheres": len(supplied),
            "raw_pair_product": raw_pairs, "candidate_pairs": candidates,
            "max_candidate_pairs": self.max_candidate_pairs, "maximum_live_pairs": min(candidates, live_chunk),
            "maximum_region_planes": plane_count, "max_candidate_plane_products": self.max_candidate_plane_products,
            "permitted_pairs": permitted_pairs, "cell_half_diagonal_m": self.cell_half_diagonal_m,
            "scope": "complete finite target-cell union; conservative whole sphere/cell intersection permission"}
        return output.reshape(spheres.shape[:-1]), enabled.reshape(spheres.shape[:-1])
# END no sampled contact point can authorize an unexamined overlap.


# BEGIN observed zero surfaces and negative affine pieces share contact inventory.
class ObservedContactGeometry:
    """Finite TSDF components for contact guidance and required surface contact.

    The component union is kept nonconvex. Every component is independently
    convex by construction; its support and point distance can therefore use
    its complete vertex set. Sphere costs guide optimization only. Source
    solid acceptance uses the complete-intersection LP helpers above.
    """

    max_cells = 250_000
    max_candidate_pairs = 2_000_000
    max_candidate_plane_products = 32_000_000
    pair_chunk = 4096
    cell_half_diagonal_m = 0.

    def __init__(self, geometry):
        # BEGIN raw contact vertices do not need a second source collision map.
        from .tsdf_geometry import iter_source_components
        # END the shared decoder keeps the same complete component inventory.

        self.gaps = list(geometry.get("query_gaps", ()))
        components, negative, surfaces = [], [], []
        for record in geometry["solids"]:
            pose = np.asarray(record["pose"], float)
            if (pose.shape != (7,) or not np.isfinite(pose).all()
                    or not np.isclose(np.linalg.norm(pose[3:]), 1., atol=1e-6)):
                raise ValueError("Observed contact components need a finite normalized native pose")
            rotation = Rotation.from_quat(pose[[4, 5, 6, 3]]).as_matrix()
            if record["kind"] == "tsdf":
                for kind, _, points in iter_source_components(record):
                    vertices = np.asarray(points, float) @ rotation.T+pose[:3]
                    components.append(vertices)
                    negative.append(kind != "surface")
                    if kind == "surface":
                        surfaces.append(vertices)
                    if len(components) > self.max_cells:
                        raise ValueError("Complete contact target exceeds its 250000-component model budget")
            elif record["kind"] == "cuboid":
                dimensions = np.asarray(record["dimensions"], float)
                if dimensions.shape != (3,) or not np.isfinite(dimensions).all() or np.any(dimensions <= 0):
                    raise ValueError("Native contact cuboids require positive finite dimensions")
                vertices = (_BOX_SIGNS*dimensions/2) @ rotation.T+pose[:3]
                components.append(vertices)
                negative.append(True)
                surfaces.extend(vertices[_BOX_SIGNS[:, axis] == sign]
                                for axis in range(3) for sign in (-1., 1.))
            else:
                self.gaps.append(f"{record['name']}: unsupported observed contact component {record['kind']}")
        if len(components) > self.max_cells:
            raise ValueError("Complete contact target exceeds its 250000-component model budget")
        self.vertices = tuple(components)
        self.negative = np.asarray(negative, bool)
        self.centres = np.asarray([points.mean(0) for points in components], float).reshape(-1, 3)
        self.radii = np.asarray([np.linalg.norm(points-centre, axis=1).max()
                                for points, centre in zip(components, self.centres)])
        self.tree = cKDTree(self.centres)
        self.maximum_cell_radius = float(self.radii.max(initial=0.))
        self._distance_models = {}
        _bind_required_surfaces(self, surfaces)
        self.last_report = None
        self.last_required_report = None

    @property
    def complete(self):
        return not self.gaps

    @property
    def storage_bytes(self):
        # Surface views refer to the owned component arrays. Native box faces
        # are separate, small arrays and are included conservatively here.
        model_bytes = sum(value.nbytes for model in self._distance_models.values()
                          for value in model.values() if isinstance(value, np.ndarray))
        return int(self.centres.nbytes+self.radii.nbytes+self.negative.nbytes
            + self.surface_centres.nbytes+self.surface_radii.nbytes
            + sum(points.nbytes for points in self.vertices)
            + sum(points.nbytes for points in self.surface_components)+model_bytes)

    def workspace_bytes(self):
        # At most twelve hull triangles per eight-vertex component are live
        # per pair, plus the bounded component/region support products.
        return int(self.pair_chunk*4096)

    def required_contact(self, policy):
        return _required_surface_contact(self, policy)

    def _distance_model(self, index):
        """Cache only queried component hulls; never hull the whole inventory."""
        model = self._distance_models.get(index)
        if model is not None:
            return model
        points = self.vertices[index]
        normals, bounds, rank = _component_support_planes(points)
        model = {"rank": rank, "normals": normals, "bounds": bounds}
        if rank == 0:
            model["point"] = points[0]
        elif rank == 1:
            direction = normals[0]
            coordinate = points @ direction
            model["endpoints"] = points[[coordinate.argmin(), coordinate.argmax()]]
        elif rank == 2:
            basis = np.linalg.svd(points-points[0], full_matrices=True)[2][:2].T
            order = ConvexHull((points-points[0]) @ basis).vertices
            faces = np.array([[order[0], order[i], order[i+1]] for i in range(1, len(order)-1)])
            model["triangles"] = points[faces]
        else:
            model["triangles"] = points[ConvexHull(points).simplices]
        self._distance_models[index] = model
        return model

    def _closest_points(self, indices, points):
        """Use maintained triangle distance code and exact convex inside planes."""
        from trimesh.triangles import closest_point

        nearest = np.empty_like(points)
        signs = np.ones(len(points))
        order = np.argsort(indices, kind="stable")
        boundaries = np.r_[0, np.flatnonzero(np.diff(indices[order]))+1, len(order)]
        for begin, end in zip(boundaries[:-1], boundaries[1:]):
            rows = order[begin:end]
            index = int(indices[rows[0]])
            model = self._distance_model(index)
            current = points[rows]
            if model["rank"] == 0:
                nearest[rows] = model["point"]
            elif model["rank"] == 1:
                first, second = model["endpoints"]
                edge = second-first
                parameter = np.clip((current-first) @ edge/(edge @ edge), 0., 1.)
                nearest[rows] = first+parameter[:, None]*edge
            else:
                triangles = model["triangles"]
                witnesses = closest_point(np.tile(triangles, (len(rows), 1, 1)),
                    np.repeat(current, len(triangles), axis=0)).reshape(len(rows), len(triangles), 3)
                squared = np.square(witnesses-current[:, None]).sum(-1)
                nearest[rows] = witnesses[np.arange(len(rows)), squared.argmin(axis=1)]
                if model["rank"] == 3:
                    values = current @ model["normals"].T-model["bounds"]
                    inside = values.max(axis=1) <= 0.
                    if self.negative[index]:
                        face = values[inside].argmax(axis=1)
                        signed = values[inside, face]
                        nearest[rows[inside]] = current[inside]-signed[:, None]*model["normals"][face]
                        signs[rows[inside]] = -1.
                    else:
                        # An all-zero affine tetrahedron is itself zero-level
                        # evidence. Its interior distance to that set is zero.
                        nearest[rows[inside]] = current[inside]
        return nearest, signs

    def query(self, spheres, policies):
        """Differentiable sphere guidance over every possible target component."""
        if not self.complete:
            raise ValueError("A partial target inventory cannot grant contact permission")
        policies = tuple(policies)
        states, count = math.prod(spheres.shape[:-2]), spheres.shape[-2]
        flat = spheres.reshape(states*count, 4)
        enabled = torch.zeros((states, count), dtype=torch.bool, device=spheres.device)
        for policy in policies:
            enabled[:, list(policy["sphere_indices"])] |= policy["active"].reshape(-1, 1)
        chosen = torch.nonzero(enabled.reshape(-1), as_tuple=False).flatten()
        supplied = flat[chosen].detach().cpu().numpy().astype(float, copy=False)
        if len(supplied) and (not np.isfinite(supplied).all() or np.any(supplied[:, 3] <= 0)):
            raise ValueError("Finite contact queries need positive finite material spheres")
        radii = np.nextafter(supplied[:, 3]+self.maximum_cell_radius, np.inf)
        lengths = self.tree.query_ball_point(supplied[:, :3], radii, return_length=True, workers=1)
        candidates = int(np.sum(lengths))
        # BEGIN zero allowance never changes a component's collision cost.
        # Keep every policy in the enabled material-sphere union above. A
        # zero allowance contributes exactly zero for every region result,
        # so only positive allowances need halfspace work or its budget.
        # Complete component distances and required-contact checks remain
        # independent of this algebraically irrelevant region calculation.
        region_policies = tuple(policy for policy in policies if policy["allowance"] != 0.)
        plane_count = max((policy["normals"].shape[-2] for policy in region_policies), default=1)
        candidate_plane_products = candidates*plane_count*len(region_policies)
        # END positive-allowance policies retain the original work estimate.
        if candidates > self.max_candidate_pairs:
            raise ValueError(f"Complete contact component query needs {candidates} candidate pairs, budget {self.max_candidate_pairs}")
        if candidate_plane_products > self.max_candidate_plane_products:
            raise ValueError("Complete component/region queries exceed the 32000000-product budget")
        neighbors = self.tree.query_ball_point(supplied[:, :3], radii, workers=1)
        ball_indices = np.repeat(np.arange(len(supplied)), lengths)
        component_indices = np.concatenate(neighbors).astype(np.int64) if candidates else np.empty(0, np.int64)
        best_values = np.zeros(len(supplied))
        best_components = np.full(len(supplied), -1, np.int64)
        best_allowances = np.zeros(len(supplied))
        best_points = np.zeros((len(supplied), 3))
        best_signs = np.ones(len(supplied))
        selected_flat = chosen.detach().cpu().numpy()
        live_chunk = min(self.pair_chunk, max(1, 131072//plane_count))
        permitted_pairs = 0
        for begin in range(0, candidates, live_chunk):
            ib = ball_indices[begin:begin+live_chunk]
            ic = component_indices[begin:begin+live_chunk]
            balls = supplied[ib]
            nearest, signs = self._closest_points(ic, balls[:, :3])
            sdf = signs*np.linalg.norm(balls[:, :3]-nearest, axis=1)
            allowance = np.zeros(len(ib))
            # BEGIN omit only region work with identically zero allowance.
            if region_policies:
                # These arrays serve only the halfspace support tests. The
                # complete distance query above never depends on padding.
                padded = np.asarray([np.pad(self.vertices[index], ((0, 8-len(self.vertices[index])), (0, 0)),
                                             mode="edge") for index in ic])
                states_index = selected_flat[ib]//count
            for policy in region_policies:
                active = (policy["active"].reshape(-1).detach().cpu().numpy()[states_index]
                    & policy["normal_valid"].reshape(-1).detach().cpu().numpy()[states_index]
                    & np.isin(selected_flat[ib] % count, policy["sphere_indices"]))
                normals = policy["normals"].detach().cpu().numpy().reshape(states, -1, 3)[states_index]
                bounds = policy["bounds"].detach().cpu().numpy().reshape(states, -1)[states_index]
                sphere_support = np.einsum("nki,ni->nk", normals, balls[:, :3])+balls[:, 3:4]*np.linalg.norm(normals, axis=-1)
                component_support = np.einsum("nki,nvi->nkv", normals, padded).max(-1)
                inside = (np.minimum(sphere_support, component_support) <= bounds).all(-1)
                allowance = np.maximum(allowance, (active & inside)*float(policy["allowance"]))
            # END base collision and positive-allowance arithmetic are unchanged.
            cost = np.maximum(balls[:, 3]-sdf-allowance, 0.)
            permitted_pairs += int(np.count_nonzero(allowance))
            order = np.lexsort((np.arange(len(ib)), -cost, ib))
            ordered = ib[order]
            winners = order[np.r_[True, ordered[1:] != ordered[:-1]]]
            winners = winners[cost[winners] > best_values[ib[winners]]]
            selected = ib[winners]
            best_values[selected] = cost[winners]
            best_components[selected] = ic[winners]
            best_allowances[selected] = allowance[winners]
            best_points[selected] = nearest[winners]
            best_signs[selected] = signs[winners]
        output = flat[:, 3]*0
        hit = np.flatnonzero(best_components >= 0)
        if len(hit):
            selected = torch.as_tensor(hit, device=spheres.device)
            balls = flat[chosen[selected]]
            # The nearest point is the minimizer of the same complete convex
            # distance. Its detached coordinates give the envelope derivative;
            # no differentiation through hull topology or an LP is required.
            sdf = spheres.new_tensor(best_signs[hit])*torch.linalg.vector_norm(
                balls[:, :3]-spheres.new_tensor(best_points[hit]), dim=-1)
            value = torch.relu(balls[:, 3]-sdf-spheres.new_tensor(best_allowances[hit]))
            output = output.index_copy(0, chosen[selected], value)
        self.last_report = {"complete_components": len(self.vertices), "enabled_spheres": len(supplied),
            "candidate_pairs": candidates, "raw_pair_product": len(supplied)*len(self.vertices),
            # BEGIN expose whole-query work separately from live storage.
            "region_policy_count": len(region_policies),
            "candidate_plane_products": candidate_plane_products,
            "max_candidate_plane_products": self.max_candidate_plane_products,
            # END this estimate matches the unchanged positive-policy guard.
            "maximum_live_pairs": min(candidates, live_chunk), "permitted_pairs": permitted_pairs,
            "scope": "optimization sphere guidance over complete TSDF surface and negative components"}
        return output.reshape(spheres.shape[:-1]), enabled.reshape(spheres.shape[:-1])


def make_contact_geometry(geometry):
    """Select the two actual data contracts; no ESDF occupancy fallback."""
    if any(record["kind"] == "tsdf" for record in geometry["solids"]):
        return ObservedContactGeometry(geometry)
    return ContactCellGeometry(geometry)
# END the same observed components support contact guidance and source proofs.


# BEGIN explicit body-selected coarse/fine contact geometry.
class LayeredContactCellGeometry:
    """Apply finite contact checks to the caller's already selected layer.

    The two inventories remain separate complete component unions. The
    PlanningScene supplies one boolean choice per body sphere and state;
    this adapter neither inspects region metadata nor chooses a layer from
    its cost. A contact rule must name spheres with one common choice in
    each state so its finite material region has exactly one target model.
    """

    def __init__(self, geometry):
        if set(geometry.get("layers", ())) != {"coarse", "fine"}:
            raise ValueError("Layered contact geometry needs exactly coarse and fine layers")
        # Retain the complete original record, including selection metadata,
        # for independent checking. Only the two inventories feed contact.
        self.geometry = deepcopy(geometry)
        self.layers = {name: make_contact_geometry(self.geometry["layers"][name])
                       for name in ("coarse", "fine")}
        self.last_report = None
        self.last_required_report = None

    def __copy__(self):
        """Share immutable inventories but isolate rollout diagnostics."""
        from copy import copy

        result = type(self).__new__(type(self))
        result.__dict__ = self.__dict__.copy()
        result.layers = {name: copy(layer) for name, layer in self.layers.items()}
        result.last_report = None
        result.last_required_report = None
        for layer in result.layers.values():
            layer.last_report = None
            layer.last_required_report = None
        return result

    @property
    def gaps(self):
        return [f"{name}: {gap}" for name, layer in self.layers.items() for gap in layer.gaps]

    @property
    def complete(self):
        return all(layer.complete for layer in self.layers.values())

    @property
    def pair_chunk(self):
        # Disclose both child allocations conservatively. Each child still
        # enforces its own unchanged per-query admission and live-pair caps.
        return sum(layer.pair_chunk for layer in self.layers.values())

    @property
    def max_candidate_pairs(self):
        return sum(layer.max_candidate_pairs for layer in self.layers.values())

    @property
    def cell_half_diagonal_m(self):
        # This maximum describes representation error only; it never selects
        # or combines contact geometry across the two independent layers.
        return max(layer.cell_half_diagonal_m for layer in self.layers.values())

    @property
    def layer_cell_counts(self):
        return {name: len(layer.centres) for name, layer in self.layers.items()}

    def workspace_bytes(self):
        return sum(layer.workspace_bytes() for layer in self.layers.values())

    @property
    def storage_bytes(self):
        return sum(layer.storage_bytes for layer in self.layers.values())

    @staticmethod
    def _validate_mask(fine_mask):
        if not isinstance(fine_mask, torch.Tensor) or fine_mask.dtype != torch.bool or fine_mask.ndim != 3:
            raise ValueError("Layered contact needs a boolean fine_mask with shape [B,H,S]")

    @staticmethod
    def _policy_layer(policy, fine_mask):
        """Validate rule membership before evaluating either native layer."""
        active = policy["active"]
        if (not isinstance(active, torch.Tensor) or active.dtype != torch.bool
                or active.shape != fine_mask.shape[:2] or active.device != fine_mask.device):
            raise ValueError("Layered contact policy active flags must match fine_mask states and device")
        indices = tuple(policy["sphere_indices"])
        if not indices or any(not isinstance(index, (int, np.integer)) or isinstance(index, bool)
                              or index < 0 or index >= fine_mask.shape[-1] for index in indices):
            raise ValueError("Layered contact policy needs valid nonempty sphere_indices")
        choices = fine_mask[..., list(indices)]
        # Check every state, including inactive ones: activity cannot repair
        # a rule whose members have no single geometric target inventory.
        if not bool((choices == choices[..., :1]).all()):
            raise ValueError("Layered contact policy sphere_indices select mixed coarse/fine layers")
        return choices[..., 0]

    @staticmethod
    def _masked_policy(policy, selected):
        # Keep halfspaces and feature tensors attached to their original
        # autograd graph; only per-state activity changes for the child call.
        return dict(policy, active=policy["active"] & selected)

    def query(self, spheres, policies, fine_mask):
        """Select exact native costs with the supplied [B,H,S] body mask."""
        self._validate_mask(fine_mask)
        if spheres.ndim != 4 or spheres.shape[-1] != 4 or spheres.shape[:-1] != fine_mask.shape:
            raise ValueError("Layered contact spheres must have shape [B,H,S,4] matching fine_mask")
        if spheres.device != fine_mask.device:
            raise ValueError("Layered contact spheres and fine_mask must use the same device")
        if not self.complete:
            raise ValueError("A partial target-cell inventory cannot grant layered contact permission")
        policies = tuple(policies)
        selections = [self._policy_layer(policy, fine_mask) for policy in policies]
        results = {}
        for name, selected_fine in (("coarse", False), ("fine", True)):
            masked = [self._masked_policy(policy, choice if selected_fine else ~choice)
                      for policy, choice in zip(policies, selections)]
            results[name] = self.layers[name].query(spheres, masked)
        self.last_report = {"layers": {name: deepcopy(layer.last_report) for name, layer in self.layers.items()},
            "layer_cell_counts": self.layer_cell_counts,
            "scope": "complete finite target cells in the caller-selected layer for each body"}
        return (torch.where(fine_mask, results["fine"][0], results["coarse"][0]),
                torch.where(fine_mask, results["fine"][1], results["coarse"][1]))

    def required_contact(self, policy, fine_mask):
        """Require an actual target surface in the contact rule's same layer."""
        self._validate_mask(fine_mask)
        if not self.complete:
            raise ValueError("A partial target-cell inventory cannot establish required layered contact")
        choice = self._policy_layer(policy, fine_mask)
        results = {}
        for name, selected in (("coarse", ~choice), ("fine", choice)):
            layer = self.layers[name]
            # The native inactive path returns zero before writing a report.
            # Clear that report so a previous solve cannot appear current.
            layer.last_required_report = None
            results[name] = layer.required_contact(self._masked_policy(policy, selected))
        self.last_required_report = {
            "layers": {name: deepcopy(layer.last_required_report) for name, layer in self.layers.items()},
            "layer_cell_counts": self.layer_cell_counts,
            "scope": "an actual target zero surface or native face in the caller-selected layer intersects the finite region"}
        return torch.where(choice, results["fine"], results["coarse"])
# END coarse/fine selection never depends on collision cost or contact residual.
