"""Finite contact intersections over the already bound target solid model.

The ordinary scene query remains responsible for every nonmaterial pair.
These helpers localize material permission to complete occupied cells or
explicit cuboids; no task identity or live object truth enters this adapter.
"""

# BEGIN bounded finite-cell contact witnesses and exact box distance gradients.
from copy import deepcopy
import math

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import torch


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


class ContactCellGeometry:
    """A complete finite box union with a bounded exact spatial broad phase.

    A KD-tree removes only cells proved too far from the queried sphere by
    the enclosing cell ball. Candidate count is obtained BEFORE allocating
    neighbor lists. At most two million candidates and 32768 live pairs enter
    a query; exceeding either supported model/query budget raises explicitly.
    The box union is the declared observed field representation, not a claim
    about hidden geometry or the continuous ESDF interpolation surface.
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
            elif record["kind"] == "voxel":
                shape = np.asarray(record["shape_xyz"], int)
                size = float(record["voxel_size"])
                indices = np.asarray(record["occupied_flat_indices"], np.int64)
                dtype = np.dtype(record["field_dtype"])
                if (shape.shape != (3,) or np.any(shape <= 0) or np.prod(shape) > 8_000_000
                        or not np.isfinite(size) or size <= 0 or indices.ndim != 1
                        or len(indices) > self.max_cells or dtype.kind != "f" or dtype.itemsize not in (2, 4, 8)):
                    raise ValueError("Contact target voxels need a complete bounded field inventory")
                field = np.frombuffer(bytes.fromhex(record["field_bytes_hex"]), dtype=dtype)
                if (field.size != np.prod(shape) or not np.isfinite(field).all()
                        # BEGIN negative interiors and exact zero surfaces follow native ESDF.
                        or record["occupancy_rule"] != "ESDF_sample_nonpositive"
                        or not np.array_equal(indices, np.flatnonzero(field <= 0))):
                        # END positive samples, including unknown sentinels, grant no solid.
                    raise ValueError("Contact occupied cells differ from the original bound field")
                local = (np.column_stack(np.unravel_index(indices, tuple(shape)))+.5-shape/2)*size
                half = np.full((len(indices), 3), size/2)
                self.cell_half_diagonal_m = max(self.cell_half_diagonal_m, math.sqrt(3)*size/2)
            else:
                self.gaps.append(f"{record['name']}: finite contact intersections require complete box cells")
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

    @property
    def complete(self):
        return not self.gaps

    def workspace_bytes(self):
        # GPU cell tensors are only gathered for one live pair chunk. CPU
        # KD-tree/list storage has its separately disclosed candidate bound.
        return int(self.pair_chunk*1600 + self.centres.nbytes+self.halves.nbytes+self.rotations.nbytes)

    def required_contact(self, policy):
        """Require some COMPLETE target cell in the finite material region.

        The material extrusion's enclosing ball gives a complete candidate
        set. Every surviving cell contributes all six native halfspaces to a
        HiGHS intersection solve. A feasible abstract patch without occupied
        target material therefore cannot satisfy required contact. This is
        geometric/model contact, still independent of measured contact force.
        """
        active = policy["active"].reshape(-1)
        shape = policy["active"].shape
        selected = torch.nonzero(active, as_tuple=False).flatten()
        points = policy["feature_points"].reshape(len(active), -1, 3)
        centre = points.mean(-2)
        radius = torch.linalg.vector_norm(points-centre[:, None], dim=-1).amax(-1)+policy["allowance"]
        output = radius*0
        if not len(selected):
            return output.reshape(shape)
        if not len(self.centres):
            self.last_required_report = {"active_states": len(selected), "candidate_cells": 0,
                                         "target_inventory_empty": True}
            return output.index_fill(0, selected, 1.).reshape(shape)
        rows = selected.detach().cpu().numpy()
        centres = centre[selected].detach().cpu().numpy()
        radii = np.nextafter(radius[selected].detach().cpu().numpy()+self.maximum_cell_radius, np.inf)
        lengths = self.tree.query_ball_point(centres, radii, return_length=True, workers=1)
        pairs = int(np.sum(lengths))
        if pairs > 4096:
            raise ValueError(f"Required complete target contact needs {pairs} cell LPs, budget 4096")
        neighbors = self.tree.query_ball_point(centres, radii, workers=1)
        missing = np.flatnonzero(lengths == 0)
        if len(missing):
            _, nearest = self.tree.query(centres[missing], workers=1)
            indices = selected[torch.as_tensor(missing, device=selected.device)]
            # This strictly positive distance bound is valid because the
            # complete enclosing-ball search found no possible intersection.
            distance = torch.linalg.vector_norm(centre[indices]-centre.new_tensor(self.centres[nearest]), dim=-1)
            value = torch.relu(distance-self.maximum_cell_radius-radius[indices])
            output = output.index_copy(0, indices, value)
        if pairs:
            state_rows = np.repeat(rows, lengths)
            cell_rows = np.concatenate(neighbors).astype(np.int64)
            states = torch.as_tensor(state_rows, device=active.device)
            rotation = centre.new_tensor(self.rotations[cell_rows])
            half = centre.new_tensor(self.halves[cell_rows])
            cell_centre = centre.new_tensor(self.centres[cell_rows])
            axes = rotation.transpose(-1, -2)
            box_normals = torch.cat((axes, -axes), dim=-2)
            box_bounds = torch.cat(((axes*cell_centre[:, None]).sum(-1)+half,
                                    (-axes*cell_centre[:, None]).sum(-1)+half), dim=-1)
            region_normals = policy["normals"].reshape(len(active), -1, 3)[states]
            region_bounds = policy["bounds"].reshape(len(active), -1)[states]
            slack = region_slack(torch.cat((region_normals, box_normals), dim=-2),
                                 torch.cat((region_bounds, box_bounds), dim=-1))
            minima = torch.full_like(output, torch.inf).scatter_reduce(0, states, slack, reduce="amin", include_self=True)
            covered = selected[torch.as_tensor(np.flatnonzero(lengths > 0), device=selected.device)]
            output = output.index_copy(0, covered, minima[covered])
        self.last_required_report = {"active_states": len(selected), "candidate_cells": pairs,
            "maximum_cell_lps": 4096, "target_inventory_empty": False,
            "scope": "at least one complete target cell intersects the finite material/patch/gap region"}
        return output.reshape(shape)

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
