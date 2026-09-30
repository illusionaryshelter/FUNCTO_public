"""Check visible target geometry against an existing parallel-jaw grasp.

This tests known tool solids and measured owner points, never a transformed
demonstration object. It neither searches new EEF poses nor proves Holding.
The open approach and the terminal, fixed-EEF conditional closure have separate
evidence. Finger timing during a moving closure suffix remains unverified.
"""

# BEGIN target-observation / real-tool grasp geometry, separate from registration.
from collections.abc import Mapping

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .motion import _pose_array, _poses_array
from .observations import backproject
from .registration_validation import _DEPTH_CONTINUITY_M, _observation

_ARITHMETIC_M = 1e-12  # Floating-point equality guard, not a physical/noise margin.


def _name(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _positive(value, label):
    if (isinstance(value, (bool, np.bool_)) or not np.isscalar(value)
            or not isinstance(value, (int, float, np.number))
            or not np.isfinite(value) or value <= 0):
        raise ValueError(f"{label} must be a finite positive metric length")
    return float(value)


def _apply(points, pose):
    return points @ pose[:3, :3].T + pose[:3, 3]


# BEGIN separate measured positions from local surface/normal support.
def _local_surface_evidence(depth, mask, K, uv, xyz):
    """Estimate only from same-owner, locally continuous measured neighbors.

    A boundary does not erase its center depth. Normals use all available
    neighbors in the fixed 3x3 image neighborhood whose depth differs from the
    center by at most the existing continuity bound. A one-sided noncollinear
    patch can supply a local normal; no missing surface is completed. SVD is a
    local tangent estimate, not a claim that a curved patch is an exact plane.

    Position support has a different purpose: a center surrounded in the image
    by those neighbors has local owner/surface support for an interior-solid
    contradiction. A merely one-sided boundary or isolated ray remains weak
    position evidence, even when a normal can be estimated. Neither condition
    removes the measured point or depends on the proposed gripper pose.
    """
    normals = np.zeros_like(xyz)
    normal_valid = np.zeros(len(uv), bool)
    position_supported = np.zeros(len(uv), bool)
    offsets = np.array([(x, y) for y in (-1, 0, 1) for x in (-1, 0, 1)
                        if x or y], dtype=int)
    height, width = depth.shape
    for index, (u, v) in enumerate(uv):
        neighbor_uv = offsets + [u, v]
        inside = ((neighbor_uv[:, 0] >= 0) & (neighbor_uv[:, 0] < width)
                  & (neighbor_uv[:, 1] >= 0) & (neighbor_uv[:, 1] < height))
        neighbor_uv = neighbor_uv[inside]
        x, y = neighbor_uv.T
        values = depth[y, x]
        usable = (mask[y, x] & np.isfinite(values) & (values > 0)
                  & (np.abs(values-depth[v, u]) <= _DEPTH_CONTINUITY_M+_ARITHMETIC_M))
        neighbor_uv = neighbor_uv[usable]
        if len(neighbor_uv) < 2:
            continue
        directions = neighbor_uv - [u, v]
        angles = np.sort(np.arctan2(directions[:, 1], directions[:, 0]))
        gaps = np.diff(np.r_[angles, angles[0]+2*np.pi])
        # Strictly less than a half-circle means the center lies inside the
        # 2-D neighbor footprint. This is a combinatorial image-boundary test,
        # not a confidence score or a calibrated metric noise tolerance.
        position_supported[index] = bool(gaps.max() < np.pi-1e-12)
        patch = np.vstack((xyz[index], backproject(depth, K, neighbor_uv)))
        _, singular, vectors = np.linalg.svd(patch-patch.mean(0), full_matrices=False)
        # With a line, or a tied smallest singular direction, no unique local
        # surface normal is observable. A measured position can still remain
        # locally supported independently of that directional ambiguity.
        if singular[1] <= 1e-8 or singular[1]-singular[2] <= _ARITHMETIC_M:
            continue
        normal = vectors[-1]
        if normal @ xyz[index] > 0:
            normal = -normal
        normals[index] = normal
        normal_valid[index] = True
    return normals, normal_valid, position_supported


def _cloud(views, reference_frame):
    """Keep every finite owner depth, with separate local evidence flags.

    Whole-object registration needs conservative stable projection rays, but
    grasp geometry must not delete a measured thin surface merely because its
    neighborhood includes background. Interaction masks restrict semantic
    contact support only, never the owner geometry used for interference.
    """
    points, normals, normal_valid, position_supported = [], [], [], []
    semantic, part_evidenced, pixels, names = [], [], [], []
    owner_ids, summaries = set(), {}
    for view_id in sorted(views):
        _name(view_id, "view ID")
        view = views[view_id]
        owner_ids.add(_name(view.get("owner_id"), "owner_id"))
        if view.get("reference_frame") != reference_frame:
            raise ValueError("Every view must share the declared output frame")
        observed = _observation(view)
        pose = _pose_array(view["T_output_camera"], "T_output_camera")
        depth, mask = observed["depth"], np.asarray(view["owner_mask"])
        rows, cols = np.nonzero(mask & np.isfinite(depth) & (depth > 0))
        uv = np.column_stack((cols, rows))
        part = view.get("interaction_mask")
        if part is not None:
            part = np.asarray(part)
            if part.dtype != np.bool_ or part.shape != observed["depth"].shape:
                raise ValueError("interaction_mask must be boolean and aligned with owner depth")
        xyz = backproject(depth, observed["K"], uv)
        normal, valid_normal, supported_position = _local_surface_evidence(
            depth, mask, observed["K"], uv, xyz)
        points.append(_apply(xyz, pose))
        normals.append(normal @ pose[:3, :3].T)
        normal_valid.append(valid_normal)
        position_supported.append(supported_position)
        semantic.append(np.ones(len(uv), bool) if part is None else part[rows, cols])
        part_evidenced.append(np.full(len(uv), part is not None, bool))
        pixels.extend(uv.tolist())
        names.extend([view_id] * len(uv))
        summaries[view_id] = {"measured_owner_samples": len(uv),
                              "locally_supported_position_samples": int(supported_position.sum()),
                              "weak_position_samples": int((~supported_position).sum()),
                              "observable_normal_samples": int(valid_normal.sum()),
                              "interaction_mask_supplied": part is not None,
                              "positive_semantic_samples": int(semantic[-1].sum())}
    if len(owner_ids) != 1:
        raise ValueError("All views must be bound to the same target owner")
    return {"points": np.concatenate(points), "normals": np.concatenate(normals),
            "normal_valid": np.concatenate(normal_valid),
            "position_supported": np.concatenate(position_supported),
            "semantic": np.concatenate(semantic), "part_evidenced": np.concatenate(part_evidenced),
            "pixels": pixels, "views": names,
            "owner_id": next(iter(owner_ids)), "summaries": summaries}
# END retain thin measured geometry without inventing surface support.


def _tool(tool, current_joint_positions_m):
    """Validate the small explicit linear parallel-jaw geometry protocol."""
    opened = tool.at(tool.open_joint_positions_m)
    closed = tool.at(tool.closed_joint_positions_m)
    current = tool.at(current_joint_positions_m)
    half = tool.at((np.asarray(tool.open_joint_positions_m)
                    + np.asarray(tool.closed_joint_positions_m)) / 2)
    geometries = {}
    radius = 0.
    for geom in tool.collision_geometries:
        name = geom["geom_id"]
        if name in geometries or geom.get("role") not in {"palm", "finger", "pad"}:
            raise ValueError("Tool solids need unique IDs and explicit palm/finger/pad roles")
        vertices, planes = np.asarray(geom["vertices"], float), np.asarray(geom["halfspaces"], float)
        if (vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 4
                or planes.ndim != 2 or planes.shape[1] != 4 or len(planes) < 4
                or not np.isfinite(vertices).all() or not np.isfinite(planes).all()
                or not np.allclose(np.linalg.norm(planes[:, :3], axis=1), 1., atol=1e-8, rtol=0)):
            raise ValueError("Known convex solids need finite vertices and unit-normal halfspaces")
        geometries[name] = {**geom, "vertices": vertices, "halfspaces": planes,
                            "low": vertices.min(0), "high": vertices.max(0)}
    if {g["role"] for g in geometries.values()} != {"palm", "finger", "pad"}:
        raise ValueError("Known tool geometry must cover palm, fingers and pads")
    states = []
    for state in (opened, closed, current, half):
        if state["tool_id"] != tool.tool_id or state["tool_frame"] != tool.tool_frame:
            raise ValueError("Tool states must preserve tool identity and coordinates")
        poses = {g["geom_id"]: _pose_array(g["T_tool_geom"], "T_tool_geom") for g in state["geoms"]}
        if set(poses) != set(geometries):
            raise ValueError("Every explicit q must position the same known tool solids")
        if len(state["pads"]) != 2:
            raise ValueError("This consumer requires two explicitly opposing pad faces")
        for pad in state["pads"]:
            _pose_array(pad["T_tool_pad"], "T_tool_pad")
            size = np.asarray(pad["half_sizes_xy_m"], float)
            if size.shape != (2,) or not np.isfinite(size).all() or np.any(size <= 0):
                raise ValueError("Each pad needs its real finite contact-face extent")
            if pad["geom_id"] not in geometries or geometries[pad["geom_id"]]["role"] != "pad":
                raise ValueError("Pad faces must belong to declared pad solids")
        radius = max(radius, *(float(np.linalg.norm(_apply(geometries[g]["vertices"], pose), axis=1).max())
                              for g, pose in poses.items()))
        states.append(poses)
    for name in geometries:
        if (not all(np.allclose(states[0][name][:3, :3], row[name][:3, :3], atol=1e-9, rtol=0)
                    for row in states[1:])
                or not np.allclose(states[3][name][:3, 3],
                                   (states[0][name][:3, 3]+states[1][name][:3, 3])/2,
                                   atol=1e-9, rtol=0)):
            raise ValueError("Conditional closure requires the declared linear tool motion")
    return geometries, opened, closed, current, radius


def _witness(cloud, index):
    return {"sample_index": int(index), "view": cloud["views"][index],
            "pixel_uv": cloud["pixels"][index], "point_output": cloud["points"][index],
            "locally_supported_position": bool(cloud["position_supported"][index]),
            "observable_normal": bool(cloud["normal_valid"][index])}


def _interference(pose, state, geometries, cloud, tree, radius, penetration):
    """Report measured samples strictly inside a tool solid with their support.

    Convex-hull halfspaces are supplied by the existing tool adapter. Their
    maximum normalized plane value tests deep interior membership. We do not
    treat that value as the Euclidean signed distance outside a polyhedron.
    Locally supported positions establish a contradiction; boundary/isolated
    positions establish only a possible conflict. A normal is not required to
    check a measured position inside a solid. The penetration band is separate
    from the positive contact proximity.
    """
    indices = np.asarray(tree.query_ball_point(pose[:3, 3], radius, return_sorted=True), dtype=int)
    if not len(indices):
        return []
    local = _apply(cloud["points"][indices], np.linalg.inv(pose))
    found = []
    for row in state["geoms"]:
        geom = geometries[row["geom_id"]]
        xyz = _apply(local, np.linalg.inv(row["T_tool_geom"]))
        inside_box = np.all((xyz >= geom["low"]) & (xyz <= geom["high"]), axis=1)
        selected = np.flatnonzero(inside_box)
        if not len(selected):
            continue
        planes = geom["halfspaces"]
        interior = -(xyz[selected] @ planes[:, :3].T + planes[:, 3]).max(axis=1)
        bad = np.flatnonzero(interior > penetration + _ARITHMETIC_M)
        if len(bad):
            supported = cloud["position_supported"][indices[selected[bad]]]
            for reliable, status in ((True, "rejected"), (False, "unresolved")):
                group = bad[supported == reliable]
                if not len(group):
                    continue
                worst = group[np.argmax(interior[group])]
                found.append({"geom_id": geom["geom_id"], "role": geom["role"],
                              "evidence_status": status,
                              "observed_conflict_points": len(group),
                              "deepest_interior_margin_m": float(interior[worst]),
                              "witness": _witness(cloud, indices[selected[worst]])})
    return found


def _record_interference(report, record):
    """Retain weak witnesses without stopping a search for real contradictions.

    One first weak record per phase is sufficient to preserve its uncertainty;
    keep a later reliable contradiction as well. This bounds the report size
    while the geometry loop still checks every subsequent sampled tool state.
    """
    if not record["solids"]:
        return False
    contradicted = any(row["evidence_status"] == "rejected" for row in record["solids"])
    if contradicted or not any(row["phase"] == record["phase"] for row in report["interference"]):
        report["interference"].append(record)
    return contradicted


def _approach_samples(poses, last_index, radius, spacing):
    """Sample, without changing the trajectory, at a declared tool-motion step.

    Translation plus radius times rotation angle bounds vertex motion on one
    interpolation interval. This is a discretely sampled check, not continuous
    collision certification or a new execution-time/finger controller.
    """
    yield 0, poses[0]
    for index in range(last_index):
        first, second = poses[index:index+2]
        rotvec = Rotation.from_matrix(first[:3, :3].T @ second[:3, :3]).as_rotvec()
        bound = np.linalg.norm(second[:3, 3]-first[:3, 3])+radius*np.linalg.norm(rotvec)
        count = max(1, int(np.ceil(bound/spacing)))
        if count > 100000:
            raise ValueError("A grasp approach interval exceeds the CPU geometry check scope")
        for fraction in np.arange(1, count+1)/count:
            pose = np.eye(4)
            pose[:3, :3] = first[:3, :3] @ Rotation.from_rotvec(fraction*rotvec).as_matrix()
            pose[:3, 3] = (1-fraction)*first[:3, 3]+fraction*second[:3, 3]
            yield index+1, pose


def _closure(pose, opened, closed, cloud, contact, penetration):
    """Intersect observed opposing-face contact intervals along native closure.

    With fixed EEF, a pad plane moves c(a)=c0+a*v. For an observed point p,
    signed face distance is n.(p-c0)-a*n.v. Requiring this to lie between
    -penetration and +contact gives a closed interval in a. Earliest observed
    target surfaces bound each side; incompatible intervals require unmodelled
    object recentering and remain unresolved, never invented force dynamics.
    """
    local = _apply(cloud["points"], np.linalg.inv(pose))
    normals = cloud["normals"] @ pose[:3, :3]
    intervals, sides, travels = [], [], []
    for first, last in zip(opened["pads"], closed["pads"]):
        start, end = np.asarray(first["T_tool_pad"]), np.asarray(last["T_tool_pad"])
        if (first["geom_id"] != last["geom_id"]
                or not np.allclose(start[:3, :3], end[:3, :3], atol=1e-9, rtol=0)):
            raise ValueError("The same material pad faces must translate through closure")
        normal = start[:3, 2]
        travel = float(normal @ (end[:3, 3]-start[:3, 3]))
        if (travel <= 0 or not np.allclose(end[:3, 3]-start[:3, 3], normal*travel, atol=1e-9, rtol=0)):
            raise ValueError("Pad closure must move along its inward normal")
        coords = _apply(local, np.linalg.inv(start))
        tangential = np.all(np.abs(coords[:, :2]) <= np.asarray(first["half_sizes_xy_m"])+1e-12, axis=1)
        swept = (tangential & (coords[:, 2] >= -penetration-_ARITHMETIC_M)
                 & (coords[:, 2] <= travel+contact+_ARITHMETIC_M))
        indices = np.flatnonzero(swept)
        if not len(indices):
            return {"status": "unresolved", "reason": "No observed target surface in both pad closure sweeps",
                    "sides": sides, "closure_fraction": None}
        first_fraction = float(coords[indices, 2].min()/travel)
        interval = [max(0., first_fraction-contact/travel), min(1., first_fraction+penetration/travel)]
        intervals.append(interval)
        travels.append(travel)
        sides.append({"pad_index": int(first["pad_index"]), "geom_id": first["geom_id"],
                      "earliest_observed_fraction": first_fraction,
                      "allowed_fraction_interval": interval,
                      "swept_observed_points": len(indices)})
    if not np.allclose(np.asarray(opened["pads"][0]["T_tool_pad"])[:3, 2],
                       -np.asarray(opened["pads"][1]["T_tool_pad"])[:3, 2], atol=1e-9, rtol=0):
        raise ValueError("Two material pad normals must oppose each other")
    lower, upper = max(row[0] for row in intervals), min(row[1] for row in intervals)
    # Re-expression in another world gauge must not turn touching closed
    # physical intervals into a 1e-15-wide artificial gap. Express this solely
    # arithmetic tolerance in closure-fraction coordinates and snap a touching
    # interval to one point; the declared metric contact limits are unchanged.
    if lower > upper + _ARITHMETIC_M/min(travels):
        return {"status": "unresolved", "reason": "Observed bilateral closure requires unmodelled target recentering",
                "sides": sides, "closure_fraction": None}
    if lower > upper:
        lower = upper = (lower+upper)/2
    fraction = float(np.clip(min(row["earliest_observed_fraction"] for row in sides), lower, upper))
    gap = (1-fraction)*opened["signed_gap_m"]+fraction*closed["signed_gap_m"]
    supported, witness_sets = True, []
    witness_indices = []
    for side, first, last in zip(sides, opened["pads"], closed["pads"]):
        pad = np.asarray(first["T_tool_pad"]).copy()
        pad[:3, 3] = (1-fraction)*pad[:3, 3]+fraction*np.asarray(last["T_tool_pad"])[:3, 3]
        coordinates = _apply(local, np.linalg.inv(pad))
        selected = (np.all(np.abs(coordinates[:, :2]) <= np.asarray(first["half_sizes_xy_m"])+1e-12, axis=1)
                    & (coordinates[:, 2] >= -penetration-1e-12)
                    & (coordinates[:, 2] <= contact+1e-12)
                    & cloud["normal_valid"]
                    & ((normals @ pad[:3, 2]) < -1e-6) & cloud["semantic"])
        ids = np.flatnonzero(selected)
        # BEGIN a locally observed surface may touch a pad at a point or line.
        # Its normal already came from a noncollinear measured neighborhood.
        # Requiring that whole neighborhood to lie inside the contact band
        # would introduce an unjustified planar-area contact requirement.
        supported &= len(ids) > 0
        witness_indices.append(ids)
        witness_sets.append(set(map(int, ids)))
        side.update(positive_contact_samples=len(ids),
                    samples_with_part_mask_evidence=int(cloud["part_evidenced"][ids].sum()),
                    samples_relying_on_upstream_functional_semantics=int((~cloud["part_evidenced"][ids]).sum()),
                    contact_witnesses=[_witness(cloud, index) for index in ids[:8]])
    # A positive pad gap alone is insufficient: contact tolerance can leave
    # both pads near the same off-center zero-thickness plane. For left/right
    # material witnesses pL,pR, actual axial separation is n.(pR-pL), not pad
    # gap. Require an ordered pair with positive separation using only the
    # arithmetic zero guard; no minimum object thickness of 2*contact is added.
    separation, opposing_pair = None, None
    if all(len(ids) for ids in witness_indices):
        axis = np.asarray(opened["pads"][0]["T_tool_pad"])[:3, 2]
        left = witness_indices[0][np.argmin(local[witness_indices[0]] @ axis)]
        right = witness_indices[1][np.argmax(local[witness_indices[1]] @ axis)]
        separation = float((local[right]-local[left]) @ axis)
        opposing_pair = [_witness(cloud, left), _witness(cloud, right)]
    independent = (gap > _ARITHMETIC_M and not bool(witness_sets[0] & witness_sets[1])
                   and separation is not None and separation > _ARITHMETIC_M)
    supported &= independent
    return {"status": "supported" if supported else "unresolved",
            "reason": ("Independent opposing surface contacts within declared proximity and tool limits" if supported
                       else "Insufficient independent bilateral surface/normal/semantic evidence"),
            "closure_fraction": fraction, "allowed_fraction_interval": [lower, upper],
            "signed_gap_m": float(gap), "independent_opposing_witnesses": bool(independent), "sides": sides,
            "opposing_witness_separation_m": separation, "opposing_witness_pair": opposing_pair,
            "force_closure_proven": False, "contact_or_holding_measured": False}
    # END point/line contact support; no invented finite surface-contact area.


def screen_grasp_candidates(candidates, views, tool, *, reference_frame,
                            current_eef_pose, current_joint_positions_m,
                            contact_tolerance_m, penetration_tolerance_m):
    """Report visible grasp geometry while retaining every supplied candidate.

    candidates maps IDs to existing grasp consumer results. views maps IDs to
    owner_id/reference_frame/depth/K/owner_mask/T_output_camera, optionally an
    interaction_mask. Missing part masks retain upstream functional semantics;
    no precise current part segmentation or fixed semantic point count is
    required. The source/current object pose and object mesh are never inputs.

    current_joint_positions_m is measured robot feedback, not a source closing
    width. The approach's full-open state is an explicit geometric plan; its
    initial opening motion is checked separately. Terminal conditional closure
    uses mechanical joint limits and observed surfaces. This is not a measured
    finger trajectory, force-closure test or whole-arm/environment certificate.

    `supported` means opposing visible contact geometry and no target surface
    interference in the stated sampled scope. Missing evidence is unresolved.
    Inputs are action alternatives for an already selected owner and function,
    not mutually exclusive scene-identity hypotheses. An unverified alternative
    does not invalidate a supported action. Select the sole supported action,
    leave multiple supported actions ambiguous, and retain every candidate.
    Target/semantic ambiguity belongs upstream; this module never ranks costs.
    """
    _name(reference_frame, "reference_frame")
    if not isinstance(candidates, Mapping) or not candidates:
        raise ValueError("A nonempty mapping of already generated grasp candidates is required")
    if not isinstance(views, Mapping) or not views:
        raise ValueError("At least one actual target RGB-D view is required")
    contact = _positive(contact_tolerance_m, "contact_tolerance_m")
    penetration = _positive(penetration_tolerance_m, "penetration_tolerance_m")
    current_pose = _pose_array(current_eef_pose, "current_eef_pose")
    geometry, opened, closed, measured, radius = _tool(tool, current_joint_positions_m)
    cloud = _cloud(views, reference_frame)
    tree = cKDTree(cloud["points"])
    result = {"status": "unresolved", "selected_candidate_id": None,
              "reference_frame": reference_frame, "owner_id": cloud["owner_id"],
              "contact_tolerance_m": contact, "penetration_tolerance_m": penetration,
              "observations": cloud["summaries"], "candidates": {},
              "eligible_candidate_ids": [], "supported_candidate_ids": [],
              "rejected_candidate_ids": [], "unresolved_candidate_ids": [],
              "scope": "sampled open target-local approach and terminal conditional visible closure geometry",
              "moving_closure_finger_timing_verified": False, "full_robot_collision_verified": False,
              "actual_holding_verified": False, "cost_ranking_used": False}
    for name in sorted(candidates):
        _name(name, "candidate ID")
        candidate = candidates[name]
        poses = _poses_array(candidate["eef_poses"], "Candidate EEF poses")
        frames = np.asarray(candidate["frame_indices"])
        if (candidate.get("reference_frame") != reference_frame
                or candidate.get("tool_id") != tool.tool_id or candidate.get("tool_frame") != tool.tool_frame
                or frames.shape != (len(poses),) or frames.dtype.kind not in "iu"
                or np.any(np.diff(frames) <= 0)):
            raise ValueError("Candidate poses, source frame IDs and tool conventions must agree")
        closing = [event["frame_index"] for event in candidate["gripper_events"] if event["command"] == "close"]
        if not closing or closing[0] not in frames:
            raise ValueError("A complete grasp candidate must preserve its original closing intention")
        close_index = int(np.flatnonzero(frames == closing[0])[0])
        report = {"status": "unresolved", "interference": [], "approach_samples": 0,
                  "supported_scope": result["scope"], "moving_closure_suffix_verified": False,
                  "part_semantics_scope": "Optional visible part masks or upstream functional binding; owner geometry alone does not prove the contacted part",
                  "source_width_reused": False, "actual_holding_verified": False,
                  "current_joint_positions_m": np.asarray(measured["joint_positions_m"]).copy(),
                  "planned_approach_joint_positions_m": np.asarray(opened["joint_positions_m"]).copy(),
                  "approach_tool_motion_sample_step_m": contact}
        result["candidates"][name] = report
        opening_declared = any(event["command"] == "open" and event["frame_index"] <= closing[0]
                               for event in candidate["gripper_events"])
        already_open = np.allclose(measured["joint_positions_m"], opened["joint_positions_m"],
                                   atol=_ARITHMETIC_M, rtol=0)
        if candidate.get("connection_required") or not np.allclose(poses[0], current_pose, atol=1e-8, rtol=0):
            report["reason"] = "Candidate lacks its verified connection from current robot feedback"
        elif not opening_declared and not already_open:
            report["reason"] = "No opening intention or measured open state supports the planned approach"
        elif not len(cloud["points"]):
            report["reason"] = "No valid current owner depth samples"
        else:
            q0 = np.asarray(measured["joint_positions_m"])
            qopen = np.asarray(opened["joint_positions_m"])
            opening_steps = max(1, int(np.ceil(np.abs(qopen-q0).max()/contact)))
            for alpha in np.linspace(0., 1., opening_steps+1):
                state = tool.at((1-alpha)*q0+alpha*qopen)
                found = _interference(current_pose, state, geometry, cloud, tree, radius, penetration)
                if _record_interference(report, {"phase": "initial_opening",
                        "opening_fraction": float(alpha), "solids": found}):
                    break
            if not any(solid["evidence_status"] == "rejected"
                       for record in report["interference"] for solid in record["solids"]):
                for index, pose in _approach_samples(poses, close_index, radius, contact):
                    report["approach_samples"] += 1
                    found = _interference(pose, opened, geometry, cloud, tree, radius, penetration)
                    if _record_interference(report, {"phase": "planned_open_approach",
                            "frame_index": int(frames[index]), "solids": found}):
                        break
            closure = _closure(poses[-1], opened, closed, cloud, contact, penetration)
            report["terminal_closure"] = closure
            # The palm does not translate with finger closure. A witnessed
            # palm conflict is valid even when contact width is unresolved.
            palm = [row for row in _interference(poses[-1], opened, geometry, cloud, tree, radius, penetration)
                    if row["role"] == "palm"]
            _record_interference(report, {"phase": "terminal_palm", "solids": palm})
            if closure["closure_fraction"] is not None:
                alpha = closure["closure_fraction"]
                joint = (1-alpha)*qopen+alpha*np.asarray(closed["joint_positions_m"])
                closure["conditional_joint_positions_m"] = joint
                # Check every sampled native closure configuration, not just
                # its endpoint: an intermediate finger solid may intersect a
                # surface that is no longer inside it at the final opening.
                steps = max(1, int(np.ceil(np.abs(joint-qopen).max()/min(contact, penetration))))
                closure["conditional_closure_samples"] = 0
                for fraction in np.linspace(0., 1., steps+1):
                    state = tool.at((1-fraction)*qopen+fraction*joint)
                    closure["conditional_closure_samples"] += 1
                    found = _interference(poses[-1], state, geometry, cloud, tree, radius, penetration)
                    if _record_interference(report, {"phase": "terminal_conditional_closure",
                            "closure_fraction": float(fraction*alpha), "solids": found}):
                        break
            # BEGIN weak measured boundary evidence remains unknown, never a
            # hard contradiction or an excuse to stop checking later poses.
            terminal_conflicts = any(row["phase"].startswith("terminal")
                                     and any(s["evidence_status"] == "rejected" for s in row["solids"])
                                     for row in report["interference"])
            approach_conflicts = any(not row["phase"].startswith("terminal")
                                     and any(s["evidence_status"] == "rejected" for s in row["solids"])
                                     for row in report["interference"])
            terminal_unknown = any(row["phase"].startswith("terminal")
                                   and any(s["evidence_status"] == "unresolved" for s in row["solids"])
                                   for row in report["interference"])
            approach_unknown = any(not row["phase"].startswith("terminal")
                                   and any(s["evidence_status"] == "unresolved" for s in row["solids"])
                                   for row in report["interference"])
            report["approach_requires_replanning"] = approach_conflicts
            report["approach_visible_geometry_status"] = ("rejected" if approach_conflicts else
                                                         "unresolved" if approach_unknown else "supported")
            report["terminal_visible_geometry_status"] = ("rejected" if terminal_conflicts else
                                                         "unresolved" if terminal_unknown else closure["status"])
            if terminal_conflicts or approach_conflicts:
                report.update(status="rejected", reason=("Terminal conditional tool geometry penetrates observed target"
                              if terminal_conflicts else "This reference approach needs replanning; terminal grasp status is reported separately"))
            elif terminal_unknown or approach_unknown:
                report.update(status="unresolved", reason="Locally unsupported boundary positions indicate possible tool interference")
            else:
                report.update(status=closure["status"], reason=closure["reason"])
            # END distinct position evidence; normals do not gate solid tests.
        result[report["status"]+"_candidate_ids"].append(name)
    result["eligible_candidate_ids"] = list(result["supported_candidate_ids"])
    eligible, unknown = result["eligible_candidate_ids"], result["unresolved_candidate_ids"]
    # BEGIN action alternatives do not compete as exclusive scene hypotheses.
    if len(eligible) == 1:
        result.update(status="selected", selected_candidate_id=eligible[0])
    elif eligible:
        result["status"] = "ambiguous"
    elif unknown:
        result["status"] = "unresolved"
    else:
        result["status"] = "rejected"
    # END select from supported actions without inventing a cost preference.
    return result
# END visible grasp geometry; no body registration, new grasp search or execution claim.
