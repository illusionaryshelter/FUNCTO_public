"""Export an explicitly named native robot into a cuRobo planning fixture.

The input is an expanded, fixed-base MuJoCo subtree with scalar joints,
box/convex-mesh arm solids and separately retained fixed box/cylinder
installation primitives. Native names, axes, signs and transforms are retained.
``yourdfpy`` owns URDF serialization and independent FK; no planner or live
simulator scene state is admitted to this CPU-only model export.
"""

# BEGIN native robot export and continuous-volume sphere-cover certificates.
from __future__ import annotations

import argparse
from itertools import product
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import h5py
from mpmath.ctx_iv import MPIntervalContext
import mujoco
import numpy as np
from scipy.cluster.vq import kmeans2
from scipy.spatial import ConvexHull, HalfspaceIntersection
from scipy.spatial.transform import Rotation
from scipy.stats import qmc
import trimesh
from yourdfpy import URDF, Robot, Link, Joint, Limit, Inertial, Geometry, Mesh, Box, Cylinder, Collision
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from manipulation.tool_geometry import _collision_geometry
from check_recorded_robot_binding import _robot_model


def _pose(position, quaternion):
    value = np.eye(4)
    value[:3, :3] = Rotation.from_quat(np.asarray(quaternion)[[1, 2, 3, 0]]).as_matrix()
    value[:3, 3] = position
    return value


def _pose_dict(transform):
    return {"position": transform[:3, 3].tolist(), "quaternion_wxyz":
            Rotation.from_matrix(transform[:3, :3]).as_quat()[[3, 0, 1, 2]].tolist()}


# BEGIN preserve native geom quaternion scale in complete-volume covers.
def _native_geom_pose(model, geom_id):
    """Use MuJoCo's actual homogeneous geom matrix, not SciPy normalization."""
    position, quaternion = model.geom_pos[geom_id], model.geom_quat[geom_id]
    if not np.isfinite(position).all() or not np.isfinite(quaternion).all():
        raise ValueError("Native collision geom poses must be finite")
    result, rotation = np.eye(4), np.empty(9)
    mujoco.mju_quat2Mat(rotation, quaternion)
    result[:3, :3], result[:3, 3] = rotation.reshape(3, 3), position
    return result
# END preserve native geom quaternion scale in complete-volume covers.


def _cover_cells(vertices, *, count=None, box_halfsize=None, subdivisions=None):
    """Partition the complete native solid, rather than sampled surfaces.

    Mesh partitions are exact intersections of convex-hull halfspaces with
    Voronoi halfspaces. Sobol points only choose sites; they are not coverage
    evidence. A ball enclosing every vertex encloses its entire convex cell.
    Box cells use an analytic Cartesian partition with the same certificate.
    """
    if box_halfsize is not None:
        bins = np.asarray(subdivisions, int)
        if bins.shape != (3,) or np.any(bins < 1):
            raise ValueError("Box cover requires three positive subdivision counts")
        step = 2 * box_halfsize / bins
        offsets = np.asarray(list(product((-.5, .5), repeat=3))) * step
        return [(-box_halfsize + (np.asarray(index) + .5) * step) + offsets
                for index in product(*(range(value) for value in bins))], None
    hull = ConvexHull(vertices)
    planes = hull.equations
    points = qmc.Sobol(3, scramble=False).random_base2(12)
    points = vertices.min(0) + points * np.ptp(vertices, axis=0)
    points = points[np.all(points @ planes[:, :3].T + planes[:, 3] < -1e-9, axis=1)]
    if count is None or not 1 <= count <= min(128, len(points)):
        raise ValueError("Mesh cover requires a bounded positive cell count")
    sites, _ = kmeans2(points, count, iter=15, minit="++", seed=10)
    cells = []
    for index, site in enumerate(sites):
        others = np.delete(sites, index, axis=0)
        normals = others - site
        offsets = (np.einsum("ij,ij->i", others, others) - site @ site) / -2
        distances = np.linalg.norm(normals, axis=1)
        if np.any(distances < 1e-12):
            raise ValueError("Degenerate Voronoi sites cannot certify a cover")
        cuts = np.column_stack((normals, offsets)) / distances[:, None]
        cells.append(HalfspaceIntersection(np.concatenate((planes, cuts)), site).intersections)
    return cells, sites


def _geom_cover(model, geom_id, specification, destination):
    geometry = _collision_geometry(model, geom_id, "native_collision_proxy")
    # BEGIN preserve the native geom transform and reproducible partition input.
    local = _native_geom_pose(model, geom_id)
    is_box = model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_BOX
    if set(specification) != ({"subdivisions"} if is_box else {"cells"}):
        raise ValueError("Each supported native geom needs exactly its existing partition input")
    # END preserve the native geom transform and reproducible partition input.
    cells, sites = _cover_cells(geometry["vertices"],
        count=specification.get("cells"),
        box_halfsize=model.geom_size[geom_id] if is_box else None,
        subdivisions=specification.get("subdivisions"))
    spheres, proof_vertices, offsets, margins = [], [], [0], []
    for cell in cells:
        body_vertices = cell @ local[:3, :3].T + local[:3, 3]
        center, _ = trimesh.nsphere.minimum_nsphere(body_vertices)
        # cuRobo consumes float32. Recompute and round the radius outward AFTER
        # rounding the serialized center, retaining a 1 micrometre allowance.
        center = np.asarray(center, dtype=np.float32).astype(float)
        distances = np.linalg.norm(body_vertices - center, axis=1)
        radius = float(np.nextafter(np.float32(distances.max() + 1e-6), np.float32(np.inf)))
        spheres.append({"center": center.tolist(), "radius": radius})
        proof_vertices.extend(body_vertices)
        offsets.append(len(proof_vertices))
        margins.append(radius - distances.max())
    centers = np.asarray([sphere["center"] for sphere in spheres])
    radii = np.asarray([sphere["radius"] for sphere in spheres])
    planes = geometry["halfspaces"]
    # BEGIN evaluate metric body-space support planes under the native matrix.
    normals_body = np.linalg.solve(local[:3, :3].T, planes[:, :3].T).T
    normal_lengths = np.linalg.norm(normals_body, axis=1)
    offsets_body = planes[:, 3]-normals_body @ local[:3, 3]
    excess = (centers @ normals_body.T + offsets_body) / normal_lengths + radii[:, None]
    # END evaluate metric body-space support planes under the native matrix.
    np.savez_compressed(destination, native_vertices=geometry["vertices"],
                        native_faces=geometry["faces"], native_halfspaces=planes,
                        T_body_geom=local, cell_vertices_body=np.asarray(proof_vertices),
                        cell_vertex_offsets=np.asarray(offsets), sphere_centers_body=centers,
                        sphere_radii=radii, cell_enclosure_margin=np.asarray(margins),
                        partition_sites_geom=np.empty((0, 3)) if sites is None else sites)
    record = {"geom": model.geom(geom_id).name, "link": model.body(model.geom_bodyid[geom_id]).name,
              "representation": geometry["collision_representation"], "T_body_geom": local.tolist(),
              "certificate_path": str(destination), "sphere_count": len(spheres),
              # BEGIN record the exact current partition input for independent reload.
              "cover_specification": specification,
              # END record the exact current partition input for independent reload.
              "minimum_cell_enclosure_margin_m": float(min(margins)),
              "maximum_support_plane_excess_m": float(excess.max()),
              "source_contype": int(model.geom_contype[geom_id]),
              "source_conaffinity": int(model.geom_conaffinity[geom_id]),
              "scope": "collision proxy only; not RGB-D occupied-solid evidence"}
    return spheres, record, geometry


def _native_ignored_pairs(model, collision_links):
    """Keep exactly the native welded/parent and collision-mask exclusions.

    A link pair is ignored only when all its native geom pairs are excluded.
    Coarse proxy overlap is not a reason to remove an enabled collision pair.
    """
    result = {}
    filter_parent = not (model.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_FILTERPARENT))
    for i, name in enumerate(collision_links):
        body = model.body(name).id
        first = np.flatnonzero(model.geom_bodyid == body)
        for other_name in collision_links[i+1:]:
            other = model.body(other_name).id
            second = np.flatnonzero(model.geom_bodyid == other)
            weld_a, weld_b = int(model.body_weldid[body]), int(model.body_weldid[other])
            parent_a = int(model.body_weldid[model.body_parentid[weld_a]])
            parent_b = int(model.body_weldid[model.body_parentid[weld_b]])
            welded = weld_a == weld_b
            parent = filter_parent and weld_a != 0 and weld_b != 0 and (parent_a == weld_b or parent_b == weld_a)
            masked = all(not ((model.geom_contype[a] & model.geom_conaffinity[b]) or
                              (model.geom_contype[b] & model.geom_conaffinity[a]))
                         for a in first for b in second)
            if welded or parent or masked:
                result.setdefault(name, []).append(other_name)
    return result


# BEGIN native solid separation certificates, independent of sphere fitting.
def _interval_bounds(value):
    """Serialize an interval without rounding either endpoint inward.

    mpmath owns all interval arithmetic. The single nextafter at this output
    boundary only encloses conversion of each arbitrary-precision endpoint
    to the binary64 number which JSON can round-trip exactly.
    """
    bounds = [float(np.nextafter(float(value.a), -np.inf)),
              float(np.nextafter(float(value.b), np.inf))]
    if not np.isfinite(bounds).all():
        raise ValueError("A finite collision certificate cannot contain infinite bounds")
    return bounds


def _interval_rotation(context, quaternion, *, normalize):
    """Enclose the native body/geom quaternion conventions using iv.

    Compiled floating values are the inputs, not decimal reconstructions.
    Native body FK normalizes its accumulated quaternion. Native geom FK does
    NOT normalize the geom quaternion again: its homogeneous quadratic matrix
    retains the compiled norm residual. Applying normalization to a geom here
    would silently change its support points at large geometry scales.
    C(q_body_unit*q_geom) = R_body*C(q_geom), so the native geom matrix can be
    projected locally without assuming that it is an exact unit rotation.
    """
    values = np.asarray(quaternion, dtype=float)
    if values.shape != (4,) or not np.isfinite(values).all():
        raise ValueError("Compiled collision quaternions must be finite four-vectors")
    q = [context.mpf(float(value)) for value in values]
    norm = context.sqrt(sum(value * value for value in q))
    if not (norm.a > 0 and abs(norm - 1).b <= 64 * np.finfo(float).eps):
        raise ValueError("Compiled collision quaternions must be normalized")
    w, x, y, z = [value / norm for value in q] if normalize else q
    return [[w*w+x*x-y*y-z*z, 2*(x*y-z*w), 2*(x*z+y*w)],
            [2*(x*y+z*w), w*w-x*x+y*y-z*z, 2*(y*z-x*w)],
            [2*(x*z-y*w), 2*(y*z+x*w), w*w-x*x-y*y+z*z]]


def _hinge_separation_certificate(model, parent, child):
    """Prove separation of complete native solids for every hinge angle.

    Each mesh is its original native convex hull; a linear projection of that
    hull is bounded by projections of ALL compiled vertices, including a
    nonconvex authored mesh's vertices. Boxes use their existing eight-corner
    geometry. Hinge rotation preserves axial projections, so this argument
    contains no angle samples and is unaffected by ancestor/descendant motion.
    This exporter still only admits an origin-centred, single scalar joint.
    Unknown geometry or an unproved separation leaves the query enabled.
    """
    if model.body_parentid[child] != parent or model.body_jntnum[child] != 1:
        return None
    joint = int(model.body_jntadr[child])
    if (model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_HINGE
            or np.any(model.jnt_pos[joint] != 0)):
        return None
    ids = [np.flatnonzero((model.geom_bodyid == body)
                         & ((model.geom_contype != 0) | (model.geom_conaffinity != 0)))
           for body in (parent, child)]
    if any(not len(group) for group in ids):
        return None
    enabled = [(int(a), int(b)) for a in ids[0] for b in ids[1]
               if (model.geom_contype[a] & model.geom_conaffinity[b])
               or (model.geom_contype[b] & model.geom_conaffinity[a])]
    if not enabled:
        return None
    if any(model.geom_type[index] not in (mujoco.mjtGeom.mjGEOM_BOX,
                                          mujoco.mjtGeom.mjGEOM_MESH)
           for group in ids for index in group):
        return None
    context = MPIntervalContext()
    context.prec = 113
    dot = lambda left, right: sum(a*b for a, b in zip(left, right))
    inputs = [model.body_pos[child], model.jnt_axis[joint], model.jnt_range[joint],
              model.geom_margin[np.concatenate(ids)], model.geom_gap[np.concatenate(ids)],
              np.asarray([model.opt.o_margin, model.opt.ccd_tolerance])]
    if any(not np.isfinite(value).all() for value in inputs) or model.opt.ccd_tolerance < 0:
        raise ValueError("Native certificate geometry, margins and tolerance must be finite")
    axis = [context.mpf(float(value)) for value in model.jnt_axis[joint]]
    norm = context.sqrt(dot(axis, axis))
    if not (norm.a > 0 and abs(norm-1).b <= 64*np.finfo(float).eps):
        raise ValueError("Compiled hinge axis must be normalized")
    axis = [value/norm for value in axis]
    parent_rotation = _interval_rotation(context, model.body_quat[child], normalize=True)
    parent_axis = [dot(row, axis) for row in parent_rotation]
    parent_offset = dot(parent_axis, [context.mpf(float(value))
                                      for value in model.body_pos[child]])
    records, intervals = [], []
    for group, body_axis, offset in zip(ids, (parent_axis, axis), (parent_offset, context.mpf(0))):
        body_intervals = []
        for index in group:
            index = int(index)
            geometry = _collision_geometry(model, index, "native_self_separation")
            vertices = geometry["vertices"]
            position = np.asarray(model.geom_pos[index])
            if not np.isfinite(vertices).all() or not np.isfinite(position).all():
                raise ValueError("Every enabled native collision vertex and pose must be finite")
            rotation = _interval_rotation(context, model.geom_quat[index], normalize=False)
            # Pull the common joint-axis linear functional into geom-local
            # coordinates. Only this affine projection is applied to vertices;
            # compiled mesh scaling/recentering must never be applied twice.
            coefficients = [dot(body_axis, column) for column in zip(*rotation)]
            translation = dot(body_axis, [context.mpf(float(value)) for value in position])-offset
            projections = [dot(coefficients, [context.mpf(float(value)) for value in vertex])
                           + translation for vertex in vertices]
            bounds = _interval_bounds(context.mpf([min(value.a for value in projections),
                                                    max(value.b for value in projections)]))
            body_intervals.append(bounds)
            records.append({"geom": model.geom(index).name,
                            "body": model.body(model.geom_bodyid[index]).name,
                            "representation": geometry["collision_representation"],
                            "vertex_count": len(vertices),
                            "geom_position": position.tolist(),
                            "geom_quaternion_wxyz": model.geom_quat[index].tolist(),
                            "contype": int(model.geom_contype[index]),
                            "conaffinity": int(model.geom_conaffinity[index]),
                            "margin_m": float(model.geom_margin[index]),
                            "gap_m": float(model.geom_gap[index]),
                            "axis_projection_interval_m": bounds})
        intervals.append([min(row[0] for row in body_intervals),
                          max(row[1] for row in body_intervals)])
    separations = [context.mpf(intervals[1][0])-context.mpf(intervals[0][1]),
                   context.mpf(intervals[0][0])-context.mpf(intervals[1][1])]
    direction = max(range(2), key=lambda index: separations[index].a)
    separation = separations[direction]
    override = bool(model.opt.enableflags & int(mujoco.mjtEnableBit.mjENBL_OVERRIDE))
    required = context.mpf(0)
    pair_parameters = []
    for first, second in enabled:
        margin = float(model.opt.o_margin if override else
                       max(model.geom_margin[first], model.geom_margin[second]))
        gap = float(max(model.geom_gap[first], model.geom_gap[second]))
        include = context.mpf(margin)-context.mpf(gap)
        # Search contacts at margin, not just margin-gap. Recording both also
        # handles a negative gap conservatively instead of hiding its effect.
        required = max((required, context.mpf(margin), include), key=lambda value: value.b)
        pair_parameters.append({"geoms": [model.geom(first).name, model.geom(second).name],
                                "effective_margin_m": margin, "effective_gap_m": gap,
                                "include_margin_interval_m": _interval_bounds(include)})
    required += context.mpf(float(model.opt.ccd_tolerance))
    if not separation.a > required.b:
        return None
    return {"kind": "direct_hinge_complete_solid_axial_separation",
            "joint": model.joint(joint).name,
            "parent_body": model.body(parent).name, "child_body": model.body(child).name,
            "joint_axis_child": model.jnt_axis[joint].tolist(),
            "joint_pivot_child": model.jnt_pos[joint].tolist(),
            "child_position_parent": model.body_pos[child].tolist(),
            "child_quaternion_parent_wxyz": model.body_quat[child].tolist(),
            "native_joint_range": model.jnt_range[joint].tolist(),
            "domain": "all real hinge angles; other joints and common rigid motion unrestricted",
            "all_enabled_geometries": records, "native_geom_pair_parameters": pair_parameters,
            "body_axis_projection_intervals_m": intervals,
            "separation_direction": "child_above_parent" if direction == 0 else "parent_above_child",
            "separation_lower_bound_m": _interval_bounds(separation)[0],
            "required_distance_upper_bound_m": _interval_bounds(required)[1],
            "certified_slack_lower_bound_m": _interval_bounds(separation-required)[0],
            "margin_override_enabled": override, "option_o_margin_m": float(model.opt.o_margin),
            "native_ccd_tolerance_m": float(model.opt.ccd_tolerance),
            "arithmetic": "local mpmath.iv, 113 bits; compiled binary inputs; outward binary64 endpoints",
            "native_pair_remains_enabled": True,
            "scope": "this native self pair only; no environment, object or other-link query removed"}


# BEGIN explicit model contact permission, separate from physical filtering.
def _mechanical_contact_contract(model, links, pairs, native_exclusions):
    """Map exact geom permissions only when native link ignores are exact.

    MuJoCo contact enablement asks physics to resolve a pair; it is not itself
    a requirement that the two solids always remain separated. The caller may
    therefore explicitly permit mechanical contact between named native geoms.
    This declaration never changes their native contact masks or model state.

    The maintained cuRobo YAML interface ignores whole link pairs. Use it only
    when every native-enabled geom pair between those links is explicitly
    permitted. A mixed link pair with an unpermitted geom is unsupported here,
    rather than broadening permission to that geom or editing kernel buffers.
    """
    if not isinstance(pairs, list):
        raise ValueError("mechanical_contact_pairs must be an explicit list; use [] for no permission")
    disabled = int(mujoco.mjtDisableBit.mjDSBL_CONTACT) | int(mujoco.mjtDisableBit.mjDSBL_CONSTRAINT)
    if pairs and int(model.opt.disableflags) & disabled:
        raise ValueError("Mechanical contact permission requires native contact and constraint processing")
    link_order = {name: index for index, name in enumerate(links)}
    declared, requested, link_pairs = [], set(), set()

    def enabled(first, second):
        return bool((model.geom_contype[first] & model.geom_conaffinity[second])
                    or (model.geom_contype[second] & model.geom_conaffinity[first]))

    for row in pairs:
        if (not isinstance(row, dict) or set(row) != {"first", "second", "source"}
                or any(not isinstance(row[key], str) or not row[key].strip()
                       for key in ("first", "second", "source"))):
            raise ValueError("Every mechanical contact needs exactly first, second and nonempty source strings")
        first, second = row["first"], row["second"]
        identity = frozenset((first, second))
        if len(identity) != 2 or identity in requested:
            raise ValueError("Mechanical contact geom pairs must be distinct and unique in either order")
        ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in (first, second)]
        if any(index < 0 for index in ids):
            raise ValueError("Mechanical contact permission names an absent native geom")
        bodies = [int(model.geom_bodyid[index]) for index in ids]
        names = [model.body(index).name for index in bodies]
        if any(name not in link_order for name in names):
            raise ValueError("Mechanical contact geoms must belong to the complete sphere-cover link scope")
        if (bodies[0] == bodies[1] or names[1] in native_exclusions.get(names[0], ())
                or names[0] in native_exclusions.get(names[1], ()) or not enabled(*ids)):
            raise ValueError("Mechanical contact permission must retain a native-enabled geom pair")
        requested.add(identity)
        link_pairs.add(tuple(sorted(names, key=link_order.__getitem__)))
        # Preserve the explicit declaration and its provenance without
        # inferring permission from geom names, materials, poses or contacts.
        declared.append({key: row[key] for key in ("first", "second", "source")})

    exclusions = {}
    for first, second in sorted(link_pairs, key=lambda pair: (link_order[pair[0]], link_order[pair[1]])):
        first_geoms = np.flatnonzero(model.geom_bodyid == model.body(first).id)
        second_geoms = np.flatnonzero(model.geom_bodyid == model.body(second).id)
        for a in first_geoms:
            for b in second_geoms:
                if enabled(a, b) and frozenset((model.geom(a).name, model.geom(b).name)) not in requested:
                    raise ValueError(
                        "Partial geom mechanical contact permission cannot use a whole-link ignore: "
                        f"unpermitted native-enabled pair {model.geom(a).name} / {model.geom(b).name}")
        exclusions.setdefault(first, []).append(second)
    return declared, exclusions


def _collision_query_contract(model, kinematics, *, mechanical_contact_pairs):
    """Separate native filtering, explicit mechanical contact and separation proofs."""
    if (model.npair or model.nexclude or model.neq or model.ntendon or model.nflex or model.nplugin
            or mujoco.get_mjcb_contactfilter() is not None):
        raise ValueError("Collision certificates require the exporter's explicit native policy without extensions")
    links = kinematics["collision_link_names"]
    spheres = kinematics["collision_spheres"]
    if len(links) != len(set(links)) or set(links) != set(spheres):
        raise ValueError("Collision links and sphere geometry must be complete and unique")
    native = _native_ignored_pairs(model, links)
    mechanical, contact_exclusions = _mechanical_contact_contract(
        model, links, mechanical_contact_pairs, native)
    completed, queries, original_count = [], [], 0
    for index, first in enumerate(links):
        for second in links[index+1:]:
            if second in native.get(first, []) or first in native.get(second, []):
                continue
            original_count += len(spheres[first])*len(spheres[second])
            if second in contact_exclusions.get(first, ()) or first in contact_exclusions.get(second, ()):
                # Contact remains active in physics. Only this completely
                # authorized link pair leaves the planner's separation set.
                continue
            a, b = model.body(first).id, model.body(second).id
            parent, child = (a, b) if model.body_parentid[b] == a else (b, a)
            # Current explicit interface has zero sphere/self buffers. A
            # nonzero or unmodelled clearance must retain all native queries;
            # geometric approximation excess itself is not such a clearance.
            clearances = [kinematics["collision_sphere_buffer"],
                          kinematics["self_collision_buffer"][first],
                          kinematics["self_collision_buffer"][second]]
            if not np.isfinite(clearances).all():
                raise ValueError("Collision buffers must be finite explicit values")
            certificate = (_hinge_separation_certificate(model, parent, child)
                           if all(value == 0 for value in clearances) else None)
            if certificate is None:
                queries.append([first, second])
            else:
                completed.append({"links": [first, second], **certificate})
    return {"native_policy_exclusions": native,
            "native_enabled_sphere_pair_count": original_count,
            "mechanical_contact_pairs": mechanical,
            "mechanical_contact_link_exclusions": contact_exclusions,
            "mechanical_contact_sphere_pair_count": sum(
                len(spheres[first])*len(spheres[second])
                for first, others in contact_exclusions.items() for second in others),
            "certificate_completed_pairs": completed, "effective_query_pairs": queries,
            "effective_query_sphere_pair_count": sum(len(spheres[a])*len(spheres[b]) for a, b in queries)}


def _effective_ignored_pairs(contract):
    result = {name: list(others) for name, others in contract["native_policy_exclusions"].items()}
    for first, others in contract["mechanical_contact_link_exclusions"].items():
        result.setdefault(first, []).extend(others)
    for certificate in contract["certificate_completed_pairs"]:
        first, second = certificate["links"]
        result.setdefault(first, []).append(second)
    return result
# END mechanical contact never disables native physics or widens a geom permission.


def _source_collision_names(scene, collision_root):
    """Require the existing expanded-source contract before using any proof."""
    if scene.find("default") is not None:
        raise ValueError("Provide expanded native XML without unresolved collision defaults")
    if any(scene.find(tag) is not None and len(scene.find(tag))
           for tag in ("contact", "equality", "tendon")):
        raise ValueError("Explicit contact/equality/tendon rules require a matching exporter")
    source_root = next((node for node in scene.iter("body") if node.get("name") == collision_root), None)
    if source_root is None:
        raise ValueError("The explicit arm/hand collision and visual root is absent")
    return {node.get("name") for node in source_root.iter("geom")
            if int(node.get("contype", "1")) or int(node.get("conaffinity", "1"))}


# BEGIN retain known fixed installation primitives separately from arm sphere covers.
def _fixed_installation_records(model, *, root_body, sphere_geom_names, sphere_links):
    """Export native fixed robot geometry and only its native pair exclusions.

    These bodies belong to the calibrated robot subtree. They are complete
    static collision owners, not environment reconstructions or extra moving
    spheres. In particular a cylinder remains an analytic cylinder here; the
    cuRobo adapter separately records its conservative polygonal approximation.
    """
    root = model.body(root_body).id
    bodies = {root}
    for body in range(root + 1, model.nbody):
        if int(model.body_parentid[body]) in bodies:
            bodies.add(body)
    enabled = [index for index in range(model.ngeom)
               if int(model.geom_bodyid[index]) in bodies
               and (model.geom_contype[index] or model.geom_conaffinity[index])]
    extras = [index for index in enabled if model.geom(index).name not in sphere_geom_names]
    if len(extras) > 32:
        raise ValueError("Known robot installation exceeds the bounded 32-primitive contract")
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    base = _pose(data.xpos[root], data.xquat[root])
    result = []
    for index in extras:
        body = int(model.geom_bodyid[index])
        cursor = body
        while cursor != root:
            if cursor not in bodies or model.body_jntnum[cursor]:
                raise ValueError("Geometry outside the arm sphere root must be rigidly fixed to the robot base")
            cursor = int(model.body_parentid[cursor])
        if model.body_jntnum[root]:
            raise ValueError("A fixed installation requires the existing fixed robot base")
        kind = model.geom_type[index]
        if kind == mujoco.mjtGeom.mjGEOM_BOX:
            shape_kind, dimensions = "box", (2 * model.geom_size[index]).tolist()
        elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
            shape_kind = "cylinder"
            dimensions = [float(model.geom_size[index, 0]), float(2 * model.geom_size[index, 1])]
        else:
            raise ValueError("Fixed robot installation accepts explicit native box or cylinder primitives")
        if not np.isfinite(dimensions).all() or np.any(np.asarray(dimensions) <= 0):
            raise ValueError("Native fixed primitives require finite positive metric dimensions")
        link = model.body(body).name
        excluded = []
        for other in sphere_links:
            # The existing body-pair policy requires every native geom pair
            # to be excluded. It never borrows a sphere separation certificate.
            policy = _native_ignored_pairs(model, [other, link])
            if link in policy.get(other, ()):
                excluded.append(other)
        native_world = np.eye(4)
        native_world[:3, :3] = data.geom_xmat[index].reshape(3, 3)
        native_world[:3, 3] = data.geom_xpos[index]
        result.append({
            "geom": model.geom(index).name, "link": link,
            "shape_kind": shape_kind, "dimensions": dimensions,
            "T_body_geom": _native_geom_pose(model, index).tolist(),
            "T_base_geom": (np.linalg.inv(base) @ native_world).tolist(),
            "robot_excluded_links": excluded,
            "source_contype": int(model.geom_contype[index]),
            "source_conaffinity": int(model.geom_conaffinity[index]),
            "policy_source": "native welded-body, enabled parent filter and collision masks only",
            "source": f"Complete fixed native robot installation below {root_body}"})
    return result


def _validate_fixed_installation(fixture, models, sphere_geom_names, sphere_links):
    """Rebuild installation identity, rigid placement and pair policy on reload."""
    if (fixture["robot_visual_root"] != fixture["base_link"]
            or not np.array_equal(fixture["T_base_visual_root"], np.eye(4))):
        raise ValueError("Visual self-filter must contain the complete calibrated robot assembly")
    supplied = fixture["fixed_collision_geometries"]
    for model in models:
        expected = _fixed_installation_records(
            model, root_body=fixture["base_link"], sphere_geom_names=sphere_geom_names,
            sphere_links=sphere_links)
        if len(supplied) != len(expected):
            raise ValueError("Every native fixed installation primitive must remain in the fixture")
        for actual, native in zip(supplied, expected):
            if set(actual) != set(native):
                raise ValueError("Fixed installation records require the explicit native schema")
            for key in native:
                same = (np.allclose(actual[key], native[key], atol=1e-12, rtol=0)
                        if key in {"T_body_geom", "T_base_geom"} else actual[key] == native[key])
                if not same:
                    raise ValueError(f"Fixed installation native geometry or policy changed: {native['geom']}/{key}")
    return len(supplied)


def fixed_installation_proxy(record):
    """Prepare a bounded, outward-rounded cuRobo shape without device access.

    Native cylinders are retained in the manifest for independent checking.
    cuRobo's default cylinder tessellation is inscribed, so this adapter uses
    a circumscribed 64-sided prism and checks its serialized float32 support
    planes. Its small, reported enlargement is collision padding only: no
    volume is added to observed-space evidence.
    """
    kind, dimensions = record["shape_kind"], np.asarray(record["dimensions"], float)
    if (dimensions.shape != ((3,) if kind == "box" else (2,))
            or kind not in {"box", "cylinder"} or not np.isfinite(dimensions).all()
            or np.any(dimensions <= 0)):
        raise ValueError("Fixed installation needs native positive box dimensions or cylinder radius/full length")
    identity = [0., 0., 0., 1., 0., 0., 0.]
    if kind == "box":
        outward = np.nextafter(dimensions.astype(np.float32), np.float32(np.inf)).astype(float)
        if not np.isfinite(outward).all() or np.any(outward < dimensions):
            raise ValueError("Fixed native box cannot be enclosed by the serialized finite dimensions")
        return {"kind": "cuboid", "arguments": {"name": record["geom"], "pose": identity,
                                                   "dims": outward.tolist()},
                "approximation": {"kind": "outward-rounded native box",
                                  "maximum_face_expansion_m": float(((outward-dimensions)/2).max())}}
    sections = 64
    radius, length = dimensions
    # The micrometre guard covers vertex quantization, independent of sensor
    # uncertainty and intentional physical/contact tolerances.
    outer_radius = float(np.nextafter(np.float32(radius/np.cos(np.pi/sections)+1e-6), np.float32(np.inf)))
    half_length = float(np.nextafter(np.float32(length/2+1e-6), np.float32(np.inf)))
    mesh = trimesh.creation.cylinder(radius=outer_radius, height=2*half_length, sections=sections)
    vertices = np.asarray(mesh.vertices, dtype=np.float32).astype(float)
    planes = ConvexHull(vertices).equations
    radial = np.linalg.norm(planes[:, :2], axis=1)
    sides = radial > .5
    radial_allowance = float((-planes[sides, 3]/radial[sides]).min()-radius)
    axial_allowance = float(min(vertices[:, 2].max(), -vertices[:, 2].min())-length/2)
    if (not np.isfinite(vertices).all() or radial_allowance < 0 or axial_allowance < 0
            or np.any(np.abs(planes[sides, 2]) > 1e-8)):
        raise ValueError("Serialized installation prism does not enclose its native cylinder")
    return {"kind": "mesh", "arguments": {"name": record["geom"], "pose": identity,
                                              "vertices": vertices.tolist(), "faces": mesh.faces.tolist()},
            "approximation": {"kind": "circumscribed native cylinder prism", "sections": sections,
                              "maximum_radial_expansion_m": float(np.linalg.norm(vertices[:, :2], axis=1).max()-radius),
                              "maximum_axial_expansion_m": float(half_length-length/2),
                              "minimum_radial_cover_allowance_m": radial_allowance,
                              "minimum_axial_cover_allowance_m": axial_allowance}}


def create_fixed_installation_owners(fixture, *, device_cfg, T_output_base, sphere_indices_by_link):
    """Bind validated robot primitives as existing complete static scene owners.

    The runner first calls validate_fixture_collision_contract. Link indices
    here come from the actual loaded cuRobo model, never from an assumed YAML
    or URDF ordering. These owners remain obstacles to payloads and every
    robot link whose native installation pair is enabled.
    """
    from curobo._src.geom.collision.collision_scene import SceneCollision, SceneCollisionCfg
    from curobo._src.geom.types import Cuboid, Mesh as SceneMesh, SceneCfg
    from manipulation.motion import _pose_array
    from manipulation.planning_problem import PosePath
    from manipulation.planning_scene import CollisionOwner, CuroboSphereQuery

    base = _pose_array(T_output_base, "current calibrated robot installation base")
    records = fixture["fixed_collision_geometries"]
    expected_links = {row["link"] for row in fixture["collision_geometries"]}
    if set(sphere_indices_by_link) != expected_links:
        raise ValueError("Installation policy requires exactly the native sphere-bearing robot links")
    flattened = [int(index) for link in expected_links for index in sphere_indices_by_link[link]]
    if sorted(flattened) != list(range(fixture["sphere_count"])):
        raise ValueError("Installation policy requires the complete actual native sphere/link layout")
    owners, reports = [], []
    for record in records:
        prepared = fixed_installation_proxy(record)
        owner_id = f"robot_installation/{record['geom']}"
        if prepared["kind"] == "cuboid":
            world = SceneCfg(cuboid=[Cuboid(**prepared["arguments"])])
            cache = {"cuboid": 1}
        else:
            world = SceneCfg(mesh=[SceneMesh(**prepared["arguments"])])
            cache = {"mesh": 1}
        native = SceneCollision.from_config(SceneCollisionCfg(
            device_cfg=device_cfg, scene_model=world, cache=cache))
        query = CuroboSphereQuery(native, owner_ids=(owner_id,), source=record["source"])
        pose = base @ _pose_array(record["T_base_geom"], "native fixed primitive base transform")
        excluded = tuple(int(index) for link in record["robot_excluded_links"]
                         for index in sphere_indices_by_link[link])
        owners.append(CollisionOwner(
            owner_id, query, PosePath(np.array([0., 1.]), np.stack((pose, pose)), record["source"]),
            record["source"], geometry_complete=True, moving=False,
            robot_excluded_spheres=excluded,
            robot_exclusion_source=record["policy_source"] if excluded else None,
            robot_exclusion_geometry_source=fixture["robot_geometry_source"] if excluded else None,
            # BEGIN exact native identity for the independent installation pair policy.
            native_geometry_names=(record["geom"],)))
            # END the owner remains present for payload/world queries.
        reports.append({"owner_id": owner_id, "native_geom": record["geom"],
                        "native_shape_kind": record["shape_kind"], "native_dimensions": record["dimensions"],
                        "T_output_geom": pose.tolist(), "approximation": prepared["approximation"],
                        "robot_excluded_links": record["robot_excluded_links"],
                        "robot_excluded_sphere_indices": list(excluded),
                        "pair_policy_source": record["policy_source"],
                        "geometry_complete": True, "moving_actor": False,
                        "occupied_space_evidence_supplied": False})
    return tuple(owners), reports
# END separate fixed installation ownership without increasing the sphere budget.


# BEGIN bind the actual planner FK and all padding to the certified native model.
def _validate_fixture_kinematics(fixture, kinematics, models):
    """Bind the consumed URDF's entire tree, not just the certified hinge.

    The current exporter has one explicit configuration shape. Extra loader
    fields can replace fixed transforms, add links or change joint offsets;
    accepting them would let the GPU move spheres using unproved kinematics.
    Feedback-dependent lock/default positions remain free because the proof
    covers every angle. All buffers remain the exporter's explicit zero.
    """
    fields = {"format_version", "urdf_path", "asset_root_path", "base_link", "tool_frames",
              "use_global_cumul", "collision_link_names", "mesh_link_names", "collision_spheres",
              "collision_sphere_buffer", "self_collision_buffer", "self_collision_ignore",
              "lock_joints", "cspace"}
    cspace_fields = {"joint_names", "default_joint_position", "cspace_distance_weight",
                     "null_space_weight", "max_acceleration", "max_jerk"}
    if set(kinematics) != fields or set(kinematics["cspace"]) != cspace_fields:
        raise ValueError("Fixture kinematics must use only the explicit exporter fields")
    path = Path(kinematics["urdf_path"])
    if (not path.is_absolute() or path != Path(fixture["urdf_path"])
            or Path(kinematics["asset_root_path"]) != path.parent
            or kinematics["format_version"] != 2.0
            or kinematics["base_link"] != fixture["base_link"]
            or kinematics["tool_frames"] != [fixture["tcp_site"]]
            or fixture["tool_link"] != fixture["tcp_site"]
            or kinematics["mesh_link_names"] != kinematics["collision_link_names"]
            or kinematics["use_global_cumul"] is not True):
        raise ValueError("Fixture planner paths, base, tools or representation changed")
    buffers = kinematics["self_collision_buffer"]
    if (not isinstance(buffers, dict) or set(buffers) != set(kinematics["collision_link_names"])
            or kinematics["collision_sphere_buffer"] != 0
            or any(value != 0 for value in buffers.values())):
        raise ValueError("Every global and per-link collision buffer must remain explicitly zero")
    joint_names = fixture["joint_names"]
    if (kinematics["cspace"]["joint_names"] != joint_names
            or len(joint_names) != len(set(joint_names))
            or not set(kinematics["lock_joints"]).issubset(joint_names)):
        raise ValueError("Fixture configuration-space joints no longer match the native binding")
    # BEGIN bind physical position bounds before any planner dtype conversion.
    # An outward physical endpoint such as 0.04 is not interchangeable with
    # its float32 planner representation. Preserve the source's named doubles
    # exactly; numerical FK tolerances must never widen hardware limits.
    position_limits = fixture.get("position_limits")
    if not isinstance(position_limits, dict) or set(position_limits) != set(joint_names):
        raise ValueError("Fixture position_limits must name every native joint exactly once")
    for name, bounds in position_limits.items():
        values = np.asarray(bounds)
        if (values.shape != (2,) or values.dtype.kind not in "iuf"
                or not np.isfinite(values).all() or values[0] > values[1]):
            raise ValueError(f"Fixture position_limits require finite ordered native bounds: {name}")
    # END named physical bounds remain independent of planner precision.
    actual = URDF.load(path, load_meshes=False)
    actual_links = [link.name for link in actual.robot.links]
    actual_joints = [joint.name for joint in actual.robot.joints]
    if len(actual_links) != len(set(actual_links)) or len(actual_joints) != len(set(actual_joints)):
        raise ValueError("Fixture URDF links and joints must be unique")

    def require_joint(name, parent, child, kind, position, quaternion, axis=None, limits=None):
        joint = actual.joint_map.get(name)
        pose = _pose(position, quaternion)
        # yourdfpy writes xyz/RPY with Python's exact float round-trip and
        # reloads them with these same mature trimesh transformations. Compare
        # that canonical serialized matrix exactly, not a loose FK tolerance.
        serialized_pose = trimesh.transformations.compose_matrix(
            translate=trimesh.transformations.translation_from_matrix(pose),
            angles=trimesh.transformations.euler_from_matrix(pose))
        if (joint is None or (joint.parent, joint.child, joint.type) != (parent, child, kind)
                or joint.mimic is not None or not np.array_equal(joint.origin, serialized_pose)
                or (axis is not None and not np.array_equal(joint.axis, axis))
                or (limits is not None and (joint.limit is None or
                    not np.array_equal([joint.limit.lower, joint.limit.upper], limits)))):
            raise ValueError(f"Actual planner URDF joint differs from original native FK: {name}")

    for model in models:
        root = model.body(fixture["base_link"]).id
        if model.body_parentid[root] != 0 or model.body_jntnum[root]:
            raise ValueError("Fixture native base must remain a fixed direct world child")
        bodies = {root}
        for body in range(root+1, model.nbody):
            if model.body_parentid[body] in bodies:
                bodies.add(body)
        expected_links = {model.body(body).name for body in bodies} | {fixture["tcp_site"]}
        if set(actual_links) != expected_links:
            raise ValueError("Actual planner URDF does not contain the complete native base subtree")
        expected_joints, native_joints = set(), set()
        for body in sorted(bodies-{root}):
            name = model.body(body).name
            count = model.body_jntnum[body]
            if count > 1:
                raise ValueError("Fixture FK requires at most one native scalar joint per body")
            joint_name, kind, axis, limits = f"{name}_fixed", "fixed", None, None
            if count:
                index = int(model.body_jntadr[body])
                if (model.jnt_type[index] not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)
                        or np.any(model.jnt_pos[index] != 0) or not model.jnt_limited[index]
                        or model.qpos0[model.jnt_qposadr[index]] != 0):
                    # This URDF has no joint offset. A native nonzero ref
                    # changes q into q-ref even though the static pose agrees.
                    raise ValueError("Fixture FK requires origin-centred bounded scalar joints with zero ref")
                joint_name = model.joint(index).name
                native_joints.add(joint_name)
                kind = "revolute" if model.jnt_type[index] == mujoco.mjtJoint.mjJNT_HINGE else "prismatic"
                axis, limits = model.jnt_axis[index], model.jnt_range[index]
                # BEGIN independently recheck the original and robot-only doubles.
                if not np.array_equal(position_limits[joint_name], limits):
                    raise ValueError(f"Fixture position_limits differ from original native bounds: {joint_name}")
                # END no physical epsilon or float32 round-trip enters this check.
            expected_joints.add(joint_name)
            require_joint(joint_name, model.body(model.body_parentid[body]).name, name, kind,
                          model.body_pos[body], model.body_quat[body], axis, limits)
        site = model.site(fixture["tcp_site"]).id
        if model.site_bodyid[site] not in bodies:
            raise ValueError("Fixture TCP must remain attached to its native base subtree")
        tcp_joint = f"{fixture['tcp_site']}_fixed"
        expected_joints.add(tcp_joint)
        require_joint(tcp_joint, model.body(model.site_bodyid[site]).name, fixture["tcp_site"], "fixed",
                      model.site_pos[site], model.site_quat[site])
        if set(actual_joints) != expected_joints or set(joint_names) != native_joints:
            raise ValueError("Actual planner URDF joint inventory differs from original native FK")
# END bind the actual planner FK and all padding to the certified native model.


def validate_fixture_collision_contract(fixture, robot_config):
    """Reload source geometry and re-prove every completed self query on CPU.

    A saved list of ignored names is never sufficient evidence. Every native
    enabled geometry is matched to its original cover input, the serialized
    spheres are matched to the enclosing-cell evidence, and the query contract
    is recomputed from BOTH the original source and exported robot model.
    Pose feedback/default joint positions are deliberately not certificate
    inputs: this certificate is valid throughout its stated hinge domain.
    """
    # BEGIN require an explicit shared model-level mechanical contact policy.
    # Old artifacts do not acquire implicit permissions. Both original and
    # exported native models must independently validate the same declaration.
    if "mechanical_contact_pairs" not in fixture:
        raise ValueError("Fixture needs explicit mechanical_contact_pairs; re-export with [] for no permission")
    mechanical_contact_pairs = fixture["mechanical_contact_pairs"]
    # END permission provenance is required even when the explicit list is empty.
    records = fixture["collision_geometries"]
    names = [row["geom"] for row in records]
    scene = ET.parse(fixture["source_scene_xml"]).getroot()
    enabled = _source_collision_names(scene, fixture["robot_collision_root"])
    if len(names) != len(set(names)) or set(names) != enabled:
        raise ValueError("Fixture certificates must include every original enabled collision geometry")
    kinematics = robot_config["robot_cfg"]["kinematics"]
    links = kinematics["collision_link_names"]
    if set(links) != {row["link"] for row in records}:
        raise ValueError("Fixture collision-link ownership does not match all native geometries")
    robot = mujoco.MjModel.from_xml_path(fixture["robot_mjcf_path"])
    source = mujoco.MjModel.from_xml_path(fixture["source_scene_xml"])
    # BEGIN require the exact FK tree actually consumed with these spheres.
    _validate_fixture_kinematics(fixture, kinematics, (robot, source))
    # END require the exact FK tree actually consumed with these spheres.
    # BEGIN validate the separately retained, complete fixed robot installation.
    fixed_count = _validate_fixed_installation(fixture, (robot, source), set(names), links)
    # END installation validation does not change articulated self-query policy.
    offsets = {name: 0 for name in links}
    minimum_allowance = np.inf
    for row in records:
        with np.load(row["certificate_path"], allow_pickle=False) as saved:
            for model in (robot, source):
                index = model.geom(row["geom"]).id
                native = _collision_geometry(model, index, "native_certificate_revalidation")
                expected_pose = _native_geom_pose(model, index)
                if (model.body(model.geom_bodyid[index]).name != row["link"]
                        or row["representation"] != native["collision_representation"]
                        or not np.array_equal(saved["native_vertices"], native["vertices"])
                        or not np.array_equal(saved["T_body_geom"], expected_pose)
                        or row["source_contype"] != model.geom_contype[index]
                        or row["source_conaffinity"] != model.geom_conaffinity[index]):
                    raise ValueError(f"Native geometry or pose changed for {row['geom']}")
            centers, radii = saved["sphere_centers_body"], saved["sphere_radii"]
            count = len(radii)
            vertex_offsets = saved["cell_vertex_offsets"]
            vertices = saved["cell_vertices_body"]
            if (centers.shape != (count, 3) or count != row["sphere_count"] or count < 1
                    or vertex_offsets.shape != (count+1,) or vertex_offsets[0] != 0
                    or vertex_offsets[-1] != len(vertices) or np.any(np.diff(vertex_offsets) < 4)
                    or vertices.ndim != 2 or vertices.shape[1] != 3
                    or not all(np.isfinite(value).all() for value in (centers, radii, vertices))
                    or np.any(radii <= 0)):
                raise ValueError("Invalid complete-cell sphere-cover evidence")
            # Rebuild the complete partition from the ORIGINAL convex solid,
            # not from a saved subset of cell vertices. Matching only native
            # vertices plus arbitrary enclosed points would permit holes.
            specification = row["cover_specification"]
            is_box = source.geom_type[index] == mujoco.mjtGeom.mjGEOM_BOX
            cells, sites = _cover_cells(native["vertices"], count=specification.get("cells"),
                box_halfsize=source.geom_size[index] if is_box else None,
                subdivisions=specification.get("subdivisions"))
            rebuilt = [cell @ expected_pose[:3, :3].T+expected_pose[:3, 3] for cell in cells]
            rebuilt_offsets = np.cumsum([0]+[len(cell) for cell in rebuilt])
            if (not np.array_equal(vertex_offsets, rebuilt_offsets)
                    or not np.array_equal(vertices, np.concatenate(rebuilt))
                    or not np.array_equal(saved["partition_sites_geom"],
                                          np.empty((0, 3)) if sites is None else sites)):
                raise ValueError("Saved sphere cells do not match a complete native-solid partition")
            start = offsets[row["link"]]
            balls = kinematics["collision_spheres"][row["link"]][start:start+count]
            if (len(balls) != count
                    or not np.array_equal([ball["center"] for ball in balls], centers)
                    or not np.array_equal([ball["radius"] for ball in balls], radii)):
                raise ValueError(f"Serialized spheres differ from their cover evidence: {row['geom']}")
            for index in range(count):
                cell = vertices[vertex_offsets[index]:vertex_offsets[index+1]]
                allowance = radii[index]-np.linalg.norm(cell-centers[index], axis=1).max()
                if allowance < 1e-6:
                    raise ValueError("A serialized sphere lost its required complete-cell enclosure allowance")
                minimum_allowance = min(minimum_allowance, allowance)
            offsets[row["link"]] += count
    if (any(offsets[name] != len(kinematics["collision_spheres"][name]) for name in links)
            or sum(offsets.values()) != fixture["sphere_count"]
            or not 1 <= fixture["sphere_count"] <= 480):
        raise ValueError("Fixture sphere count or complete collision ownership changed")
    # BEGIN share contact declarations while re-deriving both native contracts.
    expected = _collision_query_contract(robot, kinematics, mechanical_contact_pairs=mechanical_contact_pairs)
    source_expected = _collision_query_contract(source, kinematics, mechanical_contact_pairs=mechanical_contact_pairs)
    # END no saved link ignore is sufficient evidence of an exact geom permission.
    if source_expected != expected:
        raise ValueError("Exported collision certificate no longer matches its original native source")
    if any(fixture[key] != value for key, value in expected.items()):
        raise ValueError("Saved collision policy/certificate/effective queries failed fresh validation")
    if kinematics["self_collision_ignore"] != _effective_ignored_pairs(expected):
        raise ValueError("Effective self-collision exclusions differ from native policy, mechanical permissions and valid certificates")
    return {"sphere_count": fixture["sphere_count"],
            "complete_native_geometry_count": len(records),
            "complete_fixed_installation_geometry_count": fixed_count,
            "native_enabled_sphere_pair_count": expected["native_enabled_sphere_pair_count"],
            # BEGIN report mechanical permission separately from native filtering and proofs.
            "mechanical_contact_pairs": expected["mechanical_contact_pairs"],
            "mechanical_contact_link_exclusions": expected["mechanical_contact_link_exclusions"],
            "mechanical_contact_sphere_pair_count": expected["mechanical_contact_sphere_pair_count"],
            # END explicit permission leaves native physical contacts enabled.
            "certificate_completed_pairs": [row["links"] for row in expected["certificate_completed_pairs"]],
            "effective_query_sphere_pair_count": expected["effective_query_sphere_pair_count"],
            "minimum_cell_enclosure_allowance_m": float(minimum_allowance),
            "source_and_robot_certificate_equal": True,
            "scope": "CPU static geometry and self-query contract; no feedback-pose, scene or trajectory claim"}
# END native solid separation certificates, independent of sphere fitting.


# BEGIN require mechanical permission as an explicit export input, including [].
def export_fixture(*, scene_path, catalog_path, episode_path, root_body, collision_root, tcp_site,
                   arm_joint_names, finger_joint_names, velocity_limits, effort_limits,
                   cover_specifications, initial_frame, output, binding_audit_path,
                   max_acceleration, max_jerk, mechanical_contact_pairs):
    """Build one exact native model and its explicitly budgeted collision cover.

    Velocity/effort limits and the initial recorded frame are explicit inputs;
    MJCF has no velocity-limit field. No coordinate is clipped or re-signed.
    Mechanical contact permissions are explicit native geom pairs with a
    model-level source; [] declares that there is no such permission.
    Unsupported source features fail before artifacts can be used for planning.
    """
    # END no compatibility default silently supplies a missing model policy.
    scene_path, catalog_path, episode_path = [Path(value).resolve() for value in
                                            (scene_path, catalog_path, episode_path)]
    output = Path(output).resolve()
    # BEGIN preserve prior fixtures; each export owns a fresh artifact directory.
    output.mkdir(parents=True, exist_ok=False)
    # END preserve prior fixtures; each export owns a fresh artifact directory.
    scene = ET.parse(scene_path).getroot()
    # BEGIN share the same complete-source gate with reload validation.
    enabled_geoms = _source_collision_names(scene, collision_root)
    if enabled_geoms != set(cover_specifications):
        raise ValueError("Cover specification must include every native enabled geom of the explicit arm/hand root")
    # BEGIN keep fixed native installation geometry in the calibrated robot model.
    # Sphere export still has its existing explicit arm/hand scope and budget.
    # All other enabled geometry under the robot root must be rigid installation
    # geometry; _fixed_installation_records rejects an omitted movable subtree.
    assembly_geoms = _source_collision_names(scene, root_body)
    if not enabled_geoms <= assembly_geoms:
        raise ValueError("The declared arm collision root must belong to the calibrated robot assembly")
    # END share the same complete-source gate with reload validation.
    names = list(arm_joint_names) + list(finger_joint_names)
    if len(set(names)) != len(names) or set(velocity_limits) != set(names) or set(effort_limits) != set(names):
        raise ValueError("Native scalar joints need unique names and explicit velocity/effort limits")
    model = _robot_model(scene_path, root_body, sorted(assembly_geoms))
    if model.nq != len(names) or set(names) != {model.joint(i).name for i in range(model.njnt)}:
        raise ValueError("Named arm/finger joints must cover the robot exactly")
    # Preserve options used by the native collision-pair filter. This model
    # contains the robot only and is never used as a substitute scene rollout.
    robot_mjcf = output / "robot_only.xml"
    mujoco.mj_saveLastXML(str(robot_mjcf), model)
    xml = ET.parse(robot_mjcf)
    if scene.find("option") is not None:
        old = xml.getroot().find("option")
        if old is not None:
            xml.getroot().remove(old)
        from copy import deepcopy
        xml.getroot().append(deepcopy(scene.find("option")))
        xml.write(robot_mjcf, encoding="unicode")
        model = mujoco.MjModel.from_xml_path(str(robot_mjcf))
    catalog = json.loads(catalog_path.read_text())
    catalog_joints = {joint["name"]: joint for joint in catalog["joints"]}
    binding = []
    for name in names:
        row = catalog_joints[name]
        if row["qpos_count"] != 1 or row["qvel_count"] != 1:
            raise ValueError("Every exported recorded joint must be scalar")
        binding.append({"recorded_name": name, "qpos_column": row["qpos_start"],
                        "qvel_column": row["qvel_start"]})
    with h5py.File(episode_path, "r") as stream:
        q0 = np.asarray([stream["states/qpos"][initial_frame, row["qpos_column"]] for row in binding])
        frames = np.unique(np.linspace(0, len(stream["states/qpos"])-1, 17, dtype=int))
        recorded = np.column_stack([stream["states/qpos"][frames, row["qpos_column"]] for row in binding])
    data = mujoco.MjData(model)
    sphere_links = list(dict.fromkeys(model.body(model.geom_bodyid[model.geom(name).id]).name
                                     for name in sorted(enabled_geoms)))
    fixed_geometries = _fixed_installation_records(
        model, root_body=root_body, sphere_geom_names=enabled_geoms, sphere_links=sphere_links)
    fixed_by_geom = {record["geom"]: record for record in fixed_geometries}
    # END preserve installation primitives without approximating them as robot spheres.
    links, joints, covers, records = [], [], {}, []
    mesh_directory = output / "collision_meshes"
    mesh_directory.mkdir(exist_ok=True)
    root_id = model.body(root_body).id
    for body in range(1, model.nbody):
        name = model.body(body).name
        inertial = Inertial(origin=_pose(model.body_ipos[body], model.body_iquat[body]),
                            mass=float(model.body_mass[body]), inertia=np.diag(model.body_inertia[body]))
        collisions = []
        for geom_id in np.flatnonzero(model.geom_bodyid == body):
            geom_name = model.geom(int(geom_id)).name
            # BEGIN serialize native fixed primitives into the same robot FK tree.
            if geom_name in fixed_by_geom:
                fixed = fixed_by_geom[geom_name]
                if fixed["shape_kind"] == "box":
                    shape = Geometry(box=Box(size=np.asarray(fixed["dimensions"])))
                else:
                    shape = Geometry(cylinder=Cylinder(radius=fixed["dimensions"][0],
                                                       length=fixed["dimensions"][1]))
                collisions.append(Collision(name=geom_name, origin=np.asarray(fixed["T_body_geom"]),
                                            geometry=shape))
                continue
            # END static installation has no additional moving-sphere cover.
            spheres, record, geometry = _geom_cover(model, geom_id, cover_specifications[geom_name], output/f"{geom_name}.npz")
            covers.setdefault(name, []).extend(spheres)
            if model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_BOX:
                shape = Geometry(box=Box(size=2 * model.geom_size[geom_id]))
            else:
                mesh_path = mesh_directory / f"{geom_name}.obj"
                trimesh.Trimesh(vertices=geometry["vertices"], faces=geometry["faces"], process=False).convex_hull.export(mesh_path)
                shape = Geometry(mesh=Mesh(filename=str(mesh_path)))
                record["collision_mesh_path"] = str(mesh_path)
            collisions.append(Collision(name=geom_name, origin=_pose(model.geom_pos[geom_id], model.geom_quat[geom_id]), geometry=shape))
            records.append(record)
        links.append(Link(name=name, inertial=inertial, collisions=collisions))
        if body == root_id:
            if model.body_jntnum[body] or model.body_parentid[body] != 0:
                raise ValueError("The exported root must be fixed and directly below world")
            continue
        if model.body_jntnum[body] > 1:
            raise ValueError("This exporter supports at most one scalar joint per native body")
        joint = Joint(name=f"{name}_fixed", type="fixed", parent=model.body(model.body_parentid[body]).name,
                      child=name, origin=_pose(model.body_pos[body], model.body_quat[body]))
        if model.body_jntnum[body]:
            joint_id = int(model.body_jntadr[body])
            native = model.joint(joint_id).name
            if np.any(model.jnt_pos[joint_id] != 0) or model.jnt_type[joint_id] not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
                raise ValueError("Only origin-centred scalar hinge/slide joints are supported")
            # BEGIN reject a native offset absent from the explicit URDF contract.
            if model.qpos0[model.jnt_qposadr[joint_id]] != 0:
                raise ValueError("Exported scalar joints require zero native ref")
            # END reject a native offset absent from the explicit URDF contract.
            if not model.jnt_limited[joint_id] or min(velocity_limits[native], effort_limits[native]) <= 0:
                raise ValueError("Exported scalar joints require finite explicit native limits")
            joint.name, joint.type = native, "revolute" if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_HINGE else "prismatic"
            joint.axis = model.jnt_axis[joint_id].copy()
            joint.limit = Limit(lower=float(model.jnt_range[joint_id, 0]), upper=float(model.jnt_range[joint_id, 1]),
                                velocity=float(velocity_limits[native]), effort=float(effort_limits[native]))
        joints.append(joint)
    site_id = model.site(tcp_site).id
    links.append(Link(name=tcp_site))
    joints.append(Joint(name=f"{tcp_site}_fixed", type="fixed", parent=model.body(model.site_bodyid[site_id]).name,
                        child=tcp_site, origin=_pose(model.site_pos[site_id], model.site_quat[site_id])))
    urdf = URDF(Robot(name="native_recorded_robot", links=links, joints=joints), load_meshes=False)
    urdf_path = output / "robot.urdf"
    urdf.write_xml_file(str(urdf_path))
    # Reloading verifies the serialized RPY representation, not an in-memory
    # transform shared with the exporter. Samples include open/closed limits.
    reference = URDF.load(urdf_path, load_meshes=False)
    limits = np.asarray([model.jnt_range[model.joint(name).id] for name in names])
    if not np.isfinite(q0).all() or np.any(q0 < limits[:, 0]) or np.any(q0 > limits[:, 1]):
        raise ValueError("Initial measured robot feedback is outside native joint limits")
    probes = np.random.default_rng(7341).uniform(limits[:, 0], limits[:, 1], size=(16, len(names)))
    probes[0], probes[1] = limits[:, 0], limits[:, 1]
    max_position, max_rotation = 0., 0.
    for sample in np.concatenate((recorded, probes)):
        for name, value in zip(names, sample):
            data.qpos[model.joint(name).qposadr[0]] = value
        mujoco.mj_kinematics(model, data)
        base = _pose(data.xpos[root_id], data.xquat[root_id])
        reference.update_cfg(dict(zip(names, sample)))
        for body in range(1, model.nbody):
            predicted = base @ reference.get_transform(model.body(body).name, frame_from=root_body)
            actual = _pose(data.xpos[body], data.xquat[body])
            max_position = max(max_position, float(np.linalg.norm(predicted[:3, 3]-actual[:3, 3])))
            max_rotation = max(max_rotation, float(Rotation.from_matrix(predicted[:3, :3].T @ actual[:3, :3]).magnitude()))
        actual_tcp = np.eye(4)
        actual_tcp[:3, :3], actual_tcp[:3, 3] = data.site_xmat[site_id].reshape(3, 3), data.site_xpos[site_id]
        predicted_tcp = base @ reference.get_transform(tcp_site, frame_from=root_body)
        max_position = max(max_position, float(np.linalg.norm(predicted_tcp[:3, 3]-actual_tcp[:3, 3])))
        max_rotation = max(max_rotation, float(Rotation.from_matrix(predicted_tcp[:3, :3].T @ actual_tcp[:3, :3]).magnitude()))
    if max_position > 1e-9 or max_rotation > 1e-9:
        raise ValueError(f"Serialized URDF differs from native FK: {max_position} m, {max_rotation} rad")
    sphere_count = sum(map(len, covers.values()))
    if sphere_count > 480:
        raise ValueError("Local fixture exceeds the explicitly bounded 480-sphere budget")
    config = {"robot_cfg": {"load_dynamics": False, "kinematics": {
        "format_version": 2.0, "urdf_path": str(urdf_path), "asset_root_path": str(output),
        "base_link": root_body, "tool_frames": [tcp_site], "use_global_cumul": True,
        "collision_link_names": list(covers), "mesh_link_names": list(covers),
        "collision_spheres": covers, "collision_sphere_buffer": 0.,
        "self_collision_buffer": {name: 0. for name in covers},
        "self_collision_ignore": _native_ignored_pairs(model, list(covers)),
        "lock_joints": dict(zip(finger_joint_names, q0[len(arm_joint_names):].tolist())),
        "cspace": {"joint_names": names, "default_joint_position": q0.tolist(),
                   "cspace_distance_weight": [1.] * len(names), "null_space_weight": [1.] * len(names),
                   "max_acceleration": float(max_acceleration), "max_jerk": float(max_jerk)}}}}
    # BEGIN replace only the queries completed by a native-entity certificate.
    # BEGIN compile exact mechanical permissions through the maintained native YAML interface.
    collision_contract = _collision_query_contract(
        model, config["robot_cfg"]["kinematics"], mechanical_contact_pairs=mechanical_contact_pairs)
    # END completed proofs and declared contacts remain separate manifest records.
    config["robot_cfg"]["kinematics"]["self_collision_ignore"] = _effective_ignored_pairs(collision_contract)
    # END replace only the queries completed by a native-entity certificate.
    config_path = output / "robot.yml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    finger_roots = {int(model.jnt_bodyid[model.joint(name).id]) for name in finger_joint_names}
    finger_links = []
    for name in covers:
        ancestor = model.body(name).id
        while ancestor and ancestor not in finger_roots:
            ancestor = int(model.body_parentid[ancestor])
        if ancestor in finger_roots:
            finger_links.append(name)
    # BEGIN separate collision-cover selection from complete visual self filtering.
    collision_root_id = model.body(collision_root).id
    cursor = collision_root_id
    while cursor != root_id:
        if cursor == 0 or model.body_jntnum[cursor]:
            raise ValueError("The explicit collision root must be rigidly fixed below the model base")
        cursor = int(model.body_parentid[cursor])
    # Exclusion rows use the actual collision-link order consumed by cuRobo.
    fixed_geometries = _fixed_installation_records(
        model, root_body=root_body, sphere_geom_names=enabled_geoms, sphere_links=list(covers))
    manifest = {"source_scene_xml": str(scene_path), "catalog_path": str(catalog_path),
        "episode_path": str(episode_path), "binding_audit_path": str(Path(binding_audit_path).resolve()),
        "robot_mjcf_path": str(robot_mjcf), "urdf_path": str(urdf_path), "robot_config_path": str(config_path),
        "base_link": root_body, "base_pose": _pose_dict(base), "tool_link": tcp_site, "tcp_site": tcp_site,
        "robot_collision_root": collision_root,
        "robot_visual_root": root_body, "T_base_visual_root": np.eye(4).tolist(),
        "tool_transform": _pose_dict(np.eye(4)), "joint_names": names,
        "arm_joint_names": list(arm_joint_names), "finger_joint_names": list(finger_joint_names),
        "finger_link_names": finger_links,
        "planning_joint_names": list(arm_joint_names), "joint_binding": binding,
        "initial_frame": int(initial_frame), "initial_joint_positions": q0.tolist(),
        # BEGIN preserve native double endpoints for the physical execution contract.
        "position_limits": {name: model.jnt_range[model.joint(name).id].tolist() for name in names},
        # END planner float32 limits are not hardware metadata.
        "velocity_limits": velocity_limits, "effort_limits": effort_limits,
        "collision_geometries": records, "fixed_collision_geometries": fixed_geometries,
        "sphere_count": sphere_count,
        "sphere_materials": [None] * sphere_count,
        "robot_geometry_source": "compiled native MuJoCo arm/hand collision geoms below robot_collision_root; complete convex-cell sphere cover; remaining calibrated fixed installation retained as native static primitives",
        "coverage_proof": "All native enabled convex solids below robot_collision_root partitioned into convex cells; each serialized float32 sphere encloses every cell vertex with at least 1e-6 m allowance",
        "coverage_limit": "Conservative collision proxies, not RGB-D occupied-solid evidence; support-plane excess is not a Euclidean Hausdorff bound",
        "fk_audit": {"recorded_frames": frames.tolist(), "hypothetical_probes": len(probes),
                     "maximum_translation_error_m": max_position, "maximum_rotation_error_rad": max_rotation,
                    "consumer": "yourdfpy reloaded serialized URDF; cuRobo native FK is a separate runner check"}}
    # END one calibrated assembly owns both visual self filtering and collision geometry.
    # BEGIN persist native policy, certified queries and effective queries separately.
    manifest.update(collision_contract)
    # Validate deserialized primitive values and the written YAML/NPZ before
    # publishing a usable manifest. This does not trust the generating model.
    manifest["collision_contract_validation"] = validate_fixture_collision_contract(
        json.loads(json.dumps(manifest, allow_nan=False)), yaml.safe_load(config_path.read_text()))
    # END persist native policy, certified queries and effective queries separately.
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return manifest


# BEGIN bind exact native material geometry to the runtime sphere ordering.
def bind_fixture_contact_materials(fixture, robot_cfg, sphere_indices_by_link,
                                  feedback, material_faces):
    """Bind explicit pad faces without labeling any adjacent finger or palm.

    The caller supplies the sphere indices from its actual loaded kinematics.
    Per-link cover records are checked against both serialized YAML spheres
    and their saved native certificates before they acquire a material ID.
    Only the isolated robot MJCF is loaded; task-object truth is never read.
    """
    from manipulation.contact_binding import calibrate_box_material_features

    kinematics = robot_cfg["robot_cfg"]["kinematics"]
    links = kinematics["collision_link_names"]
    if set(sphere_indices_by_link) != set(links):
        raise ValueError("Material binding needs every actual native collision-link sphere index")
    indices = [int(index) for link in links for index in sphere_indices_by_link[link]]
    if (len(indices) != fixture["sphere_count"] or sorted(indices) != list(range(len(indices)))
            or any(len(sphere_indices_by_link[link]) != len(kinematics["collision_spheres"][link]) for link in links)):
        raise ValueError("Runtime sphere ownership must be an exact partition of the fixture")
    declarations = {row["native_geometry"]: row for row in material_faces}
    if len(declarations) != len(material_faces):
        raise ValueError("One native geometry cannot acquire multiple material identities")
    model = mujoco.MjModel.from_xml_path(fixture["robot_mjcf_path"])
    materials = calibrate_box_material_features(model, tcp_site=fixture["tcp_site"],
        joint_positions=dict(zip(feedback["joint_names"], feedback["start_q"])),
        finger_joint_names=feedback["finger_joint_names"], material_faces=material_faces)
    offsets = {name: 0 for name in links}
    sphere_materials = [None]*len(indices)
    bound_geometries = {}
    for record in fixture["collision_geometries"]:
        link, count = record["link"], record["sphere_count"]
        start = offsets[link]
        offsets[link] += count
        if record["geom"] not in declarations:
            continue
        declaration = declarations[record["geom"]]
        material = materials[declaration["material_id"]]
        geom = model.geom(record["geom"]).id
        native = _collision_geometry(model, geom, "material_certificate_revalidation")
        with np.load(record["certificate_path"], allow_pickle=False) as saved:
            centers, radii = saved["sphere_centers_body"], saved["sphere_radii"]
            balls = kinematics["collision_spheres"][link][start:start+count]
            if (count != len(radii) or model.body(model.geom_bodyid[geom]).name != link
                    or not np.array_equal(saved["native_vertices"], native["vertices"])
                    or not np.array_equal(saved["T_body_geom"], _native_geom_pose(model, geom))
                    or not np.array_equal([ball["center"] for ball in balls], centers)
                    or not np.array_equal([ball["radius"] for ball in balls], radii)):
                raise ValueError(f"Material geometry differs from its native sphere cover: {record['geom']}")
            normal_geom = np.eye(3)[declaration["face_axis"]]*declaration["face_sign"]
            pose = saved["T_body_geom"]
            normal_body = pose[:3, :3] @ normal_geom
            normal_body /= np.linalg.norm(normal_body)
            face_body = pose[:3, 3]+pose[:3, :3] @ (normal_geom*model.geom_size[geom, declaration["face_axis"]])
            # This is a directional support excess, not a claimed Euclidean
            # mesh-error bound or an automatically granted contact allowance.
            material["normal_cover_excess_m"] = float(np.max((centers-face_body) @ normal_body+radii))
        selected = [int(index) for index in sphere_indices_by_link[link][start:start+count]]
        material["sphere_indices"] = selected
        material["cover_certificate_path"] = record["certificate_path"]
        for index in selected:
            sphere_materials[index] = declaration["material_id"]
        bound_geometries[record["geom"]] = declaration["material_id"]
    if (set(bound_geometries) != set(declarations)
            or any(offsets[link] != len(sphere_indices_by_link[link]) for link in links)):
        raise ValueError("Native material or complete collision-sphere ownership was not found")
    return {"robot_sphere_materials": sphere_materials,
            "native_geometry_materials": bound_geometries, "materials": materials,
            "source": "Explicit native robot box faces, exact cover certificates and runtime sphere identities; no object-contact permission inferred"}
# END all nonmaterial robot geometry keeps its original collision contract.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True, help="Explicit exporter keyword arguments as JSON")
    args = parser.parse_args()
    manifest = export_fixture(**json.loads(args.spec.read_text()))
    print(json.dumps({"manifest": str(Path(json.loads(args.spec.read_text())["output"]).resolve()/"manifest.json"),
                      "sphere_count": manifest["sphere_count"], "fk_audit": manifest["fk_audit"]}, indent=2))


if __name__ == "__main__":
    main()
# END native robot export and continuous-volume sphere-cover certificates.
