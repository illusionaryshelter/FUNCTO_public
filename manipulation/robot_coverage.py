"""CPU new-volume coverage for an explicitly valid initial robot model.

The contract is V(q) \\ (V0 union F) = empty. V0 is the SAME native convex
robot model at the supplied complete measured state, under an explicit valid
initial-state assumption. It is not observed free space or physical-material
evidence. F remains the original union of whole observed-free grid cells.

Manifold owns all solid Boolean operations. Acceptance requires a successful
empty result, never a volume, distance, or sliver threshold. These are bounded
floating-point solid queries, not exact arithmetic or continuous-path proof.
Collision policy, moving payloads, and trajectory time subdivision remain
separate callers' responsibilities.
"""

# BEGIN model-bound initial-volume subtraction, separate from collision balls.
from collections.abc import Sequence

import manifold3d as manifold
import numpy as np
from scipy.spatial import ConvexHull

from .robot_geometry_check import NativeSolid, RobotGeometryChecker, _name, _rigid_pose


_MAX_WORKING_BYTES = 2 * 1024**3
_MAX_GRID_CELLS = 8_000_000
_MAX_FREE_BOXES = 8192
_MAX_SOLIDS = 32
_MAX_VERTICES = 10_000
_MAX_INPUT_FACES = 20_000
_MAX_RESULT_FACES = 65_536
_BOX_BATCH = 32


class CoverageBudgetExceeded(ValueError):
    """Finite proof work was refused; this is never a coverage success."""


def _readonly(value):
    result = np.asarray(value).copy()
    result.setflags(write=False)
    return result


def _free_rectangles(mask):
    """Merge exact cell runs along x, then identical rectangles along y/z.

    Every box is a disjoint union of input True cells. No bounding box around
    disconnected observations, dilation, or hole filling is used. Caps are
    checked while accumulating each slice, before an unbounded box list can
    be materialized. Returned upper integer indices are exclusive.
    """
    result, previous = [], {}
    for z, plane in enumerate(mask):
        rectangles, active = [], {}
        for y, row in enumerate(plane):
            edges = np.flatnonzero(np.diff(np.pad(row.astype(np.int8), (1, 1))))
            if len(edges) // 2 > _MAX_FREE_BOXES:
                raise CoverageBudgetExceeded("Observed free row exceeds the exact box budget")
            current = {(int(a), int(b)) for a, b in edges.reshape(-1, 2)}
            for key in tuple(active):
                if key not in current:
                    rectangles.append((*key, active.pop(key), y))
            for key in current:
                active.setdefault(key, y)
            if len(rectangles) + len(active) > _MAX_FREE_BOXES:
                raise CoverageBudgetExceeded("Observed free slice exceeds the exact box budget")
        rectangles.extend((*key, start, len(plane)) for key, start in active.items())
        current = set(rectangles)
        for key in tuple(previous):
            if key not in current:
                x0, x1, y0, y1 = key
                result.append(((x0, y0, previous.pop(key)), (x1, y1, z)))
        for key in current:
            previous.setdefault(key, z)
        if len(result) + len(previous) > _MAX_FREE_BOXES:
            raise CoverageBudgetExceeded("Observed free grid exceeds the exact merged-box budget")
    for (x0, x1, y0, y1), start in previous.items():
        result.append(((x0, y0, start), (x1, y1, len(mask))))
    return np.asarray(result, dtype=np.int64).reshape(-1, 2, 3)


def _oriented_hull(vertices):
    """Use maintained Qhull on one declared native convex solid, not an assembly."""
    hull = ConvexHull(vertices)
    faces = hull.simplices.copy()
    normal = np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]],
                      vertices[faces[:, 2]] - vertices[faces[:, 0]])
    reversed_faces = np.einsum("ij,ij->i", normal, hull.equations[:, :3]) < 0
    faces[reversed_faces] = faces[reversed_faces][:, [0, 2, 1]]
    return faces, hull.equations.copy()


_BOX_SIGNS = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)])
_BOX_FACES = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5],
                       [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6],
                       [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]], dtype=np.uint64)


class RobotCoverage:
    """New native robot volume versus a frozen model baseline and original F.

    ``checker`` supplies the same complete named scalar FK as the independent
    robot validator. ``solids`` explicitly selects native CONVEX moving robot
    components; static installation obstacles and attached owners are absent.
    The source adapter must verify this inventory against the chosen model.

    Free cells are merged once into exact rectangles, not globally Boolean
    unioned. Each candidate component subtracts V0, then only intersecting
    free boxes, in bounded batches. Input/result faces, box counts, cumulative
    state queries, and a conservative native-work estimate are all bounded.
    The memory estimate is a preflight budget, not a native allocator promise.
    """

    def __init__(self, *, checker, solids, joint_names, baseline_joint_positions,
                 T_reference_base, free_grid, origin, voxel_size, reference_frame,
                 initialization_source, max_queries=2048,
                 max_working_bytes=_MAX_WORKING_BYTES):
        if not isinstance(checker, RobotGeometryChecker):
            raise TypeError("Coverage requires the existing complete named native FK checker")
        if (type(max_queries) is not int or not 0 < max_queries <= 4096
                or type(max_working_bytes) is not int
                or not 0 < max_working_bytes <= _MAX_WORKING_BYTES):
            raise ValueError("Coverage needs finite state-query and at-most-2-GiB CPU budgets")
        self._checker = checker
        self.reference_frame = _name(reference_frame, "coverage reference frame")
        self.initialization_source = _name(initialization_source, "valid initial-model assumption source")
        self._base = _rigid_pose(T_reference_base)
        self.baseline_joint_names = tuple(checker.joint_names)
        self._baseline_q = self._named(joint_names, baseline_joint_positions)
        self._baseline_float32 = self._baseline_q.astype(np.float32).astype(float)
        if not np.isfinite(self._baseline_float32).all():
            raise ValueError("Initial named coordinates need a finite float32 representation")
        self.max_queries, self.max_working_bytes = max_queries, max_working_bytes
        self._queries = 0
        self._maximum_work_estimate = 0
        self._maximum_native_tolerance = 0.
        mask = np.asarray(free_grid)
        if (mask.dtype != np.bool_ or mask.ndim != 3 or min(mask.shape) < 2
                or mask.size > _MAX_GRID_CELLS):
            raise ValueError("Coverage F must be a bounded original boolean free-cell grid")
        self._free = _readonly(mask)
        self.origin = _readonly(np.asarray(origin, float))
        if (self.origin.shape != (3,) or not np.isfinite(self.origin).all()
                or isinstance(voxel_size, (bool, np.bool_))
                or not np.isfinite(voxel_size) or voxel_size <= 0):
            raise ValueError("Free-cell grid needs a finite origin and positive metric resolution")
        self.voxel_size = float(voxel_size)
        indices = _free_rectangles(mask)
        # Shared integer cell boundaries are converted with the same formula.
        # Mesh vertices below use these endpoints directly, avoiding a second
        # centre/size conversion that could introduce seams between boxes.
        self._free_boxes = self.origin + (indices.astype(float) - .5) * self.voxel_size
        if not isinstance(solids, Sequence) or not 0 < len(solids) <= _MAX_SOLIDS:
            raise ValueError("Coverage needs one to thirty-two declared native convex components")
        self._solids, names, vertices_count, faces_count = [], set(), 0, 0
        for solid in solids:
            if not isinstance(solid, NativeSolid):
                raise TypeError("Coverage accepts native convex solids; no implicit primitive tessellation")
            if (solid.name in names or solid.name not in checker.geometry_names
                    or checker.geometry_sources[solid.name] != solid.source):
                raise ValueError("Coverage components must retain unique source-verified checker identities")
            _name(solid.frame, "coverage geometry frame")
            values = np.asarray(solid.vertices_frame)
            if (values.ndim != 2 or values.shape[1] != 3 or len(values) < 4
                    or values.dtype.kind not in "iuf" or not np.isfinite(values).all()):
                raise ValueError("Native coverage vertices must be finite full-dimensional Nx3 values")
            vertices_count += len(values)
            if vertices_count > _MAX_VERTICES:
                raise CoverageBudgetExceeded("Native coverage input exceeds the vertex budget")
            vertices = np.asarray(values, dtype=float).copy()
            faces, planes = _oriented_hull(vertices)
            faces_count += len(faces)
            if faces_count > _MAX_INPUT_FACES:
                raise CoverageBudgetExceeded("Native coverage input exceeds the face budget")
            shape = self._mesh(vertices, faces)
            self._solids.append({"name": solid.name, "frame": solid.frame, "source": solid.source,
                                 "vertices_frame": vertices, "faces": faces,
                                 "halfspaces_frame": planes, "shape": shape})
            names.add(solid.name)
        self.required_link_names = tuple(dict.fromkeys(row["frame"] for row in self._solids))
        baseline = checker.query(joint_names=self.baseline_joint_names,
                                 joint_positions=self._baseline_q,
                                 T_reference_base=self._base, pairs=(), frames=self.required_link_names)
        if baseline["joint_limit_violations"]:
            raise ValueError("The declared valid initial model violates its native joint limits")
        self._baseline_poses = {name: np.asarray(baseline["frame_poses"][name], float)
                                for name in self.required_link_names}
        world_shapes = []
        for row in self._solids:
            pose = self._baseline_poses[row["frame"]]
            row["baseline_pose"] = pose.copy()
            planes = row["halfspaces_frame"].copy()
            planes[:, :3] = planes[:, :3] @ pose[:3, :3].T
            planes[:, 3] -= planes[:, :3] @ pose[:3, 3]
            row["baseline_halfspaces_world"] = planes
            world_shape = self._check_native(row["shape"].transform(pose[:3]))
            if world_shape.is_empty():
                raise ValueError("A native baseline component collapsed under its reference transform")
            world_shapes.append(world_shape)
        self._baseline = self._boolean(world_shapes, manifold.OpType.Add)
        if self._baseline.is_empty():
            raise ValueError("The supplied native baseline must have a nonempty solid volume")

    @property
    def baseline_joint_positions(self):
        return _readonly(self._baseline_q)

    def _named(self, names, positions):
        names, values = tuple(names), np.asarray(positions)
        if (len(names) != len(set(names)) or set(names) != set(self.baseline_joint_names)
                or values.shape != (len(names),) or values.dtype.kind not in "iuf"
                or not np.isfinite(values).all()):
            raise ValueError("Coverage coordinates must name every actual robot joint exactly once")
        return np.asarray([values[names.index(name)] for name in self.baseline_joint_names], float)

    def _check_native(self, shape):
        status = shape.status()
        if status != manifold.Error.NoError:
            raise RuntimeError(f"Native solid operation failed: {status}")
        if shape.num_tri() > _MAX_RESULT_FACES:
            raise CoverageBudgetExceeded("Native solid result exceeds the face budget")
        tolerance = float(shape.get_tolerance())
        if not np.isfinite(tolerance) or tolerance < 0:
            raise RuntimeError("Native solid operation returned invalid numerical metadata")
        self._maximum_native_tolerance = max(self._maximum_native_tolerance, tolerance)
        return shape

    def _mesh(self, vertices, faces):
        return self._check_native(manifold.Manifold(manifold.Mesh64(
            np.ascontiguousarray(vertices, dtype=np.float64),
            np.ascontiguousarray(faces, dtype=np.uint64))))

    def _boolean(self, shapes, operation):
        # Bound potential intersecting face-pair work before handing it to the
        # maintained native engine. AABB-disjoint operands cannot create such
        # pairs. Work is intentionally refused instead of trying an unbounded
        # all-grid union or weakening the geometric predicate.
        shapes = list(shapes)
        counts = np.asarray([shape.num_tri() for shape in shapes], dtype=np.int64)
        bounds = np.asarray([shape.bounding_box() for shape in shapes], float).reshape(-1, 2, 3)
        pairs = 0
        for index in range(len(shapes)):
            overlap = np.all(bounds[index, 0] <= bounds[:index, 1], axis=1)
            overlap &= np.all(bounds[:index, 0] <= bounds[index, 1], axis=1)
            pairs += int(counts[index]) * int(counts[:index][overlap].sum())
        estimated = int(self._free.nbytes + self._free_boxes.nbytes
                        + counts.sum() * 2048 + pairs * 128)
        self._maximum_work_estimate = max(self._maximum_work_estimate, estimated)
        if estimated > self.max_working_bytes:
            raise CoverageBudgetExceeded("Native Boolean face-pair work exceeds the CPU budget")
        return self._check_native(manifold.Manifold.batch_boolean(shapes, operation))

    def guidance_geometry(self):
        """Return copies for a caller's differentiable, non-certifying guidance."""
        return {"reference_frame": self.reference_frame,
                "baseline_joint_names": list(self.baseline_joint_names),
                "baseline_joint_positions": self._baseline_q.copy(),
                "initialization_source": self.initialization_source,
                "solids": [{key: value.copy() if isinstance(value, np.ndarray) else value
                            for key, value in row.items() if key != "shape"}
                           for row in self._solids]}

    def describe(self):
        return {"predicate": "V(q) minus (V0 union F) is empty",
                "reference_frame": self.reference_frame,
                "initialization_source": self.initialization_source,
                "initial_validity_is_assumption": True,
                "hidden_environment_validated": False,
                "new_free_cells_added": 0,
                "native_solid_names": [row["name"] for row in self._solids],
                "required_link_names": list(self.required_link_names),
                "baseline_joint_names": list(self.baseline_joint_names),
                "baseline_joint_positions": self._baseline_q.tolist(),
                "T_reference_base": self._base.tolist(),
                "free_cells": int(self._free.sum()), "free_boxes": len(self._free_boxes),
                "max_working_bytes": self.max_working_bytes, "max_queries": self.max_queries,
                "queries": self._queries,
                "maximum_native_work_estimate_bytes": self._maximum_work_estimate,
                "maximum_native_tolerance_m": self._maximum_native_tolerance,
                "numerical_tolerance_is_error_bound": False,
                "acceptance_uses_volume_threshold": False,
                "continuous_path_certified": False,
                "scope": "native-model whole-volume floating-point state queries only"}

    def _witness(self, bad, row, pose):
        """Find an interior unknown point; a Boolean cut-face corner is insufficient.

        Ray intervals inside the native result supply candidates. Independent
        convex/box membership checks confirm that a returned point is inside
        this current robot component and outside both V0 and F. Failure to
        obtain a numerically separated point leaves the witness absent; it
        never turns the already nonempty Boolean result into acceptance.
        """
        mesh = bad.to_mesh64()
        vertices, faces = np.asarray(mesh.vert_properties)[:, :3], np.asarray(mesh.tri_verts)
        extent = float(np.linalg.norm(vertices.max(0) - vertices.min(0)))
        if not np.isfinite(extent) or extent <= 0:
            return None
        for face in faces[:16]:
            triangle = vertices[face]
            normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
            length = np.linalg.norm(normal)
            if not np.isfinite(length) or length == 0:
                continue
            normal /= length
            centre = triangle.mean(axis=0)
            hits = bad.ray_cast(centre + normal * (2 * extent), centre - normal * (2 * extent))
            for first, second in zip(hits[::2], hits[1::2]):
                point = (np.asarray(first.position) + np.asarray(second.position)) / 2
                local = (point - pose[:3, 3]) @ pose[:3, :3]
                planes = row["halfspaces_frame"]
                if np.any(planes[:, :3] @ local + planes[:, 3] > 0):
                    continue
                if any(np.all(old["baseline_halfspaces_world"][:, :3] @ point
                              + old["baseline_halfspaces_world"][:, 3] <= 0)
                       for old in self._solids):
                    continue
                if len(self._free_boxes) and np.any(np.all(point >= self._free_boxes[:, 0], axis=1)
                                                   & np.all(point <= self._free_boxes[:, 1], axis=1)):
                    continue
                return {"solid": row["name"], "frame": row["frame"],
                        "point_reference": point.tolist(), "point_frame": local.tolist()}
        return None

    def check_state(self, *, joint_names, joint_positions, entrance=False):
        """Check one complete named state; only the explicit entrance is canonicalized.

        Native float32 FK may represent the measured double q0 only by its
        exact float32 round trip. At the separately identified entrance this
        one vector maps back to q0. No nearby vector, later sample, partial
        joint layout, or tolerance band gets the same treatment.
        """
        q = self._named(joint_names, joint_positions)
        canonicalized = False
        if type(entrance) is not bool:
            raise ValueError("Entrance identity must be explicit boolean metadata")
        if entrance:
            if np.array_equal(q, self._baseline_q):
                pass
            elif np.array_equal(q, self._baseline_float32):
                q, canonicalized = self._baseline_q.copy(), True
            else:
                raise ValueError("Entrance coordinates are neither measured q0 nor its exact float32 representation")
        self._queries += 1
        report = {"passed": False, "failure": None, "canonicalized_initial_float32": canonicalized,
                  "joint_names": list(self.baseline_joint_names), "joint_positions": q.tolist(),
                  "unknown_volume_m3": None, "witness": None,
                  "continuous_path_certified": False}
        if self._queries > self.max_queries:
            report["failure"] = "state_query_budget"
            return report
        try:
            state = self._checker.query(joint_names=self.baseline_joint_names, joint_positions=q,
                                        T_reference_base=self._base, pairs=(), frames=self.required_link_names)
            if state["joint_limit_violations"]:
                report.update(failure="native_joint_limits", joint_limit_violations=state["joint_limit_violations"])
                return report
            for row in self._solids:
                pose = np.asarray(state["frame_poses"][row["frame"]], float)
                # Exact pose identity is an exact set identity, including
                # unchanged root bodies while other joints move. It avoids
                # unnecessary Booleans, not unknown queries for new volume.
                if np.array_equal(pose, row["baseline_pose"]):
                    continue
                candidate = self._check_native(row["shape"].transform(pose[:3]))
                if candidate.is_empty():
                    raise RuntimeError("A native candidate component collapsed under its reference transform")
                bad = self._boolean((candidate, self._baseline), manifold.OpType.Subtract)
                if bad.is_empty():
                    continue
                bounds = np.asarray(bad.bounding_box(), float).reshape(2, 3)
                selected = self._free_boxes[
                    np.all(self._free_boxes[:, 0] <= bounds[1], axis=1)
                    & np.all(bounds[0] <= self._free_boxes[:, 1], axis=1)]
                for start in range(0, len(selected), _BOX_BATCH):
                    boxes = [self._mesh(np.where(_BOX_SIGNS, high, low), _BOX_FACES)
                             for low, high in selected[start:start + _BOX_BATCH]]
                    bad = self._boolean((bad, *boxes), manifold.OpType.Subtract)
                    if bad.is_empty():
                        break
                if not bad.is_empty():
                    volume = float(bad.volume())
                    if not np.isfinite(volume):
                        raise RuntimeError("Native new-volume query returned a nonfinite volume diagnostic")
                    report.update(failure="new_volume_in_unknown", solid=row["name"],
                                  unknown_volume_m3=volume,
                                  volume_scope="first failing native component; diagnostic only",
                                  unknown_boundary_faces=int(bad.num_tri()),
                                  witness=self._witness(bad, row, pose))
                    return report
            report.update(passed=True, unknown_volume_m3=0.)
            return report
        except CoverageBudgetExceeded as error:
            report.update(failure="native_work_budget", error=str(error))
            return report
        except (RuntimeError, ValueError) as error:
            report.update(failure="native_geometry_error", error=str(error))
            return report
        finally:
            report["numerics"] = {"maximum_native_tolerance_m": self._maximum_native_tolerance,
                                  "tolerance_is_error_bound": False,
                                  "volume_threshold_used": False}
# END unknown coverage never mutates observations or grants collision permission.
