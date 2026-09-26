"""Local geometric goals for object motion, independent of tool landmarks.

Coordinates and lengths are expressed in the trajectory's reference frame and
metres. A containment region describes *known free interior*, never an object's
outer bounding box. These checks establish geometry, not force stability.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import LinearConstraint, linprog, minimize
from scipy.spatial import ConvexHull, QhullError


def _points(value, name, minimum=1):
    points = np.asarray(value, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < minimum:
        raise ValueError(f"{name} must contain at least {minimum} XYZ points")
    if not np.isfinite(points).all():
        raise ValueError(f"{name} contains non-finite coordinates")
    return points


def _pose(value):
    pose = np.asarray(value, dtype=float)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError("Goal pose must be a finite 4x4 matrix")
    if not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-8):
        raise ValueError("Invalid homogeneous pose row")
    rotation = pose[:3, :3]
    if (not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(rotation), 1, atol=1e-6)):
        raise ValueError("Goal rotation must belong to SO(3)")
    return pose.copy()


@dataclass
class SupportSurface:
    """A finite convex support patch: origin, tangent basis, outward normal.

    ``boundary`` is a convex set of XY coordinates in ``basis``. It describes
    observed/known support area, rather than extending a fitted plane forever.
    """

    origin: np.ndarray
    basis: np.ndarray
    normal: np.ndarray
    boundary: np.ndarray
    source: str = "observed_support_patch"

    def __post_init__(self):
        self.origin = np.asarray(self.origin, float)
        self.basis = np.asarray(self.basis, float)
        self.normal = np.asarray(self.normal, float)
        self.boundary = np.asarray(self.boundary, float)
        if (self.origin.shape != (3,) or self.normal.shape != (3,)
                or self.basis.shape != (3, 2)):
            raise ValueError("Invalid support coordinate dimensions")
        if not all(np.isfinite(v).all() for v in (
                self.origin, self.normal, self.basis, self.boundary)):
            raise ValueError("Support geometry must be finite")
        axes = np.column_stack((self.basis, self.normal))
        if not np.allclose(axes.T @ axes, np.eye(3), atol=1e-6):
            raise ValueError("Support axes must be orthonormal")
        if self.boundary.ndim != 2 or self.boundary.shape[1] != 2:
            raise ValueError("Support boundary must contain planar XY vertices")
        try:
            self._hull = ConvexHull(self.boundary)
        except QhullError as error:
            raise ValueError("Support patch has no usable area") from error

    def to_dict(self):
        return {"origin": self.origin.tolist(), "basis": self.basis.tolist(),
                "normal": self.normal.tolist(), "boundary": self.boundary.tolist(),
                "source": self.source}


@dataclass
class InteriorRegion:
    """Known convex free space, expressed by unit-normal halfspaces A x + b<=0.

    Vertices/halfspaces must refer to the interior free region supplied by
    observed cavity geometry or upstream scene geometry. An outer object mask
    cannot establish this region. ``source`` remains in the output evidence.
    """

    halfspaces: np.ndarray
    source: str

    def __post_init__(self):
        self.halfspaces = np.asarray(self.halfspaces, float)
        if (self.halfspaces.ndim != 2 or self.halfspaces.shape[1] != 4
                or len(self.halfspaces) < 4 or not np.isfinite(self.halfspaces).all()):
            raise ValueError("Interior needs at least four finite halfspaces")
        norms = np.linalg.norm(self.halfspaces[:, :3], axis=1)
        if (norms < 1e-10).any() or not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("Interior needs nonzero normals and an evidence source")
        self.halfspaces = self.halfspaces / norms[:, None]
        # Halfspace count alone does not establish a bounded, nonempty cavity.
        # A small LP finds an interior ball; axis extrema reject open regions.
        matrix, offset = self.halfspaces[:, :3], self.halfspaces[:, 3]
        ball = linprog([0., 0., 0., -1.], A_ub=np.column_stack((matrix, np.ones(len(matrix)))),
                       b_ub=-offset, bounds=[(None, None)]*3 + [(0., None)], method="highs")
        if not ball.success or ball.x[3] <= 1e-9:
            raise ValueError("Interior must enclose a nonempty finite volume")
        for direction in np.concatenate((np.eye(3), -np.eye(3))):
            extreme = linprog(direction, A_ub=matrix, b_ub=-offset,
                              bounds=[(None, None)]*3, method="highs")
            if not extreme.success:
                raise ValueError("Interior must be bounded in every direction")

    @classmethod
    def from_vertices(cls, vertices, source):
        points = _points(vertices, "Interior vertices", minimum=4)
        try:
            hull = ConvexHull(points)
        except QhullError as error:
            raise ValueError("Interior vertices do not enclose a volume") from error
        return cls(hull.equations, source)

    def to_dict(self):
        return {"halfspaces": self.halfspaces.tolist(), "source": self.source}


def _nearest_feasible(preferred, matrix, upper):
    """Project a desired translation onto finite linear geometric conditions."""
    preferred = np.asarray(preferred, float)
    constraint = LinearConstraint(matrix, -np.inf, upper)
    result = minimize(lambda x: 0.5 * np.sum((x - preferred)**2), preferred,
                      jac=lambda x: x-preferred, constraints=[constraint],
                      method="SLSQP", options={"ftol": 1e-12, "maxiter": 200})
    if (not result.success or not np.isfinite(result.x).all()
            or np.max(matrix @ result.x-upper) > 1e-7):
        raise ValueError("Object geometry does not fit the requested target region")
    return result.x


def place_on(preferred_pose, body_points, surface, *, margin=0.0):
    """Preserve desired orientation; find the closest supported translation.

    The projected footprint of all supplied object points must lie inside the
    support patch. This sufficient geometric condition does not prove static
    stability or infer unobserved parts of the object.
    """
    pose = _pose(preferred_pose)
    body = _points(body_points, "Object geometry", minimum=3)
    if not np.isfinite(margin) or margin < 0:
        raise ValueError("Support margin must be finite and nonnegative")
    rotated = body @ pose[:3, :3].T
    footprint = rotated @ surface.basis
    equations = surface._hull.equations
    matrix, offset = equations[:, :2], equations[:, 2]
    upper = -offset-margin-np.max(footprint @ matrix.T, axis=0)
    preferred_xy = (pose[:3, 3]-surface.origin) @ surface.basis
    xy = _nearest_feasible(preferred_xy, matrix, upper)
    height = -np.min(rotated @ surface.normal)
    pose[:3, 3] = surface.origin + surface.basis @ xy + height * surface.normal
    diagnostics = check_on(pose, body, surface)
    if diagnostics["support_gap_m"] > 1e-7 or diagnostics["boundary_violation_m"] > 1e-7:
        raise ValueError("Support placement failed its independent geometric check")
    return pose, diagnostics


def check_on(pose, body_points, surface):
    pose = _pose(pose)
    world = _points(body_points, "Object geometry") @ pose[:3, :3].T + pose[:3, 3]
    local = world-surface.origin
    xy = local @ surface.basis
    equations = surface._hull.equations
    boundary_values = xy @ equations[:, :2].T + equations[:, 2]
    return {"support_gap_m": float(abs(np.min(local @ surface.normal))),
            "boundary_violation_m": float(max(0, boundary_values.max())),
            "minimum_boundary_clearance_m": float(-boundary_values.max()),
            "geometry_source": surface.source, "checked_points": len(world)}


def place_inside(preferred_pose, body_points, region, *, margin=0.0):
    """Fit the whole supplied object geometry within a known free region."""
    pose = _pose(preferred_pose)
    body = _points(body_points, "Object geometry", minimum=4)
    if not np.isfinite(margin) or margin < 0:
        raise ValueError("Interior margin must be finite and nonnegative")
    rotated = body @ pose[:3, :3].T
    matrix, offset = region.halfspaces[:, :3], region.halfspaces[:, 3]
    upper = -offset-margin-np.max(rotated @ matrix.T, axis=0)
    pose[:3, 3] = _nearest_feasible(pose[:3, 3], matrix, upper)
    return pose, check_inside(pose, body, region)


def check_inside(pose, body_points, region):
    pose = _pose(pose)
    world = _points(body_points, "Object geometry") @ pose[:3, :3].T + pose[:3, 3]
    values = world @ region.halfspaces[:, :3].T + region.halfspaces[:, 3]
    return {"interior_violation_m": float(max(0, values.max())),
            "minimum_clearance_m": float(-values.max()),
            "geometry_source": region.source, "checked_points": len(world)}
