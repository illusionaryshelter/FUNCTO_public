"""Volume anchors and finite support geometry of the original observed set.

The field coordinates, denoted B, remain the coordinates of the RGB-D TSDF.
Its represented material is the nonpositive trilinear subset of cells whose
eight samples were observed. A volume centre introduces only the translated
coordinate frame G; it does not alter that material or its observation support.
"""

# BEGIN one observed field supplies volume anchors and directional support bounds.
import numpy as np

from .goals import _nearest_feasible, _pose
from .implicit_distance import CORNERS, trilinear
from .observed_collision import query_minimum_height


def estimate_occupied_volume_center(field):
    """Estimate the represented occupied-volume centre in field coordinates B.

    Every original candidate cell is divided in two along each axis. The eight
    subcell midpoints have equal volume ``(voxel_size / 2)**3``; only midpoints
    with the original trilinear value <= 0 contribute to the volume and first
    moment. ``field.cells`` already contains only fully observed cells. Its
    cells are candidates, so their whole volumes cannot be counted as occupied.

    The fixed midpoint quadrature estimates volume; it is independent of the
    TSDF observation weights and of the optimizer's material-point sampling.
    A zero sampled volume has no centre and raises ``ValueError``. The result
    records the quadrature spacing and sample count alongside the estimate.
    """
    midpoints = .25 + .5 * CORNERS
    sample_volume = (field.voxel_size / 2) ** 3
    first_moment = np.zeros(3)
    sample_count = 0
    # Match the original field's bounded cell batches. No dense refined grid
    # or additional geometry index is retained by the centre calculation.
    for start in range(0, len(field.cells), 1024):
        stop = min(start + 1024, len(field.cells))
        values, _, _ = trilinear(field._cell_values[start:stop, None, :], midpoints)
        points = (field.cell_origins[start:stop, None, :]
                  + field.voxel_size * midpoints)
        material = points[values <= 0]
        sample_count += len(material)
        first_moment += sample_volume * material.sum(axis=0)
    if sample_count == 0:
        raise ValueError("Observed occupied-volume quadrature has zero volume")
    volume = sample_count * sample_volume
    return {
        "center_local": (first_moment / volume).tolist(),
        "volume_m3": float(volume),
        "quadrature_spacing_m": float(field.voxel_size / 2),
        "occupied_sample_count": int(sample_count),
    }


def support_extrema(field, pose, surface, *, precision_m):
    """Bound normal minima and finite-patch edge maxima of the original field.

    ``pose`` maps field coordinates B into the same reference frame as
    ``surface``. Every scalar projection is relative to ``surface.origin``.
    The normal result is the original minimum-height query record. An edge
    with planar equation ``a @ xy + b <= 0`` has reference-frame direction
    ``surface.basis @ a``. Its maximum interval is obtained by negating and
    exchanging the minimum query's bounds along the opposite direction.

    Witnesses are feasible represented points in B, furnishing the numerical
    extrema's inner bounds and rigid-motion derivatives. They need not be
    exact extremizers. Each original query status is retained, including a
    missing observation's ``None`` bounds and witness. Edge offsets are kept
    separately so consumers use the original finite halfspaces.
    """
    pose = _pose(pose)
    normal = query_minimum_height(
        field, pose, surface.origin, surface.normal, precision_m=precision_m)
    equations = surface._hull.equations
    directions = (surface.basis @ equations[:, :2].T).T
    queries = [query_minimum_height(
        field, pose, surface.origin, -direction, precision_m=precision_m)
        for direction in directions]
    return {
        "normal": normal,
        "footprint": {
            "directions_world": directions.tolist(),
            "offsets": equations[:, 2].tolist(),
            "lower_m": [None if q["upper_m"] is None else -q["upper_m"] for q in queries],
            "upper_m": [None if q["lower_m"] is None else -q["lower_m"] for q in queries],
            "witnesses_local": [q["witness_local"] for q in queries],
            "statuses": [q["status"] for q in queries],
            "visited_boxes": [q.get("visited_boxes") for q in queries],
        },
    }


def place_observed_on(preferred_center_pose, field, surface, *, center_local,
                      margin, precision_m):
    """Keep the preferred rotation and project its centre onto finite support.

    The input pose describes G, whose origin is ``center_local`` in B and
    whose axes equal B's axes. The returned pose describes B. The desired
    centre's tangent coordinates are projected onto the support polygon using
    continuous occupied-set edge bounds and the existing translation solver.
    The normal centre height is the negative lower bound of
    ``min n @ R @ (p - center_local)``. Thus the represented minimum gap is
    bounded by zero and the original height query's numerical interval width.

    The evidence preserves the centred shape queries, final gap interval,
    and final edge inequality intervals including ``margin``. An unresolved
    finite query remains unresolved evidence; this initializer adds no task
    acceptance rule. A missing occupied set cannot define a support placement.
    """
    preferred = _pose(preferred_center_pose)
    center = np.asarray(center_local, dtype=float)
    if not np.isfinite(margin) or margin < 0:
        raise ValueError("Support margin must be finite and nonnegative")
    rotation = preferred[:3, :3]
    # This pose places G at the support origin. Its projections are exactly
    # those of R(p-c), while all queries continue to use the original field B.
    centered_pose = preferred.copy()
    centered_pose[:3, 3] = surface.origin - rotation @ center
    extrema = support_extrema(field, centered_pose, surface, precision_m=precision_m)
    normal, footprint = extrema["normal"], extrema["footprint"]
    if normal["lower_m"] is None:
        raise ValueError("Support placement requires represented observed occupancy")
    equations = surface._hull.equations
    matrix, offset = equations[:, :2], equations[:, 2]
    upper = -offset - margin - np.asarray(footprint["upper_m"])
    preferred_xy = (preferred[:3, 3] - surface.origin) @ surface.basis
    xy = _nearest_feasible(preferred_xy, matrix, upper)
    height = -normal["lower_m"]
    center_world = surface.origin + surface.basis @ xy + height * surface.normal
    pose = preferred.copy()
    pose[:3, 3] = center_world - rotation @ center
    edge_translation = matrix @ xy + offset + margin
    footprint = {
        **footprint,
        "inequality_lower_m": (edge_translation + footprint["lower_m"]).tolist(),
        "inequality_upper_m": (edge_translation + footprint["upper_m"]).tolist(),
    }
    evidence = {
        "geometry_source": surface.source,
        "precision_m": float(precision_m),
        "margin_m": float(margin),
        "center_local": center.tolist(),
        "center_world": center_world.tolist(),
        "center_xy": xy.tolist(),
        "normal": normal,
        "minimum_gap_interval_m": [float(normal["lower_m"] + height),
                                    float(normal["upper_m"] + height)],
        "footprint": footprint,
    }
    return pose, evidence
# END centre estimation and support placement preserve the original observed set.
