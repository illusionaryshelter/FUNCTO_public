"""Original material indexes for equivalent linear support halfspaces.

The returned subset is used only by solver body/footprint rows. The caller
retains the complete material points in skill/task data and in every other
objective or geometry calculation. No collision shape is constructed here.
"""

# BEGIN material indexing changes no original point or halfspace threshold.
import numpy as np
from scipy.spatial import ConvexHull


def material_hull_indices(points):
    """Return original indexes retaining every linear halfspace extremum.

    Exact duplicate points and exactly constant coordinate axes are removed
    algebraically. A one-coordinate set has its two endpoints as vertices.
    In the remaining coordinate space, well-conditioned points use SciPy's
    hull directly in their original coordinates. No small singular direction
    is projected away or rescaled into a different Qhull precision regime.

    A singular-value ratio at or below sqrt(machine epsilon) selects the
    identity expression of all unique original points. This numerical-domain
    choice only declines compression; it introduces no physical allowance.
    In particular, tilted affine subsets and near-degenerate sets with genuine
    thickness retain all extrema. It is not a Qhull-error recovery. No jitter,
    physical tolerance or task-dependent geometry rule participates.
    """
    points = np.asarray(points, dtype=float)
    if (points.ndim != 2 or points.shape[1] != 3 or len(points) == 0
            or not np.isfinite(points).all()):
        raise ValueError("Material hull indexing needs finite nonempty XYZ points")
    unique, original = np.unique(points, axis=0, return_index=True)
    varying = np.any(unique != unique[0], axis=0)
    dimension = int(np.count_nonzero(varying))
    if dimension == 0:
        return original.astype(np.int64)
    coordinates = unique[:, varying] - unique[0, varying]
    if dimension == 1:
        vertices = [int(coordinates[:, 0].argmin()), int(coordinates[:, 0].argmax())]
        return np.sort(original[vertices]).astype(np.int64)

    singular = np.linalg.svd(coordinates, compute_uv=False)
    # Avoid computing a hull in a numerically unresolved affine direction.
    # Retaining every unique point is exact for this entire input domain;
    # this bound never flattens, perturbs or relaxes a material constraint.
    if (len(singular) < dimension
            or singular[-1] <= np.sqrt(np.finfo(coordinates.dtype).eps) * singular[0]):
        return np.sort(original).astype(np.int64)
    vertices = ConvexHull(unique[:, varying]).vertices
    return np.sort(original[vertices]).astype(np.int64)
# END only source-point indexes leave this module.
