"""Support halfspaces implied by retained position balls at the same query.

For a TCP E, carrier C and functional event F, a material point satisfies
    x = p_F + R_E (R_EC c + t_EC - t_EF).
The proper rotation R_E preserves the norm of the parenthesized vector. A
retained ball ||p_F-p_target|| <= epsilon therefore implies
    a.T x + b <= a.T p_target + b + ||a|| (epsilon + radius).
No orientation bound is needed. Material coordinates, event progress and
halfspaces come from the ordinary task declaration, not a task identifier.

The static bound uses mpmath's installed interval arithmetic for the frame
composition, dot products and norms. The final positive products and sums
round upward in NumPy. This avoids an unexplained multiple of machine epsilon
when deciding the sign of a nearly tangent halfspace. Only a strictly negative
upper endpoint removes a solver row; original task data and scores remain.
"""

from time import perf_counter

from mpmath import iv
import numpy as np
import torch

from manipulation.planning_problem import _first_event_query, _quaternion


def _vector(values):
    """Regard supplied binary floats as exact endpoints, not decimal strings."""
    return tuple(iv.mpf(float(value)) for value in values)


def _dot(first, second):
    """Evaluate a short scalar product with outward interval rounding."""
    return sum((a*b for a, b in zip(first, second)), iv.mpf(0))


def _norm(vector):
    """The interval square, unlike x*x, handles intervals crossing zero."""
    return iv.sqrt(sum((value**2 for value in vector), iv.mpf(0)))


def _upper(value):
    """Enclose an interval endpoint even if its precision exceeds float64."""
    return np.nextafter(float(value.b), np.inf)


def support_halfspace_bounds(requirement, event, material_indices, numerical):
    """Return body/footprint upper bounds in their original material order.

    T_EC and t_EF are fixed task transforms; target/origin are in the calibrated
    world frame. The event's functional rotation cancels from this norm bound.
    The canonical quaternion is the same Rotation.from_matrix representation
    used by PlanningProblem. Normalizing it expresses that rigid rotation in
    interval arithmetic. All matrix and vector dimensions follow the typed
    SupportRequirement/PoseEventRequirement interfaces.
    """
    # BEGIN compute the radius about the functional origin in TCP coordinates.
    q = _vector(_quaternion(requirement.tool_from_carrier))
    length = _norm(q)
    w, x, y, z = (value/length for value in q)
    rotation = ((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)),
                (2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)),
                (2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)))
    offset = tuple(a-b for a, b in zip(_vector(requirement.tool_from_carrier[:3, 3]),
                                        _vector(event.tool_from_function[:3, 3])))
    radii = []
    for point in requirement.body_points[material_indices]:
        material = _vector(point)
        radii.append(_upper(_norm(tuple(_dot(row, material)+shift
                                        for row, shift in zip(rotation, offset)))))
    radii = np.nextafter(np.asarray(radii)+event.position_tolerance_m, np.inf)
    # END no inverse functional rotation or orientation-specific branch is used.

    # BEGIN express the producer's existing body/footprint rows as halfspaces.
    surface = requirement.surface
    center = tuple(a-b for a, b in zip(_vector(event.target[:3, 3]), _vector(surface.origin)))
    planes = [tuple(-value for value in _vector(surface.normal))]
    offsets = [-iv.mpf(float(numerical.position_m))]
    basis = tuple(_vector(row) for row in surface.basis)
    for equation in surface._hull.equations:
        edge = _vector(equation[:2])
        planes.append(tuple(_dot(row, edge) for row in basis))
        offsets.append(iv.mpf(float(equation[2]))+iv.mpf(float(requirement.margin))
                       -iv.mpf(float(numerical.position_m)))
    centers = np.asarray([_upper(_dot(normal, center)+offset_)
                           for normal, offset_ in zip(planes, offsets)])
    norms = np.asarray([_upper(_norm(normal)) for normal in planes])
    products = np.nextafter(radii[:, None]*norms[None], np.inf)
    upper = np.nextafter(centers[None]+products, np.inf)
    return {'body': upper[:, 0], 'footprint': upper[:, 1:].ravel()}
    # END nonunit world normals retain their explicit norm and metric units.


def implied_support_rows(computation, row_keys, analytic, blocks):
    """Select redundant rows using only actual retained required position rows.

    row_keys describes the pre-presolve variable task rows, after the ordinary
    fixed-row checks. A prior or inactive event has no position row here.
    Matching the producer's earliest physical query handles progress plateaus
    and duplicate time samples without a separate time-tolerance policy.

    Returns a Boolean mask over those original rows and compact certificates:
    condition identity, material source rows, the retained position premise,
    query column and the largest strictly negative bound of each group.
    """
    started = perf_counter()
    removed = np.zeros(len(row_keys), dtype=bool)
    premises = {}

    def query_column(requirement):
        mask = _first_event_query(computation.progress, requirement.progress, computation.times)[0]
        return int(mask.to(torch.int64).argmax()) if bool(mask.any()) else None

    for row, key in enumerate(row_keys):
        parts = key[2].split(':')
        if key[1] == 'inequality' and parts[0] == 'function_pose' and parts[2] == 'position':
            index = int(parts[1])
            event = computation.problem.pose_events[index]
            column = query_column(event)
            if column is not None:
                premises.setdefault(column, []).append((index, event))

    factors = {}
    for row, (kind, name, local) in analytic.items():
        if kind == 1:
            factors.setdefault(name, ([], []))[0].append(row)
            factors[name][1].append(local)

    certificates = []
    for index, support in enumerate(computation.problem.supports):
        names = [(component, f'support:{index}:{component}')
                 for component in ('body', 'footprint')
                 if f'support:{index}:{component}' in factors]
        column = query_column(support)
        if not names or column not in premises:
            continue
        material = computation.support_material_indices[index]
        choices = premises[column]
        bounds = [support_halfspace_bounds(support, event, material, computation.problem.numerical)
                  for _, event in choices]
        for component, name in names:
            rows, locals_ = (np.asarray(value, dtype=np.int64) for value in factors[name])
            upper = np.stack([value[component][locals_] for value in bounds])
            chosen = upper.argmin(axis=0)
            best = upper[chosen, np.arange(len(rows))]
            redundant = best < 0.
            removed[rows[redundant]] = True
            factor = blocks[1][name][1]
            for premise, (event_index, _) in enumerate(choices):
                selected = redundant & (chosen == premise)
                if selected.any():
                    certificates.append(dict(condition=name, query_column=column,
                        position_condition=f'function_pose:{event_index}:position',
                        source_rows=factor.source_rows[locals_[selected]].tolist(),
                        maximum_upper_bound_m=float(best[selected].max())))
    return removed, dict(removed_task_rows=int(removed.sum()), certificates=certificates,
        elapsed_seconds=perf_counter()-started,
        scope='Support halfspaces implied by retained position balls at the same physical query')
