"""Fixed numerical coordinates for the existing free trajectory coefficients.

The complete reference Jacobian and the declared joint ranges define a
positive metric for each coefficient's joint block. Its Cholesky factor
changes numerical coordinates once, before IPOPT. Every value producer still
receives the original coefficients; no cost, tolerance or model is changed.
"""

# BEGIN a fixed invertible coordinate map preserves the original physical NLP.
from time import perf_counter

import numpy as np
from scipy.linalg import block_diag, cholesky, solve_triangular

from .reference_initialization import SymbolicKinematics


def coefficient_coordinates(computation, point):
    """Return S for C = C_initial + S z and its numerical definition.

    The matrix is block diagonal in the original free-control order. Each
    block uses the full reference's physical quadrature and declared position
    and rotation scales. The joint-range diagonal defines natural coordinates
    in directions that do not move the TCP. It belongs only to this reversible
    change of variables. IPOPT continues to approximate the full Lagrangian
    with its own limited-memory updates.
    """
    started = perf_counter()
    reference = computation.reference_objective
    declaration = reference.declaration
    complete_q, _, names, _, _ = point.candidate_state
    samples = complete_q.numpy()[0, computation.ordinary_indices]
    model = SymbolicKinematics(computation.kinematics.model, joint_names=names,
        fixed_positions={}, frame=computation.problem.tool_link,
        reference_from_base=computation.problem.base_pose,
        frame_from_tool=computation.problem.tool_transform)
    packed = np.asarray(model.jacobian.map(len(samples))(samples.T))
    jacobian = packed.reshape(12, len(samples), len(names)).transpose(1, 0, 2)
    columns = [names.index(name) for name in computation.joint_names]
    jacobian = jacobian[:, :, columns]
    scales = np.r_[np.full(3, 1/declaration.position_scale_m),
                   np.full(9, 1/(np.sqrt(2)*declaration.rotation_scale_rad))]
    jacobian = jacobian*scales[None, :, None]
    weights = (reference.weights.numpy()*computation.constraint_weight
               *declaration.relative_weight)
    local_metric = 2*np.einsum('nki,nkj,n->nij', jacobian, jacobian, weights)
    matrix, _ = computation.full_domain._curve_map.query(
        reference.physical_times.numpy()/computation.duration, 0)
    coefficients = matrix@computation.terminal.matrix
    joint_range = np.diff(computation.bounds.numpy()[0], axis=0)[0]
    natural_metric = np.diag(1/joint_range**2)
    metrics = (np.einsum('nr,nij->rij', coefficients**2, local_metric)
               +natural_metric[None])
    blocks = [solve_triangular(cholesky(metric, lower=False),
        np.eye(len(columns)), lower=False) for metric in metrics]
    transform = block_diag(*blocks)
    spectrum = np.linalg.eigvalsh(metrics)
    return transform, dict(kind='fixed_block_coefficient_coordinates',
        formula='C = C_initial + S z; S_block.T M_block S_block = I',
        metric='diag(1/joint_range^2) + 2 J_reference_block.T J_reference_block',
        block_count=len(blocks), block_size=len(columns),
        block_condition_numbers=(spectrum[:, -1]/spectrum[:, 0]).tolist(),
        minimum_metric_eigenvalue=float(spectrum.min()),
        maximum_metric_eigenvalue=float(spectrum.max()),
        elapsed_seconds=perf_counter()-started,
        scope='Numerical coordinates only; original objective and all constraints are unchanged')
# END no curvature callback or additional optimization problem is introduced.
