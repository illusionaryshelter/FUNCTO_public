"""One batched reverse query for the unchanged, selected constraint vector.

The existing Pinocchio bridge already implements its analytic backward with
batch-preserving tensor contractions. Torch's maintained batched-VJP API can
therefore evaluate all submitted rows together. This changes derivative work
and storage, while preserving each row, its physical units and its ordering.
"""

import torch


# BEGIN batch the original first derivative without changing the geometry rows.
def task_jacobian(values, controls, *, check_request, admit_work):
    """Differentiate the supplied Torch graph in its original column layout.

    ``admit_work`` counts the full batched workspace before the output-space
    basis is allocated. ``check_request`` checks the same planning request at
    both boundaries of this single native reverse query. Constant rows have
    an explicit zero dependence and thus retain an exact zero derivative.
    The original solver invokes this API only for a nonempty selected vector.
    Coal rows use the original query's approximate closest-feature direction;
    batching preserves that graph and does not improve native query accuracy.
    """
    check_request()
    admit_work()
    connected = values + controls.sum() * 0.
    basis = torch.eye(values.numel(), dtype=values.dtype, device=values.device)
    derivative, = torch.autograd.grad(connected, controls, grad_outputs=basis,
                                      is_grads_batched=True)
    check_request()
    return derivative.detach().numpy().reshape(values.numel(), controls.numel())
# END every row uses the same analytic Pinocchio and scalar geometry derivatives.


# BEGIN make the batched reverse allocation explicit at both solver admissions.
def batch_derivative_storage_bytes(*, rows, variables, samples, frames, joints,
                                  geometry_workspace):
    """Return batch storage in addition to the original pending geometry.

    Counts describe float64 CPU arrays. ``samples`` is the original task query
    count; ``geometry_workspace`` is GeometryConstraints.workspace_bytes().
    Its forward entity/permission/scalar envelopes supply dimensions for batched
    adjoints, which are new allocations rather than duplicated forward data.

    Replace the original solver's scalar ``pinocchio_reverse_buffers`` with
    this dictionary. Keep the original pending geometry dictionary once,
    removing its ``additional_largest_pinocchio_reverse`` entry because this
    dictionary counts the complete reverse allocation at both boundaries. The
    returned dense Jacobian is listed because it coexists with a previous
    cached matrix during a new derivative callback. No scene samples, native
    model, TSDF support mask or Python witness record is multiplied by rows.
    """
    rows, variables, samples, frames, joints = map(int,
        (rows, variables, samples, frames, joints))
    # A task-free problem still differentiates its scalar objective. It needs
    # one reverse direction while allocating no task basis or task Jacobian.
    directions = max(1, rows)
    memory = {
        "batched_output_basis_and_connected_values": 8 * (rows * rows + rows),
        "batched_returned_dense_jacobian": 8 * rows * variables,
        # Same largest Pinocchio product, spatial covectors and joint sum as
        # the maintained scalar counter, with one leading derivative-row axis.
        "batched_base_pinocchio_reverse": 8 * directions * (
            samples * (6 * frames * joints + 12 * frames + joints) + 2),
        # Pose and named q/dq adjoints coexist with their accumulated inputs.
        "batched_base_pose_and_curve_adjoints": 8 * directions * (
            samples * (14 * frames + 4 * joints) + variables),
    }
    if geometry_workspace:
        memory["batched_added_geometry_pinocchio_reverse"] = directions * (
            geometry_workspace["additional_largest_pinocchio_reverse"])
        memory["batched_geometry_pose_and_permission_adjoints"] = directions * (
            geometry_workspace["entity_pose_transforms_and_matrices"]
            + geometry_workspace["finite_permission_transform_graph"])
        # Only scalar tensor slots need a derivative-row axis; the original
        # Python witness provenance remains shared by all derivative rows.
        memory["batched_witness_scalar_adjoints"] = directions * (
            geometry_workspace["witness_scalar_graphs"])
    return memory
# END forward scene/model storage continues to be charged by its original owner.
