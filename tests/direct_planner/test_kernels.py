"""Small independent derivative checks for the two expression-level changes.

The maintained task fixtures provide the robot, mixed task declarations and
ordinary Torch value producer. The library's sparse assembly and
kinematic bridge are under test; no scene or full planner is constructed.
"""

import numpy as np
import pytest
from scipy.sparse import csr_matrix
import torch

from manipulation.direct_planner.differentiable_kinematics import DifferentiablePinKinematics
from manipulation.direct_planner.sparse_task import SparseTaskRows
from manipulation.direct_planner.support_linearization import linearized_robot_state, task_linearization
from manipulation.pin_kinematics import PinKinematics
from manipulation.unified_solver import task_jacobian
from .test_support_linearization import (
    finger_case, original_values_in_solver_units, pin_fixture, support_case,
)
from .test_sparse_computation import all_task_case


@pytest.mark.parametrize('analytic', [False, True])
def test_mixed_sparse_slots_match_independent_full_task_derivatives(analytic):
    """AD-only and split analytic blocks retain named columns and row order."""
    bound, initial = support_case(analytic=analytic)
    controls = torch.tensor(initial, dtype=torch.float64, requires_grad=True)
    robot, jacobians = linearized_robot_state(bound, *bound.states(controls)[:2])
    declarations = ({}, {})
    values, blocks = task_linearization(bound, controls, solver_rows_out=declarations,
        robot_state=robot, jacobians=jacobians,
        support_material_indices=bound.support_material_indices)
    rows = SparseTaskRows(bound, values, blocks, declarations, {'robot_state': robot})
    actual = csr_matrix((rows.jacobian_values(values, controls, blocks,
        check_request=lambda: None), rows.pattern.columns, rows.pattern.indptr),
        shape=rows.pattern.shape).toarray()
    reference_controls = torch.tensor(initial, dtype=torch.float64, requires_grad=True)
    reference = torch.cat(original_values_in_solver_units(bound, reference_controls,
        masks=({}, {})))[rows.source_indices]
    wanted = task_jacobian(reference, reference_controls,
        check_request=lambda: None, admit_work=lambda: None)
    np.testing.assert_allclose(rows.select_values(values).detach(), reference.detach(),
                               rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(actual, wanted, rtol=2e-12, atol=2e-12)
    assert bool(rows.analytic_slots) == analytic
    if analytic:
        assert len(rows.ad_rows) > 0 and len(rows.analytic_slots) == 3


def test_four_argument_bridge_preserves_named_pose_twist_and_vjp(monkeypatch):
    """Nonzero rates, independent fingers and calibration exercise real consumers."""
    fixture, _, _ = finger_case()
    source = fixture.kinematics
    calibration = np.eye(4)
    calibration[:3, :3], calibration[:3, 3] = source._rotation, source._translation
    new = DifferentiablePinKinematics(source.model, frames=source.frames,
                                      T_reference_base=calibration)
    old = PinKinematics(source.model, frames=source.frames, T_reference_base=calibration)
    names, position, velocity = pin_fixture.sample()
    position.requires_grad_()
    velocity.requires_grad_()
    # BEGIN independent reference exercises Pinocchio's acceleration overload.
    import pinocchio as pin
    native = pin.computeForwardKinematicsDerivatives
    with monkeypatch.context() as patch:
        patch.setattr(pin, "computeForwardKinematicsDerivatives",
            lambda model, data, q, dq: native(model, data, q, dq, np.zeros(model.nv)))
        wanted = pin_fixture.packed(*old.states(position, velocity, names))
    # END production uses the four-argument overload for the same first derivatives.
    poses, twists, jacobians = new.states_with_jacobians(position, velocity, names)
    actual = pin_fixture.packed(poses, twists)
    torch.testing.assert_close(actual, wanted, rtol=0., atol=0.)
    actual_vjp = torch.autograd.grad((actual.square()+actual.sin()).sum(), (position, velocity))
    wanted_vjp = torch.autograd.grad((wanted.square()+wanted.sin()).sum(), (position, velocity))
    for left, right in zip(actual_vjp, wanted_vjp):
        torch.testing.assert_close(left, right, rtol=2e-13, atol=2e-13)
    for frame in poses:
        torch.testing.assert_close((jacobians[frame] @ velocity[..., None]).squeeze(-1),
                                   torch.cat(twists[frame], -1), rtol=2e-13, atol=2e-13)


def test_other_task_families_keep_the_same_sparse_layout():
    """The reduction is chosen by factors, including in mechanism/interior tasks."""
    from manipulation.direct_planner.support_linearization import task_linearization as original_linearization

    bound, initial = all_task_case(.2)
    controls = torch.tensor(initial, dtype=torch.float64, requires_grad=True)
    robot, jacobians = linearized_robot_state(bound, *bound.states(controls)[:2])
    declarations = ({}, {})
    values, blocks = task_linearization(bound, controls, solver_rows_out=declarations,
        robot_state=robot, jacobians=jacobians,
        support_material_indices=bound.support_material_indices)
    new = SparseTaskRows(bound, values, blocks, declarations, {'robot_state': robot})
    old_declarations = ({}, {})
    old_values, old_blocks = original_linearization(bound, controls,
        solver_rows_out=old_declarations, robot_state=robot, jacobians=jacobians,
        support_material_indices=bound.support_material_indices)
    old = SparseTaskRows(bound, old_values, old_blocks, old_declarations, {'robot_state': robot}, support_presolve=False)
    assert new.scoring_row_keys == old.row_keys
    assert new.row_keys == tuple(old.row_keys[row] for row in new.solver_selection)
    np.testing.assert_array_equal(new.scoring_values(values).detach(), old.select_values(old_values).detach())
    old_dense = csr_matrix((old.jacobian_values(old_values, controls, old_blocks,
        check_request=lambda: None), old.pattern.columns, old.pattern.indptr), shape=old.pattern.shape).toarray()
    new_dense = csr_matrix((new.jacobian_values(values, controls, blocks,
        check_request=lambda: None), new.pattern.columns, new.pattern.indptr), shape=new.pattern.shape).toarray()
    np.testing.assert_array_equal(new_dense, old_dense[new.solver_selection])
