"""Original metric sets and derivatives of normalized event inequalities."""

# BEGIN independent error geometry checks the equivalent event representation.
import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from manipulation.planning_problem import PoseEventRequirement
from manipulation.direct_planner.support_linearization import normalized_pose_event


@pytest.mark.parametrize('yaw', [None, 0., .6, np.pi])
def test_normalized_event_has_original_position_swing_and_twist_feasible_set(yaw):
    target, local = np.eye(4), np.eye(4)
    target[:3, :3] = Rotation.from_rotvec([.2, -.1, .15]).as_matrix()
    target[:3, 3] = [.1, -.2, .3]
    local[:3, :3] = Rotation.from_rotvec([-.1, .3, .15]).as_matrix()
    local[:3, 3] = [.02, -.01, .03]
    event = PoseEventRequirement(.5, local, target, .1, .45, source='independent geometry',
        yaw_axis=None if yaw is None else np.array([0., 1., 0.]), yaw_tolerance_rad=yaw)
    rng = np.random.default_rng(9)
    position = torch.tensor(rng.uniform(-.3, .3, (128, 3)), dtype=torch.float64)
    quaternion = torch.tensor(Rotation.random(128, random_state=rng).as_quat()[:, [3, 0, 1, 2]], dtype=torch.float64)
    actual = normalized_pose_event(event, position, quaternion)
    errors = event.error_components(position, quaternion)
    for name, value in actual.items():
        expected = errors[name] <= event.component_tolerances[name]
        torch.testing.assert_close(value <= 0., expected)
    assert ('yaw' in actual) == (yaw is not None and 0. < yaw < np.pi)


def test_event_rows_have_finite_zero_error_curvature_and_correct_first_derivative():
    event = PoseEventRequirement(.5, np.eye(4), np.eye(4), .002, .02, source='independent derivative')

    def values(state):
        rows = normalized_pose_event(event, state[:3], state[3:])
        return torch.stack(tuple(rows.values()))

    zero = torch.tensor([0., 0., 0., 1., 0., 0., 0.], dtype=torch.float64)
    hessian = torch.autograd.functional.hessian(lambda x: values(x).sum(), zero)
    assert torch.isfinite(hessian).all()
    np.testing.assert_allclose(hessian.numpy()[:3, :3], np.eye(3)*2/.002**2, rtol=1e-14)
    state = zero+torch.tensor([.001, -.0004, .0003, -.01, .004, -.003, .005], dtype=torch.float64)
    analytic = torch.autograd.functional.jacobian(values, state).numpy()
    directions = np.eye(7)*1e-7
    numeric = np.stack([(values(state+torch.tensor(d)).numpy()-values(state-torch.tensor(d)).numpy())/2e-7
                        for d in directions], axis=1)
    np.testing.assert_allclose(analytic, numeric, rtol=3e-7, atol=1e-7)
# END zero-yaw equalities retain their original signed-angle producer.
