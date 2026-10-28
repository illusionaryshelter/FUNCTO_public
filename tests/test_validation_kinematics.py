"""Explicit candidate FK ownership preserves complete original numerical paths."""
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from manipulation.pin_kinematics import PinKinematics
from manipulation.robot_geometry_check import PairQuery
from manipulation.validation_kinematics import FKSession
from test_pin_kinematics import robot, sample, packed
from test_native_distance_batch import make_checker, assert_bits_equal


def test_pose_velocity_calibration_and_input_order_are_byte_exact(robot):
    checker, bridge, calibration = robot
    names, q, dq = sample()
    shifted = calibration.copy(); shifted[0, 3] += .001
    other = PinKinematics(checker.model, frames=tuple(reversed(bridge.frames)), T_reference_base=shifted)
    with FKSession(checker.model, frames=bridge.frames) as session:
        for consumer in (bridge, other):
            for rate in (dq, dq*.5):
                expected = packed(*consumer.states(q, rate, names)).numpy()
                actual = packed(*consumer.states(q, rate, names, fk_session=session)).numpy()
                assert actual.tobytes() == expected.tobytes()
            assert packed(consumer.poses(q, names, fk_session=session)).numpy().tobytes() == packed(consumer.poses(q, names)).numpy().tobytes()
        assert session.describe()['hits'] > 0
        assert session.describe()['snapshots'] == 2
    assert not session._states and session._data is None


def test_complete_fingers_adjacent_q_and_independent_velocity_remain_distinct(robot):
    checker, bridge, _ = robot
    names, q, dq = sample()
    changed_finger = q.clone(); changed_finger[..., names.index('left_slide')] += .002
    adjacent = q.clone(); adjacent[..., 0] = torch.nextafter(adjacent[..., 0], torch.full_like(adjacent[..., 0], float('inf')))
    with FKSession(checker.model, frames=bridge.frames) as session:
        for changed in (q, changed_finger, adjacent):
            expected = packed(*bridge.states(changed, dq, names)).numpy()
            actual = packed(*bridge.states(changed, dq, names, fk_session=session)).numpy()
            assert actual.tobytes() == expected.tobytes()
        assert session.describe()['snapshots'] == 6


def test_differentiable_consumer_keeps_original_graph(robot):
    checker, bridge, _ = robot
    names, q, dq = sample(); q.requires_grad_(); dq.requires_grad_()
    with FKSession(checker.model, frames=bridge.frames) as session:
        original = packed(*bridge.states(q, dq, names))
        shared = packed(*bridge.states(q, dq, names, fk_session=session))
        torch.testing.assert_close(shared, original, rtol=0, atol=0)
        left = torch.autograd.grad(original.sum(), (q, dq))
        right = torch.autograd.grad(shared.sum(), (q, dq))
        for a, b in zip(left, right):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        assert session.describe()['computed'] == 0


@pytest.mark.parametrize('parallel', [False, True])
def test_native_all_columns_fk_reuse_and_cancellation_positions(tmp_path, parallel):
    checker = make_checker(tmp_path)
    pairs = (PairQuery('prism', 'box', 'required', 'registered pair'),)
    states = np.array([[0., .03*i, 1.+.02*i] for i in range(12)])
    args = dict(joint_names=('qx','qy','qz'), joint_positions=states,
        T_reference_base=np.eye(4), pairs=pairs, projection_mask=np.ones((12, 1), bool))
    original_polls, shared_polls = Mock(), Mock()
    expected = list(checker.query_pair_batches(**args, check_request=original_polls))
    with FKSession(checker.model, frames=('base', 'z')) as session:
        actual = list(checker.query_pair_batches(**args, check_request=shared_polls,
            fk_session=session, parallel=parallel))
        assert_bits_equal(expected, actual)
        assert shared_polls.call_count == original_polls.call_count
        # A second consumer reuses placements without changing batch ownership.
        again = list(checker.query_pair_batches(**args, fk_session=session, parallel=parallel))
        assert_bits_equal(expected, again)
        assert session.describe()['hits'] == 12
        calls = 0
        def stop_after_preparation():
            nonlocal calls
            calls += 1
            if calls == original_polls.call_count:
                raise TimeoutError('cancel before delivery')
        with pytest.raises(TimeoutError, match='cancel before delivery'):
            next(checker.query_pair_batches(**args, fk_session=session,
                parallel=parallel, check_request=stop_after_preparation))
