"""Numerical tracking selection and the installed native seed allocation.

These CPU tests use declared candidate energies and joint states. They check
the proposal/seed interface, without asserting robot reachability or whole-
trajectory qualification. GPU pose derivatives are verified separately on
the production-created native IK instance.
"""

# BEGIN one tracking candidate preserves pose quality and the fixed exploration budget.
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from curobo._src.solver.manager_seed import SeedManager
from curobo._src.types.device_cfg import DeviceCfg
from manipulation.planning_ik import select_ik_tracking_candidate


def test_pose_energy_precedes_distance_and_only_supplied_proposals_are_ranked():
    values = np.array([[0., 0.], [1.5, 1.], [.01, .01]])
    costs = np.array([0., .02, .8])
    # Row zero is outside the caller's supplied proposal set. The best
    # eligible pose is farther away than row two, but its energy is smaller.
    assert select_ik_tracking_candidate(values, [2, 1], costs, [[0., 0.]]) == 1


def test_equal_pose_energy_uses_joint_continuity_then_original_index():
    values = np.array([[.4, .3], [2., 0.], [-.4, -.3]])
    costs = np.ones(3)
    assert select_ik_tracking_candidate(values, [1, 2, 0], costs, [[0., 0.]]) == 0
    assert select_ik_tracking_candidate(values, [0, 1, 2], costs, [[-.1, -.1]]) == 2


def test_empty_proposal_set_has_no_tracking_index():
    assert select_ik_tracking_candidate(np.empty((0, 2)), [], np.empty(0), [[0., 0.]]) is None


@pytest.mark.parametrize("total_slots", [1, 2, 4])
def test_native_seed_manager_preserves_one_tracking_row_and_fills_only_remaining_slots(total_slots):
    values = np.array([[-.7, .1], [.4, -.2], [.2, .8]])
    selected = select_ik_tracking_candidate(values, [0, 1, 2], [.4, .1, .3], [[0., 0.]])
    warm = torch.as_tensor(values[[selected]][None], dtype=torch.float32)
    manager = SeedManager(DeviceCfg("cpu"), 2, torch.full((2,), -1.),
                          torch.full((2,), 1.), random_seed=123)
    manager.generate_random_actions = Mock(wraps=manager.generate_random_actions)
    first = manager.prepare_action_seeds(1, total_slots, seed_config=warm).clone()
    second = manager.prepare_action_seeds(1, total_slots, seed_config=warm).clone()
    assert first.shape == second.shape == (total_slots, 1, 2)
    torch.testing.assert_close(first[0, 0], warm[0, 0], rtol=0., atol=0.)
    torch.testing.assert_close(second[0, 0], warm[0, 0], rtol=0., atol=0.)
    assert bool(((first >= -1.) & (first <= 1.)).all())
    assert bool(((second >= -1.) & (second <= 1.)).all())
    if total_slots == 1:
        manager.generate_random_actions.assert_not_called()
        torch.testing.assert_close(first, second, rtol=0., atol=0.)
    else:
        assert manager.generate_random_actions.call_count == 2
        assert all(call.args == (1, total_slots-1)
                   for call in manager.generate_random_actions.call_args_list)
        assert not torch.equal(first[1:], second[1:])
# END no additional seed slot or native solve is created by tracking selection.
