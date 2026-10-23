"""Shared original-query zero values preserve classification and gradients."""

import numpy as np
import pytest
import torch

from manipulation.implicit_distance import ObservedField
from manipulation.observed_collision import pair_residual


# BEGIN raw observed fields exercise shared and standalone zero baselines.
@pytest.mark.parametrize("shift,status", [(.09, "clear"), (.01, "unresolved"), (.07, "intersection")])
def test_pair_shared_zero_preserves_classification_value_and_gradient(shift, status):
    axis = np.linspace(-.05, .05, 7)
    points = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), -1)
    field = ObservedField(np.max(np.abs(points), -1) - .04, np.ones((7, 7, 7), bool),
                          origin=[-.05] * 3, voxel_size=axis[1] - axis[0])
    if status == "unresolved":
        # A constant negative supported cell touches its translated copy at
        # one exact cell face; no interpolated zero-location approximation is
        # used to construct this boundary case.
        field = ObservedField(-np.ones((2, 2, 2)), np.ones((2, 2, 2), bool),
                              origin=[0.] * 3, voxel_size=.01)
    quaternion = torch.tensor([1., 0., 0., 0.], dtype=torch.float64)
    matrix = np.eye(4)
    matrix[0, 3] = shift
    results = []
    for shared in (False, True):
        first = torch.zeros((2, 3), dtype=torch.float64, requires_grad=True)
        second = torch.tensor([[shift, 0., 0.], [0., 0., 0.]],
                              dtype=torch.float64, requires_grad=True)
        zero = (first.sum(-1) * 0 + second.sum(-1) * 0)[0] if shared else None
        residual, evidence = pair_residual(field, (first[0], quaternion), np.eye(4),
            field, (second[0], quaternion), matrix, zero=zero)
        assert evidence["status"] == status
        if shared and status != "intersection":
            assert residual is zero
        gradients = torch.autograd.grad(residual, (first, second))
        results.append((residual.detach(), evidence, gradients))
    assert results[0][1] == results[1][1]
    torch.testing.assert_close(results[0][0], results[1][0], rtol=0, atol=0)
    for actual, expected in zip(results[0][2], results[1][2]):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
# END batch sharing changes only zero construction, including unresolved queries.
