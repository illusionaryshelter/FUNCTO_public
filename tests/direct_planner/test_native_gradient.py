"""Compare the compiled scalar/batch covector pullback with independent NumPy."""

import numpy as np
import pytest

from manipulation.direct_planner.native_gradient import native_joint_gradient


def original_gradient(numeric, first, second, jacobian_first, jacobian_second, common):
    """Independent original vector expression, including structural zero columns."""
    gradient = np.zeros(numeric.shape[:-1]+(len(common),))
    normal = numeric[..., 7:10]
    for side, sign, origin, jacobian in ((0, -1., first, jacobian_first),
                                        (1, 1., second, jacobian_second)):
        offset = numeric[..., 1+3*side:4+3*side]-origin
        gradient += sign*(np.einsum('...i,...ij->...j', normal, jacobian[..., :3, :])
            +np.einsum('...i,...ij->...j', np.cross(offset, normal), jacobian[..., 3:, :]))
    gradient[..., common] = 0.
    return gradient


def arrays(leading, joints):
    """Use noncontiguous numeric records, origins, Jacobians and masks."""
    rng = np.random.default_rng(5801)
    numeric = rng.normal(size=(*leading, 24))[..., ::2]
    origins = [rng.normal(size=(*leading, 6))[..., ::2] for _ in range(2)]
    jacobians = [rng.normal(size=(*leading, 12, 2*joints))[..., ::2, ::2] for _ in range(2)]
    mask = np.zeros(joints*2, dtype=bool)[::2]
    mask[::3] = True
    return numeric, *origins, *jacobians, mask


@pytest.mark.parametrize('leading', [(), (17,), (3, 4)])
@pytest.mark.parametrize('joints', [2, 9, 11])
def test_compiled_gradient_matches_original_strided_scalar_and_batch(leading, joints):
    inputs = arrays(leading, joints)
    np.testing.assert_array_equal(native_joint_gradient(*inputs), original_gradient(*inputs))


def test_inactive_nan_rows_keep_common_ancestor_zero_and_other_columns_unknown():
    inputs = arrays((4,), 9)
    inputs[0][1] = np.nan
    expected = original_gradient(*inputs)
    actual = native_joint_gradient(*inputs)
    np.testing.assert_array_equal(actual, expected)
    assert np.isnan(actual[1, ~inputs[-1]]).all()
    np.testing.assert_array_equal(actual[1, inputs[-1]], 0.)
