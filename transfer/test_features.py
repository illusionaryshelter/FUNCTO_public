"""Shared response-rank and interpolation contracts used by single-view joint.

These tests retain the existing numerical checks from the retired fusion test
module. They do not require any choice about combining two camera responses.
"""

import unittest

import numpy as np

from transfer.features import centered_rank_maps, rank_sampling_plan, sample_rank_maps


class RankEvidenceTests(unittest.TestCase):
    def test_midrank_and_constant_response(self):
        allowed = np.ones((2, 2), bool)
        allowed[1, 1] = False
        scores = np.array([[[2., 2.], [4., 1000.]]])
        np.testing.assert_allclose(centered_rank_maps(scores, allowed)[0][allowed], [-1/3, -1/3, 2/3])
        for value in (0., 9., -12.):
            self.assertFalse(centered_rank_maps(np.full((1, 2, 2), value), allowed).any())

    def test_monotone_invariance_requires_rank_before_interpolation(self):
        # Raw interpolation reverses these candidates after cubing; ranking
        # first must preserve the ordering for strictly increasing transforms.
        raw = np.array([[[0., 2.], [1.1, 1.1]]])
        self.assertLess(np.mean(raw[0, 0]), raw[0, 1, 0])
        self.assertGreater(np.mean(raw[0, 0]**3), raw[0, 1, 0]**3)
        allowed = np.ones((2, 2), bool)
        plan = rank_sampling_plan(np.array([[.5, 0.], [.5, 1.]]), allowed)
        reference = sample_rank_maps(centered_rank_maps(raw, allowed), plan)
        for transformed in (raw+20, raw*.3, raw**3, raw-100):
            np.testing.assert_array_equal(sample_rank_maps(centered_rank_maps(transformed, allowed), plan), reference)

    def test_allowed_domain_and_mask_normalization(self):
        mask = np.array([[True, False], [False, False]])
        plan = rank_sampling_plan(np.array([[.5, .5], [1., 1.], [-.01, 0.], [np.nan, 0.]]), mask)
        ranks = np.array([[[.8, 999.], [999., 999.]]])
        np.testing.assert_allclose(sample_rank_maps(ranks, plan), [[.8, 0., 0., 0.]])
        np.testing.assert_array_equal(plan['supported'], [True, False, False, False])
        logits = np.array([[[3., 4.], [5., 6.]]])
        first = centered_rank_maps(logits, mask)
        logits[:, ~mask] = 1e12
        np.testing.assert_array_equal(centered_rank_maps(logits, mask), first)

    def test_negative_response_rank_is_translation_invariant(self):
        # This assertion was previously mixed with the retired fusion-specific
        # positive-maximum/tie policy; raw response sign is a separate contract.
        raw = np.array([[[-8., -7.], [-6., -5.]]])
        np.testing.assert_array_equal(centered_rank_maps(raw, np.ones((2, 2), bool)),
                                      centered_rank_maps(raw+100, np.ones((2, 2), bool)))


if __name__ == '__main__':
    unittest.main()
