"""Numerical invariants and counterexamples for the new two-view matching rule."""
import unittest

import numpy as np

from .multiview_features import (centered_rank_maps, rank_sampling_plan,
                                sample_rank_maps, select_fused_candidates,
                                unique_target_pairs)


class RankEvidenceTests(unittest.TestCase):
    def test_midrank_and_constant_response(self):
        allowed = np.ones((2, 2), bool)
        allowed[1, 1] = False
        scores = np.array([[[2., 2.], [4., 1000.]]])
        np.testing.assert_allclose(centered_rank_maps(scores, allowed)[0][allowed], [-1/3, -1/3, 2/3])
        for value in (0., 9., -12.):
            self.assertFalse(centered_rank_maps(np.full((1, 2, 2), value), allowed).any())

    def test_monotone_invariance_requires_rank_before_interpolation(self):
        # With raw interpolation, B>A before cubing, whereas A>B after cubing.
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

    def test_unknown_is_neutral_and_joint_support_can_win(self):
        # Each view's private peak loses to a mutually supported lower peak.
        a = np.array([[.9, -.8, .7]])
        b = np.array([[-.8, .9, .7]])
        fused = select_fused_candidates((a+b)/2)
        self.assertEqual(fused['target'][0], 2)
        np.testing.assert_array_equal(select_fused_candidates((b+a)/2)['target'], fused['target'])
        np.testing.assert_array_equal(select_fused_candidates((a+np.zeros_like(a))/2)['target'], [0])

    def test_positive_exact_ties_abstain_and_negative_raw_is_not_invalid(self):
        selected = select_fused_candidates(np.array([[.4, .4], [0., 0.], [-.1, -.2], [.3, .1]]))
        np.testing.assert_array_equal(selected['valid'], [False, False, False, True])
        self.assertEqual(selected['reason'][0], 'ambiguous_fused_maximum')
        raw = np.array([[[-8., -7.], [-6., -5.]]])
        np.testing.assert_array_equal(centered_rank_maps(raw, np.ones((2, 2), bool)),
                                      centered_rank_maps(raw+100, np.ones((2, 2), bool)))

    def test_target_duplicates_keep_strongest_then_original_query(self):
        selection = select_fused_candidates(np.array([[.5, .1], [.8, .2], [.8, .3], [.1, .7]]))
        np.testing.assert_array_equal(unique_target_pairs(np.array([10, 20, 30, 40]), selection), [1, 3])

    def test_empty_candidates_and_empty_queries(self):
        result = select_fused_candidates(np.zeros((3, 0)))
        self.assertFalse(result['valid'].any())
        self.assertTrue(np.all(result['reason'] == 'no_observed_candidate'))
        self.assertEqual(len(select_fused_candidates(np.zeros((0, 2)))['target']), 0)


if __name__ == '__main__':
    unittest.main()
