"""Boolean acceptance equivalence; no GPU, FK, or geometry approximation."""

import unittest
from unittest.mock import patch

import torch

from manipulation.planning import _ompl_terms_accepted


# BEGIN OMPL combines only completed boolean flags at the host boundary.
class OmplTermAcceptanceTest(unittest.TestCase):
    @staticmethod
    def original(terms):
        return all(bool(torch.isfinite(term).all()) and bool((term <= 0).all())
                   for term in terms.values())

    def test_exact_original_verdict_for_nonfinite_zero_and_adjacent_values(self):
        for dtype in (torch.float32, torch.float64):
            zero = torch.tensor(0., dtype=dtype)
            positive = torch.nextafter(zero, torch.tensor(torch.inf, dtype=dtype))
            negative = torch.nextafter(zero, torch.tensor(-torch.inf, dtype=dtype))
            values = [0., -0., -1., 1., -torch.inf, torch.inf, torch.nan,
                      positive.item(), negative.item(), torch.finfo(dtype).max,
                      -torch.finfo(dtype).max]
            for first in values:
                for second in values:
                    terms = {"process": torch.tensor([[first, 0.]], dtype=dtype),
                             "scene": torch.tensor([[second]], dtype=dtype),
                             "empty": torch.empty((1, 0), dtype=dtype)}
                    before = {name: value.numpy().tobytes() for name, value in terms.items()}
                    with self.subTest(dtype=dtype, first=first, second=second):
                        self.assertEqual(_ompl_terms_accepted(terms), self.original(terms))
                        self.assertEqual(before,
                            {name: value.numpy().tobytes() for name, value in terms.items()})

    def test_nonempty_collection_reads_one_boolean_scalar(self):
        for first in (0., 1., torch.nan):
            terms = {"process": torch.tensor([[first]]), "scene": torch.zeros((2, 3)),
                     "native": torch.full((1, 1), -1.)}
            original_item, original_bool = torch.Tensor.item, torch.Tensor.__bool__
            reads, implicit_reads = [], []

            def read(tensor, *args, **kwargs):
                reads.append((tensor.dtype, tensor.numel()))
                return original_item(tensor, *args, **kwargs)

            def implicit_read(tensor):
                implicit_reads.append(tensor)
                return original_bool(tensor)

            expected = self.original(terms)
            with patch.object(torch.Tensor, "item", read), \
                    patch.object(torch.Tensor, "__bool__", implicit_read):
                actual = _ompl_terms_accepted(terms)
            self.assertEqual(actual, expected)
            self.assertEqual(reads, [(torch.bool, 1)])
            self.assertEqual(implicit_reads, [])

    def test_empty_collection_keeps_all_identity_without_scalar_read(self):
        with patch.object(torch.Tensor, "item", side_effect=AssertionError("unexpected read")):
            self.assertTrue(_ompl_terms_accepted({}))
# END boolean flags do not alter finite-value reductions or failure reports.


if __name__ == "__main__":
    unittest.main()
