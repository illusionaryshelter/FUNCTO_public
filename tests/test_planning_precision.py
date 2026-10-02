"""CPU lifecycle checks for the planner's full-float32 numerical contract.

The tests exercise actual request construction/cleanup with native allocation
stubbed out. They prove flag ownership and exceptional cleanup, not CUDA
geometry accuracy; the frozen-input GPU precision audit covers that evidence.
"""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from manipulation.planning import _CuroboPlanner, plan_trajectory


class PlanningPrecisionTests(unittest.TestCase):
    def setUp(self):
        self.original = (torch.backends.cuda.matmul.allow_tf32,
                         torch.backends.cudnn.allow_tf32)
        self.cuda_initialized = torch.cuda.is_initialized()
        self.addCleanup(self.restore_original)

    def restore_original(self):
        torch.backends.cuda.matmul.allow_tf32 = self.original[0]
        torch.backends.cudnn.allow_tf32 = self.original[1]
        self.assertEqual(torch.cuda.is_initialized(), self.cuda_initialized)

    def flags(self):
        return (torch.backends.cuda.matmul.allow_tf32,
                torch.backends.cudnn.allow_tf32)

    def test_request_restores_both_callers_and_repeated_close_does_not_reapply(self):
        for enabled in (True, False):
            with self.subTest(prior_tf32=enabled):
                torch.backends.cuda.matmul.allow_tf32 = enabled
                torch.backends.cudnn.allow_tf32 = not enabled

                # BEGIN the request initializer now binds an optional continuation.
                def initialize(planner, *_args, continuation=None):
                    self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
                    # Both actual solver destructors must still run within the
                    # request precision scope, including any native cleanup.
                    planner.ik = Mock()
                    planner.ik.destroy.side_effect = lambda: self.assertFalse(
                        torch.backends.cuda.matmul.allow_tf32)

                with patch.object(_CuroboPlanner, "_initialize", autospec=True,
                                  side_effect=initialize):
                    planner = _CuroboPlanner(None, None, None, None, "cpu")
                planner.close()
                self.assertEqual(self.flags(), (enabled, not enabled))
                torch.backends.cuda.matmul.allow_tf32 = not enabled
                planner.close()
                self.assertEqual(self.flags(), (not enabled, not enabled))

    def test_initialization_failure_restores_settings_after_dependency_side_effect(self):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

        def failing_initialize(*_args, continuation=None):
            # Reproduce seed_ik_solver's import-time assignments before a
            # subsequent initialization failure can reach the normal reset.
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            raise ValueError("native initialization failed")

        with patch.object(_CuroboPlanner, "_initialize", side_effect=failing_initialize):
            with self.assertRaisesRegex(ValueError, "native initialization failed"):
                _CuroboPlanner(None, None, None, None, "cpu")
        self.assertEqual(self.flags(), (False, False))

    def test_native_destroy_failure_still_restores_precision(self):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = False
        with patch.object(_CuroboPlanner, "_initialize"):
            planner = _CuroboPlanner(None, None, None, None, "cpu")
        planner.ik = SimpleNamespace(destroy=Mock(side_effect=RuntimeError("destroy failed")))
        with self.assertRaisesRegex(RuntimeError, "destroy failed"):
            planner.close()
        self.assertEqual(self.flags(), (True, False))

    def test_diagnostic_failure_closes_the_real_request_scope(self):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = False
        scene = SimpleNamespace(describe=Mock(side_effect=ValueError("description failed")))

        def initialize(planner, *_args, continuation=None):
            from manipulation.planning import PlanningBudget
            planner.budget = PlanningBudget()
            planner.scene_workspace_bytes = 0
            planner.anchor_fk_workspace_bytes = 0
            planner.anchor_capacity = 0
        # END fixtures retain the production initialization contract.

        with patch.object(_CuroboPlanner, "_initialize", autospec=True, side_effect=initialize):
            with self.assertRaisesRegex(ValueError, "description failed"):
                plan_trajectory(None, scene, None, device="cpu")
        self.assertEqual(self.flags(), (True, False))


if __name__ == "__main__":
    unittest.main()
