"""CPU lifecycle checks for the planner's full-float32 numerical contract.

The tests exercise actual request construction/cleanup with native allocation
stubbed out. They prove flag ownership and exceptional cleanup, not CUDA
geometry accuracy; the frozen-input GPU precision audit covers that evidence.
"""

from types import SimpleNamespace
from copy import deepcopy
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from manipulation.planning import _CuroboPlanner, plan_trajectory


class PlanningPrecisionTests(unittest.TestCase):
    def setUp(self):
        self.original = (torch.backends.cuda.matmul.allow_tf32,
                         torch.backends.cudnn.allow_tf32)
        self.cuda_initialized = torch.cuda.is_initialized()
        self.addCleanup(self.restore_original)
        # BEGIN mocked initialization still receives an explicit source-model binding.
        self.robot_validator = object()
        # END this lifecycle sentinel is never used as a physical geometry oracle.

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
                def initialize(planner, *_args, robot_validator, continuation=None, goal_identities=None):
                    self.assertIs(robot_validator, self.robot_validator)
                    self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
                    # Both actual solver destructors must still run within the
                    # request precision scope, including any native cleanup.
                    planner.ik = Mock()
                    planner.ik.destroy.side_effect = lambda: self.assertFalse(
                        torch.backends.cuda.matmul.allow_tf32)

                with patch.object(_CuroboPlanner, "_initialize", autospec=True,
                                  side_effect=initialize):
                    planner = _CuroboPlanner(None, None, None, None, "cpu", robot_validator=self.robot_validator)
                planner.close()
                self.assertEqual(self.flags(), (enabled, not enabled))
                torch.backends.cuda.matmul.allow_tf32 = not enabled
                planner.close()
                self.assertEqual(self.flags(), (not enabled, not enabled))

    def test_initialization_failure_restores_settings_after_dependency_side_effect(self):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

        def failing_initialize(*_args, robot_validator, continuation=None, goal_identities=None):
            # Reproduce seed_ik_solver's import-time assignments before a
            # subsequent initialization failure can reach the normal reset.
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            raise ValueError("native initialization failed")

        with patch.object(_CuroboPlanner, "_initialize", side_effect=failing_initialize):
            with self.assertRaisesRegex(ValueError, "native initialization failed"):
                _CuroboPlanner(None, None, None, None, "cpu", robot_validator=self.robot_validator)
        self.assertEqual(self.flags(), (False, False))

    def test_native_destroy_failure_still_restores_precision(self):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = False
        with patch.object(_CuroboPlanner, "_initialize"):
            planner = _CuroboPlanner(None, None, None, None, "cpu", robot_validator=self.robot_validator)
        planner.ik = SimpleNamespace(destroy=Mock(side_effect=RuntimeError("destroy failed")))
        with self.assertRaisesRegex(RuntimeError, "destroy failed"):
            planner.close()
        self.assertEqual(self.flags(), (True, False))

    def test_diagnostic_failure_closes_the_real_request_scope(self):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = False
        # BEGIN scene binding is mocked along with all native initialization.
        scene = SimpleNamespace(describe=Mock(side_effect=ValueError("description failed")), gripper=None)
        # END this test isolates precision cleanup after the later diagnostic error.
        scene.clone_for_rollout = lambda: deepcopy(scene)

        def initialize(planner, *_args, robot_validator, continuation=None, goal_identities=None):
            from manipulation.planning import PlanningBudget
            planner.budget = PlanningBudget()
            planner.ik = SimpleNamespace(core=SimpleNamespace(optimizers=[]), destroy=Mock())
            planner.trajopt = SimpleNamespace(core=SimpleNamespace(optimizers=[]), destroy=Mock())
            planner.scene_workspace_bytes = 0
            planner.scene_cpu_workspace_bytes = planner.scene_gpu_workspace_bytes = 0
            planner.anchor_fk_workspace_bytes = 0
            planner.anchor_capacity = 0
        # END fixtures retain the production initialization contract.

        with patch.object(_CuroboPlanner, "_initialize", autospec=True, side_effect=initialize):
            # BEGIN failed requests retain accounting and restore caller precision.
            from manipulation.planning_problem import PlanningProblem, MotionPhase, PosePath
            path = PosePath(np.array([0., 1.]), np.stack([np.eye(4), np.eye(4)]), "reference")
            problem = PlanningProblem(("q",), [0.], [0.], "tool", "world", np.eye(4), np.eye(4),
                (MotionPhase("phase", path),), (), source_timestamps=np.array([0., 1.]),
                source_progress=np.array([0., 1.]))
            result = plan_trajectory(problem, scene, None,
                                     robot_validator=self.robot_validator, device="cpu")
            self.assertFalse(result.success)
            self.assertEqual(result.diagnostics["reason"], "description failed")
            self.assertEqual(result.request_ledger["consumed_planning"]["initializations"], 1)
            # END a structured failure does not bypass native cleanup.
        self.assertEqual(self.flags(), (True, False))


# BEGIN optimizer conditioning respects each goal's real functional accuracy.
class FunctionalGoalScalingTests(unittest.TestCase):
    def test_equal_relative_errors_have_equal_objectives_without_relaxing_raw_goals(self):
        from manipulation.planning_problem import PlanningProblem, PoseEventRequirement
        objective = []
        for tolerance in (1e-2, 1e-4):
            goal = PoseEventRequirement(1., np.eye(4), np.eye(4), tolerance, .01, "declared precision")
            # BEGIN this pose-only fixture explicitly has no mechanism directions.
            problem = SimpleNamespace(pose_relations=(), pose_events=(goal,), coordinate_events=(), mechanisms=())
            # END the full problem contract remains strict in production.
            p = torch.tensor([[[2*tolerance, 0., 0.]]], requires_grad=True)
            q = torch.tensor([[[1., 0., 0., 0.]]])
            value = PlanningProblem.optimizer_preferences(problem, p, q, torch.ones(1, 1))
            objective.append(float(value.detach()))
            self.assertGreater(float(goal.evaluate(p, q).detach()), 0.)
            derivative = torch.autograd.grad(value.sum(), p)[0]
            self.assertTrue(torch.isfinite(derivative).all())
            self.assertGreater(float(derivative[0, 0, 0]), 0.)
            inside = torch.tensor([[[.5*tolerance, 0., 0.]]])
            self.assertEqual(float(goal.evaluate(inside, q)), 0.)
        self.assertAlmostEqual(*objective, places=5)
# END optimizer scaling never alters raw acceptance or invents target freedom.


if __name__ == "__main__":
    unittest.main()
