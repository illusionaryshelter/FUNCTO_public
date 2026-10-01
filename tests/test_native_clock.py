"""CPU contract checks for the production native interpolation adapter."""

# BEGIN isolated exact-clock adapter regression without a CUDA context.
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from manipulation.planning import _formal_native_clock


class NativeClockAdapterTests(unittest.TestCase):
    def test_native_clock_adapter_keeps_two_seed_buffers_and_rejects_knot_drift(self):
        # Execute the actual nested adapter on CPU with only its CUDA call
        # substituted. The separate archive replay exercises the real kernel.
        import sys
        from curobo._src.types.control_space import ControlSpace
        from tests.planning.replay_native_clock import interpolation_method

        configured_type = object()
        kernel = Mock(side_effect=lambda state, output, dt, **kwargs: output)
        modules = {
            "curobo._src.curobolib.cuda_ops.trajectory": SimpleNamespace(get_bspline_interpolation=kernel),
            "curobo._src.util.trajectory": SimpleNamespace(
                TrajInterpolationType=SimpleNamespace(BSPLINE_KNOTS_CUDA=configured_type)),
        }
        joint_state = SimpleNamespace(zeros=lambda shape, device, joint_names:
            SimpleNamespace(shape=shape, position=torch.zeros(shape), joint_names=joint_names))
        method = interpolation_method(torch, joint_state, _formal_native_clock)
        managers = [SimpleNamespace(set_dense_counts=Mock()) for _ in range(2)]
        goal = SimpleNamespace(current_js=object(), seed_goal_js=object(),
            idxs_current_js=object(), idxs_seed_goal_js=object(), seed_enable_implicit_goal_js=object())
        optimizer = SimpleNamespace(config=SimpleNamespace(minimum_trajectory_dt=.2,
            maximum_trajectory_dt=.2, interpolation_dt=.025, interpolation_buffer_size=768,
            interpolation_type=configured_type, num_seeds=2), action_horizon=16, action_dim=7,
            horizon=81, interpolation_steps=4,
            transition_model=SimpleNamespace(control_space=ControlSpace.BSPLINE_3),
            device_cfg=SimpleNamespace(device="cpu"), joint_names=[f"joint_{index}" for index in range(7)],
            _interpolated_traj_buffer=None, _interpolation_dt_buffer=None,
            core=SimpleNamespace(_goal_buffer=goal),
            additional_metrics_rollouts={"interpolated_rollout": SimpleNamespace(
                constraint_manager=managers[0], metrics_constraint_manager=managers[1])})
        state = SimpleNamespace(position=torch.zeros((2, 81, 7)), knot=torch.arange(224).reshape(2, 16, 7).float(),
                                dt=torch.full((2,), .2), knot_dt=torch.full((2,), .8))
        original_controls = state.knot.clone()
        before_cuda = torch.cuda.is_initialized()
        with patch.dict(sys.modules, modules), patch("torch.cuda.init", side_effect=AssertionError("CPU contract only")):
            output, counts, resized = method(optimizer, state)
            self.assertFalse(resized)
            self.assertEqual(output.shape, (2, 768, 7))
            torch.testing.assert_close(counts, torch.tensor([641, 641], dtype=torch.int32), rtol=0, atol=0)
            self.assertEqual(kernel.call_args.kwargs["bspline_degree"], 3)
            self.assertIs(kernel.call_args.kwargs["current_state"], goal.current_js)
            self.assertIs(kernel.call_args.kwargs["use_implicit_goal_state"], goal.seed_enable_implicit_goal_js)
            for manager in managers:
                self.assertIs(manager.set_dense_counts.call_args.args[0], counts)
            repeated, _, _ = method(optimizer, state)
            self.assertIs(output, repeated)
            state.knot_dt[1] = torch.nextafter(state.knot_dt[1], torch.tensor(float("inf")))
            with self.assertRaisesRegex(ValueError, "fixed request-wide spline clock"):
                method(optimizer, state)
            self.assertEqual(kernel.call_count, 2)
            state.knot_dt.fill_(.8)
            optimizer.config.interpolation_buffer_size = 640
            with self.assertRaisesRegex(ValueError, "fixed buffer capacity"):
                method(optimizer, state)
            self.assertEqual(kernel.call_count, 2)
        torch.testing.assert_close(state.knot, original_controls, rtol=0, atol=0)
        self.assertEqual(torch.cuda.is_initialized(), before_cuda)
# END kernel numerics are checked by the separate original-archive replay.
