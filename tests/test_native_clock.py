"""CPU contract checks for the production native interpolation adapter."""

# BEGIN isolated exact-clock adapter regression without a CUDA context.
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
import numpy as np

from manipulation.planning import _formal_native_clock, _joint_state_sample_times, _hermite_at_parameters
from manipulation.cubic_trajectory import CubicTrajectoryStep


class NativeClockAdapterTests(unittest.TestCase):
    # BEGIN nested and nonnested grids retain their generating physical instants.
    def test_regular_dense_and_candidate_times_round_the_real_clock_once(self):
        for native_dt, dense_subdivisions in ((.1875, 6), (.17, 32)):
            step = CubicTrajectoryStep(18, 4, native_dt, device="cpu")
            output_dt = step.knot_dt/dense_subdivisions
            count = step.curve.intervals*dense_subdivisions+1
            regular = SimpleNamespace(position=torch.zeros(1, 81, 1), dt=torch.tensor([step.native_dt]))
            dense = SimpleNamespace(position=torch.zeros(1, count+3, 1), dt=torch.tensor([output_dt]))
            regular_times = np.arange(81)*step.native_dt
            dense_times = np.arange(count)*output_dt
            actual_regular = _joint_state_sample_times(regular, torch.tensor([81]))
            actual_dense = _joint_state_sample_times(dense, torch.tensor([count]))
            torch.testing.assert_close(actual_regular, torch.tensor(regular_times[None], dtype=torch.float32), rtol=0, atol=0)
            torch.testing.assert_close(actual_dense[:, :count], torch.tensor(dense_times[None], dtype=torch.float32), rtol=0, atol=0)
            _, regular_indices, dense_indices = np.intersect1d(regular_times, dense_times, return_indices=True)
            torch.testing.assert_close(actual_regular[:, regular_indices], actual_dense[:, dense_indices], rtol=0, atol=0)
            self.assertEqual(len(np.union1d(regular_times, dense_times)), count if dense_subdivisions == 32 else count+40)
            self.assertTrue(bool((actual_dense[:, count:] == actual_dense[:, count-1:count]).all()))

    def test_exact_anchor_state_and_time_use_the_original_double_parameter(self):
        dt, duration = .1875, 15.
        times = np.arange(81)*dt
        q = torch.tensor(times[None, :, None], dtype=torch.float32)
        parameter = np.array([[.3000000049, .73123456789]])
        position, velocity = _hermite_at_parameters(q, torch.ones_like(q),
            torch.tensor([dt]), torch.tensor([81]), torch.tensor(parameter, dtype=torch.float64))
        expected_time = torch.tensor(parameter*duration, dtype=torch.float32)
        torch.testing.assert_close(position[..., 0], expected_time, rtol=0, atol=0)
        torch.testing.assert_close(velocity, torch.ones_like(velocity), rtol=0, atol=0)
    # END these identities preserve event clocks without tolerance-based merging.

    def test_native_clock_adapter_keeps_two_seed_buffers_and_rejects_knot_drift(self):
        # BEGIN execute the actual exporter and shared tensor step on CPU.
        # The obsolete native interpolation entry is a failure sentinel.
        import sys
        from curobo._src.types.control_space import ControlSpace
        from tests.planning.replay_native_clock import interpolation_method

        configured_type = object()
        kernel = Mock(side_effect=AssertionError("Old fixed-prefix CUDA interpolation was called"))
        modules = {
            "curobo._src.curobolib.cuda_ops.trajectory": SimpleNamespace(get_bspline_interpolation=kernel),
            "curobo._src.util.trajectory": SimpleNamespace(
                TrajInterpolationType=SimpleNamespace(BSPLINE_KNOTS_CUDA=configured_type)),
        }
        joint_state = SimpleNamespace(zeros=lambda shape, device, joint_names:
            SimpleNamespace(shape=shape, position=torch.zeros(shape), joint_names=joint_names))
        method = interpolation_method(torch, joint_state, _formal_native_clock)
        managers = [SimpleNamespace(set_dense_counts=Mock()) for _ in range(2)]
        goal = SimpleNamespace(current_js=SimpleNamespace(position=torch.zeros(2, 7),
            velocity=torch.full((2, 7), 1e-5)), seed_goal_js=object(),
            idxs_current_js=torch.arange(2, dtype=torch.int32), idxs_seed_goal_js=object(),
            seed_enable_implicit_goal_js=torch.zeros(2, dtype=torch.uint8))
        step = CubicTrajectoryStep(18, 4, .2, device="cpu")
        optimizer = SimpleNamespace(config=SimpleNamespace(minimum_trajectory_dt=.2,
            maximum_trajectory_dt=.2, interpolation_dt=.025, interpolation_buffer_size=768,
            interpolation_type=configured_type, num_seeds=2), action_horizon=18, action_dim=7,
            horizon=81, interpolation_steps=4,
            transition_model=SimpleNamespace(control_space=ControlSpace.BSPLINE_3, _rollout_step_fn=step),
            device_cfg=SimpleNamespace(device="cpu"), joint_names=[f"joint_{index}" for index in range(7)],
            _interpolated_traj_buffer=None, _interpolation_dt_buffer=None,
            core=SimpleNamespace(_goal_buffer=goal),
            additional_metrics_rollouts={"interpolated_rollout": SimpleNamespace(
                constraint_manager=managers[0], metrics_constraint_manager=managers[1])})
        state = SimpleNamespace(position=torch.zeros((2, 81, 7)), knot=torch.arange(252).reshape(2, 18, 7).float(),
                                dt=torch.full((2,), .2), knot_dt=torch.full((2,), .8))
        original_controls = state.knot.clone()
        before_cuda = torch.cuda.is_initialized()
        with patch.dict(sys.modules, modules), patch("torch.cuda.init", side_effect=AssertionError("CPU contract only")):
            output, counts, resized = method(optimizer, state)
            self.assertFalse(resized)
            self.assertEqual(output.shape, (2, 768, 7))
            torch.testing.assert_close(counts, torch.tensor([641, 641], dtype=torch.int32), rtol=0, atol=0)
            kernel.assert_not_called()
            torch.testing.assert_close(output.position[:, 0], goal.current_js.position, rtol=0, atol=0)
            torch.testing.assert_close(output.velocity[:, 0], goal.current_js.velocity, rtol=0, atol=0)
            for manager in managers:
                self.assertIs(manager.set_dense_counts.call_args.args[0], counts)
            repeated, _, _ = method(optimizer, state)
            self.assertIs(output, repeated)
            state.knot_dt[1] = torch.nextafter(state.knot_dt[1], torch.tensor(float("inf")))
            with self.assertRaisesRegex(ValueError, "fixed request-wide spline clock"):
                method(optimizer, state)
            kernel.assert_not_called()
            state.knot_dt.fill_(.8)
            optimizer.config.interpolation_buffer_size = 640
            with self.assertRaisesRegex(ValueError, "fixed buffer capacity"):
                method(optimizer, state)
            kernel.assert_not_called()
        torch.testing.assert_close(state.knot, original_controls, rtol=0, atol=0)
        self.assertEqual(torch.cuda.is_initialized(), before_cuda)
# END CPU tests cover the real shared map; robot FK is exercised by the GPU smoke.
