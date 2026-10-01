"""CPU regressions for retaining distinct native trajectory representations."""
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "planning"))
from recorded_task_planning import save_curve, save_native_configuration, save_native_result, verification_joint_limits


# BEGIN effective-bound fixtures stay distinct from input robot configuration.
def native_verifier():
    """Supply a live test cost with a deliberately different named order."""
    limits = SimpleNamespace(joint_names=["a", "b"],
        position=torch.tensor([[-8., -9.], [8., 9.]]),
        velocity=torch.tensor([[-2., -3.], [2., 3.]]),
        acceleration=torch.tensor([[-5., -.1], [5., .1]]),
        jerk=torch.tensor([[-60., -70.], [60., 70.]]))
    cost = SimpleNamespace(enabled=True, config=SimpleNamespace(joint_limits=limits))
    return SimpleNamespace(get_cost={"cspace": cost}.get)
# END saved limits must come from this instance, not a YAML scalar.


def test_native_record_keeps_seed_layout_padding_and_distinct_clocks(tmp_path):
    """Control points, rollout and valid formal samples cannot share one clock."""
    def state(length, dt, knot_dt):
        q = torch.arange(2*length*3, dtype=torch.float32).reshape(2, length, 3)
        return SimpleNamespace(position=q, velocity=q+1, acceleration=None, jerk=None,
                               dt=torch.tensor([dt, dt]), knot_dt=torch.tensor([knot_dt]),
                               joint_names=["joint_b", "joint_a", "joint_c"])

    result = SimpleNamespace(batch_size=1, num_seeds=2,
        solution=torch.full((1, 2, 6, 3), 31.), optimized_seeds=None,
        success=torch.tensor([[False, True]]), seed_rank=torch.tensor([[1, 0]]),
        seed_cost=torch.tensor([[8., 3.]]), interpolated_last_tstep=torch.tensor([7, 5]),
        js_solution=state(8, .8, 1.6), interpolated_trajectory=state(9, .025, 1.65))
    target = tmp_path / "native.npz"
    save_native_result(target, result, ("joint_b", "joint_a", "joint_c"))
    # Mutating the native buffer afterwards must not alter retained evidence.
    result.solution.fill_(-99.)
    with np.load(target, allow_pickle=False) as saved:
        assert saved["solution"].shape == (1, 2, 6, 3)
        assert np.all(saved["solution"] == 31.)
        np.testing.assert_array_equal(saved["seed_rank"], [[1, 0]])
        np.testing.assert_array_equal(saved["interpolated_last_tstep"], [7, 5])
        assert saved["interpolated_trajectory.position"].shape == (2, 9, 3)
        np.testing.assert_allclose(saved["js_solution.dt"], [.8, .8])
        np.testing.assert_allclose(saved["interpolated_trajectory.dt"], [.025, .025])
        np.testing.assert_allclose(saved["interpolated_trajectory.knot_dt"], [1.65])
        assert "optimized_seeds" not in saved
    metadata = json.loads(target.with_suffix(".json").read_text())
    assert metadata["classification"] == "native_diagnostic_not_executable"
    assert metadata["states"]["js_solution"]["joint_names"][0] == "joint_b"


def test_effective_config_uses_constructed_values_and_rejects_opaque_objects(tmp_path):
    """A modified effective weight must survive without repr or live traversal."""
    class Space(Enum):
        BSPLINE = 3

    @dataclass
    class Config:
        weight: torch.Tensor
        control_space: Space
        robot_config: object
        optional: object = None

    config = Config(torch.tensor([2., 5.]), Space.BSPLINE, object())
    planner = SimpleNamespace(ik=SimpleNamespace(config=config),
        trajopt=SimpleNamespace(config=config, horizon=8, action_horizon=6),
        robot_dict={"kinematics": {"joint_names": ["b", "a"]}}, joint_names=("b", "a"),
        progress_map=SimpleNamespace(u=np.array([0., 1.]), s=np.array([0., 2.])),
        _verification_native=native_verifier())
    config.weight[0] = 9.
    target = tmp_path / "configuration"
    save_native_configuration(target, planner)
    document = json.loads(target.with_suffix(".json").read_text())
    assert document["ik"]["robot_config"]["reference"] == "bounded_robot_input"
    assert document["ik"]["control_space"]["enum"].endswith("Space.BSPLINE")
    with np.load(target.with_suffix(".npz"), allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved[document["ik"]["weight"]["npz_key"]], [9., 5.])
        # BEGIN all four runtime matrices retain native ordering and dtype.
        effective = document["verification_joint_limits"]
        assert effective["joint_names"] == ["a", "b"]
        limits = planner._verification_native.get_cost("cspace").config.joint_limits
        for label in ("position", "velocity", "acceleration", "jerk"):
            array = saved[effective[label]["npz_key"]]
            assert array.dtype == np.float32
            np.testing.assert_array_equal(array, getattr(limits, label).numpy())
        # END no live solver object or repr replaces effective numerical bounds.
    config.optional = object()
    with pytest.raises(TypeError, match="configuration.ik.optional"):
        save_native_configuration(tmp_path / "unsupported", planner)
    assert not (tmp_path / "unsupported.json").exists()


# BEGIN candidate reports reuse production calculus without reclassification.
@pytest.mark.parametrize("classification", ["planner_accepted", "rejected_diagnostic_not_executable"])
def test_curve_record_keeps_classification_and_separates_full_curve_from_barrier(tmp_path, classification):
    from manipulation.planning import JointTrajectory

    # Coordinate a is t**3 then linear, with a left-limit acceleration of six
    # at t=1. The original barrier at .5 permits only the prefix whose peak is
    # three. A record must expose both scopes without declaring a new plan.
    events = ({"event_id": "verify", "skill_id": "observed_skill", "kind": "verification",
               "time": .5, "requires_evidence": "actual synchronized evidence"},)
    curve = JointTrajectory(("b", "a"), np.array([0., 1., 2.]),
        np.array([[0., 0.], [0., 1.], [0., 4.]]),
        np.array([[0., 0.], [0., 3.], [0., 3.]]),
        np.array([0., 1.]), np.array([0., 1.]), events, (), {"unchanged": 0.})
    planner = SimpleNamespace(joint_names=curve.joint_names, _verification_native=native_verifier())
    limits = verification_joint_limits(planner)
    source = planner._verification_native.get_cost("cspace").config.joint_limits
    source.acceleration.fill_(99.)
    target = tmp_path / "curve.npz"
    save_curve(target, curve, classification=classification, execution_limits={"source": "explicit test contract"},
               joint_limits=limits)
    metadata = json.loads(target.with_suffix(".json").read_text())
    assert metadata["classification"] == classification
    assert metadata["residuals"] == {"unchanged": 0.}
    np.testing.assert_array_equal(metadata["verification_joint_limits"]["acceleration"], limits["acceleration"])
    full, prefix = (metadata["cubic_derivatives"][key] for key in ("full_curve", "executable_prefix"))
    assert full["checked_interval_s"] == [0., 2.]
    assert full["acceleration"]["maximum_bound_excess"] == 1.
    assert full["continuity"]["acceleration_maximum_absolute_jump"] == [0., 6.]
    assert prefix["checked_interval_s"] == [0., .5]
    assert prefix["acceleration"]["maximum"] == [0., 3.]
    assert prefix["acceleration"]["bounds_satisfied"]
    with np.load(target, allow_pickle=False) as arrays:
        assert arrays["classification"].item() == classification
        np.testing.assert_array_equal(arrays["times"], curve.times)
        np.testing.assert_array_equal(arrays["position"], curve.position)
        np.testing.assert_array_equal(arrays["velocity"], curve.velocity)


def test_native_effective_limits_require_enabled_complete_real_matrices():
    planner = SimpleNamespace(joint_names=("b", "a"), _verification_native=native_verifier())
    cost = planner._verification_native.get_cost("cspace")
    cost.enabled = False
    with pytest.raises(ValueError, match="enabled"):
        verification_joint_limits(planner)
    cost.enabled = True
    cost.config.joint_limits.joint_names = ["b", "b"]
    with pytest.raises(ValueError, match="every active joint"):
        verification_joint_limits(planner)
    cost.config.joint_limits.joint_names = ["a", "b"]
    cost.config.joint_limits.acceleration[0, 0] = float("nan")
    with pytest.raises(ValueError, match="acceleration bounds"):
        verification_joint_limits(planner)
# END diagnostic reports retain actual limits and the original executable clock.
