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
from recorded_task_planning import save_native_configuration, save_native_result


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
        progress_map=SimpleNamespace(u=np.array([0., 1.]), s=np.array([0., 2.])))
    config.weight[0] = 9.
    target = tmp_path / "configuration"
    save_native_configuration(target, planner)
    document = json.loads(target.with_suffix(".json").read_text())
    assert document["ik"]["robot_config"]["reference"] == "bounded_robot_input"
    assert document["ik"]["control_space"]["enum"].endswith("Space.BSPLINE")
    with np.load(target.with_suffix(".npz"), allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved[document["ik"]["weight"]["npz_key"]], [9., 5.])
    config.optional = object()
    with pytest.raises(TypeError, match="configuration.ik.optional"):
        save_native_configuration(tmp_path / "unsupported", planner)
    assert not (tmp_path / "unsupported.json").exists()
