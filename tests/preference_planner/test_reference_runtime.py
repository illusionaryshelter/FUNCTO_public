"""CPU checks of the planner-owned ordinary reference IK numerical policy."""
from dataclasses import replace
import pytest
from curobo._src.optim.gradient.lbfgs import LBFGSOptCfg
from curobo._src.solver.solver_core_cfg import SolverCoreCfg
from curobo._src.solver.solver_ik_cfg import IKSolverCfg
from curobo._src.types.device_cfg import DeviceCfg
from manipulation.preference_planner.native_setup import (
    configure_reference_ik, reference_ik_configuration)


def native_configuration() -> IKSolverCfg:
    """Supply configuration objects without allocating a native robot solver."""
    device = DeviceCfg("cpu")
    optimizer = LBFGSOptCfg(num_iters=80, inner_iters=1, fixed_iters=True,
                            convergence_iteration=10, device_cfg=device)
    core = SolverCoreCfg(
        metrics_rollout_config=object(), optimizer_rollout_configs=[object()],
        optimizer_configs=[optimizer], device_cfg=device, use_cuda_graph=False,
    )
    return IKSolverCfg(core_cfg=core, robot_config=object(), use_lm_seed=False,
                       exit_early=False, num_seeds=1)


@pytest.mark.parametrize("iterations", [20, 40, 79, 80])
def test_native_budget_and_convergence_keep_the_original_models(iterations):
    """Budget grouping keeps exact work and the original cost/model objects."""
    old = native_configuration()
    old.optimizer_configs[0].update_niters(iterations)
    extra = replace(old.optimizer_configs[0], solver_name="other_stage")
    old.core_cfg.optimizer_configs.insert(0, extra)
    new = configure_reference_ik(old)
    assert new is not old and new.core_cfg is not old.core_cfg
    assert new.robot_config is old.robot_config
    assert new.core_cfg.optimizer_rollout_configs is old.core_cfg.optimizer_rollout_configs
    assert new.core_cfg.metrics_rollout_config is old.core_cfg.metrics_rollout_config
    assert new.optimizer_configs[0] == extra
    assert old.optimizer_configs[1].inner_iters == 1
    result = reference_ik_configuration(new)
    optimizer = result['optimizer']
    assert optimizer['num_iters'] == iterations
    assert optimizer['outer_iters'] * optimizer['inner_iters'] == iterations
    assert not optimizer['fixed_iters']
    for field in ('cost_convergence', 'cost_delta_threshold', 'cost_relative_threshold',
                  'converged_ratio', 'convergence_iteration', 'minimum_iters'):
        assert optimizer[field] == getattr(old.optimizer_configs[1], field)
    assert new.position_tolerance == old.position_tolerance
    assert new.orientation_tolerance == old.orientation_tolerance
    assert new.use_lm_seed == old.use_lm_seed
    assert new.use_cuda_graph and not new.exit_early
    assert not old.use_cuda_graph
