"""Installed native seed allocation for the fixed ordinary/functional instances."""

# BEGIN the maintained SeedManager determines actual solve shape.
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from curobo._src.solver.manager_seed import SeedManager
from curobo._src.types.device_cfg import DeviceCfg


# BEGIN native decoding may replace the caller's nested dictionaries in place.
def test_two_solver_construction_separates_raw_configs_before_native_decoding():
    import ast
    from copy import deepcopy
    import inspect
    from types import SimpleNamespace
    from manipulation.planning import _ConstrainedPlanner

    tree = ast.parse(inspect.getsource(inspect.getmodule(_ConstrainedPlanner)))
    planner = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                   and node.name == '_ConstrainedPlanner')
    initialize = next(node for node in planner.body if isinstance(node, ast.FunctionDef)
                      and node.name == '_initialize')
    begin = next(i for i, node in enumerate(initialize.body)
                 if isinstance(node, ast.FunctionDef) and node.name == 'configuration')
    end = next(i for i, node in enumerate(initialize.body[begin:], begin)
               if isinstance(node, ast.Assign) and any(isinstance(target, ast.Attribute)
                   and target.attr == 'allow_tf32' for target in node.targets))
    program = ast.fix_missing_locations(ast.Module(body=initialize.body[begin:end], type_ignores=[]))
    decoded = []

    def create(**kwargs):
        optimizer = kwargs['optimizer_configs'][0]
        # The real decoder mutates its supplied optimizer/metrics structures.
        # Replacing the dict reproduces the observed construction failure.
        inputs = [optimizer, kwargs['metrics_rollout']]
        modes = []
        for source in inputs:
            values = source['rollout']['constraint_cfg']
            modes.append(values.get('process_mode'))
            source['rollout']['constraint_cfg'] = SimpleNamespace(**values)
        decoded.append((kwargs, modes))
        return SimpleNamespace(**kwargs)

    def yaml(_name):
        return {'optimizer': {}, 'rollout': {'constraint_cfg': {}, 'cost_cfg': {
            'tool_pose_cfg': {'weight': [10000., 500.]}, 'cspace_cfg': {}}}}

    owner = SimpleNamespace(device_cfg=SimpleNamespace(device='cpu'), _manager_type=object())
    budget = SimpleNamespace(ik_seeds=4, ik_iterations=80, random_seed=123,
        max_device_used_bytes=10**10, max_host_rss_bytes=12*10**9)
    scope = dict(self=owner, budget=budget, problem=SimpleNamespace(
        numerical=SimpleNamespace(position_m=.001, rotation_rad=.01)), robot_dict={},
        deepcopy=deepcopy, load_yaml=yaml, join_path=lambda root, name: name,
        get_task_configs_path=lambda: '',
        InverseKinematicsCfg=SimpleNamespace(create=create),
        InverseKinematics=lambda config: SimpleNamespace(config=config),
        require_device_memory=Mock(), _require_planning_host_memory=Mock(),
        workspace=SimpleNamespace(pending_cpu_bytes=0), torch=torch)
    exec(compile(program, '<production IK construction>', 'exec'), scope)
    assert [row[1] for row in decoded] == [['seed_optimizer', 'seed_metrics'], [None, None]]
    assert owner.ik.config.num_seeds == 4 and owner.reference_ik.config.num_seeds == 1
    assert owner.reference_ik.config.optimization_dt is None
    assert owner.reference_ik.config.velocity_regularization_weight == .01
    assert owner.reference_ik.config.acceleration_regularization_weight == .01
    assert all(not solver.config.use_lm_seed and not solver.config.exit_early
               for solver in (owner.ik, owner.reference_ik))
    scope['require_device_memory'].assert_called_once()
    scope['_require_planning_host_memory'].assert_called_once()
# END construction uses two raw configurations rather than decoding a typed one again.


@pytest.mark.parametrize("total_slots", [1, 2, 4])
def test_native_seed_manager_preserves_the_supplied_state_and_fills_only_configured_slots(total_slots):
    values = np.array([[-.7, .1], [.4, -.2], [.2, .8]])
    warm = torch.as_tensor(values[[0]][None], dtype=torch.float32)
    manager = SeedManager(DeviceCfg("cpu"), 2, torch.full((2,), -1.),
                          torch.full((2,), 1.), random_seed=123)
    manager.generate_random_actions = Mock(wraps=manager.generate_random_actions)
    first = manager.prepare_action_seeds(1, total_slots, seed_config=warm).clone()
    second = manager.prepare_action_seeds(1, total_slots, seed_config=warm).clone()
    assert first.shape == second.shape == (total_slots, 1, 2)
    torch.testing.assert_close(first[0, 0], warm[0, 0], rtol=0., atol=0.)
    torch.testing.assert_close(second[0, 0], warm[0, 0], rtol=0., atol=0.)
    assert bool(((first >= -1.) & (first <= 1.)).all())
    assert bool(((second >= -1.) & (second <= 1.)).all())
    if total_slots == 1:
        manager.generate_random_actions.assert_not_called()
        torch.testing.assert_close(first, second, rtol=0., atol=0.)
    else:
        assert manager.generate_random_actions.call_count == 2
        assert all(call.args == (1, total_slots-1)
                   for call in manager.generate_random_actions.call_args_list)
        assert not torch.equal(first[1:], second[1:])
# END one supplied seed only means one native solve when the instance has one slot.
