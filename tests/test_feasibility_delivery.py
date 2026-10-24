"""A real CPU constrained solve reaches the formal first-candidate binding."""

# BEGIN this fixture declares no collision entities; its task and FK remain real.
from dataclasses import replace
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from manipulation.execution_binding import check_result
from manipulation.physical_joint_domain import CubicJointDomain
from manipulation.planning import MotionBackend, _ConstrainedPlanner
from manipulation.planning_problem import PoseEventRequirement
from manipulation.unified_solver import TrajectoryComputation
from test_motion_backend_contract import Native, Scene
from test_unified_solver import computation


def test_real_feasibility_solve_qualifies_and_binds_through_motion_backend():
    original, _ = computation(entry_yaw_event=True, terminal_yaw_tolerance=np.pi)
    problem = replace(original.problem, start_q=np.zeros(3), start_dq=np.zeros(3),
        pose_events=(PoseEventRequirement(1., np.eye(4), np.eye(4), .02, .1,
            'declared terminal pose'),), couplings=(), events=(),
        source_timestamps=np.array([0., original.duration]), source_progress=np.array([0., 1.]))
    domain = CubicJointDomain(original.bounds, torch.zeros(3, dtype=torch.float64),
        torch.zeros(3, dtype=torch.float64), .5, 6)

    class EmptyScene(Scene):
        """An explicit empty geometric domain with ordinary scene interfaces."""
        owners, contacts, required_progress_anchors = (), (), ()
        last_query_report = {}
        robot_model = SimpleNamespace(_geometries={})
        robot_validator = SimpleNamespace(T_reference_base=np.eye(4))
        kinematics = original.kinematics

        def evaluate(self, spheres, position, quaternion, progress, joints, names, **kwargs):
            return torch.zeros_like(progress)

        def optimization_objective(self, position, quaternion, progress, joints, names, **kwargs):
            return joints.sum()*0.

        def to_execution_binding(self):
            return {'kind': 'declared_empty_geometric_domain', 'entities': []}

    scene = EmptyScene()
    bound = TrajectoryComputation(problem=problem, scene=scene, kinematics=original.kinematics,
        domain=domain, joint_names=original.joint_names, parameters=original.parameters,
        progress=original.progress.numpy()[0], ordinary_indices=original.ordinary_indices,
        motion_configuration=original.motion_configuration, native_dt=original.native_dt,
        constraint_weight=original.constraint_weight, max_host_rss_bytes=original.max_host_rss_bytes)
    initial = np.zeros((domain.knots, 3))
    initial[:, 0] = np.linspace(0., .08, domain.knots)
    native = Native([])
    native.problem, native.scene, native.computation = problem, scene, bound
    native.physical_domain, native.pin_kinematics = domain, original.kinematics
    native.joint_names, native.action_horizon = bound.joint_names, domain.knots
    native.robot_metadata, native.finger_trajectory = {}, None
    native.anchors = problem.anchors
    native.motion_configuration = bound.motion_configuration
    native.formal_clock = {'knot_dt': .5, 'duration': bound.duration, 'native_dt': bound.native_dt}
    limits = SimpleNamespace(joint_names=list(bound.joint_names), position=bound.bounds[0],
        velocity=bound.bounds[1], acceleration=bound.bounds[2], jerk=bound.bounds[3])
    native._verification_native = SimpleNamespace(get_cost=lambda _: SimpleNamespace(
        config=SimpleNamespace(joint_limits=limits)))
    for name in ('_controls_curve', '_candidate_queries', '_candidate_robot_state',
            '_candidate_objective', '_candidate_metrics', '_candidate_geometry',
            '_candidate_domain_validation', 'evaluate_candidate', 'evaluate_reference',
            'select_candidate', 'optimize', 'export'):
        setattr(native, name, MethodType(getattr(_ConstrainedPlanner, name), native))
    native.reference_seeds = lambda joint_reference=None: (torch.tensor(initial)[None, None], [])

    # Record the actual preference-stage calls without substituting any values.
    stages, preferences = [], problem.optimizer_preferences
    def record_preferences(*args, phase, **kwargs):
        stages.append(phase)
        return preferences(*args, phase=phase, **kwargs)
    problem.optimizer_preferences = record_preferences

    backend = MotionBackend({}, robot_validator=object())
    try:
        # Allocation is replaced at the GPU-owning constructor only. The
        # declared empty scene needs no geometry workspace; all numerical,
        # qualification, clock, ledger and execution-binding calls stay real.
        with patch('manipulation.planning._ConstrainedPlanner', return_value=native), \
             patch('manipulation.planning._planning_workspace_bytes',
                   return_value=SimpleNamespace(pending_cpu_bytes=0)):
            result = backend.plan(problem, scene)
        assert result.success, result.diagnostics.get('failure_traceback', result.diagnostics)
        assert not result.diagnostics['reference_candidate']['accepted']
        assert result.request_ledger['consumed_planning']['trajectory_solves'] == 1
        assert result.request_ledger['consumed_planning']['validation_passes'] == 2
        runtime, = native.last_optimizer_runtime
        assert runtime['solver_called'] and runtime['objective_evaluations'] > 1
        assert stages.count('feasibility') == runtime['objective_evaluations']
        assert 'quality' in stages
        assert all(value <= 0. for value in result.trajectory.residuals.values())
        assert all(value <= 0. for value in result.diagnostics['returned_candidate']['residuals'].values())
        assert check_result(result)['request_ledger'] == result.request_ledger
        assert backend._request_owner is None
    finally:
        backend.close()
# END binding succeeds only after unchanged strict task and physical qualification.
