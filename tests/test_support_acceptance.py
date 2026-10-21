"""One support declaration serves hard checks and complementary soft guidance."""

# BEGIN exercise real task/manager contracts with a small analytic joint tree.
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from manipulation.curobo_solver_components import PhysicalGradientCore
from manipulation.execution_binding import bind_candidate
from manipulation.functional_augmented_lagrangian import FunctionalAugmentedLagrangian
from manipulation.goals import SupportSurface
from manipulation.planning import JointTrajectory
from manipulation.planning_problem import (
    MotionPhase, PlanningProblem, PosePath, SupportRequirement,
)
from test_scene_query_performance import manager_fixture


def support_problem(case):
    """Specify four independent geometric conditions without a skill branch."""
    contact = np.array([[x, y, 0.] for x in (-.02, .02) for y in (-.02, .02)])
    body = np.concatenate((contact, contact + [0., 0., .05]))
    surface = SupportSurface(np.zeros(3), np.eye(3)[:, :2], [0., 0., 1.],
        [[-.1, -.1], [.1, -.1], [.1, .1], [-.1, .1]], "Observed support test data")
    # The same component is evaluated under each input. These are independent
    # mathematical fixtures, not runtime task-name conditions or tolerances.
    inputs = {
        "gap": (.04, 0., 0., .001),
        "normal": (.0005, .0005, 0., .001),
        "body": (-.002, 0., 0., .01),
        "footprint": (.0005, 0., .12, .001),
    }
    height, tilt, offset, gap = inputs[case]
    tool = np.eye(4)
    tool[:3, :3] = Rotation.from_rotvec([tilt, 0., 0.]).as_matrix()
    tool[:3, 3] = [offset, 0., height]
    requirement = SupportRequirement(.5, np.eye(4), body, contact, [0., 0., -1.],
        surface, gap, .0001, .002, "Declared support feature")
    reference = PosePath(np.array([0., 1.]), np.stack((tool, tool)), "Source reference")
    return PlanningProblem(("first_joint", "second_joint"), np.zeros(2), np.zeros(2),
        "tool", "world", np.eye(4), tool, (MotionPhase("source", reference),),
        supports=(requirement,))


@pytest.mark.parametrize("case", ["gap", "normal", "body", "footprint"])
def test_support_hard_checks_and_soft_guidance_preserve_source_geometry(case):
    problem = support_problem(case)
    position = torch.zeros((1, 3, 3), dtype=torch.float64, requires_grad=True)
    quaternion = torch.tensor([1., 0., 0., 0.], dtype=torch.float64).expand(1, 3, 4)
    p, r = problem.transform_tool_pose(position, quaternion)
    progress = torch.tensor([[0., .5, 1.]], dtype=torch.float64)
    accepted = problem.residuals(p, r, progress)["support:0"]
    # BEGIN support preferences retain their declared quality-stage scope.
    preference = problem.optimizer_preferences(p, r, progress, phase="quality")
    assert not problem.optimizer_preferences(p, r, progress, phase="feasibility").any()
    # END geometric measurements and hard residuals remain phase-independent.
    signed, _ = problem.functional_inequalities(p, r, progress)
    assert set(signed) == {"support:0:body", "support:0:footprint"}
    assert set(signed) == set(problem.functional_constraint_keys)
    assert bool((accepted <= 0).all()) is (case in {"gap", "normal"})
    projected = problem.goal_residual(p, r, progress, goal_ids=(("supports", 0),))
    torch.testing.assert_close(projected.sum(-1)[0, 1], accepted[0, 1])
    if case in {"gap", "normal"}:
        assert preference[0, 1] > 0
    raw = problem.supports[0].raw_inequalities(p, r, problem.numerical)[case]
    # Independent expected violations establish the units and event selection.
    expected = {
        "gap": .039,
        "normal": 2 * np.sin(.0005 / 2) - 2 * np.sin(.0001 / 2),
        "body": .002 - problem.numerical.position_m,
        "footprint": .12 + .02 - .1 + .002 - problem.numerical.position_m,
    }[case]
    torch.testing.assert_close(raw[0, 1], raw.new_tensor(expected), rtol=1e-10, atol=1e-12)
    if case in {"gap", "normal"}:
        scale = (problem.supports[0].gap_tolerance if case == "gap" else
                 2*np.sin(problem.numerical.rotation_rad/2))
        torch.testing.assert_close(preference[0, 1], preference.new_tensor((expected/scale)**2),
                                   rtol=1e-10, atol=1e-12)
    expected_axis = {"gap": (2, 1.), "body": (2, -1.), "footprint": (0, 1.)}
    if case in expected_axis:
        gradient = torch.autograd.grad(raw[0, 1], position)[0]
        assert not gradient[:, (0, 2)].any()
        axis, value = expected_axis[case]
        torch.testing.assert_close(gradient[0, 1, axis], gradient.new_tensor(value))


@pytest.mark.parametrize("case", ["gap", "normal", "body", "footprint"])
def test_manager_retention_metrics_and_binding_share_support_delivery_subset(case, manager_fixture):
    from curobo.types import JointState
    from curobo._src.state.state_robot import RobotState

    problem = support_problem(case)
    expected = case in {"gap", "normal"}
    outputs = {}
    for mode in ("trajectory_optimizer", "trajectory_metrics"):
        with manager_fixture(mode, required=(.5,), capacity=4, task=problem) as fixture:
            fixture.scene.evaluate = lambda spheres, p, r, progress, q, names, **kwargs: q.sum(-1) * 0
            q = torch.zeros((1, 3, 2), requires_grad=True)
            joints = JointState.from_position(q, joint_names=list(problem.joint_names))
            joints.velocity = torch.zeros_like(q)
            joints.dt = torch.ones(1)
            state = RobotState(joint_state=joints, cuda_robot_model_state=fixture.kinematics(joints))
            costs = fixture.manager.compute_costs(state)
            values = dict(zip(costs.names, costs.values))
            if mode == "trajectory_optimizer":
                augmented, feasible, signed, _ = fixture.manager._optimizer_evidence
                assert bool(feasible.all()) is expected
                assert bool(augmented.sum() > 0) is (not expected)
                assert len(signed) == 2
                assert values["skill_pose_equality_preference"].sum().item() == 0.
                # Preference violations preserve the same feasible candidate
                # rank used when every declared hard condition is satisfied.
                tracker = SimpleNamespace(config=SimpleNamespace(num_problems=1, num_particles=1),
                    action_horizon=3, action_dim=2,
                    _candidate_action=torch.full_like(q, 99.),
                    _candidate_objective=torch.full((1,), float("inf")),
                    _candidate_valid=torch.zeros(1, dtype=torch.bool))
                PhysicalGradientCore._retain_sampled_candidate(tracker, q, feasible,
                    torch.zeros(1))
                assert bool(tracker._candidate_valid[0]) is expected
            else:
                violations = values["skill_process"].detach().numpy().reshape(-1)
                assert bool((violations <= 0).all()) is expected
                outputs["violations"] = violations

    trajectory = JointTrajectory(problem.joint_names, np.array([0., 1., 2.]),
        np.zeros((3, 2)), np.zeros((3, 2)), np.array([0., 1.]), np.array([0., 1.]), (), (), {})
    violations = outputs["violations"]
    arguments = dict(problem=problem, scene=SimpleNamespace(to_execution_binding=lambda: {}),
        validation={"violations": violations, "u": np.linspace(0., 1., len(violations))},
        native_report={"accepted": expected})
    if expected:
        binding = bind_candidate(trajectory, **arguments)
        assert binding["validation"]["completed"]
        # Serialization retains the exact source tolerances and feature data.
        assert binding["problem"]["supports"][0]["normal_tolerance"] == .0001
        assert binding["problem"]["supports"][0]["contact_points"] == problem.supports[0].contact_points.tolist()
        assert binding["problem"]["supports"][0]["hard_components"] == ["body", "footprint"]
    else:
        with pytest.raises(ValueError, match="completed accepted"):
            bind_candidate(trajectory, **arguments)
# END optimization, candidate retention and delivery consume explicit meanings.


# BEGIN phase regression uses actual pose derivatives and the original AL terms.
@pytest.mark.parametrize("case", ["gap", "normal"])
def test_feasibility_excludes_support_quality_gradient_at_a_hard_feasible_pose(case):
    problem = support_problem(case)
    position = torch.zeros((1, 3, 3), dtype=torch.float64, requires_grad=True)
    angle = torch.zeros((1, 3), dtype=torch.float64, requires_grad=True)
    quaternion = torch.stack((torch.cos(angle/2), torch.sin(angle/2), angle*0, angle*0), -1)
    p, r = problem.transform_tool_pose(position, quaternion)
    progress = torch.tensor([[0., .5, 1.]], dtype=torch.float64)
    times = torch.tensor([[0., 1., 2.]], dtype=torch.float64)
    signed, _ = problem.functional_inequalities(p, r, progress, trajectory_time=times)
    augmented = FunctionalAugmentedLagrangian(problem.functional_constraint_keys)
    augmented.bind(progress)
    hard_cost = sum(augmented.terms(signed).values()).sum()
    assert hard_cost.item() == 0.
    preferences = {phase: problem.optimizer_preferences(p, r, progress,
        trajectory_time=times, phase=phase) for phase in ("feasibility", "quality")}
    assert not preferences["feasibility"].any()
    assert preferences["quality"][0, 1] > 0
    assert not preferences["quality"][:, (0, 2)].any()
    first = torch.autograd.grad(hard_cost+preferences["feasibility"].sum(),
        (position, angle), retain_graph=True)
    for derivative in first:
        torch.testing.assert_close(derivative, torch.zeros_like(derivative), rtol=0, atol=0)
    second = torch.autograd.grad(hard_cost+preferences["quality"].sum(), (position, angle))
    for derivative in second:
        assert not derivative[:, (0, 2)].any()
    # Lowering a lifted face and reducing a positive tilt are the original
    # geometric preference directions at this same hard-feasible event pose.
    if case == "gap":
        assert second[0][0, 1, 2] > 0
    else:
        assert second[1][0, 1] > 0
# END phase selection changes optional gradients, not support acceptance.


# BEGIN changing a task declaration changes every hard consumer together.
@pytest.mark.parametrize("case", ["gap", "normal"])
def test_explicit_support_hard_components_and_continuation_identity(case):
    from dataclasses import replace
    from manipulation.planning import _obligation_contract
    problem = support_problem(case)
    problem.supports = (replace(problem.supports[0], hard_components=("gap", "body", "footprint", "normal")),)
    position = torch.zeros((1, 1, 3), dtype=torch.float64, requires_grad=True)
    angle = torch.zeros((1, 1), dtype=torch.float64, requires_grad=True)
    quaternion = torch.stack((torch.cos(angle/2), torch.sin(angle/2), angle*0, angle*0), -1)
    p, r = problem.transform_tool_pose(position, quaternion)
    progress = torch.tensor([[.5]], dtype=torch.float64)
    assert problem.evaluate(p, r, progress).item() > 0
    assert problem.goal_residual(p, r, progress, goal_ids=(("supports", 0),)).sum().item() > 0
    assert len(problem.functional_constraint_keys) == 4
    signed, _ = problem.functional_inequalities(p, r, progress)
    augmented = FunctionalAugmentedLagrangian(problem.functional_constraint_keys)
    augmented.bind(progress)
    hard_cost = sum(augmented.terms(signed).values()).sum()
    derivatives = []
    for phase in ("feasibility", "quality"):
        preference = problem.optimizer_preferences(p, r, progress, phase=phase)
        assert preference.item() == 0.
        derivatives.append(torch.autograd.grad(hard_cost+preference.sum(),
            (position, angle), retain_graph=True))
    for first, second in zip(*derivatives):
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    if case == "gap":
        assert derivatives[0][0][0, 0, 2] > 0
    else:
        assert derivatives[0][1][0, 0] > 0
    assert _obligation_contract(problem)["supports"][0]["hard_components"] == ["gap", "body", "footprint", "normal"]
# END the declared input, not a query purpose, determines the feasible set.
