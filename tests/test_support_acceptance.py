"""One support declaration serves hard checks and complementary soft guidance."""

# BEGIN signed rows, original residuals and execution binding share the same support declaration.
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from manipulation.execution_binding import bind_candidate
from manipulation.goals import SupportSurface
from manipulation.planning import JointTrajectory
from manipulation.planning_problem import (
    MotionPhase, PlanningProblem, PosePath, SupportRequirement,
)


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
    _, values = problem.constraint_components(p, r, progress, trajectory_time=progress)
    signed = {key: value for key, value in values.items() if key.startswith("support:")}
    assert set(signed) == {"support:0:body", "support:0:footprint"}
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
def test_signed_rows_metrics_and_binding_share_support_delivery_subset(case):
    problem = support_problem(case)
    expected = case in {"gap", "normal"}
    progress = torch.tensor([[0., .5, 1.]], dtype=torch.float64)
    p, r = problem.sample_reference(progress)
    equalities, signed = problem.constraint_components(p, r, progress, trajectory_time=2*progress)
    assert not equalities
    support_rows = [value for key, value in signed.items() if key.startswith("support:")]
    assert len(support_rows) == 2
    assert all(bool((value <= 0).all()) for value in support_rows) is expected
    violations = problem.residuals(p, r, progress, trajectory_time=2*progress)["support:0"].numpy().reshape(-1)
    assert bool((violations <= 0).all()) is expected
    trajectory = JointTrajectory(problem.joint_names, np.array([0., 1., 2.]),
        np.zeros((3, 2)), np.zeros((3, 2)), np.array([0., 1.]), np.array([0., 1.]), (), (), {})
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
# END numerical constraints, original metrics and delivery consume explicit meanings.


# BEGIN phase regression compares original hard violations and complementary soft derivatives.
@pytest.mark.parametrize("case", ["gap", "normal"])
def test_feasibility_excludes_support_quality_gradient_at_a_hard_feasible_pose(case):
    problem = support_problem(case)
    position = torch.zeros((1, 3, 3), dtype=torch.float64, requires_grad=True)
    angle = torch.zeros((1, 3), dtype=torch.float64, requires_grad=True)
    quaternion = torch.stack((torch.cos(angle/2), torch.sin(angle/2), angle*0, angle*0), -1)
    p, r = problem.transform_tool_pose(position, quaternion)
    progress = torch.tensor([[0., .5, 1.]], dtype=torch.float64)
    times = torch.tensor([[0., 1., 2.]], dtype=torch.float64)
    _, signed = problem.constraint_components(p, r, progress, trajectory_time=times)
    hard_cost = sum(torch.relu(value).square().sum() for key, value in signed.items() if key.startswith("support:"))
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
    _, values = problem.constraint_components(p, r, progress, trajectory_time=progress)
    signed = {key: value for key, value in values.items() if key.startswith("support:")}
    assert len(signed) == 4
    hard_cost = sum(torch.relu(value).square().sum() for value in signed.values())
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
