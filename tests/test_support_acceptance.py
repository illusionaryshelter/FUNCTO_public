"""Support targets remain in optimization while delivery uses its own subset."""

# BEGIN exercise real task/manager contracts with a small analytic joint tree.
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from manipulation.curobo_solver_components import PhysicalGradientCore
from manipulation.execution_binding import bind_candidate
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
def test_support_nominal_value_and_gradient_survive_acceptance_subset(case):
    problem = support_problem(case)
    position = torch.zeros((1, 3, 3), dtype=torch.float64, requires_grad=True)
    quaternion = torch.tensor([1., 0., 0., 0.], dtype=torch.float64).expand(1, 3, 4)
    p, r = problem.transform_tool_pose(position, quaternion)
    progress = torch.tensor([[0., .5, 1.]], dtype=torch.float64)
    nominal = problem.residuals(p, r, progress)["support:0"]
    accepted = problem.residuals(p, r, progress, for_acceptance=True)["support:0"]
    signed, _ = problem.functional_inequalities(p, r, progress)
    assert set(signed) == {f"support:0:{key}" for key in ("gap", "body", "footprint", "normal")}
    assert set(problem.functional_acceptance_keys) == {"support:0:body", "support:0:footprint"}
    assert nominal[0, 1] > 0
    assert bool((accepted <= 0).all()) is (case in {"gap", "normal"})
    # Independent expected violations establish the units and event selection.
    expected = {
        "gap": .039,
        "normal": 2 * np.sin(.0005 / 2) - 2 * np.sin(.0001 / 2),
        "body": .002 - problem.numerical.position_m,
        "footprint": .12 + .02 - .1 + .002 - problem.numerical.position_m,
    }[case]
    torch.testing.assert_close(nominal[0, 1], nominal.new_tensor(expected), rtol=1e-10, atol=1e-12)
    gradient = torch.autograd.grad(nominal.sum(), position)[0]
    assert not gradient[:, (0, 2)].any()
    expected_axis = {"gap": (2, 1.), "body": (2, -1.), "footprint": (0, 1.)}
    if case in expected_axis:
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
                augmented, feasible, signed, eligible, _ = fixture.manager._optimizer_evidence
                assert not feasible.any()  # All four inputs violate a nominal target.
                assert bool(eligible.all()) is expected
                assert augmented.sum() > 0
                assert len(signed) == 4
                # Retention uses the manager's actual flags. An otherwise
                # deliverable curve survives with rank one despite a gap or
                # normal target that the original optimizer still pursues.
                tracker = SimpleNamespace(config=SimpleNamespace(num_problems=1, num_particles=1),
                    action_horizon=3, action_dim=2,
                    _candidate_action=torch.full_like(q, 99.),
                    _candidate_objective=torch.full((1,), float("inf")),
                    _candidate_rank=torch.zeros(1, dtype=torch.uint8))
                PhysicalGradientCore._retain_sampled_candidate(tracker, q, eligible, feasible,
                    torch.zeros(1))
                assert int(tracker._candidate_rank[0]) == (1 if expected else 0)
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
    else:
        with pytest.raises(ValueError, match="completed accepted"):
            bind_candidate(trajectory, **arguments)
# END optimization, candidate retention and delivery consume explicit meanings.
