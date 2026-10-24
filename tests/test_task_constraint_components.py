"""Signed task rows retain the original finite-query allowed sets and Jacobian.

The fixtures are explicit analytic declarations in metres, radians and seconds.
They exercise task semantics without a simulator, robot model or skill branch.
"""

# BEGIN complete task-component contracts for the constrained CPU trajectory solve.
from dataclasses import replace

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from manipulation.goals import SupportSurface
from manipulation.motion import MotionModel
from manipulation.planning_problem import (
    CoordinateDirectionRequirement, CoordinateEventRequirement, InteriorRequirement,
    JointCoupling, MechanismRequirement, MotionPhase, PlanningEvent, PlanningProblem,
    PoseEventRequirement, PosePath, PoseProcessRequirement, SupportRequirement,
)


def task(**requirements):
    """Supply a reference that declares no task condition by itself."""
    reference = PosePath([0., 1.], np.repeat(np.eye(4)[None], 2, axis=0), "source reference")
    return PlanningProblem(("a", "b", "c"), np.zeros(3), np.zeros(3), "tool", "world",
        np.eye(4), np.eye(4), (MotionPhase("reference", reference),), **requirements)


def full_task():
    """Include each existing geometric, actuator and directional declaration."""
    directions = tuple(CoordinateDirectionRequirement(0., 1., 0, kind, 1,
        .08 if kind != "monotone" else 0., .01, "declared direction")
        for kind in ("net", "monotone", "local_rate"))
    model = MotionModel("prismatic", np.zeros((1, 1)), np.eye(4),
        Rotation.from_euler("y", np.pi/2).as_matrix(), np.zeros(3), .03, .04,
        {"source": "observed mechanism declaration"})
    mechanism = MechanismRequirement(0., 1., model, np.eye(4), [[-.2, .5]], [0.],
                                    "declared mechanism", directions)
    surface = SupportSurface(np.zeros(3), np.eye(3)[:, :2], [0., 0., 1.],
        [[-.3, -.3], [.3, -.3], [.3, .3], [-.3, .3]], "observed support")
    points = np.array([[-.02, -.03, 0.], [.025, -.03, .04], [.025, .03, .02]])
    support = SupportRequirement(.5, np.eye(4), points, points[:1], [0., 0., -1.],
        surface, .02, .05, .001, "material support declaration", ("body", "footprint", "gap", "normal"))
    halves = np.concatenate((np.eye(3), -np.eye(3)), axis=0)
    interior = InteriorRequirement(0., 1., np.eye(4), points,
        np.column_stack((halves, np.full(6, -.4))), .001, "observed interior")
    target = np.eye(4); target[0, 3] = .1
    pose_event = PoseEventRequirement(.5, np.eye(4), target, .04, .05,
        "declared material event", yaw_axis=[0., 0., 1.], yaw_tolerance_rad=.2)
    relation = PoseProcessRequirement(0., 1.,
        PosePath([0., 1.], np.repeat(np.eye(4)[None], 2, axis=0), "required relation"),
        np.array([[-.4, .4]]*3+[[-.1, .1]]*3), "explicit process relation")
    return task(mechanisms=(mechanism,), supports=(support,), interiors=(interior,),
        pose_events=(pose_event,), coordinate_events=(CoordinateEventRequirement(
            1., mechanism, .2, .02, "native event"),), pose_relations=(relation,),
        couplings=(JointCoupling(("b", "a"), [[1., -2.]], [.03], "actuator equation"),),
        events=(PlanningEvent("stop", "instance", 1., 2, "hold", "explicit stop", stop=True),))


def arguments(values):
    """Map independent smooth test parameters to all physical query inputs."""
    rows = values.reshape(1, 3, 18)
    rotation = rows[..., 3:6]
    angle = torch.linalg.vector_norm(rotation, dim=-1, keepdim=True)
    quaternion = torch.cat((torch.cos(angle/2), rotation*torch.sin(angle/2)/angle), -1)
    return dict(position=rows[..., :3], quaternion=quaternion,
        progress=values.new_tensor([[0., .5, 1.]]), trajectory_time=values.new_tensor([[0., .7, 1.4]]),
        joint_position=rows[..., 6:9], joint_names=("a", "b", "c"), joint_velocity=rows[..., 9:12],
        tool_linear_velocity=rows[..., 12:15], tool_angular_velocity=rows[..., 15:18])


def variables():
    values = np.arange(54, dtype=float).reshape(3, 18)*.001+.003
    values[:, 0] = [.01, .11, .21]
    values[:, 12] = [.15, .19, .21]
    return torch.tensor(values.ravel(), dtype=torch.float64)


def flattened(problem, values):
    equalities, inequalities = problem.constraint_components(**arguments(values))
    return torch.cat([value.reshape(-1) for group in (equalities, inequalities) for value in group.values()])


def test_complete_inventory_has_stable_rows_and_preserves_all_declared_components():
    problem = full_task()
    eq, iq = problem.constraint_components(**arguments(variables()))
    assert not eq
    assert set(iq) == {
        "progress:lower", "progress:upper", "pose_relation:0:lower", "pose_relation:0:upper",
        "mechanism:0:position", "mechanism:0:rotation", "mechanism_limits:0:lower", "mechanism_limits:0:upper",
        "mechanism:0:direction:0:net", "mechanism:0:direction:1:monotone", "mechanism:0:direction:2:local_rate",
        "support:0:body", "support:0:footprint", "support:0:gap", "support:0:normal", "interior:0",
        "function_pose:0:position", "function_pose:0:rotation", "function_pose:0:yaw",
        "function_coordinate:0:lower", "function_coordinate:0:upper",
        "coupling:0:lower", "coupling:0:upper", "stop:0:lower", "stop:0:upper"}
    assert iq["support:0:body"].shape == (1, 3)
    assert iq["support:0:footprint"].shape == (1, 3, 4)
    assert iq["support:0:gap"].shape == (1, 1, 2)
    assert iq["interior:0"].shape == (1, 3, 3, 6)
    changed = problem.constraint_components(**arguments(variables()+.13))[1]
    assert [(key, value.shape) for key, value in iq.items()] == [(key, value.shape) for key, value in changed.items()]


def test_complete_signed_jacobian_matches_independent_central_difference():
    from manipulation.unified_solver import task_jacobian
    problem, values = full_task(), variables()
    actual = torch.autograd.functional.jacobian(lambda x: flattened(problem, x), values).numpy()
    step = 1e-6
    expected = np.column_stack([(flattened(problem, values+step*direction)-
        flattened(problem, values-step*direction)).numpy()/(2*step)
        for direction in torch.eye(len(values), dtype=values.dtype)])
    np.testing.assert_allclose(actual, expected, atol=2e-8, rtol=2e-6)
    # Include mechanisms, net/monotone/local-rate directions, material point
    # rows, process poses, coupling, coordinate events and stop velocities in
    # the production derivative-order comparison, with every original row.
    controls = values.clone().requires_grad_()
    columns = task_jacobian(flattened(problem, controls), controls,
                            check_request=lambda: None, admit_work=lambda: None)
    np.testing.assert_allclose(columns, actual, atol=2e-13, rtol=2e-13)


@pytest.mark.parametrize("collection,old_prefix,new_prefix", [
    ("pose_relations", "pose_relation:", "pose_relation:"),
    ("mechanisms", "mechanism", "mechanism"),
    ("supports", "support:", "support:"),
    ("interiors", "interior:", "interior:"),
    ("pose_events", "function_pose:", "function_pose:"),
    ("coordinate_events", "function_coordinate:", "function_coordinate:"),
    ("couplings", "coupling:", "coupling:"),
    ("events", "stop:", "stop:"),
])
def test_each_declaration_matches_existing_acceptance(collection, old_prefix, new_prefix):
    full = full_task()
    declarations = {collection: getattr(full, collection)}
    if collection == "coordinate_events":
        declarations["mechanisms"] = full.mechanisms
    problem = task(**declarations)
    # Distinct candidates cross both geometry and actuator tolerances. The
    # comparison is per declaration, avoiding unrelated violations masking it.
    for offset in (-.5, -.05, 0., .03, .5):
        inputs = arguments(variables()+offset)
        old = problem.residuals(**inputs)
        new = problem.constraint_components(**inputs)[1]
        accepted_old = all(bool((value <= 0).all()) for key, value in old.items() if key.startswith(old_prefix))
        accepted_new = all(bool((value <= 0).all()) for key, value in new.items() if key.startswith(new_prefix))
        assert accepted_new == accepted_old


def test_feasible_signed_directions_and_event_bounds_keep_nonzero_derivatives():
    problem = full_task()
    values = variables().requires_grad_()
    iq = problem.constraint_components(**arguments(values))[1]
    # Increasing the terminal native coordinate increases net displacement;
    # increasing an instantaneous velocity improves its lower-rate bound.
    net = iq["mechanism:0:direction:0:net"][0, -1]
    rate = iq["mechanism:0:direction:2:local_rate"][0, 1]
    assert net < 0 and rate < 0
    net_gradient = torch.autograd.grad(net, values, retain_graph=True)[0].reshape(3, 18)
    rate_gradient = torch.autograd.grad(rate, values)[0].reshape(3, 18)
    assert net_gradient[-1, 0] == pytest.approx(-1.)
    assert rate_gradient[1, 12] == pytest.approx(-1.)


def test_free_yaw_has_no_row_and_zero_yaw_is_signed_equality():
    source = PoseEventRequirement(.5, np.eye(4), np.eye(4), .01, .02, "axis event",
                                  yaw_axis=[0., 0., 1.], yaw_tolerance_rad=np.pi)
    problem = task(pose_events=(source,))
    for angle in (.3, -1.2, 2.8):
        p = torch.zeros((1, 3, 3), dtype=torch.float64)
        yaw = torch.tensor(angle, dtype=torch.float64, requires_grad=True)
        q = torch.stack((torch.cos(yaw/2), yaw*0, yaw*0, torch.sin(yaw/2))).expand(1, 3, 4)
        inputs = dict(position=p, quaternion=q, progress=p.new_tensor([[0., .5, 1.]]),
                      trajectory_time=p.new_tensor([[0., 1., 2.]]))
        eq, iq = problem.constraint_components(**inputs)
        assert not eq and not any("yaw" in key for key in iq)
        assert iq["function_pose:0:rotation"].item() == pytest.approx(-.02)
        assert abs(torch.autograd.grad(iq["function_pose:0:rotation"].sum(), yaw, retain_graph=True)[0]) < 1e-14
        exact = task(pose_events=(replace(source, yaw_tolerance_rad=0.),))
        equality, bounded = exact.constraint_components(**inputs)
        assert set(equality) == {"function_pose:0:yaw"}
        assert "function_pose:0:yaw" not in bounded
        assert equality["function_pose:0:yaw"].item() == pytest.approx(angle)
        assert torch.autograd.grad(equality["function_pose:0:yaw"].sum(), yaw)[0].item() == pytest.approx(1.)


def test_support_hard_complement_remains_a_quality_preference():
    full = full_task()
    support = replace(full.supports[0], hard_components=("body", "footprint"))
    problem = task(supports=(support,))
    inputs = arguments(variables())
    _, iq = problem.constraint_components(**inputs)
    assert {key for key in iq if key.startswith("support:")} == {"support:0:body", "support:0:footprint"}
    args = {name: inputs[name] for name in ("position", "quaternion", "progress", "trajectory_time")}
    assert problem.optimizer_preferences(**args, phase="quality").sum() > 0
    assert problem.optimizer_preferences(**args, phase="feasibility").sum() == 0


def test_event_uses_first_physical_arrival_and_ignores_later_plateau_queries():
    event = PoseEventRequirement(.5, np.eye(4), np.eye(4), .01, .02, "event")
    problem = task(pose_events=(event,))
    p = torch.tensor([[[.2, 0., 0.], [.003, 0., 0.], [.8, 0., 0.]]], dtype=torch.float64,
                     requires_grad=True)
    q = torch.tensor([1., 0., 0., 0.], dtype=torch.float64).expand(1, 3, 4)
    _, iq = problem.constraint_components(p, q, p.new_tensor([[.5, .5, .5]]),
                                         trajectory_time=p.new_tensor([[3., 1., 2.]]))
    value = iq["function_pose:0:position"]
    assert value.item() == pytest.approx(-.007)
    gradient = torch.autograd.grad(value.sum(), p)[0]
    np.testing.assert_allclose(gradient.numpy(), [[[0., 0., 0.], [1., 0., 0.], [0., 0., 0.]]])


def test_two_turn_event_retains_full_trajectory_lift_and_physical_order():
    model = MotionModel("revolute", np.zeros((1, 1)), np.eye(4), np.eye(3), np.zeros(3),
                        .01, .01, {"source": "declared angular model"})
    mechanism = MechanismRequirement(0., 1., model, np.eye(4), [[-1., 5*np.pi]], [0.], "angular scope")
    event = CoordinateEventRequirement(1., mechanism, 4*np.pi, .01, "two complete turns")
    problem = task(mechanisms=(mechanism,), coordinate_events=(event,))
    angles = torch.linspace(0., 4*np.pi, 17, dtype=torch.float64)
    q = torch.stack((torch.cos(angles/2), angles*0, angles*0, torch.sin(angles/2)), -1)[None]
    order = torch.tensor([0, 8, 2, 11, 1, 6, 10, 3, 4, 14, 7, 5, 13, 9, 12, 15, 16])
    p = torch.zeros((1, 17, 3), dtype=torch.float64)
    progress = torch.linspace(0., 1., 17, dtype=torch.float64)[None]
    _, iq = problem.constraint_components(p, q[:, order], progress[:, order], trajectory_time=progress[:, order]*2.)
    assert iq["function_coordinate:0:lower"].item() == pytest.approx(-.01)
    assert iq["function_coordinate:0:upper"].item() == pytest.approx(-.01)
# END finite-query task constraints do not claim continuous-interval coverage.


# BEGIN fixed solver rows are produced beside the original physical conditions.
@pytest.mark.parametrize("bounded_sides", [(), (0,), (1,), (0, 1)])
def test_solver_selection_preserves_one_sided_and_unbounded_pose_coordinates(bounded_sides):
    bounds = np.tile([-np.inf, np.inf], (6, 1))
    for side in bounded_sides:
        bounds[:, side] = -.4 if side == 0 else .4
    relation = replace(full_task().pose_relations[0], start=.25, end=.75, pose_bounds=bounds)
    problem = task(pose_relations=(relation,))
    masks = ({}, {})
    values = problem.constraint_components(**arguments(variables()), solver_rows_out=masks)[1]
    expected_keys = [f"pose_relation:0:{('lower', 'upper')[side]}" for side in bounded_sides]
    assert [key for key in values if key.startswith("pose_relation:")] == expected_keys
    for key in expected_keys:
        assert values[key].shape == (1, 3, 6)
        torch.testing.assert_close(masks[1][key],
            torch.tensor([[[False]*6, [True]*6, [False]*6]]))
        assert torch.isfinite(values[key]).all()


def test_solver_selection_preserves_full_components_and_scoped_direction_rows():
    original = full_task()
    mechanism = replace(original.mechanisms[0], start=.25, end=.75,
        directions=tuple(replace(value, start=.25, end=.75)
                         for value in original.mechanisms[0].directions))
    problem = replace(original, mechanisms=(mechanism,),
        pose_relations=tuple(replace(value, start=.25, end=.75) for value in original.pose_relations),
        interiors=tuple(replace(value, start=.25, end=.75) for value in original.interiors),
        coordinate_events=tuple(replace(value, progress=.75, mechanism=mechanism)
                                for value in original.coordinate_events))
    inputs = arguments(variables())
    sample_columns = [0, 1, 1, 1, 2]
    inputs = {key: value[:, sample_columns] if isinstance(value, torch.Tensor) else value
              for key, value in inputs.items()}
    inputs.update(progress=torch.tensor([[0., .25, .5, .75, 1.]], dtype=torch.float64),
                  trajectory_time=torch.tensor([[0., .4, .8, 1.2, 1.6]], dtype=torch.float64))
    masks = ({}, {})
    groups = problem.constraint_components(**inputs, solver_rows_out=masks)
    complete = problem.constraint_components(**inputs)
    for group, mask_group, full in zip(groups, masks, complete):
        assert list(group) == list(mask_group) == list(full)
        for key, values in group.items():
            torch.testing.assert_close(values, full[key], rtol=0., atol=0.)
            assert mask_group[key].shape == values.shape
            assert mask_group[key].dtype == torch.bool
            assert bool((values[~mask_group[key]] <= 0.).all())
    retained = masks[1]
    assert not retained["progress:lower"].any() and not retained["progress:upper"].any()
    expected = torch.tensor([[False, True, True, True, False]])
    torch.testing.assert_close(retained["mechanism:0:position"], expected)
    torch.testing.assert_close(retained["pose_relation:0:lower"], expected[..., None].expand(1, 5, 6))
    torch.testing.assert_close(retained["interior:0"], expected[..., None, None].expand(1, 5, 3, 6))
    for kind, selected in (("net", [False, False, False, True, False]),
                           ("monotone", [False, False, True, True, False]),
                           ("local_rate", [False, True, True, True, False])):
        index = ("net", "monotone", "local_rate").index(kind)
        torch.testing.assert_close(retained[f"mechanism:0:direction:{index}:{kind}"],
                                   torch.tensor([selected]))
    assert retained["support:0:body"].all() and retained["support:0:footprint"].all()
    assert retained["function_pose:0:yaw"].all() and retained["stop:0:lower"].all()
    # Changing candidate values cannot change the declaration selection.
    altered = dict(inputs, position=inputs["position"]+.17,
                   joint_position=inputs["joint_position"]-.13,
                   tool_linear_velocity=inputs["tool_linear_velocity"]+.21)
    changed_masks = ({}, {})
    problem.constraint_components(**altered, solver_rows_out=changed_masks)
    for first, second in zip(masks, changed_masks):
        for key in first:
            torch.testing.assert_close(first[key], second[key], rtol=0., atol=0.)


def test_direction_selection_uses_physical_order_and_retains_missing_net_endpoint():
    problem = full_task()
    inputs = arguments(variables())
    columns = [2, 1, 0, 1, 1]
    inputs = {key: value[:, columns] if isinstance(value, torch.Tensor) else value
              for key, value in inputs.items()}
    inputs.update(progress=torch.tensor([[1., .5, 0., .5, .5]], dtype=torch.float64),
                  trajectory_time=torch.tensor([[2., 1., 0., 1., .8]], dtype=torch.float64))
    masks = ({}, {})
    problem.constraint_components(**inputs, solver_rows_out=masks)
    torch.testing.assert_close(masks[1]["mechanism:0:direction:0:net"],
        torch.tensor([[True, False, False, False, False]]))
    torch.testing.assert_close(masks[1]["mechanism:0:direction:1:monotone"],
        torch.tensor([[True, True, False, False, True]]))
    assert masks[1]["mechanism:0:direction:2:local_rate"].all()

    missing = arguments(variables())
    missing["progress"] = torch.tensor([[0., .5, .75]], dtype=torch.float64)
    masks = ({}, {})
    values = problem.constraint_components(**missing, solver_rows_out=masks)[1]
    key = "mechanism:0:direction:0:net"
    assert torch.isinf(values[key][0, -1]) and masks[1][key][0, -1]
    assert int(masks[1][key].sum()) == 1
# END fixed rows follow declared time layout, including invalid retained conditions.
