"""Original robot/task values and derivatives under coefficient coordinates."""

# BEGIN a real robot curve checks the solver map independently of IPOPT.
import numpy as np

from manipulation.direct_planner.coefficient_coordinates import coefficient_coordinates
from .test_direct_integration import robot_case


def test_coefficient_map_preserves_curve_values_and_derivative_chain():
    planner, initial = robot_case()
    bound = planner.computation.bind_solver(initial, check_request=lambda: None)
    point = bound.evaluate(initial)
    transform, report = coefficient_coordinates(planner.computation, point)
    rng = np.random.default_rng(29)
    delta = rng.normal(0., .01, initial.size)
    numerical = np.linalg.solve(transform, delta)
    restored = initial+(transform@numerical).reshape(initial.shape)
    np.testing.assert_allclose(restored, initial+delta.reshape(initial.shape), atol=3e-16)
    assert report['minimum_metric_eigenvalue'] > 0.
    direction = rng.normal(size=initial.size)
    direction /= np.linalg.norm(direction)
    physical_direction = transform@direction
    epsilon = 1e-6
    plus = bound.evaluate(initial+epsilon*physical_direction.reshape(initial.shape))
    minus = bound.evaluate(initial-epsilon*physical_direction.reshape(initial.shape))
    expected_objective = point.objective_gradient()@physical_direction
    numerical_objective = (plus.objective_value-minus.objective_value)/(2*epsilon)
    np.testing.assert_allclose(expected_objective, numerical_objective, rtol=1e-7, atol=1e-6)
    jacobian = np.zeros(bound.pattern.shape)
    jacobian[bound.pattern.rows, bound.pattern.columns] = point.jacobian_values()
    np.testing.assert_allclose(jacobian@physical_direction,
        (plus.constraint_values-minus.constraint_values)/(2*epsilon), rtol=2e-6, atol=1e-7)
# END reported task and physical values remain in their original units.
