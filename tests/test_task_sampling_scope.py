"""A finite pose-process query does not certify its intervening cubic segment."""

# BEGIN explicit continuous-time counterexample for the task-query contract.
import numpy as np
from scipy.interpolate import CubicHermiteSpline
import torch

from manipulation.planning_problem import (
    MotionPhase, PlanningProblem, PosePath, PoseProcessRequirement,
)


def test_pose_process_nodes_can_pass_while_same_cubic_violates_between_them():
    """The target-local x band is 10 mm; the same saved cubic reaches 20 mm.

    The robot-independent tool path x(t)=0.08*t*(1-t), t in [0,1] seconds,
    is exactly represented by its endpoint positions and velocities. Both
    endpoint queries satisfy the declared process, while the interior
    stationary point violates it. This distinguishes task sampling from the
    independent exact polynomial extrema used for physical joint bounds.
    """
    target = PosePath([0., 1.], np.repeat(np.eye(4)[None], 2, axis=0),
                      "Declared constant reference frame")
    bounds = np.array([[-.01, .01]]*3+[[-.1, .1]]*3)
    relation = PoseProcessRequirement(0., 1., target, bounds,
                                      "Explicit Cartesian process band")
    problem = PlanningProblem(("joint",), np.zeros(1), np.zeros(1), "tool", "world",
        np.eye(4), np.eye(4), (MotionPhase("reference", target),),
        pose_relations=(relation,))
    curve = CubicHermiteSpline([0., 1.], [[0., 0., 0.], [0., 0., 0.]],
                               [[.08, 0., 0.], [-.08, 0., 0.]])

    def query(times):
        progress = torch.tensor([times], dtype=torch.float64)
        position = torch.tensor(curve(times)[None])
        quaternion = torch.zeros((*position.shape[:-1], 4), dtype=position.dtype)
        quaternion[..., 0] = 1.
        inputs = dict(position=position, quaternion=quaternion, progress=progress,
                      trajectory_time=progress)
        inequalities = problem.constraint_components(**inputs)[1]
        residual = problem.residuals(**inputs)["pose_relation:0"]
        return inequalities, residual

    nodes, node_residual = query([0., 1.])
    assert all(bool((value <= 0).all()) for value in nodes.values())
    assert bool((node_residual == 0).all())
    interior, interior_residual = query([0., .5, 1.])
    assert curve(.5)[0] == .02
    assert interior["pose_relation:0:upper"][0, 1, 0] > .009
    assert interior_residual[0, 1] > .009
# END finite query acceptance retains its stated temporal scope.
