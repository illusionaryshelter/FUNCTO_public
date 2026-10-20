"""Initial support retains exact source-cell identity and observed height."""

# BEGIN implicit-field regression cases replace removed solid representations.
import numpy as np
import torch

from manipulation.initial_support import InitialSupportRelation, PreparedInitialSupport
from manipulation.implicit_distance import query_distance
from manipulation.observed_mapping import FrozenObservedField
from manipulation.planning_problem import PosePath
from manipulation.planning_scene import Attachment, CollisionOwner


def make_field(name, shape=(2, 2, 2)):
    return FrozenObservedField(-np.ones(shape), np.ones(shape, bool), weights=np.ones(shape),
        origin=[-.002]*3, voxel_size=.004, name=name, source="synthetic supported samples")


def owners(shape=(2, 2, 2)):
    def path():
        return PosePath(np.array([0., 1.]), np.stack((np.eye(4), np.eye(4))), "declared entrance")
    actor=CollisionOwner("payload", make_field("payload"), path(), "observed payload",
        moving=True, attachments=(Attachment((0., 1.), [0., 0., 0.], [1., 0., 0., 0.], "held", None),))
    target=CollisionOwner("support", make_field("support", shape), path(), "observed plane layer")
    return actor, target


def relation():
    return InitialSupportRelation("entry", "payload", "support", (0., 1.), [0., 0., 0.], [0., 0., 1.],
        dict(kind="observed_plane_with_declared_initial_support", observation_ids=["camera"],
             assumption="Declared current support", scope="Observed plane layer"))


def pose(z=0., requires_grad=False):
    return (torch.tensor([[[0., 0., z]]], dtype=torch.float64, requires_grad=requires_grad),
            torch.tensor([[[1., 0., 0., 0.]]], dtype=torch.float64))


def test_complete_selected_cell_permission_uses_identity_at_its_boundary():
    actor, target=owners()
    model=PreparedInitialSupport(relation(), actor, target, 1e-6)
    assert model.selected_cells == (0,)
    result=query_distance(actor.query, np.eye(4), target.query, np.eye(4),
                          permission=model.permission(np.eye(4)))
    assert result["status"] == "clear"


def test_touching_neighbor_is_unresolved_and_remains_outside_permission():
    actor, target=owners((3, 2, 2))
    model=PreparedInitialSupport(relation(), actor, target, 1e-6)
    assert model.selected_cells == (0,)
    result=query_distance(actor.query, np.eye(4), target.query, np.eye(4),
                          permission=model.permission(np.eye(4)))
    assert result["status"] == "unresolved"
    # BEGIN exact contact does not expand permission to the adjacent cell.
    moved = np.eye(4)
    moved[0, 3] = .001
    result = query_distance(actor.query, moved, target.query, np.eye(4),
                            permission=model.permission(np.eye(4)))
    assert result["status"] == "intersection"
    # END subsequent penetration is still checked against the unpermitted cell.


def test_whole_observed_height_detects_deepening_and_preserves_direction():
    actor, target=owners()
    model=PreparedInitialSupport(relation(), actor, target, 1e-6)
    current=pose(-.001, requires_grad=True)
    progress=torch.tensor([[.5]], dtype=torch.float64)
    values=model.components(current, pose(), progress)
    assert values["initial_support/entry"].item() > .000998
    assert values["unresolved/initial_support/entry"].item() == 0
    values["initial_support/entry"].sum().backward()
    assert current[0].grad[0, 0, 2].item() == -1
    for z in (0., .02):
        assert all(value.item() == 0 for value in model.components(pose(z), pose(), progress).values())
    assert all(value.item() == 0 for value in model.components(current, pose(), progress*2).values())


def test_entrance_unresolved_neighbor_does_not_reject_clear_lifted_state():
    actor, target=owners((3, 2, 2))
    model=PreparedInitialSupport(relation(), actor, target, 1e-6)
    assert model.unresolved_cells
    values=model.components(pose(.02), pose(), torch.tensor([[.5]], dtype=torch.float64))
    assert all(value.item() == 0 for value in values.values())
# END current collision queries separately retain all unpermitted components.
