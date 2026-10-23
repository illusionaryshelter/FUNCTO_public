"""Attached tool members and payloads share internal collision membership."""

# BEGIN real native entities and raw fields exercise assembly pair semantics.
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from manipulation.observed_mapping import FrozenObservedField
from manipulation.planning_problem import PosePath
from manipulation.planning_scene import Attachment, CollisionOwner
from test_native_scene_contract import make_scene


def payload(name="payload", *, release=None, attached=True):
    field = FrozenObservedField(-np.ones((2, 2, 2)), np.ones((2, 2, 2), bool),
        weights=np.ones((2, 2, 2)), origin=[-.005] * 3, voxel_size=.04,
        name=name, source="supported synthetic observation")
    relations = (Attachment((0., .75), [0., 0., 0.], [1., 0., 0., 0.],
                            f"{name}:attachment", release),) if attached else ()
    return CollisionOwner(name, field,
        PosePath(np.array([0., 1.]), np.stack((np.eye(4), np.eye(4))), "declared path"),
        "observed entity", attachments=relations)


def assembly(tmp_path, *, members=("first", "second"), owners=None, background=False):
    scene = make_scene(tmp_path, occupied_background=background)
    scene.tool_geometry_names = tuple(members)
    scene.owners = tuple(owners) if owners is not None else (payload(),)
    return scene


def evaluate(scene, progress=(.5,), *, permission_progress=None):
    count = len(progress)
    q = torch.tensor([[[.04, -.02]] * count], dtype=torch.float64, requires_grad=True)
    return scene.components(None, torch.zeros((1, count, 3), dtype=torch.float64),
        torch.tensor([[[1., 0., 0., 0.]] * count], dtype=torch.float64),
        torch.tensor([progress], dtype=torch.float64), q, ("left_slide", "right_slide"),
        trajectory_time=torch.zeros((1, count)),
        contact_permission_progress=(None if permission_progress is None else
                                     torch.tensor([permission_progress], dtype=torch.float64)))


def test_all_declared_tool_members_are_internal_with_attached_payload(tmp_path):
    scene = assembly(tmp_path)
    values = evaluate(scene)
    assert values["collision/robot/payload"].item() == 0.
    assert scene.last_query_report["internal_assembly_pair_count"] == 2
    assert values["collision/robot/self"].item() == 0.
    saved = scene.to_execution_binding()
    assert saved["tool_geometry_names"] == ["first", "second"]
    assert "holding_contacts" not in saved


def test_payload_still_collides_with_robot_entities_outside_the_tool(tmp_path):
    scene = assembly(tmp_path, members=("first",))
    assert evaluate(scene)["collision/robot/payload"].item() > 0.
    rows = scene.last_query_report["nonclear_pairs"]
    assert any(row["first"] == "second" and row["second"] == "payload" for row in rows)


def test_tool_and_payload_each_keep_their_external_background_collision(tmp_path):
    scene = assembly(tmp_path, background=True)
    values = evaluate(scene)
    assert values["collision/robot/payload"].item() == 0.
    assert values["collision/robot/background"].item() > 0.
    assert values["collision/payload/background"].item() > 0.


def test_separate_owner_remains_an_external_obstacle_for_every_member(tmp_path):
    scene = assembly(tmp_path, owners=(payload(), payload("obstacle", attached=False)))
    values = evaluate(scene)
    assert values["collision/robot/payload"].item() == 0.
    assert values["collision/robot/obstacle"].item() > 0.
    assert values["collision/payload/obstacle"].item() > 0.


@pytest.mark.parametrize("released", [False, True])
def test_only_the_declared_release_ends_internal_membership(tmp_path, released):
    scene = assembly(tmp_path, owners=(payload(release="release" if released else None),))
    values = evaluate(scene, (.5, .75, 1.))
    expected = [False, released, released]
    assert (values["collision/robot/payload"][0] > 0).tolist() == expected
    assert scene.last_query_report["internal_assembly_pair_count"] == (2 if released else 6)


def test_two_owners_on_the_same_tool_are_internal_until_one_is_released(tmp_path):
    scene = assembly(tmp_path, owners=(payload(), payload("second_payload", release="release")))
    values = evaluate(scene, (.5, .75))
    assert (values["collision/payload/second_payload"][0] > 0).tolist() == [False, True]


def test_attachment_model_validation_needs_no_material_contact_ids(tmp_path):
    scene = assembly(tmp_path)
    condition = dict(condition_id="payload:attachment", carrier_id="payload",
                     kind="declared_attachment", progress=0., T_carrier_eef=np.eye(4))
    problem = SimpleNamespace(reference_frame="world", conditions=(condition,),
                              pose_events=(), events=())
    scene.validate_problem(problem)
    assert "contact_rule_ids" not in condition


def test_finite_contact_boundary_does_not_advance_attachment_membership(tmp_path):
    owner = payload()
    owner.attachments[0].interval = (.5, .75)
    scene = assembly(tmp_path, owners=(owner,))
    values = evaluate(scene, (.5 - 1e-6,), permission_progress=(.5,))
    assert values["collision/robot/payload"].item() > 0.
    assert scene.last_query_report["internal_assembly_pair_count"] == 0


def test_source_topology_includes_palm_and_pads_but_not_the_arm(tmp_path):
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent / "planning"))
    from robot_validation_fixture import tool_geometry_names
    path = tmp_path / "tool.xml"
    path.write_text("""<mujoco><worldbody><body name="arm">
      <geom name="arm_shape" type="box" size=".01 .01 .01"/>
      <body name="tool" pos="0 0 .1"><geom name="palm" type="box" size=".02 .02 .01"/>
        <site name="tcp"/><body name="a"><joint name="drive_a" type="slide"/>
          <geom name="a_body" type="box" size=".01 .01 .01"/>
          <body name="a_tip"><geom name="a_pad" type="box" size=".001 .01 .01"/></body>
        </body><body name="b"><joint name="drive_b" type="slide"/>
          <geom name="b_body" type="box" size=".01 .01 .01"/>
          <body name="b_tip"><geom name="b_pad" type="box" size=".001 .01 .01"/></body>
        </body>
      </body></body></worldbody></mujoco>""")
    fixture = dict(robot_mjcf_path=str(path), finger_joint_names=["drive_a", "drive_b"],
        collision_geometries=[dict(geom=name) for name in
            ("arm_shape", "palm", "a_body", "a_pad", "b_body", "b_pad")])
    assert tool_geometry_names(fixture) == ("palm", "a_body", "a_pad", "b_body", "b_pad")
# END tool membership neither enlarges geometry nor suppresses external pairs.
