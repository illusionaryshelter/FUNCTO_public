"""CPU adapter regression: complete collision assembly and measured feedback."""

# BEGIN compiled robot adapter checks without a GL renderer or scene object state.
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "planning"))
from recorded_observation import classify_robot_observation
from manipulation.robot_self_filter import ENVIRONMENT, SELF


def test_collision_groups_fixed_mount_feedback_and_unrelated_scene_exclusion(tmp_path):
    path = tmp_path / "known.xml"
    path.write_text('''<mujoco><worldbody>
      <body name="robot"><geom name="mount" type="cylinder" size=".1 .1" pos="0 0 2" group="0"/>
        <geom name="visual_only" type="box" size=".1 .1 .1" pos="1 0 1" contype="0" conaffinity="0" group="1"/>
        <body name="moving" pos="0 0 2"><joint name="slide" type="slide" axis="1 0 0"/>
          <geom name="moving_collision" type="box" size=".1 .1 .1" group="5"/></body>
      </body><body name="object"><geom name="forbidden_object" type="sphere" size=".2" pos="0 0 .5"/></body>
    </worldbody></mujoco>''')
    calibration = {"width": 3, "height": 3, "K": np.array([[2., 0., 1.], [0., 2., 1.], [0., 0., 1.]]),
                   "T_world_camera_cv": np.eye(4), "near_m": .01, "far_m": 10.}
    depth = np.full((3, 3), 2.)
    result = classify_robot_observation(scene_xml=path, robot_root="robot", joint_map=[("q", "slide", 1.)],
        joint_positions={"q": 1.}, T_world_robot_root=np.eye(4), calibration=calibration,
        observed_depth=depth, padding_scale=1., padding_offset_m=0., source="synchronized CPU test")
    assert result["self_filter"]["labels"][1, 1] == SELF  # fixed installation
    assert result["self_filter"]["labels"][1, 2] == SELF  # measured translated collision geom
    assert result["self_filter"]["labels"][1, 0] == ENVIRONMENT
    assert {row["name"] for row in result["provenance"]["collision_geometries"]} == {"mount", "moving_collision"}
    assert set(result["provenance"]["body_names"]) == {"robot", "moving"}
    assert not result["provenance"]["renderer_used"]
    np.testing.assert_array_equal(depth, np.full((3, 3), 2.))
# END fixed mount and moving robot ownership retain separate native solids.
