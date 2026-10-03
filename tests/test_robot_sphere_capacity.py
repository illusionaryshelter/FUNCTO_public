"""CPU boundary checks for complete refined robot collision covers."""

# BEGIN fixture export/reload and planner admission share one bounded capacity.
from copy import deepcopy
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import pytest
import yaml

from manipulation.planning import PlanningBudget

sys.path.insert(0, str(Path(__file__).parent / "planning"))
from robot_fixture import export_fixture, validate_fixture_collision_contract


def _arguments(root, second_subdivisions):
    """Two native boxes retain full cells and a bounded independent joint."""
    source = root / "scene.xml"
    source.write_text('''<mujoco><compiler angle="radian"/><asset/><worldbody>
      <body name="anchor"><geom name="fixed_box" type="box" size=".1 .1 .05"/>
        <body name="rotor" pos="0 0 .4">
          <joint name="spin" type="hinge" axis="0 0 1" limited="true" range="-1 1"/>
          <geom name="moving_box" type="box" size=".1 .1 .05"/>
          <site name="tool" pos="0 0 .1"/>
        </body>
      </body></worldbody></mujoco>''')
    catalog = root / "catalog.json"
    catalog.write_text(json.dumps({"joints": [{"name": "spin", "qpos_count": 1,
        "qvel_count": 1, "qpos_start": 0, "qvel_start": 0}]}))
    episode = root / "episode.h5"
    with h5py.File(episode, "w") as stream:
        stream.create_dataset("states/qpos", data=np.array([[-.2], [.2]]))
    return dict(scene_path=source, catalog_path=catalog, episode_path=episode,
        root_body="anchor", collision_root="anchor", tcp_site="tool",
        arm_joint_names=["spin"], finger_joint_names=[], velocity_limits={"spin": 1.},
        effort_limits={"spin": 2.}, cover_specifications={
            "fixed_box": {"subdivisions": [5, 8, 8]},
            "moving_box": {"subdivisions": second_subdivisions}},
        initial_frame=0, output=root / "fixture", binding_audit_path=catalog,
        max_acceleration=2., max_jerk=10., mechanical_contact_pairs=[])


def test_complete_capacity_boundary_exports_reloads_and_rejects_missing_sphere(tmp_path):
    """Exactly 640 certified balls survive export; losing one remains invalid."""
    fixture = export_fixture(**_arguments(tmp_path, [5, 8, 8]))
    configuration = yaml.safe_load(Path(fixture["robot_config_path"]).read_text())
    checked = validate_fixture_collision_contract(fixture, configuration)
    assert fixture["sphere_count"] == checked["sphere_count"] == 640
    assert checked["complete_native_geometry_count"] == 2
    assert PlanningBudget(max_robot_spheres=640).max_robot_spheres == 640
    damaged = deepcopy(configuration)
    damaged["robot_cfg"]["kinematics"]["collision_spheres"]["rotor"].pop()
    with pytest.raises(ValueError, match="Serialized spheres differ"):
        validate_fixture_collision_contract(fixture, damaged)


def test_above_capacity_is_rejected_without_dropping_native_cells(tmp_path):
    """A 641-ball complete cover fails export and planner resource admission."""
    with pytest.raises(ValueError, match="640-sphere budget"):
        export_fixture(**_arguments(tmp_path, [3, 107, 1]))
    assert not (tmp_path / "fixture" / "manifest.json").exists()
    with pytest.raises(ValueError, match="bounded local single-batch"):
        PlanningBudget(max_robot_spheres=641)
# END the larger capacity preserves complete-volume and finite-budget rejection.
