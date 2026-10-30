"""Check native fluid geometry semantics and preservation of the physical model."""

from pathlib import Path
import sys

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "planning"))
from simulation_model import build_simulation_model


def model_file(tmp_path, inertia):
    """Declare a robot assembly and an independent body under the same medium."""
    path = tmp_path / 'physical.xml'
    path.write_text(f'''<mujoco>
      <option density="1.2" viscosity="0.00002" gravity="0 0 0"/>
      <worldbody>
        <body name="assembly">
          <body name="moving">
            <joint name="axis" type="hinge"/>
            <inertial mass="0.01" diaginertia="{inertia} {inertia} {inertia}" pos="0 0 0"/>
            <geom name="physical" type="box" size="0.008 0.004 0.008" contype="0" conaffinity="1"/>
            <geom name="visual" type="box" size="0.008 0.004 0.008" contype="0" conaffinity="0"/>
          </body>
          <body name="fixed_frame" pos="0 0 0.1">
            <inertial mass="0.02" diaginertia="0.001 0.001 0.001" pos="0 0 0"/>
          </body>
        </body>
        <body name="independent" pos="2 0 0">
          <joint name="other_axis" type="hinge"/>
          <geom name="other_physical" type="box" size="0.1 0.1 0.1"/>
        </body>
      </worldbody>
    </mujoco>''')
    return path


def fluid(model):
    data = mujoco.MjData(model)
    data.qvel[:] = [1.3, -0.7]
    mujoco.mj_forward(model, data)
    return data.qfrc_fluid.copy()


def test_declared_robot_geometry_uses_or_mask_and_excludes_visual_copy(tmp_path):
    path = model_file(tmp_path, .01)
    original = mujoco.MjModel.from_xml_path(str(path))
    model, report = build_simulation_model(path, robot_root='assembly')
    assert [row['name'] for row in report['geoms']] == ['physical']
    assert model.geom_fluid[model.geom('physical').id, 0] == 1.
    assert model.geom_fluid[model.geom('visual').id, 0] == 0.
    assert model.geom_fluid[model.geom('other_physical').id, 0] == 0.
    assert report['native_inertia_bodies'] == ['fixed_frame']
    # The independent body retains exactly its original medium/model response.
    np.testing.assert_allclose(fluid(model)[1], fluid(original)[1], rtol=0., atol=0.)


def test_native_geometry_fluid_is_independent_of_supplied_inertial_dimensions(tmp_path):
    path = model_file(tmp_path, .01)
    original = mujoco.MjModel.from_xml_path(str(path))
    first, _ = build_simulation_model(path, robot_root='assembly')
    path = model_file(tmp_path, .02)
    second, _ = build_simulation_model(path, robot_root='assembly')
    np.testing.assert_allclose(fluid(first), fluid(second), rtol=1e-13, atol=1e-15)
    assert abs(fluid(first)[0]) < abs(fluid(original)[0]) / 1000.


def test_official_robot_build_preserves_inertia_collision_and_state_layout():
    root = HERE.parents[1]
    path = root / 'FUNCTO_public/scene/drawer_original/scene.xml'
    original = mujoco.MjModel.from_xml_path(str(path))
    model, report = build_simulation_model(path, robot_root='robot0_link0')
    assert (model.nq, model.nv, model.nu, model.nbody, model.ngeom) == (
        original.nq, original.nv, original.nu, original.nbody, original.ngeom)
    # These arrays jointly define rigid-body motion, collision geometry,
    # contact parameters, actuator limits and the saved integration-state order.
    fields = ('body_mass', 'body_inertia', 'body_ipos', 'body_iquat', 'body_pos',
        'body_quat', 'body_parentid', 'jnt_type', 'jnt_qposadr', 'jnt_dofadr',
        'jnt_range', 'dof_armature', 'dof_damping', 'dof_frictionloss',
        'geom_type', 'geom_bodyid', 'geom_size', 'geom_pos', 'geom_quat',
        'geom_contype', 'geom_conaffinity', 'geom_friction', 'geom_solref',
        'geom_solimp', 'geom_dataid', 'mesh_vert', 'mesh_face',
        'actuator_trnid', 'actuator_gear', 'actuator_ctrlrange',
        'actuator_forcerange', 'actuator_gainprm', 'actuator_biasprm')
    for field in fields:
        np.testing.assert_array_equal(getattr(model, field), getattr(original, field), err_msg=field)
    assert model.opt.density == original.opt.density
    assert model.opt.viscosity == original.opt.viscosity
    assert 'robot0_right_hand' in report['native_inertia_bodies']
    selected = {model.geom(row['name']).id for row in report['geoms']}
    for index in range(model.ngeom):
        if index not in selected:
            np.testing.assert_array_equal(model.geom_fluid[index], original.geom_fluid[index])
