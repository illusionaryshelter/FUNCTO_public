"""Bind fluid geometry to the declared fixed robot's existing physical shapes.

MuJoCo otherwise infers a body's fluid shape from its mass and inertia. Those
quantities specify rigid-body dynamics; an explicitly supplied inertia need
not describe the outer dimensions of that body's registered geometry. The
native ellipsoid API keeps the inertial parameters and derives the fluid
approximation from each participating geom instead.
"""

from pathlib import Path

import mujoco


def build_simulation_model(scene_path, *, robot_root):
    """Compile the same scene with explicit fluid shapes for its fixed robot.

    ``robot_root`` names the root body of the already declared official robot
    configuration, including its fixed end effector. Its geometry is reusable
    across tasks because this robot and configuration remain fixed. This
    function runs only while constructing the physical simulator, before any
    state restoration or command. It receives no trajectory, target state,
    contact result or evaluation outcome.

    Collision participation is MuJoCo's existing OR of contype/conaffinity.
    Visual-only copies do not contribute a second fluid shape. The existing
    fluid coefficients, masses, inertias, contact geometry and medium remain
    unchanged. Bodies without a participating geom retain their native fluid
    definition; the returned record makes that scope explicit.
    """
    spec = mujoco.MjSpec.from_file(str(Path(scene_path).resolve()))
    root = spec.body(robot_root)
    root_name = root.name
    robot_bodies = set()
    pending = [root]
    while pending:
        body = pending.pop()
        robot_bodies.add(body.name)
        child = body.first_body()
        while child is not None:
            pending.append(child)
            child = body.next_body(child)

    # BEGIN the official robot's existing physical geometry owns fluid shape.
    selected = []
    for geom in spec.geoms:
        if geom.parent.name in robot_bodies and (geom.contype != 0 or geom.conaffinity != 0):
            geom.fluid_ellipsoid = 1.
            selected.append(dict(name=geom.name, body=geom.parent.name,
                contype=int(geom.contype), conaffinity=int(geom.conaffinity),
                fluid_coefficients=list(geom.fluid_coefs)))
    # END no shape, inertia or task-specific selection is introduced.
    model = spec.compile()
    geometrically_defined = {row['body'] for row in selected}
    native_inertia_bodies = [body.name for body in spec.bodies
        if body.name in robot_bodies and body.name not in geometrically_defined
        and model.body_mass[model.body(body.name).id] > 0.]
    report = dict(source_scene=str(Path(scene_path).resolve()), robot_root=root_name,
        api='MuJoCo MjSpec / geom.fluid_ellipsoid = 1',
        geometry_source='Existing physical geoms of the declared fixed official robot and end effector',
        selection='Robot subtree and (contype != 0 or conaffinity != 0)',
        fluid_model='Native ellipsoid approximation with existing per-geom coefficients',
        geoms=selected, native_inertia_bodies=native_inertia_bodies,
        medium_density=float(model.opt.density), medium_viscosity=float(model.opt.viscosity),
        scope='Physical simulation model construction; no evaluation-to-control data path')
    return model, report
