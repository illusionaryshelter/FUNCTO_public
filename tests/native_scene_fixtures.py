"""Explicit fixed native solids for small scene-policy tests."""
# BEGIN tests bind real native geometry instead of an implicit sphere fallback.
import numpy as np


def bind_fixed_robot(scene, *, centers=None, radii=None, self_pairs=()):
    """Bind independent exact spheres to a zero-DOF URDF for policy tests.

    ``self_pairs`` names sphere indices only in this fixture; production
    receives the robot validator's named complete-solid policy. Existing
    one-sphere body groups keep their material identity for Holding tests.
    """
    count = len(scene.robot_sphere_materials)
    centers = np.zeros((count, 3)) if centers is None else np.asarray(centers, float)
    radii = np.full(count, .01) if radii is None else np.asarray(radii, float)
    if not scene.robot_collision_bodies:
        scene.robot_collision_bodies = tuple({
            'body_id': f'body_{index}', 'frame': 'base', 'geometry_names': [f'native_{index}'],
            'sphere_indices': [index], 'local_spheres': [[*centers[index], radii[index]]],
            'source': 'explicit fixed native sphere fixture'} for index in range(count))
    names = {}
    for body in scene.robot_collision_bodies:
        index, = body['sphere_indices']
        name, = body['geometry_names']
        names[index] = name
    geometry = []
    for index in range(count):
        pose = np.eye(4)
        pose[:3, 3] = centers[index]
        geometry.append({'name': names[index], 'frame': 'base', 'kind': 'sphere',
                         'dimensions': [radii[index]], 'T_frame_geometry': pose.tolist(),
                         'source': 'explicit test native sphere'})
    scene.bind_online_robot(
        robot_sphere_geometry=[{'link_name': 'base', 'sphere_indices': list(range(count)),
                                'centers_local': centers.tolist(), 'radii': radii.tolist()}],
        native_robot_model={'urdf_xml': '<robot name="fixed"><link name="base"/></robot>',
                            'geometries': geometry}, native_joint_names=(),
        native_self_pairs=[{'first': names[first], 'second': names[second], 'margin': 0.}
                           for first, second in self_pairs],
        robot_model={'T_reference_base': np.eye(4).tolist()})
    return scene
# END fixture geometry is explicit and carries the same current scene contract.
