"""Known parallel-jaw tool geometry, separate from observed object geometry.

The Panda adapter reads only the installed, standalone robosuite tool asset.
It never accepts a scene model, an object pose, a contact table, or a recorded
demonstration as current robot feedback. All lengths and slide coordinates are
metres. A query must supply both native finger coordinates explicitly; those
coordinates may be measured feedback or a declared hypothetical configuration,
but this adapter cannot establish which they are.
"""

# BEGIN standalone robot-tool collision geometry and explicit finger FK.
from importlib.util import find_spec
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull


def _pose(position, rotation):
    value = np.eye(4)
    value[:3, :3] = np.asarray(rotation, dtype=float).reshape(3, 3)
    value[:3, 3] = np.asarray(position, dtype=float)
    return value


def _convex_planes(vertices):
    """Use the actual collision vertices, not a fitted axis-aligned box.

    MuJoCo uses convex collision geometry for mesh geoms. SciPy's hull therefore
    gives the corresponding solid halfspaces, ``n @ p + d <= 0``. Normalize the
    equations explicitly so their signed plane values have metric units. They
    are halfspace tests, not a claimed Euclidean mesh signed-distance field.
    """
    hull = ConvexHull(vertices)
    planes = np.asarray(hull.equations, dtype=float).copy()
    planes /= np.linalg.norm(planes[:, :3], axis=1)[:, None]
    return hull, planes


def _collision_geometry(model, geom_id, role):
    """Extract a compiled geom-local mesh and its native convex collision solid.

    MuJoCo's compiled ``mesh_vert`` already includes asset scaling and mesh
    recentering / principal-axis rotation. The matching compiled geom transform
    includes the compensating mesh pose. Applying ``mesh_pos``, ``mesh_quat`` or
    asset scale again would move the hand and fingers twice. Callers must pair
    these vertices with ``data.geom_xpos / geom_xmat`` from the same model.
    """
    import mujoco

    kind = model.geom_type[geom_id]
    if kind == mujoco.mjtGeom.mjGEOM_MESH:
        mesh_id = int(model.geom_dataid[geom_id])
        vertex_start = int(model.mesh_vertadr[mesh_id])
        vertex_count = int(model.mesh_vertnum[mesh_id])
        face_start = int(model.mesh_faceadr[mesh_id])
        face_count = int(model.mesh_facenum[mesh_id])
        vertices = np.asarray(model.mesh_vert[
            vertex_start:vertex_start + vertex_count], dtype=float).copy()
        faces = np.asarray(model.mesh_face[
            face_start:face_start + face_count], dtype=np.int64).copy()
        representation = "native_mesh_convex_collision_hull"
        _, planes = _convex_planes(vertices)
    elif kind == mujoco.mjtGeom.mjGEOM_BOX:
        vertices = np.array([[x, y, z]
                             for x in (-1., 1.)
                             for y in (-1., 1.)
                             for z in (-1., 1.)]) * model.geom_size[geom_id]
        hull, planes = _convex_planes(vertices)
        faces = np.asarray(hull.simplices, dtype=np.int64).copy()
        # ConvexHull supplies matching facet normals; orient the box triangles
        # outward as well, so visual consumers do not need to repair the mesh.
        cross = np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]],
                         vertices[faces[:, 2]] - vertices[faces[:, 0]])
        flip = np.einsum("ij,ij->i", cross, planes[:, :3]) < 0
        faces[flip] = faces[flip][:, [0, 2, 1]]
        representation = "native_box"
    else:
        raise ValueError("The tool adapter only admits explicit mesh or box collision geoms")
    if (faces.ndim != 2 or faces.shape[1] != 3
            or np.any(faces < 0) or np.any(faces >= len(vertices))):
        raise ValueError("Compiled mesh faces must index their own geom-local vertices")
    return {
        "geom_id": int(geom_id),
        "name": model.geom(geom_id).name,
        "role": role,
        "vertices": vertices,
        "faces": faces,
        "halfspaces": planes,
        "collision_representation": representation,
    }


class ParallelJawToolGeometry:
    """A standalone two-slide tool with explicitly identified opposing pads.

    Construct this through the robot-specific asset factory below. The object
    owns its FK scratch data; it does not read or mutate a simulator's live
    state. Static collision meshes stay geom-local. ``at(q)`` supplies their
    current transforms in the named tool frame without copying the meshes.
    """

    def __init__(self, model, *, tool_id, tool_frame, native_tool_site,
                 joint_names, collision_roles, pad_names,
                 open_joint_positions_m, closed_joint_positions_m, asset_path):
        import mujoco

        for label, value in (("tool_id", tool_id), ("tool_frame", tool_frame)):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{label} must be an explicit nonempty name")
        if len(joint_names) != 2 or len(pad_names) != 2 or model.nq != 2:
            raise ValueError("This adapter requires a standalone two-slide tool asset")
        self.tool_id = tool_id
        self.tool_frame = tool_frame
        self.asset_path = str(asset_path)
        self.joint_names = tuple(joint_names)
        self._model = model
        self._data = mujoco.MjData(model)
        self._site = model.site(native_tool_site).id
        self._joint_ids = np.array([model.joint(name).id for name in joint_names])
        if (np.any(model.jnt_type[self._joint_ids] != mujoco.mjtJoint.mjJNT_SLIDE)
                or not np.all(model.jnt_limited[self._joint_ids])):
            raise ValueError("Tool joints must be limited slides measured in metres")
        self._qpos_addresses = model.jnt_qposadr[self._joint_ids].copy()
        self.joint_ranges_m = model.jnt_range[self._joint_ids].copy()
        self.open_joint_positions_m = self._positions(open_joint_positions_m)
        self.closed_joint_positions_m = self._positions(closed_joint_positions_m)
        self.collision_geometries = [
            _collision_geometry(model, model.geom(name).id, role)
            for name, role in collision_roles]
        self._pad_ids = tuple(model.geom(name).id for name in pad_names)
        if any(model.geom_type[index] != mujoco.mjtGeom.mjGEOM_BOX
               for index in self._pad_ids):
            raise ValueError("Opposing contact pads must be explicit box faces")

        # Identify the inward face from the declared open configuration, then
        # keep that material face through closure. No current object geometry
        # and no assumption about a world-horizontal grasp enters this choice.
        tool_inverse = self._forward(self.open_joint_positions_m)
        pad_poses = [tool_inverse @ _pose(self._data.geom_xpos[index],
                                         self._data.geom_xmat[index])
                     for index in self._pad_ids]
        between = pad_poses[1][:3, 3] - pad_poses[0][:3, 3]
        if np.linalg.norm(between) <= 1e-12:
            raise ValueError("Open pad centres must be distinct")
        self.closing_axis_tool = between / np.linalg.norm(between)
        self.joint_axes_tool = np.array([
            tool_inverse[:3, :3] @ self._data.xaxis[index]
            for index in self._joint_ids])
        if not np.allclose(np.abs(self.joint_axes_tool @ self.closing_axis_tool),
                           1., atol=1e-9, rtol=0):
            raise ValueError("Both slide axes must be parallel to the opposing pad axis")
        self._pad_faces = []
        for side, pose in enumerate(pad_poses):
            inward = self.closing_axis_tool * (1. if side == 0 else -1.)
            projections = pose[:3, :3].T @ inward
            axis = int(np.argmax(np.abs(projections)))
            if not np.isclose(abs(projections[axis]), 1., atol=1e-9, rtol=0):
                raise ValueError("A pad face must be perpendicular to the closing axis")
            self._pad_faces.append((axis, float(np.sign(projections[axis]))))
        opened = self.at(self.open_joint_positions_m)
        closed = self.at(self.closed_joint_positions_m)
        if opened["signed_gap_m"] <= max(0., closed["signed_gap_m"]):
            raise ValueError("Declared open coordinates must create a larger positive pad gap")
        self.signed_gap_range_m = np.array([
            closed["signed_gap_m"], opened["signed_gap_m"]])

    def _positions(self, value):
        values = np.asarray(value)
        if (values.shape != (2,) or values.dtype.kind not in "iuf"
                or not np.isfinite(values).all()):
            raise ValueError("joint_positions_m must contain two finite native slide coordinates")
        values = values.astype(float, copy=True)
        if (np.any(values < self.joint_ranges_m[:, 0])
                or np.any(values > self.joint_ranges_m[:, 1])):
            raise ValueError("joint_positions_m lie outside the actual tool joint limits")
        return values

    def _forward(self, positions):
        import mujoco

        self._data.qpos[self._qpos_addresses] = positions
        # Kinematics only: no simulation step, actuator command, force/contact
        # prediction, or change to a live robot is involved.
        mujoco.mj_kinematics(self._model, self._data)
        return np.linalg.inv(_pose(self._data.site_xpos[self._site],
                                   self._data.site_xmat[self._site]))

    def at(self, joint_positions_m):
        """Return exact tool-local geometry for an explicit native joint tuple.

        ``T_tool_pad`` has +z pointing into the jaw opening and x/y tangent to
        its actual inner contact face. ``signed_gap_m`` is separation of those
        faces, not q1-q2: native soft-contact pads may overlap at a joint limit.
        A negative native gap is reported honestly, never silently clamped.
        No force-closure, collision-free approach, or stable grasp is certified.
        """
        positions = self._positions(joint_positions_m)
        tool_inverse = self._forward(positions)
        geoms = []
        by_id = {}
        for geometry in self.collision_geometries:
            index = geometry["geom_id"]
            pose = tool_inverse @ _pose(self._data.geom_xpos[index],
                                        self._data.geom_xmat[index])
            by_id[index] = pose
            geoms.append({"geom_id": index, "T_tool_geom": pose})
        pads = []
        for side, (index, face) in enumerate(zip(self._pad_ids, self._pad_faces)):
            axis, sign = face
            pose = by_id[index]
            normal = pose[:3, axis] * sign
            tangent_axes = [other for other in range(3) if other != axis]
            x_axis = pose[:3, tangent_axes[0]]
            y_axis = np.cross(normal, x_axis)
            center = (pose[:3, 3]
                      + normal * self._model.geom_size[index, axis])
            pads.append({
                "geom_id": index,
                "pad_index": side,
                "T_tool_pad": _pose(center, np.column_stack((x_axis, y_axis, normal))),
                "half_sizes_xy_m": self._model.geom_size[index, tangent_axes].copy(),
            })
        gap = float((pads[1]["T_tool_pad"][:3, 3]
                     - pads[0]["T_tool_pad"][:3, 3])
                    @ pads[0]["T_tool_pad"][:3, 2])
        return {
            "tool_id": self.tool_id,
            "tool_frame": self.tool_frame,
            "joint_positions_m": positions,
            "geoms": geoms,
            "pads": pads,
            "signed_gap_m": gap,
        }


def load_robosuite_panda_tool(*, tool_id, tool_frame):
    """Load the installed Panda hand asset without importing a robot scene.

    Robot-specific names belong in this adapter, not in grasp filtering. The
    external tool name is bound to the asset's native ``grip_site``; neither is
    interpreted as a finger-pad centre. No command-to-q or demo-to-current-state
    substitution is made. The Panda's second native slide has negative q when
    open, unlike robot descriptions that use two positive finger coordinates.
    """
    import mujoco

    spec = find_spec("robosuite")
    if spec is None or not spec.submodule_search_locations:
        raise ImportError("The installed robosuite Panda tool asset is required")
    asset = (Path(next(iter(spec.submodule_search_locations)))
             / "models/assets/grippers/panda_gripper.xml")
    model = mujoco.MjModel.from_xml_path(str(asset))
    return ParallelJawToolGeometry(
        model, tool_id=tool_id, tool_frame=tool_frame, native_tool_site="grip_site",
        joint_names=("finger_joint1", "finger_joint2"),
        collision_roles=(("hand_collision", "palm"),
                         ("finger1_collision", "finger"),
                         ("finger2_collision", "finger"),
                         ("finger1_pad_collision", "pad"),
                         ("finger2_pad_collision", "pad")),
        pad_names=("finger1_pad_collision", "finger2_pad_collision"),
        open_joint_positions_m=(.04, -.04), closed_joint_positions_m=(0., 0.),
        asset_path=asset)
# END standalone robot-tool collision geometry and explicit finger FK.
