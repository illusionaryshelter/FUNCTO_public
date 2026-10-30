"""Known fixed-robot gravity from named encoder positions.

The supplied model contains the official robot and its fixed end-effector
configuration. Their shape, inertia and fixed component relationships are
reusable across tasks; calibration places the robot base in the gravity frame.
Task/environment geometry and current contact forces are not inputs.
"""
from __future__ import annotations

import mujoco
import numpy as np


class RobotGravity:
    """Own scratch dynamics data for an explicitly supplied robot-only model.

    ``joint_names`` defines encoder input/output order and ``source`` identifies
    the calibrated fixed model. Revolute coordinates use rad and output N m;
    slide coordinates use m and N.
    """
    def __init__(self, model, joint_names, *, source):
        self.model, self.source = model, str(source)
        self.joint_names = tuple(joint_names)
        self.data = mujoco.MjData(model)
        joints = np.asarray([model.joint(name).id for name in self.joint_names])
        self.qpos = model.jnt_qposadr[joints]
        self.dofs = model.jnt_dofadr[joints]
        if (model.nq != len(joints) or model.nv != len(joints)
                or len(set(self.qpos)) != model.nq):
            raise ValueError("Gravity input must cover the complete named scalar robot state")
        self._force = np.empty(model.nv)

    def compute(self, q):
        """Evaluate g(q) with kinematics and recursive Newton–Euler dynamics.

        Zero velocity and acceleration belong only to this scratch calculation.
        No physical step, scene constraint solve or feedback mutation occurs.
        The caller supplies measured robot q at its current control time.
        """
        self.data.qpos[self.qpos] = np.asarray(q, dtype=float)
        self.data.qvel[:] = 0.
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        mujoco.mj_comVel(self.model, self.data)
        mujoco.mj_rne(self.model, self.data, 0, self._force)
        return self._force[self.dofs].copy()
