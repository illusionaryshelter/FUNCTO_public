"""Owned original robot FK shared by one saved-candidate verification.

The caller supplies the same registered official robot model and declares
all frame consumers. Complete model-order q, velocity and calibration values
identify reusable results. Native homogeneous products and Pin quaternion
projections retain their original, distinct floating-point operations.
"""

import numpy as np
import pinocchio as pin


class FKSession:
    """Own FK values while explicit acceptance calls share them.

    Public methods do not install callbacks or modify classes. A caller passes
    this owner to PinKinematics and RobotGeometryChecker; differentiable
    calculations continue through the existing analytic derivative API.
    """
    def __init__(self, model, *, frames):
        self.model = model
        self.frames = tuple(dict.fromkeys(frames))
        if any(not model.existFrame(frame) for frame in self.frames):
            raise ValueError('FKSession frames must belong to its declared robot model')
        self._frame_ids = tuple(model.getFrameId(frame) for frame in self.frames)
        self._data = model.createData()
        self._states = {}
        self._counts = dict(hits=0, computed=0, snapshots=0, frames=0,
                            velocity_results=0, projected_frames=0, numeric_bytes=0,
                            peak_numeric_bytes=0)

    def __enter__(self):
        return self

    def __exit__(self, *exception):
        self._states.clear()
        self._data = None
        self._counts['numeric_bytes'] = 0

    def placements(self, model, q):
        """Return owned model-base SE3 values for the declared frame IDs."""
        state, _ = self._state(model, q)
        return state[0]

    def describe(self):
        """Return computation counts and owned numeric storage, not timing."""
        return {**self._counts,
                'scope': 'one candidate check; complete model-order q; original Pinocchio FK',
                'storage_scope': 'owned SE3 numeric values, complete input keys, projected poses and LOCAL twist arrays; Python containers and one Data are separate'}

    def _charge(self, count):
        self._counts['numeric_bytes'] += count
        self._counts['peak_numeric_bytes'] = max(
            self._counts['peak_numeric_bytes'], self._counts['numeric_bytes'])

    def _state(self, model, q, dq=None, frame_ids=()):
        """Return owned original placements and requested LOCAL velocities.

        The q-only and q/dq Pinocchio overloads compute identical placements
        for the same complete q.  A missing velocity request is computed by
        the maintained q/dq overload, with its own full input and frame key.
        No velocity, Jacobian or model relationship is reconstructed here.
        """
        if model is not self.model:
            raise ValueError('FKSession consumers must share its declared robot model instance')
        data, states = self._data, self._states
        q_key = q.tobytes()
        state = states.get(q_key)
        velocity_key = None if dq is None else (dq.tobytes(), tuple(frame_ids))
        if state is not None and (dq is None or velocity_key in state[1]):
            self._counts['hits'] += 1
            return state, None if dq is None else state[1][velocity_key]

        if dq is None:
            pin.forwardKinematics(model, data, q)
        else:
            pin.forwardKinematics(model, data, q, dq)
        pin.updateFramePlacements(model, data)
        self._counts['computed'] += 1
        if state is None:
            # Copy native SE3 values instead of converting through quaternion
            # or changing the source matrix layout used by later products.
            placements = {frame: pin.SE3(data.oMf[frame]) for frame in self._frame_ids}
            state = placements, {}, {}
            states[q_key] = state
            self._counts['snapshots'] += 1
            self._counts['frames'] += len(placements)
            self._charge(len(q_key) + len(placements) * 12 * 8)
        if dq is None:
            return state, None
        twists = np.empty((len(frame_ids), 6))
        for index, frame in enumerate(frame_ids):
            motion = pin.getFrameVelocity(model, data, frame, pin.LOCAL)
            twists[index, :3] = motion.linear
            twists[index, 3:] = motion.angular
        twists.flags.writeable = False
        state[1][velocity_key] = twists
        self._counts['velocity_results'] += 1
        self._charge(len(velocity_key[0]) + twists.nbytes)
        return state, twists

    def evaluate(self, bridge, position, velocity=None):
        leading = position.shape[:-1]
        count = int(np.prod(leading))
        joints, frames = len(bridge.joint_names), len(bridge.frames)
        values = position.reshape(count, joints)
        rates = None if velocity is None else velocity.reshape(count, joints)
        poses = np.empty((count, frames, 7))
        twists = None if rates is None else np.empty((count, frames, 6))
        q, dq = np.empty(bridge.model.nq), np.empty(bridge.model.nv)
        calibration = bridge._rotation.tobytes(), bridge._translation.tobytes()
        for sample in range(count):
            q[bridge._q_columns] = values[sample]
            if rates is not None:
                dq[bridge._v_columns] = rates[sample]
            state, local = self._state(bridge.model, q,
                None if rates is None else dq, bridge._frame_ids)
            placements, _, projections = state
            if calibration not in projections:
                projections[calibration] = {}
                self._charge(sum(map(len, calibration)))
            projected = projections[calibration]
            for index, frame in enumerate(bridge._frame_ids):
                if frame not in projected:
                    placement = placements[frame]
                    rotation = bridge._rotation @ placement.rotation
                    pose = np.empty(7)
                    pose[:3] = bridge._rotation @ placement.translation + bridge._translation
                    pose[3:] = pin.Quaternion(rotation).coeffs()[[3, 0, 1, 2]]
                    pose.flags.writeable = False
                    projected[frame] = pose
                    self._counts['projected_frames'] += 1
                    self._charge(pose.nbytes)
                poses[sample, index] = projected[frame]
            if twists is not None:
                twists[sample] = local
        result = poses.reshape(*leading, frames, 7), None
        if twists is not None:
            return *result, twists.reshape(*leading, frames, 6), None, None
        return result
