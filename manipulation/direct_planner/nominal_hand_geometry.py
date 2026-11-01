"""Explicit position-control nominal geometry on the existing finger clock.

The factory selects one declared model: retain the existing held shape until
its original open/release boundary, then use bounded free-motion Hermite
responses to the actual command targets until the next declared contact.
Inputs remain the original command, conditions, occurrence clock and measured
entrance. Output is the maintained NominalFingerGeometry, not a device command
or a future measurement. The fixed official hand geometry continues to use
these named coordinates through the existing robot kinematics interface.
"""

# BEGIN local nominal-model assembly; original commands and bindings stay owned.
from dataclasses import replace

import numpy as np

from manipulation.gripper_trajectory import (
    NominalFingerGeometry, _incoming_values, _segment, bind_nominal_finger_geometry,
)
from manipulation.planning_problem import event_occurrence_key


def bind_position_controlled_finger_geometry(reference, conditions, event_clock, *,
                                            progress_interval, source):
    """Bind the declared held-to-free model using existing occurrence identities.

    Selecting this factory is the explicit position-control model declaration.
    ``source`` names that declaration. The maintained binder supplies the
    initial model and its first invalidating occurrence. An initially free
    model is retained directly, including its measured q/dq and contact cutoff.

    For a held model, the endpoint of its valid prefix supplies predicted q/dq
    for the free response. It is neither the actuator command q/dq nor a new
    encoder sample. If release precedes open, the current active command target
    remains in force until open; the same bounded interpolation handles it.
    Future contact activation ends validity without moving that event or any
    execution boundary. The complete stored polynomial holds its final shape
    after validity, where collision consumers already retain unknown status.
    """
    if not isinstance(source, str) or not source.strip():
        raise ValueError("Position-controlled nominal geometry needs its model declaration source")
    prefix = bind_nominal_finger_geometry(reference, conditions, event_clock,
                                          progress_interval=progress_interval)
    declaration = (source + "; model=held-shape then position-controlled bounded free response; "
        "event anchors are predicted model states, not future measurements; initial=" + prefix.source)
    held = any(row.get("kind") == "declared_attachment" for row in conditions)
    if not held or prefix.valid_until_s == reference.duration:
        return replace(prefix, source=declaration)

    start = float(prefix.valid_until_s)
    begin_progress, end_progress = progress_interval
    # FingerTrajectory validates source events; consume their actual native
    # applications in stable physical order, including continuation clocks.
    active = sorted((event for event in reference.events if not event["superseded"]),
                    key=lambda event: event["applied_time_s"])
    contacts = []
    for row in conditions:
        if row.get("kind") != "grasp_contact_activation" or row["interval"][0] > end_progress:
            continue
        identity = event_occurrence_key(row["activation_event"])
        if identity not in event_clock:
            if row["interval"][0] <= begin_progress:
                activation_time = None  # Original occurrence was consumed before this block.
            else:
                raise ValueError("Contact activation inside this motion block requires its original bound event occurrence")
        else:
            activation_time = float(event_clock[identity]["activation_time_s"])
        # The compiled interval ends at its original next open, or at the
        # skill endpoint if no open exists. Resolve that actual application;
        # equal progress on a plateau does not imply an event already happened.
        endings = [float(event["applied_time_s"]) for event in active
                   if event["kind"] == "open" and event["progress"] == row["interval"][1]
                   and (activation_time is None or event["applied_time_s"] >= activation_time)]
        if endings and min(endings) <= start:
            continue
        if not endings and row["interval"][1] < begin_progress:
            continue
        # A past activation whose interval still spans this physical boundary
        # is not cleared by an unrelated held-condition or matching carrier.
        contacts.append((start if activation_time is None else max(start, activation_time), identity))
    valid_until = min((time_s for time_s, _ in contacts), default=reference.duration)
    cutoff_identities = [identity for time_s, identity in contacts if time_s == valid_until]
    q, dq = (value[0].copy() for value in prefix.sample(np.array([start])))
    anchor = dict(time_s=start, position=q.tolist(), velocity=dq.tolist(),
                  source="predicted endpoint of the declared held-shape prefix")
    declaration += "; free_anchor=" + repr(anchor) + "; next_contact=" + repr(cutoff_identities)
    if valid_until == start:
        return replace(prefix, source=declaration)

    # Read the named command target, never the command polynomial's q/dq as
    # a measured or predicted shape. Earlier command events replace that
    # target; later events supply the original application times unchanged.
    goal = _incoming_values(reference.incoming_command, reference.joint_names)[2]
    future = []
    for event in active:
        moment = float(event["applied_time_s"])
        if moment <= start:
            goal = np.asarray(event["finger_positions"], dtype=float)
        elif moment < valid_until:
            future.append(event)

    prefix_times = prefix.times[prefix.times < start]
    times = [float(value) for value in np.r_[prefix_times, start]]
    positions, velocities = (list(value.copy()) for value in prefix.sample(np.asarray(times)))

    def append(moment, position, velocity):
        if moment > times[-1]:
            times.append(float(moment))
            positions.append(position.copy())
            velocities.append(velocity.copy())

    windows = []
    cursor = start
    for event in future:
        moment = float(event["applied_time_s"])
        windows.append((cursor, moment, goal))
        cursor, goal = moment, np.asarray(event["finger_positions"], dtype=float)
    windows.append((cursor, valid_until, goal))
    for begin, end, target in windows:
        finish = _segment(q, dq, target, begin, end, reference.timestep,
                          reference.joint_names, reference._limits())
        zero = np.zeros_like(dq)
        append(finish, target, zero)
        append(end, target, zero)
        q, dq = target.copy(), zero
    # NominalFingerGeometry owns a complete polynomial but qualifies it only
    # through valid_until. The invalid suffix introduces no movement claim.
    append(reference.duration, q, np.zeros_like(dq))
    return NominalFingerGeometry(reference.joint_names, np.asarray(times), np.asarray(positions),
        np.asarray(velocities), reference.duration, valid_until, declaration)
# END source events, command polynomial, measured feedback and relation data are unchanged.
