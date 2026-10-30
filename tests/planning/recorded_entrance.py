"""Convert a saved robot reference entrance for the shared execution API.

The original executor compares these named values with current robot feedback;
reference identity and live feedback identity remain separate records.
"""
from copy import deepcopy


def reference_entrance(planned_feedback):
    """Express the existing saved robot sample in the executor's input schema."""
    result = {key: deepcopy(planned_feedback[key]) for key in (
        'joint_names', 'arm_joint_names', 'finger_joint_names', 'base_pose',
        'tool_link', 'tool_transform', 'reference_frame', 'feedback_id', 'timestamp_s', 'source')}
    result.update(q=deepcopy(planned_feedback['start_q']), dq=deepcopy(planned_feedback['start_dq']),
        tool_pose=deepcopy(planned_feedback['eef_pose']), sim_time=planned_feedback['sim_time_s'])
    return result
