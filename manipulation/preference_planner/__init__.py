"""Planning with explicitly declared required and preferred pose events.

Bind complete event declarations with bind_pose_objectives before calling
plan_trajectory, or reuse native resources through PreferenceMotionBackend.
The original manipulation.planning entry remains independently available.
"""

from .backend import PreferenceMotionBackend, PreferencePlanner
from .lifecycle import plan_trajectory
from .pose_roles import bind_pose_objectives, prior_goal_ids

__all__ = ["PreferenceMotionBackend", "PreferencePlanner", "plan_trajectory",
           "bind_pose_objectives", "prior_goal_ids"]
