"""Direct joint-trajectory optimization with explicit reference preferences.

The caller supplies a PlanningProblem, OnlinePlanningScene, official robot
configuration, measured entrance and declared numerical preferences. The
backend owns request resources; plan_trajectory owns the cumulative request
clock and delivers a qualified curve with the existing execution binding.
"""

from .direct_preference import DirectPreferenceBackend, DirectPreferencePlanner
from .direct_lifecycle import plan_trajectory
from .reference_objective import ReferenceObjective
from manipulation.preference_planner.pose_roles import bind_pose_objectives

__all__ = ["DirectPreferenceBackend", "DirectPreferencePlanner", "ReferenceObjective",
           "plan_trajectory", "bind_pose_objectives"]
