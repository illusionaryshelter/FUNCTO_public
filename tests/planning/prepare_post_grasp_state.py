"""Build a measured-hand robot model from a verified live grasp handoff.

Run after closure, fresh robot/RGB-D observation and holding verification.
The independent physics evaluator's object truth is not an input provider for
this command. Missing verification is an error; no next skill is executed.
"""

# BEGIN explicit post-grasp handoff from actual feedback and current evidence.
import argparse
import json
from pathlib import Path
import sys

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from manipulation.gripper_state import build_post_grasp_robot_config
from prepare_recorded_task import read_json


def run(args):
    """Validate the current binding before writing a new standalone hand model."""
    from robot_fixture import validate_fixture_collision_contract

    fixture = read_json(args.robot_fixture)
    feedback, scene = read_json(args.feedback), read_json(args.scene)
    robot_path = Path(fixture["robot_config_path"])
    if robot_path.stat().st_size > 16 * 1024**2:
        raise ValueError("Robot configuration exceeds the bounded metadata input size")
    robot = yaml.safe_load(robot_path.read_text())
    validate_fixture_collision_contract(fixture, robot)
    binding = build_post_grasp_robot_config(robot, feedback, scene)
    # The geometry certificate is independent of these measured finger locks.
    # Recheck the exact config that will be supplied to native kinematics.
    validate_fixture_collision_contract(fixture, binding["robot_config"])
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "robot.yml").write_text(yaml.safe_dump(binding["robot_config"], sort_keys=False))
    for name, value in (("planning-feedback", binding["planning_feedback"]),
                        ("measured-feedback", binding["full_feedback"]),
                        ("hand-binding", binding["hand_binding"]),
                        ("current-scene", scene)):
        (output / f"{name}.json").write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    report = {
        "status": "prepared", "robot_config_path": str(output / "robot.yml"),
        "feedback_id": feedback["feedback_id"], "timestamp_s": feedback["timestamp_s"],
        "complete_joint_names": feedback["joint_names"],
        "active_joint_names": binding["planning_feedback"]["joint_names"],
        "source_fixture": str(args.robot_fixture.resolve()),
        "source_feedback": str(args.feedback.resolve()), "source_scene": str(args.scene.resolve()),
        "planning_called": False, "execution_tested": False,
        "scope": "Current measured hand robot model only; no new holding claim, payload reconstruction or barrier release",
    }
    (output / "preparation-report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, allow_nan=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("robot-fixture", "feedback", "scene", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    run(parser.parse_args())
# END explicit post-grasp handoff from actual feedback and current evidence.
