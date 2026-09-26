"""Run numerical object-manipulation examples without models or a robot.

These are analytic geometry checks, not demonstrations of semantic perception.
Every output uses the production stage/sequence and visualization code.
"""

import argparse
from copy import deepcopy
import itertools
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from manipulation.pipeline import run_prepared, write_json
from manipulation.visualization import render_result


def pose(position):
    value = np.eye(4)
    value[:3, 3] = position
    return value


def box(half):
    return np.asarray(list(itertools.product(*[(-size, size) for size in half])))


def record(poses, kind, relation="transfer", interaction="none"):
    return {"task": {"motion": kind, "relation": relation, "interaction": interaction},
            "carrier_id": "object", "reference_id": "reference", "reference_frame": "world",
            "body_frame_id": "object/material", "demo_poses": poses,
            "scene": {"initial_pose": poses[0], "reference_transform": np.eye(4),
                      "body_points": box([.035, .025, .02]), "goal_pose": poses[-1],
                      "geometry_coverage": "complete analytic cuboid vertices"},
            "evidence": {"source": "analytic geometry; no model/physical validation"}}


def cases():
    """Return varied supported cases; task names never enter production routing."""
    output, t = {}, np.linspace(0., 1., 30)
    free = np.array([pose([.15*np.sin(np.pi*x), .2*x, .3-.2*x]) for x in t])
    output["free_transport"] = record(free, "free")
    slide = np.array([pose([.25*(1-x), .1, .1]) for x in t])
    output["push_without_grasp"] = record(slide, "prismatic", "articulation", "push")
    axis = np.array([1., -2., 3.]); axis /= np.linalg.norm(axis)
    center, offset = np.array([.1, .2, .3]), np.array([.2, .04, -.03])
    hinge = np.repeat(np.eye(4)[None], len(t), axis=0)
    hinge[:, :3, :3] = Rotation.from_rotvec((1.7*t)[:, None]*axis).as_matrix()
    hinge[:, :3, 3] = center + np.einsum("nij,j->ni", hinge[:, :3, :3], offset)
    output["arbitrary_hinge"] = record(hinge, "revolute", "articulation", "push")
    tilt = Rotation.from_euler("xyz", [.3, -.4, .2]).as_matrix()
    planar = np.repeat(np.eye(4)[None], len(t), axis=0)
    planar[:, :3, 3] = np.array([.1, -.2, .3]) + np.column_stack((.2*t, .08*np.sin(np.pi*t), t*0)) @ tilt.T
    planar[:, :3, :3] = Rotation.from_rotvec(t[:, None]*tilt[:, 2]).as_matrix() @ tilt
    output["tilted_planar"] = record(planar, "planar", interaction="push")
    planar_fixed = planar.copy(); planar_fixed[:, :3, :3] = tilt
    output["tilted_fixed_orientation"] = record(planar_fixed, "translation_plane", interaction="push")
    surface = {"origin": [0., 0., 0.], "basis": [[1., 0.], [0., 1.], [0., 0.]],
               "normal": [0., 0., 1.], "boundary": [[-.2, -.2], [.2, -.2], [.2, .3], [-.2, .3]],
               "source": "known finite support"}
    placement = record(free, "free", "on", "grasp")
    placement["scene"]["support"] = surface
    placement["scene"]["preferred_goal_pose"] = placement["scene"].pop("goal_pose")
    output["place_on"] = placement
    containment = record(free, "free", "inside", "grasp")
    containment["scene"]["interior"] = {"vertices": box([.1, .1, .1])+[.2, .2, .1],
                                         "source": "known convex free cavity"}
    containment["scene"]["preferred_goal_pose"] = containment["scene"].pop("goal_pose")
    output["place_inside"] = containment
    opening = record(np.array([pose([.3*x, 0, 0]) for x in t]), "prismatic", "articulation", "pull")
    opening.update(carrier_id="drawer", reference_id="cabinet", body_frame_id="drawer/material")
    middle = deepcopy(placement)
    middle.update(reference_id="drawer", reference_body_frame_id="drawer/material")
    middle["scene"]["reference_pose"] = np.eye(4)
    middle["scene"]["support"] = deepcopy(surface)
    closing = deepcopy(opening)
    closing["task"]["interaction"] = "push"
    closing["demo_poses"] = opening["demo_poses"][::-1].copy()
    closing["scene"].update(initial_pose=closing["demo_poses"][0], goal_pose=closing["demo_poses"][-1])
    output["open_place_close"] = {"stages": [opening, middle, closing]}
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports = {}
    for name, observation in cases().items():
        source = args.output_dir / f"{name}.json"
        if source.exists():
            raise FileExistsError(source)
        write_json(source, observation)
        destination = args.output_dir / name
        result = run_prepared(source, destination)
        reports[name] = {"status": result["status"], **render_result(result, destination)}
        print(name, result["status"], flush=True)
    write_json(args.output_dir / "summary.json", reports)


if __name__ == "__main__":
    main()
