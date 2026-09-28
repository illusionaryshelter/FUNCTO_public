"""Ground a demonstrated object stage using FUNCTO's existing local models.

SAM/Gemini choose actual entities; pair descriptors bind visible reference and
carrier features. Metric geometry comes from calibrated depth and rigid fits.
Reference binding is an estimate from visible correspondences, not knowledge
of hidden joints or complete container interiors.
"""

from pathlib import Path
import json

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from .grounding import infer_task, match_points, select_roles
from .motion import transfer_motion
from .observations import backproject, load_rgbd, rigid_registration, track_rgbd
from .pipeline import motion_kind, run_stage, write_json
from .support_grounding import infer_support


def _anchors(mask, depth, count=24):
    """Choose spatially distributed, valid-depth observations inside one ROI."""
    import cv2
    valid = np.asarray(mask, bool) & np.isfinite(depth) & (depth > 0)
    # Depth at object boundaries often belongs to the background. Erosion is a
    # measurement choice, not a synthetic keypoint or a guessed depth value.
    interior = cv2.erode(valid.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    if interior.sum() >= count:
        valid = interior
    y, x = np.nonzero(valid)
    if len(x) < 6:
        raise ValueError("Insufficient valid depth in the selected entity")
    candidates = np.column_stack((x, y)).astype(float)
    if len(candidates) > 6000:
        candidates = candidates[np.linspace(0, len(candidates)-1, 6000).astype(int)]
    selected = [int(np.argmin(np.sum((candidates-candidates.mean(0))**2, axis=1)))]
    distances = np.full(len(candidates), np.inf)
    while len(selected) < min(count, len(candidates)):
        distances = np.minimum(distances, np.sum((candidates-candidates[selected[-1]])**2, axis=1))
        index = int(np.argmax(distances))
        if distances[index] <= 0:
            break
        selected.append(index)
    return candidates[selected], valid


def _binding(demo_image, scene_image, demo_depth, scene_depth, demo_camera, scene_camera,
             demo_mask, scene_mask, extractor):
    pixels, valid_demo = _anchors(demo_mask, demo_depth)
    valid_scene = np.asarray(scene_mask, bool) & np.isfinite(scene_depth) & (scene_depth > 0)
    def crop(image, mask):
        yy, xx = np.nonzero(mask)
        if not len(xx):
            raise ValueError("Selected entity has no valid depth")
        padding = max(2, int(max(xx.max()-xx.min()+1, yy.max()-yy.min()+1)*.05))
        x0, y0 = max(0, int(xx.min())-padding), max(0, int(yy.min())-padding)
        x1, y1 = min(image.width, int(xx.max())+padding+1), min(image.height, int(yy.max())+padding+1)
        return image.crop((x0, y0, x1, y1)), mask[y0:y1, x0:x1], np.array([x0, y0])
    first_image, first_mask, first_offset = crop(demo_image, valid_demo)
    second_image, second_mask, second_offset = crop(scene_image, valid_scene)
    targets, matching = match_points(first_image, second_image, first_mask, second_mask,
                                     pixels-first_offset, extractor=extractor)
    targets += second_offset
    matching["crop_offsets_xy"] = [first_offset.tolist(), second_offset.tolist()]
    scores = np.asarray(matching["scores"])
    selected, seen = [], set()
    for index in np.argsort(-scores):
        pixel_id = tuple(np.rint(targets[index]).astype(int))
        # Adjacent dense pixels can have almost equal scores on the same valid
        # correspondence. Keep that localization diagnostic; metric consensus
        # and distinct target pixels decide whether a rigid binding is usable.
        if scores[index] <= 0 or pixel_id in seen:
            continue
        seen.add(pixel_id)
        selected.append(int(index))
    if len(selected) < 6:
        raise ValueError("Descriptor binding is ambiguous or has too few distinct correspondences")
    first = backproject(demo_depth, demo_camera.K, pixels[selected], mask=valid_demo)
    second = backproject(scene_depth, scene_camera.K, targets[selected], mask=valid_scene)
    transform, geometry = rigid_registration(first, second, max_error=0.01)
    return transform, {"source_pixels": pixels[selected].tolist(),
                       "target_pixels": targets[selected].tolist(),
                       "matching": matching, "rigid_fit": geometry,
                       "source": "visible semantic correspondences + calibrated RGB-D"}


def _body_points(depth, camera, mask, initial_pose):
    cloud = backproject(depth, camera.K, mask=mask)
    if len(cloud) > 3000:
        cloud = cloud[np.linspace(0, len(cloud)-1, 3000).astype(int)]
    if len(cloud) < 6:
        raise ValueError("Selected carrier has insufficient observed geometry")
    return (cloud-initial_pose[:3, 3]) @ initial_pose[:3, :3]


def ground_rgbd(demo_dir, scene_dir, instruction, *, client=None, detector=None,
                extractor=None, tracker=None, output_dir=None):
    """Produce a numerical stage from RGB-D and task semantics, without new labels.

    Optional ``scene_dir/geometry.json`` may supply known interior/support
    geometry from another system component. It is never synthesized from an
    outer bounding box. Raw depth and intrinsics are needed for metric output;
    no additional task-specific training or offline motion dataset is used.
    """
    demo_dir, scene_dir = Path(demo_dir), Path(scene_dir)
    demo_image, demo_depth, demo_camera = load_rgbd(demo_dir)
    scene_image, scene_depth, scene_camera = load_rgbd(scene_dir)
    indices = sorted({int(path.stem) for path in (demo_dir / "rgb").iterdir()
                      if path.stem.isdigit() and path.suffix.lower() in {".jpg", ".png", ".jpeg"}})
    if len(indices) < 2:
        raise ValueError("The demonstrated stage requires at least two RGB-D frames")
    final_image, _, _ = load_rgbd(demo_dir, indices[-1])
    task = infer_task(instruction, demo_image, scene_image,
                      demo_final_image=final_image, client=client)
    # BEGIN one observation path for tools, transported objects and moving parts.
    # A tool's semantic annotations specify material points, not its subsequent
    # motion. Track its carrier/reference exactly as for the other task families.
    if task["relation"] == "tool_use" and task["motion"] not in {"auto", "free"}:
        raise ValueError("The tool-use optimizer cannot jointly enforce function events and "
                         "a constrained motion family; refusing to discard the motion constraint")
    demo_roles = select_roles(demo_image, task, "demo", detector=detector, client=client)
    masks = {}

    def role_masks(domain, roles):
        carrier = roles["carrier"]["mask"]
        # A parent cabinet mask may contain the moving drawer/door. Only its
        # remaining visible fixed part is allowed to serve as motion reference.
        reference = roles["reference"]["mask"] & ~carrier
        if reference.sum() < 12:
            raise ValueError(f"{domain} has insufficient visible fixed reference outside the moving part")
        masks[domain] = {"carrier": carrier, "reference": reference}
        if output_dir is not None:
            destination = Path(output_dir) / domain
            destination.mkdir(parents=True, exist_ok=True)
            (demo_image if domain == "demo" else scene_image).save(destination / "initial.png")
            for role, mask in masks[domain].items():
                Image.fromarray(mask.astype(np.uint8)*255).save(destination / f"{role}_mask.png")
    tracker = track_rgbd if tracker is None else tracker
    role_masks("demo", demo_roles)
    track = tracker(demo_dir, masks["demo"]["carrier"], reference_mask=masks["demo"]["reference"])
    demo_poses = np.asarray(track["poses"], float)
    if len(demo_poses) != len(indices) or not np.array_equal(track["frame_indices"], indices):
        raise ValueError("Tracked poses do not cover the selected demonstration frames")
    if task["relation"] == "tool_use":
        # The released tool subset already grounds its destination semantics.
        # Repeating descriptor matching here would add work with no consumer.
        return {"tool_use": True, "task": task, "instruction": instruction, "tracking": track}
    scene_roles = select_roles(scene_image, task, "scene", detector=detector, client=client)
    role_masks("scene", scene_roles)
    # END common observation path; destination grounding remains task-specific.
    actual_extractor = extractor
    if actual_extractor is None:
        from local_models.sd_dino import extract_pair
        actual_extractor = extract_pair

    bindings, evidence = {}, {}
    for role in ("reference", "carrier"):
        bindings[role], evidence[role] = _binding(
            demo_image, scene_image, demo_depth, scene_depth, demo_camera, scene_camera,
            masks["demo"][role], masks["scene"][role], actual_extractor)
    reference = bindings["reference"]
    # The body frame is transferred from demo-visible body features. Scene
    # geometry is then expressed in that fixed frame, retaining shape changes
    # instead of pretending that the whole new object equals the demonstration.
    initial_pose = bindings["carrier"] @ demo_poses[0]
    body = _body_points(scene_depth, scene_camera, masks["scene"]["carrier"], initial_pose)
    goal = reference @ demo_poses[-1]
    scene = {"initial_pose": initial_pose, "reference_transform": reference,
             "body_points": body, "preferred_goal_pose": goal,
             "goal_source": "demonstrated endpoint bound through visible reference correspondences",
             "geometry_coverage": "visible RGB-D surface samples; hidden geometry is unknown"}
    geometry_path = scene_dir / "geometry.json"
    if geometry_path.is_file():
        geometry = json.loads(geometry_path.read_text())
        if geometry.get("reference_frame") != "scene_camera":
            raise ValueError("Optional scene geometry must declare reference_frame=scene_camera")
        for key in ("support", "interior", "entry_region", "entry_start_index", "margin"):
            if key in geometry:
                scene[key] = geometry[key]
    if task["relation"] == "on" and "support" not in scene:
        # A reference object's largest plane may be a side panel. Select an
        # observed plane by the task and retain only its observed support area.
        # Orient its side from the demonstrated goal, since a carried object
        # may start below its destination support before being lifted onto it.
        # Use a material point average, not the arbitrary body-frame origin,
        # so a different valid body coordinate convention cannot flip the side.
        goal_center = goal[:3, :3] @ body.mean(axis=0) + goal[:3, 3]
        surface, surface_evidence = infer_support(
            scene_image, scene_depth, scene_camera.K, masks["scene"]["reference"], goal_center,
            instruction=instruction + "\n" + task.get("goal_description", ""), client=client)
        scene["support"] = surface.to_dict()
        evidence["support"] = surface_evidence
    # BEGIN demonstrated containment target without invented hidden geometry.
    # The reference-bound endpoint already specifies the demonstrated inside
    # motion. Missing cavity geometry limits containment verification, rather
    # than requiring another offline input before that motion can transfer.
    # run_stage preserves this distinction and still validates supplied regions.
    # END demonstrated containment target without invented hidden geometry.
    if task["relation"] == "articulation" and task["motion"] == "free":
        raise ValueError("Articulation cannot be silently represented by unconstrained free motion")
    # Noisy image bindings produce an estimated pose, not an exact six-DoF
    # semantic command. For constrained motion, make any within-tolerance model
    # projection explicit; incompatible binding remains an error.
    fitted = transfer_motion(demo_poses, initial_pose, reference_transform=reference, kind=motion_kind(task))
    if fitted["model"].kind != "free":
        coordinates = fitted["model"].coordinates_of(goal[None])
        projected = fitted["model"].poses(coordinates)[0]
        evidence["goal_projection"] = {
            "position_correction_m": float(np.linalg.norm(goal[:3, 3]-projected[:3, 3])),
            "rotation_correction_rad": float(Rotation.from_matrix(goal[:3, :3] @ projected[:3, :3].T).magnitude()),
            "meaning": "projection of estimated correspondence geometry within motion-fit tolerances",
        }
        scene["preferred_goal_pose"] = projected
    evidence["tracking"] = {key: value for key, value in track.items() if key not in {"poses", "points"}}
    return {"task": task, "instruction": instruction, "demo_poses": demo_poses,
            "frame_indices": track.get("frame_indices", indices),
            "carrier_id": f"carrier:{scene_roles['carrier']['candidate_id']}",
            "reference_id": f"reference:{scene_roles['reference']['candidate_id']}",
            "reference_frame": "scene_camera", "scene": scene, "evidence": evidence}


def run_rgbd(demo_dir, scene_dir, instruction, output_dir, **dependencies):
    """Execute semantic grounding, observations and trajectory generation once."""
    destination = Path(output_dir)
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise FileExistsError(f"Use an empty output directory: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    report = {"status": "running", "instruction": instruction,
              "demo": str(Path(demo_dir).resolve()), "scene": str(Path(scene_dir).resolve())}
    write_json(destination / "run.json", report)
    try:
        observation = ground_rgbd(demo_dir, scene_dir, instruction, output_dir=destination, **dependencies)
        # BEGIN save the actual motion source before either downstream consumer.
        write_json(destination / "observations.json", observation)
        if observation.get("tool_use"):
            from .tool_use import run_tool_use
            result = run_tool_use(demo_dir, scene_dir, observation["task"], instruction,
                                  destination, tracking=observation["tracking"], client=dependencies.get("client"))
        else:
            result = run_stage(observation)
            write_json(destination / "trajectory.json", result)
        # END common motion provenance and task-specific trajectory generation.
        report["status"] = "passed"
        report["output_type"] = result["output_type"]
    except BaseException as error:
        # The preserved optimizer may signal failure through SystemExit.
        # Record that failure (or interruption) before propagating it unchanged.
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(destination / "run.json", report)
    return result
