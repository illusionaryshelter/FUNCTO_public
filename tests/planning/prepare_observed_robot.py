"""Create link-local implicit fields from a recorded demonstration.

The recording supplies RGB-D, source identity labels, joint coordinates and
camera/base calibration. Only identity-to-link and joint-column metadata are
read from its catalog. The kinematic URDF must contain no geometry elements.
"""

# BEGIN a recording adapter prepares the same entity fields used by scenes.
import argparse
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import h5py
import numpy as np
import torch
import yourdfpy

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from manipulation.observed_mapping import integrate_entity


def fixed_components(tree):
    """Map every link to the root of its fixed-joint connected component."""
    parents = {joint.find("child").get("link"): (
        joint.find("parent").get("link"), joint.get("type")) for joint in tree.findall("joint")}
    result = {}
    for link in tree.findall("link"):
        name = link.get("name")
        root = name
        while root in parents and parents[root][1] == "fixed":
            root = parents[root][0]
        result[name] = root
    return result


def prepare(args):
    from curobo.perception import MapperCfg

    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    tree = ET.parse(args.urdf).getroot()
    if tree.findall(".//visual") or tree.findall(".//collision") or tree.findall(".//mesh"):
        raise ValueError("Observation preparation requires a pure kinematic URDF")
    groups = fixed_components(tree)
    model = yourdfpy.URDF.load(args.urdf, load_meshes=False, load_collision_meshes=False,
                              build_scene_graph=True, build_collision_scene_graph=False)
    catalog = json.loads(Path(args.catalog).read_text())
    body_names = {row["id"]: row["name"] for row in catalog["bodies"]}
    labels = {}
    for row in catalog["geoms"]:
        link = body_names[row["body_id"]]
        if link in groups:
            # Preserve link-local source partitions within a rigid assembly.
            # In particular, a contact declaration on one attached pad must
            # not expand to the complete finger when fixed links are grouped.
            labels.setdefault(link, []).append(row["id"])
    columns = {row["name"]: row["qpos_start"] for row in catalog["joints"]
               if row["name"] in model.joint_map and row["qpos_count"] == 1}
    independent = {name for name, joint in model.joint_map.items()
                   if joint.type != "fixed" and joint.mimic is None}
    if set(columns) != independent or len(set(columns.values())) != len(columns):
        raise ValueError("Source scalar joint columns must cover the complete independent kinematic state")
    calibration = json.loads(Path(args.calibration).read_text())
    K = np.asarray(calibration["K"], dtype=float)
    camera = np.asarray(calibration["T_world_camera_cv"], dtype=float)
    world_base = np.asarray(json.loads(Path(args.base_calibration).read_text())["T_world_base"], float)
    root = model.base_link
    with h5py.File(args.episode, "r") as episode:
        frames = sorted(set(range(0, len(episode["frames/depth_m"]), args.frame_stride))
                        | set(args.include_frame))
        poses = {}
        for frame in frames:
            q = episode["states/qpos"][frame]
            model.update_cfg({name: float(q[column]) for name, column in columns.items()})
            poses[frame] = {name: world_base @ model.get_transform(name, frame_from=root)
                            for name in labels}

        records = []
        for name, identities in labels.items():
            lower, upper = np.full(3, np.inf), np.full(3, -np.inf)
            pixel_counts = {}
            # Bounds come from the same measured endpoints submitted below.
            # Only the current image and its selected points are resident.
            for frame in frames:
                mask = np.isin(episode["frames/seg_geom_id"][frame], identities)
                depth = episode["frames/depth_m"][frame]
                mask &= np.isfinite(depth) & (depth >= .05) & (depth <= 5.)
                y, x = np.nonzero(mask)
                pixel_counts[str(frame)] = len(x)
                if not len(x):
                    continue
                points = np.column_stack((x, y, np.ones(len(x)))) @ np.linalg.inv(K).T
                points *= depth[y, x, None]
                transform = np.linalg.inv(poses[frame][name]) @ camera
                points = points @ transform[:3, :3].T + transform[:3, 3]
                lower, upper = np.minimum(lower, points.min(0)), np.maximum(upper, points.max(0))
            if not np.isfinite(lower).all():
                records.append({"frame": name, "rigid_component": groups[name],
                                "status": "missing_observation",
                                "source_label_ids": identities, "pixel_counts": pixel_counts})
                continue
            padding = 4 * args.voxel_size
            lower, upper = lower - padding, upper + padding
            count = np.maximum(2, np.ceil((upper - lower) / args.voxel_size).astype(int) + 1)
            extent = count * args.voxel_size
            config = MapperCfg(
                extent_meters_xyz=tuple(extent), voxel_size=args.voxel_size,
                extent_esdf_meters_xyz=tuple(extent), esdf_voxel_size=args.voxel_size,
                grid_center=torch.tensor((lower + upper) / 2, dtype=torch.float32),
                truncation_distance=2 * args.voxel_size, image_height=calibration["height"],
                image_width=calibration["width"], depth_minimum_distance=.05,
                depth_maximum_distance=5., device="cuda:0",
            )

            def observations():
                for frame in frames:
                    if not pixel_counts[str(frame)]:
                        continue
                    yield {
                        "image": episode["frames/rgb"][frame],
                        "depth": episode["frames/depth_m"][frame],
                        "mask": np.isin(episode["frames/seg_geom_id"][frame], identities),
                        "K": K, "T_entity_camera": np.linalg.inv(poses[frame][name]) @ camera,
                    }

            print(json.dumps({"stage": "integrating", "frame": name,
                              "sample_shape_xyz": count.tolist(),
                              "observed_pixels": sum(pixel_counts.values())}), flush=True)
            field, report = integrate_entity(
                observations(), config=config, name=name,
                source=f"source_demonstration:{Path(args.episode).resolve()}",
                max_host_rss_bytes=args.max_host_rss_mib * 1024 ** 2,
                max_device_used_bytes=args.max_device_used_mib * 1024 ** 2,
                block_checkpoint=destination / f"{name}.pt",
            )
            field_path = destination / f"{name}.json"
            field.save(field_path)
            records.append({"frame": name, "rigid_component": groups[name],
                            "status": "observed", "field": str(field_path.resolve()),
                            "block_checkpoint": str((destination / f"{name}.pt").resolve()),
                            "source_label_ids": identities, "pixel_counts": pixel_counts,
                            "summary": report})
            print(json.dumps({"stage": "saved", "frame": name, **field.summary()}), flush=True)
            del field
    result = {
        "kind": "observed_robot_v1", "urdf_path": str(Path(args.urdf).resolve()),
        "base_frame": root, "source_frames": frames, "fixed_component_frames": groups,
        "fields": records, "source": {
            "episode": str(Path(args.episode).resolve()), "camera_calibration": args.calibration,
            "base_calibration": args.base_calibration,
            "catalog_use": "source segmentation identity-to-link and scalar joint columns",
        },
    }
    (destination / "observed-robot.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("episode", "catalog", "calibration", "base-calibration", "urdf", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--frame-stride", type=int, default=24)
    parser.add_argument("--include-frame", type=int, nargs="*", default=[])
    parser.add_argument("--voxel-size", type=float, default=.002)
    parser.add_argument("--max-host-rss-mib", type=int, default=8192)
    parser.add_argument("--max-device-used-mib", type=int, default=6144)
    return parser.parse_args()


if __name__ == "__main__":
    prepare(parse_args())
# END no current-run state or shape resource enters the source observation model.
