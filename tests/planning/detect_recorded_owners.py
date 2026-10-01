"""Run the existing isolated SAM worker on explicitly selected current RGBs.

The caller schedules this command only after other local GPU work has exited.
Every nonempty SAM instance remains an independent candidate. No source mask,
current simulator truth, detector-score selection or merged owner mask is
introduced. Semantic target confirmation is a separate recorded input.
"""

# BEGIN manifest-selected current owner detection through the existing worker.
import argparse
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _reject_hidden_state(event, args):
    """Keep this perception process limited to visible image/metadata inputs."""
    if event != "open" or not isinstance(args[0], (str, bytes)):
        return
    value = args[0].decode() if isinstance(args[0], bytes) else args[0]
    path = Path(value)
    if ("evaluation" in path.parts or path.suffix in {".h5", ".hdf5"}
            or path.name in {"scene.xml", "model_catalog.json", "seg_geom_id.npy",
                             "initial_scene_truth.npz"}):
        raise PermissionError(f"Current owner detection cannot read hidden object state: {path}")


def run(args):
    from local_models.sam31 import detect

    manifest = json.loads(args.manifest.read_text())
    root = Path(manifest["workspace_root"])
    segment = next(item for item in manifest["segments"] if item["id"] == args.segment)
    args.output.mkdir(parents=True, exist_ok=False)
    sys.addaudithook(_reject_hidden_state)
    report = {"segment_id": args.segment, "entry_frame": segment["snapshot"]["entry_frame"],
              "queries": args.labels, "backend": "existing local_models.sam31.detect isolated worker",
              "selection_scope": "all detected nonempty instances; no target chosen",
              "current_object_state_read": False, "views": []}
    _write_json(args.output / "run.json", report)
    for view in segment["snapshot"]["views"]:
        folder = args.output / view["id"]
        folder.mkdir()
        path = root / view["rgb_path"]
        item = {"view_id": view["id"], "input_rgb_path": str(path.resolve()), "status": "running"}
        _write_json(folder / "run.json", item)
        started = time.monotonic()
        try:
            with Image.open(path) as opened:
                rgb = opened.convert("RGB")
            if rgb.width * rgb.height > 1920*1080:
                raise ValueError("This recorded detection runner admits at most a two-megapixel source image")
            candidates = detect(rgb, args.labels)
            masks = (np.stack([row["mask"] for row in candidates]) if candidates
                     else np.empty((0, rgb.height, rgb.width), dtype=bool))
            metadata = [{"candidate_id": index+1, "mask_index": index,
                         "label": row["label"], "score": row["score"], "bbox": row["bbox"],
                         "visible_pixels": int(row["mask"].sum())}
                        for index, row in enumerate(candidates)]
            np.savez_compressed(folder / "all_masks.npz", masks=masks)
            _write_json(folder / "detections.json", metadata)
            # One numbered panel per independent mask makes later semantic
            # inspection possible without merging the masks or ranking them.
            panels = Image.new("RGB", (rgb.width, rgb.height*max(1, len(candidates))))
            if not candidates:
                panels.paste(rgb, (0, 0))
            for index, row in enumerate(candidates):
                overlay = Image.composite(Image.new("RGB", rgb.size, (30, 220, 70)), rgb,
                                          Image.fromarray(row["mask"].astype(np.uint8)*105))
                painter = ImageDraw.Draw(overlay)
                painter.rectangle(row["bbox"], outline=(255, 255, 0), width=2)
                painter.text((8, 8), f"{index+1}: {row['label']}", fill=(255, 255, 0))
                panels.paste(overlay, (0, index*rgb.height))
            panels.save(folder / "candidates.png")
            item.update(status="detected" if candidates else "no_detections", candidate_count=len(candidates),
                        resolution=[rgb.width, rgb.height], selected_candidate_id=None,
                        masks_path=str((folder / "all_masks.npz").resolve()),
                        detections_path=str((folder / "detections.json").resolve()))
        except Exception as error:
            item.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
            (folder / "exception.txt").write_text(traceback.format_exc())
        finally:
            item["elapsed_s"] = time.monotonic() - started
            _write_json(folder / "run.json", item)
            report["views"].append(item)
            _write_json(args.output / "run.json", report)
    report["status"] = "failed" if any(item["status"] == "failed" for item in report["views"]) else "completed"
    _write_json(args.output / "run.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--segment", required=True)
    parser.add_argument("--labels", required=True, nargs="+")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps(report))
    return 1 if report["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
# END manifest-selected current owner detection through the existing worker.
