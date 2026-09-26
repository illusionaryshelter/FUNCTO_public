"""Transfer an object/part demonstration, or replay grounded observations.

Examples:
  python manipulate.py --demo demo_dir --scene scene_dir --instruction 'push the drawer closed' --output-dir runs/close
  python manipulate.py --prepared observations.json --output-dir runs/replay

RGB-D inputs use aligned rgb/, depth/ and intrinsic.json. The output is an
object/part trajectory, not a robot end-effector or force-control command.
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", type=Path, help="One demonstrated RGB-D stage")
    parser.add_argument("--scene", type=Path, help="Current calibrated initial RGB-D observation")
    parser.add_argument("--instruction", help="Task instruction for the demonstrated stage")
    parser.add_argument("--prepared", type=Path, help="Replay already grounded observations or ordered stages")
    parser.add_argument("--output-dir", type=Path, required=True, help="Empty directory for this run")
    parser.add_argument("--no-render", action="store_true", help="Skip the bounded offline trajectory plot")
    args = parser.parse_args()
    if args.prepared:
        if any((args.demo, args.scene, args.instruction)):
            parser.error("--prepared is an alternative to --demo/--scene/--instruction")
        from manipulation.pipeline import run_prepared
        result = run_prepared(args.prepared, args.output_dir)
    else:
        if not all((args.demo, args.scene, args.instruction)):
            parser.error("Provide --demo, --scene and --instruction, or use --prepared")
        from manipulation.perception import run_rgbd
        result = run_rgbd(args.demo, args.scene, args.instruction, args.output_dir)
    if not args.no_render:
        from manipulation.visualization import render_result
        from manipulation.pipeline import write_json
        report_path = args.output_dir / "run.json"
        report = json.loads(report_path.read_text())
        # Numeric generation and rendering are separately inspectable. The CLI
        # requests both by default, so a rendering failure cannot leave its
        # overall run marked passed even though the numeric artifact is valid.
        report.update(status="rendering", trajectory_status="passed", visualization_status="running")
        write_json(report_path, report)
        try:
            report["visualization"] = render_result(result, args.output_dir)
            report.update(status="passed", visualization_status="passed")
        except BaseException as error:
            report.update(status="failed", visualization_status="failed",
                          error=f"Visualization {type(error).__name__}: {error}")
            raise
        finally:
            write_json(report_path, report)
    print(f"FUNCTO manipulation completed: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
