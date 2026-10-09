"""Own one live native trial across separately planned, observed skill requests.

This is a simulation test adapter. Native task scores decide whether the test
may submit its next stage; they never supply object poses, attachments, contact
geometry or online effect observations. A stage uses the ordinary executor.
The simulator remains alive while the caller performs perception and planning.
"""

# BEGIN continuous test-session ownership; no task or dataset names in this API.
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
from PIL import Image

from manipulation.execution import _feedback_json
from run_recorded_simulation import run


def _read(path):
    return json.loads(Path(path).read_text())


def _write(path, value):
    """Create a new observation/report, preserving any prior trial evidence."""
    with Path(path).open("x") as stream:
        json.dump(_feedback_json(value), stream, indent=2, allow_nan=False)
        stream.write("\n")


class ContinuousSimulationSession:
    """One restore, explicit actual feedback, and independent per-stage scores.

    The caller may run expensive child processes between calls. The integration
    state is compared bit for bit before the next stage so a reset, hidden step,
    velocity edit or actuator rewrite cannot masquerade as continuous execution.
    These native arrays stay private to this audit and are never planner input.
    """

    def __init__(self, *, manifest_path, fixture_path, controller_part, output):
        self.output = Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=False)
        self.manifest_path, self.fixture_path = Path(manifest_path), Path(fixture_path)
        self.manifest, self.fixture = _read(manifest_path), _read(fixture_path)
        self.root = Path(self.manifest["workspace_root"])
        recording = self.root / self.manifest["source_recording"]
        if Path(self.fixture["episode_path"]).resolve() != (recording / "episode.h5").resolve():
            raise ValueError("The continuous fixture and recording must identify one episode")
        self.environment = _read(recording / "manifest.json")["environment"]
        self.controller_part = controller_part
        controller = self.environment["controller_config"]["body_parts"][controller_part]
        if (controller["type"] != "JOINT_POSITION" or controller["input_type"] != "absolute"
                or controller["interpolation"] is not None):
            raise ValueError("The current sequence requires the declared absolute native joint controller")
        # The first ordinary runner owns the sole restoration and preserves
        # the planned historical feedback identity. Its lifecycle hook hands
        # ownership of that very backend to this session before control starts.
        # Later requests use fresh native identities throughout planning and
        # execution; no initial-feedback alias or curve metadata is rewritten.
        self.simulation = None
        self.trial_id = None
        self.stages = []
        self._boundary = None
        self._gate = None
        self.finalization_errors = []

    def _adopt(self, simulation, restored):
        if self.simulation is not None:
            raise RuntimeError("A continuous session cannot adopt a second restored backend")
        self.simulation = simulation
        self.trial_id = simulation.proprioception()["trial_id"]
        self._boundary = self._integration_state()
        _write(self.output / "session-start.json", {
            "trial_id": self.trial_id, "restore_count": 1, "restore": restored,
            "initial_feedback": self.simulation.proprioception(),
            # BEGIN input provenance is owned by the caller's explicit experiment.
            "stage_result_policy": "Independent simulation outcome gates test stages; geometry/control inputs retain their separately declared sources, and this result gate supplies no geometry",
            # END a GT-input experiment must not be mislabeled as vision-only.
            "between_stage_timing": "Physics is paused without changing state while perception/planning uses wall time"})

    def _integration_state(self):
        kind = mujoco.mjtState.mjSTATE_INTEGRATION
        state = np.empty(mujoco.mj_stateSize(self.simulation.native_model, kind))
        mujoco.mj_getState(self.simulation.native_model, self.simulation.native_data, state, kind)
        return state

    def assert_boundary_unchanged(self):
        if (self.simulation is None or self._boundary is None
                or self.simulation.proprioception()["trial_id"] != self.trial_id
                or not np.array_equal(self._integration_state(), self._boundary)):
            raise RuntimeError("Native trial/state changed between consecutive skill requests")

    # BEGIN continuous callers retain separate assumption and predicate inputs.
    def execute(self, specification, *, attachment_assumptions=None, observation_callback=None):
        """Execute one new skill from the same live backend, then grade it.

        ``specification`` contains ordinary single-stage runner arguments. It
        cannot replace the native backend. After the first stage, the previous
        full physical outcome must be T. U/F or execution faults stop the test;
        the caller cannot use a new skill identity to bypass them.

        ``attachment_assumptions`` forwards explicit bound-condition sources to
        the executor. It is never an observation. ``observation_callback`` is
        the independent predicate-layer interface; a future S3E adapter can
        supply current image-based estimates through that existing contract.
        Neither input is inferred from the independent native physics grade.
        """
        spec = deepcopy(_read(specification) if isinstance(specification, (str, Path)) else specification)
        if self.stages and (self._gate is None or self._gate["value"] != "T"):
            raise RuntimeError("The preceding simulation stage has not passed its independent result gate")
        if self.simulation is not None:
            self.assert_boundary_unchanged()
        for key in ("manifest", "robot_fixture", "feedback", "trajectory", "finger_targets",
                    "execution_limits", "evaluator_evidence", "output"):
            spec[key] = Path(spec[key]).resolve()
        if spec["robot_fixture"] != self.fixture_path.resolve():
            raise ValueError("Every stage must use the same bound robot/native fixture")
        args = SimpleNamespace(**spec)
        before = None if self.simulation is None else self.simulation.proprioception()
        report = None
        primary_error = None
        try:
            inputs = {"attachment_assumptions": attachment_assumptions,
                      "observation_callback": observation_callback}
            report = (run(args, on_simulation_ready=self._adopt, **inputs) if self.simulation is None else
                      run(args, simulation=self.simulation, initial_feedback=before, **inputs))
        except BaseException as error:
            primary_error = error
        # A failed execution may perform its ordinary abort hold. Keep that
        # actual state. A subsequent audit/I/O error cannot replace the first
        # physical exception or authorize another stage.
        after, errors = None, []
        if self.simulation is not None:
            try:
                after = self.simulation.proprioception()
            except Exception as error:
                errors.append(error)
            try:
                self._boundary = self._integration_state()
            except Exception as error:
                self._boundary = None
                errors.append(error)
        self._gate = {"value": "T" if primary_error is None and not errors and report is not None
            and report["status"] == "completed" and report["physics_success"] is True else "U",
            "source": {"kind": "independent_simulation_result_gate",
                "assessment_path": str(args.output / "evaluation-report.json")}}
        if report is not None and report["physics_success"] is False:
            self._gate["value"] = "F"
        entry = {"stage_index": len(self.stages), "segment_id": args.segment,
            "trial_id": self.trial_id,
            "native_start_time_s": (None if report is None else report.get("native_start_time_s"))
                if before is None else before["timestamp_s"],
            "native_end_time_s": None if after is None else after["timestamp_s"],
            "initial_feedback": before, "final_feedback": after, "result_gate": deepcopy(self._gate),
            "simulation_report": str(args.output / "simulation-report.json"),
            "same_backend_instance": self.simulation is not None,
            "restore_count": int(self.simulation is not None),
            "primary_error": None if primary_error is None else repr(primary_error),
            "audit_errors": [repr(error) for error in errors]}
        self.stages.append(entry)
        try:
            _write(self.output / f"stage-{len(self.stages):02d}.json", entry)
        except Exception as error:
            errors.append(error)
        self.finalization_errors.extend(repr(error) for error in errors)
        if errors:
            self._gate["value"] = "U"
        if primary_error is not None:
            raise primary_error
        if errors:
            raise RuntimeError("Continuous stage audit or report persistence failed: " +
                               "; ".join(repr(error) for error in errors)) from errors[0]
        return report
    # END native continuity and stage-result gates keep their original duties.

    def capture(self, destination, *, views, tool_id, result_predicate):
        """Export only synchronized camera measurements and named robot feedback.

        Camera-native segmentation is deliberately discarded. The returned
        manifest exposes no simulator body ID, object pose, force, contact or
        integration-state array. The same observed images may serve both task
        perception and collision mapping, with their masks produced separately.
        """
        from collect.camera import FixedCamera
        from recorded_observation import camera_config_from_calibration

        self.assert_boundary_unchanged()
        if self._gate is None or self._gate["value"] != "T":
            raise RuntimeError("A failed or ungraded stage cannot authorize the next observed request")
        if not isinstance(result_predicate, str) or not result_predicate.strip():
            raise ValueError("The test caller must name its independently evaluated transition predicate")
        destination = Path(destination).resolve()
        destination.mkdir(parents=True, exist_ok=False)
        feedback = self.simulation.proprioception()
        capture_id = f"{self.trial_id}:boundary:{len(self.stages)}"
        feedback = {**feedback, "start_q": feedback["q"], "start_dq": feedback["dq"],
            "eef_pose": feedback["tool_pose"], "sim_time_s": feedback["timestamp_s"],
            "tool_id": tool_id, "tool_frame": feedback["tool_link"]}
        _write(destination / "feedback.json", feedback)
        captured = []
        for view in views:
            calibration = _read(self.root / view["calibration_path"])
            folder = destination / view["id"]
            folder.mkdir()
            camera = FixedCamera(self.simulation.native_model, self.simulation.native_data,
                width=calibration["width"], height=calibration["height"], camera_name=None,
                camera_config=camera_config_from_calibration(calibration))
            capture_error = None
            try:
                native = camera.calibration
                for key in ("K", "T_world_camera_cv"):
                    if not np.allclose(native[key], calibration[key], atol=1e-9, rtol=0):
                        raise ValueError("Boundary camera calibration changed")
                frame = camera.capture(self.simulation.native_data)
                Image.fromarray(frame["rgb"]).save(folder / "rgb.png")
                np.save(folder / "depth_m.npy", frame["depth_m"])
                _write(folder / "calibration.json", calibration)
            except Exception as error:
                capture_error = error
            try:
                camera.close()
            except Exception as error:
                self.finalization_errors.append("boundary_camera_close: " + repr(error))
                if capture_error is None:
                    capture_error = error
            if capture_error is not None:
                raise capture_error
            captured.append({"id": view["id"], "rgb_path": f"{view['id']}/rgb.png",
                "depth_path": f"{view['id']}/depth_m.npy", "calibration_path": f"{view['id']}/calibration.json"})
        self.assert_boundary_unchanged()
        _write(destination / "capture.json", {"capture_id": capture_id,
            "sim_time_s": feedback["timestamp_s"], "feedback_path": "feedback.json", "views": captured,
            "source": "Actual camera RGB-D and same-tick named robot feedback in the unchanged live trial"})
        _write(destination / "result-gate.json", {"predicate": result_predicate, "value": "T",
            "capture_id": capture_id, "source": deepcopy(self._gate["source"])})
        return destination / "capture.json"
# END test-result gating cannot supply any of the next stage's object geometry.
