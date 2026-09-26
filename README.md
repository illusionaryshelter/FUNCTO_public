# An Official Code Implementation of MimicFunc: Imitating Tool Manipulation from a Single Human Video via Functional Correspondence (CoRL 2025)
[[Preprint]](https://arxiv.org/abs/2502.11744) [[website]](https://sites.google.com/view/mimicfunc) [[video]](https://youtu.be/H0sEzJufpeE)

[Chao Tang](https://mkt1412.github.io/), Anxing Xiao, Yuhong Deng, Tianrun Hu, Wenlong Dong, Hanbo Zhang, David Hsu, and Hong Zhang  

If you find this work useful, please cite:
```
@article{tang2025functo,
  title={Functo: Function-centric one-shot imitation learning for tool manipulation},
  author={Tang, Chao and Xiao, Anxing and Deng, Yuhong and Hu, Tianrun and Dong, Wenlong and Zhang, Hanbo and Hsu, David and Zhang, Hong},
  journal={arXiv preprint arXiv:2502.11744},
  year={2025}
}
  ```

## Local pipeline

This workspace uses one `recovery` Python environment for local **SAM3.1 +
SD-DINO** perception. Gemini performs semantic keypoint selection and optional
pose selection through the existing s3e gateway backend. There are no OWLv2,
SAM, SD-DINO or LLaVA model servers to start.

- `local_models/sam31.py`: SAM3.1 text detection and segmentation from the local
  multiplex checkpoint. Boxes and masks always come from the same instance.
- `local_models/sd_dino.py`: SD v1.3, timestep 0, DINOv2 ViT-B/14 token layer 11,
  native shared two-image co-PCA (256 dimensions per SD scale), branch L2
  normalization and equal-weight concatenation. SD960 and DINO840 produce two
  FP32 `[1,1536,60,60]` descriptors. The old remote feature service's implementation
  was not provided; this is an explicit local feature definition.
- `local_models/runtime.py`: sequential, short-lived local worker processes
  release model RAM, GPU allocations and autocast state before dense matching.
  Network access is disabled inside perception workers; missing weights fail
  explicitly. The worker CUDA allocator is capped at 8 GiB, combined caller and
  worker RSS at 12 GiB, with at least 1.5 GiB host memory kept available.
  <!-- BEGIN host allocator reclamation before resource rejection. -->
  On host-memory pressure, the Linux worker first collects unreachable objects
  and returns unused glibc allocator pages, then rechecks the same limits.
  Live tensors, model calls and numerical settings are unchanged.
  <!-- END host allocator reclamation. -->
- `local_models/gemini.py`: reuses the sibling `s3e` source backend. It reads
  `SUB2API_API_KEY` only when called. `SUB2API_BASE_URL` defaults to the user's
  local gateway `http://127.0.0.1:18080/antigravity`; `GEMINI_MODEL` defaults to
  `gemini-3.7-flash-tiered`. Override these environment variables if needed.
  Annotated images are sent to that authorized gateway. API failures and invalid
  candidate indices raise errors instead of using old predictions.

Expected sibling repositories are `sam3`, `sd-dino` and `s3e`. SAM weights live at
`../models/sam3.1/sam3.1_multiplex.pt`; SD/ODISE/OpenCLIP and DINO use the existing
local caches checked by the adapter. No entry point installs dependencies or
downloads weights. The geometry stages require Open3D, SciPy, OpenCV and CasADi
with IPOPT, in addition to the existing model dependencies.

The raw SD path retains the accepted unused-component GPU optimization. It does
not share the two different CLIP models, change precision, cache pair features,
or change FUNCTO's interpolation, cosine similarity, mask multiplication or
global argmax.

## Demo

`main.py` implements released steps 8--12: it consumes the prepared demonstration
annotations and trajectory, detects the test scene, transfers points and poses,
then optimizes a trajectory. It does not implement raw human-video preparation
or robot execution.

The task configuration is `utils_IL/config/config_pour.yaml`. SD-DINO is enabled
by default. `vp_flag: false` disables only optional pose selection; semantic
function/grasp point selection still uses Gemini.

```bash
PYTHONNOUSERSITE=1 MPLCONFIGDIR=/tmp/functo-mpl OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  python main.py --output-dir ./runs/pour-local
```

The output directory must be empty. Only the initial test RGB-D observation and
its metadata are copied there, and all predictions are generated anew. Without
`--output-dir`, each invocation creates a timestamped directory under `runs/`.
`--data-root` selects a root containing `demo_data/<task>_demo` and
`test_data/<task>_test`; `--config` selects another YAML task configuration.
Paths to prompts and default configuration are independent of the current
working directory. The bundled RGB-D example uses the original Azure Kinect
intrinsics and millimeter depth units in `utils_IL/params.py`; custom recordings
need matching calibration before geometric processing.

After success, `run.json` records stage completion and validates finite 2D/3D
points and trajectory arrays. The final files are
`tool_traj_transfer_output/test_tool_trajectory.json` and
`tool_traj_transfer_output/test_tool_traj_pc.ply` inside that run directory.
Historical results under `test_data/` remain example data, never substitutes for
a completed new run.

<!-- BEGIN offline visualization of saved local predictions. -->
## Visual validation

Render a completed run without model inference or Gemini calls:

```bash
PYTHONNOUSERSITE=1 MPLCONFIGDIR=/tmp/functo-mpl OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  python visualize_run.py ./runs/pour-local
```

Open `visualization/index.html` inside the run. It links the mask/keypoint
overlays, saved semantic selections, full trajectory MP4, an offline interactive
3D replay with a frame slider, and numerical diagnostics. `--demo-dir` overrides
the demonstration path recorded in `run.json` if the data has moved.

The renderer reuses the original point-cloud reconstruction and compares its
poses against every tool snapshot in the saved PLY. Only displayed point counts
are reduced. The static original tool is removed from the background; hidden
surfaces stay unknown. White crosses in the image show the actual 3D points
reprojected through the saved calibration, exposing depth-neighbor adjustments
to the 2D matches. Trajectory translations are **function-point** positions;
video FPS is illustrative because the saved poses have no timestamps. These
views support semantic and geometric inspection, not collision, dynamics or
robot-execution certification.
<!-- END offline visualization of saved local predictions. -->

## Contract checks

The focused tests use synthetic CPU data and mock model/API boundaries. They
check instance association, semantic response parsing, fresh output isolation,
and result validation; they do not download weights or initialize GPU models.

```bash
PYTHONNOUSERSITE=1 MPLBACKEND=Agg PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  python -m pytest -q tests cloud_services/test_scene_graph_contract.py
```

<!-- BEGIN task-adaptive object/part manipulation. -->
## Object and moving-part manipulation

`manipulate.py` adds a separate path for object transport, placement, containment
and articulated parts. Motion, goal relation and interaction are independent:
an object does not need a tool function point, and a pushed part does not need a
grasp point. The existing `main.py` tool-use example remains available unchanged.

```bash
PYTHONNOUSERSITE=1 MPLCONFIGDIR=/tmp/functo-mpl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python manipulate.py --demo /path/to/demonstrated-stage --scene /path/to/current-scene \
  --instruction "push the drawer closed" --output-dir runs/drawer-close
```

Each RGB-D directory contains `rgb/00000.jpg` (or PNG), registered
`depth/00000.png` (or NPY), and `intrinsic.json`. The demonstration contains the
whole stage's numbered frames. Open3D's column-major `intrinsic_matrix`, width
and height are read from calibration. `depth_unit_m` explicitly specifies metres
per stored value; Open3D `depth_scale` specifies its divisor. If neither is
provided, the original millimetre convention applies. Geometry is no longer
restricted to the original one-metre range or hard-coded camera intrinsics.

The new path uses Gemini for role/relationship selection, SAM3.1 for entity and
part masks, SD-DINO for visible cross-instance correspondences, and bounded CPU
RGB-D tracking for demonstrated rigid motion. Parent-object masks exclude the
moving part when tracking a fixed reference. Registration diagnostics remain
in the result; texture loss, ambiguous correspondences and incompatible motion
are errors, rather than a fallback to fabricated points or unconstrained motion.
Raw tool-use classification routes to the existing annotated tool subset, which
still requires its original demonstration annotations and calibration.

An `on` goal selects a task-relevant observed plane with Gemini, then uses a
bounded patch that remains inside its observed mask, including any holes. It
does not treat the largest face of a reference object as its support surface.
The supplied object extent must fit the patch. An `inside` goal requires known **free interior geometry**, never the
outer box of a container. An optional `geometry.json` in the current-scene
directory can carry geometry produced by another component:

```json
{
  "reference_frame": "scene_camera",
  "interior": {
    "vertices": [[0,0,0.5], [0.2,0,0.5], [0,0.2,0.5], [0.2,0.2,0.5],
                 [0,0,0.7], [0.2,0,0.7], [0,0.2,0.7], [0.2,0.2,0.7]],
    "source": "known free cavity geometry"
  }
}
```

This example describes a convex free region in metres. Partial RGB-D does not
reveal an occluded cavity, and neither a semantic label nor an outer mask is used
to invent it. Optional `entry_region` and `entry_start_index` check the supplied
object geometry against a known entry corridor at the generated trajectory
nodes, using the supplied `margin` in metres. They do not certify interpolation
between those nodes. These local checks are not complete-scene collision or force planning.

The numerical API `manipulation.pipeline.run_stage` accepts already grounded
observations. `run_sequence` accepts an ordered list and updates carrier and
reference states. Repeated entities require registered, fixed `body_frame_id`
values; moving references require `reference_body_frame_id` and their capture
pose. A placed object is not assumed rigidly attached to its support/container:
if that reference later moves, its world state becomes `needs_observation`.
This sequence API consumes a provided order; it does not discover a new task
plan from a long instruction.

`scene.goal_pose` is an exact target and is never projected into a convenient
answer. `scene.preferred_goal_pose` is an estimate that placement may adjust
within the requested geometric relation. Goal geometry and motion must both
pass. The core supports free SE(3), planar SE(2), fixed-orientation planar
translation, fixed-axis rotation, and fixed-direction translation.

To replay a saved grounding without loading models or calling an API:

```bash
PYTHONNOUSERSITE=1 python manipulate.py --prepared runs/drawer-close/observations.json \
  --output-dir runs/drawer-close-replay
```

Every new run writes `run.json`, grounded `observations.json` for the object path,
and `trajectory.json` containing fixed-body-to-reference 4×4 poses in metres,
entity IDs, frame conventions, model/goal checks and available interaction
evidence. These are object/part trajectories, not end-effector commands. The
offline `trajectory.png` and `coordinate_checks.json` visualize bounded
keyframes and verify the underlying point transformations without invented
function or grasp markers. A successful geometry run does not certify contact
forces, hidden surfaces, physical task completion or robot execution.

Run the analytic examples without models to inspect all five spaces, placement
and an open/place/close sequence:

```bash
PYTHONNOUSERSITE=1 MPLCONFIGDIR=/tmp/functo-mpl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python examples/manipulation_geometry.py --output-dir runs/manipulation-geometry
```

These examples use known synthetic geometry. Actual new task coverage still
requires its own demonstrated stage and scene; the bundled pouring example
alone cannot verify real drawer, door or container operation.
<!-- END task-adaptive object/part manipulation. -->
