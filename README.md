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
