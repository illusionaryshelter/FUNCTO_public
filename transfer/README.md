# Observed-region transfer

The supported transfer schemes are:

| Scheme | Native registration and candidate search | Entry point |
|---|---|---|
| Single target view | TEASER++ proposal plus the existing six/twelve-point joint candidates | `python -m transfer.joint_experiment` |
| Two target views | TEASER++, joint observed surface, six/twelve-point candidates and fixed-domain observed-evidence scoring | `python -m transfer.dual_view.experiment` |

Both transfer a demonstrated location preference to actual observed target
surfaces, then sample nearby visible points. The source demonstration mask is
evidence; its outline is not copied to the target. Sampling radius and spacing
are physical lengths and are not multiplied by the estimated object scale.
Output acceptance concerns observed localization evidence. It does not certify
object fit, collision freedom, stability, reachability or an accurate full pose.

The inference code accepts observed arrays and calibration, never target region
labels, joint models, object names or hidden geometry. Task identifiers group
input records and reporting; they do not select numerical rules. Evaluation is
a separate process after predictions have been frozen. Robot planning and
execution interfaces are outside this package.

## Single-view scheme

`joint_features.source_queries` selects source demonstration and entity-context
queries. `appearance_evidence` retains multiple MARCO response peaks and the
native decoder's observed pixel. `joint_region.infer_joint_region` evaluates
six- and twelve-point distance-consistent proposals together with the admitted
TEASER++ native transform. It preserves the existing partial injective
assignment, fixed-support similarity refit, geometric gates and local sampler.

The first valid candidate in the common score order is primary; competing
surfaces and warnings remain visible. Independently processed target views can
be compared by `select_observation` only when their source evidence is identical.
This is distinct from reconstructing a joint surface from two target views.

`joint_experiment` runs only this selected scheme. It writes `joint.json`,
`joint.npz`, raw pairs, observed evidence and explicit missing-input records.
Its diagnostics identify `native_backend="teaserpp"`. The six/twelve-point
candidate rules live in `joint_region.py` and are independent of the driver.

## Dual-view scheme

`dual_view.pipeline.run` computes these stages from current resident inputs:

1. `surface.py` converts two complete calibrated depth images into actual
   surface elements, original-pixel provenance, oriented planes and adjacency.
   Exact CPU mutual-nearest searches overlap CUDA local geometry. Each element
   retains an actual measurement; missing entity features do not erase depth.
2. `geometry.prepare_source` prepares source queries, independent check points
   and demonstration geometry. Source and target local planes use the same
   checked CUDA Jacobi eigensolver.
3. `pairs.native_region_pairs` computes current MARCO correspondences using
   eight CPU Torch threads and invocation-local Gaussian/grid reuse. It retains
   original query batching, decoding and observed-pixel identities.
4. `io.appearance_field` constructs per-view response maps. `proposals.rank_domain`
   builds the fixed target rankings and empirical response references.
5. `io.native_transforms` fits TEASER++ from the newly computed raw pairs.
   `proposals.build_proposals` adds the complete declared six/twelve-point
   candidate set. CUDA handles correspondence growth; original CPU random
   calls, boundary certification and similarity fits remain explicit.
6. `candidate_events` prepares sixteen parent candidates per batch. CUDA
   projection and evidence evaluation feed the unchanged CPU refits, capacity
   assignment, geometry decisions and observed-region sampler. Candidate and
   interruption events retain their original commit order.

`inference.infer` is the candidate-stage API. `pipeline.run` is the complete
resident-input API and returns the final result together with the actual
computed surface, source, pairs, response field and ranking domain. These
intermediate outputs support persistence and auditing; no historical prediction
is an input to a fresh call.

```python
from transfer.dual_view import io, pipeline
from transfer.multiview_experiment import read_manifest

index, description = read_manifest("paired-index.json")
task = description["tasks"][0]
loaded = io.load_observations(index, task)
features = io.load_feature_pairs(loaded)
pipeline.prepare()  # Installed native code, CUDA setup and memory cap.
outputs = pipeline.run(loaded, features)
result = outputs["result"]
```

The input manifest supplies `data` and paired `tasks`, each with two `cases`.
Each prepared case points to observed source/target arrays, RGB crops and MARCO
feature pairs. Full depth, camera intrinsics and world-from-camera transforms
come from the observation manifest. The two cases must retain identical source
arrays and model-input pixels. An unavailable target entity/feature observation
has an explicit reason; that view's complete depth still contributes geometry.

All observation uploads and region reuse belong to one inference call. The
region cache is bounded to 64 MiB and requires identical normal-eligibility
inputs before reusing a sampled region. CUDA failures raise or produce the
existing explicit interrupted-search state; they never select another backend.

`accepted`, decision status, rejection reasons, quality warnings and diagnostic
regions are separate outputs. A finite candidate winner is not a global
optimality or scale/rotation certificate. Missing appearance, unknown geometry,
conflicts, ambiguity and incomplete searches remain visible. Masks and samples
index `surface['xyz']`; the original observation-to-element mapping is retained.

## Commands and runtime

Run from `FUNCTO_public` in the existing recovery environment. Output directories
must be fresh. No package installation or third-party source changes are needed.

```bash
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/shelter/miniforge3/envs/recovery/bin/python -m transfer.joint_experiment \
  --index SINGLE_VIEW_INDEX.json --output SINGLE_VIEW_RESULTS

PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/shelter/miniforge3/envs/recovery/bin/python -m transfer.dual_view.experiment \
  --index PAIRED_INDEX.json --output DUAL_VIEW_RESULTS
```

The dual-view driver fixes TEASER++ and computes fresh surfaces and pairs. It
does not accept an old prediction cache, a surface cache or alternate native
backends. Native C++/CUDA sources are packaged under `dual_view`; generated
build products are ignored by Git. Runtime requires the already installed
TEASER++, Torch/CUDA and compiler dependencies and fails explicitly if absent.

`pipeline.prepare` caps the Torch allocator at the smaller of 6 GiB and half
the device memory, disables TF32 and builds extensions serially. The process
driver checks resident-input RSS against 7 GiB. Full-process/device supervision
is still needed for large experiments because the allocator cap does not
include every native allocation. Run model extraction and inference serially.

The continuous online timer starts with resident observations and feature
pairs and includes all seven numerical stages, transfers and synchronization.
It excludes model execution, observation/crop preparation, source-query
selection, native-code preparation, file I/O and evaluation. It is not
camera-to-output latency.

Shared modules such as `core.py`, `sim3.py`, `registration.py`, `features.py` and
`multiview_geometry.py` provide numerical primitives and independent historical
experiments. Their older experiment entry points do not change the two selected
schemes above. Historical artifacts remain evidence of their recorded code;
experimental scripts that rewrite old function bodies require that recorded
source revision for replay.
