# Multi-view interaction grounding — synthetic validation harness

## What this establishes

The scripts test **inference-time** grounding with a **single right-camera source point cloud** and **two calibrated target RGB-D cameras**. Left/right 3D feature correspondences remain separate and are used to localize one target interaction reference point. There is no CAD, mesh reconstruction, custom training, or averaged multiview descriptor.

The 64 recorded trials use synthetic point-sampled shelves/open drawers, camera z-buffers, viewpoint-specific foreground occluders, and a **synthetic MARCO-like sparse correspondence error generator**. The code DOES NOT run actual MARCO inference and DOES NOT measure real-world placement success. Source points unseen from the right are not given to any inference solver. Ground-truth geometries/IDs are used ONLY inside the synthetic data generator and evaluator.

## Files

- `multiview_grounding_experiment.py` — reproducible scene simulation, two-view matching errors, source-capped distance-vote grounding, robust full-rank affine grounding, and benchmark runner.
- `test_multiview_grounding_geometry.py` — visibility-state, anisotropic affine, metric sampling, and missing-observation tests.
- `multiview_grounding_full_results.csv` — all 64 cases x tested methods.
- `multiview_grounding_summary.json` — aggregate metrics.
- `multiview_grounding_final.png` — per-scenario success chart.

Run (Python with `numpy`, `scipy`, `matplotlib`):

```bash
OPENBLAS_NUM_THREADS=1 python multiview_grounding_experiment.py --trials 8 --output mv_results
OPENBLAS_NUM_THREADS=1 python test_multiview_grounding_geometry.py
```

The run used 8 scenarios x 8 seeds; the runner accepts `--scenario-indices 0,1` to split large runs. Some machines may need a longer runtime limit for `--trials 8` with all scenarios.

## Grounding input contract for real MARCO integration

- `src_xyz`: **source observed right-camera** points in meters, shape `[N,3]`.
- `src_region_xyz` or mask: demonstration interaction surface in that observed source cloud.
- `target_cloud_L`, `target_cloud_R`: **independent SAM-filtered** target RGB-D point sets transformed to the **same calibrated robot/world frame**.
- `matches_L`, `matches_R`: independent MARCO source-target associations: source point index, 3D target point in common world coordinates, matching confidence, original pixel indices, depth validity. Keep the source point index shared between both views.
- `K_L,K_R`, camera-to-world poses, original (not SAM-masked) depth maps, depth uncertainty, SAM masks, and timestamps for visibility and collision decisions.
- Object footprint/dimensions in **physical meters**, plus world gravity, for feasible-rotation-dependent placement sampling.

Do NOT pass synthetic scene full points, oracle correspondences, or GT target location to inference.

## Intended production grounding (requires validation with user's real MARCO data)

1. Cache source right-view MARCO features and relative interaction center/normal.
2. Acquire *time-synchronized* left/right images, target SAM masks, depths, and calibration.
3. Compute left/right MARCO maps in parallel/batch; independently match source sparse anchors to each target view. Carry multiple candidate matches rather than only top-1.
4. Lift valid matches to 3D in a common frame, reject depth discontinuities, score cross-view 3D agreement and correspondence spatial coverage. Cap evidence once per source anchor.
5. Generate geometric hypotheses: robust similarity and regularized, bounded-anisotropy affine alignment; plus an invariant anchor-distance grounding fallback. Retain distinct modes.
6. Rank with feature correspondence inlier support, model complexity, anchor spread, independent view support, 3D support surface, and visibility. **Unknown/occluded pixels are neutral; measured free space contradicting a surface receives a penalty.**
7. If ambiguous or unsupported, abstain; do not invent a hidden drawer surface. If an opened drawer remains invisible in both views, output insufficient observation.
8. Recover a *reference point*, not a copied polygon. Use metric-size placement samples truncated by visible support geometry eroded by the block footprint and by observed occupied/free/unknown space, with access constraints.
9. Evaluate reference localization, accepted vs rejected cases, Top-K feasible placement success, calibration degradation, performance P50/P95, and failure modes on real source-right / target-left+right RGB-D recordings.

The standalone synthetic evaluator exercises the **grounding core** and selected visibility semantics. It does not implement image models, robotic grasp execution, robot collision planning, full support/occupancy checking, or statistically calibrated confidence. Its computed latency excludes these stages and MUST NOT be reported as full system latency.
