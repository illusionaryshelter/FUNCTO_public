"""Native AnyGrasp prediction with the fixed official robot's aperture binding.

The fixed robot and installed gripper reuse their official collision shape and
component relations across tasks. Only current observed RGB-D points enter the
learned detector. Native predictions, voxel masks and sampled seed identities
return to the CPU before any following pose model or TSDF allocation begins.
"""
from argparse import Namespace
from pathlib import Path

import numpy as np

# BEGIN native model transport and the preserved official robot binding.
NATIVE_LABEL_ROTATIONS = np.stack((np.eye(3), np.diag([-1., -1., 1.])))

class AnyGraspBackend:
    """Bind fixed official geometry and launch one detector lifetime per call.

    ``gripper_geometry`` is the existing official ``grasp_generation_geometry``
    record; ``tool_geometry`` is the same model's public geometry object. The
    robot and installed gripper are fixed across tasks, so their official shape
    and component relations are reusable. Object geometry enters only as the
    current RGB-D points supplied to ``generate_and_rank``.
    """

    def __init__(self, *, checkpoint_path, gripper_geometry, tool_geometry):
        self.T_tool_grasp = np.asarray(gripper_geometry['T_tool_grasp']).copy()
        self.geometry_source = dict(gripper_geometry['source'])
        self.open_joints = np.asarray(tool_geometry.open_joint_positions_m).copy()
        self.closed_joints = np.asarray(tool_geometry.closed_joint_positions_m).copy()
        opened = tool_geometry.at(self.open_joints)
        closed = tool_geometry.at(self.closed_joints)
        self.open_gap = float(opened['signed_gap_m'])
        self.closed_gap = float(closed['signed_gap_m'])
        inverse_binding = np.linalg.inv(self.T_tool_grasp)
        poses = {row['geom_id']: row['T_tool_geom'] for row in opened['geoms']}
        finger_points = []
        for part in tool_geometry.collision_geometries:
            if part['role'] == 'finger':
                transform = inverse_binding @ poses[part['geom_id']]
                finger_points.append(part['vertices'] @ transform[:3, :3].T + transform[:3, 3])
        sweep = gripper_geometry['sweep_volume_params']
        self.tip_anchor_grasp = np.asarray(sweep['offset_open'], dtype=float).copy()
        # Native depth locates the leading finger plane, not the pad centre.
        # Keeping the official pad midpoint's transverse coordinates defines
        # the central point of that plane without selecting a mesh vertex.
        self.tip_anchor_grasp[2] = np.concatenate(finger_points)[:, 2].max()
        self.geometry_source.update(
            native_tip_binding='official_finger_leading_support_plane_centre',
            tip_anchor_grasp_m=self.tip_anchor_grasp.copy(),
            pad_center_grasp_m=np.asarray(sweep['offset_open']).copy(),
            pad_leading_depth_grasp_m=float(sweep['offset_open'][2] + sweep['extents_open'][2]/2),
            open_pad_gap_m=self.open_gap, closed_pad_gap_m=self.closed_gap,
            finger_joint_names=list(tool_geometry.joint_names),
            native_label_rotations_grasp=NATIVE_LABEL_ROTATIONS.copy())
        # Native widths are float32. Declare the greatest representable bound
        # within the official aperture so decoded widths map to real joint
        # limits without resizing a candidate or expanding those limits.
        native_max_gap = np.float32(self.open_gap)
        if float(native_max_gap) > self.open_gap:
            native_max_gap = np.nextafter(native_max_gap, np.float32(0.))
        config = Namespace(checkpoint_path=str(Path(checkpoint_path).resolve()),
            max_gripper_width=float(native_max_gap),
            gripper_height=float(sweep['extents_open'][1]))
        self.config = vars(config)
        self.device = None
        self.model_sources = dict(
            checkpoint_path=config.checkpoint_path,
            backend='anygrasp_sdk.create_detector',
            weight_source='existing_local_release',
            native_roll_period_rad=float(np.pi),
            score_domain='native_parallel_gripper_pi_label_class',
            configuration=vars(config))

    def predict(self, points_camera, controls, *, seed):
        """Run the SDK once in the shared bounded worker and retain raw rows."""
        from .runtime import run_array_worker

        arrays = run_array_worker("anygrasp", dict(
            points_camera_m=points_camera, region_mask=controls["region_steering"],
            approach_camera=controls["approach_steering"]), dict(
            **self.config, seed=int(seed), approach_thresh=float(controls["approach_thresh"])))
        observed = bool(arrays["model_forward_observed"])
        self.prepared_points_camera = arrays["prepared_points_camera_m"] if observed else None
        self.prepared_region_mask = arrays["model_region_mask"] if observed else None
        self.sampled_seed_indices = arrays["model_seed_indices"] if observed else None
        self.device = str(arrays["device"])
        return arrays["native_rows"]

    def joint_positions_for_widths(self, widths):
        """Map physical inner-pad separation to the declared opposed slides.

        The model has two opposing prismatic joints. Their declared open and
        closed configurations determine the affine aperture law; this is a
        robot-model relation, not a task-specific gripper command or force law.
        """
        fraction = (np.asarray(widths, dtype=float)-self.closed_gap)/(self.open_gap-self.closed_gap)
        return self.closed_joints + fraction[:, None]*(self.open_joints-self.closed_joints)

def worker_main(arrays, output, *, checkpoint_path, max_gripper_width, gripper_height,
                seed, approach_thresh):
    """Expose the complete native SDK rows and actual network-input evidence."""
    import sys
    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from anygrasp_sdk import create_detector

    config = Namespace(checkpoint_path=checkpoint_path,
        max_gripper_width=max_gripper_width, gripper_height=gripper_height)
    detector = create_detector(config)
    observed = {}

    def remember_cloud(model, args):
        observed["points"] = args[0]["point_clouds"][0].detach().cpu().numpy().copy()
        region = args[2]
        observed["region"] = region.detach().cpu().numpy().reshape(-1).copy()

    def remember_sampling(model, args, prediction):
        observed["seeds"] = prediction["stage2_seed_inds"].detach().cpu().numpy().reshape(-1).copy()

    detector.net.register_forward_pre_hook(remember_cloud)
    detector.net.register_forward_hook(remember_sampling)
    np.random.seed(seed)
    torch.manual_seed(seed)
    controls = dict(dense_grasp=False, collision_detection=False,
        region_steering=arrays["region_mask"], approach_steering=arrays["approach_camera"],
        approach_thresh=float(approach_thresh))
    native = detector.get_grasp(arrays["points_camera_m"], controls)
    forwarded = "points" in observed
    np.savez(output,
        native_rows=np.empty((0, 17)) if native is None else np.asarray(native.grasp_group_array),
        model_forward_observed=np.asarray(forwarded),
        prepared_points_camera_m=observed["points"] if forwarded else np.empty((0, 3), np.float32),
        model_region_mask=observed["region"] if forwarded else np.empty(0, bool),
        model_seed_indices=observed["seeds"] if forwarded else np.empty(0, np.int64),
        device=np.asarray(str(next(detector.net.parameters()).device)))
# END native model transport and the preserved official robot binding.
