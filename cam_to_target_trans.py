import os
import numpy as np
import open3d as o3d
import cv2
from PIL import Image
import json

# BEGIN local SAM 3.1 integration: one model supplies both boxes and masks.
from local_models.sam31 import detect
# END local SAM 3.1 integration.
from utils_IL.geometry_utils import compute_point_cloud
from utils_IL.params import CAM_MATRIX

# BEGIN local SAM 3.1 integration: preserve the detection_output file contract.
def demo_test_deteciton(data_path, tool_label, target_label):
    """Detect both roles locally and save each role's highest-scoring instance.

    The model layer returns all instances without assigning task-specific roles.
    This pipeline selects one tool and one target, retaining the pixel xyxy JSON,
    boolean NPY mask, binary PNG mask, and annotated image used downstream.
    """
    output_path = os.path.join(data_path, 'detection_output')
    os.makedirs(output_path, exist_ok=True)
    initial_frame_file_path = os.path.join(data_path, 'rgb', '00000.jpg')
    with Image.open(initial_frame_file_path) as opened:
        initial_frame_pil = opened.convert('RGB')
    detected_objects = detect(initial_frame_pil, [tool_label, target_label])

    # Resolve every required role before writing files, so missing detections
    # fail explicitly instead of producing an apparently complete partial run.
    selected_objects = {}
    for role, label in [('tool', tool_label), ('target', target_label)]:
        candidates = [obj for obj in detected_objects if obj['label'] == label]
        if not candidates:
            raise RuntimeError(f"SAM 3.1 detected no {role} for text label {label!r}")
        selected_objects[role] = max(candidates, key=lambda obj: obj['score'])

    for role, detection in selected_objects.items():
        box = detection['bbox']
        mask = detection['mask']
        annotated_frame = np.array(initial_frame_pil)
        top_left = (int(box[0]), int(box[1]))
        bottom_right = (int(box[2]), int(box[3]))
        cv2.rectangle(annotated_frame, top_left, bottom_right, (255, 0, 0), 2)
        cv2.putText(
            annotated_frame, detection['label'], top_left,
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA,
        )
        output_image_path = os.path.join(output_path, f'{role}_bbox.png')
        if not cv2.imwrite(output_image_path, cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)):
            raise OSError(f"Could not save detection image: {output_image_path}")
        with open(os.path.join(output_path, f'{role}.json'), 'w') as json_file:
            json.dump(box, json_file, indent=4)
        Image.fromarray(mask.astype(np.uint8) * 255).save(
            os.path.join(output_path, f'{role}_mask.png')
        )
        np.save(os.path.join(output_path, f'{role}_mask.npy'), mask)
        print(f"SAM 3.1 {role} detection saved in {output_path}")
    return None
# END local SAM 3.1 integration.

def cam_to_target_trans(data_path):

    """"
    transform from camera to target object frame
    """
    output_path = os.path.join(data_path, 'cam_to_target_trans_output')
    os.makedirs(output_path, exist_ok=True)

    initial_frame_index = 0
    initial_frame_filenname = f'{initial_frame_index:05d}.jpg'
    initial_frame_file_path = os.path.join(data_path, 'rgb', initial_frame_filenname)
    initial_frame_depth_filename = f'{initial_frame_index:05d}.png'
    initial_frame_depth_file_path = os.path.join(data_path, 'depth', initial_frame_depth_filename)

    initial_frame_pil = Image.open(initial_frame_file_path).convert('RGB')
    initial_frame_depth_pil = Image.open(initial_frame_depth_file_path)
    initial_frame = np.array(initial_frame_pil)
    initial_frame_depth = np.array(initial_frame_depth_pil)

    ############ 3D target object detection ############
    # reconstruct the scene point cloud
    # mask = (initial_frame_depth == 0).astype(np.uint8)
    initial_frame_depth = initial_frame_depth.astype(np.float32)
    points, colors, _ = compute_point_cloud(initial_frame, initial_frame_depth, CAM_MATRIX, [])
    scene_point_cloud = o3d.geometry.PointCloud()
    scene_point_cloud.points = o3d.utility.Vector3dVector(points)
    scene_point_cloud.colors = o3d.utility.Vector3dVector(colors)

     # plane fitting
    plane_model, inliers = scene_point_cloud.segment_plane(distance_threshold=0.01,
                                                 ransac_n=3,
                                                 num_iterations=1000)
    [a, b, c, d] = plane_model
    print(f"plane equation: {a}x + {b}y + {c}z + {d} = 0")
    normal_vector = np.array([a, b, c])
    normal_vector /= np.linalg.norm(normal_vector)  # the normal vector goes down
    z_axis = np.array([0, 0, -1])  # Z-axis vector
    rotation_axis = np.cross(normal_vector, z_axis)
    rotation_axis /= np.linalg.norm(rotation_axis)
    angle = np.arccos(np.dot(normal_vector, z_axis))
    rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_axis * angle)

    # rotate the scene point cloud for visualization
    rotated_scene_point_cloud = scene_point_cloud.rotate(rotation_matrix, center=[0, 0, 0])

    # get target object point cloud
    target_mask_path = os.path.join(data_path, 'detection_output', 'target_mask.npy')
    target_mask = np.load(target_mask_path)
    target_mask_uint8 = target_mask.astype(np.uint8)
    masked_initial_frame = cv2.bitwise_and(initial_frame, initial_frame, mask=target_mask_uint8)
    masked_initial_frame_depth = cv2.bitwise_and(initial_frame_depth, initial_frame_depth, mask=target_mask_uint8)
    points, colors, points_3d = compute_point_cloud(masked_initial_frame, masked_initial_frame_depth, CAM_MATRIX, [])
    target_point_cloud = o3d.geometry.PointCloud()
    target_point_cloud.points = o3d.utility.Vector3dVector(points)
    target_point_cloud.colors = o3d.utility.Vector3dVector(colors)

    # remove target object outliers and rotate
    cl, ind = target_point_cloud.remove_statistical_outlier(nb_neighbors=100, std_ratio=0.75)
    filtered_target_point_cloud = target_point_cloud.select_by_index(ind)
    rotated_target_point_cloud = filtered_target_point_cloud.rotate(rotation_matrix, center=[0, 0, 0])

    # get axis-aligned bounding box and compute the center point
    aabb = rotated_target_point_cloud.get_axis_aligned_bounding_box()
    box_points = np.asarray(aabb.get_box_points())
    box_dim = aabb.get_max_bound()- aabb.get_min_bound()
    center_point = np.mean(box_points, axis=0)
    translated_box_points = box_points - center_point

    # move the origin to the target object
    rotated_scene_point_cloud_pts = np.asarray(rotated_scene_point_cloud.points)
    rotated_scene_point_cloud_pts -= center_point
    rotated_scene_point_cloud.points = o3d.utility.Vector3dVector(rotated_scene_point_cloud_pts)
    o3d.io.write_point_cloud(os.path.join(output_path, 'transformed_scene_point_cloud.ply'), rotated_scene_point_cloud)
    rotated_target_point_cloud_pts = np.asarray(rotated_target_point_cloud.points)
    rotated_target_point_cloud_pts -= center_point
    rotated_target_point_cloud.points = o3d.utility.Vector3dVector(rotated_target_point_cloud_pts)
    print("target object frame transformation done.")

    # save rotation matrix, rotattion center, centroid, box dim, and box points
    cam_to_target_trans = {}
    cam_to_target_trans['rotation_matrix'] = rotation_matrix.tolist()
    cam_to_target_trans['rotation_center'] = [0, 0, 0]
    cam_to_target_trans['centroid'] = center_point.tolist()
    cam_to_target_trans['target_dim'] = box_dim.tolist()
    cam_to_target_trans['box_points'] = translated_box_points.tolist()

    with open(os.path.join(output_path, 'cam_to_target_trans.json' ), 'w') as json_file:
        json.dump(cam_to_target_trans, json_file, indent=4)

    ########### 3D target object detection visualization ############
    spheres = []
    for point in translated_box_points:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)  # Adjust the radius as needed
        sphere.translate(point)
        sphere.paint_uniform_color([0, 1, 0])  # Green color
        spheres.append(sphere)
    # Combine the transformed target object point cloud and the spheres
    spheres_combined = o3d.geometry.PointCloud()
    for sphere in spheres:
        sphere_pc = sphere.sample_points_poisson_disk(100)
        spheres_combined += sphere_pc
    combined_point_cloud = rotated_target_point_cloud + spheres_combined
    o3d.io.write_point_cloud(os.path.join(output_path, 'target_point_cloud_with_aabb_corners.ply'), combined_point_cloud)

    return None

 
