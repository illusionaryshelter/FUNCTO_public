import os
from PIL import Image
import numpy as np
import cv2
import json
# BEGIN both transfer entry points use the common MARCO correspondence contract.
from pathlib import Path
from manipulation.grounding import match_points
# END common correspondence import.

from utils_IL.perception_utils import fit_gaussian, convert_to_serializable
from utils_IL.vp_utils import propose_candidate_keypoints, annotate_visual_prompts, load_prompts, generate_vertices_and_update, keypoint_transfer


# BEGIN default MARCO refinement; semantic-only mode remains explicit.
def grasp_point_transfer(data_path, test_data_path, test_tool_label='mug', test_target_label='bowl', task_label='pour', num_candidate_keypoints=8, feature_matching=True):
    # END local default.
    grasp_det_path = os.path.join(data_path, 'grasp_det_output')
    output_path = os.path.join(test_data_path, 'grasp_transfer_output')
    os.makedirs(output_path, exist_ok=True)

    # BEGIN repository-owned prompts do not depend on the output dataset path.
    prompt_path = Path(__file__).resolve().parent / 'utils_IL' / 'prompts'
    # END prompt path resolution.
    prompts = load_prompts(prompt_path)

    # load test initial keyframe
    test_init_frame_idx = 0
    test_init_frame_path = os.path.join(test_data_path, 'rgb', f'{test_init_frame_idx:05}.jpg')
    test_init_frame = cv2.imread(test_init_frame_path)
    demo_init_frame_pil_crop = Image.open(os.path.join(test_data_path, 'func_point_transfer_output', 'demo_tool_crop.png')).convert('RGB')
    demo_init_frame_crop = np.array(demo_init_frame_pil_crop)
    test_init_frame_pil_crop = Image.open(os.path.join(test_data_path, 'func_point_transfer_output', 'test_tool_crop.png')).convert('RGB')
    test_init_frame_mask_crop = np.load(os.path.join(test_data_path, 'func_point_transfer_output', 'test_tool_mask_crop.npy'))

    ############ Demo tool processing ############ 

    # load demo tool bbox
    demo_tool_box_path = os.path.join(data_path, 'detection_output', 'tool.json')
    with open(demo_tool_box_path, 'r') as file:
        demo_tool_box = json.load(file)
    x_min, y_min, x_max, y_max = demo_tool_box
    x_min *= 0.95
    y_min *= 0.95
    x_max *= 1.03
    y_max *= 1.03
    demo_tool_box = [x_min, y_min, x_max, y_max]

    # load demo grasp point
    grasp_point_path = os.path.join(grasp_det_path, 'grasp_points.json')
    with open(grasp_point_path, 'r') as json_file:
        grasp_points = json.load(json_file)
    # compute 2d demo grasp point
    mean, _ = fit_gaussian(grasp_points)
    x, y = int(mean[0]), int(mean[1])
    demo_grasp_point = [y - demo_tool_box[0],  x - demo_tool_box[1]]

    # save cropped demo grasp point
    with open(os.path.join(output_path, 'demo_init_frame_crop_grasp_point_out.json'), "w") as json_file:
        json.dump([int(demo_grasp_point[0]), int(demo_grasp_point[1])], json_file, default=convert_to_serializable, indent=4)
    scatter_size = 3
    cv2.circle(demo_init_frame_crop, (int(demo_grasp_point[0]), int(demo_grasp_point[1])), scatter_size, (0, 0, 255), -1)
    cv2.imwrite(os.path.join(output_path, 'demo_init_frame_grasp_point_vis.jpg'), demo_init_frame_crop)

    ############ Test tool processing ############ 

    # load test tool bbox
    test_tool_box_path = os.path.join(test_data_path, 'detection_output', 'tool.json')
    with open(test_tool_box_path, 'r') as file:
        test_tool_box = json.load(file)
    x_min, y_min, x_max, y_max = test_tool_box
    x_min *= 0.95
    y_min *= 0.95
    x_max *= 1.03
    y_max *= 1.03
    test_tool_box = [x_min, y_min, x_max, y_max]
            
    ############ Coarse-grained region proposal ############ 
    segmasks = {}
    segmasks[test_tool_label] = {'mask': test_init_frame_mask_crop}

    task_instruction = f'use a {test_tool_label} to {task_label} {test_target_label}'

    # ignore target objects
    task = {}
    task['object_grasped'] = test_tool_label
    task['object_unattached'] = ''
    task['task_instruction'] = task_instruction

    segmasks = generate_vertices_and_update(segmasks)

    # Annotate visual marks.
    candidate_keypoints = propose_candidate_keypoints(
        task,
        segmasks, 
        num_samples=num_candidate_keypoints)

    annotated_image = annotate_visual_prompts(
                test_init_frame_pil_crop,
                candidate_keypoints)
    annotated_image.save(os.path.join(output_path, 'visual_prompting_candidates.png'))
    
    print("zero-shot grasp point transfer")
    context, context_cor, vp_img = keypoint_transfer(
        task,
        test_init_frame_pil_crop,
        annotated_image,
        candidate_keypoints,
        prompts=prompts['select_motion_grasp'], 
        debug=True,
        grasp=True)
    
    print(task)
    print(context)

    vp_grasp_point = context_cor['selected_keypoint']
    vp_img.save(os.path.join(output_path, 'visual_prompting_result.png'))

    print("coarse-grained region proposal done.")

    x_offset, y_offset, _, _ = test_tool_box
    # BEGIN native MARCO refinement in original crop pixels, without a second resize.
    if feature_matching:
        source_mask = np.asarray(np.load(os.path.join(test_data_path, 'func_point_transfer_output', 'demo_tool_mask_crop.npy'), allow_pickle=False), dtype=bool)
        target_mask = np.asarray(test_init_frame_mask_crop, dtype=bool)
        yy, xx = np.indices(target_mask.shape)
        # Retain the semantic proposal's 15% crop-size window; MARCO searches
        # only the intersection with the observed object mask.
        cx, cy = vp_grasp_point
        candidates = target_mask & (np.abs(xx-cx) <= .075*test_init_frame_pil_crop.width)
        candidates &= np.abs(yy-cy) <= .075*test_init_frame_pil_crop.height
        points, diagnostic = match_points(
            demo_init_frame_pil_crop, test_init_frame_pil_crop,
            source_mask, candidates, [demo_grasp_point])
        with open(os.path.join(output_path, 'feature_matching.json'), 'w') as stream:
            json.dump(diagnostic, stream, indent=2)
        original_x, original_y = points[0]
    else:
        original_x, original_y = vp_grasp_point
    test_frame_x, test_frame_y = int(round(original_x + x_offset)), int(round(original_y + y_offset))
    # END model failure propagates; semantic-only mode is not an error fallback.

    cv2.circle(test_init_frame, (test_frame_x, test_frame_y), 3, (0, 0, 255), -1)
    cv2.imwrite(os.path.join(output_path, 'test_init_frame_grasp_point_vis.jpg'), cv2.cvtColor(test_init_frame, cv2.COLOR_BGR2RGB))

    with open(os.path.join(output_path, 'test_init_frame_grasp_point_out.json'), 'w') as json_file:
        json.dump([test_frame_x, test_frame_y], json_file, indent=4)

    print("grasp point transfer done.")



 
    
