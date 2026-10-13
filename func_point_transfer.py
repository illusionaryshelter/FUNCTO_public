
import os
from PIL import Image
import numpy as np
import cv2
import json
# BEGIN both transfer entry points use the common MARCO correspondence contract.
from pathlib import Path
from manipulation.grounding import match_points
# END common correspondence import.

from utils_IL.perception_utils import convert_to_serializable
from utils_IL.vp_utils import propose_candidate_keypoints, annotate_visual_prompts, annotate_candidate_keypoints, load_prompts, generate_vertices_and_update, keypoint_transfer

# BEGIN default MARCO refinement; semantic-only mode remains explicit.
def func_point_transfer(data_path, test_data_path, test_tool_label='mug', test_target_label='bowl', task_label='pour', num_candidate_keypoints=8, feature_matching=True):
    # END local default.
    """
    Function point transfer
    """

    output_path = os.path.join(test_data_path, 'func_point_transfer_output')
    os.makedirs(output_path, exist_ok=True)

    # BEGIN repository-owned prompts do not depend on the output dataset path.
    prompt_path = Path(__file__).resolve().parent / 'utils_IL' / 'prompts'
    # END prompt path resolution.
    prompts = load_prompts(prompt_path)

    # load demo function point in the initial keyframe
    demo_func_point_init_path = os.path.join(data_path, 'func_point_track_output', 'func_point_2d_init.json')
    with open(demo_func_point_init_path, 'r') as file:
        demo_func_point_init = json.load(file) 
    demo_func_point_init_x, demo_func_point_init_y = demo_func_point_init

    # load demo initial keyframe
    demo_init_frame_index = 0
    demo_init_frame_filenname = f'{demo_init_frame_index:05d}.jpg'
    demo_init_frame_file_path = os.path.join(data_path, 'rgb', demo_init_frame_filenname)
    demo_init_frame_pil = Image.open(demo_init_frame_file_path).convert('RGB')
    demo_init_frame_mask_path = os.path.join(data_path, 'detection_output', 'tool_mask.npy')
    demo_init_frame_mask = np.load(demo_init_frame_mask_path)

    # load test initial keyframe
    test_init_frame_idx = 0
    test_init_frame_filename = f'{test_init_frame_idx:05d}.jpg'
    test_init_frame_file_path = os.path.join(test_data_path, 'rgb', test_init_frame_filename)
    test_init_frame_pil =  Image.open(test_init_frame_file_path).convert('RGB')
    test_init_frame = np.array(test_init_frame_pil)
    test_init_frame_mask_path = os.path.join(test_data_path, 'detection_output', 'tool_mask.npy')
    test_init_frame_mask = np.load(test_init_frame_mask_path)

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

    # crop demo tool and plot function point
    demo_init_frame_pil_crop = demo_init_frame_pil.crop(demo_tool_box)
    demo_init_frame_pil_crop.save(os.path.join(output_path, 'demo_tool_crop.png'))
    demo_func_point_2d_init_x_crop, demo_func_point_2d_init_y_crop = int(demo_func_point_init_x - demo_tool_box[0]), int(demo_func_point_init_y - demo_tool_box[1])
    demo_func_point_2d_init_crop = [demo_func_point_2d_init_x_crop, demo_func_point_2d_init_y_crop]
    demo_annotate_keypoints = {'grasped': [demo_func_point_2d_init_crop]}
    demo_init_frame_pil_crop_annotated = annotate_candidate_keypoints(demo_init_frame_pil_crop, demo_annotate_keypoints, add_caption=False)
    # hack: flip for better global alignment
    demo_init_frame_pil_crop_annotated = demo_init_frame_pil_crop_annotated.transpose(Image.FLIP_LEFT_RIGHT)
    demo_init_frame_pil_crop_annotated.save(os.path.join(output_path, 'demo_tool_func_point_init_vis.jpg'))

    # crop demo tool mask
    demo_init_frame_mask_pil = Image.fromarray(demo_init_frame_mask)
    demo_init_frame_mask_pil_crop = demo_init_frame_mask_pil.crop(demo_tool_box)
    demo_init_frame_mask_crop = np.array(demo_init_frame_mask_pil_crop).astype(np.uint8)
    np.save(os.path.join(output_path, 'demo_tool_mask_crop.npy'), demo_init_frame_mask_crop)
    demo_init_frame_mask_pil_crop.save(os.path.join(output_path, 'demo_tool_mask_crop.png'))

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

    # crop test tool and mask
    test_init_frame_pil_crop = test_init_frame_pil.crop(test_tool_box)
    test_init_frame_pil_crop.save(os.path.join(output_path, 'test_tool_crop.png'))
    test_init_frame_mask = np.load(os.path.join(test_data_path, 'detection_output', 'tool_mask.npy'))
    test_init_frame_mask_pil = Image.fromarray(test_init_frame_mask)
    test_init_frame_mask_pil_crop = test_init_frame_mask_pil.crop(test_tool_box)
    test_init_frame_mask_crop = np.array(test_init_frame_mask_pil_crop).astype(np.uint8)
    np.save(os.path.join(output_path, 'test_tool_mask_crop.npy'), test_init_frame_mask_crop)
    test_init_frame_mask_pil_crop.save(os.path.join(output_path, 'test_tool_mask_crop.png'))

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
    
    print("in-context function point transfer")
    context, context_cor, vp_img = keypoint_transfer(
        task,
        test_init_frame_pil_crop,
        [demo_init_frame_pil_crop_annotated, annotated_image],
        candidate_keypoints,
        prompts=prompts['select_motion_func_demo'], 
        debug=True)
    
    print(task)
    print(context)

    vp_func_point = context_cor['selected_keypoint']
    vp_img.save(os.path.join(output_path, 'visual_prompting_result.png'))

    print("coarse-grained region proposal done.")

    x_offset, y_offset, _, _ = test_tool_box
    # BEGIN native MARCO refinement in original crop pixels, without a second resize.
    if feature_matching:
        source_mask = np.asarray(demo_init_frame_mask_crop, dtype=bool)
        target_mask = np.asarray(test_init_frame_mask_crop, dtype=bool)
        yy, xx = np.indices(target_mask.shape)
        # Retain the semantic proposal's 15% crop-size window; MARCO searches
        # only the intersection with the observed object mask.
        cx, cy = vp_func_point
        candidates = target_mask & (np.abs(xx-cx) <= .075*test_init_frame_pil_crop.width)
        candidates &= np.abs(yy-cy) <= .075*test_init_frame_pil_crop.height
        points, diagnostic = match_points(
            demo_init_frame_pil_crop, test_init_frame_pil_crop,
            source_mask, candidates, [demo_func_point_2d_init_crop])
        with open(os.path.join(output_path, 'feature_matching.json'), 'w') as stream:
            json.dump(diagnostic, stream, indent=2)
        original_x, original_y = points[0]
    else:
        original_x, original_y = vp_func_point
    test_init_frame_x, test_init_frame_y = int(round(original_x + x_offset)), int(round(original_y + y_offset))
    # END model failure propagates; semantic-only mode is not an error fallback.

    cv2.circle(test_init_frame, (test_init_frame_x, test_init_frame_y), 3, (0, 0, 255), -1)
    cv2.imwrite(os.path.join(output_path, 'test_init_frame_func_point_vis.jpg'), test_init_frame)

    with open(os.path.join(output_path, 'test_init_frame_func_point_out.json'), "w") as json_file:
        json.dump([test_init_frame_x, test_init_frame_y], json_file, default=convert_to_serializable, indent=4)

    print("function point transfer done.")
