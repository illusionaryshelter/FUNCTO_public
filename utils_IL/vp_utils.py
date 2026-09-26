import numpy as np
import os
import io
import json
import cv2
from PIL import Image
import re
from scipy.spatial import distance
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects

# BEGIN Gemini migration: importing visual prompting no longer needs credentials.
from local_models.gemini import Gemini
# END Gemini migration.

def load_prompts(prompt_dir):
    """
    Load prompts from files.
    """
    prompts = dict()
    for filename in os.listdir(prompt_dir):
        path = os.path.join(prompt_dir, filename)
        if os.path.isfile(path) and path[-4:] == '.txt':
            with open(path, 'r') as f:
                value = f.read()
            key = filename[:-4]
            prompts[key] = value
    return prompts

def generate_vertices_and_update(segmasks):
    for obj_name, obj_data in segmasks.items():
        mask = obj_data['mask'].astype(np.uint8)
        
        # Find contours using OpenCV
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Collect vertices
        obj_vertices = []
        for contour in contours:
            for point in contour:
                obj_vertices.append(point[0])
        
        # Update the dictionary with vertices
        segmasks[obj_name]['vertices'] = np.array(obj_vertices)
    
    return segmasks

def point_2_object_dis(vertices, mask):
    """
    compute the distance distribution of candidate function points
    """
    vertices_dis_list = []
    mask_points = np.argwhere(mask)
    for ver in vertices:
        distances = distance.cdist([ver], mask_points, metric='euclidean')
        ver_dis = np.sum(distances)
        vertices_dis_list.append(ver_dis)
    
    vertices_dis_array = np.array(vertices_dis_list)
    mean_value = np.mean(vertices_dis_array)
    std_value = np.std(vertices_dis_array)
    normalized_vertices_dis_array = (vertices_dis_array - mean_value) / std_value

    # apply softmax to get a distribution
    exp_values = np.exp(normalized_vertices_dis_array)
    softmax_values = exp_values / np.sum(exp_values)
    
    return softmax_values

def fps(points, n_samples):
    """
    Farthest Point Sampling (FPS) with inclusion of the leftmost and rightmost points.
    
    Args:
        points: [N, 2] array containing the whole point cloud.
        n_samples: Number of points to sample, including the leftmost and rightmost points.
    
    Returns:
        Sampled points: [n_samples, 2] array of the sampled point cloud.
    """
    points = np.array(points)

    # Find indices of the leftmost and rightmost points
    leftmost_idx = np.argmin(points[:, 0])
    rightmost_idx = np.argmax(points[:, 0])

    # Initialize sampled indices with the leftmost and rightmost points
    sample_inds = [rightmost_idx, leftmost_idx]
    
    # Reduce the number of points to sample since we've already included two
    remaining_samples = n_samples - 2

    # Represent the points by their indices in points
    points_left = np.delete(np.arange(len(points)), sample_inds)

    # Initialize distances to inf
    dists = np.ones(len(points_left)) * float("inf")

    # Iteratively select the remaining points
    for _ in range(remaining_samples):
        # Calculate distances from the last added sampled point
        last_added = sample_inds[-1]
        dist_to_last_added = ((points[last_added] - points[points_left]) ** 2).sum(axis=1)

        # Update distances to reflect the nearest sampled point
        dists = np.minimum(dists, dist_to_last_added)

        # Select the point with the maximum distance
        farthest_idx = np.argmax(dists)
        sample_inds.append(points_left[farthest_idx])

        # Remove the selected point from points_left and corresponding distance
        points_left = np.delete(points_left, farthest_idx)
        dists = np.delete(dists, farthest_idx)

    # Return the sampled points
    return points[sample_inds]

def get_keypoints_from_segmentation(segmasks, subtask, num_samples=5, include_center=True, dis_thresh=0.1):
    """
    Args:
        mask: a dict of segmentation masks, each mask is a dict with keys: 'mask', 'bbox', 'score'
        objects: a list of objects, each string is a query
    """
    object_vertices = {}
    for object_name in segmasks.keys():
        vertices = segmasks[object_name]["vertices"]
        mask = segmasks[object_name]["mask"]

        center_point = vertices.mean(0)
        # if mask[int(center_point[1])][
        #     int(center_point[0])
        # ] and include_center:  # ignore if geometric mean is not in mask
        if include_center:
            vertices = np.concatenate([center_point[None, ...], vertices], axis=0)

        if vertices.shape[0] > num_samples:
            kps = fps(vertices, num_samples)
        else:
            kps = vertices

        kps = np.concatenate([kps[[0]], kps[1:][kps[1:, 1].argsort()]], axis=0)
        object_vertices[object_name] = kps

    tool_kps = object_vertices[subtask['object_grasped']]
    if subtask['object_unattached'] != '':
        target_mask = segmasks[subtask['object_unattached']]['mask']
        kps_dis_dist = point_2_object_dis(tool_kps, target_mask)

        # remove points far away
        mask = kps_dis_dist <= dis_thresh
        filtered_tool_kps = tool_kps[mask]
    else:
        filtered_tool_kps = tool_kps
    object_vertices[subtask['object_grasped']] = filtered_tool_kps

    return object_vertices

def propose_candidate_keypoints(subtask, segmasks, num_samples):
    """
    Propose candidate keypoints for object_grasped and object_unattached.
    """
    input_object_names = []
    if subtask['object_grasped'] != '':
        input_object_names.append(subtask['object_grasped'])

    if subtask['object_unattached'] != '':
        input_object_names.append(subtask['object_unattached'])

    object_vertices = get_keypoints_from_segmentation(
        segmasks,
        subtask,
        num_samples=num_samples,
        include_center=False)

    candidate_keypoints = {
        'grasped': None,
        'unattached': None,
    }

    if subtask['object_grasped'] != '':
        candidate_keypoints['grasped'] = object_vertices[
            subtask['object_grasped']
        ]

    return candidate_keypoints

def plot_keypoints(
        ax, image_size, keypoints, color, prefix='', annotate_index=True,
        add_caption=True):
    if keypoints is None:
        return
        
    (h, w) = image_size
    for i, keypoint in enumerate(keypoints):
        if keypoint is None:
            continue

        ax.plot(
            keypoint[0], keypoint[1],
            color=color, alpha=0.4,
            marker='o', markersize=15,
            markeredgewidth=2, markeredgecolor='black',
        )

        if add_caption:
            text = ''
            if annotate_index:
                text = text + str(i + 1)
            text = prefix + text

            xytext = (
                min(max(30, keypoint[0]), h - 30),
                min(max(30, keypoint[1]), w - 30),
            )

            if prefix == 'P':
                # ax.annotate(text, keypoint, xytext, size=15, color='white')
                annotation = ax.annotate(text, keypoint, xytext, size=15)
                annotation.set_path_effects([
                path_effects.Stroke(linewidth=3, foreground='white'),
                path_effects.Normal()])
            else:
                ax.annotate(text, keypoint, xytext, size=15, color='white')
            # ax.annotate(text, keypoint, xytext, size=15)

def annotate_candidate_keypoints(
        image,
        candidate_keypoints,
        add_caption=True
):
    fig, ax = plt.subplots(1, 1)
    ax.imshow(image)
    ax.axis('off')

    image_size = image.size[:2]

    if 'grasped' in candidate_keypoints.keys():
        plot_keypoints(
            ax, image_size, candidate_keypoints['grasped'], 'r', prefix='P', add_caption=add_caption)

    if 'unattached' in candidate_keypoints.keys():
        plot_keypoints(
            ax, image_size, candidate_keypoints['unattached'], 'b', prefix='Q', add_caption=add_caption)

    buf = io.BytesIO()
    # fig.savefig(buf, transparent=True, bbox_inches='tight',
    #             pad_inches=0, format='jpg', dpi=int(image_size[0] / fig.get_size_inches()[0]))
    fig.savefig(buf, transparent=True, bbox_inches='tight',
                pad_inches=0, format='jpg', dpi=int(image_size[0]))
    buf.seek(0)
    # close the figure to prevent it from being displayed
    plt.close(fig)
    return Image.open(buf)

def annotate_visual_prompts(
        obs_image,
        candidate_keypoints,
        log_dir=None,
):
    """
    nnotate the visual prompts on the image.
    """

    annotated_image = annotate_candidate_keypoints(
        obs_image,
        candidate_keypoints,
    )

    if log_dir is not None:
        annotated_image.save(os.path.join(log_dir, 'keypoints.png'))
    # BEGIN headless pipeline: the caller saves/inspects the returned image.
    # Do not create an extra blocking GUI figure when log_dir is omitted.
    # END headless pipeline.

    return annotated_image

# BEGIN Gemini semantic selection: preserve prompts/images, validate returned
# candidate IDs, and surface errors instead of retrying malformed JSON forever.
def _request_context(message, images, meta_prompt, example_images=None,
                     example_responses=None):
    """Send a semantic selection through the authorized Gemini gateway.

    Images retain their supplied order. Optional demonstrations precede the
    current observation; the text explicitly associates each example response
    with its image so a multi-image request remains unambiguous.
    """
    images = images if isinstance(images, list) else [images]
    if any(not isinstance(image, Image.Image) for image in images):
        raise TypeError("Visual prompting requires PIL images")
    text = "\n".join(message) if isinstance(message, list) else message
    if example_images is not None or example_responses is not None:
        if example_images is None or example_responses is None:
            raise ValueError("Example images and responses must be supplied together")
        if len(example_images) != len(example_responses):
            raise ValueError("Each example image needs one response")
        examples = "\n".join(
            f"Example image {index + 1}: {response}"
            for index, response in enumerate(example_responses)
        )
        text = f"{examples}\nThe remaining images are the current request.\n{text}"
        images = list(example_images) + images
    return Gemini().chat_json(text, images, system_prompt=meta_prompt)


def _selected_keypoint(context, candidate_keypoints, grasp=False):
    """Map a displayed, one-based Pi to the exact candidate that Gemini saw."""
    field = 'grasp_keypoint' if grasp else 'function_keypoint'
    label = context.get(field)
    match = re.fullmatch(r'P([1-9][0-9]*)', label) if isinstance(label, str) else None
    points = candidate_keypoints['grasped']
    if match is None or not 1 <= int(match.group(1)) <= len(points):
        raise ValueError(f"Invalid Gemini {field}: {label!r}; expected P1..P{len(points)}")
    return points[int(match.group(1)) - 1]


def annotate_motion(image, context, log_dir=None, add_caption=True):
    """Return a saved visualization without opening a blocking GUI window."""
    fig, ax = plt.subplots(1, 1)
    ax.imshow(image)
    ax.axis('off')
    image_size = image.size[:2]
    if 'keypoints_2d' in context:
        function_keypoint = context['keypoints_2d']['function']
    else:
        function_keypoint = context['selected_keypoint']
    if function_keypoint is not None:
        plot_keypoints(ax, image_size, [function_keypoint], 'y', 'function', False,
                       add_caption=add_caption)
    with io.BytesIO() as buf:
        fig.savefig(buf, transparent=True, bbox_inches='tight', pad_inches=0, format='jpg')
        buf.seek(0)
        result = Image.open(buf).copy()
    plt.close(fig)
    return result


def request_motion(subtask, obs_image_reshaped, annotated_image, candidate_keypoints,
                   prompts, debug=False, example_images=None, example_responses=None,
                   loaded_context=None, use_center=False, log_dir=None, suffix='',
                   add_caption=True, grasp=False):
    """Generate a motion point with the same task-specific visual prompts."""
    selection = 'select_motion_grasp' if grasp else 'select_motion_func'
    context_json = loaded_context
    if context_json is None:
        context_json = _request_context(
            [f"Task: {subtask}"], annotated_image, prompts[selection],
            example_images, example_responses)
    # An empty grasped object legitimately has no selected point, as specified
    # by the existing motion prompts. Do not turn that valid case into an index
    # error; a nonempty object still requires a validated visible candidate.
    field = 'grasp_keypoint' if grasp else 'function_keypoint'
    if not subtask.get('object_grasped') and context_json.get(field) == '':
        point = None
    else:
        point = _selected_keypoint(context_json, candidate_keypoints, grasp)
    context = {'keypoints_2d': {'function': point}}
    log_img = None
    if debug:
        log_img = annotate_motion(obs_image_reshaped, context, add_caption=add_caption)
        if log_dir is not None:
            log_img.save(os.path.join(log_dir, f'motion{suffix}.png'))
            with open(os.path.join(log_dir, f'context{suffix}.json'), 'w') as output:
                json.dump(context_json, output)
    return context, context_json, log_img


def keypoint_transfer(subtask, obs_image_reshaped, annotated_image, candidate_keypoints,
                      prompts, waypoint_grid_size=None, debug=False, example_images=None,
                      example_responses=None, loaded_context=None, use_center=False,
                      log_dir=None, suffix='', add_caption=False, grasp=False):
    """Select a coarse semantic point; SD-DINO performs the existing refinement."""
    context = loaded_context
    if context is None:
        context = _request_context([f"Task: {subtask}"], annotated_image, prompts,
                                   example_images, example_responses)
    context_cor = {'selected_keypoint': _selected_keypoint(context, candidate_keypoints, grasp)}
    log_img = None
    if debug:
        print(json.dumps(context, ensure_ascii=False))
        log_img = annotate_motion(obs_image_reshaped, context_cor, add_caption=add_caption)
    return context, context_cor, log_img


def pose_selection(subtask, obs_image, prompts, debug=False):
    """Validate the zero-based pose index against the supplied rendered images."""
    images = obs_image if isinstance(obs_image, list) else [obs_image]
    context = _request_context([f"Task: {subtask}"], images, prompts)
    index = context.get('selected_idx')
    if type(index) is not int or not 0 <= index < len(images):
        raise ValueError(f"Invalid Gemini selected_idx: {index!r}; expected 0..{len(images)-1}")
    if debug:
        print(json.dumps(context, ensure_ascii=False))
    return context
# END Gemini semantic selection.
