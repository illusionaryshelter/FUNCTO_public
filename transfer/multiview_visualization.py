"""Render frozen joint-surface regions on their original RGB observations.

Region pixels retain their saved observation-to-element identity. Actual
sample coordinates are projected through depth verification; an occluded
sample is never painted as if the other camera observed it.
"""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .multiview_geometry import project_observations


# BEGIN original-observation rendering for the joint-surface contract.
REGION_COLOR = (48, 208, 119)
VIEW_COLORS = ((244, 158, 65), (177, 112, 241))
SAMPLE_COLOR = '#ffed64'


def read_rgb(path):
    return np.asarray(Image.open(path).convert('RGB'))


def overlay(rgb, mask, color, alpha=.48):
    """Apply an original-image Boolean mask without resizing its pixel chart."""
    if np.asarray(mask).dtype != np.bool_ or np.asarray(mask).shape != rgb.shape[:2]:
        raise ValueError('Overlay mask must be Boolean and original-image aligned')
    result = np.array(rgb, copy=True)
    result[mask] = np.rint((1 - alpha) * result[mask] + alpha * np.asarray(color)).astype(np.uint8)
    return result


def source_picture(rgb, region_mask, source=None, query_indices=None, query_region=None):
    """Display the supplied demonstration and, when available, fixed queries."""
    image = Image.fromarray(overlay(rgb, region_mask, (255, 70, 94)))
    if query_indices is not None:
        if source is None or query_region is None:
            raise ValueError('Source queries require their original pixel identities')
        draw = ImageDraw.Draw(image)
        for (x, y), is_region in zip(source['pixels'][query_indices], query_region):
            color = '#ffed64' if is_region else '#63e8ff'
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color, outline='#172222')
    return image


def single_view_picture(rgb, observation, result):
    """Render historical frozen primary/diagnostic output without rerunning it."""
    image = Image.fromarray(rgb)
    if observation is None or not result['modes']:
        return image
    mode = result['modes'][0]
    mask = np.asarray(mode['mask'])
    if mask.dtype != np.bool_ or mask.shape != (len(observation['xyz']),):
        raise ValueError('Historical mode mask must index its original target observation')
    pixels = observation['pixels']
    painted = np.zeros(rgb.shape[:2], dtype=bool)
    xy = pixels[mask]
    painted[xy[:, 1], xy[:, 0]] = True
    image = Image.fromarray(overlay(rgb, painted, REGION_COLOR))
    draw = ImageDraw.Draw(image)
    for x, y in pixels[mode['sample_indices']]:
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=SAMPLE_COLOR, outline='#172222')
    x, y = pixels[mode['seed_index']]
    draw.line((x - 7, y, x + 7, y), fill='white', width=2)
    draw.line((x, y - 7, x, y + 7), fill='white', width=2)
    return image


def joint_region_picture(rgb, surface, mode, view_index, *, show_samples=True):
    """Return an original-view image and auditable sample visibility counts.

    A region element may contain observations from both views. Its original
    pixels are painted only in their own frame. Colors report which cameras
    support the element geometrically, not which camera supplied its MARCO
    candidate. A sample marker always refers to the saved 3-D representative.
    """
    mask = np.asarray(mode['mask'])
    if mask.dtype != np.bool_ or mask.shape != (len(surface['xyz']),):
        raise ValueError('Joint mode mask must index frozen surface elements')
    samples = np.asarray(mode['sample_indices'])
    if (samples.ndim != 1 or samples.dtype.kind not in 'iu' or np.any(samples < 0)
            or np.any(samples >= len(mask)) or not mask[samples].all()):
        raise ValueError('Joint samples must belong to the saved element region')
    observations = np.flatnonzero(surface['observation_view_index'] == view_index)
    elements = surface['observation_to_element'][observations]
    valid = elements >= 0
    selected = valid.copy()
    selected[valid] &= mask[elements[valid]]
    observations, elements = observations[selected], elements[selected]
    pixels = surface['observation_pixels'][observations]
    image_rgb = np.array(rgb, copy=True)
    observed = np.asarray(surface['observed_views'], dtype=bool)
    common = observed[elements].all(axis=1)
    for selection, color in ((common, REGION_COLOR),
                             (~common & observed[elements, 0], VIEW_COLORS[0]),
                             (~common & observed[elements, 1], VIEW_COLORS[1])):
        image_mask = np.zeros(rgb.shape[:2], dtype=bool)
        xy = pixels[selection]
        image_mask[xy[:, 1], xy[:, 0]] = True
        image_rgb = overlay(image_rgb, image_mask, color)
    image = Image.fromarray(image_rgb)
    projection = project_observations(surface, surface['xyz'][samples], view_index)
    supported = np.asarray(projection['supported'], dtype=bool)
    coordinates = np.asarray(projection['pixels'])[supported]
    draw = ImageDraw.Draw(image)
    if show_samples:
        for x, y in coordinates:
            draw.ellipse((x - 2.5, y - 2.5, x + 2.5, y + 2.5), fill=SAMPLE_COLOR, outline='#172222')
    center = project_observations(surface, np.asarray(mode['center'])[None], view_index)
    if show_samples and center['supported'][0]:
        x, y = center['pixels'][0]
        draw.line((x - 7, y, x + 7, y), fill='white', width=2)
        draw.line((x, y - 7, x, y + 7), fill='white', width=2)
    counts = {'region_observation_pixels': int(len(pixels)), 'sample_count': int(len(samples)),
              'visible_samples': int(supported.sum()), 'conflicting_samples': int(np.sum(projection['conflict'])),
              'unknown_samples': int(np.sum(projection['unknown']))}
    return image, counts


def save_picture(image, path):
    """Save a local PNG without altering the original cached RGB file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path
# END original-observation rendering.
