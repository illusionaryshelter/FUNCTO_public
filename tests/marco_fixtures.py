"""Small-channel native-coordinate feature fixtures for geometry consumers.

These arrays replace learned features only. Real MARCO sampling, decoding,
window masks, RGB-D backprojection and frame reconstruction remain active.
An 840-grid gives exact resized-pixel centres for tests of geometric identities;
the production worker separately validates its learned 240-grid output.
"""

import numpy as np

from local_models.marco import RESOLUTION, resize_scales


def labeled_pair(first, second):
    """Encode nonzero red-channel labels as isolated high-confidence features."""
    labels = [np.asarray(image)[:, :, 0] for image in (first, second)]
    channels = max(int(values.max()) for values in labels) + 1
    if channels > 16:
        raise ValueError("Synthetic feature labels must be small integers")
    pair = []
    for image, values in zip((first, second), labels):
        features = np.zeros((1, channels, RESOLUTION, RESOLUTION), dtype=np.float32)
        y, x = np.nonzero(values)
        locations = np.rint(np.column_stack((x, y)) * resize_scales(image.size)).astype(int)
        # The high amplitude makes each isolated identity win without a
        # near-uniform softmax; no target positions enter production code.
        features[0, values[y, x], locations[:, 1], locations[:, 0]] = 1000.
        pair.append(features)
    return pair
