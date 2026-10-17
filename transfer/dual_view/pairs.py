"""Native MARCO region pairs with invocation-local Gaussian and grid reuse.

The matching, mask, 32-query batches, score ordering and observed-pixel output
follow transfer.features.native_region_pairs. Only the repeated integer-offset
Gaussian values and coordinate grid are reused within one invocation. Native
MARCO softmax/scaling helpers remain direct read-only dependencies; no global
function, source file, module namespace or third-party implementation is patched.
"""
import numpy as np

from ..core import fps_indices
from ..features import sample_descriptors


def prepare():
    """Resolve maintained MARCO pure geometry before a measured invocation."""
    from local_models.marco import _geometry
    return _geometry()


class _GaussianGrid:
    """Own fresh lookup tensors for exactly one native region-pair call.

    Native unit-spaced grid locations and argmax centers are integer cells.
    The full offset table therefore evaluates the same square/add/negate/
    divide/exp expression without a separable-product approximation. The
    original default floating dtype and actual reduction shapes are retained.
    """

    def __init__(self):
        self.geometry = prepare()
        self.tables, self.grids = {}, {}

    def grid(self, height, width, gap=1, device='cpu'):
        import torch
        key = (height, width, gap, str(device), torch.get_default_dtype())
        if key not in self.grids:
            self.grids[key] = self.geometry.create_grid(height, width, gap, device)
        return self.grids[key]

    def gaussian(self, scoremaps, sigma=7):
        import torch
        batch, queries, height, width = scoremaps.shape
        device = scoremaps.device
        key = (height, width, sigma, str(device), torch.get_default_dtype())
        if key not in self.tables:
            dtype = torch.get_default_dtype()
            dx = torch.arange(1-width, width, dtype=dtype, device=device)
            dy = torch.arange(1-height, height, dtype=dtype, device=device)
            table = torch.exp(-(dx[None, :] ** 2 + dy[:, None] ** 2) / (2 * sigma ** 2))
            x = torch.arange(width, device=device).view(1, 1, 1, width)
            y = torch.arange(height, device=device).view(1, 1, height, 1)
            self.tables[key] = (table, x, y)
        table, x, y = self.tables[key]
        index = torch.max(scoremaps.view(batch, queries, -1), dim=-1).indices
        centre_y = (index // width).view(batch, queries, 1, 1)
        centre_x = (index % width).view(batch, queries, 1, 1)
        weights = table[y-centre_y+height-1, x-centre_x+width-1]
        return weights * scoremaps

    def decode(self, logits, allowed):
        """Original project decoder with explicit local Gaussian/grid owners.

        Mask before finding the Gaussian center and again before softmax, so
        forbidden pixels cannot acquire probability. Native reduction order,
        temperature, grid layout and align-corner scaling remain unchanged.
        """
        import torch
        from local_models.marco import RESOLUTION, SIGMA, TEMPERATURE
        valid = torch.as_tensor(allowed, dtype=torch.bool, device=logits.device)[None, None]
        if not bool(valid.any()):
            raise ValueError('No MARCO feature cell is supported by the target mask')
        bounded = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
        weighted = self.gaussian(bounded, sigma=SIGMA).masked_fill(~valid, -torch.inf)
        probabilities = self.geometry.softmax_with_temperature(weighted.flatten(2), TEMPERATURE, -1)
        height, width = logits.shape[-2:]
        grid = self.grid(height, width, device=logits.device).reshape(1, 1, -1, 2)
        coordinates = (probabilities[..., None] * grid).sum(dim=2)
        return self.geometry.scaling_coordinates(coordinates, (height, width), (RESOLUTION, RESOLUTION))


def native_region_pairs(source, target, source_region_mask, feature_pair, count=160):
    """Decode region queries and retain the original invalid-observation schema.

    The caller controls CPU thread scheduling, using eight Torch threads for
    this selected matching stage. No thread setting changes inside the API.
    Each call owns its Gaussian/grid cache, so another target or invocation
    cannot reuse sampled evidence or retain these tensors after return.
    """
    import torch
    from local_models.marco import allowed_grid, resize_scales

    decoder = _GaussianGrid()
    mask = np.asarray(source_region_mask)
    if mask.dtype != np.bool_ or mask.shape != (len(source['xyz']),):
        raise ValueError('Source region must be a point-aligned Boolean mask')
    region_indices = np.flatnonzero(mask)
    selected = fps_indices(source['xyz'][region_indices], count)
    source_indices = region_indices[selected]
    if not len(source_indices):
        raise ValueError('Source region has no visible points')
    descriptors = sample_descriptors(feature_pair[0], source['pixels'][source_indices],
                                     source['object_mask'].shape[::-1])
    target_mask = np.zeros_like(target['object_mask'], dtype=bool)
    pixels = np.asarray(target['pixels'], dtype=int)
    target_mask[pixels[:, 1], pixels[:, 0]] = True
    allowed = allowed_grid(target_mask, feature_pair[1].shape[-2:])
    if not allowed.any():
        raise ValueError('No native feature cell observes the target entity')
    target_map = torch.from_numpy(feature_pair[1])
    decoded, scores = [], []
    with torch.inference_mode():
        for start in range(0, len(descriptors), 32):
            queries = torch.from_numpy(descriptors[start:start + 32])[None]
            logits = torch.bmm(queries, target_map.flatten(2)).reshape(
                1, -1, *target_map.shape[-2:])
            decoded.append(decoder.decode(logits, allowed)[0].numpy())
            scores.append(logits[0, :, allowed].max(dim=1).values.numpy())
    output_pixels = np.concatenate(decoded) / resize_scales(target_mask.shape[::-1])
    nearest = np.rint(output_pixels).astype(int)
    point_index = np.full(target_mask.shape, -1, dtype=np.int64)
    point_index[pixels[:, 1], pixels[:, 0]] = np.arange(len(pixels))
    in_image = ((nearest[:, 0] >= 0) & (nearest[:, 0] < target_mask.shape[1])
                & (nearest[:, 1] >= 0) & (nearest[:, 1] < target_mask.shape[0]))
    target_indices = np.full(len(nearest), -1, dtype=np.int64)
    target_indices[in_image] = point_index[nearest[in_image, 1], nearest[in_image, 0]]
    valid = target_indices >= 0
    # One observed depth point has capacity one for fitting. Keep every raw
    # duplicate and choose the original first strongest pair for that point.
    raw_scores = np.concatenate(scores)
    fitting = []
    for target_index in np.unique(target_indices[valid]):
        candidates = np.flatnonzero(valid & (target_indices == target_index))
        fitting.append(int(candidates[np.argmax(raw_scores[candidates])]))
    fitting = np.asarray(fitting, dtype=np.int64)
    return {
        'src_anchor_idx': source_indices[fitting],
        'tgt_anchor_idx': target_indices[fitting],
        'scores': raw_scores[fitting],
        'raw_source_idx': source_indices,
        'raw_target_idx': target_indices,
        'raw_target_pixels': output_pixels,
        'raw_valid': valid,
        'raw_scores': raw_scores,
        'matching_rule': 'native MARCO masked Gaussian soft-argmax; rounded pixel for RGB-D',
    }
