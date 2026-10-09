"""Geometric optimization costs for already queried sphere collision streams.

Hard collision acceptance remains with the scene. This module only changes
how its nonnegative per-sphere proposals guide a complete trajectory.
"""
# BEGIN one shared geometric objective, independent of task and scene backends.
from dataclasses import dataclass
from typing import Mapping

import torch
from torch.utils.checkpoint import checkpoint


# BEGIN bound recomputation scratch independently of robot/task dimensions.
_COLLISION_PATH_SAMPLE_ELEMENTS = 65_536


def _collision_path_chunk_size(states, spheres):
    """Keep one sphere when a state batch alone exceeds the sample bound."""
    return min(spheres, max(1, _COLLISION_PATH_SAMPLE_ELEMENTS // states)) if spheres else 0


def _sphere_chunk_objective(c, x, radius, valid, edges, order, begin, end, *mode_keys):
    """Recompute one sphere block, including its time-ordering copies.

    Complete raw tensors are checkpoint inputs; slicing and gathering happen
    here so the graph retains neither ordered centres nor edge intermediates
    for every block. Integer bounds are explicit arguments, not loop closures.
    """
    def ordered(value):
        if order is None:
            return value
        indices = order.reshape(*order.shape, *((1,) * (value.ndim - 2))).expand_as(value)
        return value.gather(1, indices)

    values = torch.where(valid[..., None], ordered(c[..., begin:end]), 0.)
    centres = ordered(x[..., begin:end, :])
    peak = values.amax(dim=1)
    connected = edges[..., None].expand(*edges.shape, end - begin)
    for key in mode_keys:
        key = ordered(key if key.shape[-1] == 1 else key[..., begin:end])
        connected = connected & (key[:, 1:] == key[:, :-1])
    # Mask endpoints before differences: NaN padding multiplied by zero
    # after a norm would still propagate NaN through values and gradients.
    first = torch.where(connected[..., None], centres[:, :-1], 0.)
    second = torch.where(connected[..., None], centres[:, 1:], 0.)
    length = torch.linalg.vector_norm(second - first, dim=-1)
    first_cost = torch.where(connected, values[:, :-1], 0.)
    second_cost = torch.where(connected, values[:, 1:], 0.)
    arc = (((first_cost + second_cost) * .5) * length).sum(dim=1)
    return (peak + arc / radius[begin:end]).sum(dim=-1)
# END blocks preserve every sphere and only regroup the final sphere sum.


@dataclass(frozen=True)
class SphereCollisionPath:
    """One actor/target stream with stable sphere identities.

    ``residual`` is [B,T,S]; ``centres`` is [B,T,S,3] in target coordinates.
    ``radii`` is the fixed finite positive [S] model scale, validated once by
    the scene binding and never an optimizer variable. The reducer only checks
    metadata, avoiding device synchronization inside optimization iterations.
    Each mode key is [B,T,1] or [B,T,S]. All keys must agree across
    an edge before that edge can represent motion in one collision model.
    """

    residual: torch.Tensor
    centres: torch.Tensor
    radii: torch.Tensor
    mode_keys: tuple[torch.Tensor, ...] = ()


def collision_path_objective(
    paths: Mapping[str, SphereCollisionPath],
    times: torch.Tensor,
    valid: torch.Tensor,
    *,
    order: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return [B] sum of per-sphere peak plus trapezoidal arc cost/radius.

    The caller supplies physical-time/same-time-mode ordering in ``order``;
    without it, samples must already be ordered. Raw geometry is retained and
    each sphere block gathers its samples inside a Torch checkpoint. This
    bounds temporary ordering and derivative storage without rerunning scene
    geometry or changing any sample. A
    duplicate time never creates an edge; mode changes never create a spatial
    jump. Every valid point still participates in its sphere's global peak,
    including endpoints, isolated modes and completely stationary collisions.

    Arc cost uses target-relative displacement, not TCP speed or elapsed
    time. It is invariant to timing of the same sampled geometric path. For
    nonlinear fields, trapezoidal quadrature remains an approximation, not a
    continuous collision certificate. No path-length denominator can dilute
    an existing collision by appending unrelated collision-free travel.

    Raw residuals and geometry retain their Torch derivatives. In particular,
    vector_norm supplies a finite zero subgradient for a stationary edge.
    Invalid padded points are masked before arithmetic; nonfinite valid
    residuals remain nonfinite costs and cannot become feasible evidence.
    """
    if (times.ndim != 2 or not times.is_floating_point() or times.shape[1] < 1
            or valid.shape != times.shape or valid.dtype != torch.bool
            or valid.device != times.device):
        raise ValueError("Collision paths require aligned floating times and Boolean validity")
    batch, horizon = times.shape
    if order is not None:
        if order.shape != times.shape or order.dtype != torch.long or order.device != times.device:
            raise ValueError("Collision path order must be an aligned int64 permutation")
        times, valid = times.gather(1, order), valid.gather(1, order)
    total = times.new_zeros(batch)
    edges = valid[:, :-1] & valid[:, 1:] & (times[:, 1:] > times[:, :-1])
    for path in paths.values():
        c, x, radius = path.residual, path.centres, path.radii
        if (c.ndim != 3 or c.shape[:2] != times.shape or c.shape[2] < 1
                or x.shape != (*c.shape, 3) or radius.shape != (c.shape[2],)
                or any(value.device != times.device or value.dtype != times.dtype
                       for value in (c, x, radius)) or radius.requires_grad):
            raise ValueError("Sphere collision streams require aligned samples and fixed model radii")
        for key in path.mode_keys:
            if (key.ndim != 3 or key.shape[:2] != times.shape
                    or key.shape[2] not in (1, c.shape[2]) or key.device != times.device
                    or key.requires_grad):
                raise ValueError("Collision mode keys must align with time and sphere identity")
        step = _collision_path_chunk_size(batch * horizon, c.shape[2])
        gradients = torch.is_grad_enabled() and (c.requires_grad or x.requires_grad)
        for begin in range(0, c.shape[2], step):
            args = (c, x, radius, valid, edges, order, begin,
                    min(begin + step, c.shape[2]), *path.mode_keys)
            value = (checkpoint(_sphere_chunk_objective, *args,
                                use_reentrant=False, preserve_rng_state=False)
                     if gradients else _sphere_chunk_objective(*args))
            total = total + value
    return total
# END no hard scene residual, permission, sampling budget or task weight changes.


# BEGIN both scene estimators reserve the same optional geometric objective.
def collision_path_workspace_allocation(batch, horizon, *, geometry_spheres, path_spheres,
                                        max_stream_spheres, mode_bytes_per_state):
    """Bound optional fixed metadata and one live forward/backward path graph.

    Counts are CPU model metadata. ``geometry_spheres`` counts each actor's
    physical sphere capacity once; ``path_spheres`` counts it once per target.
    Raw exports retain at most five floats per sphere: a target-local sphere
    array (four coordinates, because a centre view retains its backing array)
    and one residual. Reserve another twelve floats for three complete c/x
    gradient arrays (four coordinates each): the current checkpoint block's
    return, the previously accumulated gradient, and their addition result.

    Only one checkpointed block retains sorting/edge/reduction intermediates.
    Its sixty-four-scalar scratch bound covers selection, ordering, endpoint
    differences and norms, products, peaks and their backward temporaries.
    Full input gradients above remain separate even with one-sphere blocks.
    ``mode_bytes_per_state`` counts actual pair-scoped Boolean sphere arrays
    and broadcast int64 attachment columns; no all-owner mode cross-product.
    """
    states = int(batch)*int(horizon)
    path_elements = states*int(path_spheres)
    block_elements = states*_collision_path_chunk_size(states, int(max_stream_spheres))
    # Eight int64 arrays bound sorting indices, side identities and their
    # temporary permutations. Four scalar arrays bound physical times and
    # their ordered copies. These are per state, never per sphere/target.
    indices_and_modes = states*(8*8+int(mode_bytes_per_state))
    scalar_elements = 17*path_elements+64*block_elements+4*states
    return {'cpu_shared': int(geometry_spheres)*48,
            'gpu_shared': int(geometry_spheres)*28,
            'cpu_live': scalar_elements*8+indices_and_modes,
            'gpu_live': scalar_elements*4+indices_and_modes}
# END no scene-specific policy or optimizer configuration enters this helper.
