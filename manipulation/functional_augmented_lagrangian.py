"""Per-seed PHR parameters for the planner's fixed-budget inner blocks.

The problem supplies ``g = error / original_tolerance - 1 <= 0``. Parameters
are constant within one native inner block; neither their update nor a small
augmented objective certifies feasibility. Original metrics own acceptance.
"""

# BEGIN independent seed parameters; line-search particles never own dual state.
import torch


class FunctionalAugmentedLagrangian:
    """One multiplier vector and penalty per independently optimized seed.

    ``bind`` declares the seed layout before evaluation/capture. Rollout rows
    are seed-major, then line-search-particle-major, exactly as GradientOptCore
    reshapes actions. ``update`` requires one actual continuation point per
    seed. It cannot average particles or implicitly select a different seed.
    """

    initial_penalty = 2.0
    maximum_penalty = 1e6
    penalty_growth = 10.0
    required_reduction = 0.5

    def __init__(self, keys):
        self.keys = tuple(keys)
        if (any(not isinstance(key, str) or not key for key in self.keys)
                or len(set(self.keys)) != len(self.keys)):
            raise ValueError("Functional inequalities require distinct nonempty keys")
        self._multiplier_buffer = None
        self._penalty_buffer = None
        self.num_seeds = 1

    def bind(self, like, num_seeds=None):
        """Keep addresses for an unchanged layout; a changed layout starts fresh.

        Manager rebinding only supplies dtype/device and therefore retains the
        existing seed count. The core explicitly declares its count at each
        solve entry, before either initial or ordinary rollout evaluation.
        """
        if not like.is_floating_point():
            raise ValueError("Functional AL parameters require a floating tensor")
        count = self.num_seeds if num_seeds is None else num_seeds
        if not isinstance(count, int) or count < 1:
            raise ValueError("Functional AL requires a positive seed count")
        if (self._multiplier_buffer is not None and count == self.num_seeds
                and self._multiplier_buffer.device == like.device
                and self._multiplier_buffer.dtype == like.dtype):
            return
        self.num_seeds = count
        self._multiplier_buffer = like.new_zeros((count, len(self.keys)))
        self._penalty_buffer = like.new_empty(count)
        self._previous_residual = like.new_empty(count)
        self._updates = torch.zeros(count, dtype=torch.int64, device=like.device)
        self._penalty_capped = torch.zeros(count, dtype=torch.bool, device=like.device)
        self.reset()

    def reset(self):
        """Reset an independent solve in place, preserving captured addresses."""
        if self._multiplier_buffer is None:
            return
        self._multiplier_buffer.zero_()
        self._penalty_buffer.fill_(self.initial_penalty)
        self._previous_residual.fill_(float("inf"))
        self._updates.zero_()
        self._penalty_capped.zero_()

    def _residuals(self, signed):
        if set(signed) != set(self.keys):
            raise ValueError("Functional inequality identities changed during continuation")
        if self._multiplier_buffer is None:
            raise RuntimeError("Bind functional AL device parameters before evaluation")
        if not self.keys:
            return self._multiplier_buffer
        values = [signed[key] for key in self.keys]
        like = values[0]
        if (like.ndim != 1 or like.numel() % self.num_seeds
                or like.device != self._multiplier_buffer.device
                or like.dtype != self._multiplier_buffer.dtype
                or any(value.shape != like.shape or value.device != like.device
                       or value.dtype != like.dtype for value in values)):
            raise ValueError("Functional residuals must match the seed-major device, dtype and batch layout")
        return torch.stack(values, dim=-1)

    def terms(self, signed):
        """Nonnegative PHR terms, broadcasting only within each seed's particles.

        Omitting the x-independent ``-lambda**2/(2*rho)`` preserves gradients
        and minimizers while retaining the native tracker's nonnegative cost
        convention. Such costs cannot be compared across parameter updates.
        """
        if not self.keys and not signed:
            return {}
        residual = self._residuals(signed)
        particles = residual.shape[0] // self.num_seeds
        residual = residual.view(self.num_seeds, particles, len(self.keys))
        penalty = self._penalty_buffer[:, None, None]
        shifted = torch.relu(self._multiplier_buffer[:, None, :]+penalty*residual)
        costs = (shifted.square()/(2*penalty)).reshape(-1, len(self.keys))
        return {key: costs[:, index] for index, key in enumerate(self.keys)}

    @torch.no_grad()
    def update(self, signed):
        """Return changed seed rows after one block's inexact PHR update.

        ``max(abs(max(g, -lambda/rho)))`` measures feasibility/complementarity
        progress, not stationarity or full KKT convergence. The first update
        establishes the comparison value. Finite steps may approach a boundary
        from its infeasible side indefinitely; acceptance stays outside here.
        """
        residual = self._residuals(signed)
        if residual.shape != self._multiplier_buffer.shape:
            raise ValueError("Update functional AL from exactly one continuation point per seed")
        if not bool(torch.isfinite(residual).all()):
            raise ValueError("Functional AL cannot update from missing or nonfinite events")
        if not self.keys:
            return torch.zeros(self.num_seeds, dtype=torch.bool, device=residual.device)
        measure = torch.maximum(residual, -self._multiplier_buffer/self._penalty_buffer[:, None]).abs().amax(-1)
        multipliers = torch.relu(self._multiplier_buffer+self._penalty_buffer[:, None]*residual)
        if not bool(torch.isfinite(multipliers).all()):
            raise ValueError("Functional AL multiplier update exceeded finite arithmetic")
        increase = measure > self.required_reduction*self._previous_residual
        proposed_penalty = self._penalty_buffer*self.penalty_growth
        penalty = torch.where(increase, proposed_penalty.clamp_max(self.maximum_penalty), self._penalty_buffer)
        changed = (multipliers != self._multiplier_buffer).any(-1) | (penalty != self._penalty_buffer)
        self._penalty_capped.copy_(increase & (proposed_penalty > self.maximum_penalty))
        self._multiplier_buffer.copy_(multipliers)
        self._penalty_buffer.copy_(penalty)
        self._previous_residual.copy_(measure)
        self._updates.add_(1)
        return changed

    def snapshot(self):
        """Copy small per-seed diagnostics only at a completed solve boundary."""
        if self._multiplier_buffer is None:
            return {"constraint_keys": list(self.keys), "seeds": []}
        multipliers = self._multiplier_buffer.detach().cpu().tolist()
        penalties = self._penalty_buffer.detach().cpu().tolist()
        previous = self._previous_residual.detach().cpu().tolist()
        updates = self._updates.cpu().tolist()
        capped = self._penalty_capped.cpu().tolist()
        return {"constraint_keys": list(self.keys), "maximum_penalty": self.maximum_penalty,
                "seeds": [{"seed_index": index, "multipliers": dict(zip(self.keys, multipliers[index])),
                    "penalty": penalties[index], "outer_updates": updates[index],
                    "previous_complementarity_residual": previous[index] if updates[index] else None,
                    "penalty_capped": capped[index]} for index in range(self.num_seeds)]}

    @property
    def report(self):
        return self.snapshot()
# END hard tolerances, collision acceptance and iteration budgets are external.
