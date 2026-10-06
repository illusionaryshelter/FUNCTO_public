"""Signed functional inequalities for one continued trajectory candidate.

The planner supplies ``g = error / original_tolerance - 1 <= 0`` for each
named event component. This module owns only the Powell--Hestenes--Rockafellar
objective and its outer updates; physical validation and the solve budget
remain owned by the planner. No scalar augmented cost certifies feasibility.
"""

# BEGIN standard inequality AL with explicit, fixed-address device parameters.
import math


class FunctionalAugmentedLagrangian:
    """A single continuation's multipliers, shared only while evaluating it.

    Independent initial seeds may use the same zero multipliers. After the
    planner selects one candidate, ``update`` accepts only that candidate's
    scalar residual for every key. A new independent initializer must call
    ``reset``; updates must never average different seeds or line-search trials.

    ``bind`` runs before graph capture. ``terms`` reads its two device buffers;
    ``update`` and ``reset`` copy new values into those same allocations outside
    capture. The caller must re-evaluate current cost/gradient and reset solver
    history after any parameter change, even when a graph remains replayable.
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
        self.reset()

    def bind(self, like):
        """Allocate once for the current solver's floating device and dtype.

        Rebinding to another device or dtype requires discarding any old graph
        first. Repeated manager initialization on the same device retains the
        allocations, so all rollouts of the current solver read the same state.
        """
        if not like.is_floating_point():
            raise ValueError("Functional AL parameters require a floating tensor")
        if (self._multiplier_buffer is not None
                and self._multiplier_buffer.device == like.device
                and self._multiplier_buffer.dtype == like.dtype):
            return
        self._multiplier_buffer = like.new_tensor(self._multipliers)
        self._penalty_buffer = like.new_tensor(self._penalty)

    def _check_keys(self, signed):
        if set(signed) != set(self.keys):
            raise ValueError("Functional inequality identities changed during continuation")

    def terms(self, signed):
        """Return nonnegative, constant-shifted PHR terms with input batch shape.

        Each value in ``signed`` has shape ``[batch]`` (or a scalar for one
        unbatched curve). All entries of a line-search batch use the unchanged
        continuation parameters. The result is already a scalar cost term and
        must not be squared again when folded into the event's query slot.

        Full PHR subtracts ``lambda**2/(2*rho)``. This constant is independent
        of the candidate within each inner solve, so omitting it preserves all
        gradients and the subproblem minimizers. Native cuRobo best tracking
        divides improvement by the old cost and therefore requires nonnegative
        objectives. Parameters are frozen within a solve; scalar best costs
        must be reset when parameters change, even with this constant omitted.
        """
        import torch
        self._check_keys(signed)
        if not self.keys:
            return {}
        if self._multiplier_buffer is None:
            raise RuntimeError("Bind functional AL device parameters before evaluation")
        values = [signed[key] for key in self.keys]
        like = values[0]
        if (like.device != self._multiplier_buffer.device
                or like.dtype != self._multiplier_buffer.dtype
                or any(value.shape != like.shape or value.device != like.device
                       or value.dtype != like.dtype for value in values)):
            raise ValueError("Functional residuals must match the bound device, dtype and batch")
        residual = torch.stack(values, dim=-1)
        shifted = torch.relu(self._multiplier_buffer+self._penalty_buffer*residual)
        costs = shifted.square()/(2*self._penalty_buffer)
        return {key: costs[..., index] for index, key in enumerate(self.keys)}

    def _copy_parameters(self):
        """Keep graph input addresses unchanged; call only between inner solves."""
        if self._multiplier_buffer is not None:
            self._multiplier_buffer.copy_(self._multiplier_buffer.new_tensor(self._multipliers))
            self._penalty_buffer.fill_(self._penalty)

    def reset(self):
        """Start an independent candidate without retaining another seed's duals."""
        self._multipliers = [0.0]*len(self.keys)
        self._penalty = self.initial_penalty
        self._previous_residual = math.inf
        self._updates = 0
        self._copy_parameters()

    def snapshot(self):
        """Return host diagnostics in normalized, pre-constraint-weight units."""
        return {"constraint_keys": list(self.keys),
                "multipliers": dict(zip(self.keys, self._multipliers)),
                "penalty": self._penalty, "maximum_penalty": self.maximum_penalty,
                "outer_updates": self._updates,
                "previous_complementarity_residual": (
                    self._previous_residual if math.isfinite(self._previous_residual) else None)}

    @property
    def report(self):
        return self.snapshot()

    def update(self, signed):
        """Update from one selected curve, using the previous lambda and rho.

        The residual ``max(abs(max(g, -lambda/rho)))`` measures constraint and
        complementarity progress; it does not include stationarity and is not
        a full KKT residual. If it fails to halve, grow rho by ten, capped at
        ``maximum_penalty``. The first update establishes the comparison value.
        A reached cap is reported, never interpreted as successful convergence.

        This is the standard PHR update also used by NLopt's augmented method:
        https://github.com/stevengj/nlopt/blob/master/src/algs/auglag/auglag.c
        """
        import torch
        self._check_keys(signed)
        if any(value.numel() != 1 for value in signed.values()):
            raise ValueError("Update functional AL from exactly one selected candidate")
        if self.keys:
            values = torch.stack([signed[key].reshape(()) for key in self.keys])
            residuals = values.detach().to(device="cpu", dtype=torch.float64).tolist()
        else:
            residuals = []
        if any(not math.isfinite(value) for value in residuals):
            raise ValueError("Functional AL cannot update from missing or nonfinite events")
        before = self.snapshot()
        measure = max((abs(max(value, -multiplier/self._penalty))
                       for value, multiplier in zip(residuals, self._multipliers)), default=0.)
        multipliers = [max(0., multiplier+self._penalty*value)
                       for value, multiplier in zip(residuals, self._multipliers)]
        if any(not math.isfinite(value) for value in multipliers):
            raise ValueError("Functional AL multiplier update exceeded finite arithmetic")
        increase = measure > self.required_reduction*self._previous_residual
        penalty = min(self.maximum_penalty, self._penalty*self.penalty_growth) if increase else self._penalty
        changed = multipliers != self._multipliers or penalty != self._penalty
        capped = increase and self._penalty*self.penalty_growth > self.maximum_penalty
        self._multipliers, self._penalty = multipliers, penalty
        self._previous_residual = measure
        self._updates += 1
        self._copy_parameters()
        return {**self.snapshot(), "changed": changed, "before": before,
                "signed_residuals": dict(zip(self.keys, residuals)),
                "maximum_normalized_violation": max((max(0., value) for value in residuals), default=0.),
                "complementarity_residual": measure,
                "penalty_increased": penalty > before["penalty"], "penalty_capped": capped}
# END hard tolerances, collision acceptance and iteration budgets are external.
