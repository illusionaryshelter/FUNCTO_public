"""Declared task row identities and structural cubic dependencies.

PlanningProblem remains the value producer for every task type. This module
binds its fixed masks and sample scopes to reduced control columns. Historical
mechanism coordinates retain their complete query history; point and local
process rows use their actual event/query rows. No trial Jacobian determines
which columns exist.
"""

from time import perf_counter

import numpy as np
import torch

from manipulation.planning_problem import _first_event_query
from .sparse_rows import SparseRows, coefficient_columns
# BEGIN retained-row selection uses the ordinary task's geometric implication.
from .support_presolve import implied_support_rows
# END the task producer remains the complete original declaration.


class SparseTaskRows:
    """Select declared variable rows and bind their sparse derivative slots."""

    def __init__(self, computation, values, blocks, declarations, context, *, support_presolve=True):
        self.computation = computation
        self.joints = len(computation.joint_names)
        self.variables = computation.terminal.matrix.shape[1]*self.joints
        self.coefficients = tuple(row.detach().numpy() @ computation.terminal.matrix
                                  for row, _ in computation.rows[:2])
        self.complete_joint_names = tuple(context['robot_state'][2])
        columns, source_indices, keys, analytic = [], [], [], {}
        fixed, full_offset, selected_equalities = [], 0, 0
        fixed_equalities = set(map(int, computation.fixed_equality_rows))
        for kind, (vector, group) in enumerate(zip(values, declarations)):
            group_offset = 0
            detached = vector.detach().numpy()
            for name, mask in group.items():
                shape = tuple(mask.shape)
                active = mask.detach().numpy().ravel().copy()
                size = active.size
                declared_columns = self._columns(name, shape)
                factor_entry = blocks[kind].get(name)
                factor = None if factor_entry is None else factor_entry[1]
                omitted_rows, omitted_values = [], []
                for local in range(size):
                    original = group_offset+local
                    retained = bool(active[local]) and len(declared_columns[local]) > 0
                    if kind == 0 and original in fixed_equalities:
                        retained = False
                    if not retained:
                        omitted_rows.append(local if factor is None else int(factor.source_rows[local]))
                        omitted_values.append(float(detached[original]))
                        continue
                    target = len(columns)
                    columns.append(declared_columns[local])
                    source_indices.append(full_offset+original)
                    source_row = local if factor is None else int(factor.source_rows[local])
                    keys.append(('task', 'equality' if kind == 0 else 'inequality', name, source_row))
                    if factor is not None:
                        analytic[target] = (kind, name, local)
                    selected_equalities += int(kind == 0)
                if omitted_rows:
                    numbers = np.asarray(omitted_values)
                    feasible = np.isfinite(numbers) & (numbers == 0. if kind == 0 else numbers <= 0.)
                    fixed.append({'kind': 'equality' if kind == 0 else 'inequality', 'key': name,
                                  'row_indices': omitted_rows, 'values': omitted_values,
                                  'finite': np.isfinite(numbers).tolist(), 'feasible': bool(feasible.all())})
                group_offset += size
            full_offset += vector.numel()
        # BEGIN original scoring rows and solver rows have distinct selections.
        # Constant rows were handled above. Only variable support halfspaces
        # implied by a retained required position row leave the NLP layout.
        self.scoring_source_indices = np.asarray(source_indices, dtype=np.int64)
        self.scoring_row_keys = tuple(keys)
        # BEGIN explicit selection retains either the reduced or complete rows.
        if support_presolve:
            removed, presolve = implied_support_rows(computation, keys, analytic, blocks)
        else:
            removed = np.zeros(len(keys), dtype=bool)
            presolve = dict(removed_task_rows=0, certificates=[], elapsed_seconds=0.,
                scope="Complete variable task rows; support presolve disabled")
        presolve["enabled"] = support_presolve
        # END no automatic retry or second solver is attached to this option.
        self.solver_selection = np.flatnonzero(~removed)
        columns = [columns[row] for row in self.solver_selection]
        source_indices = self.scoring_source_indices[self.solver_selection]
        keys = [keys[row] for row in self.solver_selection]
        analytic = {target: analytic[int(original)]
                    for target, original in enumerate(self.solver_selection) if int(original) in analytic}
        # END initialization retains the same original complete violation score.
        self.pattern = SparseRows(columns, self.variables)
        self.columns_by_row = self.pattern.columns_by_row
        self.source_indices = np.asarray(source_indices, dtype=np.int64)
        self.row_keys = tuple(keys)
        # BEGIN bind each analytic product to its final sparse output slot.
        # Factor H differentiates one task block with respect to named joints;
        # B maps reduced spline coefficients to that event. The existing CSR
        # pattern supplies output slots even when AD rows split the block.
        started = perf_counter()
        grouped = {}
        for row, (kind, name, local) in analytic.items():
            block = grouped.setdefault((kind, name), ([], []))
            block[0].append(row)
            block[1].append(local)
        self.analytic_slots = {}
        for key, (rows, locals_) in grouped.items():
            slots = np.concatenate([np.arange(self.pattern.indptr[row],
                                               self.pattern.indptr[row+1]) for row in rows])
            counts = np.diff(self.pattern.indptr)[rows]
            local_rows = np.repeat(locals_, counts)
            columns_ = self.pattern.columns[slots]
            self.analytic_slots[key] = (slots, local_rows,
                                        columns_ % self.joints, columns_ // self.joints)
        slot_binding_seconds = perf_counter()-started
        # END callback work consists only of H/B gathers and one product.
        self.ad_rows = np.asarray([row for row in range(len(columns)) if row not in analytic], dtype=np.int64)
        column_slots = {}
        for local, row in enumerate(self.ad_rows):
            for column, slot in zip(self.pattern.columns_by_row[row],
                                    range(self.pattern.indptr[row], self.pattern.indptr[row+1])):
                column_slots.setdefault(int(column), ([], []))
                column_slots[int(column)][0].append(local)
                column_slots[int(column)][1].append(slot)
        self.ad_column_slots = {column: (np.asarray(rows, dtype=np.int64), np.asarray(slots, dtype=np.int64))
                                for column, (rows, slots) in column_slots.items()}
        self.equalities = selected_equalities
        self.fixed_constraints = fixed
        self.fixed_feasible = all(row['feasible'] for row in fixed)
        self.report = {'complete_task_equalities': values[0].numel(),
                       'complete_task_inequalities': values[1].numel(),
                       'solver_task_equalities': self.equalities,
                       'solver_task_inequalities': len(columns)-self.equalities,
                       'fixed_task_constraints': fixed,
                       'analytic_task_rows': len(analytic),
                       'automatic_task_rows': len(columns)-len(analytic),
                       'support_presolve': presolve,
                       'analytic_slot_binding_seconds': slot_binding_seconds,
                       'task_jacobian_method': ('producer_local_slots_and_scalar_rows'
                           if len(self.ad_rows) < len(self.ad_column_slots)
                           else 'producer_local_slots_and_covector_columns')}

    def _event_column(self, progress):
        mask = _first_event_query(self.computation.progress, progress, self.computation.times)[0]
        return int(mask.to(torch.int64).argmax())

    def _columns(self, name, shape):
        """Resolve all current generic PlanningProblem scalar row producers."""
        parts = name.split(':')
        family = parts[0]
        size = int(np.prod(shape))
        all_joints = np.arange(self.joints, dtype=np.int64)

        def at_sample(sample, derivative=0, joint_columns=all_joints):
            return coefficient_columns(self.coefficients[derivative][[sample]], joint_columns, self.joints)

        def repeated(columns):
            return tuple(columns for _ in range(size))

        if family == 'progress':
            return repeated(np.empty(0, dtype=np.int64))
        if family in {'support', 'function_pose'}:
            collection = self.computation.problem.supports if family == 'support' else self.computation.problem.pose_events
            sample = self._event_column(collection[int(parts[1])].progress)
            return repeated(at_sample(sample))
        if family == 'stop':
            sample = self._event_column(self.computation.problem.events[int(parts[1])].progress)
            controlled = {name: index for index, name in enumerate(self.computation.joint_names)}
            return tuple(at_sample(sample, 1, [controlled[joint]]) if joint in controlled
                         else np.empty(0, dtype=np.int64) for joint in self.complete_joint_names)
        if family == 'coupling':
            coupling = self.computation.problem.couplings[int(parts[1])]
            controlled = {name: index for index, name in enumerate(self.computation.joint_names)}
            component_joints = tuple(np.asarray(sorted(controlled[name]
                for name, nonzero in zip(coupling.joint_names, row != 0.) if nonzero and name in controlled), dtype=np.int64)
                for row in coupling.matrix)
            return tuple(at_sample(sample, joint_columns=component_joints[component])
                         for _, sample, component in np.ndindex(shape))
        if family in {'pose_relation', 'interior'} or (
                family == 'mechanism' and parts[2] in {'position', 'rotation'}):
            return tuple(at_sample(index[1]) for index in np.ndindex(shape))
        if family in {'mechanism', 'mechanism_limits', 'function_coordinate'}:
            # Native lifted coordinates and ordered direction declarations may
            # use all earlier samples, rates and angular-history anchors. Their
            # full declared history is the structural scope, including ties.
            history = np.concatenate(self.coefficients, axis=0)
            return repeated(coefficient_columns(history, all_joints, self.joints))
        raise ValueError(f'No declared dependency scope for task row producer {name!r}')

    def select_values(self, values):
        """Take the same fixed scalar selection from this point's task vector."""
        return torch.cat(values)[self.source_indices]

    # BEGIN full initialization scores reuse the already produced task values.
    def scoring_values(self, values):
        """Return the original variable task rows before implied-row elimination."""
        return torch.cat(values)[self.scoring_source_indices]
    # END the final task evaluator independently retains the original declarations.

    def jacobian_values(self, values, controls, blocks, *, check_request):
        """Write local analytic slots or one scalar VJP directly per row.

        Use the smaller number of scalar row or declared control-column
        queries. A column query differentiates J.T@v only with respect to the
        independent covector v; it requests no robot derivative with respect
        to q twice. Each returned vector writes directly to its sparse slots.
        """
        selected = self.select_values(values)
        result = np.empty(self.pattern.nnz)
        # BEGIN fill the same H[row,joint]*B[coefficient] products by factor.
        for (kind, name), (slots, rows, joints, coefficients) in self.analytic_slots.items():
            check_request()
            factor = blocks[kind][name][1]
            result[slots] = (factor.joint_jacobian[rows, joints]
                             * factor.coefficient_row[coefficients])
            check_request()
        # END explicit indexed assignment writes into the shared result array.
        if len(self.ad_rows) < len(self.ad_column_slots):
            for row in self.ad_rows:
                check_request()
                scalar = selected[row]+controls.sum()*0.
                gradient, = torch.autograd.grad(scalar, controls, retain_graph=True)
                result[self.pattern.slices[row]] = gradient.detach().numpy().ravel()[self.columns_by_row[row]]
        elif len(self.ad_rows):
            connected = selected[self.ad_rows]+controls.sum()*0.
            covector = torch.zeros_like(connected, requires_grad=True)
            product, = torch.autograd.grad(connected, controls, grad_outputs=covector,
                                           create_graph=True, retain_graph=True)
            for column, (rows, slots) in self.ad_column_slots.items():
                check_request()
                derivative, = torch.autograd.grad(product.reshape(-1)[column], covector, retain_graph=True)
                result[slots] = derivative.detach().numpy()[rows]
        check_request()
        return result
