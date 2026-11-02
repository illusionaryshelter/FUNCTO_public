"""One physical trajectory objective and one explicit task linearization.

The library's JointComputation retains the curve, event layout, continuous
joint domain and complete differentiable task function. This class supplies
the solver's factored support derivatives and adds the already bound TCP
reference to the same objective used by saved-curve scoring.
"""

from time import perf_counter

import numpy as np
import torch

from manipulation.preference_planner.joint_computation import JointComputation
from manipulation.unified_solver import TrajectoryComputation
from .material_hull import material_hull_indices
from .reference_objective import trajectory_objective_components
from .support_linearization import linearized_robot_state, task_linearization
from .sparse_physical import SparsePhysicalRows
from .sparse_rows import SparseRows
from .sparse_task import SparseTaskRows
from .coefficient_coordinates import coefficient_coordinates
from .constraint_envelope import ConstraintEnvelope


class DirectComputation(JointComputation):
    """Bind full task rows, analytic support factors and one reference clock."""

    def __init__(self, *, base, terminal, geometry, reference_objective, support_presolve=True):
        super().__init__(base=base, terminal=terminal, geometry=geometry)
        # BEGIN propagate the public solver-row selection into fixed binding.
        self.support_presolve = support_presolve
        # END complete task values remain independent of this option.
        self.reference_objective = reference_objective
        # BEGIN each objective query reports its declared relationship terms.
        self.scene_relation_report = {}
        # END inherited geometry_report is not used by this objective.
        # BEGIN immutable material declarations bind one solver-local hull index.
        self.support_material_indices = tuple(material_hull_indices(requirement.body_points)
                                              for requirement in self.problem.supports)
        # END complete material arrays remain owned by the original requirements.

    # BEGIN one point context shares native FK across every original cost/task.
    def shared_context(self, controls):
        """Evaluate original states and named frame Jacobians once at this x.

        The base query order retains all original event/first-release anchors.
        Geometry may add its own query states while reusing these exact values.
        Frame Jacobians use reference-axis linear/angular rows and named input
        columns, including the existing nominal hand completion.
        """
        full = self.terminal.expand(controls)
        states = tuple((row @ full + offset)[None] for row, offset in self.rows)
        robot_state, jacobians = linearized_robot_state(self, *states[:2])
        return dict(full_controls=full, states=states, robot_state=robot_state,
                    frame_jacobians=jacobians, parameters=self.parameters,
                    times=self.times, progress=self.progress, joint_names=self.joint_names)

    def task(self, controls, *, solver_rows_out=None):
        """Return the original complete declared task rows for inspection.

        Geometry has its own fixed solver block. The candidate evaluator reads
        the retained task states and geometry queries of the returned point.
        """
        return TrajectoryComputation.task(self, controls, solver_rows_out=solver_rows_out)

    def task_linearization(self, controls, *, solver_rows_out=None, context=None):
        """Return values and factored analytic blocks in the original row order.

        Support blocks identify slices of the original complete task arrays.
        Body/footprint rows use the same local material-hull selection for
        values, masks and factors. The complete source point arrays remain
        available through task() and the original quality objective.
        """
        if context is None:
            context = self.shared_context(controls)
        return task_linearization(self, controls, solver_rows_out=solver_rows_out,
            robot_state=context['robot_state'], jacobians=context['frame_jacobians'],
            support_material_indices=self.support_material_indices)

    def objective_components(self, controls, *, context=None, producer=None):
        """Use one FK result for original costs and the complete TCP reference."""
        if context is None:
            context = self.shared_context(controls)
        # BEGIN required contacts consume the geometry owner's one fixed inventory.
        samples = {}
        if any(rule.required for rule in self.scene.contacts):
            self.geometry._prepare_materials()
            samples = self.scene._objective_material_samples
        # END empty contact declarations need no material construction for scoring.
        costs = trajectory_objective_components(
            reference_objective=self.reference_objective,
            material_samples=samples,
            problem=self.problem, scene=self.scene, states=context['states'],
            robot_state=context['robot_state'], progress=self.progress,
            times=self.times, ordinary_indices=self.ordinary_indices,
            bounds=self.bounds, motion_configuration=self.motion_configuration,
            native_dt=self.native_dt, constraint_weight=self.constraint_weight,
            scene_relation_report=self.scene_relation_report, producer=producer)
        return costs

    def bind_solver(self, initial, *, check_request):
        """Create one fixed task/geometry/physical sparse numerical problem."""
        return BoundComputation(self, initial, check_request=check_request)
    # END complete trial archives are emitted only at the solver-result boundary.


class BoundComputation:
    """Own one immutable row/column layout and original value producers.

    The task declarations, query clock and registered entities determine its
    fixed sparse layout. This object owns numerical producers for one solve.
    """

    def __init__(self, computation, initial, *, check_request):
        self.computation, self.check_request = computation, check_request
        self.shape = np.asarray(initial).shape
        self.variables = int(np.prod(self.shape))
        # BEGIN objective subtimings use the same request checks and counters.
        self.counts = {name: 0 for name in ('shared_state', 'objective', 'objective_base',
                                           'scene_relations', 'reference_prior', 'tool_angular_motion',
                                           'task', 'geometry', 'physical',
                                           'objective_gradient', 'task_jacobian', 'geometry_jacobian')}
        # END child objective times are included in the objective total.
        self.seconds = {name: 0. for name in self.counts}
        controls = torch.tensor(np.asarray(initial), dtype=torch.float64, requires_grad=True)
        context = self.produce('shared_state', lambda: computation.shared_context(controls))
        declarations = ({}, {})
        values, blocks = self.produce('task', lambda: computation.task_linearization(
            controls, solver_rows_out=declarations, context=context))
        self.task = SparseTaskRows(computation, values, blocks, declarations, context,
            support_presolve=computation.support_presolve)
        self.geometry = computation.geometry.bind_solver(computation.terminal)
        # BEGIN each declared entity pair supplies one equivalent inequality.
        # Original physical clocks, event sides and stationary witnesses stay
        # in the geometry point. Their maximum is <= 0 exactly when all are.
        self.geometry_envelope = ConstraintEnvelope(self.geometry.pattern,
            [('geometry', *key[-3:]) for key in self.geometry.row_keys])
        # END grouping identities come from registered entities and conditions.
        # BEGIN one immutable problem replaces geometry-revision identities.
        self.problem_identity = (id(computation),)
        # END no material-exchange revision participates in solver state.
        self.physical = SparsePhysicalRows(computation.full_domain, computation.terminal)
        self.row_keys = self.task.row_keys + self.geometry_envelope.row_keys + self.physical.row_keys
        # BEGIN initialization row identity includes the original support rows.
        self.scoring_row_keys = self.task.scoring_row_keys + self.geometry_envelope.row_keys + self.physical.row_keys
        # END geometric maxima and physical conditions retain the same ordering.
        self.pattern = SparseRows((*self.task.columns_by_row, *self.geometry_envelope.pattern.columns_by_row,
                                   *self.physical.columns_by_row), self.variables)
        self.lower = np.full(len(self.row_keys), -np.inf)
        self.lower[:self.task.equalities] = 0.
        self.upper = np.zeros(len(self.row_keys))
        self.fixed_feasible = self.task.fixed_feasible
        # BEGIN phase snapshots separate initializer work from main callbacks.
        self.producer_snapshots = {}
        self.report = {**self.task.report, 'solver_geometry_inequalities': len(self.geometry_envelope.row_keys),
                       'original_geometry_inequalities': len(self.geometry.row_keys),
                       'solver_physical_inequalities': len(self.physical.row_keys),
                       'solver_jacobian_nonzeros': self.pattern.nnz,
                       'solver_jacobian_shape': list(self.pattern.shape),
                       'producer_evaluations': self.counts, 'producer_seconds': self.seconds,
                       'producer_snapshots': self.producer_snapshots}
        self._initial = (np.asarray(initial).copy(), controls, context, values, blocks)
        self.snapshot_producers('solver_binding')

    def snapshot_producers(self, label):
        """Copy completed counts/times at an actual caller stage boundary."""
        self.producer_snapshots[label] = dict(counts=self.counts.copy(), seconds=self.seconds.copy())
        # END snapshots are passive records, never inputs to a planning decision.

    def produce(self, name, function):
        """Charge actual completed/failed producer work to this fixed problem."""
        self.check_request()
        started = perf_counter()
        self.counts[name] += 1
        try:
            result = function()
        finally:
            self.seconds[name] += perf_counter()-started
        self.check_request()
        return result

    # BEGIN the bound reference defines one fixed numerical coordinate map.
    def solver_coordinates(self, point):
        """Keep coefficient conditioning separate from physical value producers."""
        self.check_request()
        result = coefficient_coordinates(self.computation, point)
        self.check_request()
        return result
    # END geometry, task and joint rows retain their original coordinates.

    def evaluate(self, controls):
        """Build an exact-x point; same-point caching belongs to the solver."""
        value = np.asarray(controls, dtype=float).reshape(self.shape)
        initial, self._initial = self._initial, None
        if initial is not None and np.array_equal(value, initial[0]):
            _, tensor, context, values, blocks = initial
        else:
            tensor = torch.tensor(value, dtype=torch.float64, requires_grad=True)
            context = self.produce('shared_state', lambda: self.computation.shared_context(tensor))
            values, blocks = self.produce('task', lambda: self.computation.task_linearization(tensor, context=context))
        return ComputationPoint(self, tensor, context, values, blocks)


class ComputationPoint:
    """One completed set of values with lazy exact first-order numeric slots."""

    def __init__(self, bound, controls, context, task_values, task_blocks):
        self.bound, self.controls, self.context = bound, controls, context
        self.task_values, self.task_blocks = task_values, task_blocks
        components = bound.produce('objective', lambda: bound.computation.objective_components(
            controls, context=context, producer=bound.produce))
        self.objective_tensor = sum(components.values())
        self.objective_value = float(self.objective_tensor.detach())
        self.objective_components = {name: float(value.detach()) for name, value in components.items()}
        # BEGIN snapshot only plain relationship evidence for this exact x.
        self.scene_relations = dict(bound.computation.scene_relation_report)
        # END the original graph is released after lazy first derivatives.
        full_controls = context['full_controls'].detach().numpy()
        # BEGIN retain plain candidate inputs independently of the Torch graph.
        self.full_controls = full_controls.copy()
        q, dq, names, _, tool, twist = context['robot_state']
        self.candidate_state = (q.detach(), dq.detach(), names,
            tuple(value.detach() for value in tool), tuple(value.detach() for value in twist))
        # END derivative completion may release context without losing results.
        self.geometry = bound.produce('geometry', lambda: bound.geometry.evaluate(
            full_controls, shared_context=context))
        self.physical = bound.produce('physical', lambda: bound.physical.evaluate(
            full_controls, check_request=bound.check_request))
        # BEGIN retain complete task values for initialization scoring.
        task = bound.task.scoring_values(task_values).detach().numpy()
        # END the same production supplies the solver selection below.
        # BEGIN the optimizer consumes grouped maxima; candidate evidence is original.
        geometry, self.geometry_winners = bound.geometry_envelope.select(self.geometry.values)
        # BEGIN one value production serves full scoring and the reduced NLP.
        self.scoring_constraint_values = np.concatenate((task, geometry, self.physical.values))
        self.constraint_values = np.concatenate((task[bound.task.solver_selection], geometry, self.physical.values))
        # END discarded support rows never change initialization branch ranking.
        # END no physical or task condition is dropped by the max identity.
        self._gradient = self._jacobian = None

    # BEGIN initial proposals are ranked by the original full condition set.
    def initialization_score(self):
        """Original sum of positive violations, then the original objective.

        Task equalities occupy the leading rows and have zero lower/upper
        bounds; all other original task, geometry and physical rows are <= 0.
        This consumes completed arrays, without additional FK or geometry.
        """
        values = self.scoring_constraint_values
        violation = np.maximum(values, 0.)
        equalities = self.bound.task.equalities
        violation[:equalities] = np.abs(values[:equalities])
        worst = int(np.argmax(violation))
        return dict(total_positive_violation=float(violation.sum()),
            maximum_positive_violation=float(violation[worst]),
            positive_condition_count=int(np.count_nonzero(violation)),
            worst_condition=self.bound.scoring_row_keys[worst],
            objective=self.objective_value, objective_components=self.objective_components)
    # END IPOPT callbacks and derivatives use the separate retained-row vector.

    def objective_gradient(self):
        """Use the shared native first-order graph once for the complete cost."""
        if self._gradient is None:
            def derivative():
                gradient, = torch.autograd.grad(self.objective_tensor, self.controls, retain_graph=True)
                return gradient.detach().numpy().ravel()
            self._gradient = self.bound.produce('objective_gradient', derivative)
        return self._gradient

    def jacobian_values(self):
        """Concatenate producer-owned sparse values, retaining no dense matrix."""
        if self._jacobian is None:
            self.objective_gradient()
            task = self.bound.produce('task_jacobian', lambda: self.bound.task.jacobian_values(
                self.task_values, self.controls, self.task_blocks, check_request=self.bound.check_request))
            geometry = self.bound.produce('geometry_jacobian', self.geometry.jacobian_values)
            geometry = self.bound.geometry_envelope.jacobian_values(geometry, self.geometry_winners)
            self._jacobian = np.concatenate((task, geometry, self.physical.jacobian_values()))
            # All first derivatives now own plain arrays. Release this point's
            # Torch graph together; repeated Callback calls reuse only arrays.
            self.controls = self.context = self.task_values = self.task_blocks = self.objective_tensor = None
        return self._jacobian
