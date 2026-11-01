"""One fixed batch geometry block on the original trajectory and entity models.

The physical query clock, registered pairs and declared attachment/release
sources determine a fixed sparse layout. Batched original-shape distances
provide values and pose covectors; the same result supplies candidate scene
diagnostics. Numerical iterations do not discover materials or revise rows.
"""

# BEGIN one batch provider retains the original model and event interfaces.
from copy import copy
from itertools import combinations
from time import perf_counter

import coal
import numpy as np
from scipy.spatial.transform import Rotation
import torch

from .batch_distance import BatchDistance
from .native_interval_distance import NativeIntervalDistance
from manipulation._coal_scene_batch import MAX_BATCH_SIZE, NativePairTable
from manipulation.implicit_distance import ObservedField
from manipulation.online_geometry import _matrices
from manipulation.planning_problem import quaternion_multiply, quaternion_rotate
from manipulation.preference_planner.geometry import GeometryConstraints as ScenePermissions
from manipulation.sampled_geometry_objective import MaterialSamples
from .sparse_rows import SparseRows, coefficient_columns
from .support_linearization import linearized_robot_state


def pair_identity(record):
    """Keep all physical, relation, permission and declared pair identities."""
    return tuple(record[name] for name in (
        'time_s', 'progress', 'permission_progress', 'first', 'second', 'component'))


def _query_records(base, progress_map):
    """Use the original exact/left/right scene-boundary clock declarations.

    Physical q/time stays fixed on either side. At a robot contact arrival the
    left relation side uses the original exact material permission, matching
    planning._scene_mode_residual. Exact base queries always precede sides so
    the original first-release gather keeps its physical anchor.
    """
    times, progress = base.times[0].numpy(), base.progress[0].numpy()
    exact = [dict(time_s=float(t), progress=float(p), permission_progress=float(p))
             for t, p in zip(times, progress)]
    left, right = np.zeros(len(times), bool), np.zeros(len(times), bool)
    for anchor in base.scene.required_progress_anchors:
        if not progress_map.s[0] <= anchor <= progress_map.s[-1]:
            continue
        parameters = progress_map.anchor_parameters([anchor])[:, 0]
        matching = progress == np.asarray(anchor, dtype=progress.dtype)
        if anchor > progress_map.s[0]:
            left |= matching & (base.parameters <= parameters[0])
        if anchor < progress_map.s[-1]:
            right |= matching & (base.parameters >= parameters[-1])
    sides = []
    for columns, direction in ((np.flatnonzero(left), -np.inf), (np.flatnonzero(right), np.inf)):
        for column in columns:
            current = progress[column]
            side = float(np.nextafter(current, direction))
            permission = side
            if direction < 0 and any(rule.actor == 'robot' and
                    current == np.asarray(rule.interval[0], dtype=progress.dtype)
                    for rule in base.scene.contacts):
                permission = float(current)
            sides.append(dict(time_s=float(times[column]), progress=side,
                              permission_progress=permission))
    return exact + sides


def _pair_records(scene):
    """Enumerate the same registered pairs as the original scene objective."""
    pairs = [dict(first=f'robot/{pair.first}', second=f'robot/{pair.second}',
                  component='collision/robot/self') for pair in scene.robot_validator.required_pairs]
    fields = [(f'owner/{owner.owner_id}', owner.owner_id) for owner in scene.owners]
    fields.append(('background', 'background'))
    moving = {owner.owner_id for owner in scene.owners if owner.moving}
    for name in scene.robot_model._geometries:
        for target, identity in fields:
            if name not in scene.fixed_geometry_names or identity in moving:
                pairs.append(dict(first=f'robot/{name}', second=target,
                                  component=f'collision/robot/{identity}'))
    pairs.extend(dict(first=f'owner/{owner.owner_id}', second='background',
                      component=f'collision/{owner.owner_id}/background')
                 for owner in scene.owners if owner.moving)
    for first, second in combinations(scene.owners, 2):
        if first.moving or second.moving:
            if not first.moving:
                first, second = second, first
            pairs.append(dict(first=f'owner/{first.owner_id}', second=f'owner/{second.owner_id}',
                              component=f'collision/{first.owner_id}/{second.owner_id}'))
    return pairs


def _state_layout(base, records):
    """Retain original base ordering and share exact added state identities."""
    states = [dict(time_s=float(t), progress=float(p), permission_progress=float(p))
              for t, p in zip(base.times[0].numpy(), base.progress[0].numpy())]
    lookup = {}
    for column, row in enumerate(states):
        lookup.setdefault(tuple(row.values()), column)
    columns = []
    for record in records:
        state = {key: record[key] for key in ('time_s', 'progress', 'permission_progress')}
        key = tuple(state.values())
        if key not in lookup:
            lookup[key] = len(states)
            states.append(state)
        columns.append(lookup[key])
    return states, np.asarray(columns, dtype=np.int64)


class GeometryConstraints:
    """Bind registered entities and reuse the scene's finite permission API."""

    _permission = ScenePermissions._permission

    def __init__(self, computation, *, progress_map):
        self.computation, self.scene = computation, computation.scene
        self.progress_map = progress_map
        self._unknown_geometries = (set() if self.scene.gripper is None else
            set(self.scene.robot_model.geometry_depending_on(self.scene.gripper.joint_names)))
        pairs = _pair_records(self.scene)
        self.records = [{**state, **pair} for state in _query_records(computation, progress_map) for pair in pairs]
        # The installed robot's official geometry and fixed component relations
        # are reusable across tasks. Named feedback and calibration place it.
        # Task entities retain their observed TSDF representation under motion.
        self.geometries = {f'robot/{name}': row[1] for name, row in self.scene.robot_model._geometries.items()}
        self.geometries.update({f'owner/{owner.owner_id}': owner.query for owner in self.scene.owners})
        self.geometries['background'] = self.scene.background
        self.distance = None
        self._native_tables = {}
        self.last_evaluation = {}

    def _prepare_materials(self):
        """Prepare original field/native arrays once for this observed scene."""
        if self.distance is None:
            query_sources = {name for name in self.geometries if name.startswith('robot/')
                             and name[6:] not in self.scene.fixed_geometry_names}
            query_sources.update('owner/'+owner.owner_id for owner in self.scene.owners if owner.moving)
            self.distance = BatchDistance(self.geometries, query_sources=query_sources)
            for rule in self.scene.contacts:
                name = 'owner/'+rule.target_owner
                if rule.required and name not in self.distance.samples:
                    self.distance.samples[name] = MaterialSamples(self.geometries[name])
            self.scene._objective_material_samples = {
                (('robot', name[6:]) if name.startswith('robot/') else
                 ('field', name[6:] if name.startswith('owner/') else name)): samples
                for name, samples in self.distance.samples.items()}

    @property
    def row_count(self):
        return len(self.records)

    def workspace_bytes(self):
        """Count fixed row arrays; runtime process/device limits own total use."""
        rows, joints = len(self.records), len(self.computation.joint_names)
        return dict(pair_values_covectors_and_joint_factors=rows*8*(16+2*joints),
                    registered_native_tables=sum(table.storage_nbytes for table in self._native_tables.values()))

    # BEGIN immutable query layout belongs to the bound numerical problem.
    def _context(self, full_controls, bound, *, shared_context=None):
        """Share base FK/J and compute only added physical/relation query states."""
        base = self.computation
        states, columns = bound.states, bound.record_columns
        times, progress = bound.times, bound.progress
        permission_progress = bound.permission_progress
        controls = torch.as_tensor(full_controls, dtype=torch.float64)
        count = len(base.parameters)

        def batch(start):
            query = copy(base)
            query.times, query.progress = times[:, start:], progress[:, start:]
            q, dq = [(row[start:] @ controls + offset[start:])[None]
                     for row, offset in bound.state_rows]
            return linearized_robot_state(query, q, dq)

        with torch.no_grad():
            if shared_context is None:
                robot, jacobians = batch(0)
            else:
                first = shared_context['robot_state']
                first_j = shared_context['frame_jacobians']
                if len(states) == count:
                    # BEGIN numerical geometry retains values, not the objective graph.
                    links = {key: tuple(value.detach() for value in pair)
                             for key, pair in first[3].items()}
                    robot = (first[0].detach(), first[1].detach(), first[2], links,
                             tuple(value.detach() for value in first[4]),
                             tuple(value.detach() for value in first[5]))
                    jacobians = {key: value.detach() for key, value in first_j.items()}
                    # END candidate diagnostics keep these same independent values.
                else:
                    added, added_j = batch(count)
                    join = lambda a, b: torch.cat((a.detach(), b.detach()), dim=1)
                    links = {key: tuple(join(a, b) for a, b in zip(first[3][key], added[3][key]))
                             for key in first[3]}
                    robot = (join(first[0], added[0]), join(first[1], added[1]), first[2], links,
                             tuple(join(a, b) for a, b in zip(first[4], added[4])),
                             tuple(join(a, b) for a, b in zip(first[5], added[5])))
                    jacobians = {key: join(first_j[key], added_j[key]) for key in first_j}
            complete_q, complete_dq, names, links, tool, _ = robot
            hand = (None if self.scene.gripper is None else self.scene.gripper.state(
                progress, complete_q, names, trajectory_time=times))
            poses = {}
            for name, (frame, local_p, local_q) in bound.robot_offsets.items():
                p, q = links[frame]
                poses[f'robot/{name}'] = (p + quaternion_rotate(q, local_p),
                    quaternion_multiply(q, local_q))
            relationship_components = {}
            owners = self.scene._owner_poses(*tool, progress, None, relationship_components)
            poses.update({f'owner/{key}': value for key, value in owners.items()})
            identity = (torch.zeros_like(tool[0]), torch.zeros_like(tool[1]))
            identity[1][..., 0] = 1.
            poses['background'] = identity
            policies, contact_conditions = {}, {}
            for rule in self.scene.contacts:
                actor = tool if rule.actor == 'robot' else owners[rule.actor]
                policy, normal, uncertain = self.scene._permission(rule, actor, owners[rule.target_owner],
                    permission_progress, None if hand is None else hand.coordinates,
                    uncertain_hand=None if hand is None else hand.uncertain)
                policies[rule.rule_id] = policy
                contact_conditions[rule.rule_id] = (normal, uncertain)
        return dict(states=states, record_columns=columns, poses=poses, geometries=self.geometries,
            matrices={key: _matrices(value) for key, value in poses.items()},
            links=links, frame_jacobians=jacobians, complete_joint_names=names,
            complete_q=complete_q, complete_dq=complete_dq,
            policies=policies, contact_conditions=contact_conditions, hand=hand, progress=progress,
            relationship_components=relationship_components, owner_poses=owners,
            query_state_count=len(states), added_fk_state_count=0 if shared_context is not None
                and len(states) == count else len(states)-count if shared_context is not None else len(states))
    # END current controls change values, not the query clock or registered offsets.

    def bind_solver(self, terminal):
        """Bind one immutable sparse layout to the same scene query arrays."""
        self._prepare_materials()
        return BoundGeometry(self, terminal)

    def describe(self):
        """Expose actual query cost and the scene location of its worst row."""
        return dict(scope='original_scene_queries_and_bracketed_native_stationary_times',
            rows=self.records, last_evaluation=self.last_evaluation,
            preparation_seconds=None if self.distance is None else self.distance.preparation_seconds,
            fixed_sample_counts={} if self.distance is None else
                {name: len(points.points) for (name, excluded), points in self.distance.inventories.items()
                 if not excluded},
            native_pair_table_count=len(self._native_tables))


class BoundGeometry:
    """Own fixed coefficient slots and original current/release state sources."""

    def __init__(self, owner, terminal):
        self.owner, self.terminal = owner, terminal
        self.records = tuple(dict(row) for row in owner.records)
        self.states, self.record_columns = _state_layout(owner.computation, self.records)
        base = owner.computation
        self.joints = len(base.joint_names)
        self.variables = terminal.matrix.shape[1]*self.joints
        times = np.asarray([row['time_s'] for row in self.states])
        # BEGIN physical clocks, spline rows and official fixed offsets are bound once.
        # State identities include event-side relation/permission coordinates;
        # reusing these arrays preserves their distinct rows and first-release anchor.
        self.times = torch.tensor(times[None], dtype=torch.float64)
        self.progress = torch.tensor([[row['progress'] for row in self.states]], dtype=torch.float64)
        self.permission_progress = torch.tensor(
            [[row['permission_progress'] for row in self.states]], dtype=torch.float64)
        self.state_rows = tuple(tuple(torch.tensor(value, dtype=torch.float64) for value in
            base.domain._curve_map.query(times/base.duration, order)) for order in (0, 1))
        self.coefficients = self.state_rows[0][0].numpy() @ terminal.matrix
        self.robot_offsets = {name: (frame, torch.tensor(local[:3, 3], dtype=torch.float64),
            torch.tensor(Rotation.from_matrix(local[:3, :3]).as_quat()[[3, 0, 1, 2]], dtype=torch.float64))
            for name, (frame, _, _, local) in owner.scene.robot_model._geometries.items()}
        self.record_times = np.array([row['time_s'] for row in self.records])
        # END only immutable model data is shared across coefficient evaluations.
        self.sources = {(name, column): self._source(name, column)
                        for name in owner.geometries for column in range(len(self.states))}
        columns = []
        self.source_samples = np.full((len(self.records), 2), -1, dtype=np.int64)
        self.frame_groups = {}
        groups = {}
        for ordinal, (row, column) in enumerate(zip(self.records, self.record_columns)):
            samples = []
            for side, name in enumerate((row['first'], row['second'])):
                origin = self.sources[name, int(column)]
                if origin is not None:
                    frame, sample = origin
                    self.source_samples[ordinal, side] = sample
                    self.frame_groups.setdefault((name, frame, side), []).append(ordinal)
                    samples.append(sample)
            columns.append(coefficient_columns(self.coefficients[samples], np.arange(self.joints), self.joints)
                           if samples else np.empty(0, dtype=np.int64))
            if row['first'].startswith('robot/') and row['second'].startswith('robot/'):
                groups.setdefault((row['first'], row['second']), []).append(ordinal)
        # BEGIN each original physical query owns its own local inequality.
        # Native interval minima below extend only their left row's support.
        self.row_keys = tuple(('geometry', *pair_identity(row)) for row in self.records)
        # END query coordinates and the feasible set retain their original scope.
        self.native_rows = {pair: np.asarray(rows, dtype=np.int64) for pair, rows in groups.items()}
        self.frame_groups = {key: np.asarray(rows, dtype=np.int64) for key, rows in self.frame_groups.items()}
        # BEGIN declarations identify the rows requiring finite permission math.
        owner_by_name = {value.owner_id: value for value in owner.scene.owners}
        permission_rows, hand_rows = [], []
        for ordinal, record in enumerate(self.records):
            first, second = record['first'], record['second']
            robot_names = {name[6:] for name in (first, second) if name.startswith('robot/')}
            if robot_names & owner._unknown_geometries:
                hand_rows.append(ordinal)
            if first.startswith('robot/') and second.startswith('robot/'):
                continue
            actor = 'robot' if first.startswith('robot/') else first[6:]
            target = second.removeprefix('owner/')
            attachment = (target in owner_by_name and bool(owner_by_name[target].attachments)
                and ((actor == 'robot' and first[6:] in owner.scene.tool_geometry_names)
                     or (actor in owner_by_name and bool(owner_by_name[actor].attachments))))
            contact = any((actor == 'robot' and rule.actor == 'robot' and rule.target_owner == target
                           and rule.robot_body == first[6:]) or
                          (actor != 'robot' and {rule.actor, rule.target_owner} == {actor, target})
                          for rule in owner.scene.contacts)
            support = any(model.relation.actor == actor and model.relation.target_owner == target
                          for model in owner.scene._initial_support_models.values())
            if attachment or contact or support:
                permission_rows.append(ordinal)
        self.permission_rows = np.asarray(permission_rows, dtype=np.int64)
        self.hand_rows = np.asarray(hand_rows, dtype=np.int64)
        # END trial signs never change these registered structural dependencies.
        for pair in self.native_rows:
            if pair not in owner._native_tables:
                owner._native_tables[pair] = NativePairTable(
                    [owner.geometries[name] for name in pair], np.array([[0, 1]], dtype=np.intp))
        # BEGIN pair groups share the same ordered clock and activation rule.
        # Native row activation depends only on the declared hand uncertainty.
        # Equal clock/hand membership therefore permits one multi-pair query;
        # the row matrix restores every original pair identity after the call.
        grouped = {}
        for pair, rows in self.native_rows.items():
            key = (tuple(self.record_columns[rows]), tuple(np.isin(rows, self.hand_rows)))
            grouped.setdefault(key, []).append(pair)
        self.native_groups = []
        for pairs in grouped.values():
            names = tuple(dict.fromkeys(name for pair in pairs for name in pair))
            indices = np.array([[names.index(a), names.index(b)] for a, b in pairs], dtype=np.intp)
            rows = np.stack([self.native_rows[pair] for pair in pairs], axis=1)
            capacity = min(MAX_BATCH_SIZE, len(rows))
            self.native_groups.append(dict(pairs=tuple(pairs), names=names, indices=indices, rows=rows,
                table=NativePairTable([owner.geometries[name] for name in names], indices),
                poses=np.broadcast_to(np.eye(4), (capacity, len(names), 4, 4)).copy(),
                projection=np.zeros((capacity, len(pairs)), dtype=bool)))
        # END scalar interval queries retain their separate original pair tables.
        self.native_request = coal.DistanceRequest()
        self.native_request.enable_signed_distance = True
        # This unit only normalizes inequalities. No physical clearance or
        # shape is changed by expressing native metres in observed-grid units.
        self.distance_unit = min(geometry.voxel_size for geometry in owner.geometries.values()
                                 if isinstance(geometry, ObservedField))
        # BEGIN one native interval owns every possible interior spline column.
        self.temporal = NativeIntervalDistance(self)
        for row, extra in self.temporal.support_columns.items():
            columns[row] = np.union1d(columns[row], extra)
        self.pattern = SparseRows(columns, self.variables)
        self.columns_by_row = self.pattern.columns_by_row
        # END field rows and first-release sources retain their original support.

    def _source(self, name, column):
        """Resolve original pose composition to a named frame/query or constant."""
        scene, base = self.owner.scene, self.owner.computation
        if name.startswith('robot/'):
            return scene.robot_model._geometries[name[6:]][0], column
        if name == 'background':
            return None
        owner = next(row for row in scene.owners if row.owner_id == name[6:])
        progress = self.states[column]['progress']
        source = None
        for index, relation in enumerate(owner.attachments):
            if bool(relation.active(np.asarray(progress))):
                source = base.problem.tool_link, column
            elif relation.release_condition_id is not None and progress >= relation.interval[1]:
                if (owner.owner_id, index) in scene._fixed_release_poses:
                    source = None
                else:
                    # This is the original _release_pose event membership and
                    # first occurrence, including same-progress later plateaus.
                    matches = [i for i, row in enumerate(self.states)
                               if abs(row['progress']-relation.interval[1]) <= 1e-7]
                    if not matches:
                        raise ValueError(f'Missing declared release query for owner {owner.owner_id!r}')
                    source = base.problem.tool_link, matches[0]
        return source

    def evaluate(self, full_controls, *, shared_context=None):
        """Evaluate one coefficient point with grouped numerical library calls."""
        started = perf_counter()
        check = self.owner.scene.check_request
        # BEGIN an interrupted batch retains its own coordinates and work boundary.
        report = dict(state='producing', stage='robot_states',
            full_controls=np.asarray(full_controls).tolist(),
            fixed_row_count=len(self.records), completed_row_count=0,
            native_batches_submitted=0, native_batches_completed=0,
            native_pair_queries_submitted=0, native_pair_queries_completed=0,
            seconds=dict(native_query=0.))
        self.owner.last_evaluation = report
        # END later failures cannot inherit another numerical point's result.
        # BEGIN the context consumes the already bound original query layout.
        context = self.owner._context(full_controls, self, shared_context=shared_context)
        # END no per-iterate record-to-state reconstruction is needed.
        seconds = dict(context=perf_counter()-started, permissions=0., native_query=0.,
                       field_pairs=0., rigid_to_joint=0.)
        stamp = perf_counter()
        permissions = [None]*len(self.records)
        modes = np.full(len(self.records), 'active', dtype=object)
        if context['hand'] is not None:
            uncertain = context['hand'].uncertain[0].numpy()
            modes[self.hand_rows[uncertain[self.record_columns[self.hand_rows]]]] = 'unmodelled_motion'
        for ordinal in self.permission_rows:
            if modes[ordinal] == 'active':
                permissions[ordinal], modes[ordinal] = self.owner._permission(
                    context, self.records[ordinal], int(self.record_columns[ordinal]))
        active = modes == 'active'
        seconds['permissions'] = perf_counter()-stamp
        stamp = perf_counter()
        report['stage'] = 'field_submission'
        pending = self.owner.distance.submit(records=self.records,
            columns=self.record_columns, matrices=context['matrices'], permissions=permissions,
            active=active, check_request=check)
        values, covectors, witnesses = pending.values, pending.covectors, pending.witnesses
        field_submission = perf_counter()-stamp
        native_count = 0
        native_distances = np.full(len(self.records), np.nan)
        native_numeric = np.full((len(self.records), 12), np.nan)
        stamp = perf_counter()
        report['stage'] = 'native_pairs'
        # BEGIN all native pairs with equal clocks share one maintained C++ call.
        for group in self.native_groups:
            selected = group['rows'][active[group['rows'][:, 0]]]
            capacity = len(group['poses'])
            for start in range(0, len(selected), capacity):
                check()
                block = selected[start:start+capacity]
                columns = self.record_columns[block[:, 0]]
                matrices = group['poses'][:len(block)]
                for side, name in enumerate(group['names']):
                    matrices[:, side] = context['matrices'][name][0, columns]
                rows = block.ravel()
                report['native_batches_submitted'] += 1
                report['native_pair_queries_submitted'] += len(rows)
                numeric = group['table'].query(matrices, group['projection'][:len(block)],
                                               self.native_request).reshape(-1, 12)
                report['native_batches_completed'] += 1
                report['native_pair_queries_completed'] += len(rows)
                report['seconds']['native_query'] = perf_counter()-stamp
                origins = [matrices[:, group['indices'][:, side], :3, 3].reshape(-1, 3)
                           for side in range(2)]
                # BEGIN the same negative tail removes inactive-distance noise.
                # max(-d/h,-1) <= 0 is exactly d >= 0; this numerical mapping
                # changes neither registered shape nor physical clearance.
                # The unclipped metre distance remains in native_distances.
                near = numeric[:, 0] < self.distance_unit
                normal = numeric[:, 7:10]/self.distance_unit*near[:, None]
                values[rows] = np.maximum(-numeric[:, 0]/self.distance_unit, -1.)
                # END far rows have the derivative of their constant tail.
                covectors[rows, 0, :3], covectors[rows, 1, :3] = normal, -normal
                covectors[rows, 0, 3:] = np.cross(numeric[:, 1:4]-origins[0], normal)
                covectors[rows, 1, 3:] = -np.cross(numeric[:, 4:7]-origins[1], normal)
                witnesses[rows], native_distances[rows] = numeric[:, 1:4], numeric[:, 0]
                native_numeric[rows] = numeric
                native_count += len(rows)
        seconds['native_query'] = perf_counter()-stamp
        # END exact native numeric rows retain their original index and witness.
        # BEGIN independent CPU interval work runs before the common GPU wait.
        stamp = perf_counter()
        report['stage'] = 'native_temporal'
        interval_winners, temporal_report = self.temporal.evaluate(full_controls, context, native_numeric, active)
        seconds['native_temporal'] = perf_counter()-stamp
        stamp = perf_counter()
        report['stage'] = 'field_consumption'
        values, covectors, observed = pending.finish()
        seconds['field_pairs'] = field_submission+perf_counter()-stamp
        witnesses = observed.pop('witness_world')
        unknown_counts = observed.pop('unknown_counts')
        query_counts = observed.pop('query_counts')
        # END subsequent rigid derivative assembly uses both completed providers.
        stamp = perf_counter()
        joint_factors = np.zeros((len(self.records), 2, self.joints))
        controlled = [context['complete_joint_names'].index(name) for name in self.owner.computation.joint_names]
        jacobian_arrays = {frame: value[0].detach().numpy()[:, :, controlled]
                           for frame, value in context['frame_jacobians'].items()}
        frame_origins = {frame: value[0][0].detach().numpy() for frame, value in context['links'].items()}
        for (name, frame, side), rows in self.frame_groups.items():
            samples = self.source_samples[rows, side]
            jacobian = jacobian_arrays[frame][samples]
            frame_origin = frame_origins[frame][samples]
            entity_origin = context['matrices'][name][0, self.record_columns[rows], :3, 3]
            linear, angular = covectors[rows, side, :3], covectors[rows, side, 3:]
            shifted = angular+np.cross(entity_origin-frame_origin, linear)
            joint_factors[rows, side] = (np.einsum('ni,nij->nj', linear, jacobian[:, :3])
                                        + np.einsum('ni,nij->nj', shifted, jacobian[:, 3:]))
        seconds['rigid_to_joint'] = perf_counter()-stamp
        # BEGIN original interval extrema share this value and derivative point.
        # BEGIN each point owns only its mutable actual-witness time array.
        query_times = self.record_times.copy()
        # END fixed original row times remain in the bound layout.
        for winner in interval_winners:
            row, numeric = winner['row'], winner['numeric']
            values[row] = -numeric[0]/self.distance_unit
            native_distances[row], witnesses[row] = numeric[0], numeric[1:4]
            query_times[row] = winner['time_s']
        # END original rows retain explicit actual-time witness provenance.
        index = int(np.argmax(values))
        # BEGIN the spatial witness names its actual time; row identity stays explicit.
        identity = dict(self.records[index])
        if query_times[index] != identity['time_s']:
            mapping = self.owner.progress_map
            progress = float(np.interp(query_times[index]/self.owner.computation.duration, mapping.u, mapping.s))
            identity.update(time_s=float(query_times[index]), progress=progress, permission_progress=progress)
        worst = dict(identity=identity, row_identity=dict(self.records[index]),
            row=index, value=float(values[index]),
            mode=modes[index], unknown_query_count=int(unknown_counts[index]),
            queried_point_count=int(query_counts[index]),
            witness_first_world=None if not np.isfinite(witnesses[index]).all() else witnesses[index].tolist(),
            signed_distance_m=None if not np.isfinite(native_distances[index]) else float(native_distances[index]))
        # END display and rejection records consume the same original-shape witness.
        self.owner.last_evaluation = dict(state='completed', full_controls=np.asarray(full_controls).tolist(),
            fixed_row_count=len(self.records), completed_row_count=len(self.records),
            active_count=int(active.sum()), inactive_count=int((~active).sum()),
            maximum_condition=worst, native_pair_queries_completed=native_count,
            native_temporal=temporal_report, native_batches_completed=report['native_batches_completed'],
            observed_queries=observed, seconds=seconds, elapsed_seconds=perf_counter()-started,
            query_state_count=context['query_state_count'],
            scope='Original scene queries plus bracketed robot-distance stationary times')
        check()
        point = GeometryPoint(self, values, joint_factors, witnesses, unknown_counts, query_counts)
        point.native_distances_m = native_distances
        point.interval_winners = interval_winners
        point.query_times = query_times
        point.context = context
        point.report = self.owner.last_evaluation
        return point


class GeometryPoint:
    """Retain one value result and write its sparse coefficient derivatives."""

    def __init__(self, bound, values, factors, witnesses, unknown_counts, query_counts):
        self.bound, self.spatial_values, self.factors = bound, values, factors
        self.values = values
        self.witnesses = witnesses
        self.unknown_counts, self.query_counts = unknown_counts, query_counts

    def jacobian_values(self):
        pattern, bound = self.bound.pattern, self.bound
        values = np.zeros(pattern.nnz)
        for side in range(2):
            records = pattern.rows
            samples = bound.source_samples[records, side]
            selected = np.flatnonzero(samples >= 0)
            columns, rows = pattern.columns[selected], records[selected]
            values[selected] += (bound.coefficients[samples[selected], columns//bound.joints]
                * self.factors[rows, side, columns % bound.joints])
        # BEGIN envelope derivatives use the active interior state's own basis.
        for winner in self.interval_winners:
            target = pattern.slices[winner['row']]
            columns = pattern.columns[target]
            basis = (bound.owner.computation.domain._curve_map.query(
                np.array([winner['time_s']/bound.owner.computation.duration]), 0)[0]
                @bound.terminal.matrix)[0]
            values[target] = (-winner['gradient'][columns % bound.joints]/bound.distance_unit
                              *basis[columns//bound.joints])
        # END stationarity removes the derivative of the selected root time.
        return values
# END geometry rows share one value/derivative and candidate-diagnostic source.
