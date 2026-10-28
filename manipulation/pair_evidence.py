"""One owned scene-query result, shared by indexed validation consumers.

The numerical native block has Coal's original float64 columns.  Observed
queries retain their complete original dictionaries, including variable-sized
witness enclosures and model-specific fields.  The common scalar columns are
consumer indices over those results; they introduce no geometric calculation.
Only selected diagnostics, optimization witnesses and completed output records
are materialized as ordinary row dictionaries.
"""

from dataclasses import dataclass

import numpy as np


CLEAR, INTERSECTION, UNRESOLVED, UNMODELLED, INTERNAL = range(5)
STATUS_NAMES = ("clear", "intersection", "unresolved", "unmodelled_motion", "internal_assembly")
STATUS_CODES = {name: index for index, name in enumerate(STATUS_NAMES)}
NONE, FINITE_REGION, RIGID_ASSEMBLY = range(3)
PERMISSION_NAMES = ("none", "finite_region", "declared_rigid_assembly")
PERMISSION_CODES = {name: index for index, name in enumerate(PERMISSION_NAMES)}


@dataclass(frozen=True)
class PairDescriptor:
    """Declared pair identity, shared across every state of one scene query.

    ``first`` and ``second`` are the original display names.  Namespaced entity
    identities distinguish robot solids, observed owners and the background.
    Native policy/source fields retain the caller's fixed PairQuery contract.
    """

    component: str
    first: str
    second: str
    entity_first: str
    entity_second: str
    policy: str = None
    policy_source: str = None

    @property
    def key(self):
        return self.component, self.entity_first, self.entity_second

    def display(self):
        """Build the small identity record used by final interval reports."""
        return {"component": self.component, "first": self.first, "second": self.second,
                "geometry_entity_key": {"first": self.entity_first, "second": self.entity_second}}


class PairBatch:
    """Complete rectangular [expanded query state, declared entity pair] data.

    A physical time may have several relation-side queries.  Its expanded
    ``state_index`` and actual progress/permission progress remain independent
    of ``query_index`` and ``side``, which are assigned by the scene-boundary
    caller.  This object owns one call only; subsequent calls allocate another
    object.  Array users read the result after ``seal``.  Observed payloads are
    transferred from their original producer and are subsequently read only.
    """

    def __init__(self):
        self.descriptors = []
        self.pair_columns = {}
        self.observed_payloads = {}
        self._observed_permissions = {}
        self.native_numeric = None
        self.native_count = 0
        self._query_layout_bound = False

    def bind_clock(self, times, progress, permission_progress):
        """Own the original clock dtype and expanded two-axis query layout."""
        self.state_shape = tuple(np.asarray(times).shape)
        self.state_index = np.indices(self.state_shape, dtype=np.intp).reshape(2, -1).T.copy()
        self.time_s = np.asarray(times).reshape(-1).copy()
        self.progress = np.asarray(progress).reshape(-1).copy()
        self.permission_progress = np.asarray(permission_progress).reshape(-1).copy()
        self.state_count = self.time_s.size
        self.query_index = self.state_index.copy()
        self.side = np.zeros(self.state_count, dtype=np.int8)

    @property
    def native_pair_count(self):
        """Number of leading descriptor columns owned by the native kernel."""
        return self.native_count

    def _register(self, descriptor):
        column = len(self.descriptors)
        self.descriptors.append(descriptor)
        self.pair_columns[descriptor.key] = column
        return column

    def begin_native(self, pairs, *, boundary, include_bounds):
        """Reserve native output once; each unchanged kernel chunk fills a slice.

        The transient native chunk retains the original kernel's allocation
        and error boundary.  Its values are copied into the one result owner;
        no Python float/list objects are created for clear native pairs.
        """
        self.native_count = len(pairs)
        self.native_boundary = float(boundary)
        self.native_include_bounds = bool(include_bounds)
        self.native_numeric = np.empty((self.state_count, len(pairs), 12), dtype=np.float64)
        self._native_unmodelled = np.zeros((self.state_count, len(pairs)), dtype=bool)
        for pair in pairs:
            self._register(PairDescriptor("collision/robot/self", pair.first, pair.second,
                f"robot/{pair.first}", f"robot/{pair.second}", pair.policy, pair.source))

    def retain_native_chunk(self, start, numeric, unmodelled):
        """Retain every raw column, including rows whose motion is unmodelled."""
        end = start + len(numeric)
        self.native_numeric[start:end] = numeric
        self._native_unmodelled[start:end] = unmodelled

    def append_observed(self, index, component, first, second, evidence, *,
                        geometry_entity_key, permission_kind="none"):
        """Keep one original observed/declared-relation payload without wrapping it.

        The registered scene enumerates each observed pair over every expanded
        state, including skipped internal or uncertain states.  Model-specific
        payload keys are preserved intact; a twelve-column native layout does
        not define the schema of an observed-field query.
        """
        descriptor = PairDescriptor(component, first, second,
            geometry_entity_key["first"], geometry_entity_key["second"])
        column = self.pair_columns.get(descriptor.key)
        if column is None:
            column = self._register(descriptor)
            self.observed_payloads[column] = [None] * self.state_count
            self._observed_permissions[column] = np.empty(self.state_count, dtype=np.int8)
        state = int(np.ravel_multi_index(index, self.state_shape))
        self.observed_payloads[column][state] = evidence
        self._observed_permissions[column][state] = PERMISSION_CODES[permission_kind]

    def seal(self):
        """Prepare common scalar columns and one original traversal index.

        Missing bounds are represented by a separate Boolean validity column.
        A numeric NaN therefore remains a numeric NaN; it is not substituted
        for an absent value.  The complete original payload still determines
        field presence when a final row is materialized.
        """
        shape = (self.state_count, len(self.descriptors))
        self.status = np.empty(shape, dtype=np.int8)
        self.permission_kind = np.zeros(shape, dtype=np.int8)
        self.lower = np.zeros(shape, dtype=np.float64)
        self.full_lower = np.zeros(shape, dtype=np.float64)
        self.upper = np.zeros(shape, dtype=np.float64)
        self.lower_valid = np.zeros(shape, dtype=bool)
        self.full_lower_valid = np.zeros(shape, dtype=bool)
        self.upper_valid = np.zeros(shape, dtype=bool)
        self.empty_set = np.zeros(shape, dtype=bool)
        self.witness_present = np.zeros(shape, dtype=bool)
        count = self.native_count
        if count:
            distance = self.native_numeric[..., 0]
            status = np.where(distance > self.native_boundary, CLEAR,
                np.where(distance < -self.native_boundary, INTERSECTION, UNRESOLVED))
            self.status[:, :count] = np.where(self._native_unmodelled, UNMODELLED, status)
            self.witness_present[:, :count] = ~self._native_unmodelled
            if self.native_include_bounds:
                self.lower[:, :count] = self.native_numeric[..., 11]
                self.lower_valid[:, :count] = ~self._native_unmodelled
        for column, payloads in self.observed_payloads.items():
            self.permission_kind[:, column] = self._observed_permissions[column]
            for state, payload in enumerate(payloads):
                self.status[state, column] = STATUS_CODES[payload["status"]]
                for field, values, valid in (
                        ("lower_m", self.lower, self.lower_valid),
                        ("full_entity_lower_m", self.full_lower, self.full_lower_valid),
                        ("upper_m", self.upper, self.upper_valid)):
                    value = payload.get(field)
                    if value is not None:
                        values[state, column] = value
                        valid[state, column] = True
                self.empty_set[state, column] = bool(payload.get("empty_set"))
                self.witness_present[state, column] = (
                    payload.get("witness_first") is not None and payload.get("witness_second") is not None)
        # Native rows were consumed state-major; observed rows pair-major.
        # Keeping that order also preserves first-witness tie selection and
        # the order in which relation-side groups were originally inserted.
        native = np.indices((self.state_count, count), dtype=np.intp).reshape(2, -1).T
        observed = np.indices((len(self.descriptors) - count, self.state_count),
                              dtype=np.intp).reshape(2, -1)[::-1].T
        observed[:, 1] += count
        self.order = np.concatenate((native, observed), axis=0)
        self.descriptors = tuple(self.descriptors)
        for value in (self.time_s, self.progress, self.permission_progress, self.state_index,
                self.status, self.permission_kind, self.lower, self.lower_valid,
                self.full_lower, self.full_lower_valid, self.upper, self.upper_valid,
                self.empty_set, self.witness_present, self.order):
            value.flags.writeable = False
        if self.native_numeric is not None:
            self.native_numeric.flags.writeable = False
        return self

    def set_query_layout(self, query_index, side):
        """Bind exact physical query identities after relation-side expansion."""
        self.query_index = np.asarray(query_index, dtype=np.intp).reshape(self.state_count, 2).copy()
        self.side = np.asarray(side, dtype=np.int8).reshape(self.state_count).copy()
        self.query_index.flags.writeable = self.side.flags.writeable = False
        self._query_layout_bound = True

    def materialize(self, state, column, *, query_layout=True):
        """Return one original complete row at a diagnostic/output boundary."""
        state, column = int(state), int(column)
        descriptor = self.descriptors[column]
        row = {"state_index": self.state_index[state].tolist(),
            "time_s": float(self.time_s[state]), "progress": float(self.progress[state]),
            "permission_progress": float(self.permission_progress[state]),
            **descriptor.display(),
            "permission_kind": PERMISSION_NAMES[int(self.permission_kind[state, column])]}
        if column < self.native_count:
            if self._native_unmodelled[state, column]:
                row.update(status="unmodelled_motion", lower_m=None)
            else:
                numeric = self.native_numeric[state, column]
                distance = float(numeric[0])
                relation = ("separated" if distance > self.native_boundary else "penetrating"
                            if distance < -self.native_boundary else "numerical_boundary")
                row.update(policy=descriptor.policy, policy_source=descriptor.policy_source,
                    signed_distance_m=distance, relation=relation,
                    witness_first=numeric[1:4].tolist(), witness_second=numeric[4:7].tolist(),
                    normal_first_to_second=numeric[7:10].tolist(),
                    witness_identity_error_m=float(numeric[10]),
                    status={"separated": "clear", "penetrating": "intersection",
                            "numerical_boundary": "unresolved"}[relation],
                    lower_m=float(numeric[11]) if self.native_include_bounds else None,
                    distance_scope="full_entities")
        else:
            row.update(self.observed_payloads[column][state])
        if query_layout and self._query_layout_bound:
            row.update(query_index=self.query_index[state].tolist(), side=int(self.side[state]))
        return row


@dataclass(frozen=True)
class EvidenceState:
    """One physical time/relation side, retaining every expanded source row.

    Rectangular scene queries emit every registered pair for every source
    row, including skipped relations. Duplicate rows keep their original
    order; interval lookup has the original dictionary's last-row semantics.
    Progress outside the block is the original physical query's progress.
    """
    time_s: float
    side: int
    progress: float
    residual: float
    batch: PairBatch
    source_rows: tuple

    @property
    def pair_columns(self):
        return self.batch.pair_columns if self.source_rows else {}

    @property
    def source(self):
        return self.source_rows[-1]

    def tokens(self):
        """Yield indices in the original within-state evidence order."""
        for source in self.source_rows:
            for pair in range(self.batch.native_pair_count):
                yield source, pair
        for pair in range(self.batch.native_pair_count, len(self.batch.descriptors)):
            for source in self.source_rows:
                yield source, pair

    def pair_record(self, pair):
        row = self.batch.materialize(self.source, pair)
        return {key: value for key, value in row.items()
                if key not in {'state_index', 'query_index', 'time_s', 'side'}}

def _geometry_query_states(evidence, times, progress, residuals):
    """Group state indices once; entity identities stay in the shared block."""
    groups = {(float(time), 0): [] for time in times}
    columns = {(float(time), 0): column for column, time in enumerate(times)}
    # Empty pair sets originally create only the physical exact states.
    if evidence.descriptors:
        for source, (_, column) in enumerate(evidence.query_index):
            key = float(times[column]), int(evidence.side[source])
            groups.setdefault(key, []).append(source)
            columns.setdefault(key, column)
    return [EvidenceState(time, side, float(progress[columns[(time, side)]]),
                float(residuals[columns[(time, side)]]), evidence, tuple(sources))
            for (time, side), sources in groups.items()]

def selected_witness_records(*, states=None, batch=None):
    """Select original witnesses from arrays, then build only selected records.

    Initial task checks consume grouped state order. Later interval queries
    consume the scene's original production order. These two orders differ
    for observed pairs; both preserve first-on-equal ranking and discoveries.
    """
    intersections, closest = [], {}
    total = 0

    def sources():
        if states is not None:
            for state in states:
                for source, pair in state.tokens():
                    yield state.batch, source, pair, state.time_s, state.progress
        else:
            for source, pair in batch.order:
                yield batch, source, pair, batch.time_s[source], batch.progress[source]

    for block, source, pair, time, progress in sources():
        total += 1
        status = block.status[source, pair]
        if status in (UNMODELLED, INTERNAL):
            continue
        if status == INTERSECTION:
            intersections.append((block, source, pair, time, progress))
        elif block.witness_present[source, pair]:
            descriptor = block.descriptors[pair]
            key = descriptor.entity_first, descriptor.entity_second
            if pair < block.native_pair_count:
                lower = float(block.lower[source, pair]) if block.lower_valid[source, pair] else None
                upper = float(block.upper[source, pair]) if block.upper_valid[source, pair] else None
            else:
                # Observed payloads already own these original Python values.
                # Retaining them preserves Python tuple comparison identity,
                # including two rows sharing the same numeric NaN object.
                payload = block.observed_payloads[pair][source]
                lower, upper = payload.get('lower_m'), payload.get('upper_m')
            rank = float(lower or 0.), float(upper or np.inf), float(time)
            if key not in closest or rank < closest[key][0]:
                closest[key] = rank, block, source, pair, time, progress
    selected = intersections + [value[1:] for value in closest.values()]
    records = []
    for block, source, pair, time, progress in selected:
        row = block.materialize(source, pair, query_layout=False)
        if states is not None:
            row = {key: value for key, value in row.items()
                   if key not in {'state_index', 'query_index', 'time_s', 'side'}}
        entity = row['geometry_entity_key']
        records.append({**row, 'time_s': float(time),
            'progress': float(row.get('progress', progress)),
            'permission_progress': float(row.get('permission_progress', progress)),
            'first': entity['first'], 'second': entity['second']})
    return records, total
