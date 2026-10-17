"""Fresh CUDA evidence in the original finite-candidate commit order.

Sixteen parent proposals are prepared at a time. CPU refits keep their original
arithmetic, and a future interruption is exposed only after earlier events.
"""
from time import perf_counter
import numpy as np
from .candidate_batch import DenseProjector
from .cuda_evidence import CudaEvidence
from .features import ResponseField
from .proposals import transform_key
from .refinement import partial_refits

# The CUDA evaluator implements this exact sampling contract. Freeze its
# identity so later monkey-patching cannot silently change the claimed API.
_RESPONSE_SAMPLE = ResponseField.sample


def validate_appearance(appearance, surface):
    """Reject another sampling implementation instead of silently bypassing it."""
    bound = getattr(appearance, 'sample', None)
    if (not isinstance(appearance, ResponseField)
            or appearance.surface is not surface
            or getattr(bound, '__func__', None) is not _RESPONSE_SAMPLE
            or getattr(bound, '__self__', None) is not appearance):
        raise ValueError('Online batching requires the same surface and an unmodified bound ResponseField.sample implementation')
# END inherited methods are valid; overridden or rebound methods are not.


def _mapped_candidate(source_pack, transform):
    """Preserve each original CPU transform call's shape and operation order."""
    from transfer.sim3 import apply_sim3
    queries = np.asarray(source_pack['query_xyz'], dtype=float)
    normals = np.asarray(source_pack['query_normals'], dtype=float)
    mapped, oriented = apply_sim3(queries, *transform), normals @ transform[1].T
    checks = apply_sim3(source_pack['check_xyz'], *transform)
    check_normals = np.asarray(source_pack['check_normals']) @ transform[1].T
    center, normal = apply_sim3(source_pack['center'], *transform), transform[1] @ source_pack['normal']
    return dict(query_points=mapped, query_normals=oriented, check_points=checks,
        check_normals=check_normals, center=center, normal=normal)



def _project_evidence_records(records, projector, evidence):
    """Keep baseline projection shapes; download geometry and scores together."""
    import torch
    from transfer.multiview_geometry import STATE_NAMES
    outputs, measured = [None]*len(records), [None]*len(records)
    for view in range(2):
        selected = [(index, record) for index, record in enumerate(records) if record['view'] == view]
        if not selected:
            continue
        for _, record in selected:
            if record['role'] == 'query' and len(record['points']) != len(evidence.offsets)-1:
                raise ValueError('Each candidate query call must contain the frozen Q rows')
        points = projector.tensor(np.concatenate([r['points'] for _, r in selected]), torch.float64)
        normals = projector.tensor(np.concatenate([r['normals'] for _, r in selected]), torch.float64)
        query_rows = np.concatenate([np.arange(len(r['points']), dtype=np.int64)
            if r['role'] == 'query' else np.full(len(r['points']), -1, dtype=np.int64)
            for _, r in selected])
        edges = np.r_[0, np.cumsum([len(r['points']) for _, r in selected])]
        groups = list(zip(edges[:-1].tolist(), edges[1:].tolist()))
        packed = projector.project(points, normals, view)
        sampled = evidence.sample_view(packed, projector.tensor(query_rows, torch.int64), view, groups)
        pixels, integer, boolean = [value.cpu().numpy() for value in packed]
        values = [value.cpu().numpy() for value in sampled]
        offset = 0
        for index, record in selected:
            stop = offset+len(record['points'])
            rows = slice(offset, stop)
            result = {name: boolean[rows, column] for column, name in enumerate(
                ('supported', 'conflict', 'unknown', 'position_supported', 'plane_valid',
                 'oriented_supported', 'entity_supported', 'query_normal_valid'))}
            result.update(pixels=pixels[rows], observation_indices=integer[rows, 0],
                states=integer[rows, 1].astype(np.int8), state_names=list(STATE_NAMES))
            outputs[index] = result
            measured[index] = [value[rows] for value in values]
            offset = stop
    torch.cuda.synchronize()
    return outputs, measured


def candidate_events(proposals, source_pack, surface, domain, seen, appearance,
                     stats, evidence_stats):
    """Prepare bounded speculative work, then expose the original event order.

    Refit reports and duplicate origins are not committed during lookahead.
    An interrupted future refit becomes an event after its parent's evaluation;
    earlier candidates are still evaluated first. A projection resource error
    leaves prior committed blocks intact and never supplies a CPU substitute.
    """
    projector = evidence = None
    proposal_batch = 16
    for begin in range(0, len(proposals), proposal_batch):
        events, prepared = [], {}
        interrupted = False
        parents = proposals[begin:begin+proposal_batch]
        for proposal in parents:
            events.append(dict(kind='evaluate', transform=proposal['transform'], origins=proposal['origins']))
            started = perf_counter()
            try:
                refined, report = partial_refits(source_pack, surface, domain, proposal)
                refit_seconds = perf_counter()-started
            except (MemoryError, InterruptedError) as error:
                events.append(dict(kind='error', error=error))
                interrupted = True
                break
            events.append(dict(kind='refinement', report=report,
                seconds=refit_seconds, retained=len(refined)))
            events.extend(dict(kind='evaluate', transform=item['transform'], origins=item['origins'])
                          for item in refined)

        mapping_started = perf_counter()
        for event_index, event in enumerate(events):
            if event['kind'] != 'evaluate':
                continue
            # BEGIN delay a future mapping interruption to its event position.
            try:
                key = transform_key(event['transform'])
                if key not in seen and key not in prepared:
                    prepared[key] = _mapped_candidate(source_pack, event['transform'])
            except (MemoryError, InterruptedError) as error:
                events = events[:event_index]+[dict(kind='error', error=error)]
                interrupted = True
                break
            # END completed earlier candidates can still be projected/committed.
            event['prepared'] = prepared.get(key)
        mapping_seconds = perf_counter()-mapping_started
        records, bindings = [], []
        for mapped in prepared.values():
            for role, points, normals in (
                ('query', mapped['query_points'], mapped['query_normals']),
                ('check', mapped['check_points'], mapped['check_normals']),
                ('center', mapped['center'][None], mapped['normal'][None])):
                for view in range(2):
                    records.append(dict(points=points, normals=normals, view=view, role=role))
                    bindings.append((mapped, role, view))
                mapped[role+'_projection'] = [None, None]

        projection_started = perf_counter()
        if records:
            try:
                if projector is None:
                    projector = DenseProjector(surface, device='cuda')
                if evidence is None:
                    evidence = CudaEvidence(appearance, domain, projector, evidence_stats)
                outputs, evidence_outputs = _project_evidence_records(records, projector, evidence)
            except Exception as error:
                # A device resource error remains an interrupted finite search.
                import torch
                if isinstance(error, torch.cuda.OutOfMemoryError):
                    raise MemoryError('CUDA candidate projection allocation failed: '+str(error)) from error
                raise
            for output, (mapped, role, view) in zip(outputs, bindings):
                mapped[role+'_projection'][view] = output
            if evidence is not None:
                for values, (mapped, role, view) in zip(evidence_outputs, bindings):
                    if role == 'query':
                        mapped.setdefault('evidence_views', [None, None])[view] = values
                for mapped in prepared.values():
                    sampled, utilities, elements = evidence.assemble(
                        mapped['query_projection'], mapped.pop('evidence_views'))
                    mapped.update(sampled=sampled, utilities=utilities, elements=elements)
        if stats is not None:
            stats.append(dict(proposal_begin=begin, proposal_batch=proposal_batch,
                backend='cuda', evaluation_events=sum(e['kind']=='evaluate' for e in events),
                unique_transform_count=len(prepared), projection_calls=len(records),
                projection_point_rows=sum(len(r['points']) for r in records),
                mapping_seconds=mapping_seconds, projection_seconds=perf_counter()-projection_started,
                interrupted_preparation=interrupted))
        yield from events
        if interrupted:
            return
