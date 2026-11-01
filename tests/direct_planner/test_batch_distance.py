"""Small numerical contracts for the batched original-geometry interface."""

# BEGIN generic analytic fixtures exercise values, derivatives and unknowns.
import coal
import numpy as np

from manipulation.direct_planner.batch_distance import BatchDistance, NativePointQuery
from manipulation.implicit_distance import ObservedField


def field(values, observed=None):
    return ObservedField(values, np.ones(values.shape, bool) if observed is None else observed,
                         origin=np.array([-.5, -.5, -.5]), voxel_size=.1)


def evaluate(provider, translation):
    first, second = np.eye(4), np.eye(4)
    first[:3, 3] = translation
    return provider.submit(records=[dict(first='robot/body', second='owner/field')],
        columns=np.array([0]), matrices={'robot/body': first[None, None],
                                        'owner/field': second[None, None]},
        permissions=[None], active=np.array([True]), check_request=lambda: None).finish()


def test_batched_field_value_rigid_derivative_and_containment():
    xyz = np.stack(np.meshgrid(*([np.linspace(-.5, .5, 11)]*3), indexing='ij'), axis=-1)
    observed = field(np.linalg.norm(xyz, axis=-1)-.3)
    provider = BatchDistance({'robot/body': coal.Box(.1, .12, .08),
                              'owner/field': observed}, query_sources={'robot/body'}, device='cpu')
    position = np.array([.271, .023, .017])
    value, covector, report = evaluate(provider, position)
    step = 1e-6
    numeric = (evaluate(provider, position+[step, 0, 0])[0]
               - evaluate(provider, position-[step, 0, 0])[0])/(2*step)
    np.testing.assert_allclose(covector[0, 0, 0], numeric[0], atol=1e-5)
    assert report['query_count'] > 0
    assert evaluate(provider, np.zeros(3))[0][0] > 0


def test_unsupported_field_remains_unknown():
    values = np.full((4, 4, 4), -.1)
    provider = BatchDistance({'robot/body': coal.Box(.1, .1, .1),
        'owner/field': field(values, np.zeros(values.shape, bool))}, query_sources={'robot/body'}, device='cpu')
    value, _, report = evaluate(provider, np.array([-.35]*3))
    assert value[0] < 0
    assert report['unknown_query_count'] == report['query_count'] > 0


# BEGIN fixed installations are valid targets without a source inventory.
def test_observed_moving_owner_queries_fixed_robot_target():
    observed = field(np.full((4, 4, 4), -.1))
    provider = BatchDistance({'robot/body': coal.Box(1., 1., 1.),
        'owner/field': observed}, query_sources={'owner/field'}, device='cpu')
    assert 'robot/body' not in provider.samples
    value, _, report = evaluate(provider, np.zeros(3))
    assert value[0] > 0 and report['query_count'] > 0
# END the target lookup follows geometry registration rather than motion role.


def test_exact_registered_analytic_shapes_match_coal():
    points = np.array([[.31, .04, .02], [.03, .01, .01], [.02, -.04, .21]])
    request = coal.DistanceRequest()
    request.enable_signed_distance = True
    for shape in (coal.Box(.2, .3, .4), coal.Cylinder(.12, .3)):
        provider = NativePointQuery(shape, device='cpu')
        values, gradients = provider.sample(points)
        exact = []
        for point in points:
            result = coal.DistanceResult()
            exact.append(coal.distance(shape, coal.Transform3s(), coal.Sphere(0.),
                coal.Transform3s(np.eye(3), point), request, result))
        np.testing.assert_allclose(values, exact, atol=1e-12)
        for axis in range(3):
            offset = np.zeros_like(points); offset[:, axis] = 1e-6
            finite = (provider.sample(points+offset)[0]-provider.sample(points-offset)[0])/2e-6
            np.testing.assert_allclose(gradients[:, axis], finite, atol=1e-6)
# END no scene or task data is replaced by these test-only analytic fields.


# BEGIN asynchronous grouping preserves direction order and the original witness.
def test_pending_groups_keep_first_direction_ties_and_native_rows():
    geometries = {name: field(np.full((4, 4, 4), -.1))
                  for name in ('owner/a', 'owner/b', 'owner/c')}
    geometries.update({'robot/a': coal.Box(.1, .1, .1), 'robot/b': coal.Box(.1, .1, .1)})
    provider = BatchDistance(geometries,
        query_sources={'owner/a', 'owner/b', 'owner/c'}, device='cpu')
    matrices = {name: np.eye(4)[None, None].copy() for name in geometries}
    matrices['owner/a'][0, 0, 0, 3] = .05
    records = [dict(first='owner/a', second=second) for second in ('owner/b', 'owner/c')]
    records.append(dict(first='robot/a', second='robot/b'))
    pending = provider.submit(records=records, columns=np.zeros(3, dtype=int), matrices=matrices,
        permissions=[None]*3, active=np.ones(3, bool), check_request=lambda: None)
    # CPU native rows are independent of the submitted field groups.
    pending.values[2], pending.covectors[2, 0, 0] = .7, .2
    pending.witnesses[2] = [.1, .2, .3]
    values, covectors, report = pending.finish()
    points = provider.inventories['owner/a', ()].points+np.array([.05, 0., 0.])
    scalar, support, _, _ = geometries['owner/b'].sample(points)
    scores = np.where(support, -scalar/.1, -np.inf)
    np.testing.assert_array_equal(values[:2], np.repeat(scores.max(), 2))
    np.testing.assert_array_equal(report['witness_world'][:2],
        np.repeat(points[scores.argmax()][None], 2, axis=0))
    assert values[2] == .7 and covectors[2, 0, 0] == .2
    np.testing.assert_array_equal(report['witness_world'][2], [.1, .2, .3])
    assert report['submitted_groups'] == 4 and report['result_transfers'] == 1
# END support counts, strict winner order and concurrent native output share one point.
