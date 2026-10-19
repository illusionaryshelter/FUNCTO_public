"""CPU checks of the common task/environment precision contract."""
# BEGIN task identities determine precision independently of experiment names.
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from manipulation.planning_observation import observation_resolutions, scene_task_owner_ids


def test_functional_reference_and_carrier_are_fine_while_initial_support_is_environment():
    scene = {'carrier_id': 'part_a', 'reference_id': 'part_b',
             'observation_owners': [{'owner_id': 'initial_surface'}]}
    task = scene_task_owner_ids(scene)
    assert observation_resolutions(('part_a', 'part_b', 'initial_surface'),
        task_owner_ids=task, environment_voxel_size=.01) == {
            'part_a': .002, 'part_b': .002, 'initial_surface': .01}


def test_owner_names_and_inventory_order_do_not_select_task_roles():
    scene = {'carrier_id': 'environment', 'reference_id': 'table'}
    assert observation_resolutions(('table', 'mug', 'environment'),
        task_owner_ids=scene_task_owner_ids(scene), environment_voxel_size=.015) == {
            'table': .002, 'mug': .015, 'environment': .002}
    assert scene_task_owner_ids({'carrier_id': 'grasp_target'}) == ('grasp_target',)


@pytest.mark.parametrize('task', [('missing',), ('part', 'part'), 'part'])
def test_missing_or_ambiguous_task_owner_is_rejected(task):
    with pytest.raises(ValueError):
        observation_resolutions(('part',), task_owner_ids=task, environment_voxel_size=.01)


@pytest.mark.parametrize('spacing', [True, float('nan'), float('inf'), 0, -.01, .001, .002])
def test_environment_cannot_use_an_invalid_or_task_scale_grid(spacing):
    with pytest.raises(ValueError):
        observation_resolutions(('part',), task_owner_ids=('part',), environment_voxel_size=spacing)


@pytest.mark.parametrize('scene', [{}, {'carrier_id': ''}, {'carrier_id': 'part', 'reference_id': ''},
                                 {'carrier_id': 'part', 'reference_id': 'part'}])
def test_role_evidence_cannot_be_missing_or_self_referential(scene):
    with pytest.raises(ValueError):
        scene_task_owner_ids(scene)
# END no native map, simulation, planner or CUDA allocation is required.


# BEGIN both recorded-input command entry points must resolve the shared defaults.
@pytest.mark.parametrize('module_name', ['run_recorded_skill', 'probe_recorded_grasp_feasibility'])
def test_recorded_entry_help_resolves_current_mapping_api(module_name, monkeypatch, capsys):
    """Catch missing live imports before starting an expensive native request."""
    import importlib
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parent/'planning'))
    module = importlib.import_module(module_name)
    monkeypatch.setattr(sys, 'argv', [module_name, '--help'])
    with pytest.raises(SystemExit) as stopped:
        module.parse_args()
    assert stopped.value.code == 0
    assert '--voxel-size' in capsys.readouterr().out
# END help exercises the real parser without reading inputs or allocating CUDA.


# BEGIN the native scene boundary enforces the same rule before allocation.
@pytest.mark.parametrize('identity, spacing', [('part', .0005), (None, .0015)])
def test_roi_cannot_escape_task_or_environment_precision(identity, spacing):
    from unittest.mock import patch
    from test_manipulation_planning_observation import observed_owner_fixture
    from manipulation.planning_observation import build_scene_from_rgbd
    views, frames, cfg = observed_owner_fixture()
    region = {'center_m': [0., 0., 0.], 'extent_m': [.01, .01, .01],
        'voxel_size_m': spacing, 'measurement_allowance_m': .003,
        'measurement_evidence': 'Explicit synthetic bound', 'source': 'Bounded synthetic ROI'}
    with patch('manipulation.planning_observation._map_online_partition') as mapper, \
         patch('manipulation.planning_observation._known_free_voxels') as coverage:
        with pytest.raises(ValueError, match='grid spacing|coarser'):
            build_scene_from_rgbd(views, frames, mapper_cfg=cfg, depth_error_m=.003,
                task_owner_ids=('part',), reference_frame='world', robot_sphere_materials=(None,),
                robot_geometry_source='Declared test robot', has_gripper=False,
                fine_regions={identity: region}, max_mapping_bytes=1024**3)
        mapper.assert_not_called()
        coverage.assert_not_called()
# END no resource failure can turn into a lower-precision task field.
