"""Static consistency checks for the four Point2Pose + Track-On-R YAMLs.

Does not initialize models; does not promise that external SAM interfaces,
CUDA extensions or camera SDKs will run on the deployment machine.
"""
import sys
from pathlib import Path
import yaml


def validate(path):
    cfg = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    assert cfg['pipeline']['type'] == 'modular'
    pc = cfg['pipeline']['params']
    assert pc['frame_reg_mode'] in ('f2m', 'f2f')
    assert pc['frame_reg_mode'] == 'f2m'
    assert pc['use_segmenter'] is False, 'Mask fed via Frame.mask'
    assert pc['use_depth_estimate'] is False
    assert pc['max_num_obj'] >= 1
    assert pc['min_depth'] < pc['max_depth']
    assert cfg['segmenter']['type'] == 'dummy'
    assert cfg['tracker']['type'] == 'trackon'
    tr = cfg['tracker']['params']
    assert tr['vit_backbone'] == 'dinov3_s_plus'
    assert list(tr['input_size']) == [384, 512]
    assert tr['checkpoint_path'].endswith('track_on_r.pt')
    assert tr['inference_memory_size'] >= tr['memory_size']
    assert cfg['sampler']['type'] == 'super_point_balanced'
    assert cfg['sampler']['params']['num_points'] > 0
    assert cfg['sampler']['params']['max_points'] >= cfg['sampler']['params']['num_points']
    assert cfg['criterion']['type'] == 'rotation_threshold_and_min_num'
    assert cfg['reconstructor']['type'] == 'sdf_builder'
    rc = cfg['reconstructor']['params']
    assert 0.001 <= rc['sdf_voxel_size'] <= 0.03
    assert 0 <= rc['sdf_bounds_padding'] <= .2
    assert 'sdf_max_voxels' not in rc, 'key is ignored in upstream SDFBuilder'
    assert 'use_graph_optimization' not in pc, 'ignored key; use use_key_frame_graph'
    name = Path(path).name.lower()
    offline = 'offline' in name
    assert pc['use_key_frame_graph'] == offline
    assert pc['use_local_graph'] == offline
    assert cfg['register']['type'] == ('svd_cluster_ransac' if offline else 'svd_residual_outlier')
    if 'mujoco' in name:
        assert 'mujoco' in cfg
    if 'realsense' in name:
        assert 'realsense' in cfg
    return f'OK {name}: {cfg["register"]["type"]} / graph={pc["use_key_frame_graph"]}'

if __name__ == '__main__':
    args = sys.argv[1:]
    if not args:
        args = sorted(str(p) for p in Path(__file__).parent.glob('*.yaml'))
    for fn in args:
        print(validate(fn))
