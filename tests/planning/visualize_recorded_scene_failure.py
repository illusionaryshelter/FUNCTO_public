#!/usr/bin/env python3
"""Render frozen scene failures on CPU; never plan, simulate, or change evidence.

The paired exporter owns native collision queries. This renderer independently
checks world-frame FK, then shows those exact saved queries, their source RGB-D,
and native convex geometry. Independent IK states are never interpolated.
"""

# BEGIN offline review of independently verified, frozen scene queries.
import argparse
import json
import os
from pathlib import Path
import resource
import time

from visualize_recorded_task import configure_font, input_path, read_json
from visualize_current_recorded_task import escape, link
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
from PIL import Image
from plotly.offline import get_plotlyjs
from scipy.spatial import ConvexHull
import trimesh
from yourdfpy import URDF
import yaml


def transform(points, pose):
    """Apply world-from-local SE(3) to row-vector points without rounding."""
    return np.asarray(points) @ pose[:3, :3].T + pose[:3, 3]


def geometry(fixture, feedback, reports, arrays):
    """Use recorded finger coordinates, and verify every displayed arm state.

    Scene hand envelopes are already exported separately. The solid meshes here
    represent actual native collision solids at measured finger coordinates.
    They must not be confused with the full-range planning hand envelope.
    """
    robot = URDF.load(fixture['urdf_path'], build_scene_graph=True, load_meshes=False,
                     build_collision_scene_graph=False, load_collision_meshes=False)
    cfg = yaml.safe_load(Path(fixture['robot_config_path']).read_text())['robot_cfg']['kinematics']
    base = np.asarray(feedback['base_pose'])
    fixed = dict(zip(feedback['joint_names'], feedback['start_q']))
    meshes = []
    for row in fixture['collision_geometries']:
        with np.load(row['certificate_path'], allow_pickle=False) as data:
            vertices = transform(data['native_vertices'], data['T_body_geom'])
        # Native mesh collision uses its convex hull. Keep only hull vertices;
        # this is a lossless surface representation, not mesh simplification.
        hull = ConvexHull(vertices)
        used, indices = np.unique(hull.simplices, return_inverse=True)
        meshes.append({'link': row['link'], 'vertices': vertices[used].tolist(),
                       'faces': indices.reshape(-1, 3).tolist()})
    maximum = 0.
    for report, data in zip(reports, arrays):
        links = data['sphere_links'].tolist()
        local = []
        for name in dict.fromkeys(links):
            balls = cfg['collision_spheres'][name]
            if links.count(name) != len(balls):
                raise ValueError('Native sphere ordering changed')
            local.extend([[*ball['center'], ball['radius']] for ball in balls])
        local = np.asarray(local)
        for index, state in enumerate(report['states']):
            robot.update_cfg({**fixed, **dict(zip(fixture['planning_joint_names'], state['q']))})
            poses = {name: base @ robot.get_transform(name, frame_from=fixture['base_link'])
                     for name in set(links)}
            centers = np.array([transform(local[i:i+1, :3], poses[name])[0]
                                for i, name in enumerate(links)])
            error = np.max(np.abs(np.c_[centers, local[:, 3]]-data['actual_spheres'][index]))
            maximum = max(maximum, float(error))
            state['mesh_poses'] = [poses[mesh['link']].tolist() for mesh in meshes]
        if maximum > 2e-6:
            raise ValueError(f'Independent CPU sphere FK disagrees with saved native FK: {maximum}')
    return meshes, maximum


def observations(manifest, segment, entry, entry_dir, output, lower, upper):
    """Back-project original calibrated depth, with deterministic display limits.

    Cropping and thinning affect visualization only. They never change scene
    classification or imply that unseen space is free. Masks remain distinct.
    """
    root = Path(manifest['workspace_root'])
    clouds, images, evidence = [], [], []
    colors = {'background': '#8893a0', 'robot': '#e69d35', 'owner': '#159c92'}
    for view in segment['snapshot']['views']:
        calibration = read_json(input_path(view['calibration_path'], root))
        depth = np.load(input_path(view['depth_path'], root), allow_pickle=False)
        rgb = np.asarray(Image.open(input_path(view['rgb_path'], root)).convert('RGB'))
        robot = np.load(entry_dir/'observations'/view['id']/'robot_mask.npy', allow_pickle=False).astype(bool)
        owner = np.load(entry['carrier_observation']['mask_paths'][view['id']], allow_pickle=False).astype(bool)
        if robot.shape != depth.shape or owner.shape != depth.shape or np.any(robot & owner):
            raise ValueError('Saved masks must align and remain disjoint')
        v, u = np.indices(depth.shape)
        K, pose = np.asarray(calibration['K']), np.asarray(calibration['T_world_camera_cv'])
        camera = np.stack(((u-K[0, 2])/K[0, 0]*depth,
                           (v-K[1, 2])/K[1, 1]*depth, depth), axis=-1)
        points = transform(camera.reshape(-1, 3), pose)
        valid = (np.isfinite(points).all(axis=1) & (depth.ravel() > 0)
                 & (points >= lower).all(axis=1) & (points <= upper).all(axis=1))
        for category, mask, budget in [('background', ~(robot | owner), 10000),
                                        ('robot', robot, 2500), ('owner', owner, 2500)]:
            selected = np.flatnonzero(valid & mask.ravel())
            count = len(selected)
            selected = selected[::max(1, int(np.ceil(count/budget)))]
            clouds.append({'name': view['id']+' / '+category, 'category': category,
                           'color': colors[category], 'points': points[selected].tolist()})
            evidence.append({'view': view['id'], 'category': category,
                             'in_crop_before_thinning': count, 'display_count': len(selected)})
        overlay = rgb.astype(float)
        overlay[robot] = .5*overlay[robot]+.5*np.array([230, 157, 53])
        overlay[owner] = .4*overlay[owner]+.6*np.array([21, 156, 146])
        rgb_path, overlay_path = f'{view["id"]}-rgb.png', f'{view["id"]}-masks.png'
        Image.fromarray(rgb).save(output/rgb_path)
        Image.fromarray(overlay.astype(np.uint8)).save(output/overlay_path)
        images.append({'name': view['id'], 'rgb': rgb_path, 'masks': overlay_path})
    return clouds, images, evidence


def mesh_projection(ax, meshes, state, axes, center, width):
    """Orthographic projection retains metre axes; no perspective size claim."""
    for mesh, pose in zip(meshes, state['mesh_poses']):
        vertices = transform(mesh['vertices'], np.asarray(pose))[:, axes]
        hull = ConvexHull(vertices)
        ax.fill(*vertices[hull.vertices].T, color='#60788f', alpha=.28, lw=.6, ec='#30465d')
    ax.set_xlim(center[axes[0]]-width, center[axes[0]]+width)
    ax.set_ylim(center[axes[1]]-width, center[axes[1]]+width)
    ax.set_aspect('equal')
    ax.set_xlabel('XYZ'[axes[0]]+' / m'); ax.set_ylabel('XYZ'[axes[1]]+' / m')
    ax.grid(alpha=.2)


def overview(output, payload, arrays):
    """Give a shareable overview while the HTML retains all individual states."""
    configure_font()
    fig = plt.figure(figsize=(18, 11), constrained_layout=True)
    grid = fig.add_gridspec(2, 3)
    report, data = payload['reports'][0], arrays[0]
    q0, balls = report['states'][0], data['scene_spheres'][0]
    for index, view in enumerate(payload['images'][:2]):
        ax = fig.add_subplot(grid[0, index])
        ax.imshow(Image.open(output/view['masks'])); ax.axis('off')
        ax.set_title(view['name']+' 原始 RGB + 掩码\n橙：机器人观测；青：owner 观测')
    ax = fig.add_subplot(grid[0, 2])
    rows = []
    for item in payload['reports']:
        for frame in sorted({s['source_frame'] for s in item['states'] if s['kind']=='ik'}):
            states = [s for s in item['states'] if s['mode']=='native_self_collision' and s['source_frame']==frame]
            rows.append((item['candidate_id'].split('/')[0]+f' / {frame}',
                         sum(s['native_success'] for s in states),
                         sum(s['postcheck']['all_original_collision_terms_zero'] for s in states)))
    ticks = np.arange(len(rows))
    ax.barh(ticks-.18, [r[1] for r in rows], .34, label='原生自碰撞 IK 成功', color='#39846a')
    ax.barh(ticks+.18, [r[2] for r in rows], .34, label='完整场景通过', color='#b84545')
    ax.set_yticks(ticks, [r[0] for r in rows]); ax.invert_yaxis(); ax.set_xlim(0, 8)
    ax.set_xlabel('每个目标的 8 个已保存候选解'); ax.legend(loc='lower right', fontsize=9)
    ax.set_title('IK 可以成功；完整场景仍全部失败')
    ax = fig.add_subplot(grid[1, 0])
    positive = np.flatnonzero(data['background'][0] > 0)
    center = balls[positive, :3].mean(axis=0)
    mesh_projection(ax, payload['meshes'], q0, [0, 2], center, .13)
    for cloud in payload['clouds']:
        points = np.asarray(cloud['points'])
        if cloud['category']=='background' and len(points):
            mask = np.abs(points[:, 1]-center[1]) < .08
            ax.scatter(points[mask, 0], points[mask, 2], s=1, c=cloud['color'], alpha=.45)
    for i in positive:
        ax.add_patch(Circle(balls[i, [0, 2]], balls[i, 3], color='#d74a42', alpha=.3))
        ax.text(*balls[i, [0, 2]], str(i), fontsize=7)
    ax.set_title(f'q0 背景查询正项：{len(positive)}/{len(balls)} 球\n红球 = 场景包络；灰点 = 观测背景（薄 Y 切片）')
    ax = fig.add_subplot(grid[1, 1])
    center = np.asarray(q0['tool_pose'])[:3, 3]
    mesh_projection(ax, payload['meshes'], q0, [1, 2], center, .085)
    pair_ids = set()
    for pair in q0['gripper_pairs']['top'][:4]:
        pair_ids.update(pair['indices'])
        points = balls[pair['indices'], :3]
        ax.plot(points[:, 1], points[:, 2], color='#b22a57', lw=1)
    for i in pair_ids:
        ax.add_patch(Circle(balls[i, [1, 2]], balls[i, 3], color='#d44b86', alpha=.28))
        actual = data['actual_spheres'][0, i]
        ax.add_patch(Circle(actual[[1, 2]], actual[3], fill=False, ec='#248356', lw=1))
        ax.text(*balls[i, [1, 2]], str(i), fontsize=7)
    ax.set_title('q0 夹爪重叠最大的 4 对球\n粉：全行程包络；绿圈：实测指位；蓝：原生实体')
    ax = fig.add_subplot(grid[1, 2])
    clearance, origin = data['coverage_clearance'], data['coverage_origin']
    voxel = float(data['coverage_voxel_size'])
    worst = int(data['unknown'][0].argmax())
    z = int(np.clip(round((balls[worst, 2]-origin[2])/voxel), 0, len(clearance)-1))
    im = ax.imshow(clearance[z]*1000, origin='lower', extent=[origin[0]-.5*voxel,
        origin[0]+(clearance.shape[2]-.5)*voxel, origin[1]-.5*voxel,
        origin[1]+(clearance.shape[1]-.5)*voxel], cmap='viridis', aspect='equal')
    selected = np.abs(balls[:, 2]-(origin[2]+z*voxel)) <= balls[:, 3]+voxel/2
    ax.scatter(balls[selected, 0], balls[selected, 1], s=9, facecolors='none', edgecolors='#ed6262')
    ax.scatter(*balls[worst, :2], s=55, marker='x', c='red')
    ax.set_xlabel('X / m'); ax.set_ylabel('Y / m')
    ax.set_title(f'已知空间保守净空：Z = {origin[2]+z*voxel:.3f} m\n红 ×：最大未知空间项所在球；这不是占据地图')
    fig.colorbar(im, ax=ax, shrink=.6, label='保存的 coverage clearance / mm')
    fig.suptitle('失败复查：修复后的入口 q0 + 静态 IK 结果（无本轮执行轨迹）', fontsize=19)
    fig.savefig(output/'summary.png', dpi=135); plt.close(fig)


def render(args):
    started = time.monotonic()
    reports = [read_json(path/'report.json') for path in args.scene_export]
    shared = ('manifest_path', 'fixture_path', 'feedback_path', 'segment_id')
    if any(any(report[key] != reports[0][key] for key in shared) for report in reports):
        raise ValueError('Explicit exports must share the exact scene and robot provenance')
    if any(r['status'] != 'exported' or r['maximum_original_scene_term_difference'] > 2e-5 for r in reports):
        raise ValueError('Only validated native query exports are displayable')
    arrays = []
    for path in args.scene_export:
        with np.load(path/'states.npz', allow_pickle=False) as data:
            arrays.append({key: data[key].copy() for key in data.files})
    fixture = read_json(Path(reports[0]['fixture_path']))
    feedback = read_json(Path(reports[0]['feedback_path']))
    manifest = read_json(Path(reports[0]['manifest_path']))
    entry = read_json(args.entry_run)
    if entry['robot_fixture_path'] != reports[0]['fixture_path'] or entry['feedback_path'] != reports[0]['feedback_path']:
        raise ValueError('Entry observation provenance differs from exported scene')
    segment, = [s for s in manifest['segments'] if s['id']==reports[0]['segment_id']]
    meshes, maximum_fk_error = geometry(fixture, feedback, reports, arrays)
    args.output.mkdir(parents=True, exist_ok=False)
    data = arrays[0]
    origin, voxel = data['coverage_origin'], float(data['coverage_voxel_size'])
    upper = origin + voxel*np.asarray(data['coverage_clearance'].shape[::-1])
    clouds, images, cloud_counts = observations(manifest, segment, entry, args.entry_run.parent,
                                               args.output, origin-.15, upper+.15)
    unit = trimesh.creation.icosphere(subdivisions=1, radius=1.)
    payload = {'reports': reports, 'meshes': meshes, 'clouds': clouds, 'images': images,
               'base': np.asarray(feedback['base_pose'])[:3, 3].tolist(),
               'unitSphere': {'vertices': unit.vertices.tolist(), 'faces': unit.faces.tolist()},
               'data': [{key: data[key].tolist() for key in ('actual_spheres', 'scene_spheres',
                         'sphere_links', 'background', 'unknown', 'gripper_overlap', 'native_overlap')}
                        for data in arrays]}
    overview(args.output, payload, arrays)
    (args.output/'plotly.min.js').write_text(get_plotlyjs())
    (args.output/'viewer.js').write_text(Path(__file__).with_name('recorded_scene_failure_viewer.js').read_text())
    sources = ''.join('<li>'+link(path/'report.json', args.output, report['candidate_id']+' 原始查询导出')+
                      ' · '+link(Path(report['probe_report']), args.output, '静态 IK 原报告')+'</li>'
                      for path, report in zip(args.scene_export, reports))
    if args.historical_page:
        sources += '<li>'+link(args.historical_page, args.output, '历史模型的失败执行复查（非本轮结果）')+'</li>'
    encoded = json.dumps(payload, allow_nan=False, separators=(',', ':')).replace('<', '\\u003c')
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>失败场景复查</title>
<style>body{font:15px system-ui,sans-serif;margin:24px;color:#243448;background:#f5f7fa}
main{max-width:1500px;margin:auto}h1{font-size:28px}h2{font-size:20px}.card{background:white;border:1px solid #dbe2ea;border-radius:10px;padding:18px;margin:16px 0}
.note{border-left:5px solid #bf5549}label{display:inline-block;margin:5px 12px 5px 0}select,button{padding:7px;max-width:95%;margin:3px;border:1px solid #bac6d5;border-radius:5px;background:white}
#scene{height:690px}.split{display:grid;grid-template-columns:1fr 1fr;gap:18px}.photos{display:flex;gap:12px;flex-wrap:wrap}.photos figure{margin:0;flex:1;min-width:260px}.photos img{width:100%}table{width:100%;border-collapse:collapse;font-size:13px}td,th{text-align:left;padding:7px;border-bottom:1px solid #e5e9ef}code{word-break:break-all}.bad{color:#a53232}.good{color:#267354}.muted{color:#626f7d}.scroll{overflow:auto}#error{color:#a53232;white-space:pre-wrap}a{color:#2167a3}
@media(max-width:850px){.split{display:block}body{margin:10px}#scene{height:520px}}</style>
<main><h1>失败场景复查</h1><div class="card note">本轮在入口的完整场景检查失败，没有生成执行轨迹。下方为保存的 q0 与静态 IK 候选解，各状态互相独立。原生自碰撞 IK 成功不代表完整场景通过，更不代表抓取成功。</div>
<div class="card"><label>候选 <select id="candidate"></select></label><label>状态 <select id="state"></select></label><br>
<label>球显示 <select id="metric"><option value="background">背景正项 · 场景包络</option><option value="unknown">未知空间正项 · 场景包络</option><option value="gripper_overlap">夹爪自碰撞正项 · 全行程包络</option><option value="native_overlap">原生自碰撞正项 · 实测指位</option><option value="actual">全部球 · 实测指位</option><option value="none">隐藏球</option></select></label>
<label>连杆 <select id="link"><option value="all">全部</option></select></label>
<label><input id="solids" type="checkbox" checked>原生碰撞实体</label><label><input id="background" type="checkbox" checked>背景点</label><label><input id="owner" type="checkbox" checked>owner 点</label><label><input id="robot" type="checkbox">q0 机器人观测点</label><br>
<button data-view="all">全景</button><button data-view="base">基座近景</button><button data-view="hand">夹爪近景</button><span class="muted">拖动旋转 · 滚轮缩放 · 悬停查看球编号 / 连杆 / 数值</span>
<p id="status"></p><p id="error"></p><div id="scene"></div><p id="explanation"></p>
<p class="muted">蓝灰实体 = 原生凸碰撞体；点云来自入口 RGB-D，换选 IK 状态时仍保留同一观测。场景夹爪球包含整个机械行程，与实测指位的球不同。球面仅作低面数显示，查询使用的中心、半径与数值完整保留。</p></div>
<div class="split"><div class="card"><h2>原报告的完整场景残差</h2><div id="residuals"></div><p>背景项与夹爪项为多个正项之和，不能当作单处穿透深度。未知空间项取最大值。零值也不证明未观测物体的几何完整。</p></div>
<div class="card"><h2>所选显示项的球 / 球对</h2><div id="sphereSummary"></div><div class="scroll" id="spheres"></div></div></div>
<div class="card"><h2>静态 IK 汇总</h2><div id="ikSummary"></div></div>
<div class="card"><h2>入口观测</h2><p>可点开原始 RGB；橙色为机器人掩码，青色为 owner 掩码。三维点仅为范围内的确定性抽样，不是占据或自由空间证明。</p><div id="photos" class="photos"></div></div>
<div class="card"><h2>静态总览</h2><a href="summary.png"><img src="summary.png" style="width:100%" alt="入口碰撞、夹爪包络、已知空间净空及 IK 汇总"></a></div>
<div class="card"><h2>证据</h2><ul>__SOURCES__</ul><p><a href="verification-report.json">数据 / FK 校验记录</a> · <a href="README.md">范围与复查说明</a></p></div></main>
<script src="plotly.min.js"></script><script id="payload" type="application/json">__PAYLOAD__</script><script src="viewer.js"></script></html>'''
    (args.output/'index.html').write_text(page.replace('__SOURCES__', sources).replace('__PAYLOAD__', encoded))
    # Round-trip validation catches broken serialization, missing rows, and
    # accidental display rounding independently of the browser implementation.
    decoded = json.loads(encoded)
    for report, original, display in zip(reports, arrays, decoded['data']):
        for key in display:
            if not np.array_equal(np.asarray(display[key]), original[key]):
                raise ValueError('Display payload modified source array: '+key)
        if len(report['states']) != len(display['actual_spheres']):
            raise ValueError('Every state must have the corresponding saved scene query')
    verification = {'status': 'passed', 'renderer': str(Path(__file__).resolve()),
        'scene_exports': [str(p.resolve()) for p in args.scene_export],
        'candidate_count': len(reports), 'state_count': sum(len(r['states']) for r in reports),
        'maximum_independent_cpu_sphere_fk_error_m': maximum_fk_error,
        'maximum_original_scene_term_difference': max(r['maximum_original_scene_term_difference'] for r in reports),
        'serialized_arrays_exact': True, 'cloud_counts': cloud_counts,
        'point_display_crop_lower': (origin-.15).tolist(), 'point_display_crop_upper': (upper+.15).tolist(),
        'browser_webgl_interaction_tested': False, 'ik_called': False, 'simulation_called': False,
        'elapsed_seconds': time.monotonic()-started,
        'peak_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024}
    (args.output/'verification-report.json').write_text(json.dumps(verification, indent=2)+'\n')
    (args.output/'README.md').write_text('''# 失败结果可视化复查

打开同目录 `index.html`；Plotly 和数据均在本地，不需要启动服务或联网。

1. 默认查看入口 q0 的背景正项，点击「基座近景」。
2. 切换未知空间项、夹爪自碰撞项；夹爪项配合「夹爪近景」查看。
3. 选择 A/B 候选和各目标的 8 个 IK 结果，对照原生 IK 状态与完整场景残差。
4. 「实测指位」显示的球和原生实体使用冻结反馈的手指位置；场景查询使用全机械行程包络。两者不是同一个几何集合。

当前结果没有轨迹或物理执行。历史执行页面是单独证据链接，不能用作本轮执行结果。
原始逐球查询与原报告完全一致，CPU FK 和页面数组另行校验。浏览器 WebGL 实际交互未自动测试；静态 PNG 可直接复查。
背景的正查询值保留原算法含义；不能据此把显示抽样点认作精确最近碰撞点。夹爪残差为球对重叠之和，最大单对值另列。
已知空间切片显示保存的保守净空，不是占据地图。红球中心的抽样切片用于定位，不代替逐球体积查询。
''')
    print(json.dumps(verification, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene-export', type=Path, action='append', required=True)
    parser.add_argument('--entry-run', type=Path, required=True)
    parser.add_argument('--historical-page', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    render(parser.parse_args())
# END offline review of independently verified, frozen scene queries.
