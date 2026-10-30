"""Build a local scene-based result page from one completed experiment.

Saved reports own all verdicts. SciPy evaluates the recorded Hermite curves,
and Pinocchio evaluates the fixed official robot's kinematics for display.
Camera projection preserves the original grounding images, regions and sample
points. No planner, geometric query or physical step is performed here.
"""

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pinocchio as pin
from scipy.interpolate import CubicHermiteSpline
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from manipulation.execution_binding import plain
from manipulation.planning_problem import PosePath, PoseEventRequirement
from manipulation.progress_clock import progress_anchor_parameters
from manipulation.timed_reference import BoundToolReference


def read(path):
    """Read one original completed JSON record."""
    # BEGIN display consumes full evidence behind the report's file references.
    from manipulation.evidence_archive import read_evidence_json
    return read_evidence_json(path)
    # END acceptance, refusal and unknown records keep their original values.


def archive(path):
    """Load the named numeric arrays without pickle or schema conversion."""
    with np.load(path, allow_pickle=False) as record:
        return {key: record[key].copy() for key in record.files}


def project(points, calibration):
    """Project W points into the original calibrated CV image in pixels."""
    inverse = np.linalg.inv(calibration['T_world_camera_cv'])
    camera = np.asarray(points) @ inverse[:3, :3].T + inverse[:3, 3]
    homogeneous = camera @ np.asarray(calibration['K']).T
    return homogeneous[..., :2] / homogeneous[..., 2:]


def pose_track(poses, calibration, label, color):
    """Store full poses, RPY and projected 35-mm axes at every saved time."""
    offsets = np.vstack((np.zeros(3), np.eye(3) * .035))
    points = np.einsum('nij,kj->nki', poses[:, :3, :3], offsets) + poses[:, None, :3, 3]
    return dict(label=label, color=color, poses=poses,
        rpy_deg=Rotation.from_matrix(poses[:, :3, :3]).as_euler('xyz', degrees=True),
        pixels=project(poses[:, :3, 3], calibration), axes=project(points, calibration))


def robot_poses(fixture, feedback, curve, times):
    """Evaluate one saved curve using the official fixed robot's kinematics.

    The robot and its end-effector configuration have reusable, fixed model
    structure across tasks. The URDF is used only for this official robot;
    task entities are represented by their original scene images here.
    Named measured joints initialize the complete model. A saved finger curve
    supplies its own joints when present; otherwise those measured joints
    remain fixed and that source is recorded in the returned description.
    """
    arm = CubicHermiteSpline(curve['times'], curve['position'], curve['velocity'], axis=0)
    model = pin.buildModelFromUrdf(fixture['urdf_path'])
    data, q = model.createData(), pin.neutral(model)
    for name, value in zip(feedback['joint_names'], feedback['start_q']):
        q[model.joints[model.getJointId(name)].idx_q] = value
    names = list(curve['joint_names'])
    positions, velocities = arm(times), arm(times, 1)
    finger_source = 'measured entrance joints; curve contains no finger trajectory'
    if 'finger_times' in curve:
        fingers = {key: curve['finger_' + key] for key in
                   ('times', 'position', 'velocity', 'joint_names')}
    else:
        fingers = curve.get('fingers')
    if fingers is not None:
        finger = CubicHermiteSpline(fingers['times'], fingers['position'], fingers['velocity'], axis=0)
        positions = np.column_stack((positions, finger(times)))
        velocities = np.column_stack((velocities, finger(times, 1)))
        names += list(fingers['joint_names'])
        finger_source = 'this curve saved finger polynomial'
    columns = [model.joints[model.getJointId(name)].idx_q for name in names]
    frame = model.getFrameId(feedback['tool_link'])
    base, tool = np.asarray(feedback['base_pose']), np.asarray(feedback['tool_transform'])
    poses = []
    for values in positions:
        q[columns] = values
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        poses.append(base @ data.oMf[frame].homogeneous @ tool)
    return np.asarray(poses), dict(q=positions, dq=velocities,
        joint_names=np.asarray(names), finger_source=finger_source)


def nearest_indices(times, queries):
    """Map each query to its closest saved time without a dense distance array."""
    times, queries = np.asarray(times), np.asarray(queries)
    right = np.searchsorted(times, queries).clip(0, len(times) - 1)
    left = np.maximum(0, right - 1)
    return np.where(abs(times[left] - queries) <= abs(times[right] - queries), left, right)


def saved_markers(selected, audit, curve_key, curve, pose_objectives):
    """Copy spatial evidence owned by the selected, completed curve report.

    Geometry witnesses retain their registered entity names and original
    status. Task origins are populated later from that same curve's FK.
    Reports without a completed candidate verdict supply no spatial verdict.
    """
    markers = []
    for row in selected.get('observed_geometry', {}).get('nonclear_pairs', []):
        markers.append(dict(kind='registered_geometry', time_s=row['time_s'],
            world_position_m=row['witness_first'], evidence=deepcopy(row),
            text=f"登记实体 {row['first']} / {row['second']}；原状态 {row['status']}；"
                 f"曲线时间 {row['time_s']:.9f} s。红叉为该查询保存的见证点。"))
    for index, goal in enumerate(audit['pose_events']):
        name = f'function_pose:{index}'
        if name not in selected.get('residuals', {}):
            continue
        # The same shared clock API used by candidate queries preserves both
        # ends of a progress plateau. This locates original task declarations;
        # the saved aggregate residual is not relabelled as a per-time value.
        anchors = progress_anchor_parameters(np.asarray(curve['progress_u']),
            np.asarray(curve['progress_s']), [goal['progress']])
        value = selected['residuals'][name]
        # BEGIN display the actual declared role instead of interpreting zero as arrival.
        declaration = pose_objectives[index]
        role_text = ('示教位姿先验' if declaration['role'] == 'prior' else '必达位姿条件')
        # END the original target remains visible for both semantic roles.
        for parameter, _ in anchors:
            time_s = float(parameter * np.asarray(curve['times'])[-1])
            markers.append(dict(kind='task_condition', time_s=time_s, goal_index=index,
                target_world_position_m=np.asarray(goal['target'])[:3, 3],
                evidence=dict(condition=name, original_residual=value,
                    pose_objective=declaration,
                    target_pose=goal['target'], tool_from_function=goal['tool_from_function'],
                    position_tolerance_m=goal['position_tolerance_m'],
                    rotation_tolerance_rad=goal['rotation_tolerance_rad'],
                    time_mapping_source='progress_anchor_parameters on this saved curve',
                    residual_semantics='sum of positive component excesses; mixed m/rad scalar'),
                text=f"{role_text} {name}；角色 {declaration['role']}；声明查询时间 {time_s:.9f} s。"
                     '红叉为该曲线的功能坐标原点，黄圈为声明目标。'))
    for name, row in selected.get('scene_components', {}).items():
        if name.startswith('conditional/') and row['maximum'] > 0:
            markers.append(dict(kind='conditional_warning', time_s=row['time_s'],
                evidence=dict(component=name, **row),
                text=f"条件提示 {name}={row['maximum']}，曲线时间 {row['time_s']:.9f} s。"
                     '红叉定位此时工具原点；条件提示与实体相交判定分别保留。'))
    for marker in markers:
        marker['curve_key'] = curve_key
        marker['text'] = ('初始化曲线：' if curve_key == 'initial' else '当前保存曲线：') + marker['text']
    return markers


def main():
    """Render actual report states, including early errors and missing curves."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'request', 'resources', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    output, run = args.output.resolve(), args.run.resolve()
    output.mkdir(parents=True, exist_ok=True)
    relative = lambda path: os.path.relpath(Path(path).resolve(), output)
    planning = run / 'planning-run/planning'
    request, resources = read(args.request), read(args.resources)
    inputs, grounding = Path(request['inputs']), Path(request['grounding'])
    feedback, fixture, audit = [read(inputs / name) for name in
        ('feedback.json', 'robot-fixture.json', 'input-audit.json')]
    run_report = read(run / 'planning-run/run-report.json')
    report_path = planning / 'planning-report.json'
    report = read(report_path) if report_path.exists() else {}
    diagnostic = report.get('diagnostics', {})
    success = run_report.get('planning_success')
    curve_path = planning / ('trajectory.npz' if success else 'diagnostic-trajectory.npz')
    curve = archive(curve_path) if curve_path.exists() else None
    metadata = read(curve_path.with_suffix('.json')) if curve is not None else None
    initial = diagnostic.get('reference_candidate', {}).get('curve')
    clock_curve = curve if curve is not None else initial
    # BEGIN map this motion block back to the request's original pose identities.
    # The audit retains the complete skill representation; display queries
    # use only the bound block's event list and its explicit role declarations.
    pose_objectives = []
    if clock_curve is not None:
        identities = diagnostic['motion_block']['goal_identities']['pose_events']
        declarations = {tuple(row['goal_id']): row for row in request['pose_objectives']}
        audit['pose_events'] = [audit['pose_events'][index] for index in identities]
        pose_objectives = [declarations[('pose_events', index)] for index in identities]
    # END local residual indexes and original task identities have separate labels.
    selected = (diagnostic if success else diagnostic.get('diagnostic_candidate_report', {}))
    if metadata is None or metadata['validation_status'] != 'completed':
        selected = {}
    # BEGIN completed initial evidence belongs only to its own polynomial.
    markers = saved_markers(selected, audit, 'final', curve, pose_objectives)
    initial_report = diagnostic.get('reference_candidate', {})
    if initial is not None and initial_report['validation_status'] == 'completed':
        markers += saved_markers(initial_report, audit, 'initial', initial, pose_objectives)
    # END numerical optimizer trials never inherit initial qualification.
    actual_path = run / 'simulation/execution/physical-trajectory.npz'
    actual = archive(actual_path) if actual_path.exists() else None
    simulation_path = run / 'simulation/simulation-report.json'
    simulation = read(simulation_path) if simulation_path.exists() else None
    # BEGIN locate the original executor's recorded failure on the actual scene track.
    if simulation is not None and simulation.get('feedback_failure_summary') is not None:
        failure = simulation['feedback_failure_summary']
        violation = failure['first_violation']
        markers.append(dict(kind='execution_feedback_failure', curve_key='actual',
            time_s=failure['curve_time_s'],
            world_position_m=np.asarray(failure['actual_feedback']['tool_pose'])[:3, 3],
            evidence=deepcopy(failure),
            text=f"实际执行停止：{violation['kind']} / {violation['name']}；"
                 f"曲线时间 {failure['curve_time_s']:.6f} s，物理经过 {failure['physical_elapsed_s']:.6f} s。"
                 f"实测 {violation['measured']:.9g}，参考 {violation['expected']:.9g}，"
                 f"原误差上限 {violation['maximum_error']:.9g}。红叉定位该反馈时刻的实际TCP；失败关节见原记录。"))
    # END this evidence is posthoc and never changes planning or execution decisions.
    # BEGIN independent effect verdicts retain their actual measured scene time.
    evaluation_path = run / 'simulation/evaluation-report.json'
    evaluation = read(evaluation_path) if evaluation_path.exists() else None
    if evaluation is not None and actual is not None:
        for collection in ('effect_evidence', 'terminal_evidence'):
            for row in evaluation[collection]:
                elapsed = row.get('sample_elapsed_s')
                # A missing or unrecorded measurement has no physical location;
                # its original T/F/U evidence remains in the status and report.
                if elapsed is None or not actual['elapsed'][0] <= elapsed <= actual['elapsed'][-1]:
                    continue
                sample = int(nearest_indices(actual['elapsed'], elapsed))
                markers.append(dict(kind='independent_physical_effect', curve_key='actual',
                    time_s=float(actual['curve_time'][sample]),
                    world_position_m=actual['actual_tool_pose'][sample, :3, 3],
                    evidence=dict(collection=collection, measurement=deepcopy(row),
                        physical_sample_index=sample, physical_elapsed_s=float(actual['elapsed'][sample]),
                        location_source='This run actual_tool_pose at the recorded evaluation time'),
                    text=f"独立物理判定 {row['value']}：{row['predicate']}；"
                         f"物理经过 {elapsed:.6f} s；来源 {row['source']}。"
                         '红叉定位该次测量对应的实际TCP，具体对象和判定依据见原始证据。'))
    # END evaluation facts are displayed, never reconstructed as new verdicts.

    # A recorded clock owns all reference binding. An early error without a
    # saved curve has one entrance display time, with reference/final missing.
    if clock_curve is None:
        duration, times, progress = 0., np.asarray([0.]), np.asarray([0.])
    else:
        duration = float(np.asarray(clock_curve['times'])[-1])
        actual_times = [] if actual is None else actual['curve_time']
        times = np.unique(np.r_[np.linspace(0., duration, 1001), clock_curve['times'],
                               [mark['time_s'] for mark in markers], actual_times])
        progress = np.interp(times / duration, clock_curve['progress_u'], clock_curve['progress_s'])
    tracks, arrays, binding = {}, dict(time_s=times, progress=progress), None
    if clock_curve is not None:
        phases = [SimpleNamespace(reference=PosePath.from_record(row['reference']), metadata=row['metadata'])
                  for row in audit['phases']]
        reference = BoundToolReference(SimpleNamespace(phases=phases), clock_curve['progress_u'],
            clock_curve['progress_s'], duration, feedback['eef_pose'], feedback['eef_twist'])
        reference_poses = reference.sample_numpy(times)
        tracks['reference'] = (reference_poses, '迁移后入口绑定参考 · W', '#ffd267')
        arrays['reference'] = reference_poses
        binding = reference.diagnostics
    final_label = ('当前保存关节曲线' if metadata is None else
        ('正式接受关节曲线' if success else
         '尚未完成资格检查的数值试点' if metadata['validation_status'] != 'completed' else
         '完成资格检查并被拒绝的关节曲线'))
    for key, record, label, color in (
        ('initial', initial, '初始化关节曲线的官方机器人 FK · W', '#7bc8ff'),
        ('final', curve, final_label + '的官方机器人 FK · W', '#ff8bdd')):
        if record is not None:
            poses, data = robot_poses(fixture, feedback, record, times)
            tracks[key] = (poses, label, color)
            arrays[key] = poses
            arrays.update({key + '_' + name: value for name, value in data.items()})
    if not tracks:
        tracks['entrance'] = (np.asarray([feedback['eef_pose']]), '入口反馈工具位姿 · W', '#7bc8ff')
        arrays['entrance'] = tracks['entrance'][0]
    for mark in markers:
        mark['index'] = int(np.searchsorted(times, mark['time_s']))
        if 'goal_index' in mark:
            goal = audit['pose_events'][mark['goal_index']]
            function = arrays[mark['curve_key']][mark['index']] @ np.asarray(goal['tool_from_function'])
            mark['world_position_m'] = function[:3, 3]
            # BEGIN offline error uses the same original functional geometry.
            errors = PoseEventRequirement(**goal).errors_numpy(
                arrays[mark['curve_key']][mark['index']])
            mark['evidence']['nominal_pose_errors'] = {key: float(value) for key, value in errors.items()}
            mark['evidence']['measurement_scope'] = 'saved nominal curve FK; posthoc display'
            mark['text'] += (f" 此时名义功能位置偏差 {1000*float(errors['position']):.3f} mm，"
                             f"旋转分量偏差 {np.rad2deg(float(errors['rotation'])):.3f}°。")
            # END prior measurement remains visible independently of hard membership.
        elif 'world_position_m' not in mark:
            mark['world_position_m'] = arrays[mark['curve_key']][mark['index'], :3, 3]

    # Reuse the exact annotated source/target images; these own the visible
    # functional regions and sampled pixels. Nothing is reconstructed from GT.
    encoded = (grounding / 'index.html').read_text().split('const panels=', 1)[1]
    original_panels, _ = json.JSONDecoder().raw_decode(encoded)
    source = deepcopy(original_panels[0])
    source['image'] = relative(grounding / source['image'])
    source_times = np.asarray(source['paths']['eef']['times'])
    source_clock = source_times[0] + progress * (source_times[-1] - source_times[0])
    source['indices'] = nearest_indices(source_times, source_clock)
    for key, track in source['paths'].items():
        track['color'] = '#60d9ef' if key == 'eef' else '#beb7ee'
    panels = []
    for view in request['views']:
        original = next(row for row in original_panels if row['id'] == view)
        calibration = read(Path(request['observation']) / view / 'calibration.json')
        projected = [dict(mark, pixel=project(mark['world_position_m'], calibration)) for mark in markers]
        for mark in projected:
            if 'target_world_position_m' in mark:
                mark['target_pixel'] = project(mark['target_world_position_m'], calibration)
        paths = {key: pose_track(poses, calibration, label, color)
                 for key, (poses, label, color) in tracks.items()}
        if actual is not None:
            paths['actual'] = pose_track(actual['actual_tool_pose'], calibration,
                                         '本次实际物理反馈位姿 · W', '#81edb2')
        panels.append(dict(id=view, title='本次现场 ' + view,
            image=relative(grounding / original['image']), width=original['width'],
            height=original['height'], marks=projected, paths=paths))

    actual_data = None
    if actual is not None:
        actual_data = dict(times=actual['elapsed'], curve_times=actual['curve_time'],
            nominal_indices=nearest_indices(times, actual['curve_time']),
            source=np.unique(actual['tool_pose_source']).tolist(),
            independent_observation=np.unique(actual['tool_pose_is_independent_observation']).tolist())
        arrays.update(actual_time_s=actual['elapsed'], actual_curve_time_s=actual['curve_time'],
                      actual_poses=actual['actual_tool_pose'])
    else:
        arrays.update(actual_time_s=np.empty(0), actual_curve_time_s=np.empty(0),
                      actual_poses=np.empty((0, 4, 4)))
    error = run_report.get('error') or report.get('error') or diagnostic.get('failure')
    if success is True:
        outcome = '本次规划接受。'
    elif success is False:
        outcome = f'本次规划拒绝：{error}。'
    else:
        outcome = f"本次运行状态 {run_report['status']}：{error}。"
    # BEGIN formal artifacts carry binding provenance; diagnostic artifacts name their trial.
    candidate_text = '本次未保存关节候选曲线。'
    if metadata is not None:
        source_text = '本次正式执行绑定' if success else metadata['source']
        candidate_text = (f"来源 {source_text}；分类 {metadata['classification']}；"
                          f"验收状态 {metadata['validation_status']}。")
    # END the formal writer's unused diagnostic-source field is not a missing binding.
    actual_missing = '本次实际物理轨迹缺失；执行状态以运行报告为准。'
    outcome += candidate_text + ('已加载本次实际物理反馈。' if actual is not None else actual_missing)
    if simulation is not None:
        outcome += f"物理结果：{simulation['physics_outcome']}。"
        if simulation.get('error'):
            outcome += f"执行记录：{simulation['error']}。"
    status = [['规划结果', str(success)], ['运行状态', run_report['status']],
              ['候选来源与状态', candidate_text], ['实际物理记录', '存在' if actual is not None else '缺失']]
    if initial is not None:
        status.append(['初始化资格结果', json.dumps(dict(accepted=initial_report['accepted'],
            validation_status=initial_report['validation_status'],
            residuals=initial_report['residuals']), ensure_ascii=False)])
    for label, key in [('入口检查', 'initial_state'), ('数值求解', 'optimizer'),
                       ('累计规划工作', 'consumed_planning')]:
        if key in diagnostic:
            value = diagnostic[key]
            if key == 'optimizer':
                value = {name: value[name] for name in ('status', 'message', 'iterations', 'objective_evaluations') if name in value}
            status.append([label, json.dumps(value, ensure_ascii=False)])
    # BEGIN accepted motion retains optimization interruption and interval scope.
    if 'optimization_interruption' in diagnostic:
        interruption = diagnostic['optimization_interruption']
        status.append(['先验优化中断', json.dumps({key: interruption[key] for key in
            ('failure', 'failed_stage', 'reason', 'error_type')}, ensure_ascii=False)])
    if 'continuous_geometry' in selected:
        geometry = selected['continuous_geometry']
        status.append(['区间几何查询范围', json.dumps(dict(
            query_points=geometry['query_points'], hard_scene_residual=geometry['hard_scene_residual'],
            unresolved_intervals=len(geometry['unresolved_intervals']),
            scope='原有界查询结果；未决区间保留在完整规划报告'), ensure_ascii=False)])
    # END a formal accepted record does not relabel missing continuous evidence.
    if simulation is not None:
        status.append(['物理结果', json.dumps({key: simulation[key] for key in
            ('status', 'error', 'physics_outcome', 'actual_physics_steps') if key in simulation}, ensure_ascii=False)])
        if actual is not None:
            # BEGIN show the physical range actually present in the saved pose archive.
            status.append(['实际位姿记录范围', json.dumps(dict(
                elapsed_interval_s=[float(actual['elapsed'][0]), float(actual['elapsed'][-1])],
                samples=len(actual['elapsed']),
                trajectory_physics_steps=simulation['trajectory_executed_steps'],
                additional_physics_steps=simulation['actual_physics_steps']-simulation['trajectory_executed_steps'],
                scope='归档覆盖入口至执行停止；后续保持的逐步位姿不在该归档中'), ensure_ascii=False)])
            # END missing hold poses remain separate from the complete recorded motion prefix.
    # BEGIN all independent outcomes, including missing T/F/U measurements, remain visible.
    if evaluation is not None:
        status.append(['独立效果与终停证据', json.dumps({key: evaluation[key] for key in
            ('effect_evidence', 'terminal_evidence')}, ensure_ascii=False)])
    # END a completed physical trajectory does not imply a successful task effect.
    status.append(['资源峰值', f"GPU {resources['peak_device_bytes']/1e9:.3f} GB；"
                   f"进程树 RSS {resources['peak_process_tree_rss_bytes']/1e9:.3f} GB"])
    events = [dict(label='入口', index=0), *[dict(label=mark['kind'], index=mark['index']) for mark in markers],
              dict(label='保存曲线末端', index=len(times)-1)]
    link_inputs = [('本次运行报告', run/'planning-run/run-report.json'), ('本次规划报告', report_path),
        ('本次曲线说明', curve_path.with_suffix('.json')), ('本次曲线', curve_path),
        ('原迁移区域与采样点', grounding/'index.html'), ('原请求', args.request), ('资源记录', args.resources),
        ('本次结果摘要', run/'result-summary.json'), ('本次异常现场', run/'planning-run/exception.txt'),
        ('联合计算记录', run/'planning-run/planner-internals/joint-problem.json'),
        ('数值试点系数与目标', run/'planning-run/planner-internals/objective-trials.json'),
        ('物理报告', simulation_path), ('物理位姿记录', actual_path),
        # BEGIN the same report links expose model provenance and actual result evidence.
        ('独立物理评价', evaluation_path),
        ('固定机器人流体模型', run/'simulation-inputs/robot-fluid-model.json'),
        ('跟踪与完成条件复核', run/'tracking-review.json'),
        ('物理判定现场 view_a', run/'effect-nearest-view_a.png'),
        ('物理判定现场 view_b', run/'effect-nearest-view_b.png')]
        # END these are offline artifacts belonging to this run.
    links = [dict(label=label, path=relative(path)) for label, path in link_inputs if path.exists()]
    dependencies = [] if metadata is None else metadata['satisfied_dependencies']
    videos = [] if simulation is None else [dict(view=row['view_id'],
        path=relative(row['path']), capture_times=relative(row['capture_times_path']),
        timing=row['timing']) for row in simulation['videos']]
    display = dict(planning_success=success,
        task=f"当前任务：{Path(request['skill']).name} / {request['segment']}；"
             f"已完成依赖：{dependencies}（空列表表示本次无已声明完成依赖）；"
             f"入口采用 {request['feedback_id']} 的原生 {feedback['sim_time_s']:.3f} s 关节位置与速度。",
        outcome=outcome, status=status, scope='W 是本次标定世界坐标系，W_D 是源示教世界坐标系；'
            'E 是工具控制坐标系。名义轨迹由保存关节多项式及官方固定机器人运动学计算。'
            '判定证据沿用对应候选的原报告；实际物理位姿仅用于本页面事后比较。',
        times=times, progress=progress, native_start_s=feedback['sim_time_s'], source=source,
        panels=panels, markers=markers, events=events,
        initial_index=next((mark['index'] for mark in markers
                            if mark['kind'] == 'execution_feedback_failure'),
                           next((mark['index'] for mark in markers
                                 if mark['kind'] == 'independent_physical_effect'
                                 and mark['evidence']['measurement']['value'] == 'F'),
                                markers[0]['index'] if markers else 0)),
        actual=actual_data, actual_missing_text=actual_missing, failure_summary=outcome,
        # BEGIN the full binding is already linked as the curve metadata artifact.
        track_keys=list(tracks), links=links, videos=videos,
        # END the page consumes paths, poses, markers and statuses, not a duplicate field archive.
        missing_tracks=[name for name in ('reference', 'initial', 'final', 'actual')
                        if name not in panels[0]['paths']],
        provenance=dict(request=str(args.request.resolve()), run=str(run), reference_binding=binding,
            saved_curve=None if curve is None else str(curve_path),
            actual_trajectory=None if actual is None else str(actual_path),
            report_error=error, extra_planning_calls=0, extra_geometry_queries=0, new_physics_steps=0))
    checks = dict(samples=len(times), tracks=list(tracks), marker_count=len(markers),
        pose_finite=all(bool(np.isfinite(values[0]).all()) for values in tracks.values()),
        images_exist=all((output/p['image']).exists() for p in [source, *panels]),
        links_exist=all((output/link['path']).exists() for link in links),
        exact_witness_times=all(times[m['index']] == m['time_s'] for m in markers),
        actual_physical_poses=0 if actual is None else len(actual['elapsed']),
        saved_curve_source=None if metadata is None else metadata['source'],
        saved_validation_status=None if metadata is None else metadata['validation_status'],
        actual_full_sample_access=actual is None or len(actual_data['nominal_indices']) == len(actual['elapsed']),
        extra_planning_calls=0, extra_geometry_queries=0, new_physics_steps=0)
    arrays.update(source_clock=source_clock, source_display_times=source_times[source['indices']])
    np.savez_compressed(output/'display-poses.npz', **arrays)
    # BEGIN convert the identical completed display tree once for its two formats.
    display = plain(display)
    for name, value in [('display-data.json', display), ('display-checks.json', checks)]:
        (output/name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    encoded = json.dumps(display, ensure_ascii=False, allow_nan=False).replace('</', '<\\/')
    # END the standalone HTML retains the same visible data and controls.
    (output/'index.html').write_text((HERE/'result_template.html').read_text().replace('__DISPLAY_DATA__', encoded))
    print(json.dumps(checks, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
