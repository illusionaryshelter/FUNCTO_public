"""Local HTML for observed regions, nominal paths and declared boundaries.

Images are recorded RGB observations. The existing dual-view renderer paints
region membership and supported surface samples. Nominal path projections and
declared entrance/event poses show the geometric inputs to physical-time
reference binding; the planning report owns that binding's resulting path.
"""

import json
from html import escape
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_erosion
from scipy.spatial.transform import Rotation

from transfer.multiview_visualization import overlay, REGION_COLOR
from transfer.dual_view.visualization import region_picture
from .representation import sample_poses


def _project(points, calibration):
    camera, K = np.asarray(calibration["T_world_camera_cv"]), np.asarray(calibration["K"])
    local = (np.asarray(points) - camera[:3, 3]) @ camera[:3, :3]
    pixels = local @ K.T
    return [None if point[2] <= 0 else (point[:2] / point[2]).tolist() for point in pixels]


def _axes(frame, calibration):
    if frame is None:
        return None
    pose = np.asarray(frame)
    points = np.vstack([pose[:3, 3], pose[:3, 3] + .04 * pose[:3, :3].T])
    return _project(points, calibration)


# BEGIN trajectory display retains complete world poses on one shared source clock.
def _pose_track(poses, times, calibration, *, label, source):
    """Project a sampled pose track and retain its numerical orientation.

    Each pose maps the named moving frame into the panel's world frame.
    Axis endpoints are four centimetres from its origin. RPY is extrinsic
    xyz in degrees; the complete rotation matrix is the primary orientation.
    """
    poses = np.asarray(poses)
    return dict(pixels=_project(poses[:, :3, 3], calibration), times=np.asarray(times).tolist(),
                axes=[_axes(pose, calibration) for pose in poses], poses=poses.tolist(),
                rpy_deg=Rotation.from_matrix(poses[:, :3, :3]).as_euler("xyz", degrees=True).tolist(),
                label=label, source=source)


def _entity_outline(image, mask):
    """Retain the selected observed entity extent around its functional region."""
    pixels = np.array(image, copy=True)
    boundary = mask & ~binary_erosion(mask, structure=np.ones((3, 3), bool), border_value=0)
    pixels[boundary] = (85, 240, 178)
    return Image.fromarray(pixels)
# END these display operations neither change a path nor select a region.


def write_reference_html(directory, skill, views, transfer, reference, report):
    """Show source motion and transferred full poses in calibrated scene images.

    A shared source-time sample drives one pose per displayed track. Source
    poses are recorded demonstration evidence; current paths are generated
    nominal references. A grounding report has no current execution trajectory.
    """
    directory = Path(directory)
    source = skill["source"]
    original = np.asarray(source["image"])
    region_mask = source.get("region_image_mask", np.zeros(original.shape[:2], bool))
    source_picture = overlay(original, region_mask, REGION_COLOR)
    # BEGIN show the actual source-region query pixels from the native sampler.
    from transfer.joint_features import source_queries
    source_image = Image.fromarray(source_picture)
    if np.any(source["observation"]["region_mask"]):
        indices, membership = source_queries(source["observation"])
        draw = ImageDraw.Draw(source_image)
        for x, y in np.asarray(source["observation"]["pixels"])[indices[membership]]:
            draw.ellipse((x-1, y-1, x+1, y+1), fill=(255, 218, 70))
    source_image = _entity_outline(source_image, source["observation"]["object_mask"])
    source_image.save(directory / "source-region.png")
    # END these are source queries, not invented current correspondences.
    source_frame = (source["T_world_body_at_image"] @ source["L_body_region"]
                    if "L_body_region" in source else None)
    # BEGIN sample every available track at identical source-clock instants.
    representation = skill["representation"]
    source_times = np.asarray(representation["timestamps"])
    clocks = [source_times, np.linspace(source_times[0], source_times[-1], 201)]
    for name in ("object", "eef"):
        if reference and name + "_path" in reference:
            clocks.append(np.asarray(reference[name + "_timestamps"]))
    display_times = np.unique(np.concatenate(clocks))
    source_paths = {}
    for name, field, label in (("object", "T_world_carrier", "源物体实际记录轨迹"),
                                ("eef", "T_world_eef", "源工具实际记录轨迹")):
        if field in representation:
            poses = sample_poses(representation[field], source_times, display_times)
            source_paths[name] = _pose_track(poses, display_times, source["observation"],
                label=label, source="skill representation: " + field)
    # END original timestamps and event knots remain in the display clock.
    panels = [dict(id="source", title="源单视角：区域与原示教运动", image="source-region.png",
                   # BEGIN canvas coordinates use the input image's pixel grid.
                   width=original.shape[1], height=original.shape[0],
                   # END no fixed camera resolution is assumed.
                   axes=_axes(source_frame, source["observation"]), paths=source_paths,
                   world_frame="源示教世界坐标 W_D")]
    outputs = transfer.get("outputs") if transfer else None
    frame = transfer.get("reference_frame") if transfer else None
    for index, view in enumerate(views):
        image = np.asarray(view["image"])
        picture = Image.fromarray(image)
        if outputs is not None:
            modes = outputs["result"].get("modes", [])
            if modes:
                # The first native mode is the primary region (accepted or
                # diagnostic). Its actual status is displayed independently.
                picture, _ = region_picture(image, outputs["surface"], modes[0], index)
        filename = view["id"] + "-region.png"
        if view.get("observation") is not None:
            picture = _entity_outline(picture, view["observation"]["object_mask"])
        picture.save(directory / filename)
        paths = {}
        for name in ("object", "eef"):
            if reference and name + "_poses" in reference:
                # BEGIN display the complete nominal path on the shared clock.
                path = reference[name + "_path"]
                original_times = np.asarray(reference[name + "_timestamps"])
                times = display_times[(display_times >= original_times[0]) &
                                      (display_times <= original_times[-1])]
                progress = ((times-original_times[0])/(original_times[-1]-original_times[0])
                            if len(original_times) > 1 else np.zeros(len(times)))
                poses = path.sample_numpy(progress)
                paths[name] = _pose_track(poses, times, view,
                    label="物体名义路径" if name == "object" else "工具名义路径",
                    source="generate_reference: " + name + "_path")
                # END physical execution time is assigned by the planner.
        # BEGIN declared tool boundaries are separate from nominal path samples.
        # The generator supplies world-frame initial_pose and local-progress
        # goal poses. Their physical-time connection belongs to BoundToolReference.
        markers = []
        if reference and "eef_boundaries" in reference:
            declaration = reference["eef_boundaries"]
            initial = _project(np.asarray(declaration["initial_pose"])[None, :3, 3], view)[0]
            nominal = _project(reference["eef_path"].sample_numpy(np.asarray([0.]))[:, :3, 3], view)[0]
            markers.extend((dict(kind="declared_entrance", label="声明工具入口", color="#ffffff", pixel=initial),
                            dict(kind="nominal_start", label="名义路径首点", color="#62ff9b", pixel=nominal)))
            for index, goal in enumerate(declaration["goals"]):
                point = _project(np.asarray(goal["pose"])[None, :3, 3], view)[0]
                markers.append(dict(kind="nominal_event", label=f"名义事件目标 {index}", color="#ff75df",
                                    pixel=point, progress=float(goal["progress"]), source=goal["source"]))
        panels.append(dict(id=view["id"], title="当前 " + view["id"] + "：区域与采样点",
                           image=filename, width=image.shape[1], height=image.shape[0],
                           axes=_axes(frame["T_world_region"] if frame else None, view),
                           paths=paths, markers=markers, world_frame="当前运行世界坐标 W_N"))
        # END each marker retains its declared geometric role.
    details = json.dumps(report, ensure_ascii=False, indent=2)
    cards = "".join(f'<article><h2>{escape(panel["title"])}</h2>'
                    f'<div class="viewport"><canvas id="{panel["id"]}" width="{panel["width"]}" height="{panel["height"]}"></canvas></div>'
                    f'<a href="{panel["image"]}" target="_blank">打开原分辨率区域图</a>'
                    f'<pre id="pose-{panel["id"]}"></pre></article>' for panel in panels)
    data = json.dumps(panels, ensure_ascii=False).replace("</", "<\\/")
    html = r"""<!doctype html><meta charset="utf-8"><title>技能区域迁移与参考轨迹</title>
<style>body{font:16px system-ui;background:#10151d;color:#e9eff7;margin:24px}h1{font-size:24px}h2{font-size:17px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(400px,1fr));gap:18px}article{padding:14px;background:#1c2531;border-radius:8px}.viewport{overflow:auto;max-height:75vh}canvas{width:100%;display:block}a{color:#8bcdff}pre{white-space:pre-wrap;overflow-wrap:anywhere}label{margin-right:24px}.controls{position:sticky;top:0;background:#10151df5;padding:12px 0;z-index:2}.status{padding:10px;border-left:4px solid #e8b450}.legend{color:#bdc9d8}</style>
<h1>技能区域迁移与参考轨迹</h1>
<p class="status">流程：RUN_STATUS · 区域：REGION_STATUS · 名义参考：REFERENCE_STATUS。本页展示 grounding 的几何输出与边界声明。</p>
FAILURE_SUMMARY
REGION_WARNINGS
<p>源单视角与当前左右视角并排显示。绿色轮廓为实体分割，区域色块与黄色采样点来自本次计算；拒绝时保留诊断区域。源图叠加原示教实际记录，当前图叠加迁移参考。</p>
<div class="controls"><label>图像放大 <input id="zoom" type="range" min="1" max="4" value="1" step=".25"></label><label>源时间对应进度 <input id="progress" type="range" min="0" max="0" value="0" step="1"></label><span id="time"></span></div>
<p class="legend">青色为物体轨迹，橙色为工具轨迹；当前两图对应物体名义路径与工具名义路径。每条轨迹只显示所选时刻的一个位姿：红 x、绿 y、蓝 z，轴长 4 cm；虚线 F 为功能坐标。白色十字为声明工具入口，绿色圆环为名义路径首点，粉色菱形为名义事件目标。</p>
<p>位姿 T_W_E 或 T_W_B 将工具 E 或物体 B 坐标映到各图标注的世界坐标。位置单位 m；RPY 为外禀 XYZ 角度，R=Rz(yaw) Ry(pitch) Rx(roll)，完整旋转矩阵显示在图下。三个视图使用同一源示教时刻，源时间差表示示教经过时长。</p>
<p class="status">当前实际物理轨迹：缺失（本次记录为 grounding 与参考生成）。当前机器人反馈只提供声明入口；执行时钟及其实际轨迹由规划与执行记录给出。</p>
<div class="grid">CARDS</div><details open><summary>本次状态、输入来源与缺项</summary><pre>DETAILS</pre></details>
<p><a href="report.json">完整运行记录</a>REFERENCE_LINK REGION_LINK</p>
<script>const panels=DATA;const images={};const slider=document.getElementById('progress');
// BEGIN one shared source-time sample selects exactly one pose on each track.
const allTimes=[...new Set(panels.flatMap(p=>Object.values(p.paths).flatMap(v=>v.times)))].sort((a,b)=>a-b);const begin=allTimes.length?allTimes[0]:0,end=allTimes.length?allTimes[allTimes.length-1]:0;slider.max=Math.max(0,allTimes.length-1);
function dot(ctx,p,color,r){if(!p)return;ctx.fillStyle=color;ctx.beginPath();ctx.arc(p[0],p[1],r,0,2*Math.PI);ctx.fill()}
function axes(ctx,points,label,dashed=false){if(!points||!points[0])return;ctx.setLineDash(dashed?[4,3]:[]);for(let i=1;i<4;i++){if(!points[i])continue;ctx.strokeStyle=['','#ff554e','#4de07b','#559bff'][i];ctx.lineWidth=dashed?2:3;ctx.beginPath();ctx.moveTo(...points[0]);ctx.lineTo(...points[i]);ctx.stroke();ctx.fillStyle=ctx.strokeStyle;ctx.font='12px sans-serif';ctx.fillText(label+' '+['','x','y','z'][i],points[i][0]+3,points[i][1]-3)}ctx.setLineDash([])}
function numbers(path,i,name){const T=path.poses[i],p=T.slice(0,3).map(row=>row[3].toFixed(6)),rpy=path.rpy_deg[i].map(v=>v.toFixed(3)),R=T.slice(0,3).map(row=>row.slice(0,3).map(v=>v.toFixed(6)).join('  '));return path.label+'\n数据：'+path.source+'\n时间：'+path.times[i].toFixed(3)+' s\n位置 [m]：'+p.join(', ')+'\nRPY [deg]：'+rpy.join(', ')+'\nR_W_'+(name==='eef'?'E':'B')+'：\n'+R.join('\n')}
// END world-frame matrices accompany the projected orientation axes.
// BEGIN project declared entrance, nominal first point and explicit event goals.
function boundaries(ctx,p){for(const m of p.markers||[]){if(!m.pixel)continue;const [x,y]=m.pixel;ctx.strokeStyle=m.color;ctx.lineWidth=2;ctx.beginPath();if(m.kind==='declared_entrance'){ctx.moveTo(x-7,y);ctx.lineTo(x+7,y);ctx.moveTo(x,y-7);ctx.lineTo(x,y+7)}else if(m.kind==='nominal_start'){ctx.arc(x,y,9,0,2*Math.PI)}else{ctx.moveTo(x,y-8);ctx.lineTo(x+8,y);ctx.lineTo(x,y+8);ctx.lineTo(x-8,y);ctx.closePath()}ctx.stroke();ctx.fillStyle=m.color;ctx.font='12px sans-serif';ctx.fillText(m.label,x+12,y+(m.kind==='nominal_start'?18:-12))}}
// END marker roles follow the generator's explicit boundary record.
function draw(){const time=allTimes[Number(slider.value)];document.getElementById('time').textContent=allTimes.length?time.toFixed(3)+' s · 示教经过 '+(time-begin).toFixed(3)+' s':'没有可用轨迹';for(const p of panels){const c=document.getElementById(p.id),ctx=c.getContext('2d');if(!images[p.id].complete)continue;ctx.clearRect(0,0,c.width,c.height);ctx.drawImage(images[p.id],0,0);boundaries(ctx,p);axes(ctx,p.axes,'F',true);const info=[p.world_frame];for(const [name,path]of Object.entries(p.paths)){const color=name==='object'?'#40e2ed':'#ff9e38';ctx.strokeStyle=color;ctx.lineWidth=2;ctx.beginPath();let connected=false;for(const point of path.pixels){if(!point){connected=false;continue}if(connected)ctx.lineTo(...point);else ctx.moveTo(...point);connected=true}ctx.stroke();const i=path.times.indexOf(time);if(i>=0){dot(ctx,path.pixels[i],color,5);axes(ctx,path.axes[i],name==='eef'?'E':'B');info.push(numbers(path,i,name))}else info.push(path.label+'：所选时间在该轨迹区间外')}document.getElementById('pose-'+p.id).textContent=info.join('\n\n')}}
for(const p of panels)images[p.id]=new Image();for(const p of panels){images[p.id].onload=draw;images[p.id].src=p.image}slider.oninput=draw;document.getElementById('zoom').oninput=e=>document.querySelectorAll('canvas').forEach(c=>c.style.width=(100*Number(e.target.value))+'%');</script>"""
    html = html.replace("REGION_STATUS", escape(str(report.get("region_status", report.get("status")))))
    # BEGIN transport/computation failures are distinct from region rejection.
    html = html.replace("RUN_STATUS", escape(str(report.get("status", "missing"))))
    failure = (f'<p class="status">失败步骤：{escape(str(report["failed_stage"]))}；'
               f'{escape(str(report.get("error", "")))}</p>' if "failed_stage" in report else "")
    html = html.replace("FAILURE_SUMMARY", failure)
    # BEGIN accepted curved regions retain every native quality warning.
    warnings = report.get('region_warnings', [])
    warning_text = ('<p class="status">区域警告：'+escape('；'.join(warnings))+'</p>') if warnings else ''
    html = html.replace('REGION_WARNINGS', warning_text)
    html = html.replace('REGION_LINK', ' · <a href="region.json">区域候选与观测诊断</a>'
                        if (directory/'region.json').is_file() else '')
    # END warnings are visible beside the region and reference decisions.
    # END preserve the native status without presenting missing as rejection.
    html = html.replace("REFERENCE_STATUS", escape(str(reference.get("status", "missing") if reference else "missing")))
    html = html.replace("REFERENCE_LINK", ' · <a href="reference.npz">参考数值</a>'
                        + (' · <a href="reference-paths.json">完整连续路径与事件边界</a>'
                           if (directory / "reference-paths.json").is_file() else "")
                        if (directory / "reference.npz").is_file() else "")
    html = html.replace("CARDS", cards).replace("DETAILS", escape(details)).replace("DATA", data)
    (directory / "index.html").write_text(html, encoding="utf-8")
