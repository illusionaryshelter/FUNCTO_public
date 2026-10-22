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

from transfer.multiview_visualization import overlay, REGION_COLOR
from transfer.dual_view.visualization import region_picture


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


def write_reference_html(directory, skill, views, transfer, reference, report):
    """Show observed regions and nominal source-time paths with explicit goals."""
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
    source_image.save(directory / "source-region.png")
    # END these are source queries, not invented current correspondences.
    source_frame = (source["T_world_body_at_image"] @ source["L_body_region"]
                    if "L_body_region" in source else None)
    panels = [dict(id="source", title="源单视角：已给区域", image="source-region.png",
                   # BEGIN canvas coordinates use the input image's pixel grid.
                   width=original.shape[1], height=original.shape[0],
                   # END no fixed camera resolution is assumed.
                   axes=_axes(source_frame, source["observation"]), paths={})]
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
        picture.save(directory / filename)
        paths = {}
        for name in ("object", "eef"):
            if reference and name + "_poses" in reference:
                # BEGIN display the preserved nominal composed path.
                # Additional display samples do not change reference knots or
                # the numerical algorithm; they avoid drawing a chord across
                # a rotating object/tool origin with a fixed lever arm.
                path = reference[name + "_path"]
                original_times = np.asarray(reference[name + "_timestamps"])
                progress = (np.unique(np.r_[path.progress, np.linspace(0., 1., 201)])
                            if len(original_times) > 1 else np.array([0.]))
                poses = path.sample_numpy(progress)
                times = original_times[0] + progress*(original_times[-1]-original_times[0])
                # END physical-time adaptation is recorded by the planner.
                paths[name] = dict(pixels=_project(poses[:, :3, 3], view),
                                   times=times.tolist())
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
                           paths=paths, markers=markers))
        # END each marker retains its declared geometric role.
    details = json.dumps(report, ensure_ascii=False, indent=2)
    cards = "".join(f'<article><h2>{escape(panel["title"])}</h2>'
                    f'<div class="viewport"><canvas id="{panel["id"]}" width="{panel["width"]}" height="{panel["height"]}"></canvas></div>'
                    f'<a href="{panel["image"]}" target="_blank">打开原分辨率区域图</a></article>' for panel in panels)
    data = json.dumps(panels, ensure_ascii=False).replace("</", "<\\/")
    html = """<!doctype html><meta charset="utf-8"><title>技能区域迁移与参考轨迹</title>
<style>body{font:16px system-ui;background:#10151d;color:#e9eff7;margin:24px}h1{font-size:24px}h2{font-size:17px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(400px,1fr));gap:18px}article{padding:14px;background:#1c2531;border-radius:8px}.viewport{overflow:auto;max-height:75vh}canvas{width:100%;display:block}a{color:#8bcdff}pre{white-space:pre-wrap;overflow-wrap:anywhere}label{margin-right:24px}.controls{position:sticky;top:0;background:#10151df5;padding:12px 0;z-index:2}.status{padding:10px;border-left:4px solid #e8b450}.legend{color:#bdc9d8}</style>
<h1>技能区域迁移与参考轨迹</h1>
<p class="status">流程：RUN_STATUS · 区域：REGION_STATUS · 名义参考：REFERENCE_STATUS。本页展示 grounding 的几何输出与边界声明。</p>
FAILURE_SUMMARY
REGION_WARNINGS
<p>真实源图与当前双视角图并排显示。区域色块与黄色采样点来自本次计算；拒绝时仍保留诊断区域。源区域是已有输入，当前区域没有使用目标区域真值。</p>
<div class="controls"><label>图像放大 <input id="zoom" type="range" min="1" max="4" value="1" step=".25"></label><label>参考时间 <input id="progress" type="range" min="0" max="1000" value="0"></label><span id="time"></span></div>
<p class="legend">坐标轴：红 x、绿 y、蓝 z（显示长度 4 cm）。青色为物体名义路径，橙色为工具名义路径。白色十字为声明工具入口，绿色圆环为名义路径首点，粉色菱形为名义事件目标。滑块选择源示教时间；圆点表示该时间之前最近的名义曲线采样。规划报告另行展示实际物理时钟、入口速度及边界适配后的完整工具参考。</p>
<div class="grid">CARDS</div><details open><summary>本次状态、输入来源与缺项</summary><pre>DETAILS</pre></details>
<p><a href="report.json">完整运行记录</a>REFERENCE_LINK REGION_LINK</p>
<script>const panels=DATA;const images={};let ready=0;const slider=document.getElementById('progress');
const allTimes=panels.flatMap(p=>Object.values(p.paths).flatMap(v=>v.times));const begin=allTimes.length?Math.min(...allTimes):0,end=allTimes.length?Math.max(...allTimes):0;
function dot(ctx,p,color,r){if(!p)return;ctx.fillStyle=color;ctx.beginPath();ctx.arc(p[0],p[1],r,0,2*Math.PI);ctx.fill()}
// BEGIN project declared entrance, nominal first point and explicit event goals.
function boundaries(ctx,p){for(const m of p.markers||[]){if(!m.pixel)continue;const [x,y]=m.pixel;ctx.strokeStyle=m.color;ctx.lineWidth=2;ctx.beginPath();if(m.kind==='declared_entrance'){ctx.moveTo(x-7,y);ctx.lineTo(x+7,y);ctx.moveTo(x,y-7);ctx.lineTo(x,y+7)}else if(m.kind==='nominal_start'){ctx.arc(x,y,9,0,2*Math.PI)}else{ctx.moveTo(x,y-8);ctx.lineTo(x+8,y);ctx.lineTo(x,y+8);ctx.lineTo(x-8,y);ctx.closePath()}ctx.stroke();ctx.fillStyle=m.color;ctx.font='12px sans-serif';ctx.fillText(m.label,x+12,y+(m.kind==='nominal_start'?18:-12))}}
// END marker roles follow the generator's explicit boundary record.
function draw(){const time=begin+(end-begin)*Number(slider.value)/1000;document.getElementById('time').textContent=allTimes.length?time.toFixed(3)+' s':'没有可用参考';for(const p of panels){const c=document.getElementById(p.id),ctx=c.getContext('2d');if(!images[p.id].complete)continue;ctx.clearRect(0,0,c.width,c.height);ctx.drawImage(images[p.id],0,0);boundaries(ctx,p);for(const [name,path]of Object.entries(p.paths)){ctx.strokeStyle=name==='object'?'#40e2ed':'#ff9e38';ctx.lineWidth=2;ctx.beginPath();let connected=false;for(const point of path.pixels){if(!point){connected=false;continue}if(connected)ctx.lineTo(...point);else ctx.moveTo(...point);connected=true}ctx.stroke();let i=0;while(i+1<path.times.length&&path.times[i+1]<=time)i++;if(time<=path.times[path.times.length-1])dot(ctx,path.pixels[i],ctx.strokeStyle,5)}if(p.axes&&p.axes[0]){for(let i=1;i<4;i++){if(!p.axes[i])continue;ctx.strokeStyle=['','#ff554e','#4de07b','#559bff'][i];ctx.lineWidth=3;ctx.beginPath();ctx.moveTo(...p.axes[0]);ctx.lineTo(...p.axes[i]);ctx.stroke()}}}}
for(const p of panels){const im=new Image();images[p.id]=im;im.onload=()=>draw();im.src=p.image}slider.oninput=draw;document.getElementById('zoom').oninput=e=>document.querySelectorAll('canvas').forEach(c=>c.style.width=(100*Number(e.target.value))+'%');</script>"""
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
                        if (directory / "reference.npz").is_file() else "")
    html = html.replace("CARDS", cards).replace("DETAILS", escape(details)).replace("DATA", data)
    (directory / "index.html").write_text(html, encoding="utf-8")
