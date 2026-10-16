"""Original simulation images for independently frozen three-scheme results."""

from html import escape
import json
from pathlib import Path

import numpy as np
# BEGIN evaluation-only GT layer uses the existing Pillow morphology tools.
from PIL import Image, ImageDraw, ImageFilter
# END evaluation-only GT layer imports.

from ..multiview_visualization import overlay, REGION_COLOR, VIEW_COLORS
from .geometry import project_oriented


# BEGIN held-out GT overlay, composited after every prediction marker.
GT_COLOR = (255, 64, 207, 255)


def ground_truth_layer(support_mask, reference, intrinsic, world_camera, *, center_visible):
    """Return one reusable RGBA layer for a calibrated target camera.

    The contour denotes the *visible complete support surface*, not a guessed
    local transfer patch. The ring denotes the saved transformed demonstration
    mean. Visibility comes from the held-out pair annotation; an occluded
    center is deliberately drawn as a dashed projection and labelled hidden.
    No predicted region, sample, fit, or acceptance decision enters this layer.
    """
    mask = np.asarray(support_mask)
    if mask.ndim != 2 or mask.dtype != np.bool_:
        raise ValueError('GT support mask must be a full-image Boolean array')
    height, width = mask.shape
    layer = Image.new('RGBA', (width, height))
    mask_image = Image.fromarray(mask.astype(np.uint8)*255)
    # Keep the colored contour inside actual visible support pixels. A narrow
    # dark halo provides contrast against both light furniture and yellow dots.
    edge = mask & ~np.asarray(mask_image.filter(ImageFilter.MinFilter(5)), dtype=bool)
    halo = Image.fromarray(edge.astype(np.uint8)*255).filter(ImageFilter.MaxFilter(3))
    layer.paste((15, 18, 29, 255), mask=halo)
    layer.paste(GT_COLOR, mask=Image.fromarray(edge.astype(np.uint8)*255))

    reference = np.asarray(reference, dtype=float)
    intrinsic, world_camera = np.asarray(intrinsic), np.asarray(world_camera)
    if (reference.shape != (3,) or intrinsic.shape != (3, 3)
            or world_camera.shape != (4, 4) or not np.isfinite(reference).all()
            or not np.isfinite(intrinsic).all() or not np.isfinite(world_camera).all()):
        raise ValueError('GT projection requires a finite world point and calibrated camera')
    camera = (reference-world_camera[:3, 3]) @ world_camera[:3, :3]
    projected = camera @ intrinsic.T
    uv = projected[:2]/projected[2] if camera[2] > 0 and projected[2] > 0 else None
    inside = uv is not None and 0 <= uv[0] < width and 0 <= uv[1] < height
    state = ('visible' if center_visible else 'occluded') if inside else (
        'behind_camera' if uv is None else 'outside_image')
    if inside:
        x, y = uv
        draw = ImageDraw.Draw(layer)
        bounds = (x-10, y-10, x+10, y+10)
        arcs = [(0, 360)] if center_visible else [(a, a+36) for a in range(0, 360, 60)]
        for start, end in arcs:
            # Opaque outer stroke, white separator and magenta inner stroke
            # remain legible even when the prediction center coincides exactly.
            draw.arc((x-12, y-12, x+12, y+12), start, end, fill='#10121d', width=7)
            draw.arc((x-11, y-11, x+11, y+11), start, end, fill='white', width=5)
            draw.arc(bounds, start, end, fill=GT_COLOR, width=3)
        draw.ellipse((x-2, y-2, x+2, y+2), fill=GT_COLOR, outline='#10121d')
        label = 'GT' if center_visible else 'GT hidden'
        text_box = draw.textbbox((0, 0), label)
        text_width, text_height = text_box[2]-text_box[0], text_box[3]-text_box[1]
        left = min(max(int(x+15), 2), width-text_width-6)
        top = min(max(int(y-19), 2), height-text_height-6)
        draw.rectangle((left-2, top-2, left+text_width+3, top+text_height+3), fill='#10121d')
        draw.text((left-text_box[0], top-text_box[1]), label, fill=GT_COLOR)
    return layer, {'support_visible_pixels': int(mask.sum()),
        'reference_world': reference.tolist(), 'reference_pixel': None if uv is None else uv.tolist(),
        'reference_visible': bool(center_visible), 'reference_display': state,
        'contour_definition': 'visible_complete_preferred_support',
        'center_definition': 'held_out_transformed_source_demonstration_mean',
        'color': '#ff40cf', 'draw_order': 'after_prediction_regions_samples_and_center'}


def with_ground_truth(picture, layer):
    """Composite GT last; never let yellow samples overwrite its opaque marks."""
    if picture.size != layer.size or layer.mode != 'RGBA':
        raise ValueError('GT layer must share the original target image coordinates')
    return Image.alpha_composite(picture.convert('RGBA'), layer).convert('RGB')
# END held-out GT overlay.


def region_picture(rgb, surface, mode, view_index):
    """Paint real observed region pixels and strictly qualified sample markers."""
    mask = np.asarray(mode['mask'])
    if mask.dtype != np.bool_ or mask.shape != (len(surface['xyz']),):
        raise ValueError('Displayed region mask must index actual surface elements')
    observations = np.flatnonzero(surface['observation_view_index'] == view_index)
    elements = surface['observation_to_element'][observations]
    selected = elements >= 0
    selected[selected] &= mask[elements[selected]]
    observations, elements = observations[selected], elements[selected]
    pixels = surface['observation_pixels'][observations]
    # Colors identify saved measurement membership only. They make no claim
    # that both views qualify for the new oriented appearance score.
    members = np.asarray(surface['member_views'], dtype=bool)[elements]
    both = members.all(axis=1)
    painted = np.array(rgb, copy=True)
    for chosen, color in ((both, REGION_COLOR), (~both & members[:, 0], VIEW_COLORS[0]),
                          (~both & members[:, 1], VIEW_COLORS[1])):
        image_mask = np.zeros(rgb.shape[:2], dtype=bool)
        xy = pixels[chosen]
        image_mask[xy[:, 1], xy[:, 0]] = True
        painted = overlay(painted, image_mask, color)
    image = Image.fromarray(painted)
    samples = np.asarray(mode['sample_indices'])
    points = surface['xyz'][samples]
    normals = np.broadcast_to(mode['normal'], points.shape)
    projection = project_oriented(surface, points, normals, view_index)
    visible = projection['supported'] & ~projection['conflict']
    draw = ImageDraw.Draw(image)
    for x, y in projection['pixels'][visible]:
        draw.ellipse((x-2.5, y-2.5, x+2.5, y+2.5), fill='#ffed64', outline='#172222')
    center = project_oriented(surface, np.asarray(mode['center'])[None],
                              np.asarray(mode['normal'])[None], view_index)
    if center['supported'][0] and not center['conflict'][0]:
        x, y = center['pixels'][0]
        draw.line((x-7, y, x+7, y), fill='white', width=2)
        draw.line((x, y-7, x, y+7), fill='white', width=2)
    return image, {'sample_count': len(samples), 'visible_samples': int(visible.sum()),
                   'unavailable_samples': int((~visible).sum())}


# BEGIN held-out transform and measured-runtime displays, separate from regions.
def _transform_panel(metrics):
    """Display every defined error without treating a local region as a pose."""
    if metrics is None:
        return ''
    reference = metrics['reference']
    primary = metrics['primary_estimate']
    records = ([{'label': '主估计（有限搜索）', **primary}] if primary is not None else
               [{'label': f"候选 h{record['hypothesis_id']}", **record}
                for record in metrics['displayed_candidates']])
    if not records:
        return '<p>没有可评价的变换估计；保留拒绝或缺失状态。</p>'
    def number(value, digits=2):
        return '不定义' if value is None else f'{value:.{digits}f}'
    body = []
    for record in records:
        values = (record['label'], number(record['estimated_scale'], 4),
            number(record['scale_signed_error_percent']), number(record['rotation_vs_polar_deg']),
            number(record['normal_error_deg']), number(record['local_mapping_rmse_mm']),
            number(record['context_mapping_rmse_mm']))
        body.append('<tr>'+''.join(f'<td>{escape(str(value))}</td>' for value in values)+'</tr>')
    truth = (f"真统一尺度 {reference['true_scale']:.4f}；"
             if reference['truth_is_sim3'] else 'GT 为非均匀形变，没有唯一真统一尺度，尺度误差不定义；')
    role = ('接受区域相关变换' if metrics['displayed_candidate_role'] == 'accepted_region_candidates'
            else '拒绝结果的诊断变换，不是成功结果')
    return ('<div class="transform"><b>尺度与完整旋转 · 仅事后 GT 评价</b>'
        f'<p>{escape(role)}。{escape(truth)}旋转角相对于真值极分解旋转，法向角单独评价。'
        '局部/上下文 RMS 比较固定源点经过估计与真值映射后的三维位置。</p>'
        '<div style="overflow:auto"><table><thead><tr><th>估计</th><th>尺度 s</th><th>尺度误差 %</th>'
        '<th>完整旋转 °</th><th>法向 °</th><th>局部 RMS mm</th><th>上下文 RMS mm</th>'
        '</tr></thead><tbody>'+''.join(body)+'</tbody></table></div>'
        '<p class="warn">采样半径固定为 3 cm，不随估计尺度改变。区域接受及有限候选唯一都不证明完整变换可识别。'
        '有限候选范围不是置信区间。</p>'
        '<details><summary>全部展示候选的误差与范围</summary><pre>'+
        escape(json.dumps(metrics, ensure_ascii=False, indent=2))+'</pre></details></div>')


def _runtime_panel(runtime):
    """Do not present absent historical stage timings as zero-cost stages."""
    if not runtime:
        return ''
    def seconds(value):
        return '未记录' if value is None else f'{value:.3f} s'
    detail = (f"该后端 {seconds(runtime.get('backend_seconds'))}；"
              f"任务合计 {seconds(runtime.get('task_seconds'))}（含共同准备与已运行配准算法）；"
              f"共同准备 {seconds(runtime.get('preparation_seconds'))}")
    stages = {'shared_preparation': runtime.get('preparation_stage_seconds', {}),
              'native_registration': runtime.get('native_registration_seconds'),
              'backend_inference': runtime.get('timing_seconds', {})}
    shared = ('<p>共享目标排名在此任务准备中仅计一次，RANSAC 与 TEASER++ 复用。</p>'
              if 'shared_target_ranking' in stages['shared_preparation'] else '')
    return ('<p>'+escape(detail)+'</p><details><summary>记录的阶段耗时（秒）</summary><pre>'+
            escape(json.dumps(stages, ensure_ascii=False, indent=2))+
            '</pre><p>这些记录有父子包含关系：task 包含共同准备和配准算法；backend 包含原生配准、推理及保存；'
            '推理 total 包含 proposals、partial_refinement、candidate_evaluation、decision_and_display；'
            'candidate_evaluation 又包含 query_sampling、independent_geometry、capacity_assignment 和 region_construction。'
            '父项与子项不得再次相加。</p>'+shared+'</details>')


def _runtime_comparison(runtime):
    """Show measured cached-run times, including all recorded task outcomes."""
    if not runtime:
        return ''
    rows = []
    for key, label in (('original_dual', '原双视角'), ('revision', '修订双视角')):
        record = runtime.get(key)
        if record is None:
            continue
        med = record['median_task_seconds']
        low, high = record['minimum_task_seconds'], record['maximum_task_seconds']
        extent = '未记录' if low is None else f'{low:.2f}–{high:.2f}'
        supervisor = record.get('supervisor')
        wall = '未记录' if supervisor is None else f"{supervisor['wall_seconds']:.2f}"
        values = (label, record['recorded_tasks'], f"{record['sum_task_seconds']:.2f}",
                  wall, '未记录' if med is None else f'{med:.2f}', extent)
        rows.append('<tr>'+''.join(f'<td>{escape(str(value))}</td>' for value in values)+'</tr>')
    return ('<h2>缓存预测耗时</h2><p>合计是逐任务实际记录时间之和，包含共同准备与已运行配准算法；'
        '监督器墙钟时间另含进程启动与退出，单列显示，不与任务时间相加。'
        '不包括 SAM/MARCO 特征提取或被复用表面缓存的原始重建成本。修订流程的完整目标排名每任务计算一次，由 RANSAC 与 TEASER++ 共享。'
        '阶段细节保留在每个任务中，未记录的阶段不记成零。</p>'
        '<table><thead><tr><th>方案</th><th>有耗时记录的任务</th><th>任务合计 s</th><th>监督器墙钟 s</th><th>每任务中位数 s</th><th>每任务范围 s</th>'
        '</tr></thead><tbody>'+''.join(rows)+'</tbody></table>')
# END held-out transform and measured-runtime helpers.


# BEGIN revised report preserves images first and explicitly distinguishes pose metrics.
def write_html(path, cards, summary, *, runtime=None, baseline_link=None):
    """Standalone local report with original image zoom and method filtering."""
    def image(filename, caption):
        return (f'<figure><img loading="lazy" src="{escape(filename)}" alt="{escape(caption)}" '
                f'tabindex="0"><figcaption>{escape(caption)}</figcaption></figure>')

    rows = []
    for item in summary:
        median = '—' if item['median_center_error_mm'] is None else f"{item['median_center_error_mm']:.2f}"
        rows.append('<tr>' + ''.join(f'<td>{escape(str(value))}</td>' for value in (
            item['label'], item['view'], f"{item['accepted']}/{item['total']}",
            item['correct_surface'], item['wrong_surface'], item['rejected'],
            item['missing'], item['failed'], item['incomplete'], median)) + '</tr>')
    sections = []
    for card in cards:
        methods = []
        for method in card['methods']:
            panels = []
            for panel in method['panels']:
                badge = panel['display_status']
                metric = panel.get('primary_metrics')
                detail = ''
                if metric:
                    detail = (f"中心误差 {metric['center_error_m']*1000:.2f} mm；"
                              f"正确支持面 {metric['top1_preferred_support_hit']}")
                elif panel.get('diagnostic_first_mode_metrics'):
                    metric = panel['diagnostic_first_mode_metrics']
                    detail = (f"仅诊断区域：中心误差 {metric['center_error_m']*1000:.2f} mm；"
                              f"正确支持面 {metric['preferred_support_hit']}；不可执行")
                reasons = '; '.join(panel.get('reasons', []))
                warning = '; '.join(panel.get('warnings', []))
                pictures = '<div class="image-pair">' + ''.join(
                    image(p['path'], p['caption']) for p in panel['images'][:2]) + '</div>'
                extra = panel['images'][2:]
                if extra:
                    pictures += '<details><summary>竞争区域与其他诊断区域（非主输出）</summary>'
                    pictures += ''.join(image(p['path'], p['caption']) for p in extra) + '</details>'
                panels.append(f'<div class="panel"><b>{escape(panel["view"])} · {escape(badge)}</b>'
                    f'<p>{escape(detail)}</p><p class="reason">{escape(reasons)}</p>'
                    f'<p class="warn">{escape(warning)}</p>{pictures}'
                    f'{_transform_panel(panel.get("transform_metrics"))}{_runtime_panel(panel.get("runtime"))}'
                    f'<details><summary>评分与搜索记录</summary><pre>{escape(json.dumps(panel.get("search", {}), ensure_ascii=False, indent=2))}</pre></details></div>')
            methods.append(f'<section class="method" data-scheme="{method["scheme"]}">'
                f'<h3>{escape(method["label"])}</h3><div class="panels">{"".join(panels)}</div></section>')
        source = (image(card['source'], '源示教：红色区域；黄点为区域查询，青点为实体上下文查询')
                  if card['source'] is not None else '<p>源图无法核验；保留失败记录。</p>')
        # Pair the same backend's original/revised A/B panels horizontally on
        # wide displays. Each figure keeps its own full-resolution zoom action.
        compared, index = [], 0
        while index < len(methods):
            current = card['methods'][index]
            following = card['methods'][index+1] if index+1 < len(methods) else None
            pair = (following is not None and current['scheme'] == 'dual_original'
                    and following['scheme'] == 'dual'
                    and current['panels'][0].get('backend') == following['panels'][0].get('backend'))
            if pair:
                compared.append('<div class="dual-comparison">'+methods[index]+methods[index+1]+'</div>')
                index += 2
            else:
                compared.append(methods[index])
                index += 1
        sections.append(f'<article data-task="{escape(card["id"])}"><h2>{escape(card["id"])}</h2>'
            f'<div class="source">{source}</div>'
            + ''.join(compared) + '</article>')
    html = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>双视角迁移 · 修订与冻结结果对照</title><style>
body{margin:0;background:#f3f5f8;color:#1d2b38;font:15px/1.5 system-ui,sans-serif}header,main{max-width:1480px;margin:auto;padding:24px}
header{background:#fff}h1{font-size:26px}h2{font-size:21px}h3{margin:0 0 12px}p{margin:7px 0}table{border-collapse:collapse;width:100%;font-size:13px}
td,th{border-bottom:1px solid #ccd4dc;padding:8px;text-align:left}article{margin:24px 0;padding:20px;background:#fff;border-radius:12px}
.method{margin:20px 0;padding:15px;border:1px solid #cad5df;border-radius:8px}.panels{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:14px}
.dual-comparison{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px;align-items:start}.dual-comparison .panels{display:block}.dual-comparison .method{min-width:0}.dual-comparison .image-pair{grid-template-columns:repeat(2,minmax(0,1fr))}.dual-comparison:has(.method[hidden]){grid-template-columns:1fr}@media(max-width:1000px){.dual-comparison{grid-template-columns:1fr}}
.panel{min-width:0}.image-pair{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:10px}.source{max-width:640px}figure{margin:8px 0}img{display:block;width:100%;height:auto;cursor:zoom-in;background:#eee}figcaption{font-size:13px;color:#445668;padding:4px}
.reason{color:#933c2c;overflow-wrap:anywhere}.warn{color:#856016;font-size:13px;overflow-wrap:anywhere}.transform{margin-top:14px;padding:10px;background:#f2f6fa}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}
.controls{position:sticky;top:0;z-index:2;padding:12px;background:#e7eef5;display:flex;gap:16px;flex-wrap:wrap}select,input,button{font:inherit;padding:6px}dialog{width:95vw;height:90vh;padding:0;border:0;background:#12202b;color:white}
dialog::backdrop{background:#000b}#toolbar{position:sticky;top:0;padding:8px;background:#203445;display:flex;gap:12px;align-items:center}#zoomArea{overflow:auto;height:calc(100% - 55px)}#zoomImage{width:auto;max-width:none;cursor:default}body:has(dialog[open]){overflow:hidden}[hidden]{display:none!important}
</style><header><h1>双视角迁移：修订结果与原始冻结结果</h1>
<p>同批源示教与真实仿真 RGB-D；旧方案、新方案均读取固定输出。两种原生配准后端分别比较，不按目标真值调参。</p>
<p>黄色圆点为真实采样位置，白色十字为区域中心。双视角区域颜色表示实际测量来源：绿为 A/B 测量配对，橙为 A，紫为 B；颜色不表示两路外观均有效。拒绝结果中的区域仅供诊断，不能视为已接受结果。</p>
<!-- BEGIN held-out GT legend: the surface contour and local reference are distinct. -->
<p><strong style="color:#b50085">洋红色 GT：轮廓为真值支持面的可见边界（完整支持面，不是局部迁移区域）；圆环与中心点为源示教区域经真值变换后的中心。</strong>GT 最后绘制并加黑白描边，不会被黄色采样点遮住。虚线圆环及 “GT hidden” 表示中心被遮挡，仅标注其投影；支持面不可见时不绘制虚构轮廓。各图说明保留可见状态，所有方案及诊断区域使用同一份 GT。</p>
<!-- END held-out GT legend. -->
<p>每张图可点击或按 Enter 放大，再用 50%–400% 缩放并滚动比较。修订双视角把局部几何准入与实体上下文诊断分开，以固定源查询分母下的已观测支持选择候选；未知保留为未知。有限搜索估计不代表连续全局最优，也不证明尺度和完整旋转已被观测确定。</p>
BASELINE_LINK
<!-- BEGIN GT provenance link; target truth is consumed only by evaluation. -->
<p><a href="evaluation.json">完整评价</a> · <a href="summary.json">汇总数据</a> · <a href="ground-truth.json">GT 来源与逐图覆盖记录</a> · <a href="../revision-plan.md">修订冻结计划</a></p>
<!-- END GT provenance link. -->
<div style="overflow:auto"><table><thead><tr><th>方案 / 配准后端</th><th>视角</th><th>接受/全部</th><th>正确面</th><th>错误面</th><th>拒绝</th><th>缺失</th><th>失败</th><th>搜索未完成</th><th>接受误差中位数 mm</th></tr></thead><tbody>SUMMARY</tbody></table></div>RUNTIME</header>
<main><div class="controls"><label>任务 <input id="taskFilter" placeholder="输入名称筛选"></label><label>显示 <select id="schemeFilter"><option value="all">全部方案</option><option value="dual">修订双视角</option><option value="dual_original">原双视角</option><option value="old">旧方案</option><option value="new">新方案</option></select></label><span id="count"></span></div>CARDS</main>
<dialog id="viewer"><div id="toolbar"><button id="close">关闭</button><label>缩放 <select id="zoom"><option>50</option><option selected>100</option><option>150</option><option>200</option><option>300</option><option>400</option></select>%</label><span id="caption"></span></div><div id="zoomArea"><img id="zoomImage" alt="放大图"></div></dialog>
<script>
const task=document.querySelector('#taskFilter'),scheme=document.querySelector('#schemeFilter');
function filter(){let n=0;document.querySelectorAll('article').forEach(a=>{a.hidden=!a.dataset.task.toLowerCase().includes(task.value.toLowerCase());if(!a.hidden)n++;});document.querySelectorAll('.method').forEach(m=>m.hidden=scheme.value!=='all'&&m.dataset.scheme!==scheme.value);document.querySelector('#count').textContent=n+' 个任务';}
task.addEventListener('input',filter);scheme.addEventListener('change',filter);filter();
const dialog=document.querySelector('#viewer'),large=document.querySelector('#zoomImage'),zoom=document.querySelector('#zoom');
function scale(){large.style.width=(large.naturalWidth*Number(zoom.value)/100)+'px';}
function openImage(im){large.onload=scale;large.src=im.src;document.querySelector('#caption').textContent=im.alt;dialog.showModal();}
document.querySelectorAll('figure img').forEach(im=>{im.onclick=()=>openImage(im);im.onkeydown=e=>{if(e.key==='Enter')openImage(im);};});zoom.onchange=scale;document.querySelector('#close').onclick=()=>dialog.close();
</script></html>'''.replace('SUMMARY', ''.join(rows)).replace('CARDS', ''.join(sections)).replace(
        'RUNTIME', _runtime_comparison(runtime)).replace('BASELINE_LINK', '' if baseline_link is None else
        f'<p><a href="{escape(baseline_link)}">打开原双视角完整网页</a>；本页原双视角图像直接引用原资产，没有复制或重写。</p>')
    Path(path).write_text(html)
# END revised report rendering.
