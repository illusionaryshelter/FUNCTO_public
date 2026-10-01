/* BEGIN local, read-only rendering of frozen independent failure states.
 * Scene queries use the exported expanded envelope; native queries use actual
 * finger positions. No path interpolation or browser-side collision solver.
 */
'use strict';
const payload = JSON.parse(document.getElementById('payload').textContent);
const el = id => document.getElementById(id);
const safe = value => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const table = (headers, rows) => '<table><thead><tr>'+headers.map(x=>'<th>'+safe(x)+'</th>').join('')+'</tr></thead><tbody>'+rows.map(row=>'<tr>'+row.map(x=>'<td>'+safe(x)+'</td>').join('')+'</tr>').join('')+'</tbody></table>';
const mm = value => (value*1000).toFixed(4);
const xyz = points => ({x:points.map(p=>p[0]), y:points.map(p=>p[1]), z:points.map(p=>p[2])});
const indices = faces => ({i:faces.map(f=>f[0]), j:faces.map(f=>f[1]), k:faces.map(f=>f[2])});
const transform = (point, pose) => [0,1,2].map(i=>pose[i][3]+point.reduce((sum,v,j)=>sum+pose[i][j]*v,0));
let focus = 'all';
let cameraRevision = 0;

function selection() {
    const c = Number(el('candidate').value), s = Number(el('state').value);
    return {report:payload.reports[c], data:payload.data[c], state:payload.reports[c].states[s], index:s};
}

function stateOptions() {
    const report = payload.reports[Number(el('candidate').value)];
    el('state').innerHTML = report.states.map((state,index)=>'<option value="'+index+'">'+safe(state.kind==='entrance'?'入口 q0（实测）':
        (state.mode==='native_self_collision'?'原生自碰撞 IK':'仅位姿 + 限位 IK')+' · 帧 '+state.source_frame+' · 解 '+state.rank+
        ' · 原生 '+(state.native_success?'成功':'失败'))+'</option>').join('');
    el('state').value = '0';
    el('link').innerHTML = '<option value="all">全部</option>'+[...new Set(payload.data[Number(el('candidate').value)].sphere_links)].map(name=>'<option>'+safe(name)+'</option>').join('');
    render();
}

function sphereMesh(balls, selected, values, links, metric) {
    const vertices=[], faces=[], intensity=[], labels=[];
    for (const id of selected) {
        const offset=vertices.length, ball=balls[id];
        vertices.push(...payload.unitSphere.vertices.map(v=>v.map((x,j)=>ball[j]+ball[3]*x)));
        faces.push(...payload.unitSphere.faces.map(f=>f.map(x=>x+offset)));
        const label='球 '+id+' · '+links[id]+'<br>半径 '+mm(ball[3])+' mm'+
            (metric==='actual'?'':'<br>当前显示值 '+mm(values[id])+' mm');
        for(let i=0;i<payload.unitSphere.vertices.length;i++) {intensity.push(values[id]*1000); labels.push(label);}
    }
    return {type:'mesh3d',...xyz(vertices),...indices(faces),intensity,text:labels,
        hovertemplate:'%{text}<extra></extra>',opacity:.6,flatshading:false,
        colorscale:metric==='actual'?[[0,'#48a27b'],[1,'#48a27b']]:'YlOrRd',
        cmin:0,cmax:Math.max(.001,...intensity),showscale:metric!=='actual',
        colorbar:{title:{text:'逐球值 / mm'},len:.65},name:'保存的碰撞球'};
}

function axisTraces(pose, prefix, dashed) {
    return [0,1,2].map((axis)=>{
        const begin=pose.slice(0,3).map(row=>row[3]), end=begin.map((v,i)=>v+.07*pose[i][axis]);
        return {type:'scatter3d',mode:'lines',...xyz([begin,end]),name:prefix+' '+'XYZ'[axis],
            line:{color:['#dc5656','#43a26e','#4d79bd'][axis],width:dashed?3:6,dash:dashed?'dash':'solid'},
            hovertemplate:safe(prefix)+'<extra></extra>',showlegend:false};
    });
}

function render() {
    try {
        const {report,data,state,index}=selection(), metric=el('metric').value, chosen=el('link').value;
        const actual=metric==='actual'||metric==='native_overlap';
        const balls=data[actual?'actual_spheres':'scene_spheres'][index];
        const values=metric==='actual'||metric==='none'?balls.map(()=>0):data[metric][index];
        const selected=balls.map((_,i)=>i).filter(i=>(chosen==='all'||data.sphere_links[i]===chosen)&&
            metric!=='none'&&(metric==='actual'||values[i]>0));
        const traces=[];
        for(const cloud of payload.clouds) if(el(cloud.category).checked) {
            traces.push({type:'scatter3d',mode:'markers',...xyz(cloud.points),name:cloud.name,
                marker:{color:cloud.color,size:cloud.category==='background'?1.7:2.4,opacity:.6},hoverinfo:'skip'});
        }
        if(el('solids').checked) payload.meshes.forEach((mesh,i)=>{
            traces.push({type:'mesh3d',...xyz(mesh.vertices.map(p=>transform(p,state.mesh_poses[i]))),...indices(mesh.faces),
                color:'#8299b3',opacity:.38,flatshading:true,name:mesh.link,showlegend:false,
                hovertemplate:safe(mesh.link)+' · 原生凸碰撞实体<extra></extra>'});
        });
        if(selected.length) traces.push(sphereMesh(balls,selected,values,data.sphere_links,metric));
        if(metric==='gripper_overlap') {
            const lines=[];
            for(const pair of state.gripper_pairs.top) {
                if(pair.indices.some(i=>selected.includes(i))) lines.push(balls[pair.indices[0]],balls[pair.indices[1]],[null,null,null]);
            }
            traces.push({type:'scatter3d',mode:'lines',...xyz(lines),line:{color:'#b0225b',width:5},name:'最大重叠球对',hoverinfo:'skip'});
        }
        traces.push(...axisTraces(state.tool_pose,'当前 FK TCP',false));
        if(state.kind==='ik') traces.push(...axisTraces(state.target,'目标 TCP',true));
        const points=[state.tool_pose.slice(0,3).map(r=>r[3])];
        const names=['当前 TCP'];
        if(state.kind==='ik') {points.push(state.target.slice(0,3).map(r=>r[3]));names.push('目标 TCP');}
        traces.push({type:'scatter3d',mode:'markers+text',...xyz(points),text:names,textposition:'top center',
            marker:{size:5,color:['#4b4295','#da7731']},name:'TCP',hovertemplate:'%{text}<extra></extra>'});
        let center,half;
        if(focus==='base') {center=payload.base.map((v,i)=>v+(i===2?.1:0));half=.23;}
        else if(focus==='hand') {center=points[0];half=.15;}
        else {const all=[...data.actual_spheres[index].map(b=>b.slice(0,3)),...points];
            const low=[0,1,2].map(i=>Math.min(...all.map(p=>p[i]))-.15);
            const high=[0,1,2].map(i=>Math.max(...all.map(p=>p[i]))+.15);
            center=low.map((v,i)=>(v+high[i])/2);half=Math.max(...low.map((v,i)=>high[i]-v))/2;}
        const scene={aspectmode:'cube',uirevision:String(cameraRevision),camera:{eye:{x:1.55,y:-1.65,z:1.0}},
            xaxis:{title:{text:'世界 X / m'},range:[center[0]-half,center[0]+half]},
            yaxis:{title:{text:'世界 Y / m'},range:[center[1]-half,center[1]+half]},
            zaxis:{title:{text:'世界 Z / m'},range:[center[2]-half,center[2]+half]}};
        Promise.resolve(Plotly.react('scene',traces,{scene,margin:{l:0,r:0,t:5,b:0},legend:{orientation:'h',y:0},
            uirevision:String(cameraRevision),paper_bgcolor:'#fff'}, {responsive:true,displaylogo:false})).catch(showError);
        el('status').innerHTML='<b>'+safe(report.candidate_id)+'</b> · '+(state.kind==='entrance'?'实测入口 q0':
            '原生 IK：<b class="'+(state.native_success?'good':'bad')+'">'+(state.native_success?'成功':'失败')+'</b> · 位姿 / 限位复核：'+
            (state.pose_limits_passed?'通过':'失败')+' · 位置误差 '+mm(state.position_error_m)+' mm · 姿态误差 '+
            (state.rotation_error_rad*180/Math.PI).toFixed(4)+'°')+' · 完整场景：<b class="'+
            (state.postcheck.all_original_collision_terms_zero?'good':'bad')+'">'+(state.postcheck.all_original_collision_terms_zero?'通过':'失败')+'</b>';
        el('residuals').innerHTML=table(['原始项','原值（未缩放）'],Object.entries(state.postcheck.residuals).map(([k,v])=>[k,v.toPrecision(10)]));
        const explanations={background:'背景：显示正查询值的场景球；值为原背景查询输出，不对应某个显示点的精确穿透。',
            unknown:'未知空间：球半径与已知空间保守净空不足形成的正项，不能当作已观测物体碰撞。',
            gripper_overlap:'夹爪：每球颜色为其参与球对的最大正重叠。明细列最大球对；原始残差是全部球对正重叠之和。',
            native_overlap:'原生自碰撞：使用实测指位球与原生有效球对计算的几何正重叠；原生加权代价仍以原报告为准。',
            actual:'实测指位的全部原生球；未把全行程场景包络的数值混用到这些球上。',none:'碰撞球已隐藏。'};
        el('explanation').textContent=explanations[metric];
        el('sphereSummary').textContent='当前筛选显示 '+selected.length+' / '+balls.length+' 个球'+(metric==='gripper_overlap'?
            '；全部正重叠球对 '+state.gripper_pairs.positive_count+'；最大单对 '+mm(state.gripper_pairs.maximum_overlap_m)+' mm':'');
        if(metric==='gripper_overlap') {
            el('spheres').innerHTML=table(['球对','连杆','正重叠 / mm'],state.gripper_pairs.top.filter(pair=>pair.indices.some(i=>selected.includes(i)))
                .map(pair=>[pair.indices.join(' / '),pair.links.join(' / '),mm(pair.overlap_m)]));
        } else {
            el('spheres').innerHTML=table(['球编号','连杆','半径 / mm','当前项 / mm'],selected.sort((a,b)=>values[b]-values[a]).slice(0,16)
                .map(i=>[i,data.sphere_links[i],mm(balls[i][3]),metric==='actual'?'—':mm(values[i])]));
        }
        el('error').textContent='';
    } catch(error) {showError(error);}
}

function showError(error) {el('error').textContent='页面绘制失败：'+String(error)+'。可打开下方静态总览与原始证据。';}

el('candidate').innerHTML=payload.reports.map((report,i)=>'<option value="'+i+'">'+safe(report.candidate_id)+'</option>').join('');
el('candidate').value='0';
el('candidate').addEventListener('change',stateOptions);
['state','metric','link','solids','background','owner','robot'].forEach(id=>el(id).addEventListener('change',render));
document.querySelectorAll('[data-view]').forEach(button=>button.addEventListener('click',()=>{focus=button.dataset.view;cameraRevision++;render();}));
el('photos').innerHTML=payload.images.map(view=>'<figure><a href="'+safe(view.rgb)+'"><img src="'+safe(view.masks)+'" alt="'+safe(view.name)+' 掩码"></a><figcaption>'+safe(view.name)+'（点击查看原始 RGB）</figcaption></figure>').join('');
const summary=[];
for(const report of payload.reports) {
    const groups=new Map();
    for(const state of report.states) if(state.kind==='ik') {
        const key=state.mode+' / '+state.source_frame;
        if(!groups.has(key)) groups.set(key,[]);groups.get(key).push(state);
    }
    for(const [key,states] of groups) summary.push([report.candidate_id,key,
        states.filter(s=>s.native_success).length+' / '+states.length,
        states.filter(s=>s.pose_limits_passed).length+' / '+states.length,
        states.filter(s=>s.postcheck.all_original_collision_terms_zero).length+' / '+states.length]);
}
el('ikSummary').innerHTML=table(['候选','模式 / 源帧','原生成功','位姿 + 限位','完整场景通过'],summary);
stateOptions();
/* END local, read-only rendering of frozen independent failure states. */
