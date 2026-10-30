// Exercise every delivered nominal and actual display sample. Canvas stubs
// check finite coordinates and exactly one pose triad per current track.
// This is a serialization/interaction check, not a browser-rendering claim.
const fs=require('fs'),path=require('path'),vm=require('vm'),root=path.resolve(process.argv[2]);
const html=fs.readFileSync(path.join(root,'index.html'),'utf8'),registry=new Map();
let axes=0,coordinates=0;const images=[];
function finite(a){if(a.some(x=>!Number.isFinite(x)))throw Error('Nonfinite canvas coordinate');coordinates+=a.length;}
function context(){return{beginPath(){},stroke(){if(this.lineWidth===2.8)axes++;},fill(){},clearRect(){},drawImage(){},moveTo(...a){finite(a);},lineTo(...a){finite(a);},arc(...a){finite(a);}};}
class Element{constructor(){this.children=[];this.style={};this.classList={toggle(){}};this.value=0;this.context=context();}
 appendChild(x){this.children.push(x);}replaceChildren(){this.children=[];}getContext(){return this.context;}
 set innerHTML(s){for(const match of s.matchAll(/id="([^"]+)"/g))registry.set(match[1],new Element());if(s==='<th></th><td></td>')this.children=[new Element(),new Element()];}}
for(const match of html.matchAll(/id="([^"]+)"/g))registry.set(match[1],new Element());
class LocalImage{set src(s){if(!fs.existsSync(path.resolve(root,s)))throw Error('Missing image '+s);images.push(s);this.complete=true;}}
const box={document:{getElementById(id){if(!registry.has(id))throw Error('Unknown DOM id '+id);return registry.get(id);},createElement(){return new Element();}},Image:LocalImage};
vm.createContext(box);vm.runInContext(html.split('<script>')[1].split('</script>')[0],box);
const samples=vm.runInContext('D.times.length',box),physicalSamples=vm.runInContext('D.actual?D.actual.times.length:0',box);
function checkAxes(){const expected=vm.runInContext('((D.track_keys.length+(actualIndex===null?0:1))*D.panels.length+Object.keys(D.source.paths).length)*3',box);if(axes!==expected)throw Error('Current-pose axis count '+axes+' != '+expected);}
for(let i=0;i<samples;i++){axes=0;vm.runInContext(`slider.value=${i};slider.oninput();`,box);checkAxes();}
for(let i=0;i<physicalSamples;i++){axes=0;vm.runInContext(`actualSlider.value=${i};actualSlider.oninput();`,box);checkAxes();if(vm.runInContext('actualIndex',box)!==i)throw Error('Lost physical sample');}
vm.runInContext('for(const button of el("events").children)button.onclick();',box);
const links=[...html.matchAll(/href="([^"]+)"/g)].map(m=>m[1]);links.push(...vm.runInContext('D.links.map(row=>row.path)',box));
links.push(...vm.runInContext('D.videos.flatMap(row=>[row.path,row.capture_times])',box));
for(const link of links)if(link!=='javascript-checks.json'&&!fs.existsSync(path.resolve(root,link)))throw Error('Missing link '+link);
const report={status:'passed',samples,physical_samples:physicalSamples,images:images.length,links:links.length,
 canvas_coordinates_checked:coordinates,current_pose_only:true,actual_full_sample_access:true,
 scope:'All saved display timestamps and event buttons; finite canvas coordinates and local files; no browser rendering or algorithm acceptance check'};
fs.writeFileSync(path.join(root,'javascript-checks.json'),JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report));
