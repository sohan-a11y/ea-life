(() => {
"use strict";
const $=id=>document.getElementById(id);
const COLORS=["#70bbff","#72dfbc","#b7a3ff","#f3c879","#ff9c83","#8bdde7"];
let worldCanvas,popCanvas;
function esc(v){return String(v).replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]))}
function drawWorld(s){
 const c=worldCanvas,x=c.getContext("2d"),n=s.grid_size||96,r=s.resources||[],h=s.hazards||[],cell=c.width/n;
 x.fillStyle="#071018";x.fillRect(0,0,c.width,c.height);
 for(let y=0;y<n;y++)for(let z=0;z<n;z++){
  const rv=Number((r[y]||[])[z])||0,hv=Number((h[y]||[])[z])||0,px=z*cell,py=y*cell;
  if(rv>.012){x.fillStyle="rgba(243,200,121,"+Math.min(.9,rv)+")";x.fillRect(px,py,Math.ceil(cell),Math.ceil(cell))}
  if(hv>.07){x.fillStyle="rgba(241,107,114,"+Math.min(.7,hv*.7)+")";x.fillRect(px,py,Math.ceil(cell),Math.ceil(cell))}
 }
 (s.beacons||[]).forEach(b=>{x.strokeStyle="#a9ffe5";x.lineWidth=1.4;x.beginPath();x.rect((b.x+.22)*cell,(b.y+.22)*cell,cell*.56,cell*.56);x.stroke()});
 (s.agents||[]).filter(a=>a.alive).forEach(a=>{
  const col=COLORS[(Number(a.generation)||0)%COLORS.length],px=(a.x+.5)*cell,py=(a.y+.5)*cell;
  x.beginPath();x.arc(px,py,Math.max(3,cell*1.65),0,Math.PI*2);x.fillStyle=col+"35";x.fill();
  x.beginPath();x.arc(px,py,Math.max(1.8,cell*.72),0,Math.PI*2);x.fillStyle=col;x.fill();
 });
 $("worldCover").style.display=(s.resources||[]).length?"none":"grid";
}
function drawChart(events){
 const c=popCanvas,x=c.getContext("2d"),w=c.width,h=c.height,p={l:34,r:12,t:12,b:23};
 x.clearRect(0,0,w,h);x.strokeStyle="#263644";x.lineWidth=1;
 for(let i=0;i<4;i++){const y=p.t+(h-p.t-p.b)*i/3;x.beginPath();x.moveTo(p.l,y);x.lineTo(w-p.r,y);x.stroke()}
 const rows=(events||[]).filter(e=>e.type==="metrics").slice(-60);
 if(!rows.length)return;
 const max=Math.max(4,...rows.map(e=>Number(e.detail&&e.detail.alive)||0),...rows.map(e=>Number(e.detail&&e.detail.births)||0));
 const px=i=>p.l+i*(w-p.l-p.r)/Math.max(1,rows.length-1),py=v=>h-p.b-v/max*(h-p.t-p.b);
 [["alive","#70bbff"],["births","#72dfbc"]].forEach(pair=>{
  x.beginPath();x.strokeStyle=pair[1];x.lineWidth=2;
  rows.forEach((e,i)=>{const v=Number((e.detail||{})[pair[0]])||0;if(i===0)x.moveTo(px(i),py(v));else x.lineTo(px(i),py(v))});x.stroke();
 });
 x.fillStyle="#7c91a2";x.font="10px system-ui";x.fillText("0",10,h-p.b+3);x.fillText(String(max),5,p.t+3);
}
function renderEvents(events){
 const rows=(events||[]).slice(-14).reverse();
 $("events").innerHTML=rows.length?rows.map(e=>{
  const kind=String(e.type||"").toLowerCase(),cls=kind==="birth"?"birth":kind==="research"?"research":"";
  return '<div class="event '+cls+'"><time>t+'+Number(e.tick||0).toLocaleString()+'</time><div><strong>'+esc(e.message||kind)+'</strong></div></div>';
 }).join(""):'<p class="muted">The event stream is waiting for simulation activity.</p>';
}
function renderRegistry(agents){
 const rows=(agents||[]).slice().sort((a,b)=>Number(b.alive)-Number(a.alive)||Number(b.generation)-Number(a.generation)).slice(0,80);
 $("registryCount").textContent=(agents||[]).length+" ORGANISMS";
 $("registry").innerHTML=rows.length?rows.map(a=>{
  const col=COLORS[(Number(a.generation)||0)%COLORS.length],energy=Math.max(0,Math.min(100,Number(a.energy)||0));
  return '<tr><td class="agent-name"><span class="agent-dot" style="color:'+col+'">●</span>'+esc(a.id)+'</td><td>'+esc(a.role)+'</td><td>G'+Number(a.generation||0)+'</td><td><span class="energybar"><i style="width:'+energy+'%"></i></span>'+energy.toFixed(1)+'</td><td>'+Number(a.curiosity||0).toFixed(3)+'</td><td>'+Number(a.age||0).toLocaleString()+'</td><td>'+Number(a.offspring||0)+'</td><td>'+Number(a.goals_reached||0)+'</td><td>'+Number(a.facts||0)+' snippets / '+(a.known_topics||[]).length+' topics</td><td><span class="pill '+(a.alive?"alive-pill":"dead-pill")+'">'+(a.alive?"ALIVE":"INACTIVE")+'</span></td></tr>';
 }).join(""):'<tr><td colspan="10" class="muted">Waiting for organisms…</td></tr>';
}
function render(s){
 const m=s.metrics||{},agents=s.agents||[],alive=Number(m.alive)||0;
 $("status").textContent=s.finished?"Experiment stopped":s.paused?"Simulation paused":s.ok?"Live experiment connected":"Waiting for simulation";
 $("dot").style.background=s.paused?"var(--amber)":"var(--mint)";
 $("alive").textContent=alive;$("births").textContent=m.births??0;$("diversity").textContent=Number(m.genomic_diversity||0).toFixed(3);
 $("curiosity").textContent=Number(m.mean_curiosity||0).toFixed(3);$("curiosity2").textContent=Number(m.mean_curiosity||0).toFixed(3);
 $("curiosityGauge").style.width=Math.min(100,100*Number(m.mean_curiosity||0))+"%";
 $("learning").textContent=Number(m.mean_learning_progress||0).toFixed(3);
 $("learningGauge").style.width=Math.min(100,100*Number(m.mean_learning_progress||0)*4)+"%";
 $("coverage").textContent=(100*Number(m.resource_coverage||0)).toFixed(1)+"%";
 $("topics").textContent=m.research_topics_known??0;$("energy").textContent=Number(m.mean_energy||0).toFixed(1);
 $("error").textContent=Number(m.mean_prediction_error||0).toFixed(3);$("goals").textContent=m.goals_reached??0;
 $("tick").textContent=Number(s.tick||0).toLocaleString();$("tickHero").textContent="TICK "+String(s.tick||0).padStart(6,"0");
 $("gridLabel").textContent=(s.grid_size||96)+" × "+(s.grid_size||96);$("scenario").textContent=s.scenario||"mixed";
 $("worldSize").textContent=(s.grid_size||96)+" × "+(s.grid_size||96);$("totalCreated").textContent=m.total_created??agents.length;
 $("populationText").textContent=alive+" alive · "+(m.deaths||0)+" inactive";
 $("aliveBar").style.width=Math.min(100,alive/Math.max(1,agents.length)*100)+"%";
 $("birthBar").style.width=Math.min(100,Number(m.births||0)/Math.max(1,Number(m.total_created||1))*100)+"%";
 $("diversityBar").style.width=Math.min(100,100*Number(m.genomic_diversity||0))+"%";
 $("curiosityBar").style.width=Math.min(100,100*Number(m.mean_curiosity||0))+"%";
 $("pause").disabled=!s.ok||s.paused||s.finished;$("resume").disabled=!s.ok||!s.paused||s.finished;$("reset").disabled=!s.ok;
 drawWorld(s);drawChart(s.events||[]);renderEvents(s.events||[]);renderRegistry(agents);
}
async function refresh(){
 try{const r=await fetch("/api/state",{cache:"no-store"});if(!r.ok)throw new Error("HTTP "+r.status);render(await r.json())}
 catch(e){$("status").textContent="Dashboard API unavailable"}
}
async function control(action){
 if(action==="reset"&&!window.confirm("Reset the experiment and discard the current checkpoint?"))return;
 try{
  const r=await fetch("/api/control",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({action:action})});
  const data=await r.json();if(!r.ok||!data.ok)throw new Error(data.error||"request rejected");await refresh();
 }catch(e){$("status").textContent="Control rejected: "+e.message}
}
$("pause").addEventListener("click",()=>control("pause"));
$("resume").addEventListener("click",()=>control("resume"));
$("reset").addEventListener("click",()=>control("reset"));
worldCanvas=$("world");popCanvas=$("populationChart");refresh();setInterval(refresh,1400);
})();