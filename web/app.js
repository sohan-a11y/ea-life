const $ = id => document.getElementById(id);
const colors = ["#6bbcff","#74e0bf","#b6a2ff","#f5c978","#ff9a78","#9de6ee"];

function drawWorld(s) {
  const c=$("world"),x=c.getContext("2d"),w=s.world||[],n=s.grid_size||128,cell=c.width/n;
  x.fillStyle="#08121a";x.fillRect(0,0,c.width,c.height);
  for(let y=0;y<w.length;y++)for(let z=0;z<(w[y]||[]).length;z++){
    const v=Number(w[y][z])||0;
    if(v>.015){x.fillStyle="rgba(230,188,108,"+Math.min(.95,v)+")";x.fillRect(z*cell,y*cell,Math.ceil(cell),Math.ceil(cell));}
  }
  (s.agents||[]).filter(a=>a.alive).forEach(a=>{
    const color=colors[(Number(a.generation)||0)%colors.length],px=(a.x+.5)*cell,py=(a.y+.5)*cell;
    x.beginPath();x.arc(px,py,Math.max(3,cell*1.9),0,Math.PI*2);x.fillStyle=color+"33";x.fill();
    x.beginPath();x.arc(px,py,Math.max(1.7,cell*.82),0,Math.PI*2);x.fillStyle=color;x.fill();
  });
}
function drawChart(events) {
  const c=$("chart"),x=c.getContext("2d"),W=c.width,H=c.height,p={l:30,r:12,t:12,b:24};
  x.clearRect(0,0,W,H);x.strokeStyle="#263644";
  for(let i=0;i<4;i++){const y=p.t+(H-p.t-p.b)*i/3;x.beginPath();x.moveTo(p.l,y);x.lineTo(W-p.r,y);x.stroke();}
  const rows=(events||[]).slice(-50);if(!rows.length)return;
  const max=Math.max(4,...rows.map(e=>Number(e.alive)||0),...rows.map(e=>Number(e.births)||0));
  const px=i=>p.l+i*(W-p.l-p.r)/Math.max(1,rows.length-1),py=v=>H-p.b-(v/max)*(H-p.t-p.b);
  [["alive","#6bbcff"],["births","#74e0bf"]].forEach(pair=>{
    x.beginPath();x.strokeStyle=pair[1];x.lineWidth=2;
    rows.forEach((e,i)=>{const v=Number(e[pair[0]])||0;if(!i)x.moveTo(px(i),py(v));else x.lineTo(px(i),py(v));});x.stroke();
  });
}
function esc(v){return String(v).replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));}
function render(s){
  const m=s.metrics||{};
  $("status").textContent=s.finished?"Experiment complete":s.paused?"Simulation paused":s.ok?"Simulation connected":"Waiting for simulation";
  $("alive").textContent=m.alive??"—";$("births").textContent=m.births??"—";$("generation").textContent=m.max_generation??"—";
  $("curiosity").textContent=Number(m.mean_curiosity||0).toFixed(3);$("curiositySmall").textContent=Number(m.mean_curiosity||0).toFixed(3);
  $("energy").textContent=m.mean_energy==null?"—":Number(m.mean_energy).toFixed(1);$("error").textContent=Number(m.mean_prediction_error||0).toFixed(3);
  $("topics").textContent=m.known_topics??0;$("tick").textContent=Number(s.tick||0).toLocaleString();$("tickLabel").textContent=String(s.tick||0).padStart(6,"0");
  $("curiosityGauge").style.width=Math.min(100,100*Number(m.mean_curiosity||0))+"%";
  $("pauseBtn").disabled=!s.ok||s.paused||s.finished;$("resumeBtn").disabled=!s.ok||!s.paused||s.finished;
  drawWorld(s);drawChart(s.events||[]);
  const events=(s.events||[]).slice(-12).reverse();
  $("events").innerHTML=events.length?events.map(e=>'<div class="event"><time>t+'+Number(e.tick||0).toLocaleString()+'</time><div>'+esc(e.message||("Population "+e.alive+" · births "+e.births))+'</div></div>').join(""):"No events recorded yet.";
  const agents=(s.agents||[]).slice().sort((a,b)=>Number(b.alive)-Number(a.alive)||Number(b.generation)-Number(a.generation)).slice(0,50);
  $("registryCount").textContent=(s.agents||[]).length+" RECORDS";
  $("registry").innerHTML=agents.length?agents.map(a=>{
    const e=Math.max(0,Math.min(100,Number(a.energy)||0)),color=colors[(Number(a.generation)||0)%colors.length];
    return '<tr><td class="agent"><span style="color:'+color+'">●</span> '+esc(a.id)+'</td><td>G'+Number(a.generation||0)+'</td><td><span class="energy"><i style="width:'+e+'%"></i></span>'+e.toFixed(1)+'</td><td>'+Number(a.last_curiosity||0).toFixed(3)+'</td><td>'+Number(a.age||0).toLocaleString()+'</td><td>'+Number(a.offspring||0)+'</td><td>'+(a.known_topics||[]).length+' topics / '+Number(a.facts||0)+' facts</td><td><span class="pill '+(a.alive?"alive":"dead")+'">'+(a.alive?"ALIVE":"DEAD")+'</span></td></tr>';
  }).join(""):'<tr><td colspan="8">Waiting for data…</td></tr>';
}
async function refresh(){try{const r=await fetch("/api/state",{cache:"no-store"});if(!r.ok)throw new Error("HTTP "+r.status);render(await r.json());}catch(e){$("status").textContent="Dashboard API unavailable";}}
async function control(action){try{const r=await fetch("/api/control",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({action:action})});const j=await r.json();if(!r.ok||!j.ok)throw new Error(j.error||"Control failed");await refresh();}catch(e){$("status").textContent="Control rejected";}}
$("pauseBtn").addEventListener("click",()=>control("pause"));$("resumeBtn").addEventListener("click",()=>control("resume"));
refresh();setInterval(refresh,1200);
