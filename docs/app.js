(() => {
"use strict";
const $ = id => document.getElementById(id);
const N=128, ACTIONS=[[-1,-1],[0,-1],[1,-1],[-1,0],[0,0],[1,0],[-1,1],[0,1],[1,1]];
const COLORS=["#6bbcff","#74e0bf","#b6a2ff","#f5c978","#ff9a78","#9de6ee"];
const TOPICS=["curiosity and exploration","prediction and learning","artificial life","evolution and mutation","resource navigation","energy budgets","cellular automata"];
let world, agents, tick, paused, timer, history, events, born, nextId, worldCanvas, chartCanvas;
function rand(lo,hi){return lo+Math.random()*(hi-lo)}
function clamp(x,lo=0,hi=1){return Math.max(lo,Math.min(hi,x))}
function sigmoid(x){return 1/(1+Math.exp(-clamp(x,-30,30)))}
function genome(){return {food:rand(1.2,2.2),novelty:rand(.4,1.5),uncertainty:rand(.4,1.4),learning:rand(.5,1.4),risk:rand(1,2.4),cost:rand(.3,1),bias:rand(-1,-.15),threshold:rand(56,74),mutation:rand(.07,.16)}}
function mutate(g){const c={};Object.keys(g).forEach(k=>c[k]=g[k]+(Math.random()*2-1)*g.mutation);const b={food:[.1,4],novelty:[0,3],uncertainty:[0,3],learning:[0,3],risk:[.1,4],cost:[0,2],bias:[-2,1],threshold:[45,90],mutation:[.03,.25]};Object.keys(b).forEach(k=>c[k]=clamp(c[k],b[k][0],b[k][1]));return c}
function makeAgent(id,x,y,g,parent,generation,energy){return {id,x,y,genome:g||genome(),parent:parent||null,generation:generation||0,energy:energy===undefined?58:energy,age:0,offspring:0,alive:true,visits:new Map(),q:Array(9).fill(0),error:.5,progress:0,curiosity:0,topics:[],lastTopic:-999,lastBirth:-999,totalError:0,errorCount:0,reward:0}}
function seed(){
  world=new Float32Array(N*N);
  for(let i=0;i<world.length;i++)world[i]=rand(0,.1);
  [[22,24],[96,22],[24,98],[100,98],[64,64],[64,30],[32,64]].forEach(([cx,cy])=>{
    for(let y=Math.max(0,cy-8);y<Math.min(N,cy+9);y++)for(let x=Math.max(0,cx-8);x<Math.min(N,cx+9);x++){
      const d=Math.abs(x-cx)+Math.abs(y-cy),idx=y*N+x;
      world[idx]=Math.max(world[idx],Math.max(0,.95-.045*d));
    }
  });
  agents=[makeAgent("org-01",12,12),makeAgent("org-02",115,12),makeAgent("org-03",12,115),makeAgent("org-04",115,115)];
  tick=0;paused=false;history=[];events=[];born=0;nextId=5;
  addEvent("Experiment initialized with four organisms.","system");
  addEvent("Organisms are active. Resources and behavior update locally in this tab.","system");
  $("pauseBtn").textContent="Ⅱ Pause";$("statusText").textContent="Simulation running";$("statusDot").style.background="var(--mint)";
  render();if(timer)clearInterval(timer);timer=setInterval(step,95);
}
function addEvent(msg,type){events.push({tick:tick,message:msg,type:type||"system"});if(events.length>30)events.shift()}
function curiosity(a,novelty,error,risk,cost){const g=a.genome;return sigmoid(g.novelty*novelty+g.uncertainty*error+g.learning*a.progress-g.risk*risk-g.cost*cost+g.bias)}
function step(){
  if(paused)return;
  const living=agents.filter(a=>a.alive);
  if(!living.length){paused=true;$("statusText").textContent="All organisms inactive · reset to restart";$("pauseBtn").textContent="▶ Resume";render();return}
  const occupied=new Set(living.map(a=>a.y*N+a.x));
  living.forEach(a=>{
    if(!a.alive)return;
    const pos=a.y*N+a.x,visits=a.visits,seen=visits.get(pos)||0,novelty=1/Math.sqrt(1+seen);visits.set(pos,seen+1);
    if(visits.size>520){const first=visits.keys().next().value;visits.delete(first)}
    let cand=[];
    ACTIONS.forEach((d,i)=>{
      const x=a.x+d[0],y=a.y+d[1];if(x<0||y<0||x>=N||y>=N)return;
      const idx=y*N+x,local=1/Math.sqrt(1+(visits.get(idx)||0)),crowd=(occupied.has(idx)&&idx!==pos)?0.65:0;
      cand.push({i,x,y,idx,score:a.genome.food*world[idx]+a.genome.novelty*local+a.q[i]*.06-crowd+rand(-.035,.035)});
    });
    cand.sort((a,b)=>b.score-a.score);const options=cand.slice(0,Math.min(4,cand.length));const pick=options[Math.floor(Math.random()*options.length)];
    if(pick.idx!==pos){occupied.delete(pos);if(!occupied.has(pick.idx)){a.x=pick.x;a.y=pick.y}occupied.add(a.y*N+a.x)}
    const idx=a.y*N+a.x,res=world[idx],moving=pick.x!==pos%N||pick.y!==Math.floor(pos/N),reward=res*5-(moving ? .5 : .18);
    const oldQ=a.q[pick.i],error=reward-oldQ;a.q[pick.i]+=.12*error;
    const oldE=a.error;a.error=.9*oldE+.1*Math.min(1,Math.abs(error)/5);a.progress=clamp(Math.max(0,oldE-a.error)*4);
    a.totalError+=Math.min(1,Math.abs(error)/5);a.errorCount++;
    const edge=Math.min(a.x,a.y,N-1-a.x,N-1-a.y),risk=edge<2 ? .35 : 0,cost=.2+(moving ? .25 : 0);
    a.curiosity=curiosity(a,novelty,Math.min(1,Math.abs(error)/5),risk,cost);
    a.energy=clamp(a.energy+res*3-.42-(moving ? .16 : 0),0,100);world[idx]=Math.max(0,res-.055);a.age++;a.reward+=reward;
    // Built-in topic labels are local stimuli, not web research.
    if(a.curiosity>.84&&tick-a.lastTopic>115&&a.topics.length<TOPICS.length){
      const choices=TOPICS.filter(t=>!a.topics.includes(t));
      if(choices.length){const topic=choices[Math.floor(Math.random()*choices.length)];a.topics.push(topic);a.lastTopic=tick;addEvent(a.id+" explored the local concept “"+topic+"”.","learn")}
    }
    if(a.energy<=.01||a.age>3300){a.alive=false;a.visits.clear();a.q.fill(0);addEvent(a.id+" became inactive at age "+a.age+".","death")}
  });
  if(tick%8===0){for(let i=0;i<world.length;i++)if(world[i]<.1)world[i]=Math.min(.1,world[i]+.002)}
  const livingNow=agents.filter(a=>a.alive),spots=new Set(livingNow.map(a=>a.y*N+a.x));
  if(livingNow.length<16&&agents.length<300){
    for(const p of livingNow.slice()){
      if(agents.filter(a=>a.alive).length>=16||agents.length>=300)break;
      if(p.energy<p.genome.threshold||tick-p.lastBirth<80)continue;
      const choices=ACTIONS.map(d=>[p.x+d[0],p.y+d[1]]).filter(([x,y])=>x>=0&&y>=0&&x<N&&y<N&&!spots.has(y*N+x));
      if(!choices.length)continue;
      const [x,y]=choices[Math.floor(Math.random()*choices.length)],child=makeAgent("org-"+String(nextId++).padStart(3,"0"),x,y,mutate(p.genome),p.id,p.generation+1,30);
      agents.push(child);p.offspring++;p.lastBirth=tick;p.energy-=20;spots.add(y*N+x);born++;addEvent("Birth: "+p.id+" → "+child.id+" (generation "+child.generation+").","birth");
    }
  }
  tick++;
  if(tick%6===0){
    const a=agents.filter(z=>z.alive);
    history.push({tick:tick,alive:a.length,births:born});if(history.length>60)history.shift();
    if(tick%60===0)addEvent("Tick "+tick+": "+a.length+" alive, "+born+" births so far.","system");
  }
  render();
}
function drawWorld(){
  const c=worldCanvas,x=c.getContext("2d"),cell=c.width/N;x.fillStyle="#081219";x.fillRect(0,0,c.width,c.height);
  for(let y=0;y<N;y++)for(let z=0;z<N;z++){const v=world[y*N+z];if(v>.012){x.fillStyle="rgba(230,188,108,"+Math.min(.92,v)+")";x.fillRect(z*cell,y*cell,Math.ceil(cell),Math.ceil(cell))}}
  agents.filter(a=>a.alive).forEach(a=>{const col=COLORS[a.generation%COLORS.length],px=(a.x+.5)*cell,py=(a.y+.5)*cell;x.beginPath();x.arc(px,py,Math.max(3,cell*1.9),0,Math.PI*2);x.fillStyle=col+"35";x.fill();x.beginPath();x.arc(px,py,Math.max(1.7,cell*.82),0,Math.PI*2);x.fillStyle=col;x.fill()});
}
function drawChart(){
  const c=chartCanvas,x=c.getContext("2d"),w=c.width,h=c.height,p={l:32,r:12,t:13,b:24};x.clearRect(0,0,w,h);x.strokeStyle="#263745";x.lineWidth=1;
  for(let i=0;i<4;i++){const y=p.t+(h-p.t-p.b)*i/3;x.beginPath();x.moveTo(p.l,y);x.lineTo(w-p.r,y);x.stroke()}
  if(!history.length)return;
  const max=Math.max(4,...history.map(v=>v.alive),...history.map(v=>v.births));
  const px=i=>p.l+i*(w-p.l-p.r)/Math.max(1,history.length-1),py=v=>h-p.b-v/max*(h-p.t-p.b);
  [[ "alive","#6bbcff"],["births","#74e0bf"]].forEach(([key,color])=>{x.beginPath();x.strokeStyle=color;x.lineWidth=2;history.forEach((v,i)=>{if(!i)x.moveTo(px(i),py(v[key]));else x.lineTo(px(i),py(v[key]))});x.stroke()});
  x.fillStyle="#758b9c";x.font="10px system-ui";x.fillText("0",9,h-p.b+3);x.fillText(String(max),4,p.t+3);
}
function render(){
  const living=agents.filter(a=>a.alive),mCur=living.reduce((s,a)=>s+a.curiosity,0)/Math.max(1,living.length),mEnergy=living.reduce((s,a)=>s+a.energy,0)/Math.max(1,living.length),error=agents.reduce((s,a)=>s+a.totalError,0)/Math.max(1,agents.reduce((s,a)=>s+a.errorCount,0)),maxGen=Math.max(0,...agents.map(a=>a.generation)),topics=living.reduce((s,a)=>s+a.topics.length,0),learn=living.reduce((s,a)=>s+a.progress,0)/Math.max(1,living.length),coverage=world.reduce((s,v)=>s+(v>.15?1:0),0)/world.length;
  $("alive").textContent=living.length;$("births").textContent=born;$("generation").textContent=maxGen;$("curiosity").textContent=mCur.toFixed(3);$("curiositySmall").textContent=mCur.toFixed(3);$("energy").textContent=mEnergy.toFixed(1);$("error").textContent=error.toFixed(3);$("topics").textContent=topics;$("learning").textContent=learn.toFixed(3);$("resourceCoverage").textContent=(coverage*100).toFixed(1)+"%";$("tick").textContent=tick.toLocaleString();$("tickHero").textContent="TICK "+String(tick).padStart(6,"0");$("registryCount").textContent=agents.length+" ORGANISMS";
  $("aliveBar").style.width=(living.length/16*100)+"%";$("birthBar").style.width=Math.min(100,born/50*100)+"%";$("genBar").style.width=Math.min(100,maxGen/15*100)+"%";$("curiosityBar").style.width=(mCur*100)+"%";$("curiosityGauge").style.width=(mCur*100)+"%";
  $("pauseBtn").textContent=paused?"▶ Resume":"Ⅱ Pause";$("statusText").textContent=paused?"Simulation paused":"Simulation running";$("statusDot").style.background=paused?"var(--amber)":"var(--mint)";
  drawWorld();drawChart();
  $("events").innerHTML=events.slice(-12).reverse().map(e=>'<div class="event '+e.type+'"><time>t+'+e.tick.toLocaleString()+'</time><div>'+escapeHtml(e.message)+'</div></div>').join("");
  $("registry").innerHTML=agents.slice().sort((a,b)=>Number(b.alive)-Number(a.alive)||b.generation-a.generation).slice(0,40).map(a=>'<tr><td class="agent-name"><span style="color:'+COLORS[a.generation%COLORS.length]+'">●</span> '+a.id+'</td><td>G'+a.generation+'</td><td><span class="energybar"><i style="width:'+a.energy+'%"></i></span>'+a.energy.toFixed(1)+'</td><td>'+a.curiosity.toFixed(3)+'</td><td>'+a.age.toLocaleString()+'</td><td>'+a.offspring+'</td><td>'+a.topics.length+'</td><td><span class="pill '+(a.alive?"alive-pill":"dead-pill")+'">'+(a.alive?"ALIVE":"INACTIVE")+'</span></td></tr>').join("");
}
function escapeHtml(s){return String(s).replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]))}
$("pauseBtn").addEventListener("click",()=>{paused=!paused;if(!paused&&agents.every(a=>!a.alive)){seed();return}render()});
$("resetBtn").addEventListener("click",()=>seed());
worldCanvas=$("world");chartCanvas=$("chart");seed();
})();