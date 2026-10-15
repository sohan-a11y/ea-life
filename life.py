#!/usr/bin/env python3
"""Network-isolated artificial-life simulation. Standard library only."""
from __future__ import annotations
import json, math, os, random, re, time, uuid
from pathlib import Path
from typing import Any

GRID = 128
MAX_POPULATION = 16
MAX_TOTAL_ORGANISMS = 500
RUN_TICKS = int(os.environ.get("SIM_TICKS", "100000"))
TICK_SLEEP = float(os.environ.get("TICK_SLEEP", "0.05"))
SAVE_EVERY = 10
STATE_DIR = Path(os.environ.get("STATE_DIR", "/state"))
REQUEST_DIR = Path(os.environ.get("REQUEST_DIR", "/exchange/requests"))
RESULT_DIR = Path(os.environ.get("RESULT_DIR", "/exchange/results"))
CONTROL_DIR = Path(os.environ.get("CONTROL_DIR", "/control"))
RESUME = os.environ.get("RESUME", "0") == "1"
SEED = int(os.environ.get("SIM_SEED", "20261015"))
TOPICS = {
    "curiosity": "curiosity driven learning artificial intelligence",
    "cellular_automata": "cellular automata computer science",
    "evolution": "evolution mutation natural selection",
    "prediction": "prediction error machine learning",
    "energy": "energy ecology organisms",
    "memory": "memory in artificial intelligence",
    "navigation": "exploration and navigation",
}
ACTIONS = [(-1,-1),(0,-1),(1,-1),(-1,0),(0,0),(1,0),(-1,1),(0,1),(1,1)]
STOPWORDS = set("about after again also among another around because before being below between could does doing during each few from further have having here into itself more most other over same some such than that their them then there these they this those through under until very what when where which while with would your using system".split())

def clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))

def sigmoid(v: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, v))))

def new_genome(rng: random.Random) -> dict[str, Any]:
    return {
        "resource_weight": rng.uniform(1.2,2.2), "novelty_weight": rng.uniform(0.4,1.5),
        "uncertainty_weight": rng.uniform(0.4,1.4), "learning_weight": rng.uniform(0.5,1.4),
        "risk_weight": rng.uniform(1.0,2.4), "cost_weight": rng.uniform(0.3,1.0),
        "curiosity_bias": rng.uniform(-1.0,-0.15), "reproduction_threshold": rng.uniform(56.0,74.0),
        "mutation_scale": rng.uniform(0.07,0.16),
    }

def mutate_genome(parent: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    child = dict(parent)
    sigma = max(0.02, min(0.25, float(parent.get("mutation_scale",0.12))))
    for key in child:
        child[key] = float(child[key]) + rng.gauss(0,sigma)
    bounds = {
        "resource_weight":(0.1,4.0), "novelty_weight":(0.0,3.0),
        "uncertainty_weight":(0.0,3.0), "learning_weight":(0.0,3.0),
        "risk_weight":(0.1,4.0), "cost_weight":(0.0,2.0),
        "curiosity_bias":(-2.0,1.0), "reproduction_threshold":(45.0,90.0),
        "mutation_scale":(0.03,0.25),
    }
    for key,(lo,hi) in bounds.items():
        child[key] = clamp(child[key],lo,hi)
    return child

def seed_world(rng: random.Random) -> list[list[float]]:
    world = [[rng.uniform(0.0,0.10) for _ in range(GRID)] for _ in range(GRID)]
    for cx,cy in ((22,24),(96,22),(24,98),(100,98),(64,64),(64,30),(32,64)):
        for y in range(max(0,cy-7),min(GRID,cy+8)):
            for x in range(max(0,cx-7),min(GRID,cx+8)):
                world[y][x] = max(world[y][x], max(0.0,0.95-0.045*(abs(x-cx)+abs(y-cy))))
    return world

def new_agent(agent_id: str, x: int, y: int, rng: random.Random,
              genome: dict[str,Any] | None = None, parent_id: str | None = None,
              generation: int = 0, energy: float = 50.0) -> dict[str,Any]:
    return {"id":agent_id,"parent_id":parent_id,"generation":generation,"x":x,"y":y,
            "energy":energy,"age":0,"offspring":0,"alive":True,
            "genome":genome if genome is not None else new_genome(rng),"visits":{},
            "q":[0.0 for _ in ACTIONS],"counts":[0 for _ in ACTIONS],"error_ema":0.5,
            "learning_progress":0.0,"last_curiosity":0.0,"known_topics":[],"known_terms":{},
            "facts":0,"pending_request_id":None,"last_request_tick":-9999,
            "last_birth_tick":-9999,"prediction_error_sum":0.0,"prediction_count":0}

def curiosity_score(agent: dict[str,Any], novelty: float, prediction_error: float,
                    risk: float, cost: float) -> float:
    g=agent["genome"]
    score=(g["novelty_weight"]*novelty+g["uncertainty_weight"]*prediction_error+
           g["learning_weight"]*agent.get("learning_progress",0.0)-g["risk_weight"]*risk-
           g["cost_weight"]*cost+g["curiosity_bias"])
    return sigmoid(score)

def request_topic(agent: dict[str,Any], tick: int, curiosity: float, rng: random.Random) -> None:
    if curiosity<0.78 or agent["pending_request_id"] or tick-agent["last_request_tick"]<120: return
    unseen=[key for key in TOPICS if key not in agent["known_topics"]]
    if not unseen: return
    topic_id=rng.choice(unseen); request_id=uuid.uuid4().hex
    REQUEST_DIR.mkdir(parents=True,exist_ok=True)
    payload={"request_id":request_id,"organism_id":agent["id"],"topic_id":topic_id,"tick":tick}
    target=REQUEST_DIR/("req_"+request_id+".json"); tmp=target.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload),encoding="utf-8"); tmp.replace(target)
    agent["pending_request_id"],agent["last_request_tick"]=request_id,tick

def absorb_result(agent: dict[str,Any], tick: int) -> None:
    request_id=agent.get("pending_request_id")
    if not request_id: return
    try: result=json.loads((RESULT_DIR/("req_"+request_id+".json")).read_text(encoding="utf-8"))
    except (OSError,ValueError): return
    agent["pending_request_id"]=None; topic_id=result.get("topic_id")
    if not result.get("ok") or topic_id not in TOPICS: return
    if topic_id not in agent["known_topics"]: agent["known_topics"].append(topic_id)
    for item in result.get("pages",[])[:3]:
        text=(str(item.get("title",""))+" "+str(item.get("extract","")))[:900].lower()
        for word in re.findall(r"[a-z][a-z'-]{3,}",text):
            if word not in STOPWORDS: agent["known_terms"][word]=min(1000,agent["known_terms"].get(word,0)+1)
    agent["facts"] += min(3,len(result.get("pages",[])))
    print("[tick %d] %s learned from approved topic %s"%(tick,agent["id"],topic_id),flush=True)

def to_tuple(v):
    return tuple(to_tuple(x) for x in v) if isinstance(v,list) else v

def save_snapshot(tick: int, world: list[list[float]], agents: list[dict[str,Any]], start_time: float) -> None:
    STATE_DIR.mkdir(parents=True,exist_ok=True)
    fields=("id","parent_id","generation","x","y","energy","age","offspring","alive",
            "last_curiosity","learning_progress","error_ema","known_topics","facts")
    public=[{k:a[k] for k in fields} for a in agents]
    data={"next_tick":tick+1,"saved_at":time.time(),"grid_size":GRID,"world":world,
          "agents":public,"runtime_seconds":round(time.time()-start_time,2)}
    tmp=STATE_DIR/"snapshot.tmp"; tmp.write_text(json.dumps(data,separators=(",",":")),encoding="utf-8")
    tmp.replace(STATE_DIR/"snapshot.json")

def main() -> None:
    rng=random.Random(SEED)
    for folder in (STATE_DIR,REQUEST_DIR,RESULT_DIR,CONTROL_DIR): folder.mkdir(parents=True,exist_ok=True)
    (CONTROL_DIR/"pause.flag").unlink(missing_ok=True)
    events_path=STATE_DIR/"events.jsonl"; agents=[]; tick=0
    if RESUME and (STATE_DIR/"full_state.json").exists():
        try:
            saved=json.loads((STATE_DIR/"full_state.json").read_text(encoding="utf-8"))
            world,agents,tick=saved["world"],saved["agents"],int(saved["tick"])
            rng.setstate(to_tuple(saved["rng_state"]))
        except (OSError,ValueError,KeyError,TypeError):
            world=seed_world(rng); agents=[]; tick=0
    else: world=seed_world(rng)
    if not agents:
        starts=((12,12),(115,12),(12,115),(115,115))
        agents=[new_agent("org-%02d"%(i+1),x,y,rng) for i,(x,y) in enumerate(starts)]
    start_time=time.time(); total_created=len(agents)
    while tick<RUN_TICKS:
        if (CONTROL_DIR/"stop.flag").exists(): break
        if (CONTROL_DIR/"pause.flag").exists(): time.sleep(0.2); continue
        living=[a for a in agents if a["alive"]]; occupied={(a["x"],a["y"]) for a in living}
        for a in living:
            absorb_result(a,tick); x,y=a["x"],a["y"]; key=str(x)+","+str(y)
            prior=a["visits"].get(key,0); novelty=1.0/math.sqrt(1+prior); a["visits"][key]=prior+1
            if len(a["visits"]) > 512:
                a["visits"].pop(next(iter(a["visits"])))
            candidates=[]
            for idx,(dx,dy) in enumerate(ACTIONS):
                nx,ny=x+dx,y+dy
                if not (0<=nx<GRID and 0<=ny<GRID): continue
                local_key=str(nx)+","+str(ny)
                local_novelty=1.0/math.sqrt(1+a["visits"].get(local_key,0))
                crowd=0.7 if (nx,ny) in occupied and (nx,ny)!=(x,y) else 0.0
                score=a["genome"]["resource_weight"]*world[ny][nx]+a["genome"]["novelty_weight"]*local_novelty+a["q"][idx]*0.08-crowd+rng.uniform(-0.04,0.04)
                candidates.append((score,idx,dx,dy,nx,ny))
            if not candidates: continue
            _,action,dx,dy,nx,ny=rng.choice(sorted(candidates,reverse=True)[:4])
            if (nx,ny)!=(x,y):
                occupied.discard((x,y))
                if (nx,ny) not in occupied: a["x"],a["y"]=nx,ny
                occupied.add((a["x"],a["y"]))
            resource=world[a["y"]][a["x"]]; reward=resource*5.0-(0.55 if (dx or dy) else 0.2)
            predicted,old=a["q"][action],a["error_ema"]; error=reward-predicted
            a["q"][action]+=0.12*error; a["counts"][action]+=1
            a["error_ema"]=0.9*old+0.1*min(1.0,abs(error)/5.0)
            a["learning_progress"]=clamp(max(0.0,old-a["error_ema"])*4.0)
            a["prediction_error_sum"]+=min(1.0,abs(error)/5.0); a["prediction_count"]+=1
            edge=min(a["x"],a["y"],GRID-1-a["x"],GRID-1-a["y"]); risk=0.35 if edge<2 else 0.0
            cost=0.2+(0.25 if (dx or dy) else 0.0)
            a["last_curiosity"]=curiosity_score(a,novelty,min(1.0,abs(error)/5.0),risk,cost)
            a["energy"]=clamp(a["energy"]+resource*3.0-0.45-(0.18 if (dx or dy) else 0.0),0,100)
            world[a["y"]][a["x"]]=max(0.0,resource-0.06); a["age"]+=1
            request_topic(a,tick,a["last_curiosity"],rng)
            if a["energy"]<=0.01 or a["age"]>4000:
                a["alive"]=False
                a["visits"].clear()
                a["known_terms"].clear()
                a["q"]=[0.0 for _ in ACTIONS]
                a["counts"]=[0 for _ in ACTIONS]
                a["pending_request_id"]=None
        for row in world:
            for i,value in enumerate(row):
                if value<0.10: row[i]=min(0.10,value+0.002)
        living=[a for a in agents if a["alive"]]
        if len(living)<MAX_POPULATION and len(agents)<MAX_TOTAL_ORGANISMS:
            occupied={(a["x"],a["y"]) for a in living}
            for parent in list(living):
                if sum(1 for a in agents if a["alive"])>=MAX_POPULATION or len(agents)>=MAX_TOTAL_ORGANISMS: break
                if parent["energy"]<parent["genome"]["reproduction_threshold"] or tick-parent["last_birth_tick"]<80: continue
                spots=[(parent["x"]+dx,parent["y"]+dy) for dx,dy in ACTIONS if 0<=parent["x"]+dx<GRID and 0<=parent["y"]+dy<GRID and (parent["x"]+dx,parent["y"]+dy) not in occupied]
                if not spots: continue
                bx,by=rng.choice(spots); child_id="org-"+uuid.uuid4().hex[:8]
                child=new_agent(child_id,bx,by,rng,mutate_genome(parent["genome"],rng),parent["id"],parent["generation"]+1,30.0)
                agents.append(child); parent["offspring"]+=1; parent["last_birth_tick"]=tick; parent["energy"]-=20.0
                occupied.add((bx,by)); living.append(child); total_created+=1
                print("[tick %d] BIRTH %s -> %s generation %d"%(tick,parent["id"],child_id,child["generation"]),flush=True)
        if tick%SAVE_EVERY==0:
            save_snapshot(tick,world,agents,start_time)
            alive=sum(1 for a in agents if a["alive"])
            ev={"tick":tick,"type":"population","alive":alive,"total":len(agents),"births":sum(a["offspring"] for a in agents),
                "max_generation":max((a["generation"] for a in agents),default=0),
                "mean_prediction_error":round(sum(a["prediction_error_sum"] for a in agents)/max(1,sum(a["prediction_count"] for a in agents)),4),
                "mean_curiosity":round(sum(a["last_curiosity"] for a in agents if a["alive"])/max(1,alive),3),
                "known_topics":sum(len(a["known_topics"]) for a in agents),
                "message":"Population %d · births %d"%(alive,sum(a["offspring"] for a in agents))}
            with events_path.open("a",encoding="utf-8") as fp: fp.write(json.dumps(ev,separators=(",",":"))+"\n")
            (STATE_DIR/"full_state.json").write_text(json.dumps({"tick":tick+1,"world":world,"agents":agents,"rng_state":rng.getstate()}),encoding="utf-8")
        tick+=1
        if TICK_SLEEP>0: time.sleep(TICK_SLEEP)
    save_snapshot(tick,world,agents,start_time)
    summary={"ticks":tick,"total_organisms_created":total_created,"alive":sum(1 for a in agents if a["alive"]),
             "births":sum(a["offspring"] for a in agents),"max_generation":max((a["generation"] for a in agents),default=0),
             "runtime_seconds":round(time.time()-start_time,2)}
    (STATE_DIR/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print("EXPERIMENT COMPLETE",json.dumps(summary),flush=True)

if __name__=="__main__": main()
