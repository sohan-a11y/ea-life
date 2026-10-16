"""EA-Life simulation engine. Standard library only; no network access here."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import json, math, random, time, uuid
from pathlib import Path
from typing import Any
from .latent_world_model import NeuralLatentWorldModel

ACTIONS = ((-1,-1),(0,-1),(1,-1),(-1,0),(0,0),(1,0),(-1,1),(0,1),(1,1))
ROLES = ("forager","explorer","navigator","avoider")
SCENARIOS = ("mixed","forage","explore","navigate","avoid_hazards")
TOPICS = {
    "curiosity":"curiosity and intrinsic motivation in learning",
    "cellular_automata":"cellular automata artificial life",
    "evolution":"evolution mutation natural selection",
    "prediction":"prediction error and reinforcement learning",
    "memory":"memory models in artificial intelligence",
    "navigation":"navigation and spatial learning",
    "ecology":"ecology resource competition",
    "collectives":"collective behavior swarm intelligence",
}
STOPWORDS = set("about after again also among another around because before being below between could does doing during each few from further have having here into itself more most other over same some such than that their them then there these they this those through under until very what when where which while with would your using system systems based have been".split())
GENE_BOUNDS = {
    "food_drive":(0.25,3.0), "novelty_drive":(0.0,3.5), "uncertainty_drive":(0.0,2.5),
    "goal_drive":(0.0,3.5), "risk_aversion":(0.1,4.0), "crowd_penalty":(0.0,2.0),
    "learning_rate":(0.02,0.35), "discount":(0.5,0.98), "curiosity_bias":(-2.0,1.2),
    "reproduction_threshold":(52.0,88.0), "mutation_scale":(0.025,0.25), "value_weight":(0.1,2.0),
    "predictive_drive":(0.0,2.2), "collective_drive":(0.0,2.5), "signal_aversion":(0.0,2.5),
}

def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low,min(high,value))

def sigmoid(value: float) -> float:
    return 1.0/(1.0+math.exp(-max(-30.0,min(30.0,value))))

def tupleify(value: Any) -> Any:
    return tuple(tupleify(v) for v in value) if isinstance(value,list) else value

@dataclass(frozen=True)
class Config:
    grid_size: int = 96
    initial_population: int = 12
    max_population: int = 40
    max_total_organisms: int = 500
    scenario: str = "mixed"
    evolution_enabled: bool = True
    max_age: int = 2600
    max_q_states: int = 512
    max_memory: int = 32
    max_semantic_terms: int = 128
    max_world_model_states: int = 256
    world_model_enabled: bool = True
    collective_enabled: bool = True
    latent_model_enabled: bool = True
    imagination_enabled: bool = True
    imagination_horizon: int = 4
    imagination_beam_width: int = 3
    imagination_bonus: float = 0.45
    def __post_init__(self):
        if not 32 <= self.grid_size <= 192: raise ValueError("grid_size must be between 32 and 192")
        if not 1 <= self.initial_population <= self.max_population <= 128: raise ValueError("population bounds are invalid")
        if self.max_total_organisms < self.initial_population: raise ValueError("total organism cap is below starting population")
        if self.scenario not in SCENARIOS: raise ValueError("unknown scenario")
        if self.max_q_states < 16 or self.max_memory < 4: raise ValueError("memory bounds are too small")
        if not 16 <= self.max_world_model_states <= 2048: raise ValueError("world-model bound must be between 16 and 2048")
        if not 1 <= self.imagination_horizon <= 12: raise ValueError("imagination horizon must be between 1 and 12")
        if not 1 <= self.imagination_beam_width <= 8: raise ValueError("imagination beam width must be between 1 and 8")
        if not 0.0 <= self.imagination_bonus <= 10.0: raise ValueError("imagination bonus must be between 0 and 10")

def config_from_env() -> Config:
    import os
    return Config(
        grid_size=int(os.environ.get("SIM_GRID","96")),
        initial_population=int(os.environ.get("SIM_POPULATION","12")),
        max_population=int(os.environ.get("SIM_MAX_POPULATION","40")),
        max_total_organisms=int(os.environ.get("SIM_MAX_ORGANISMS","500")),
        scenario=os.environ.get("SIM_SCENARIO","mixed"),
        evolution_enabled=os.environ.get("SIM_EVOLUTION_ENABLED","1") != "0",
        max_age=int(os.environ.get("SIM_MAX_AGE","2600")),
        world_model_enabled=os.environ.get("SIM_WORLD_MODEL_ENABLED","1") != "0",
        collective_enabled=os.environ.get("SIM_COLLECTIVE_ENABLED","1") != "0",
        max_world_model_states=int(os.environ.get("SIM_WORLD_MODEL_STATES","256")),
        latent_model_enabled=os.environ.get("SIM_LATENT_MODEL_ENABLED","1") != "0",
        imagination_enabled=os.environ.get("SIM_IMAGINATION_ENABLED","1") != "0",
        imagination_horizon=int(os.environ.get("SIM_IMAGINATION_HORIZON","4")),
        imagination_beam_width=int(os.environ.get("SIM_IMAGINATION_BEAM_WIDTH","3")),
        imagination_bonus=float(os.environ.get("SIM_IMAGINATION_BONUS","0.45")),
    )

def new_genome(rng: random.Random) -> dict[str,float]:
    return {
        "food_drive":rng.uniform(.9,2.3), "novelty_drive":rng.uniform(.4,1.8),
        "uncertainty_drive":rng.uniform(.3,1.5), "goal_drive":rng.uniform(.5,2.0),
        "risk_aversion":rng.uniform(.8,2.5), "crowd_penalty":rng.uniform(.2,1.0),
        "learning_rate":rng.uniform(.06,.20), "discount":rng.uniform(.75,.94),
        "curiosity_bias":rng.uniform(-.9,-.15), "reproduction_threshold":rng.uniform(60,78),
        "mutation_scale":rng.uniform(.06,.15), "value_weight":rng.uniform(.4,1.1),
        "predictive_drive":rng.uniform(.35,1.35), "collective_drive":rng.uniform(.3,1.5),
        "signal_aversion":rng.uniform(.4,1.7),
    }

def mutate_genome(parent: dict[str,float], rng: random.Random) -> dict[str,float]:
    child = {}
    sigma = clamp(parent.get("mutation_scale",.1),.025,.25)
    for key,(low,high) in GENE_BOUNDS.items():
        value = float(parent.get(key,(low+high)/2))+rng.gauss(0,sigma*(high-low)*.14)
        child[key] = clamp(value,low,high)
    return child

def curiosity_score(genome: dict[str,float], novelty: float, uncertainty: float,
                    learning_progress: float, risk: float) -> float:
    value = (genome["novelty_drive"]*novelty + genome["uncertainty_drive"]*uncertainty
             + genome["learning_rate"]*5*learning_progress - genome["risk_aversion"]*risk
             + genome["curiosity_bias"])
    return sigmoid(value)

class Simulation:
    """Discrete 2-D environment with bounded per-organism learning state."""
    def __init__(self, config: Config | None = None, seed: int = 20261015,
                 request_dir: str | Path | None = None, result_dir: str | Path | None = None):
        self.config = config or Config()
        self.rng = random.Random(seed)
        self.request_dir = Path(request_dir) if request_dir else None
        self.result_dir = Path(result_dir) if result_dir else None
        self.latent_model = NeuralLatentWorldModel(seed=seed + 104729)
        # How much the imagined plan actually steers behaviour (see metrics()).
        self.plan_stats = {"planned":0,"adopted":0,"promoted":0}
        # Held-out evaluation: when frozen, transitions are recorded but not trained on.
        self.latent_frozen = False
        self.latent_heldout: list[dict[str,Any]] = []
        self.tick = 0
        self.total_created = self.total_births = self.total_deaths = self.goal_events = 0
        self.next_id = 1
        self.event_seq = 0
        self.events: list[dict[str,Any]] = []
        self.resources: list[float] = []
        self.hazards: list[float] = []
        self.global_visits: list[int] = []
        self.beacons: list[dict[str,int]] = []
        self.agents: list[dict[str,Any]] = []
        self._seed_world()
        self._seed_agents()

    def _idx(self,x:int,y:int)->int: return y*self.config.grid_size+x
    def _distance(self,x:int,y:int,target:dict[str,int])->int:
        return max(abs(x-target["x"]),abs(y-target["y"]))

    def _seed_world(self):
        n = self.config.grid_size
        self.resources = [self.rng.uniform(.005,.08) for _ in range(n*n)]
        self.hazards = [0.0 for _ in range(n*n)]
        self.global_visits = [0 for _ in range(n*n)]
        self.resource_trails = [0.0 for _ in range(n*n)]
        self.danger_trails = [0.0 for _ in range(n*n)]
        centres = [(max(5,n//4),max(5,n//4)),(min(n-6,n*3//4),max(5,n//4)),
                   (max(5,n//4),min(n-6,n*3//4)),(min(n-6,n*3//4),min(n-6,n*3//4)),(n//2,n//2)]
        for cx,cy in centres:
            for y in range(max(0,cy-8),min(n,cy+9)):
                for x in range(max(0,cx-8),min(n,cx+9)):
                    d=abs(x-cx)+abs(y-cy); idx=self._idx(x,y)
                    self.resources[idx]=max(self.resources[idx],max(0.0,.98-d*.052))
        for cx,cy in [(n//2,n//3),(n//3,n//2),(n*2//3,n*2//3)]:
            for y in range(max(0,cy-4),min(n,cy+5)):
                for x in range(max(0,cx-4),min(n,cx+5)):
                    self.hazards[self._idx(x,y)]=max(0.0,.75-(abs(x-cx)+abs(y-cy))*.09)
        self.beacons=[{"x":x,"y":y} for x,y in [
            (max(3,n//8),max(3,n//8)),(min(n-4,n*7//8),max(3,n//8)),
            (max(3,n//8),min(n-4,n*7//8)),(min(n-4,n*7//8),min(n-4,n*7//8))]]

    def _seed_agents(self):
        n=self.config.grid_size
        starts=[(max(1,n//10),max(1,n//10)),(n-2-max(1,n//10),max(1,n//10)),
                (max(1,n//10),n-2-max(1,n//10)),(n-2-max(1,n//10),n-2-max(1,n//10)),
                (n//2,min(n-2,n//2+10))]
        for i in range(self.config.initial_population):
            x,y=starts[i] if i<len(starts) else (self.rng.randrange(n),self.rng.randrange(n))
            self.agents.append(self._new_agent(x,y,role=ROLES[i%len(ROLES)]))
        self.total_created=len(self.agents)
        self._event("system","Experiment initialized",{"seeded_population":len(self.agents),
                    "scenario":self.config.scenario,"grid_size":n})

    def _new_agent(self,x:int,y:int,role:str="forager",genome:dict[str,float]|None=None,
                   parent_id:str|None=None,generation:int=0,energy:float=64.0,
                   inherited_terms:dict[str,int]|None=None)->dict[str,Any]:
        agent_id="org-%04d"%self.next_id
        self.next_id+=1
        return {
            "id":agent_id,"parent_id":parent_id,"generation":generation,"role":role,"x":x,"y":y,
            "energy":energy,"age":0,"offspring":0,"alive":True,
            "genome":genome if genome is not None else new_genome(self.rng),
            "visits":{},"q_table":{},"model":[[0,0.0,0.0] for _ in ACTIONS],
            "world_model":{},"world_model_error_sum":0.0,"world_model_error_count":0,
            "world_model_next_correct":0,"world_model_prediction_count":0,
            "world_model_surprise_sum":0.0,"world_model_surprise_count":0,
            "error_ema":.5,"learning_progress":0.0,"curiosity":0.0,"known_topics":[],
            "semantic_terms":dict(list((inherited_terms or {}).items())[:8]),"facts":0,
            "pending_request_id":None,"last_request_tick":-10000,
            "last_birth_tick":-10000,"goal_index":(self.next_id-2)%len(self.beacons),
            "goals_reached":0,"reward_total":0.0,"prediction_error_sum":0.0,
            "prediction_count":0,"memory":[],"last_action":None,
        }

    def _event(self,kind:str,message:str,detail:dict[str,Any]|None=None):
        self.event_seq+=1
        self.events.append({"seq":self.event_seq,"tick":self.tick,"type":kind,"message":message,"detail":detail or {}})
        self.events=self.events[-120:]

    def _task(self,agent:dict[str,Any])->str:
        scenario_map={"forage":"forager","explore":"explorer","navigate":"navigator","avoid_hazards":"avoider"}
        return scenario_map.get(self.config.scenario,agent["role"]) if self.config.scenario!="mixed" else agent["role"]

    def _goal(self,agent:dict[str,Any])->dict[str,int]:
        return self.beacons[agent["goal_index"]%len(self.beacons)]

    def _state_key(self,agent:dict[str,Any])->str:
        idx=self._idx(agent["x"],agent["y"]); target=self._goal(agent)
        dist=self._distance(agent["x"],agent["y"],target); visits=int(agent["visits"].get(str(idx),0))
        rb=min(3,int(self.resources[idx]*4)); eb=min(3,int(agent["energy"]/25))
        nb=min(3,int(3/math.sqrt(1+visits))); gb=min(3,dist*4//max(1,self.config.grid_size))
        hb=min(2,int(self.hazards[idx]*3))
        return "%d:%d:%d:%d:%d"%(rb,eb,nb,gb,hb)

    def _q_values(self,agent:dict[str,Any],key:str)->list[float]:
        table=agent["q_table"]
        if key not in table:
            if len(table)>=self.config.max_q_states: table.pop(next(iter(table)))
            table[key]=[0.0 for _ in ACTIONS]
        return table[key]

    def _model_stats(self,agent:dict[str,Any],action:int)->tuple[float,float]:
        count,mean,m2=agent["model"][action]
        if count<2: return float(mean),1/math.sqrt(count+1)
        return float(mean),min(2.0,math.sqrt(max(0.0,m2/(count-1)))/math.sqrt(count))

    def _update_model(self,agent:dict[str,Any],action:int,reward:float):
        row=agent["model"][action]; row[0]+=1; delta=reward-row[1]
        row[1]+=delta/row[0]; row[2]+=delta*(reward-row[1])

    def _world_model_key(self,state_key:str,action:int)->str:
        return state_key+"|"+str(action)

    def _world_model_stats(self,agent:dict[str,Any],state_key:str,action:int)->tuple[float,float,float]:
        """Return predicted immediate reward, confidence, and normalized uncertainty."""
        row=agent.get("world_model",{}).get(self._world_model_key(state_key,action))
        if not row:
            return 0.0,0.0,1.0
        count=int(row.get("count",0))
        variance=max(0.0,float(row.get("reward_m2",0.0))/(count-1)) if count>1 else 1.0
        uncertainty=min(1.0,1.0/math.sqrt(count+1)+min(.35,math.sqrt(variance)/12.0))
        return float(row.get("reward_mean",0.0)),count/(count+3.0),uncertainty

    def _update_world_model(self,agent:dict[str,Any],state_key:str,action:int,
                            next_state_key:str,reward:float):
        """Online bounded transition model: state/action -> next-state counts and reward moments."""
        world=agent.setdefault("world_model",{})
        key=self._world_model_key(state_key,action)
        if key not in world:
            if len(world)>=self.config.max_world_model_states:
                world.pop(next(iter(world)))
            world[key]={"count":0,"next_counts":{},"reward_mean":0.0,"reward_m2":0.0}
        row=world[key]
        count=int(row["count"])
        next_counts=row["next_counts"]
        if count:
            agent["world_model_error_sum"]+=abs(reward-float(row["reward_mean"]))
            agent["world_model_error_count"]+=1
            if next_counts:
                predicted=max(next_counts.items(),key=lambda item:item[1])[0]
                agent["world_model_next_correct"]+=int(predicted==next_state_key)
                agent["world_model_prediction_count"]+=1
            # Use tracked outcome counts after top-k pruning, keeping this diagnostic
            # normalized over the outcomes the bounded model currently retains.
            retained_total=sum(int(value) for value in next_counts.values())
            probability=(int(next_counts.get(next_state_key,0))+1)/(retained_total+len(next_counts)+1)
            surprise=min(1.0,-math.log(probability)/math.log(max(2,retained_total+len(next_counts)+1)))
            agent["world_model_surprise_sum"]+=surprise
            agent["world_model_surprise_count"]+=1
        next_counts[next_state_key]=int(next_counts.get(next_state_key,0))+1
        if len(next_counts)>24:
            drop=min((candidate for candidate in next_counts if candidate!=next_state_key),
                     key=lambda candidate:next_counts[candidate])
            del next_counts[drop]
        delta=reward-float(row["reward_mean"])
        row["reward_mean"]=float(row["reward_mean"])+delta/(count+1)
        row["reward_m2"]=float(row["reward_m2"])+delta*(reward-float(row["reward_mean"]))
        row["count"]=count+1

    def _role_weights(self,agent:dict[str,Any])->dict[str,float]:
        g=agent["genome"]; w={"food":g["food_drive"],"novelty":g["novelty_drive"],
                               "goal":g["goal_drive"],"risk":g["risk_aversion"]}
        task=self._task(agent)
        if task=="explorer": w["novelty"]*=1.65; w["food"]*=.65
        elif task=="navigator": w["goal"]*=1.85
        elif task=="avoider": w["risk"]*=1.8
        elif task=="forager": w["food"]*=1.35; w["novelty"]*=.8
        return w

    def _latent_features(self,agent:dict[str,Any])->list[float]:
        """Encode normalized local state plus the organism's task; every feature is in [0, 1].

        The task one-hot matters: rewards depend on the task (explorers are paid for novelty,
        navigators for beacon progress), and the latent model is shared by all organisms."""
        n=self.config.grid_size; x=int(agent["x"]); y=int(agent["y"])
        idx=self._idx(x,y); target=self._goal(agent); scale=max(1,n-1)
        resource=float(self.resources[idx]); hazard=float(self.hazards[idx])
        visits=int(agent.get("visits",{}).get(str(idx),0))
        nx=(1.0/math.sqrt(1.0+visits))
        local_resources=[]; local_hazards=[]
        for yy in range(max(0,y-1),min(n,y+2)):
            for xx in range(max(0,x-1),min(n,x+2)):
                cell=self._idx(xx,yy)
                local_resources.append(self.resources[cell]); local_hazards.append(self.hazards[cell])
        crowd=sum(1 for other in self.agents if other.get("alive") and other["id"]!=agent["id"]
                  and max(abs(other["x"]-x),abs(other["y"]-y))<=1)/8.0
        trail_fields=getattr(self,"_resource_trail_view",self.resource_trails)
        danger_fields=getattr(self,"_danger_trail_view",self.danger_trails)
        return [
            clamp(float(agent.get("energy",0.0))/100.0), clamp(resource), clamp(hazard), clamp(nx),
            clamp(.5+(target["x"]-x)/(2.0*scale)), clamp(.5+(target["y"]-y)/(2.0*scale)),
            clamp(self._distance(x,y,target)/scale), clamp(float(trail_fields[idx])),
            clamp(float(danger_fields[idx])), clamp(sum(local_resources)/max(1,len(local_resources))),
            clamp(sum(local_hazards)/max(1,len(local_hazards))), clamp(crowd),
            clamp(x/scale), clamp(y/scale),
        ]+[1.0 if self._task(agent)==role else 0.0 for role in ROLES]

    def _choose_action(self,agent:dict[str,Any],occupied:set[int],
                       latent_observation:list[float]|None=None)->tuple[int,float,str,float]:
        n=self.config.grid_size; x,y=agent["x"],agent["y"]; current=self._idx(x,y)
        key=self._state_key(agent); q=self._q_values(agent,key); target=self._goal(agent)
        current_dist=self._distance(x,y,target); weights=self._role_weights(agent); g=agent["genome"]
        seen=int(agent["visits"].get(str(current),0)); novelty_now=1/math.sqrt(1+seen)
        candidates=[]
        for action,(dx,dy) in enumerate(ACTIONS):
            nx,ny=x+dx,y+dy
            if not (0<=nx<n and 0<=ny<n): continue
            idx=self._idx(nx,ny); novelty=1/math.sqrt(1+int(agent["visits"].get(str(idx),0)))
            resource=self.resources[idx]; hazard=self.hazards[idx]
            crowd=1.0 if idx in occupied and idx!=current else 0.0
            progress=(current_dist-self._distance(nx,ny,target))/max(1,n//2)
            _,uncertainty=self._model_stats(agent,action)
            predicted_reward,confidence,world_uncertainty=self._world_model_stats(agent,key,action)
            resource_trail=(self._resource_trail_view[idx] if self.config.collective_enabled else 0.0)
            danger_trail=(self._danger_trail_view[idx] if self.config.collective_enabled else 0.0)
            collective=(g.get("collective_drive",0.0)*resource_trail
                        -g.get("signal_aversion",0.0)*danger_trail)
            predictive=(g.get("predictive_drive",0.0)*confidence*clamp(predicted_reward/4.0,-1.0,1.0)
                        +g["uncertainty_drive"]*world_uncertainty*.12)
            score=(weights["food"]*resource+weights["novelty"]*novelty+weights["goal"]*progress
                   -weights["risk"]*hazard-g["crowd_penalty"]*crowd
                   +g["value_weight"]*q[action]+g["uncertainty_drive"]*uncertainty
                   +collective+predictive+self.rng.uniform(-.035,.035))
            candidates.append((score,action,novelty,uncertainty))
        if not candidates: raise RuntimeError("organism has no legal actions")
        plan_action=None
        if (self.config.latent_model_enabled and self.config.imagination_enabled
                and latent_observation is not None):
            legal_actions=[item[1] for item in candidates]
            plan=self.latent_model.imagine(latent_observation,
                horizon=self.config.imagination_horizon,
                beam_width=self.config.imagination_beam_width,
                legal_first_actions=legal_actions)
            if plan is not None:
                boost=self.config.imagination_bonus
                plan_action=plan["action"]
                top_before=max(item[0] for item in candidates)
                plan_score=next(item[0] for item in candidates if item[1]==plan_action)
                plan_in_band_before=plan_score>=top_before-.03
                candidates=[(score+(boost if candidate_action==plan["action"] else 0.0),
                             candidate_action,novelty_value,uncertainty_value)
                            for score,candidate_action,novelty_value,uncertainty_value in candidates]
                top_after=max(item[0] for item in candidates)
                self.plan_stats["planned"]+=1
                # The bonus moved the plan's action into the greedy choice band.
                self.plan_stats["promoted"]+=int((plan_score+boost)>=top_after-.03 and not plan_in_band_before)
        epsilon=.035+.18*novelty_now*clamp(g["novelty_drive"]/3.5)
        if self.rng.random()<epsilon: picked=self.rng.choice(candidates)
        else:
            high=max(item[0] for item in candidates)
            picked=self.rng.choice([item for item in candidates if item[0]>=high-.03])
        if plan_action is not None:
            self.plan_stats["adopted"]+=int(picked[1]==plan_action)
        return picked[1],picked[2],key,picked[3]

    def _request_topic(self,agent:dict[str,Any]):
        if self.request_dir is None or agent["curiosity"]<.84 or agent["pending_request_id"]: return
        # Stop generating requests once the finite broker budget has been exhausted.
        if self.result_dir is not None:
            try:
                broker_state=json.loads((self.result_dir/"broker_status.json").read_text(encoding="utf-8"))
                state=broker_state.get("state")
                stale=(state=="active" and
                       time.time()-float(broker_state.get("updated_at",0.0))>30.0)
                if state=="exhausted" or stale: return
            except (OSError,json.JSONDecodeError,TypeError,ValueError):
                pass
        if len(list(self.request_dir.glob("req_*.json")))>=128: return
        if self.tick-agent["last_request_tick"]<240: return
        unseen=[key for key in TOPICS if key not in agent["known_topics"]]
        if not unseen: return
        topic=self.rng.choice(unseen); rid=uuid.uuid4().hex
        self.request_dir.mkdir(parents=True,exist_ok=True)
        payload={"request_id":rid,"organism_id":agent["id"],"topic_id":topic,"tick":self.tick}
        path=self.request_dir/("req_"+rid+".json"); tmp=path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload),encoding="utf-8"); tmp.replace(path)
        agent["pending_request_id"]=rid; agent["last_request_tick"]=self.tick

    def _absorb_research(self,agent:dict[str,Any]):
        rid=agent.get("pending_request_id")
        if self.result_dir is None or not rid: return
        if self.tick-agent["last_request_tick"]>400:
            agent["pending_request_id"]=None; return
        try: result=json.loads((self.result_dir/("req_"+rid+".json")).read_text(encoding="utf-8"))
        except (OSError,json.JSONDecodeError): return
        agent["pending_request_id"]=None; topic=result.get("topic_id")
        if not result.get("ok") or topic not in TOPICS: return
        if topic not in agent["known_topics"]: agent["known_topics"].append(topic)
        import re
        for page in result.get("pages",[])[:3]:
            text=(str(page.get("title",""))+" "+str(page.get("extract","")))[:700].lower()
            for word in re.findall(r"[a-z][a-z'-]{3,}",text):
                if word not in STOPWORDS:
                    agent["semantic_terms"][word]=min(1000,agent["semantic_terms"].get(word,0)+1)
        if len(agent["semantic_terms"])>self.config.max_semantic_terms:
            ranked=sorted(agent["semantic_terms"].items(),key=lambda pair:pair[1],reverse=True)
            agent["semantic_terms"]=dict(ranked[:self.config.max_semantic_terms])
        agent["facts"]+=min(3,len(result.get("pages",[])))
        self._event("research",agent["id"]+" absorbed a bounded reference result",
                    {"organism_id":agent["id"],"topic_id":topic,"pages":min(3,len(result.get("pages",[])))})

    def _update_trails(self):
        """End-of-tick update of the shared trail fields. With no rule installed this is the original
        hand-coded rule: each cell decays on its own (x0.985 food trail, x0.96 danger trail)."""
        rule=getattr(self,"trail_rule",None)
        if rule is None:
            for i in range(len(self.resource_trails)):
                self.resource_trails[i]*=.985
                if self.resource_trails[i]<.001: self.resource_trails[i]=0.0
            for i in range(len(self.danger_trails)):
                self.danger_trails[i]*=.96
                if self.danger_trails[i]<.001: self.danger_trails[i]=0.0
        else:
            self.resource_trails,self.danger_trails=rule.update(self.resource_trails,self.danger_trails,self.config.grid_size)

    def step(self):
        n=self.config.grid_size
        living_start=[a for a in self.agents if a["alive"]]
        # Every organism observes the same pre-tick signal fields; deposits take effect next tick.
        self._resource_trail_view=self.resource_trails.copy()
        self._danger_trail_view=self.danger_trails.copy()
        occupied={self._idx(a["x"],a["y"]) for a in living_start}
        pending_transitions:list[tuple[list[float],dict[str,Any],int,float]]=[]
        for agent in living_start:
            if not agent["alive"]: continue
            self._absorb_research(agent)
            x0,y0=agent["x"],agent["y"]; old_idx=self._idx(x0,y0)
            visits=agent["visits"]; old_seen=int(visits.get(str(old_idx),0))
            novelty_now=1/math.sqrt(1+old_seen); visits[str(old_idx)]=old_seen+1
            if len(visits)>700: visits.pop(next(iter(visits)))
            old_key=self._state_key(agent); previous_q=list(self._q_values(agent,old_key))
            latent_observation=(self._latent_features(agent) if self.config.latent_model_enabled else None)
            action,novelty,state_key,uncertainty=self._choose_action(agent,occupied,latent_observation)
            dx,dy=ACTIONS[action]; nx=max(0,min(n-1,x0+dx)); ny=max(0,min(n-1,y0+dy))
            new_idx=self._idx(nx,ny)
            if new_idx!=old_idx:
                occupied.discard(old_idx)
                if new_idx not in occupied:
                    agent["x"],agent["y"]=nx,ny; occupied.add(new_idx)
                else:
                    new_idx=old_idx; nx,ny=x0,y0; occupied.add(old_idx)
            resource=self.resources[new_idx]; hazard=self.hazards[new_idx]
            if self.config.collective_enabled:
                if resource>=.18:
                    self.resource_trails[new_idx]=min(1.0,self.resource_trails[new_idx]+resource*.24)
                if hazard>=.04:
                    self.danger_trails[new_idx]=min(1.0,self.danger_trails[new_idx]+hazard*.32)
            old_dist=self._distance(x0,y0,self._goal(agent)); new_dist=self._distance(nx,ny,self._goal(agent))
            progress=(old_dist-new_dist)/max(1,n//2); task=self._task(agent)
            food_reward=resource*2.4
            intrinsic=novelty*(.9 if task=="explorer" else .18)
            navigation=progress*(2.1 if task=="navigator" else .25)
            hazard_cost=hazard*(3.5 if task=="avoider" else 2.5)
            moving=bool(dx or dy); movement_cost=.14 if moving else .06
            crowd=sum(1 for other in self.agents if other["alive"] and other["id"]!=agent["id"]
                      and abs(other["x"]-nx)<=1 and abs(other["y"]-ny)<=1)/8.0
            reward=food_reward+intrinsic+navigation-hazard_cost-movement_cost-crowd*.12
            energy_delta=resource*4.8-.36-movement_cost*.4-hazard*(1.6 if task=="avoider" else 2.0)
            agent["energy"]=clamp(agent["energy"]+energy_delta,0.0,100.0)
            self.resources[new_idx]=max(0.0,resource-.065)
            self.global_visits[new_idx]+=1
            agent["energy"]=max(0.0,agent["energy"]-hazard*.4)
            reached=new_dist<=1
            if reached and task=="navigator":
                reward+=4.5; agent["goals_reached"]+=1; self.goal_events+=1
                self._event("goal",agent["id"]+" reached a beacon",
                            {"organism_id":agent["id"],"generation":agent["generation"]})
                agent["goal_index"]=(agent["goal_index"]+1)%len(self.beacons)
            new_key=self._state_key(agent)
            terminal_transition=(agent["energy"]<=.01 or agent["age"]+1>=self.config.max_age)
            if terminal_transition:
                target=reward
            else:
                next_q=self._q_values(agent,new_key)
                target=reward+agent["genome"]["discount"]*max(next_q)
            td_error=target-previous_q[action]
            self._q_values(agent,state_key)[action]+=agent["genome"]["learning_rate"]*td_error
            self._update_model(agent,action,reward)
            if latent_observation is not None:
                pending_transitions.append((latent_observation,agent,action,reward))
            if self.config.world_model_enabled:
                self._update_world_model(agent,state_key,action,new_key,reward)
            old_ema=agent["error_ema"]; normalized=min(3.0,abs(td_error)/4.0)
            agent["error_ema"]=.92*old_ema+.08*normalized
            agent["learning_progress"]=clamp(max(0.0,old_ema-agent["error_ema"])*3.5)
            agent["prediction_error_sum"]+=abs(td_error); agent["prediction_count"]+=1
            _,uncertainty=self._model_stats(agent,action)
            agent["curiosity"]=curiosity_score(agent["genome"],novelty_now,uncertainty,
                                               agent["learning_progress"],hazard)
            agent["age"]+=1; agent["reward_total"]+=reward; agent["last_action"]=action
            memory=agent["memory"]
            memory.append({"tick":self.tick,"x":nx,"y":ny,"action":action,
                           "reward":round(reward,4),"td_error":round(td_error,4)})
            if len(memory)>self.config.max_memory: del memory[:len(memory)-self.config.max_memory]
            self._request_topic(agent)
            if agent["energy"]<=.01 or agent["age"]>=self.config.max_age:
                agent["alive"]=False; agent["visits"].clear(); agent["q_table"].clear(); agent["memory"].clear()
                agent["world_model"]={}
                self.total_deaths+=1
                self._event("death",agent["id"]+" became inactive",
                            {"organism_id":agent["id"],"age":agent["age"],"generation":agent["generation"]})
        for i,value in enumerate(self.resources):
            if value<.09: self.resources[i]=min(.09,value+.0025)
        for i,value in enumerate(self.hazards):
            if value>.01: self.hazards[i]=max(0.0,value-.00025)
        if self.config.collective_enabled:
            self._update_trails()
        if self.config.evolution_enabled:
            self._reproduce()
        self.tick+=1
        # Train only after the entire world step, so every sample's target observes the
        # same end-of-tick fields rather than a partially updated organism iteration.
        self._resource_trail_view=self.resource_trails.copy()
        self._danger_trail_view=self.danger_trails.copy()
        for observation,agent,action,reward in pending_transitions:
            next_observation=self._latent_features(agent)
            if self.latent_frozen:
                if len(self.latent_heldout)<20000:
                    self.latent_heldout.append({"obs":observation,"action":action,
                                                "next_obs":next_observation,"reward":reward})
            else:
                self.latent_model.observe_transition(observation,action,next_observation,reward)

    def freeze_latent_model(self):
        """Stop training; later transitions are kept for a held-out evaluation."""
        self.latent_frozen=True

    def latent_heldout_metrics(self)->dict[str,Any]:
        return self.latent_model.evaluate(self.latent_heldout)

    def _reproduce(self):
        living=[a for a in self.agents if a["alive"]]
        if len(living)>=self.config.max_population or self.total_created>=self.config.max_total_organisms: return
        occupied={self._idx(a["x"],a["y"]) for a in living}
        for parent in list(living):
            if sum(1 for a in self.agents if a["alive"])>=self.config.max_population: break
            if self.total_created>=self.config.max_total_organisms: break
            if parent["energy"]<parent["genome"]["reproduction_threshold"] or self.tick-parent["last_birth_tick"]<100: continue
            x0,y0=parent["x"],parent["y"]; candidates=[]
            for dx,dy in ACTIONS:
                x,y=x0+dx,y0+dy
                if 0<=x<self.config.grid_size and 0<=y<self.config.grid_size:
                    idx=self._idx(x,y)
                    if idx not in occupied: candidates.append((x,y,idx))
            if not candidates: continue
            x,y,idx=self.rng.choice(candidates); role=parent["role"]
            if self.rng.random()<.04: role=self.rng.choice(ROLES)
            terms=dict(sorted(parent["semantic_terms"].items(),key=lambda p:p[1],reverse=True)[:8])
            child=self._new_agent(x,y,role,mutate_genome(parent["genome"],self.rng),
                                  parent["id"],parent["generation"]+1,29.0,terms)
            self.agents.append(child); occupied.add(idx)
            parent["offspring"]+=1; parent["last_birth_tick"]=self.tick
            parent["energy"]=max(0.0,parent["energy"]-19.0)
            self.total_created+=1; self.total_births+=1
            self._event("birth",parent["id"]+" reproduced as "+child["id"],
                        {"parent_id":parent["id"],"child_id":child["id"],
                         "generation":child["generation"],"role":role})

    def metrics(self)->dict[str,Any]:
        living=[a for a in self.agents if a["alive"]]
        mean=lambda key: sum(float(a.get(key,0.0)) for a in living)/max(1,len(living))
        prediction=sum(a["prediction_error_sum"] for a in self.agents)/max(1,sum(a["prediction_count"] for a in self.agents))
        genes=list(GENE_BOUNDS)
        pairs=[(living[i],living[j]) for i in range(len(living)) for j in range(i+1,len(living))]
        diversity=(sum(sum(abs(a["genome"][k]-b["genome"][k])/(GENE_BOUNDS[k][1]-GENE_BOUNDS[k][0])
                           for k in genes)/len(genes) for a,b in pairs)/len(pairs)) if pairs else 0.0
        roles={role:sum(1 for a in living if self._task(a)==role) for role in ROLES}
        coverage=sum(1 for v in self.resources if v>.15)/max(1,len(self.resources))
        wm_entries=sum(len(a.get("world_model",{})) for a in living)
        wm_error=sum(float(a.get("world_model_error_sum",0.0)) for a in self.agents)
        wm_error_count=sum(int(a.get("world_model_error_count",0)) for a in self.agents)
        wm_correct=sum(int(a.get("world_model_next_correct",0)) for a in self.agents)
        wm_predictions=sum(int(a.get("world_model_prediction_count",0)) for a in self.agents)
        wm_surprise=sum(float(a.get("world_model_surprise_sum",0.0)) for a in self.agents)
        wm_surprise_count=sum(int(a.get("world_model_surprise_count",0)) for a in self.agents)
        area=max(1,len(self.resources))
        trail_coverage=sum(1 for i in range(area)
                             if self.resource_trails[i]>.04 or self.danger_trails[i]>.04)/area
        return {
            "alive":len(living),"total_created":self.total_created,"births":self.total_births,
            "deaths":self.total_deaths,"max_generation":max((a["generation"] for a in self.agents),default=0),
            "mean_energy":round(mean("energy"),3),"mean_curiosity":round(mean("curiosity"),4),
            "mean_learning_progress":round(mean("learning_progress"),4),"mean_prediction_error":round(prediction,4),
            "genomic_diversity":round(diversity,4),"goals_reached":self.goal_events,
            "research_topics_known":sum(len(a["known_topics"]) for a in self.agents),
            "resource_coverage":round(coverage,4),"roles":roles,
            "world_model_entries":wm_entries,
            "world_model_reward_mae":round(wm_error/max(1,wm_error_count),4),
            "world_model_next_state_accuracy":round(wm_correct/max(1,wm_predictions),4),
            "world_model_predictions":wm_predictions,
            "world_model_surprise":round(wm_surprise/max(1,wm_surprise_count),4),
            "resource_trail_mass":round(sum(self.resource_trails)/area,4),
            "danger_trail_mass":round(sum(self.danger_trails)/area,4),
            "collective_signal_coverage":round(trail_coverage,4),
            **self.latent_model.metrics(),
            # Outcome the planner optimises: mean reward per organism per tick lived, all organisms.
            "reward_per_agent_tick":round(sum(float(a["reward_total"]) for a in self.agents)
                                           /max(1,sum(int(a["age"]) for a in self.agents)),4),
            "latent_plan_adoption_rate":round(self.plan_stats["adopted"]/max(1,self.plan_stats["planned"]),4),
            "latent_plan_promotion_rate":round(self.plan_stats["promoted"]/max(1,self.plan_stats["planned"]),4),
        }

    def public_snapshot(self)->dict[str,Any]:
        n=self.config.grid_size
        keys=("id","parent_id","generation","role","x","y","energy","age","offspring","alive",
              "curiosity","learning_progress","error_ema","known_topics","facts","goals_reached","last_action")
        agents=[{k:a[k] for k in keys} for a in self.agents]
        return {"schema_version":4,"tick":self.tick,"saved_at":time.time(),"grid_size":n,
                "scenario":self.config.scenario,
                "resources":[[round(self.resources[y*n+x],3) for x in range(n)] for y in range(n)],
                "hazards":[[round(self.hazards[y*n+x],3) for x in range(n)] for y in range(n)],
                "resource_trails":[[round(self.resource_trails[y*n+x],3) for x in range(n)] for y in range(n)],
                "danger_trails":[[round(self.danger_trails[y*n+x],3) for x in range(n)] for y in range(n)],
                "beacons":self.beacons,"agents":agents,"metrics":self.metrics(),"recent_events":self.events[-30:]}

    def state_dict(self)->dict[str,Any]:
        return {"config":asdict(self.config),"tick":self.tick,"total_created":self.total_created,
                "total_births":self.total_births,"total_deaths":self.total_deaths,"goal_events":self.goal_events,
                "next_id":self.next_id,"event_seq":self.event_seq,"events":self.events[-120:],
                "resources":self.resources,"hazards":self.hazards,"global_visits":self.global_visits,
                "resource_trails":self.resource_trails,"danger_trails":self.danger_trails,
                "beacons":self.beacons,"agents":self.agents,"rng_state":self.rng.getstate(),
                "latent_world_model":self.latent_model.state_dict(),"plan_stats":dict(self.plan_stats)}

    def load_state_dict(self,state:dict[str,Any]):
        saved_config = state.get("config", {})
        current_config = asdict(self.config)
        mismatches = [key for key, value in current_config.items() if saved_config.get(key) != value]
        if mismatches:
            raise ValueError("checkpoint configuration mismatch: " + ", ".join(mismatches))
        for key in ("tick","total_created","total_births","total_deaths","goal_events","next_id",
                    "event_seq","events","resources","hazards","global_visits","resource_trails","danger_trails",
                    "beacons","agents"):
            if key in state: setattr(self,key,state[key])
        self.rng.setstate(tupleify(state["rng_state"]))
        if "latent_world_model" in state:
            self.latent_model.load_state_dict(state["latent_world_model"])
        if "plan_stats" in state:
            self.plan_stats={k:int(state["plan_stats"].get(k,0)) for k in ("planned","adopted","promoted")}

    def save_checkpoint(self,path:str|Path):
        dest=Path(path); dest.parent.mkdir(parents=True,exist_ok=True)
        temp=dest.with_suffix(".tmp")
        temp.write_text(json.dumps(self.state_dict(),separators=(",",":")),encoding="utf-8")
        temp.replace(dest)
