#!/usr/bin/env python3
"""Lifecycle runner: persistence and controls live outside the organism engine."""
from __future__ import annotations
import json, os, time
from pathlib import Path
from ea_life.engine import Simulation, config_from_env

STATE_DIR=Path(os.environ.get("STATE_DIR","/state"))
REQUEST_DIR=Path(os.environ.get("REQUEST_DIR","/exchange/requests"))
RESULT_DIR=Path(os.environ.get("RESULT_DIR","/exchange/results"))
CONTROL_DIR=Path(os.environ.get("CONTROL_DIR","/control"))
MAX_TICKS=int(os.environ.get("SIM_TICKS","100000"))
TICK_SLEEP=max(0.0,float(os.environ.get("TICK_SLEEP","0.04")))
SAVE_EVERY=max(1,int(os.environ.get("SAVE_EVERY","10")))
SEED=int(os.environ.get("SIM_SEED","20261015"))
RESUME=os.environ.get("RESUME","0")=="1"
MAX_EVENT_LOG_BYTES=max(65536,int(os.environ.get("MAX_EVENT_LOG_BYTES","2000000")))

def atomic_json(path:Path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(".tmp")
    temp.write_text(json.dumps(value,separators=(",",":")),encoding="utf-8")
    temp.replace(path)

def append_event_batch(path:Path, events:list[dict], max_bytes:int=MAX_EVENT_LOG_BYTES):
    """Append a batch of events while keeping the active log under a bounded size."""
    if not events or max_bytes<1:
        return
    lines=[(json.dumps(event,separators=(",",":"))+"\n").encode("utf-8") for event in events]
    payload=b"".join(lines)
    if len(payload)>max_bytes:
        retained=[]
        size=0
        for line in reversed(lines):
            if size+len(line)>max_bytes:
                break
            retained.append(line)
            size+=len(line)
        payload=b"".join(reversed(retained))
    if not payload:
        return
    path.parent.mkdir(parents=True,exist_ok=True)
    current=path.stat().st_size if path.exists() else 0
    if current+len(payload)>max_bytes:
        rotated=path.with_name("events.previous.jsonl")
        rotated.unlink(missing_ok=True)
        if path.exists():
            path.replace(rotated)
    with path.open("ab") as stream:
        stream.write(payload)


def reset_files():
    STATE_DIR.mkdir(parents=True,exist_ok=True)
    CONTROL_DIR.mkdir(parents=True,exist_ok=True)
    for name in ("snapshot.json","full_state.json","summary.json","events.jsonl","events.previous.jsonl"):
        (STATE_DIR/name).unlink(missing_ok=True)
    for flag in ("pause.flag","reset.flag","stop.flag"):
        (CONTROL_DIR/flag).unlink(missing_ok=True)
    REQUEST_DIR.mkdir(parents=True,exist_ok=True)
    for path in REQUEST_DIR.glob("req_*.json"):
        path.unlink(missing_ok=True)
    # Results are mounted read-only in this container. Unique request IDs make stale results harmless.

def main():
    for folder in (STATE_DIR,CONTROL_DIR,REQUEST_DIR,RESULT_DIR): folder.mkdir(parents=True,exist_ok=True)
    config=config_from_env(); checkpoint=STATE_DIR/"full_state.json"
    if not RESUME: reset_files()
    sim=Simulation(config=config,seed=SEED,request_dir=REQUEST_DIR,result_dir=RESULT_DIR)
    if RESUME and checkpoint.exists():
        try:
            sim.load_state_dict(json.loads(checkpoint.read_text(encoding="utf-8")))
            print("Resuming at tick",sim.tick,flush=True)
        except (OSError,ValueError,KeyError,TypeError) as exc:
            print("Checkpoint load failed; starting clean:",type(exc).__name__,flush=True)
            reset_files(); sim=Simulation(config=config,seed=SEED,request_dir=REQUEST_DIR,result_dir=RESULT_DIR)
    saved_event_seq=sim.event_seq
    started=time.time()
    print("EA-Life simulation started",json.dumps({"grid":config.grid_size,"scenario":config.scenario,
          "population":config.initial_population,"seed":SEED}),flush=True)
    while sim.tick<MAX_TICKS:
        if (CONTROL_DIR/"stop.flag").exists(): break
        if (CONTROL_DIR/"reset.flag").exists():
            (CONTROL_DIR/"reset.flag").unlink(missing_ok=True)
            reset_files()
            sim=Simulation(config=config,seed=SEED,request_dir=REQUEST_DIR,result_dir=RESULT_DIR)
            saved_event_seq=0; started=time.time()
            print("Experiment reset from dashboard",flush=True)
        if (CONTROL_DIR/"pause.flag").exists():
            time.sleep(.2)
            continue
        sim.step()
        if sim.tick%SAVE_EVERY==0:
            metrics=sim.metrics()
            sim._event("metrics","Metric sample",{"alive":metrics["alive"],"births":metrics["births"],
                      "mean_curiosity":metrics["mean_curiosity"],"mean_prediction_error":metrics["mean_prediction_error"],
                      "genomic_diversity":metrics["genomic_diversity"]})
            atomic_json(STATE_DIR/"snapshot.json",sim.public_snapshot())
            sim.save_checkpoint(checkpoint)
            new_events=[e for e in sim.events if e["seq"]>saved_event_seq]
            if new_events:
                append_event_batch(STATE_DIR/"events.jsonl",new_events)
                saved_event_seq=max(e["seq"] for e in new_events)
            if sim.tick%100==0:
                print("[tick %d] alive=%d births=%d curiosity=%.3f diversity=%.3f"%(
                    sim.tick,metrics["alive"],metrics["births"],metrics["mean_curiosity"],metrics["genomic_diversity"]),flush=True)
        if TICK_SLEEP: time.sleep(TICK_SLEEP)
    metrics=sim.metrics()
    atomic_json(STATE_DIR/"snapshot.json",sim.public_snapshot())
    sim.save_checkpoint(checkpoint)
    atomic_json(STATE_DIR/"summary.json",{"ticks":sim.tick,"runtime_seconds":round(time.time()-started,2),
                "summary":metrics,"seed":SEED,"scenario":config.scenario,"finished_at":time.time()})
    print("EXPERIMENT STOPPED",json.dumps(metrics),flush=True)

if __name__=="__main__": main()
