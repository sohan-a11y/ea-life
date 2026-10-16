#!/usr/bin/env python3
"""Fixed-topic Wikipedia broker. The organism cannot provide a URL or free-form query."""
from __future__ import annotations
import html, json, os, re, time, urllib.parse, urllib.request
from pathlib import Path

REQUEST_DIR=Path(os.environ.get("REQUEST_DIR","/exchange/requests"))
RESULT_DIR=Path(os.environ.get("RESULT_DIR","/exchange/results"))
POLL_SECONDS=float(os.environ.get("BROKER_POLL_SECONDS","0.5"))
MIN_INTERVAL=float(os.environ.get("BROKER_MIN_INTERVAL","1.5"))
MAX_REQUESTS=int(os.environ.get("BROKER_MAX_REQUESTS","200"))
MAX_BYTES=800_000
TOPICS={
 "curiosity":"curiosity and intrinsic motivation in learning",
 "cellular_automata":"cellular automata artificial life",
 "evolution":"evolution mutation natural selection",
 "prediction":"prediction error and reinforcement learning",
 "memory":"memory models in artificial intelligence",
 "navigation":"navigation and spatial learning",
 "ecology":"ecology resource competition",
 "collectives":"collective behavior swarm intelligence",
}
ENDPOINT="https://en.wikipedia.org/w/api.php"

def retrieve(topic_id:str)->list[dict]:
    if topic_id not in TOPICS: raise ValueError("topic not allowlisted")
    query=urllib.parse.urlencode({"action":"query","list":"search","srsearch":TOPICS[topic_id],
                                  "format":"json","utf8":"1","srlimit":"3"})
    request=urllib.request.Request(ENDPOINT+"?"+query,
        headers={"User-Agent":"EA-Life-Research-Broker/2.0 (educational read-only prototype)"},method="GET")
    with urllib.request.urlopen(request,timeout=8) as response:
        if response.status!=200: raise RuntimeError("upstream status was not 200")
        body=response.read(MAX_BYTES+1)
    if len(body)>MAX_BYTES: raise ValueError("response size exceeded")
    payload=json.loads(body.decode("utf-8")); pages=[]
    for item in payload.get("query",{}).get("search",[])[:3]:
        snippet=re.sub(r"<[^>]+>"," ",str(item.get("snippet","")))
        pages.append({"title":str(item.get("title",""))[:120],
                      "extract":" ".join(html.unescape(snippet).split())[:700],
                      "source":"https://en.wikipedia.org/"})
    return pages

def write_status(state: str, completed: int) -> None:
    """Publish broker availability for the isolated simulator to observe."""
    RESULT_DIR.mkdir(parents=True,exist_ok=True)
    path=RESULT_DIR/"broker_status.json"
    temp=path.with_suffix(".tmp")
    temp.write_text(json.dumps({"state":state,"completed_requests":completed,
                                "max_requests":MAX_REQUESTS,"updated_at":time.time()}),
                    encoding="utf-8")
    temp.replace(path)

def main():
    REQUEST_DIR.mkdir(parents=True,exist_ok=True); RESULT_DIR.mkdir(parents=True,exist_ok=True)
    completed=0; last=0.0
    write_status("active",completed)
    print("Broker: allowlisted topic IDs, HTTPS GET only, no listener.",flush=True)
    while completed<MAX_REQUESTS:
        for path in sorted(REQUEST_DIR.glob("req_*.json")):
            if completed>=MAX_REQUESTS: break
            item={}
            try:
                item=json.loads(path.read_text(encoding="utf-8"))
                rid=str(item.get("request_id","")); topic=item.get("topic_id")
                if not re.fullmatch(r"[a-f0-9]{32}",rid): raise ValueError("invalid request id")
                if topic not in TOPICS: raise ValueError("topic not allowlisted")
                wait=MIN_INTERVAL-(time.monotonic()-last)
                if wait>0: time.sleep(wait)
                pages=retrieve(topic); last=time.monotonic()
                result={"request_id":rid,"topic_id":topic,"ok":True,"pages":pages,"retrieved_at":time.time()}
            except Exception as exc:
                result={"request_id":str(item.get("request_id",""))[:32],"topic_id":item.get("topic_id"),
                        "ok":False,"error":type(exc).__name__}
            finally:
                path.unlink(missing_ok=True)
            rid=str(result.get("request_id",""))
            if re.fullmatch(r"[a-f0-9]{32}",rid):
                output=RESULT_DIR/("req_"+rid+".json"); temp=output.with_suffix(".tmp")
                temp.write_text(json.dumps(result),encoding="utf-8"); temp.replace(output)
            completed+=1
        time.sleep(POLL_SECONDS)
    write_status("exhausted",completed)
    print("Broker request budget reached; marked exhausted.",flush=True)

if __name__=="__main__": main()
