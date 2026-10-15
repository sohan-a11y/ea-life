#!/usr/bin/env python3
"""Allowlisted read-only Wikipedia broker; no listener and no arbitrary URLs."""
from __future__ import annotations
import json, os, re, time, urllib.parse, urllib.request
from pathlib import Path

REQUEST_DIR=Path(os.environ.get("REQUEST_DIR","/exchange/requests"))
RESULT_DIR=Path(os.environ.get("RESULT_DIR","/exchange/results"))
POLL_SECONDS=float(os.environ.get("BROKER_POLL_SECONDS","0.5"))
MIN_INTERVAL=float(os.environ.get("BROKER_MIN_INTERVAL","2.0"))
MAX_REQUESTS=int(os.environ.get("BROKER_MAX_REQUESTS","100"))
MAX_RESPONSE_BYTES=1_000_000
TOPICS={
 "curiosity":"curiosity driven learning artificial intelligence",
 "cellular_automata":"cellular automata computer science",
 "evolution":"evolution mutation natural selection",
 "prediction":"prediction error machine learning",
 "energy":"energy ecology organisms",
 "memory":"memory in artificial intelligence",
 "navigation":"exploration and navigation",
}
ENDPOINT="https://en.wikipedia.org/w/api.php"

def fetch_topic(topic_id: str) -> list[dict]:
    if topic_id not in TOPICS: raise ValueError("topic not allowlisted")
    query=urllib.parse.urlencode({"action":"query","list":"search","srsearch":TOPICS[topic_id],"format":"json","utf8":"1","srlimit":"3"})
    req=urllib.request.Request(ENDPOINT+"?"+query,headers={"User-Agent":"EA-Life-Research-Prototype/0.1 (read-only educational experiment)"},method="GET")
    with urllib.request.urlopen(req,timeout=8) as response:
        if response.status!=200: raise RuntimeError("upstream status was not 200")
        body=response.read(MAX_RESPONSE_BYTES+1)
    if len(body)>MAX_RESPONSE_BYTES: raise ValueError("upstream response exceeded limit")
    payload=json.loads(body.decode("utf-8")); pages=[]
    for row in payload.get("query",{}).get("search",[])[:3]:
        snippet=re.sub(r"<[^>]*>"," ",str(row.get("snippet","")))
        pages.append({"title":str(row.get("title",""))[:120],"extract":" ".join(snippet.split())[:700],"source":"https://en.wikipedia.org/"})
    return pages

def main() -> None:
    REQUEST_DIR.mkdir(parents=True,exist_ok=True); RESULT_DIR.mkdir(parents=True,exist_ok=True)
    processed=0; last=0.0
    print("Broker ready: fixed HTTPS GET only; no listener.",flush=True)
    while processed<MAX_REQUESTS:
        for path in sorted(REQUEST_DIR.glob("req_*.json")):
            if processed>=MAX_REQUESTS: break
            item={}
            try:
                item=json.loads(path.read_text(encoding="utf-8"))
                rid=str(item.get("request_id","")); topic=item.get("topic_id")
                if not rid.isalnum() or len(rid)>64: raise ValueError("invalid request id")
                if topic not in TOPICS: raise ValueError("topic not allowlisted")
                wait=MIN_INTERVAL-(time.monotonic()-last)
                if wait>0: time.sleep(wait)
                pages=fetch_topic(topic); last=time.monotonic()
                result={"request_id":rid,"topic_id":topic,"ok":True,"pages":pages,"retrieved_at":time.time()}
            except Exception as exc:
                result={"request_id":str(item.get("request_id",""))[:64],"topic_id":item.get("topic_id"),"ok":False,"error":type(exc).__name__}
            finally:
                path.unlink(missing_ok=True)
            rid=str(result.get("request_id",""))
            if rid.isalnum() and len(rid)<=64:
                out=RESULT_DIR/("req_"+rid+".json"); tmp=out.with_suffix(".tmp")
                tmp.write_text(json.dumps(result),encoding="utf-8"); tmp.replace(out)
            processed+=1
        time.sleep(POLL_SECONDS)
    print("Broker request budget exhausted; stopping.",flush=True)

if __name__=="__main__": main()
