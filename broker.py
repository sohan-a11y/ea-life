#!/usr/bin/env python3
"""Fixed-topic Wikipedia broker. The organism cannot provide a URL or free-form query."""
from __future__ import annotations
import html, json, os, re, time, urllib.error, urllib.parse, urllib.request
from pathlib import Path

REQUEST_DIR=Path(os.environ.get("REQUEST_DIR","/exchange/requests"))
RESULT_DIR=Path(os.environ.get("RESULT_DIR","/exchange/results"))
POLL_SECONDS=float(os.environ.get("BROKER_POLL_SECONDS","0.5"))
MIN_INTERVAL=float(os.environ.get("BROKER_MIN_INTERVAL","1.5"))
MAX_REQUESTS=int(os.environ.get("BROKER_MAX_REQUESTS","200"))
MAX_BYTES=800_000
MAX_REQUEST_BYTES=4096
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

class AllowlistedRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Allow redirects only to the fixed HTTPS Wikipedia API host."""
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        target=urllib.parse.urlparse(urllib.parse.urljoin(req.full_url,newurl))
        if (target.scheme!="https" or target.hostname!="en.wikipedia.org"
                or target.username is not None or target.password is not None
                or target.port not in (None,443)):
            raise urllib.error.HTTPError(target.geturl(),code,"redirect destination blocked",headers,fp)
        return super().redirect_request(req,fp,code,msg,headers,newurl)

def retrieve(topic_id:str)->list[dict]:
    if topic_id not in TOPICS: raise ValueError("topic not allowlisted")
    query=urllib.parse.urlencode({"action":"query","list":"search","srsearch":TOPICS[topic_id],
                                  "format":"json","utf8":"1","srlimit":"3"})
    request=urllib.request.Request(ENDPOINT+"?"+query,
        headers={"User-Agent":"EA-Life-Research-Broker/1.0 (educational read-only prototype)"},method="GET")
    opener=urllib.request.build_opener(AllowlistedRedirectHandler())
    with opener.open(request,timeout=8) as response:
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

def parse_request(path:Path)->tuple[dict,str,str]:
    """Read and validate one bounded request from the shared, untrusted queue."""
    raw=path.read_bytes()
    if len(raw)>MAX_REQUEST_BYTES:
        raise ValueError("request too large")
    item=json.loads(raw.decode("utf-8"))
    if not isinstance(item,dict):
        raise ValueError("request must be a JSON object")
    rid=str(item.get("request_id",""))
    topic=item.get("topic_id")
    if not re.fullmatch(r"[a-f0-9]{32}",rid):
        raise ValueError("invalid request id")
    if path.name!="req_"+rid+".json":
        raise ValueError("request filename/id mismatch")
    if topic not in TOPICS:
        raise ValueError("topic not allowlisted")
    return item,rid,topic


def publish_result(result:dict) -> None:
    rid=str(result.get("request_id",""))
    if not re.fullmatch(r"[a-f0-9]{32}",rid):
        return
    output=RESULT_DIR/("req_"+rid+".json")
    temp=output.with_suffix(".tmp")
    temp.write_text(json.dumps(result),encoding="utf-8")
    temp.replace(output)


def main():
    REQUEST_DIR.mkdir(parents=True,exist_ok=True); RESULT_DIR.mkdir(parents=True,exist_ok=True)
    completed=0; last_attempt=0.0; last_status=time.monotonic()
    write_status("active",completed)
    print("Broker: allowlisted topic IDs, HTTPS GET only, no listener.",flush=True)
    while completed<MAX_REQUESTS:
        for path in sorted(REQUEST_DIR.glob("req_*.json")):
            if completed>=MAX_REQUESTS: break
            item={}
            try:
                item,rid,topic=parse_request(path)
                wait=MIN_INTERVAL-(time.monotonic()-last_attempt)
                if wait>0: time.sleep(wait)
                # Record the attempt before I/O so failures still obey rate limiting.
                last_attempt=time.monotonic()
                pages=retrieve(topic)
                result={"request_id":rid,"topic_id":topic,"ok":True,"pages":pages,"retrieved_at":time.time()}
            except Exception as exc:
                safe_item=item if isinstance(item,dict) else {}
                candidate=str(safe_item.get("request_id",""))
                safe_rid=(candidate if re.fullmatch(r"[a-f0-9]{32}",candidate)
                          and path.name=="req_"+candidate+".json" else "")
                result={"request_id":safe_rid,"topic_id":safe_item.get("topic_id"),
                        "ok":False,"error":type(exc).__name__}
            finally:
                path.unlink(missing_ok=True)
            publish_result(result)
            completed+=1
        time.sleep(max(0.05,POLL_SECONDS))
        if time.monotonic()-last_status>=10:
            write_status("active",completed)
            last_status=time.monotonic()
    write_status("exhausted",completed)
    # Return an explicit failure result for queued work so agents can clear pending IDs.
    for path in sorted(REQUEST_DIR.glob("req_*.json")):
        item={}
        try:
            item,rid,topic=parse_request(path)
            publish_result({"request_id":rid,"topic_id":topic,"ok":False,
                            "error":"RequestBudgetExhausted"})
        except Exception:
            pass
        finally:
            path.unlink(missing_ok=True)
    print("Broker request budget reached; marked exhausted.",flush=True)


if __name__=="__main__": main()
