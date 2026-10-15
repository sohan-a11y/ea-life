#!/usr/bin/env python3
"""Visual dashboard/API. It cannot access genomes or make outbound requests."""
from __future__ import annotations
import json, os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

PORT=int(os.environ.get("DASHBOARD_PORT","8000"))
STATE_DIR=Path(os.environ.get("STATE_DIR","/state"))
CONTROL_DIR=Path(os.environ.get("CONTROL_DIR","/control"))
WEB_DIR=Path(__file__).parent/"web"
MIME={".html":"text/html; charset=utf-8",".css":"text/css; charset=utf-8",".js":"text/javascript; charset=utf-8"}

def read_json(path: Path, fallback):
    try: return json.loads(path.read_text(encoding="utf-8"))
    except (OSError,ValueError): return fallback

def build_state() -> dict:
    snap=read_json(STATE_DIR/"snapshot.json",{})
    agents=snap.get("agents",[]); events=[]
    try:
        with (STATE_DIR/"events.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                try: events.append(json.loads(line))
                except ValueError: pass
    except OSError: pass
    events=events[-100:]; living=[a for a in agents if a.get("alive")]; latest=events[-1] if events else {}
    keys=("id","parent_id","generation","x","y","energy","age","offspring","alive","last_curiosity","learning_progress","error_ema","known_topics","facts")
    safe=[{k:a.get(k) for k in keys} for a in agents]
    return {"ok":bool(snap),"tick":max(0,int(snap.get("next_tick",0))-1),"saved_at":snap.get("saved_at"),
      "grid_size":snap.get("grid_size",128),"world":snap.get("world",[]),"agents":safe,"events":events,
      "metrics":{"alive":len(living),"total":len(agents),"births":sum(int(a.get("offspring",0)) for a in agents),
      "max_generation":max((int(a.get("generation",0)) for a in agents),default=0),
      "mean_energy":round(sum(float(a.get("energy",0)) for a in living)/max(1,len(living)),2),
      "mean_curiosity":round(sum(float(a.get("last_curiosity",0)) for a in living)/max(1,len(living)),3),
      "mean_prediction_error":latest.get("mean_prediction_error",0),"known_topics":latest.get("known_topics",0)},
      "paused":(CONTROL_DIR/"pause.flag").exists(),"finished":(STATE_DIR/"summary.json").exists(),
      "summary":read_json(STATE_DIR/"summary.json",None)}

class Handler(BaseHTTPRequestHandler):
    server_version="EA-Life-Dashboard/0.1"
    def send_body(self,status: int,ctype: str,body: bytes):
        self.send_response(status); self.send_header("Content-Type",ctype); self.send_header("Content-Length",str(len(body)))
        self.send_header("X-Content-Type-Options","nosniff"); self.send_header("Referrer-Policy","no-referrer")
        self.send_header("Cache-Control","no-store")
        self.send_header("Content-Security-Policy","default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self' data:")
        self.end_headers(); self.wfile.write(body)
    def do_GET(self):
        route=urlparse(self.path).path
        if route=="/api/state":
            self.send_body(200,"application/json; charset=utf-8",json.dumps(build_state(),separators=(",",":")).encode()); return
        assets={"/":WEB_DIR/"index.html","/app.css":WEB_DIR/"app.css","/app.js":WEB_DIR/"app.js"}
        path=assets.get(route)
        if not path: self.send_body(404,"text/plain; charset=utf-8",b"Not found"); return
        try: self.send_body(200,MIME[path.suffix],path.read_bytes())
        except OSError: self.send_body(500,"text/plain; charset=utf-8",b"Dashboard asset unavailable")
    def do_POST(self):
        if urlparse(self.path).path!="/api/control": self.send_body(404,"application/json",b'{"ok":false}'); return
        origin,host=self.headers.get("Origin"),self.headers.get("Host","")
        if origin and urlparse(origin).netloc!=host: self.send_body(403,"application/json",b'{"ok":false,"error":"cross-origin denied"}'); return
        if self.headers.get("Content-Type","").split(";",1)[0].lower()!="application/json":
            self.send_body(415,"application/json",b'{"ok":false,"error":"JSON required"}'); return
        try:
            length=int(self.headers.get("Content-Length","0"))
            if not 1<=length<=128: raise ValueError
            data=json.loads(self.rfile.read(length)); action=data.get("action") if isinstance(data,dict) else None
            if action not in {"pause","resume"}: raise ValueError
        except (ValueError,json.JSONDecodeError):
            self.send_body(400,"application/json",b'{"ok":false,"error":"invalid command"}'); return
        CONTROL_DIR.mkdir(parents=True,exist_ok=True); flag=CONTROL_DIR/"pause.flag"
        if action=="pause": flag.touch(exist_ok=True)
        else: flag.unlink(missing_ok=True)
        self.send_body(200,"application/json",json.dumps({"ok":True,"action":action}).encode())
    def log_message(self,fmt,*args): print("[dashboard] "+fmt%args,flush=True)

if __name__=="__main__":
    CONTROL_DIR.mkdir(parents=True,exist_ok=True)
    print("EA-Life dashboard on port",PORT,flush=True)
    ThreadingHTTPServer(("0.0.0.0",PORT),Handler).serve_forever()
