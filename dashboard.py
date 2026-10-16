#!/usr/bin/env python3
"""Read-only dashboard API with pause/resume/reset controls."""
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
MAX_BODY=128

def read_json(path:Path,fallback):
    try: return json.loads(path.read_text(encoding="utf-8"))
    except (OSError,json.JSONDecodeError): return fallback

def build_state()->dict:
    snap=read_json(STATE_DIR/"snapshot.json",{})
    events=[]
    try:
        event_path=STATE_DIR/"events.jsonl"
        # Bound both I/O and parsing cost even after a long-running experiment.
        with event_path.open("rb") as stream:
            stream.seek(0,2)
            end=stream.tell()
            stream.seek(max(0,end-262144))
            if stream.tell():
                stream.readline()  # discard a potentially partial first line
            lines=stream.readlines()[-500:]
        for line in lines:
            try: events.append(json.loads(line.decode("utf-8")))
            except (UnicodeDecodeError,json.JSONDecodeError): continue
    except OSError: pass
    summary=read_json(STATE_DIR/"summary.json",None)
    return {"ok":bool(snap),"schema_version":snap.get("schema_version",2),
      "tick":snap.get("tick",0),"saved_at":snap.get("saved_at"),"grid_size":snap.get("grid_size",96),
      "scenario":snap.get("scenario","mixed"),"resources":snap.get("resources",[]),"hazards":snap.get("hazards",[]),
      "resource_trails":snap.get("resource_trails",[]),"danger_trails":snap.get("danger_trails",[]),
      "beacons":snap.get("beacons",[]),"agents":snap.get("agents",[]),"metrics":snap.get("metrics",{}),
      "events":events[-100:],"paused":(CONTROL_DIR/"pause.flag").exists(),"finished":summary is not None,
      "summary":summary}

class Handler(BaseHTTPRequestHandler):
    server_version="EA-Life-Dashboard"
    def _send(self,status:int,content_type:str,body:bytes):
        self.send_response(status); self.send_header("Content-Type",content_type)
        self.send_header("Content-Length",str(len(body))); self.send_header("X-Content-Type-Options","nosniff")
        self.send_header("Referrer-Policy","no-referrer"); self.send_header("Cache-Control","no-store")
        self.send_header("Content-Security-Policy","default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; connect-src 'self'; img-src 'self' data:")
        self.end_headers(); self.wfile.write(body)
    def do_GET(self):
        route=urlparse(self.path).path
        if route=="/api/state":
            self._send(200,"application/json; charset=utf-8",json.dumps(build_state(),separators=(",",":")).encode()); return
        assets={"/":WEB_DIR/"index.html","/app.css":WEB_DIR/"app.css","/app.js":WEB_DIR/"app.js"}
        path=assets.get(route)
        if not path: self._send(404,"text/plain; charset=utf-8",b"Not found"); return
        try: body=path.read_bytes()
        except OSError: self._send(500,"text/plain; charset=utf-8",b"Asset unavailable"); return
        self._send(200,MIME.get(path.suffix,"application/octet-stream"),body)
    def do_POST(self):
        if urlparse(self.path).path!="/api/control":
            self._send(404,"application/json",b'{"ok":false}'); return
        origin,host=self.headers.get("Origin"),self.headers.get("Host","")
        if origin and urlparse(origin).netloc!=host:
            self._send(403,"application/json",b'{"ok":false,"error":"cross-origin denied"}'); return
        if self.headers.get("Content-Type","").split(";",1)[0].strip().lower()!="application/json":
            self._send(415,"application/json",b'{"ok":false,"error":"JSON required"}'); return
        try:
            length=int(self.headers.get("Content-Length","0"))
            if not 1<=length<=MAX_BODY: raise ValueError
            payload=json.loads(self.rfile.read(length)); action=payload.get("action") if isinstance(payload,dict) else None
            if action not in {"pause","resume","reset"}: raise ValueError
        except (ValueError,json.JSONDecodeError):
            self._send(400,"application/json",b'{"ok":false,"error":"invalid command"}'); return
        CONTROL_DIR.mkdir(parents=True,exist_ok=True); pause=CONTROL_DIR/"pause.flag"
        if action=="pause": pause.touch(exist_ok=True)
        elif action=="resume": pause.unlink(missing_ok=True)
        else:
            (CONTROL_DIR/"reset.flag").touch(exist_ok=True); pause.unlink(missing_ok=True)
        self._send(200,"application/json",json.dumps({"ok":True,"action":action}).encode())
    def log_message(self,fmt,*args): print("[dashboard] "+fmt%args,flush=True)

if __name__=="__main__":
    CONTROL_DIR.mkdir(parents=True,exist_ok=True)
    print("EA-Life dashboard on port",PORT,flush=True)
    ThreadingHTTPServer(("0.0.0.0",PORT),Handler).serve_forever()
