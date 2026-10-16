#!/usr/bin/env python3
"""Run EA-Life on this computer without Docker: the simulator plus the dashboard at http://127.0.0.1:8000.

The research broker (the only component that uses the internet) is not started, so nothing leaves this
computer. The dashboard listens on 127.0.0.1 only. Requires Python 3.10 or newer; no packages to install.

    python run_local.py                    # opens the dashboard in your browser
    python run_local.py --no-imagination   # faster: latent model on, planning off
    python run_local.py --resume           # continue the previous run from its checkpoint

Stop with Ctrl+C. Run data is kept in ./local-data (delete that folder to start fresh).
"""
from __future__ import annotations
import argparse, os, subprocess, sys, threading, time, webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> int:
    ap = argparse.ArgumentParser(description="Run EA-Life locally without Docker.")
    ap.add_argument("--data", default=str(ROOT / "local-data"), help="folder for state, checkpoints and events")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--grid", type=int, default=64)
    ap.add_argument("--population", type=int, default=12)
    ap.add_argument("--max-population", type=int, default=40)
    ap.add_argument("--max-organisms", type=int, default=500)
    ap.add_argument("--scenario", default="mixed")
    ap.add_argument("--ticks", type=int, default=100000)
    ap.add_argument("--tick-sleep", type=float, default=0.04, help="pause between ticks, in seconds")
    ap.add_argument("--seed", type=int, default=20261015)
    ap.add_argument("--no-imagination", action="store_true", help="turn off planning (much faster)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    data = Path(args.data)
    dirs = {name: data / name for name in ("state", "requests", "results", "control")}
    for folder in dirs.values():
        folder.mkdir(parents=True, exist_ok=True)
    for flag in ("stop.flag", "pause.flag", "reset.flag"):
        (dirs["control"] / flag).unlink(missing_ok=True)

    env = dict(os.environ, STATE_DIR=str(dirs["state"]), REQUEST_DIR=str(dirs["requests"]), RESULT_DIR=str(dirs["results"]),
               CONTROL_DIR=str(dirs["control"]), SIM_GRID=str(args.grid), SIM_POPULATION=str(args.population),
               SIM_MAX_POPULATION=str(args.max_population), SIM_MAX_ORGANISMS=str(args.max_organisms), SIM_SCENARIO=args.scenario,
               SIM_TICKS=str(args.ticks), TICK_SLEEP=str(args.tick_sleep), SAVE_EVERY="5", SIM_SEED=str(args.seed),
               SIM_IMAGINATION_ENABLED="0" if args.no_imagination else "1", RESUME="1" if args.resume else "0",
               PYTHONPATH=str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    os.environ.update(STATE_DIR=env["STATE_DIR"], CONTROL_DIR=env["CONTROL_DIR"], DASHBOARD_PORT=str(args.port))
    sys.path.insert(0, str(ROOT))
    import dashboard  # reads STATE_DIR and CONTROL_DIR at import

    server = None
    for port in range(args.port, args.port + 20):          # fall back if the port is taken or reserved
        try:
            server = dashboard.ThreadingHTTPServer(("127.0.0.1", port), dashboard.Handler)
            break
        except OSError as exc:
            print("Port %d is not available (%s); trying %d." % (port, type(exc).__name__, port + 1), flush=True)
    if server is None:
        print("No free port found from %d to %d; choose one with --port." % (args.port, args.port + 19), flush=True)
        return 1
    port = server.server_address[1]
    sim = subprocess.Popen([sys.executable, str(ROOT / "runner.py")], env=env, cwd=str(ROOT))
    url = "http://127.0.0.1:%d/" % port
    print("EA-Life is running. Open %s  (Ctrl+C to stop; no internet is used)" % url, flush=True)
    if not args.no_browser:
        threading.Timer(1.5, webbrowser.open, (url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        (dirs["control"] / "stop.flag").touch()
        try:
            sim.wait(timeout=15)
        except subprocess.TimeoutExpired:
            sim.terminate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
