"""Apply the v0.4.1 Windows worker-compatibility fix and retain diagnostics.

The release archive is kept byte-identical as a v0.4.0 baseline; this script patches
only the isolated temporary extraction used by Windows CI. The final successful CI
job also archives the patched v0.4.1 project as a downloadable build artifact.
"""
from pathlib import Path
import sys

project = Path(sys.argv[1]).resolve()
worker = project / "organism" / "worker.py"
supervisor = project / "organism" / "runtime_supervisor.py"
tests = project / "tests" / "test_runtime_supervisor.py"

worker_src = worker.read_text(encoding="utf-8")
worker_src = worker_src.replace(
    "import os\n",
    "import os\nimport sys\n",
    1,
)
worker_src = worker_src.replace(
    "    state = \"starting\"\n    try:\n        store = OrganismStore(str(home_path))\n",
    """    state = "starting"
    try:
        _atomic_write(status_path, {
            "organism_id": organism_id, "state": "booting", "pid": os.getpid(),
            "python": sys.version.split()[0], "started_at": time.time(),
        })
        print(f"ALIFE_WORKER_BOOT pid={os.getpid()} python={sys.version.split()[0]} home={home_path}", flush=True)
        store = OrganismStore(str(home_path))
""",
    1,
)
old_check = "            if hb.get(\"alive\") is False or not fresh or not _parent_alive(supervisor_pid):\n                state = \"orphaned\"\n                break\n"
new_check = """            # The supervisor heartbeat is authoritative. os.kill(pid, 0) is not
            # a portable parent-liveness probe on all Windows/Python combinations.
            if hb.get("alive") is False or not fresh:
                state = "orphaned"
                store.set_meta("runtime_exit_reason", {
                    "reason": "supervisor_heartbeat_lost", "heartbeat_fresh": fresh,
                    "heartbeat_alive": hb.get("alive"), "timestamp": now,
                })
                store.db.commit()
                print(f"ALIFE_WORKER_ORPHAN pid={os.getpid()} heartbeat_fresh={fresh} heartbeat_alive={hb.get('alive')}", flush=True)
                break
"""
if old_check not in worker_src:
    raise SystemExit("Could not locate OS-dependent parent PID check in worker.py")
worker_src = worker_src.replace(old_check, new_check, 1)
worker_src = worker_src.replace(
    '        _atomic_write(status_path, {"organism_id": organism_id, "state": state, "pid": os.getpid(), "last_heartbeat": time.time()})\n        return 0\n',
    '''        _atomic_write(status_path, {
            "organism_id": organism_id, "state": state, "pid": os.getpid(),
            "last_heartbeat": time.time(), "python": sys.version.split()[0],
            "exit_reason": store.get_meta("runtime_exit_reason", None) if store else None,
        })
        print(f"ALIFE_WORKER_EXIT pid={os.getpid()} state={state}", flush=True)
        return 0
''',
    1,
)
if 'ALIFE_WORKER_BOOT' not in worker_src or '_parent_alive(supervisor_pid)' in worker_src:
    raise SystemExit("Worker compatibility patch verification failed")
worker.write_text(worker_src, encoding="utf-8")

src = supervisor.read_text(encoding="utf-8")
start_marker = "        if events and len(events) >= self._restart_limit:\n"
start = src.find(start_marker)
if start < 0:
    raise SystemExit("Could not find crash-loop branch in runtime_supervisor.py")
end_marker = "            return None\n"
end = src.find(end_marker, start)
if end < 0:
    raise SystemExit("Could not find crash-loop return in runtime_supervisor.py")
end += len(end_marker)
replacement = '''        if events and len(events) >= self._restart_limit:
            # Preserve the last worker diagnostics instead of hiding them in crash-loop.
            previous = _read_json(self._status_path(row), {})
            status = {
                "organism_id": organism_id,
                "state": "crash-loop",
                "desired_state": "failed",
                "timestamp": now,
                "error": f"Restart limit reached ({self._restart_limit} within {int(self._restart_window)} seconds). Stop and explicitly start to retry.",
                "last_worker_state": previous.get("state"),
                "last_worker_error": previous.get("last_error") or previous.get("error"),
                "last_worker_traceback": previous.get("traceback"),
                "last_worker_pid": previous.get("pid"),
                "last_worker_exit_reason": previous.get("exit_reason"),
            }
            _atomic_json(self._status_path(row), status)
            self._desired[organism_id] = "failed"
            self._save_desired()
            return None
'''
supervisor.write_text(src[:start] + replacement + src[end:], encoding="utf-8")

test_src = tests.read_text(encoding="utf-8")
fail_line = '        self.fail(f"{message}; last snapshot={target if \'target\' in locals() else latest}")\n'
diagnostic = '''        snapshot = target if 'target' in locals() else latest
        if isinstance(snapshot, dict):
            worker_home = snapshot.get("home")
            if worker_home:
                for name in ("runtime_status.json", "runtime_worker_process.log"):
                    path = Path(worker_home) / name
                    try:
                        content = path.read_text(encoding="utf-8")
                        print(f"RUNTIME_DIAGNOSTIC_{name.upper()}=" + content[-12000:])
                    except OSError:
                        pass
        self.fail(f"{message}; last snapshot={snapshot}")
'''
if fail_line in test_src:
    test_src = test_src.replace(fail_line, diagnostic + "\n", 1)
tests.write_text(test_src, encoding="utf-8")
# Stamp the temporary project with its corrected version; the original uploaded baseline ZIP
# remains unchanged until the fixed build is confirmed by Windows CI.
(project / "VERSION").write_text("0.4.1\\n", encoding="utf-8")
readme = project / "README.md"
readme_text = readme.read_text(encoding="utf-8")
readme.write_text(readme_text.replace("v0.4.0", "v0.4.1", 1), encoding="utf-8")
changelog = project / "CHANGELOG.md"
previous_log = changelog.read_text(encoding="utf-8") if changelog.exists() else "# Changelog\\n"
changelog.write_text(
    "# Changelog\\n\\n## v0.4.1 — Windows child-runtime compatibility\\n"
    "- Use the supervisor heartbeat as the authoritative cross-platform parent-liveness check.\\n"
    "- Emit worker boot/exit diagnostics and preserve final worker state across crash-loop protection.\\n\\n"
    + previous_log.removeprefix("# Changelog\\n").lstrip(),
    encoding="utf-8",
)
print("Applied v0.4.1 Windows heartbeat fix and failure diagnostics to CI extraction.")
