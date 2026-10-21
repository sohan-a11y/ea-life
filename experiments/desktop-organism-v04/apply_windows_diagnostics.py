"""Temporary Windows CI diagnostic overlay for v0.4.0 child-process failures.

This preserves the last worker exception/traceback before crash-loop status overwrites
it, and prints those details if the lifecycle test times out. Remove this overlay
once a targeted runtime fix has been merged into the release bundle.
"""
from pathlib import Path
import sys

project = Path(sys.argv[1]).resolve()
supervisor = project / "organism" / "runtime_supervisor.py"
tests = project / "tests" / "test_runtime_supervisor.py"

src = supervisor.read_text(encoding="utf-8")
start_marker = "        if events and len(events) >= self._restart_limit:\n"
start = src.find(start_marker)
if start < 0:
    raise SystemExit("Diagnostic patch could not find crash-loop branch in runtime_supervisor.py")
end_marker = "            return None\n"
end = src.find(end_marker, start)
if end < 0:
    raise SystemExit("Diagnostic patch could not find crash-loop return")
end += len(end_marker)
replacement = '''        if events and len(events) >= self._restart_limit:
            # Preserve the last worker exception before crash-loop status replaces it.
            previous = _read_json(self._status_path(row), {})
            worker_error = previous.get("last_error") or previous.get("error")
            worker_traceback = previous.get("traceback") or previous.get("last_worker_traceback")
            status = {
                "organism_id": organism_id,
                "state": "crash-loop",
                "desired_state": "failed",
                "timestamp": now,
                "error": f"Restart limit reached ({self._restart_limit} within {int(self._restart_window)} seconds). Stop and explicitly start to retry.",
                "last_error": worker_error,
                "last_worker_traceback": worker_traceback,
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
if fail_line not in test_src:
    raise SystemExit("Diagnostic patch could not find lifecycle timeout assertion")
tests.write_text(test_src.replace(fail_line, diagnostic + "\n", 1), encoding="utf-8")
print("Applied child-process failure diagnostics to the extracted release.")
