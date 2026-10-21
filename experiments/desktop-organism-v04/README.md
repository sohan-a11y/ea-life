# Artificial Life Desktop v0.4.1 — Windows-validated desktop organism

This folder intentionally isolates the desktop organism from the EA-Life simulator.

## Release bundle
- `artificial_life_desktop_v4_1.zip` — generated from the patched source after the native Windows CI matrix passed. It contains a top-level `project/` folder.
- `artificial_life_desktop_v4.zip` — original v0.4.0 baseline used to reproduce the Windows/Python 3.12 worker issue.
- `VALIDATION.md` — exact verification status and limits.
- `apply_windows_diagnostics.py` — CI-only overlay that applies the heartbeat-liveness fix, adds worker boot/exit diagnostics, and updates the temporary project to v0.4.1.
- Repository-level workflow: [desktop-organism-v04-windows.yml](../../../.github/workflows/desktop-organism-v04-windows.yml).

## Windows validation
The v0.4.1 overlay passed the automated suite and the native Tk desktop + child-process lifecycle smoke test on GitHub-hosted `windows-latest` runners using Python 3.10, 3.11, and 3.12. See the [successful workflow run](https://github.com/sohan-a11y/ea-life/actions/runs/39546342539).

## Local Windows check
Extract the ZIP, open PowerShell in its `project` directory, and run:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\validate_windows.ps1
```

The native desktop smoke test opens the Tk pet and console, starts a separate child-organism process, checks pause/resume/stop, and closes the app. It needs no LLM or model download.

## Scope and caveats
This validates the tested paths in a hosted Windows environment, not every user's Windows version, display scaling, multiple-monitor layout, startup configuration, or security property. Child runtimes are background processes; each child does not yet have its own visible pet overlay. The work is not evidence of subjective consciousness.
