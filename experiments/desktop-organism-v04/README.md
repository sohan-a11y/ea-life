# Artificial Life Desktop v0.4.0 — Windows validation bundle

This folder intentionally isolates the desktop organism from the EA-Life simulator.

## Files
- `artificial_life_desktop_v4.zip` — exact release bundle. Extract it; the archive contains a top-level `project/` directory.
- `VALIDATION.md` — what has and has not been tested.
- The repository-level workflow `.github/workflows/desktop-organism-v04-windows.yml` extracts this bundle and tests it on hosted Windows runners with Python 3.10, 3.11, and 3.12.

## Local Windows check
Extract the ZIP, open PowerShell in its `project` directory, and run:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\validate_windows.ps1
```

The native desktop smoke test opens the Tk desktop pet and its console, starts a separate child-organism process, checks pause/resume/stop, and closes the app. It needs no LLM or model download.

## Scope and caveats
Tests cover defined software paths and are not proof of consciousness or security in all situations. Child runtimes are background processes; individual child pet overlays are not yet implemented. Hosted-runner success does not replace testing on the target user's Windows version, display scaling, and multi-monitor setup.

The source archive's contents are unchanged. SHA-1 Git blob ID of the ZIP: `d488c801b0edb66998ac5fca301c6d231c64cf0b`.
