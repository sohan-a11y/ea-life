# Validation status

## Before native Windows CI
- Linux unit tests: 28 passed in the build environment.
- Linux Tk desktop + child-process smoke test: passed under Xvfb.
- Native Windows run: pending until the repository's Windows GitHub Actions jobs complete.

## What Windows CI checks
1. Extracts the exact packaged v0.4.0 ZIP.
2. Installs dependencies.
3. Compiles Python source.
4. Runs the complete unittest suite.
5. Starts the native Tk desktop and console, then tests a child worker's separate PID and start/pause/resume/stop cycle.

A Windows job result must be inspected before reporting native Windows success. If the UI smoke fails because the hosted runner cannot create a Tk window, that is a real failure to diagnose rather than a passing test.

## Still outside the current verification
- Manual use on Windows 10/11 with real display scaling and multiple monitors.
- Packaging as a signed installer.
- Hard OS-enforced CPU/RAM limits.
- Independent visible overlays for every child organism.
- A claim of subjective consciousness (not established by these tests).
