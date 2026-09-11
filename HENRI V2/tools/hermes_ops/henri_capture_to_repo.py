#!/usr/bin/env python3
"""Cron wrapper: HENRI capture -> GitHub main. Runs the repo engine with the
canonical interpreter and streams its output; exit code = engine exit code.
Quiet on no-change (watchdog pattern)."""
import subprocess, sys

ENGINE = r"C:/Users/chan/Desktop/HENRI 7B SWARM/scripts/telemetry/henri_capture_to_repo.py"
r = subprocess.run([sys.executable, ENGINE], capture_output=True, text=True, timeout=900)
out = (r.stdout or "").strip()
if out:
    print(out)
if r.returncode != 0:
    print("CAPTURE_FAIL", (r.stderr or "")[:300])
sys.exit(r.returncode)
