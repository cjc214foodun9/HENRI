#!/usr/bin/env python
"""NotebookLM auth watchdog (no-agent cron).

Silent (empty stdout) while `nlm login --check` reports valid credentials.
Prints a compact alert with re-auth instructions when credentials are
stale, broken, or the CLI is missing. Exit 0 in all handled cases so the
watchdog itself never fails silently; a non-zero exit is reserved for
unexpected internal errors.

Path notes (Windows/MSYS):
- The cron scheduler resolves the script by basename under the Hermes
  scripts dir and runs .py via Python, which handles Windows paths
  natively. Keep this file on disk as a .py, NOT a .sh: bash strips the
  backslashes in C:\\... absolute paths and fails with exit 127.
"""
import shutil
import subprocess
import sys

NLM = r"C:\Users\chan\AppData\Local\hermes\hermes-agent\venv\Scripts\nlm.exe"


def main() -> int:
    if not shutil.which(NLM) and not __import__("os").path.exists(NLM):
        print("NOTEBOOKLM_AUTH_WATCHDOG: nlm CLI not found at %s "
              "(reinstall notebooklm-mcp-cli)" % NLM)
        return 0
    try:
        proc = subprocess.run(
            [NLM, "login", "--check"],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except subprocess.TimeoutExpired:
        print("NOTEBOOKLM_AUTH_WATCHDOG: nlm login --check timed out "
              "(60s). Re-auth: run 'nlm login'.")
        return 0
    out = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode == 0 and "valid" in out.lower():
        return 0  # healthy: silent, nothing delivered
    print("NOTEBOOKLM_AUTH_WATCHDOG: NotebookLM auth is STALE or broken "
          "(exit=%s). Re-auth: run 'nlm login' (opens Chrome, CDP-based)."
          % proc.returncode)
    for line in out.splitlines()[:8]:
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
