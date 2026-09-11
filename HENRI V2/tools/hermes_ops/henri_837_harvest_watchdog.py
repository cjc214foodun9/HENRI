#!/usr/bin/env python3
"""Phase 8.37 harvest watchdog — no-agent cron (Windows host).

Silent while the remote harvest runs (empty stdout = no delivery).
Prints a compact receipt when the sealed manifest appears with
record_count >= 10000. Exits 1 with the log tail if the process died
without a manifest (error alert).

Usage: python henri_837_harvest_watchdog.py
"""
import json
import subprocess
import sys

HOST = "root@107.206.71.138"
PORT = "45864"
MANIFEST = "/root/henri-837-bank/trajectories_harvest_837_20260819_manifest.json"
LOG = "/root/henri-837-bank.log"
MIN_RECORDS = 10000


def ssh(cmd: str, timeout: int = 60) -> tuple:
    r = subprocess.run(
        ["ssh", "-p", PORT, "-o", "ConnectTimeout=25", "-o", "BatchMode=yes",
         HOST, cmd],
        capture_output=True, text=True, timeout=timeout)
    return r.returncode, r.stdout.strip(), r.stderr.strip()


def main() -> int:
    rc, out, err = ssh(f"test -f {MANIFEST} && cat {MANIFEST} || echo NO_MANIFEST")
    if "NO_MANIFEST" in out:
        rc2, proc, err2 = ssh("pgrep -f cegis_self_play_sandbox | head -1 || echo DEAD")
        if proc == "DEAD":
            rc3, tail, err3 = ssh(f"tail -5 {LOG}")
            print(f"HENRI 8.37 HARVEST DIED without manifest. Log tail:\n{tail}")
            return 1
        # Still running: stay silent.
        return 0

    try:
        m = json.loads(out)
    except Exception:
        # Manifest mid-write or unparsable: stay silent this tick.
        return 0

    n = int(m.get("record_count", 0))
    if n >= MIN_RECORDS and m.get("dataset_digest"):
        print(
            f"HENRI 8.37 HARVEST COMPLETE | records={n} | "
            f"digest={m['dataset_digest'][:16]} | npz={m['npz_sha256'][:16]} | "
            f"envs={m.get('envs')} | actions={m.get('action_vocab')}"
        )
        return 0
    print(f"HENRI 8.37 MANIFEST_INCOMPLETE records={n} (< {MIN_RECORDS})")
    return 1


if __name__ == "__main__":
    sys.exit(main())
