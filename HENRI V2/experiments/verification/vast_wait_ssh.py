#!/usr/bin/env python3
"""Wait for instance readiness, then run the SSH gate. Bounded, fail-closed.

MEASURED CONTEXT
    First poll: cur_state=running but actual_status=None, and SSH returned
    `Connection refused`. Both are EXPECTED before the container is built --
    `--image` triggers a full image pull on the host (the tagged image is ~8.5 GB).
    Refused-during-loading is NOT an SSH fault; it must not be diagnosed as one.

    The endpoint is re-read on EVERY poll: Vast reassigns ssh_host/ssh_port.
"""
from __future__ import annotations
import json, subprocess, sys, time

IID = int(sys.argv[1]) if len(sys.argv) > 1 else 52161444
DEADLINE_S = int(sys.argv[2]) if len(sys.argv) > 2 else 600
KEY = "id_ed25519"

def show():
    r = subprocess.run(["vastai", "show", "instance", str(IID), "--raw"],
                       capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        return None
    return json.loads(r.stdout)

def gate(host, port):
    import os
    args = ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new",
            "-o", "ConnectTimeout=20", "-o", "IdentitiesOnly=yes"]
    k = os.path.expanduser(f"~/.ssh/{KEY}")
    if os.path.exists(k):
        args += ["-i", k]
    args += ["-p", str(port), f"root@{host}",
             'echo SSH_OK; hostname; nvidia-smi --query-gpu=name,memory.total,compute_cap '
             '--format=csv,noheader; python3 -c "import torch,sys;print(\'torch\','
             'torch.__version__,\'cuda_avail\',torch.cuda.is_available())" 2>&1 | tail -2']
    return subprocess.run(args, capture_output=True, text=True, timeout=90)

start = time.time()
last = None
while time.time() - start < DEADLINE_S:
    d = show()
    if d is None:
        print("poll: show failed"); time.sleep(15); continue
    st = d.get("actual_status")
    msg = str(d.get("status_msg") or "")[:80]
    host = d.get("ssh_host"); port = d.get("ssh_port")
    print(f"t={int(time.time()-start):3d}s actual_status={st!r} ssh={host}:{port} msg={msg}")
    if st == "running" and host and port:
        g = gate(host, port)
        out = (g.stdout or "") + (g.stderr or "")
        if "SSH_OK" in out:
            print("=== SSH GATE: PASS ===")
            print(out.strip()[:900])
            sys.exit(0)
        print(f"  gate rc={g.returncode}: {out.strip().splitlines()[-1][:140] if out.strip() else 'no output'}")
        last = (host, port)
    time.sleep(20)

print(f"=== SSH GATE: NOT_READY within {DEADLINE_S}s (last endpoint {last}) ===")
d = show() or {}
print(f"  final actual_status={d.get('actual_status')!r} msg={str(d.get('status_msg'))[:120]}")
sys.exit(2)
