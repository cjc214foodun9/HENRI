#!/usr/bin/env python3
"""Poll instance 52161444 readiness and print the LIVE SSH endpoint.

WHY A SCRIPT
    Vast reassigns ssh_host/ssh_port on every restart, so the endpoint must be
    re-read every time. During `loading` a `Connection refused` is NORMAL, not an
    SSH fault -- the container is not built yet.
"""
from __future__ import annotations
import json, subprocess, sys

IID = int(sys.argv[1]) if len(sys.argv) > 1 else 52161444

r = subprocess.run(["vastai", "show", "instance", str(IID), "--raw"],
                   capture_output=True, text=True, timeout=120)
if r.returncode != 0:
    sys.exit(f"show-instance rc={r.returncode} {r.stderr[:200]}")
d = json.loads(r.stdout)
print(f"id={d.get('id')}")
print(f"  actual_status   = {d.get('actual_status')}")
print(f"  status_msg      = {str(d.get('status_msg'))[:110]}")
print(f"  gpu_name        = {d.get('gpu_name')}")
print(f"  num_gpus        = {d.get('num_gpus')}")
print(f"  image_uuid      = {str(d.get('image_uuid'))[:80]}")
print(f"  disk_space      = {d.get('disk_space')} GB")
print(f"  dph_total       = {d.get('dph_total')}")
print(f"  ssh_host        = {d.get('ssh_host')}")
print(f"  ssh_port        = {d.get('ssh_port')}   <-- TRUST THIS, not the ports map")
print(f"  ports           = {json.dumps(d.get('ports'))[:120]}")
print(f"  public_ipaddr   = {d.get('public_ipaddr')}")
print(f"  machine_id      = {d.get('machine_id')}")
print(f"  cur_state       = {d.get('cur_state')}")
print(f"  gpu_frac        = {d.get('gpu_frac')}")
print(f"  has_volume      = {d.get('has_volume')}")
