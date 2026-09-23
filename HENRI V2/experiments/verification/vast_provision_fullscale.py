#!/usr/bin/env python3
"""STEP 3 PROVISION, fail-closed. Verify the template from LIVE data before create.

WHY DATA-DRIVEN
    The two Vast skills disagree on the create-time template hash:
        vast_lifecycle        -> 964c5aa07c7c35384023b325c574855e
        henri-vast-lifecycle  -> 91a13de9ef64245e0f152c02981561b6
    Vast churns template hashes, and a stale hash was measured to produce a
    container that never comes up. So this script reads the LIVE template record,
    verifies the image identity, and creates with the hash the API currently
    reports. If any preflight fails it exits non-zero WITHOUT creating.
"""
from __future__ import annotations
import json, pathlib, subprocess, sys

TEMPLATE_ID = 725358
EXPECT_IMAGE_PREFIX = "ghcr.io/cjc214foodun9/henri-v2-execution"
OFFER_ID = 47332370          # best offer: RTX PRO 5000 47.8GB, $0.1288/10GB, rel 0.9988
DISK_GB = 60
LABEL = "henri-v2-fullscale"

def sh(args, timeout=240):
    r = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    return r.returncode, r.stdout, r.stderr

# ---------------------------------------------------------------- preflight 0
rc, out, err = sh(["vastai", "show", "instances", "--raw"])
if rc != 0:
    sys.exit(f"PREFLIGHT_FAIL show-instances rc={rc} {err[:200]}")
inst = json.loads(out)
inst = inst if isinstance(inst, list) else inst.get("instances", [])
live = [i for i in inst if i.get("actual_status") in ("running", "loading")]
print(f"PREFLIGHT instances_total={len(inst)} live={len(live)}")
for i in live:
    print(f"   id={i.get('id')} status={i.get('actual_status')} gpu={i.get('gpu_name')}")
rc, out, err = sh(["vastai", "show", "user", "--raw"])
credit = json.loads(out).get("credit") if rc == 0 else None
print(f"PREFLIGHT credit=${float(credit):.2f}" if credit is not None else "PREFLIGHT credit=UNKNOWN")

# ---------------------------------------------------------------- preflight 1
rc, out, err = sh(["vastai", "search", "templates", "--raw"])
if rc != 0:
    sys.exit(f"PREFLIGHT_FAIL search-templates rc={rc} {err[:200]}")
d = json.loads(out)
templates = d.get("templates") if isinstance(d, dict) else d
templates = templates or []
print(f"PREFLIGHT templates_listed={len(templates)}")
t = next((x for x in templates if str(x.get("id")) == str(TEMPLATE_ID)), None)
if t is None:
    ids = [str(x.get("id")) for x in templates][:25]
    sys.exit(f"PREFLIGHT_FAIL template {TEMPLATE_ID} not in live list; sample ids={ids}")

name = t.get("name")
image = t.get("image")
hash_id = t.get("hash_id") or t.get("hash")
print(f"TEMPLATE id={t.get('id')} name={name!r}")
print(f"TEMPLATE hash_id={hash_id}")
print(f"TEMPLATE image={image!r}")
if not image or not str(image).startswith(EXPECT_IMAGE_PREFIX):
    sys.exit(f"PREFLIGHT_FAIL template image is {image!r}, expected prefix {EXPECT_IMAGE_PREFIX!r}")
if not hash_id:
    sys.exit("PREFLIGHT_FAIL template has no live hash_id")

# ---------------------------------------------------------------- preflight 2
TOP = pathlib.Path(r"C:\Users\chan\Desktop\HENRI 7B SWARM")
for rel in ("environment_files", "HENRI V2/models"):
    p = TOP / rel
    if not p.exists():
        print(f"LOCAL {rel:22} MISSING")
        continue
    tot = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    n = sum(1 for f in p.rglob("*") if f.is_file())
    print(f"LOCAL {rel:22} {n:5d} files {tot/1048576:8.1f} MiB")

# ---------------------------------------------------------------- create
cmd = ["vastai", "create", "instance", str(OFFER_ID),
       "--template_hash", str(hash_id),
       "--disk", str(DISK_GB), "--ssh", "--direct", "--cancel-unavail",
       "--label", LABEL, "--raw"]
print("\nCREATE " + " ".join(cmd))
rc, out, err = sh(cmd, timeout=300)
print(f"CREATE rc={rc}")
if err.strip():
    print(f"CREATE stderr: {err.strip()[:300]}")
try:
    res = json.loads(out)
except Exception:
    print(f"CREATE raw_stdout: {out[:600]}")
    sys.exit(3)
print(f"CREATE success={res.get('success')} new_contract={res.get('new_contract')}")
if res.get("new_contract"):
    pathlib.Path(r"C:\Users\chan\HENRI_telemetry_exports\vast_instance.txt").write_text(
        str(res["new_contract"]), encoding="utf-8")
    print(f"INSTANCE_ID={res['new_contract']}  (written to vast_instance.txt)")
else:
    print(f"CREATE no contract: {json.dumps(res)[:400]}")
    sys.exit(4)
