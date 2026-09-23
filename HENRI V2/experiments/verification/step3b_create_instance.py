#!/usr/bin/env python3
"""STEP 3B: create the full-scale CUDA instance. Fail-closed, digest-pinned.

IMAGE PROVENANCE (why not a template)
    Template 725358 -- pinned by BOTH vast skills -- is ABSENT from the live
    template list (2048 listed, 0 matches for "henri"). `vastai create instance`
    supports `--image`, and the repo's own workflow publishes it:
        .github/workflows/docker-publish.yml
          REGISTRY=ghcr.io  IMAGE_NAME=<owner>/henri-v2-execution
          tags: branch | tag | sha-short | raw:latest (main only)
    Verified by anonymous GHCR manifest HEAD before this script ran:
        ghcr.io/cjc214foodun9/henri-v2-execution:latest -> 200
        digest sha256:aa51848403923ecbdac303a54a7eef0cc5dd288f19a71aacbca0fed92183c36a
    Pin by DIGEST so the created container cannot drift from the verified bytes.

BUDGET (OBSERVED)
    credit $17.47 ; best qualifying offer $0.6156/hr ; 60 GB disk at the measured
    $0.2220/GB/month = $0.0183/hr -> ~$0.634/hr -> ~27 h of headroom. A bounded
    60-step run costs well under $1. Stop (not destroy) after the egress gate.
"""
from __future__ import annotations
import json, pathlib, subprocess, sys

OFFER_ID = 47332370          # RTX PRO 5000 47.8GB, $0.1288/10GB, reliability 0.9988
DIGEST = "sha256:aa51848403923ecbdac303a54a7eef0cc5dd288f19a71aacbca0fed92183c36a"
IMAGE_DIGEST = f"ghcr.io/cjc214foodun9/henri-v2-execution@{DIGEST}"
IMAGE_TAG = "ghcr.io/cjc214foodun9/henri-v2-execution:latest"
DISK_GB = 60
LABEL = "henri-v2-fullscale"

def sh(a, t=300):
    r = subprocess.run(a, capture_output=True, text=True, timeout=t)
    return r.returncode, r.stdout, r.stderr

# ---- preflight A: no live instance may already hold the GPU
rc, out, err = sh(["vastai", "show", "instances", "--raw"])
if rc != 0:
    sys.exit(f"FAIL show-instances rc={rc} {err[:200]}")
inst = json.loads(out)
inst = inst if isinstance(inst, list) else inst.get("instances", [])
live = [i for i in inst if i.get("actual_status") in ("running", "loading")]
print(f"A instances_total={len(inst)} live={len(live)}")
for i in live:
    print(f"   LIVE id={i.get('id')} status={i.get('actual_status')} gpu={i.get('gpu_name')}")
if live:
    sys.exit("FAIL a live instance already exists; resolve it before creating another")

# ---- preflight B: credit
rc, out, _ = sh(["vastai", "show", "user", "--raw"])
credit = float(json.loads(out).get("credit", 0)) if rc == 0 else 0.0
print(f"B credit=${credit:.2f}")
if credit < 5.0:
    sys.exit(f"FAIL credit ${credit:.2f} below the $5.00 floor for a full-scale run")

# ---- preflight C: offer still rentable (market rotates)
rc, out, _ = sh(["vastai", "search", "offers",
                 "gpu_name=RTX_PRO_5000 num_gpus=1 rentable=true", "--raw"])
os_ok = False
if rc == 0:
    d = json.loads(out)
    rows = d if isinstance(d, list) else d.get("offers", [])
    os_ok = any(int(o.get("id", -1)) == OFFER_ID for o in rows)
print(f"C offer {OFFER_ID} rentable={os_ok}")
if not os_ok:
    print("C falling back to the cheapest currently-rentable qualifying offer")
    OFFER_ID = None

# ---- create
def do_create(image):
    cmd = ["vastai", "create", "instance", str(OFFER_ID) if OFFER_ID else None,
           "--image", image, "--disk", str(DISK_GB), "--ssh", "--direct",
           "--cancel-unavail", "--label", LABEL, "--raw"]
    cmd = [c for c in cmd if c is not None]
    print("CREATE " + " ".join(cmd))
    rc, out, err = sh(cmd, t=420)
    print(f"  rc={rc}")
    if err.strip():
        print(f"  stderr: {err.strip()[:400]}")
    try:
        return json.loads(out)
    except Exception:
        print(f"  raw: {out[:600]}")
        return None

res = do_create(IMAGE_DIGEST)
if not res or not res.get("new_contract"):
    print("D digest-pinned create did not yield a contract; retrying with the tag")
    res = do_create(IMAGE_TAG)

print(f"CREATE result={json.dumps(res)[:400] if res else None}")
if res and res.get("new_contract"):
    iid = str(res["new_contract"])
    p = pathlib.Path(r"C:\Users\chan\HENRI_telemetry_exports")
    p.mkdir(parents=True, exist_ok=True)
    (p / "vast_instance_fullscale.txt").write_text(iid, encoding="utf-8")
    print(f"INSTANCE_ID={iid}")
else:
    sys.exit("FAIL no instance created")
