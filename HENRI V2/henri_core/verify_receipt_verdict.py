"""Re-derive a receipt's verdict from its OWN numbers, and audit verdict presence.

DEFECT D139 (self-caught on a stale-notification check)
    Commit c15adb9 has the subject "FHRR binding codec -- H-BIND FALSIFIED at
    this scale". The receipt it produced,
    design/zone_a/evidence/henri_binding_codec_receipt.json, stores NO verdict:

        top-level keys : ['arms', 'bounds_moved', 'n_compose_rows',
                          'n_train_rows', 'schema', 'unigram_floor_C']
        'verdict' in d : False
        raw FALSIF|SUPPORT tokens in the file : []

    So the verdict is DERIVED from the numbers by the harness criterion, not
    STORED in the artifact. The established repo convention is the opposite:
    egress_*, codec_*, h2_*, h3_*, hopfield_* and latency_* receipts all carry a
    top-level "verdict" string. A reader of a session receipt must currently
    re-read prose to learn the outcome. That is a provenance gap, not a science
    gap, and it is fixable without touching any evidence byte.

WHAT THIS SCRIPT DOES
    It never writes to a receipt. It reads raw bytes, re-applies the committed
    criterion, and prints the derived verdict plus whether the on-disk artifact
    self-describes. Repeatable, deterministic, no model in the loop.

USAGE
    python henri_core/verify_receipt_verdict.py                 # audit all
    python henri_core/verify_receipt_verdict.py --json OUT      # machine output
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys

EVID = "design/zone_a/evidence"

# ------------------------------------------------------------------ criteria
# Each entry mirrors the criterion literally coded in its producing harness.
# Keep these in sync with the producer; a mismatch is itself a defect.


def criterion_binding_codec(d: dict) -> tuple[str, str]:
    """exp_binding_codec.py line 175:
        if c > max(a, b) + 0.05 and c > ctl + 0.05:  -> SUPPORTED
    where c=C_fhrr_bind, a=A_sum_off, b=B_sum_pos, ctl=D_control,
    all on ev.C_compose.em.
    """
    em = {k: v["ev"]["C_compose"]["em"] for k, v in d["arms"].items()}
    c = em["C_fhrr_bind"]
    a = em["A_sum_off"]
    b = em["B_sum_pos"]
    ctl = em["D_control"]
    thresh = 0.05
    supported = (c > max(a, b) + thresh) and (c > ctl + thresh)
    v = "H-BIND SUPPORTED" if supported else "H-BIND FALSIFIED"
    why = (f"c={c:.6f} max(a,b)+{thresh}={max(a, b) + thresh:.6f} "
           f"ctl+{thresh}={ctl + thresh:.6f} floor={d['unigram_floor_C']:.6f}")
    return v, why


REGISTRY = {
    "henri_binding_codec_receipt.json": criterion_binding_codec,
}


# ------------------------------------------------------------------- helpers
def sha256_bytes(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def raw_mentions_verdict(path: str) -> bool:
    with open(path, "rb") as f:
        b = f.read()
    return b"FALSIFIED" in b or b"SUPPORTED" in b


def audit() -> dict:
    rows, lacks = [], []
    for p in sorted(glob.glob(os.path.join(EVID, "*.json"))):
        name = os.path.basename(p)
        try:
            with open(p, "rb") as f:
                raw = f.read()
            d = json.loads(raw)
        except Exception as e:                       # unreadable is not silent
            rows.append({"receipt": name, "error": str(e)})
            continue
        if not isinstance(d, dict):
            continue
        has_v = isinstance(d.get("verdict"), str) and bool(d["verdict"])
        ng = len(d.get("gates") or []) if isinstance(d.get("gates"), list) else 0
        row = {
            "receipt": name,
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "stored_verdict": d.get("verdict") if has_v else None,
            "self_describing": has_v,
            "gates": ng,
        }
        if name in REGISTRY and not has_v:
            v, why = REGISTRY[name](d)
            row["derived_verdict"] = v
            row["derivation"] = why
            row["raw_mentions_verdict_token"] = raw_mentions_verdict(p)
        if not has_v:
            lacks.append(name)
        rows.append(row)
    return {"rows": rows, "receipts_without_stored_verdict": lacks,
            "n_receipts": len([r for r in rows if "error" not in r]),
            "n_without": len(lacks)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=str, default=None)
    a = ap.parse_args()
    # D140 (self-caught): the first version chdir'd to the PARENT of henri_core,
    # which is "HENRI V2", but design/zone_a/evidence lives at the REPO ROOT
    # (one level higher). The glob then matched nothing and the audit reported
    # "receipts audited: 0" -- a false-empty, the same defect class as a gate
    # that cannot fail. Three levels: henri_core -> HENRI V2 -> repo root.
    repo_root = os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))
    os.chdir(repo_root)
    if not os.path.isdir(EVID):
        print(f"FATAL: evidence dir not found at {os.path.abspath(EVID)}")
        return 2
    res = audit()

    print(f"receipts audited            : {res['n_receipts']}")
    print(f"without a stored verdict    : {res['n_without']}")
    print()
    print("=== session receipts (the ones this work produced) ===")
    for r in res["rows"]:
        if "error" in r:
            print(f"  {r['receipt']:52s} UNREADABLE {r['error']}")
            continue
        if not r["receipt"].startswith("henri_"):
            continue
        tag = f"stored={r['stored_verdict']!r}" if r["self_describing"] \
            else "NO STORED VERDICT"
        print(f"  {r['receipt']:52s} bytes={r['bytes']:6d} gates={r['gates']:2d} {tag}")
        if r.get("derived_verdict"):
            print(f"      derived  : {r['derived_verdict']}")
            print(f"      criterion: {r['derivation']}")
            print(f"      raw token present: {r['raw_mentions_verdict_token']}")

    if res["n_without"]:
        print()
        print("=== ALL receipts missing a stored verdict ===")
        for n in res["receipts_without_stored_verdict"]:
            print(f"  {n}")

    if a.json:
        os.makedirs(os.path.dirname(a.json), exist_ok=True)
        with open(a.json, "w") as f:
            json.dump(res, f, indent=2)
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
