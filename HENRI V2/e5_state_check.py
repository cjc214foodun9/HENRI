"""DEFINITIVE state check. Reads everything from disk; writes one receipt.

Settles, from bytes rather than from any advisory claim:
  1. functor: Drive live sha, backup sha, defect present?, corrected present?
  2. ledger: records, head, self-verify, presence of the E5 events
  3. git: e5 HEAD + remote + dirty
  4. receipts: C2 coverage, functor certification, marginal diagnostic
  5. whether a backbone result file exists
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path

E3 = Path(r"C:\Users\chan\henri-telemetry\e3")
E5WT = Path(r"C:\Users\chan\henri-worktrees\e5-wt")
LED = Path(r"C:\Users\chan\AppData\Local\hermes\audit\henri_audit_chain.jsonl")
FUNCTOR = Path(r"G:\My Drive\HENRI_Inbox\Sagnacfunctor.txt")
BK_DRIVE = Path(r"G:\My Drive\HENRI_Inbox\Sagnacfunctor.ORIGINAL.txt")
OUT = E3 / "e5_state_check.json"
ORIG = "e60fa07be762b0b6aad559ed8020d8beef5ffa8e07a6976803da8ea0e3a8683f"


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main() -> None:
    rec = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    # 1. functor
    f = {}
    if FUNCTOR.exists():
        t = FUNCTOR.read_text(encoding="utf-8", errors="replace")
        f = {"exists": True, "bytes": FUNCTOR.stat().st_size, "sha256": sha(FUNCTOR),
             "is_original": sha(FUNCTOR) == ORIG,
             "defect_present": ("re / D" in t) and ("1.0 - sim" in t),
             "corrected_present": "np_ * na" in t,
             "has_correction_comment": "CORRECTED 2026-09-11" in t
                                       or "VACUOUS" in t}
    else:
        f = {"exists": False}
    f["backup_drive"] = ({"exists": True, "sha256": sha(BK_DRIVE),
                          "equals_original": sha(BK_DRIVE) == ORIG}
                         if BK_DRIVE.exists() else {"exists": False})
    loc = E3 / "Sagnacfunctor.CORRECTED.txt"
    f["local_corrected"] = ({"exists": True, "sha256": sha(loc)} if loc.exists()
                            else {"exists": False})
    rec["functor"] = f

    # 2. ledger
    rows = [json.loads(l) for l in LED.read_text(encoding="utf-8").splitlines() if l.strip()]
    prev, ok = "0" * 64, True
    for i, r in enumerate(rows):
        b = (f"{r['idx']}|{r['ts']}|{r['actor']}|{r['action']}|"
             f"{json.dumps(r['payload'], sort_keys=True)}|{r['prev_hash']}")
        if (r["idx"] != i or r["prev_hash"] != prev
                or hashlib.sha256(b.encode()).hexdigest() != r["hash"]):
            ok = False
            break
        prev = r["hash"]
    want = ["HENRI_E5A_COVERAGE_GATE", "HENRI_SAGNAC_DELTA_VACUITY_FOUND",
            "HENRI_E5A_COVERAGE_CONSTRUCT_FINDING", "HENRI_FUNCTOR_METRIC_CERTIFIED",
            "HENRI_E5B_C2_GATES"]
    rec["ledger"] = {"records": len(rows), "intact": ok,
                     "head": rows[-1]["hash"][:16],
                     "event_counts": {w: sum(1 for r in rows if r["action"] == w)
                                      for w in want},
                     "last5": [[r["idx"], r["action"]] for r in rows[-5:]]}

    # 3. git
    def g(*a):
        return subprocess.run(["git", *a], cwd=str(E5WT), capture_output=True,
                              text=True, timeout=90).stdout.strip()
    rec["git"] = {"head": g("rev-parse", "--short", "HEAD"),
                  "main": g("rev-parse", "--short", "origin/main"),
                  "branch": g("rev-parse", "--abbrev-ref", "HEAD"),
                  "remote": g("ls-remote", "origin",
                              "refs/heads/carrier/e5-wave-superposition").split()[:1],
                  "dirty": g("status", "--porcelain=v1", "-uall").splitlines()}

    # 4/5. receipts
    files = ["e5b_coverage_c2.json", "functor_metric_certified.json",
             "e5b_marginal_diag.json", "functor_metric_fix.json",
             "e5b_backbone_topk.pt", "e5b_backbone_topk.json",
             "e5b_gates_receipt.json", "e5b_c2_eval_contexts.pt"]
    rec["receipts"] = {}
    for n in files:
        p = E3 / n
        rec["receipts"][n] = ({"exists": True, "bytes": p.stat().st_size,
                               "sha256": sha(p)[:16]} if p.exists()
                              else {"exists": False})

    OUT.write_text(json.dumps(rec, indent=2))
    print(json.dumps(rec, indent=2))
    print("WROTE " + str(OUT) + " sha256=" + sha(OUT)[:16])


if __name__ == "__main__":
    main()
