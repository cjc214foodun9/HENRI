"""E5b COMPLIANCE DISCLOSURE: prereg-declared split != executed split, and the
executed eval region overlaps a prior carrier's consumed range.

MY DEFECT (disclosed, not hidden):
  e5b_c2_coverage_prereg.md declared:
      calibration 700,000-719,999   evaluation 800,000-800,999
  e5b_c2_v2.py EXECUTED:
      calibration 400,000-419,999   evaluation 600,000-600,999
  and 600000-601000 is the range the prereg ITSELF lists as consumed by E4c.
  The gates were therefore computed on a split that (a) differs from the declared
  one and (b) is not demonstrably fresh.

This script does NOT re-run anything. It:
  1. recovers the ACTUAL consumed regions from the E4c source on disk
  2. states the exact executed split
  3. seals a disclosure event with the line numbers that prove the overlap
  4. bounds the impact on the verdict (a FAIL is conservative under contamination)
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, r"C:\Users\chan\AppData\Local\hermes\scripts")
import henri_audit as ha

E4WT = Path(r"C:\Users\chan\henri-worktrees\e4-wt\HENRI V2")
E5WT = Path(r"C:\Users\chan\henri-worktrees\e5-wt\HENRI V2")
E3 = Path(r"C:\Users\chan\henri-telemetry\e3")
OUT = E3 / "e5b_compliance_disclosure.json"
EXECUTED = {"calib": [400_000, 420_000], "eval": [600_000, 601_000]}
DECLARED = {"calib": [700_000, 719_999], "eval": [800_000, 800_999]}


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main() -> None:
    rec = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    rec["executed_split"] = EXECUTED
    rec["prereg_declared_split"] = DECLARED
    rec["split_mismatch"] = (EXECUTED != DECLARED)

    # ---- recover the ACTUAL consumed regions from E4c sources ------------
    pat = re.compile(r"(\d{6,7})")
    found: dict[str, list] = {}
    for name in ["e4c_factorial.py", "e4c_oracle_probe.py", "e4c_probe2.py"]:
        p = E4WT / name
        if not p.exists():
            continue
        hits = []
        for ln, line in enumerate(p.read_text(encoding="utf-8",
                                              errors="replace").splitlines(), 1):
            if any(k in line for k in ("CALIB", "EVAL", "RANGE", "POS", "start",
                                       "600000", "300000", "800000", "400000")):
                nums = pat.findall(line)
                if nums:
                    hits.append({"line": ln, "text": line.strip()[:110],
                                 "nums": nums[:6]})
        found[name] = hits[:12]
    rec["e4c_sources_scanned"] = list(found)
    rec["e4c_region_lines"] = found

    # the prereg's own claim about E4c's consumed ranges
    pre = (E5WT / "experiments" / "verification" / "e5b_c2_coverage_prereg.md")
    pre_txt = pre.read_text(encoding="utf-8") if pre.exists() else ""
    rec["prereg_consumed_list_text"] = next(
        (l.strip() for l in pre_txt.splitlines() if "E4c" in l and "consumed" in l.lower()),
        None)
    rec["prereg_says_e4c_used_600k_601k"] = "600000" in pre_txt and "601000" in pre_txt
    rec["executed_eval_overlaps_that_range"] = (
        rec["prereg_says_e4c_used_600k_601k"]
        and EXECUTED["eval"][0] == 600_000)

    # ---- impact bound -----------------------------------------------------
    gates = json.loads((E3 / "e5b_gates_receipt.json").read_text())
    primary = gates["gates"]["G_C2_A_admissibility_PRIMARY"]["pass"]
    rec["verdict_computed"] = gates["VERDICT"]
    rec["primary_gate_pass"] = primary
    rec["impact_bound"] = {
        "direction": "contamination INFLATES coverage (exposed regions are easier)",
        "primary_gate_failed": (not primary),
        "measured_at_k64": gates["gates"]["G_C2_A_admissibility_PRIMARY"]["measured"],
        "k90": gates.get("k90"),
        "prereg_budget_k": 64,
        "conclusion": (
            "the PRIMARY gate failed BY 1.5x (k90=96 vs budget 64) even on a "
            "region that may be easier than fresh. A FAIL under inflation stays "
            "a FAIL. The verdict E5B_CONSTRUCT_INADMISSIBLE is therefore "
            "CONSERVATIVE and is NOT recalled. However the 'fresh split' label "
            "is WITHDRAWN: the carrier must be re-run on a verified-fresh region "
            "before any POSITIVE claim, promotion, or rank metric is used."),
        "labels": {
            "withdrawn": "CONDITIONAL_FRESH_SPLIT_SAME_CORPUS",
            "assigned": "CONDITIONAL_REUSED_EVAL_REGION_CONFIRMED",
            "verdict_effect": "none (negative result is conservative)",
            "positive_claims": "BLOCKED until a verified-fresh re-run",
        },
    }

    # ---- seal -------------------------------------------------------------
    h = ha.record_event("henri-arbiter", "HENRI_E5B_COMPLIANCE_DISCLOSURE", {
        "executed_split": EXECUTED,
        "prereg_declared_split": DECLARED,
        "split_mismatch": rec["split_mismatch"],
        "executed_eval_overlaps_e4c_consumed": rec["executed_eval_overlaps_that_range"],
        "prereg_consumed_list_text": rec["prereg_consumed_list_text"],
        "verdict_computed": rec["verdict_computed"],
        "verdict_recalled": False,
        "why_not_recalled": rec["impact_bound"]["conclusion"],
        "labels": rec["impact_bound"]["labels"],
        "prereg_sha256": sha(pre) if pre.exists() else None,
        "gates_receipt_sha256": sha(E3 / "e5b_gates_receipt.json"),
        "self_reported_defect": True,
        "no_new_run": True,
    })
    ok, msg = ha.verify_chain()
    rec["seal"] = h
    rec["chain"] = {"ok": ok, "message": msg}
    OUT.write_text(json.dumps(rec, indent=2))

    print("[mismatch] executed " + json.dumps(EXECUTED)
          + " vs declared " + json.dumps(DECLARED)
          + " -> " + str(rec["split_mismatch"]))
    print("[overlap] prereg lists E4c range 600000-601000: "
          + str(rec["prereg_says_e4c_used_600k_601k"])
          + "; my eval starts there: " + str(rec["executed_eval_overlaps_that_range"]))
    print("[e4c lines] " + str(len(found)) + " source(s) scanned")
    for f, hits in found.items():
        for x in hits[:3]:
            print("      " + f + ":" + str(x["line"]) + "  " + x["text"][:80])
    print("[verdict] " + rec["verdict_computed"] + " (NOT recalled)")
    print("[sealed] HENRI_E5B_COMPLIANCE_DISCLOSURE #" + h[:16])
    print("[chain] " + ("OK " if ok else "FAIL ") + msg)
    print("WROTE " + str(OUT) + " sha256=" + sha(OUT)[:16])


if __name__ == "__main__":
    main()
