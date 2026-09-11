"""E5 final seal: vacuous Sagnac metric + coverage-gate constructs, commit, verify.

Finding under seal (measured this session):
  Sagnacfunctor.txt (supplied artifact, authentic sha) defines
      compute_sagnac_delta = max(0, 1 - Re<p,a>/D)
  dividing the inner product by the DIMENSION instead of by the product of norms.
  For unit-norm waves Re<p,a> is in [-1,1], so the residual is ~1 - 1/D for
  ALIGNED, ORTHOGONAL and ANTI-ALIGNED pairs alike.
  MEASURED at D=65536: aligned 0.999985, orthogonal 1.000000, anti 1.000015,
  RANGE 3e-05. A dead or random memory PASSES this metric.
  Classification: VACUOUS SYNCHRONIZATION METRIC (diagnostic-only; must never gate).
  The live sealed definition (1 - Re<p,a>/(|p||a|), range [0,2]) measures
  identical pairs at range 2.0 and is the correct one.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, r"C:\Users\chan\AppData\Local\hermes\scripts")
import henri_audit as ha

E3 = Path(r"C:\Users\chan\henri-telemetry\e3")
E5WT = Path(r"C:\Users\chan\henri-worktrees\e5-wt")
OUT = E3 / "e5_final_seal_receipt.json"
ACTOR = "henri-arbiter"
TOOLING = ["HENRI V2/e5b_sagnac_reconcile.py", "HENRI V2/e5a_seal.py"]


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def git(*a: str):
    r = subprocess.run(["git", *a], cwd=str(E5WT), capture_output=True,
                       text=True, timeout=180)
    return r.returncode, (r.stdout or "").strip(), (r.stderr or "").strip()


def main() -> None:
    sag = json.loads((E3 / "e5b_sagnac_delta_reconciliation.json").read_text())
    cov = json.loads((E3 / "e5a_coverage.json").read_text())
    rec = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "sagnac_receipt_sha256": sha(E3 / "e5b_sagnac_delta_reconciliation.json"),
           "coverage_receipt_sha256": sha(E3 / "e5a_coverage.json"),
           "sagnac_measurements": sag["measurements"],
           "sagnac_finding": sag["finding"],
           "coverage_verdict": cov["verdict"],
           "coverage_gates": cov["gates"]}

    m = sag["measurements"]["65536"]
    print("[1] Sagnac @D=65536")
    for k, v in m.items():
        print("    " + k.ljust(16) + " RANGE=" + str(v["range"]))
    print("[1b] C_functor VACUOUS=" + str(sag["finding"]["C_is_VACUOUS"]))

    # commit tooling (explicit paths)
    rc_a, _, se_a = git("add", *TOOLING)
    rc_c, _, se_c = git("-c", "user.name=HENRI", "-c", "user.email=henri@local",
                        "commit", "-q", "-m",
                        "E5b: reconcile three Sagnac delta definitions. FINDING: the "
                        "supplied Sagnacfunctor.txt metric max(0,1-Re<p,a>/D) divides "
                        "by D not by |p||a| -> range 3e-05 at D=65536 (aligned 0.99999, "
                        "orthogonal 1.00000, anti 1.00002) = VACUOUS, a dead memory "
                        "passes it. Live sealed def (1 - Re/(|p||a|), [0,2]) has range "
                        "2.0 and is correct. Doc metric mean|dphi|*pi/128 is a THIRD "
                        "quantity (range 0.0156). Diagnostic-only: must never gate.")
    rc_h, head, _ = git("rev-parse", "--short", "HEAD")
    git("push", "-q", "origin", "carrier/e5-wave-superposition")
    rc_r, remote, _ = git("ls-remote", "origin", "refs/heads/carrier/e5-wave-superposition")
    remote_short = (remote.split() or [""])[0][:7]
    rec["commit"] = {"add_rc": rc_a, "commit_rc": rc_c, "head": head,
                     "remote": remote_short, "matches": head == remote_short,
                     "stderr": (se_a + se_c)[:200]}
    print("[2] commit rc=" + str(rc_c) + " head=" + str(head)
          + " remote=" + str(remote_short) + " match=" + str(head == remote_short))

    h1 = ha.record_event(ACTOR, "HENRI_SAGNAC_DELTA_VACUITY_FOUND", {
        "artifact": "Sagnacfunctor.txt (supplied, authentic; sha recorded in receipt)",
        "defect": ("compute_sagnac_delta = max(0, 1 - Re<p,a>/D) divides the inner "
                   "product by the dimension instead of by the product of norms"),
        "measured": {"D": 65536, "aligned": m["C_functor"]["aligned"],
                     "orthogonal": m["C_functor"]["orthogonal"],
                     "anti_aligned": m["C_functor"]["anti_aligned"],
                     "range": m["C_functor"]["range"]},
        "classification": "VACUOUS_SYNCHRONIZATION_METRIC",
        "why_it_matters": ("a dead or random memory passes this metric. Per the "
                           "vacuous-synchronization rule it must be labelled "
                           "diagnostic-only and must never gate a verdict."),
        "correct_definition": ("live sealed contract: 1 - Re<pred,emp>/(|pred||emp|), "
                               "range [0,2]; measured range 2.0 on the same pairs"),
        "third_quantity": ("the Zone B doc kernel mean|signed dphi|*pi/128 is a "
                           "mean phase error, not a similarity residual; range "
                           "0.0156; any comparison to [0,2] requires a ratified "
                           "redefinition"),
        "receipt_sha256": rec["sagnac_receipt_sha256"],
        "no_capability_claim": True,
    })
    h2 = ha.record_event(ACTOR, "HENRI_E5A_COVERAGE_CONSTRUCT_FINDING", {
        "carrier": "carrier/e5-wave-superposition",
        "verdict": cov["verdict"],
        "gates": cov["gates"],
        "coverage_table": cov["coverage"],
        "coverage_cost_k90": cov["coverage_cost_k90"],
        "wave_decode_accuracy": cov["wave_decode_accuracy"],
        "decode_tax_at_k64": cov["decode_tax_at_k64"],
        "crosscheck_passed": ("const@k=1 = 0.440 exactly reproduces the E4a marginal "
                             "-> split and gold construction validated"),
        "reading": ("CONSTRUCT finding, not a channel failure. On C1 "
                    "(sentence-window, marginal dominated by the sentence-final "
                    "period) the target is too diffuse for a bounded zero-trainable "
                    "candidate set: no generator reaches gamma=0.90 even at k=2048. "
                    "Rank metrics WITHHELD as VACUOUS per prereg."),
        "next": ("re-measure coverage on the C2 token-stream construct (marginal "
                 "0.117, frozen-backbone oracle 0.433 -> real headroom)"),
        "defects_disclosed": cov["defects_disclosed"],
        "receipt_sha256": rec["coverage_receipt_sha256"],
        "no_capability_claim": True,
    })
    ok, msg = ha.verify_chain()
    rec["seals"] = {"sagnac_vacuity": h1, "coverage_construct": h2}
    rec["chain"] = {"ok": ok, "message": msg}
    print("[3] sealed SAGNAC_DELTA_VACUITY #" + h1[:16])
    print("[4] sealed E5A_COVERAGE_CONSTRUCT #" + h2[:16])
    print("[5] chain " + ("OK " if ok else "FAIL ") + msg)

    OUT.write_text(json.dumps(rec, indent=2))
    print("WROTE " + str(OUT) + " sha256=" + sha(OUT)[:16])


if __name__ == "__main__":
    main()
