"""E5a seal: record the coverage-gate verdict, commit E5 tooling, verify chain.

Reads the sealed coverage receipt; seals one governance event; commits the E5
tooling to carrier/e5-wave-superposition with an explicit path list.
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
COV = E3 / "e5a_coverage.json"
OUT = E3 / "e5a_seal_receipt.json"
ACTOR = "henri-arbiter"
TOOLING = ["HENRI V2/e5a_coverage.py", "HENRI V2/e5a_promote.py",
           "HENRI V2/e4b_position_codec.py", "HENRI V2/e4b_clean.py"]


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def git(*a: str):
    r = subprocess.run(["git", *a], cwd=str(E5WT), capture_output=True,
                       text=True, timeout=180)
    return r.returncode, (r.stdout or "").strip(), (r.stderr or "").strip()


def main() -> None:
    cov = json.loads(COV.read_text())
    rec = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "coverage_receipt": str(COV), "coverage_sha256": sha(COV),
           "verdict": cov["verdict"], "gates": cov["gates"],
           "coverage": cov["coverage"], "decode_acc": cov["wave_decode_accuracy"],
           "k90": cov["coverage_cost_k90"], "decode_tax_k64": cov["decode_tax_at_k64"]}

    print("[1] verdict " + cov["verdict"] + "  gates " + json.dumps(cov["gates"]))

    # commit tooling (explicit paths only)
    rc_a, so, se = git("add", *TOOLING)
    rc_c, so2, se2 = git("-c", "user.name=HENRI", "-c", "user.email=henri@local",
                         "commit", "-q", "-m",
                         "E5a coverage gate: zero-trainable candidate-set coverage "
                         "on the sealed E4a fresh split. VERDICT E5A_COVERAGE_FAIL "
                         "(CONSTRUCT finding): const@k1=0.440 reproduces the E4a "
                         "marginal (cross-check PASS) but no arm reaches gamma=0.90 "
                         "even at k=2048 (const 0.839); wave decode acc 0.839 < 0.95. "
                         "Rank metrics withheld as VACUOUS. Next per prereg reading: "
                         "re-measure on C2 token-stream (marginal 0.117, oracle 0.433).")
    rc_h, head, _ = git("rev-parse", "--short", "HEAD")
    git("push", "-q", "origin", "carrier/e5-wave-superposition")
    rc_r, remote, _ = git("ls-remote", "origin", "refs/heads/carrier/e5-wave-superposition")
    remote_short = (remote.split() or [""])[0][:7]
    rec["commit"] = {"add_rc": rc_a, "commit_rc": rc_c, "head": head,
                     "remote": remote_short, "matches": head == remote_short,
                     "stderr": (se + se2)[:200]}
    print("[2] commit rc=" + str(rc_c) + " head=" + str(head)
          + " remote=" + str(remote_short) + " match=" + str(head == remote_short))

    # seal
    h = ha.record_event(ACTOR, "HENRI_E5A_COVERAGE_GATE", {
        "carrier": "carrier/e5-wave-superposition",
        "mechanism": "zero-trainable wave candidate-set coverage gate",
        "trainable_params": 0,
        "coverage_receipt_sha256": rec["coverage_sha256"],
        "split": "sealed E4a fresh split 11000..21999 (reused, CONDITIONAL)",
        "verdict": cov["verdict"],
        "gates": cov["gates"],
        "coverage_table": cov["coverage"],
        "coverage_cost_k90": cov["coverage_cost_k90"],
        "wave_decode_accuracy": cov["wave_decode_accuracy"],
        "decode_tax_at_k64": cov["decode_tax_at_k64"],
        "crosscheck": ("const@k=1 = 0.440 exactly reproduces the E4a marginal "
                       "baseline -> split and gold construction validated"),
        "reading_SEALED": (
            "CONSTRUCT finding, not a channel failure. On the C1 sentence-window "
            "construct (dominated by the sentence-final period) the target is too "
            "diffuse for a bounded zero-trainable candidate set: no generator "
            "reaches gamma=0.90 even at k=2048. Rank metrics are WITHHELD as "
            "VACUOUS per prereg. The wave arm is NOT penalized (decode tax -0.014, "
            "i.e. the reserved-channel decode concentrates on frequent words whose "
            "successor sets are better populated)."),
        "defects_disclosed": cov["defects_disclosed"],
        "next_carrier": ("re-measure coverage on the C2 token-stream construct, "
                         "where the marginal is 0.117 and the frozen-backbone "
                         "oracle is 0.433 -> real headroom for a bounded set"),
        "no_promotion_to_main": True, "no_capability_claim": True,
    })
    ok, msg = ha.verify_chain()
    rec["seal"] = h
    rec["chain"] = {"ok": ok, "message": msg}
    print("[3] sealed HENRI_E5A_COVERAGE_GATE #" + h[:16])
    print("[4] chain " + ("OK " if ok else "FAIL ") + msg)

    OUT.write_text(json.dumps(rec, indent=2))
    print("WROTE " + str(OUT) + " sha256=" + sha(OUT)[:16])


if __name__ == "__main__":
    main()
