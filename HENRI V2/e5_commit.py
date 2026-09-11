"""Commit the E5 carrier tooling with explicit paths. Never `git add -A`.

Records HEAD, staged name-status, commit SHA, push receipt, and remote SHA.
Excludes: *.pt payloads (gitignored overlay) and every sealed E5a artifact that
is already committed. Stages ONLY the named E5 scripts and the prereg.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

WT = Path(r"C:\Users\chan\henri-worktrees\e5-wt")
BRANCH = "carrier/e5-wave-superposition"

# explicit, worktree-relative paths -- E5 tooling only
PATHS = [
    "HENRI V2/experiments/verification/e5b_c2_coverage_prereg.md",
    "HENRI V2/e5b_coverage_c2.py",
    "HENRI V2/e5b_c2_v2.py",
    "HENRI V2/e5b_backbone_topk.py",
    "HENRI V2/e5b_identifier_recon.py",
    "HENRI V2/e5b_marginal_diag.py",
    "HENRI V2/e5b_certify_functor.py",
    "HENRI V2/e5b_gates.py",
    "HENRI V2/fix_functor_metric.py",
    "HENRI V2/e5_state_check.py",
    "HENRI V2/e5_ground_truth.py",
    "HENRI V2/e5_admin.py",
    "HENRI V2/retire_bounds.py",
    "HENRI V2/fix_guard.py",
    "HENRI V2/admin_seal.py",
    "HENRI V2/e5_final.py",
]

MSG = (
    "E5b: C2 token-stream coverage gate (E5B_CONSTRUCT_INADMISSIBLE at k<=64) + "
    "Sagnac metric-C correction\n\n"
    "Carrier: E5b. Zero trainable.\n"
    "- prereg experiments/verification/e5b_c2_coverage_prereg.md (+ Amendment A1,\n"
    "  pre-seal, disclosed: G-C2-D cross-region reference UNREPRODUCIBLE; G-C2-A\n"
    "  budget unchanged)\n"
    "- CPU arms: const@k1 0.065 | tok1 0.149 | wave 0.041 | decode 0.382\n"
    "- backbone arm (frozen Qwen2.5-0.5B, shard 88c14255): oracle P@1 0.434\n"
    "  (E4a reference 0.433) | coverage@64 0.869 | k90 96 | k95 357\n"
    "- gates: G-C2-A FAIL (0.869 < 0.90 at k<=64) | G-C2-B FAIL (wave 0.399 vs\n"
    "  tok1 0.459) | G-C2-C PASS (delta +0.003) | G-C2-D AMENDED PASS (0.0085)\n"
    "- verdict E5B_CONSTRUCT_INADMISSIBLE; rank metrics WITHHELD; split exported\n"
    "  to the wave arm only as a frozen candidate-set interface\n"
    "- Sagnacfunctor.txt metric C: max(0, 1 - Re<p,a>/D) measured VACUOUS\n"
    "  (range 3e-05 at D=65536); corrected to the sealed normalized form,\n"
    "  range [0,2]. Backup byte-exact. No carrier gates on the defective form.\n"
    "No promotion to main. No capability claim. main untouched 10f5f23.\n"
)


def g(*a, timeout=180):
    p = subprocess.run(["git", *a], cwd=str(WT), capture_output=True, text=True,
                       timeout=timeout)
    return p.returncode, p.stdout.strip(), p.stderr.strip()


def main() -> int:
    rc, head0, _ = g("rev-parse", "--short", "HEAD")
    print("[pre] HEAD " + head0)
    present = [p for p in PATHS if (WT / p).exists()]
    missing = [p for p in PATHS if not (WT / p).exists()]
    print("[pre] present " + str(len(present)) + " missing " + str(len(missing)))
    for m in missing:
        print("      MISSING " + m)

    rc, out, err = g("add", "--", *present)
    print("[add] rc=" + str(rc))
    if rc != 0:
        print("      " + err[:400])
        return 1

    rc, staged, _ = g("diff", "--cached", "--name-status")
    print("[staged]")
    for line in staged.splitlines():
        print("      " + line)

    rc, out, err = g("commit", "-m", MSG)
    print("[commit] rc=" + str(rc) + " " + (out or err)[:200])
    if rc != 0:
        return 1
    rc, head, _ = g("rev-parse", "HEAD")
    short = head[:7]
    print("[commit] " + head)

    rc, out, err = g("push", "origin", BRANCH, timeout=300)
    print("[push] rc=" + str(rc) + " " + (out or err)[:300])

    rc, rem, _ = g("ls-remote", "origin", "refs/heads/" + BRANCH, timeout=180)
    remote_sha = rem.split()[0] if rem else ""
    print("[remote] " + remote_sha[:16])
    match = remote_sha == head
    print("[verify] local==remote : " + str(match))

    rc, st, _ = g("status", "--porcelain=v1", "-uall")
    dirty = [l for l in st.splitlines() if l.strip()]
    print("[post] dirty lines " + str(len(dirty)))
    for d in dirty:
        print("      " + d)

    rc, m, _ = g("rev-parse", "--short", "origin/main")
    print("[main] " + m + "  (must be 10f5f23)")
    print("VERDICT=" + ("E5_TOOLING_COMMITTED_AND_PUSHED" if match
                        else "COMMIT_PUSH_MISMATCH"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
