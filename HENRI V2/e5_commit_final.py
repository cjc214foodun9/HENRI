"""Final commit: the three remaining E5b tooling files. Explicit paths only.

Files: e5b_index_map.py (settled the 0.117 question), e5_commit.py, e5_final_seal.py
Verifies: HEAD before/after, staged name-status, remote SHA equality, main pin.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

WT = Path(r"C:\Users\chan\henri-worktrees\e5-wt")
BRANCH = "carrier/e5-wave-superposition"
PATHS = ["HENRI V2/e5b_index_map.py", "HENRI V2/e5_commit.py", "HENRI V2/e5_final_seal.py"]
MSG = (
    "E5b: C2 index-mapping reconciliation (0.117 UNREPRODUCIBLE) + carrier tooling\n\n"
    "e5b_index_map.py tests every candidate index mapping against E4a's documented C2\n"
    "rule using PINNED matching identifiers (corpus e83889ba, tokenizer c0382117):\n"
    "  shift3 (exact from e4a_construct_audit.py:108-118) -> p1 0.033\n"
    "  shift1 (my earlier recompute)                     -> p1 0.033\n"
    "  shift3/shift1 with eval 44k-45k                   -> p1 0.056\n"
    "  calib 11k-21k                                     -> p1 0.033\n"
    "Receipt value 0.117 is not reproduced under any tested mapping. The calib gold\n"
    "UNIVERSE DOES reproduce exactly (distinct_golds 2435 == receipt), so the calib\n"
    "region is confirmed. Status: STILL_UNREPRODUCIBLE_under_every_tested_mapping.\n"
    "Recorded as an UNREPRODUCIBLE reference; not used as a gate (Amendment A1).\n"
    "No promotion. No capability claim. main untouched 10f5f23.\n"
)


def g(*a, timeout=300):
    p = subprocess.run(["git", *a], cwd=str(WT), capture_output=True, text=True,
                       timeout=timeout)
    return p.returncode, p.stdout.strip(), p.stderr.strip()


def main() -> int:
    _, head0, _ = g("rev-parse", "--short", "HEAD")
    print("[pre] HEAD " + head0)
    present = [p for p in PATHS if (WT / p).exists()]
    print("[pre] present " + str(len(present)) + "/" + str(len(PATHS)))
    rc, out, err = g("add", "--", *present)
    print("[add] rc=" + str(rc) + " " + err[:200])
    if rc != 0:
        return 1
    _, staged, _ = g("diff", "--cached", "--name-status")
    for line in staged.splitlines():
        print("      " + line)
    rc, out, err = g("commit", "-m", MSG)
    print("[commit] rc=" + str(rc) + " " + (out or err)[:160])
    if rc != 0:
        rc, st, _ = g("status", "--porcelain=v1", "-uall")
        print("[status] " + st[:300])
        return 1
    _, head, _ = g("rev-parse", "HEAD")
    print("[commit] " + head[:16])
    rc, out, err = g("push", "origin", BRANCH)
    print("[push] rc=" + str(rc) + " " + (out or err)[:200])
    _, rem, _ = g("ls-remote", "origin", "refs/heads/" + BRANCH)
    remote = rem.split()[0] if rem else ""
    print("[remote] " + remote[:16] + "  match=" + str(remote == head))
    _, st, _ = g("status", "--porcelain=v1", "-uall")
    dirty = [l for l in st.splitlines() if l.strip()]
    print("[post] dirty=" + str(len(dirty)))
    for d in dirty:
        print("      " + d)
    _, m, _ = g("rev-parse", "--short", "origin/main")
    print("[main] " + m + " (must be 10f5f23)")
    print("VERDICT=" + ("E5_WORKTREE_COMPLETE" if (remote == head and not dirty)
                        else "RESIDUAL_STATE"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
