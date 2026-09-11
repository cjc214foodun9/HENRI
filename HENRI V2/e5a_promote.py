"""E5a staging: promote the ONE demonstrated wave carrier into carrier/e5-wave-superposition.

Rationale (sealed): E4b is the only carrier this session that demonstrated real
information transport -- a RESERVED DISJOINT position channel (2048 of 8192 rows)
carrying explicit end-distance boundary markers, zero trainable, position-region
last-word-to-front cosine 0.0332 vs the g7 bag codec's 0.940, terminal recovery
32/32. E4a/E4c/E4c-bis produced no promotable mechanism (bounds reform, and a
falsified training path). So E5 opens with the position codec as its baseline.

This script copies ONLY the named files with explicit paths (never `git add -A`),
verifies every copy by SHA-256, commits with an explicit path list, pushes, and
confirms the remote object. Then it re-derives the promotion facts from disk.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

E4WT = Path(r"C:\Users\chan\henri-worktrees\e4-wt")
HV2 = E4WT / "HENRI V2"
E5WT = Path(r"C:\Users\chan\henri-worktrees\e5-wt")
E5HV2 = E5WT / "HENRI V2"
E3 = Path(r"C:\Users\chan\henri-telemetry\e3")
OUT = E3 / "e5a_promotion_receipt.json"

COPY = [
    "e4b_position_codec.py",   # the demonstrated carrier
    "e4b_clean.py",            # its clean single-path measurement harness
    "g7_highorder_codec.py",   # dependency
    "g5_separable_codec.py",   # dependency (v5_feature_cells)
    "zone_c_world_knowledge_codec.py",  # dependency (tokenize/features_of)
]


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def git(*a: str):
    r = subprocess.run(["git", *a], cwd=str(E5WT), capture_output=True,
                       text=True, timeout=180)
    return r.returncode, (r.stdout or "").strip(), (r.stderr or "").strip()


def main() -> None:
    rec: dict = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    rc, head, _ = git("rev-parse", "--short", "HEAD")
    rc2, branch, _ = git("rev-parse", "--abbrev-ref", "HEAD")
    rc3, dirty, _ = git("status", "--porcelain=v1", "-uall")
    rec["pre"] = {"head": head, "branch": branch, "dirty_lines": len(dirty.splitlines())}
    print(f"[pre] {branch} @ {head} dirty={len(dirty.splitlines())}")

    copies = []
    for name in COPY:
        s, d = HV2 / name, E5HV2 / name
        if not s.exists():
            copies.append({"file": name, "status": "SOURCE_MISSING"})
            continue
        before = sha(d) if d.exists() else None
        shutil.copy2(s, d)
        after = sha(d)
        copies.append({"file": name, "status": "COPIED",
                       "src_sha16": sha(s)[:16], "dst_sha16": after[:16],
                       "identical": sha(s) == after,
                       "pre_existing": before is not None})
    rec["copies"] = copies
    bad = [c for c in copies if c["status"] == "SOURCE_MISSING"
           or not c.get("identical", False)]
    print(f"[copy] {len(copies)} files, integrity failures: {len(bad)}")
    for c in copies:
        print(f"   {c['file']:<38} {c['status']:<14} identical={c.get('identical')}")

    # compile check on the promoted carrier
    comp = subprocess.run([sys.executable, "-m", "py_compile",
                           str(E5HV2 / "e4b_position_codec.py")],
                          capture_output=True, text=True)
    rec["compile_ok"] = comp.returncode == 0
    print(f"[compile] e4b_position_codec.py rc={comp.returncode}")

    if bad:
        rec["VERDICT"] = "E5A_PROMOTION_ABORTED_INTEGRITY"
        OUT.write_text(json.dumps(rec, indent=2))
        print("ABORTED: integrity failure")
        return

    # explicit-path staging only -- paths MUST be worktree-relative
    git_paths = ["HENRI V2/" + n for n in COPY]
    rc_add, out_add, err_add = git("add", *git_paths)
    rc, staged, _ = git("diff", "--cached", "--name-status")
    rec["staged"] = staged.splitlines()
    rec["git_add"] = {"rc": rc_add, "stdout": out_add[:300], "stderr": err_add[:300]}
    print(f"[stage] add_rc={rc_add} staged lines={len(rec['staged'])} "
          f"err={err_add[:140]}")
    if rc_add != 0:
        rec["VERDICT"] = "E5A_PROMOTION_ABORTED_STAGE_FAILED"
        OUT.write_text(json.dumps(rec, indent=2))
        print("ABORTED: git add failed")
        return
    for l in rec["staged"]:
        print("   " + l)

    msg = ("E5a staging: promote the E4b position-identifiable codec "
           "(reserved disjoint channel 2048/8192 rows, zero trainable, "
           "position last-to-front cos 0.0332 vs bag 0.940, terminal recovery 32/32) "
           "+ its g7/g5/zone_c codec dependencies from carrier/e4-construct. "
           "Provenance: teacher artifact IS the backbone tied output head (event 1374).")
    rc, out, err = git("-c", "user.name=HENRI", "-c", "user.email=henri@local",
                       "commit", "-q", "-m", msg)
    rc2, newhead, _ = git("rev-parse", "--short", "HEAD")
    rec["commit"] = {"rc": rc, "head": newhead, "stderr": err[:200]}
    print(f"[commit] rc={rc} head={newhead}")

    rc, pout, perr = git("push", "-q", "origin", "carrier/e5-wave-superposition")
    rc2, remote, _ = git("ls-remote", "origin",
                         "refs/heads/carrier/e5-wave-superposition")
    remote_short = remote.split()[0][:7] if remote else None
    rec["push"] = {"rc": rc, "remote_obj": remote_short,
                   "matches_head": remote_short == newhead, "stderr": perr[:200]}
    print(f"[push] rc={rc} remote={remote_short} matches={remote_short == newhead}")

    # post-state
    rc, dirty2, _ = git("status", "--porcelain=v1", "-uall")
    rc2, files, _ = git("ls-tree", "-r", "--name-only", "HEAD")
    rec["post"] = {
        "dirty_lines": len(dirty2.splitlines()),
        "promoted_files_in_head": [f for f in files.splitlines()
                                   if Path(f).name in COPY],
        "tree_clean": len(dirty2.splitlines()) == 0,
    }
    print(f"[post] tree_clean={rec['post']['tree_clean']} "
          f"files_in_head={len(rec['post']['promoted_files_in_head'])}")

    ok = (rec["compile_ok"] and len(rec["post"]["promoted_files_in_head"]) == len(COPY)
          and rec["push"]["matches_head"])
    rec["VERDICT"] = "E5A_PROMOTED_AND_PUSHED" if ok else "E5A_PROMOTION_INCOMPLETE"
    rec["carrier_scope_HONEST"] = (
        "the promoted object is a position-identifiable boundary-marker channel; "
        "it is NOT yet a superposition generator. 'wave-superposition' is the "
        "user-directed branch name. Any superposition claim needs its own measurement.")
    OUT.write_text(json.dumps(rec, indent=2))
    print(f"\nVERDICT={rec['VERDICT']}")
    print(f"WROTE {OUT} sha256={sha(OUT)[:16]}")


if __name__ == "__main__":
    main()
