#!/usr/bin/env python3
"""Mutation-gate verification for the wave-KB suite, run ENTIRELY on copies.

WHY: the stale batch claims "5 injected defects, all CAUGHT". That claim is what
separates "28 tests pass" (plumbing) from "the suite detects defects" (a gate).
This session's earlier mutation harnesses v1-v3 were all malformed, so the claim
is re-measured here rather than trusted.

SAFETY: a child process (deleg_bf89049e) is actively reading henri_wave_kb.py.
The live worktree is NEVER written. All mutation happens in a temp dir, and the
worktree sha256 is snapshotted before and after to prove it never changed.

INTERPRETATION RULE (the v4 lesson):
    attempted == 0                  -> INCONCLUSIVE  (never "caught")
    attempted > 0 and failed+err > 0 -> CAUGHT
    attempted > 0 and failed+err == 0-> NOT_CAUGHT
A control arm that mutates nothing (INERT) must be NOT_CAUGHT, otherwise the
harness always-fails and would "catch" everything trivially.
"""
import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

WT = Path("C:/Users/chan/henri-worktrees/aaii-v43/HENRI V2")
PY = "C:/Python314/python.exe"
MOD = "henri_wave_kb.py"
TOK = "henri_vla_tokenizer.py"
TEST = "tests/contract/test_wave_kb.py"
BASE = Path(os.environ.get("LOCALAPPDATA", ".")) / "Temp" / "wkb_mut_probe"

M1_OLD = "        eps_cal = float(torch.quantile(stress.double(), quantile).item())"
M1_NEW = "        eps_cal = float(hardcoded_epsilon)  # M1"

M2_OLD = (
    "                raise MissingPinnedBasisError(\n"
    "                    \"mode='pinned' requires a pinned basis artifact (or an explicit \"\n"
    "                    \"basis tensor). The reference implementation silently substituted \"\n"
    "                    \"qr(randn(2048, 16)) here and called the result world knowledge \"\n"
    "                    \"(defect D-5); this module fails closed instead.\"\n"
    "                )"
)
M2_NEW = (
    "                basis = torch.linalg.qr(torch.randn(int(ambient_real_dim or 2048), "
    "self.k, generator=torch.Generator().manual_seed(11))).Q[:, : self.k]  # M2")
M3_OLD = "        hits = self.contamination_report(blocklist)\n        if hits:"
M3_NEW = "        hits = []  # M3\n        if hits:"
M4_OLD = (
    "            basis = torch.zeros(int(ambient_real_dim), self.k, dtype=torch.float32)\n"
    "            basis[: self.k, : self.k] = torch.eye(self.k, dtype=torch.float32)"
)
M4_NEW = (
    "            basis = torch.linalg.qr(torch.randn(int(ambient_real_dim), self.k, "
    "generator=torch.Generator().manual_seed(7))).Q[:, : self.k]  # M4")
M5_OLD = "    if not hits or best is None or best < float(floor):"
M5_NEW = "    if False:  # M5"

MUTATIONS = [
    ("C0_IDENTITY_baseline", None, None),
    ("M1_calibrate_eps_ignores_distribution", M1_OLD, M1_NEW),
    ("M2_pinned_falls_back_to_qr_randn", M2_OLD, M2_NEW),
    ("M3_assert_clean_noop", M3_OLD, M3_NEW),
    ("M4_null_arm_from_randn", M4_OLD, M4_NEW),
    ("M5_grounded_answer_never_abstains", M5_OLD, M5_NEW),
    ("C_INERT_comment_append", None, None),
]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def fresh_root(dst):
    if dst.exists():
        shutil.rmtree(dst, ignore_errors=True)
    (dst / "tests" / "contract").mkdir(parents=True, exist_ok=True)
    shutil.copy2(WT / MOD, dst / MOD)
    shutil.copy2(WT / TOK, dst / TOK)
    shutil.copy2(WT / TEST, dst / TEST)
    return dst


def run_pytest(root):
    env = {k: v for k, v in os.environ.items()
           if k not in ("VIRTUAL_ENV", "PYTHONPATH", "PYTHONHOME")}
    env["PYTHONPATH"] = str(root)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    p = subprocess.run(
        [PY, "-m", "pytest", TEST, "-q", "-p", "no:cacheprovider",
         "--tb=no", "-rf"],
        cwd=str(root), env=env, capture_output=True, text=True, timeout=1200)
    out = (p.stdout or "") + (p.stderr or "")
    n = lambda pat: (lambda m: int(m.group(1)) if m else 0)(re.search(pat, out))
    passed = n(r"(\d+) passed")
    failed = n(r"(\d+) failed")
    errors = n(r"(\d+) error")
    skipped = n(r"(\d+) skipped")
    attempted = passed + failed + errors + skipped
    failed_names = sorted({ln.split("::")[-1].split()[0].strip()
                           for ln in out.splitlines() if ln.startswith("FAILED ")})
    if attempted == 0:
        verdict = "INCONCLUSIVE"
    elif failed + errors > 0:
        verdict = "CAUGHT"
    else:
        verdict = "NOT_CAUGHT"
    return dict(passed=passed, failed=failed, errors=errors, skipped=skipped,
                attempted=attempted, verdict=verdict, rc=p.returncode,
                failed_tests=failed_names[:8],
                tail=" | ".join(out.strip().splitlines()[-2:])[:200])


def main():
    wt_mod_sha_before = sha(WT / MOD)
    wt_test_sha_before = sha(WT / TEST)
    results = {}

    root = fresh_root(BASE)
    print("worktree module sha256 (before) =", wt_mod_sha_before)
    print("worktree test   sha256 (before) =", wt_test_sha_before)

    # SHADOWING CONTROL -- decisive. If this fails, every "NOT_CAUGHT" below is a
    # false negative because pytest would be importing the WORKTREE module.
    env = {k: v for k, v in os.environ.items()
           if k not in ("VIRTUAL_ENV", "PYTHONPATH", "PYTHONHOME")}
    env["PYTHONPATH"] = str(root)
    probe = subprocess.run(
        [PY, "-c", "import henri_wave_kb as k,sys;print(k.__file__)"],
        cwd=str(root), env=env, capture_output=True, text=True, timeout=300)
    resolved = (probe.stdout or "").strip()
    # NORMALIZE BOTH SIDES. The first version compared str(root) (raw backslashes)
    # against a slash-normalized resolved path, so it reported FALSE on a state
    # where shadowing had in fact SUCCEEDED. Never compare an unnormalized path.
    shadow_ok = str(root).replace("\\", "/").lower() in resolved.replace("\\", "/").lower()
    print("shadowing: import resolved to =", resolved)
    print("shadow_ok =", shadow_ok)
    if not shadow_ok:
        print("ABORT: shadowing failed; mutations would test the worktree module.")
        return 1

    for name, old, new in MUTATIONS:
        root = fresh_root(BASE)
        mod = root / MOD
        src = mod.read_text(encoding="utf-8")
        if old is None and name.startswith("C_INERT"):
            src2 = src + "\n# INERT_CONTROL_MUTATION\n"
        elif old is None:
            src2 = src
        else:
            cnt = src.count(old)
            if cnt != 1:
                results[name] = dict(verdict="INVALID_MUTATION",
                                     note=f"anchor occurrences={cnt} (need 1)")
                print(f"{name:38s} INVALID_MUTATION anchor_count={cnt}")
                continue
            src2 = src.replace(old, new, 1)
        mod.write_text(src2, encoding="utf-8")
        mutated_sha = sha(mod)
        r = run_pytest(root)
        r["mutated_module_sha256"] = mutated_sha
        r["module_bytes"] = mod.stat().st_size
        r["changed_bytes"] = (src2 != src)
        results[name] = r
        print(f"{name:38s} {r['verdict']:15s} "
              f"passed={r['passed']:2d} failed={r['failed']:2d} err={r['errors']:2d} "
              f"attempted={r['attempted']:2d} rc={r['rc']}")
        if r["failed_tests"]:
            print("      failed:", ", ".join(r["failed_tests"]))

    wt_mod_sha_after = sha(WT / MOD)
    wt_test_sha_after = sha(WT / TEST)
    receipt = dict(
        worktree_module_sha_before=wt_mod_sha_before,
        worktree_module_sha_after=wt_mod_sha_after,
        worktree_test_sha_before=wt_test_sha_before,
        worktree_test_sha_after=wt_test_sha_after,
        worktree_unchanged=(wt_mod_sha_before == wt_mod_sha_after
                            and wt_test_sha_before == wt_test_sha_after),
        shadowing_ok=shadow_ok, resolved_module=resolved,
        mutation_dir=str(root), n_mutations=len(MUTATIONS) - 2,
        results=results)

    print("\n=== SUMMARY ===")
    print("worktree_unchanged =", receipt["worktree_unchanged"])
    det = {k: v["verdict"] for k, v in results.items()}
    real = {k: v for k, v in det.items() if k.startswith("M")}
    print("inert control verdict   =", det.get("C_INERT_comment_append"))
    print("identity baseline       =", det.get("C0_IDENTITY_baseline"))
    print("real mutations CAUGHT   =",
          sum(1 for v in real.values() if v == "CAUGHT"), "/", len(real))
    print("real mutations NOT_CAUGHT =",
          [k for k, v in real.items() if v == "NOT_CAUGHT"])
    print("real mutations INVALID/INCONCLUSIVE =",
          [k for k, v in real.items() if v not in ("CAUGHT", "NOT_CAUGHT")])

    out = BASE.parent / "wkb_mutation_receipt.json"
    out.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print("\nreceipt:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
