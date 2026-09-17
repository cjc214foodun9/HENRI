"""MUTATION GATE for arc_tripartite_resonator.py -- prove the contract suite is a gate.

Runs the contract suite against MUTATED COPIES of the module in a temp shadow
directory. The worktree is NEVER modified: the pristine sources are copied out,
the mutation is applied to the COPY, and pytest is run with PYTHONPATH shadowing
and cwd inside the temp root so the copy is the imported module.

Shadowing is verified with NORMALIZED path comparison (case-folded, backslashes
folded to forward slashes) because a raw-vs-slash mismatch produced a false
negative earlier in this project.

Mutations
  baseline            : no change                              -> must be NOT_CAUGHT
  a_hardcoded_epsilon : iterate on the HARDCODED epsilon        -> must be CAUGHT
  b_det_no_correction : skip the det<0 rotor correction         -> must be CAUGHT
  c_identity_control  : identity arm silently = treatment       -> must be CAUGHT
  d_cap_dropped       : requested iteration cap not honored     -> must be CAUGHT
  inert_comment       : append a comment (no behaviour)         -> must be NOT_CAUGHT

Usage: python tests/contract/check_tripartite_mutation_gate.py [--json PATH]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKTREE = HERE.parents[1]                       # .../HENRI V2
PY = sys.executable

MODULE = "arc_tripartite_resonator.py"
TEST_REL = Path("tests") / "contract" / "test_tripartite_resonator.py"

PROTECTED = [
    MODULE,
    str(TEST_REL).replace("\\", "/"),
    "arc_task_functor.py",
    "henri_vla_tokenizer.py",
    "henri_wave_kb.py",
    "henri_vla_engine.py",
    "tests/contract/test_wave_kb.py",
]

MUTATIONS = [
    ("baseline", None, None,
     "no change (control for the harness itself)"),
    ("a_hardcoded_epsilon",
     "        eps_used = float(self.gate.epsilon_used)   # MUTATION GATE ANCHOR (a)",
     "        eps_used = float(self.gate.hardcoded_epsilon)   # MUTATION GATE ANCHOR (a)",
     "iterate on the document's HARDCODED epsilon instead of the calibrated one"),
    ("b_det_no_correction",
     "    if enforce_rotation and det_raw < 0.0:\n"
     "        R = R.clone()\n"
     "        R[:, -1] = -R[:, -1]\n"
     "        corrected = True",
     "    if False and enforce_rotation and det_raw < 0.0:\n"
     "        R = R.clone()\n"
     "        R[:, -1] = -R[:, -1]\n"
     "        corrected = True",
     "skip the det<0 correction so the rotor may be a REFLECTION"),
    ("c_identity_control",
     "    id_pred_c = to_complex(Xh).reshape(-1)",
     "    id_pred_c = to_complex(pred_t).reshape(-1)",
     "identity control arm silently equals the treatment"),
    ("d_cap_dropped",
     "        cap = int(self.max_iters)                  # MUTATION GATE ANCHOR (d)",
     "        cap = 200                  # MUTATION GATE ANCHOR (d)",
     "the requested iteration cap is replaced by a fixed constant (cap not honored)"),
    ("inert_comment",
     "",
     "\n# INERT CONTROL MUTATION: this comment changes no behaviour.\n",
     "append a comment (inert control)"),
]


def sha256_file(p: Path) -> str:
    if not p.is_file():
        return "<missing>"
    return hashlib.sha256(p.read_bytes()).hexdigest()


def norm(p) -> str:
    return os.path.normcase(os.path.normpath(str(p))).replace("\\", "/")


def worktree_digests() -> dict:
    return {rel: sha256_file(WORKTREE / rel) for rel in PROTECTED}


def build_shadow(root: Path, mutation) -> tuple:
    """Materialise <root>/arc_tripartite_resonator.py + <root>/tests/contract/<test>.

    Returns (applied_count, note).
    """
    (root / "tests" / "contract").mkdir(parents=True, exist_ok=True)
    src = (WORKTREE / MODULE).read_text(encoding="utf-8")
    name, old, new, _desc = mutation
    applied = -1
    if old is None:
        applied = 0
    elif old == "":
        # append-only mutation (inert control): always applies
        src = src + new
        applied = 1
    else:
        n = src.count(old)
        if n == 0:
            return 0, f"ANCHOR NOT FOUND for mutation {name}"
        if n > 1:
            return n, f"ANCHOR NOT UNIQUE ({n} matches) for mutation {name}"
        src = src.replace(old, new, 1)
        applied = 1
    (root / MODULE).write_text(src, encoding="utf-8")
    shutil.copy2(WORKTREE / TEST_REL, root / TEST_REL)
    return applied, "ok"


def run_pytest(root: Path) -> dict:
    env = {k: v for k, v in os.environ.items()
           if k not in ("VIRTUAL_ENV", "PYTHONPATH", "PYTHONHOME")}
    env["PYTHONPATH"] = os.pathsep.join([str(root), str(WORKTREE)])
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["HENRI_MUTATION_SHADOW_ROOT"] = str(root)
    cmd = [PY, "-m", "pytest", str(root / TEST_REL), "-q", "--tb=no",
           "-p", "no:cacheprovider", "-p", "no:randomly"]
    p = subprocess.run(cmd, capture_output=True, text=True, env=env,
                       cwd=str(root), timeout=3600)
    out = (p.stdout or "") + (p.stderr or "")
    failed = sorted({ln.split(" - ")[0].strip() for ln in out.splitlines()
                     if ln.startswith("FAILED")})
    errors = sorted({ln.split(" - ")[0].strip() for ln in out.splitlines()
                     if ln.startswith("ERROR")})
    passed = failed_n = 0
    import re
    for ln in out.splitlines():
        m = re.search(r"(\d+) passed", ln)
        if m:
            passed = int(m.group(1))
        m = re.search(r"(\d+) failed", ln)
        if m:
            failed_n = int(m.group(1))
    return {"returncode": p.returncode, "passed": passed, "failed": failed_n,
            "failed_tests": failed, "errors": errors,
            "tail": "\n".join(out.strip().splitlines()[-3:])}


def verify_shadowing(root: Path) -> dict:
    """Prove the imported module IS the copy, with normalized path comparison."""
    env = {k: v for k, v in os.environ.items()
           if k not in ("VIRTUAL_ENV", "PYTHONPATH", "PYTHONHOME")}
    env["PYTHONPATH"] = os.pathsep.join([str(root), str(WORKTREE)])
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    probe = ("import json, arc_tripartite_resonator as m, henri_wave_kb as k;"
             "print(json.dumps({'module': m.__file__, 'kb': k.__file__}))")
    p = subprocess.run([PY, "-c", probe], capture_output=True, text=True, env=env,
                       cwd=str(root), timeout=600)
    got = json.loads(p.stdout.strip().splitlines()[-1])
    expected_module = root / MODULE
    expected_kb = WORKTREE / "henri_wave_kb.py"
    ok = norm(got["module"]) == norm(expected_module)
    kb_ok = norm(got["kb"]) == norm(expected_kb)
    return {
        "shadowing_ok": bool(ok and kb_ok),
        "module_imported": str(got["module"]),
        "module_expected": str(expected_module),
        "module_normalized_match": bool(ok),
        "collateral_import_delegated_to_worktree": bool(kb_ok),
        "kb_imported_normalized": norm(got["kb"]),
        "kb_expected_normalized": norm(expected_kb),
        "normalization": "os.path.normcase(os.path.normpath(p)).replace('\\\\','/')",
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    ap.add_argument("--only", default=None, help="run a single mutation by name")
    args = ap.parse_args(argv)

    before = worktree_digests()
    results = []
    tmp_root = Path(tempfile.mkdtemp(prefix="henri_tripartite_mutation_"))
    shadow_check = None
    try:
        for mutation in MUTATIONS:
            name, old, new, desc = mutation
            if args.only and name != args.only:
                continue
            root = tmp_root / name
            root.mkdir(parents=True, exist_ok=True)
            applied, note = build_shadow(root, mutation)
            if shadow_check is None:
                shadow_check = verify_shadowing(root)
            if note != "ok":
                results.append({"mutation": name, "description": desc,
                                "result": "MUTATION_NOT_APPLIED", "note": note,
                                "count": applied})
                continue
            res = run_pytest(root)
            caught = res["returncode"] != 0
            results.append({
                "mutation": name,
                "description": desc,
                "result": "CAUGHT" if caught else "NOT_CAUGHT",
                "anchor_substitutions": applied,
                "pytest_returncode": res["returncode"],
                "passed": res["passed"],
                "failed": res["failed"],
                "failed_tests": res["failed_tests"],
                "errors": res["errors"],
                "pytest_tail": res["tail"],
            })
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    after = worktree_digests()
    unchanged = before == after
    table = {r["mutation"]: r["result"] for r in results}
    expected_table = {"baseline": "NOT_CAUGHT", "a_hardcoded_epsilon": "CAUGHT",
                      "b_det_no_correction": "CAUGHT", "c_identity_control": "CAUGHT",
                      "d_cap_dropped": "CAUGHT", "inert_comment": "NOT_CAUGHT"}
    gate_ok = all(table.get(k) == v for k, v in expected_table.items()
                  if k in table) and bool(table)
    receipt = {
        "schema_id": "henri.arc-tripartite-resonator.mutation-gate.v1",
        "worktree": str(WORKTREE),
        "protected_files_sha256_before": before,
        "protected_files_sha256_after": after,
        "worktree_unchanged": bool(unchanged),
        "changed_files": sorted(k for k in before if before[k] != after.get(k)),
        "shadowing": shadow_check,
        "shadowing_ok": bool(shadow_check and shadow_check["shadowing_ok"]),
        "mutations": results,
        "table": table,
        "expected_table": expected_table,
        "gate_pass": bool(gate_ok),
        "python": PY,
        "note": ("The suite is a GATE only if every injected defect is CAUGHT and the "
                 "inert and baseline controls are NOT_CAUGHT. MUTATION_NOT_APPLIED or "
                 "a non-unique anchor is reported as a harness failure, never as a pass."),
    }
    text = json.dumps(receipt, indent=2)
    print(text)
    if args.json:
        Path(args.json).write_text(text, encoding="utf-8")
    return 0 if (gate_ok and unchanged and receipt["shadowing_ok"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
