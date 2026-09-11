"""Fix two real defects in the bounds-retirement edit, then prove it.

DEFECT 1 (test-breaking): the retirement comment I inserted contains the exact
  literal `P_AT_1_BOUND = 0.285`. The inverted contract tests assert
  `"P_AT_1_BOUND = 0.285" not in src` -> they would FAIL. Reword the comment so
  the exact string appears nowhere, while `LEGACY_BOUNDS_RETIRED` (lowercase
  keys) records the historical values.

DEFECT 2 (import-time hazard): `_REG = load_registered_bounds()` ran at import,
  so any importer (e.g. e3_diag_target_gap.py importing gold_next_token) would
  SystemExit(2) if the receipt were absent. Make resolution LAZY via module
  __getattr__: import always succeeds; the bound resolves on first access and
  fails closed only then.

Also records the true post-edit hashes and compiles every touched file.
"""
from __future__ import annotations

import hashlib
import json
import py_compile
import time
from pathlib import Path

HV2 = Path(r"C:\Users\chan\henri-worktrees\e4-wt\HENRI V2")
E3 = Path(r"C:\Users\chan\henri-telemetry\e3")
OUT = E3 / "bounds_fix_receipt.json"

BAD_COMMENT = "# The E3 constants P_AT_1_BOUND = 0.285 / P_AT_5_BOUND = 0.640 are RETIRED:"
GOOD_COMMENT = "# The E3 legacy constants (historically 0.285 / 0.640) are RETIRED:"

OLD_TAIL = '_REG = load_registered_bounds()\nP_AT_1_BOUND = _REG["p1"]\nP_AT_5_BOUND = _REG["p5"]\n'
NEW_TAIL = (
    '_REG = None  # resolved lazily; import must never fail closed\n'
    '\n'
    '\n'
    'def __getattr__(name: str):\n'
    '    """Resolve retired bounds lazily (PEP 562).\n'
    '\n'
    '    Importing this module always succeeds; the bound is loaded from the\n'
    '    sealed E4a receipt on FIRST ACCESS and fails closed only then. This keeps\n'
    '    importers such as e3_diag_target_gap.py (gold_next_token) working on a\n'
    '    host that has no receipt, while the bound itself can never be\n'
    '    hard-coded again.\n'
    '    """\n'
    '    global _REG\n'
    '    if name in ("P_AT_1_BOUND", "P_AT_5_BOUND"):\n'
    '        if _REG is None:\n'
    '            _REG = load_registered_bounds()\n'
    '        return _REG["p1"] if name == "P_AT_1_BOUND" else _REG["p5"]\n'
    '    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")\n'
)


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main() -> None:
    rec: dict = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                 "targets": [], "forbidden_literals": ["P_AT_1_BOUND = 0.285",
                                                       "P_AT_5_BOUND = 0.640"]}

    for name in ("e3_calibrate.py", "verify_egress_closed_loop.py"):
        p = HV2 / name
        before = p.read_text(encoding="utf-8")
        b_sha = sha(p)
        t = before
        fixes = []
        if BAD_COMMENT in t:
            t = t.replace(BAD_COMMENT, GOOD_COMMENT, 1)
            fixes.append("comment_reworded")
        if OLD_TAIL in t:
            t = t.replace(OLD_TAIL, NEW_TAIL, 1)
            fixes.append("lazy_getattr")
        if t != before:
            p.write_text(t, encoding="utf-8", newline="")
        rr = p.read_text(encoding="utf-8")

        compile_ok = True
        try:
            py_compile.compile(str(p), doraise=True)
        except Exception as e:
            compile_ok = False
            fixes.append(f"COMPILE_FAIL:{e}")

        rec["targets"].append({
            "file": name,
            "fixes": fixes,
            "sha_before": b_sha[:16],
            "sha_after": sha(p)[:16],
            "forbidden_absent": all(f not in rr for f in rec["forbidden_literals"]),
            "has_load_registered_bounds": "load_registered_bounds" in rr,
            "has_LEGACY_BOUNDS_RETIRED": "LEGACY_BOUNDS_RETIRED" in rr,
            "has_lazy_getattr": "__getattr__" in rr,
            "compile_ok": compile_ok,
            "env_overridable_bound": ('os.environ.get("P_AT_1_BOUND"' in rr),
        })

    ok = all(r["forbidden_absent"] and r["has_load_registered_bounds"]
             and r["has_LEGACY_BOUNDS_RETIRED"] and r["has_lazy_getattr"]
             and r["compile_ok"] and not r["env_overridable_bound"]
             for r in rec["targets"])
    rec["VERDICT"] = "RETIREMENT_EDIT_CLEAN" if ok else "RETIREMENT_EDIT_DEFECTIVE"

    OUT.write_text(json.dumps(rec, indent=2))
    print(json.dumps(rec, indent=2))
    print(f"\nWROTE {OUT} sha256={sha(OUT)[:16]}")


if __name__ == "__main__":
    main()
