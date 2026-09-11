"""Retire the legacy E3 bounds (P@1 >= 0.285, P@5 >= 0.640) permanently.

WHY
---
E4a measured, on a fresh never-used split, the context-free marginal baseline at
P@1 = 0.440 and the strongest trivial baseline at P@1 = 0.483. A bound of 0.285
sits BELOW both, so it cannot separate a learned mechanism from a constant
predictor. It is a vacuous gate and must not be enforced anywhere.

WHAT THIS DOES (deterministic, exact-string, verified)
  1. e3_calibrate.py                 : constants -> per-construct bounds loaded
                                       from the sealed E4a receipt, fail-closed
  2. verify_egress_closed_loop.py    : same
  3. tests/contract/test_e3_egress_reform.py  : assert the literals are GONE
                                                and the loader is present
  4. tests/contract/test_gate31_p1pk.py       : same
Sealed prereg .md files are HISTORICAL RECORDS and are NOT touched.
Receipts / audit json are NOT touched.

Every replacement is verified by re-reading the file and re-checking the exact
strings, then checked again for the legacy literals. Writes a receipt.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

E4WT = Path(r"C:\Users\chan\henri-worktrees\e4-wt")
HV2 = E4WT / "HENRI V2"
E3 = Path(r"C:\Users\chan\henri-telemetry\e3")
RECEIPT = E3 / "e4a_construct_audit.json"
OUT = E3 / "bounds_retirement_receipt.json"

GUARD = '''# ---- legacy bounds RETIRED 2026-09-11 (see bounds_retirement_receipt.json) --
# The E3 constants P_AT_1_BOUND = 0.285 / P_AT_5_BOUND = 0.640 are RETIRED:
# E4a measured the context-free marginal at 0.440 and the strongest trivial
# baseline at 0.483 on a fresh split, so 0.285 sits BELOW the trivial baseline
# and cannot separate a mechanism from a constant predictor. Bounds are now
# loaded per construct from the sealed E4a receipt and are never hard-coded.
LEGACY_BOUNDS_RETIRED = {"p_at_1": 0.285, "p_at_5": 0.640}
E4A_RECEIPT = Path(os.environ.get(
    "HENRI_E4A_RECEIPT",
    r"C:\\Users\\chan\\henri-telemetry\\e3\\e4a_construct_audit.json"))


def load_registered_bounds(construct: str = "C1_sentence_window") -> dict:
    """Load per-construct bounds from the sealed E4a audit; fail closed."""
    if not E4A_RECEIPT.exists():
        print("BOUNDS_VERDICT=BLOCKED_INFRA reason=E4A_RECEIPT_MISSING")
        raise SystemExit(2)
    rec = json.loads(E4A_RECEIPT.read_text(encoding="utf-8"))
    b = rec["registered_bounds"][construct]
    base = rec["constructs"][construct]["best_trivial_baseline"]["p1"]
    if not (b["p1"] > base and 0 < b["ce_max"]):
        print("BOUNDS_VERDICT=BLOCKED_INFRA reason=BOUND_NOT_ABOVE_TRIVIAL")
        raise SystemExit(2)
    return b


_REG = load_registered_bounds()
P_AT_1_BOUND = _REG["p1"]
P_AT_5_BOUND = _REG["p5"]
'''


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    rec: dict = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                 "e4a_receipt_sha256": sha(RECEIPT)}
    e4a = json.loads(RECEIPT.read_text(encoding="utf-8"))
    rec["successor_bounds"] = e4a["registered_bounds"]["C1_sentence_window"]

    old_code = ("# ---- pre-registered bounds (module constants; never env-overridable) -------\n"
                "P_AT_1_BOUND = 0.285\n"
                "P_AT_5_BOUND = 0.640\n")
    old_code2 = ("P_AT_1_BOUND = 0.285\n"
                 "P_AT_5_BOUND = 0.640\n")

    edits = []
    for name in ("e3_calibrate.py", "verify_egress_closed_loop.py"):
        p = HV2 / name
        before = p.read_text(encoding="utf-8")
        bsha = sha(p)
        if old_code in before:
            after = before.replace(old_code, GUARD, 1)
        elif old_code2 in before:
            after = before.replace(old_code2, GUARD, 1)
        else:
            edits.append({"file": name, "status": "PATTERN_NOT_FOUND",
                          "sha_before": bsha[:16]})
            continue
        p.write_text(after, encoding="utf-8", newline="")
        reread = p.read_text(encoding="utf-8")
        ok = ("0.285" not in reread.split("LEGACY_BOUNDS_RETIRED")[0]
              and "load_registered_bounds" in reread
              and "P_AT_1_BOUND = _REG" in reread)
        edits.append({"file": name, "status": "RETIRED" if ok else "VERIFY_FAILED",
                      "sha_before": bsha[:16], "sha_after": sha(p)[:16],
                      "legacy_literal_remaining_outside_retirement_note":
                          reread.count("0.285") - 1})
    rec["code_edits"] = edits

    # ---- tests: invert to assert retirement --------------------------------
    t1 = HV2 / "tests" / "contract" / "test_e3_egress_reform.py"
    t2 = HV2 / "tests" / "contract" / "test_gate31_p1pk.py"
    tedits = []
    for p, marker in ((t1, "src = "), (t2, "src = ")):
        if not p.exists():
            tedits.append({"file": p.name, "status": "MISSING"})
            continue
        text = p.read_text(encoding="utf-8")
        bsha = sha(p)
        # find the assert block and replace the two literal asserts
        a1 = '    assert "P_AT_1_BOUND = 0.285" in src\n'
        a2 = '    assert "P_AT_5_BOUND = 0.640" in src\n'
        if a1 in text and a2 in text:
            new = ('    # RETIRED 2026-09-11: the legacy 0.285/0.640 bound sits below the\n'
                   '    # measured trivial baseline and must never be reinstated.\n'
                   '    assert "P_AT_1_BOUND = 0.285" not in src\n'
                   '    assert "P_AT_5_BOUND = 0.640" not in src\n'
                   '    assert "load_registered_bounds" in src\n'
                   '    assert "LEGACY_BOUNDS_RETIRED" in src\n')
            text = text.replace(a1, new, 1).replace(a2, "", 1)
            p.write_text(text, encoding="utf-8", newline="")
            rr = p.read_text(encoding="utf-8")
            ok = ("not in src" in rr and "load_registered_bounds" in rr)
            tedits.append({"file": p.name,
                           "status": "INVERTED" if ok else "VERIFY_FAILED",
                           "sha_before": bsha[:16], "sha_after": sha(p)[:16]})
        else:
            tedits.append({"file": p.name, "status": "PATTERN_NOT_FOUND",
                           "sha_before": bsha[:16]})
    rec["test_edits"] = tedits

    # ---- final proof: no live code still *enforces* the legacy literal ------
    offenders = []
    for p in HV2.rglob("*"):
        if not p.is_file() or p.suffix not in (".py", ".yml", ".yaml"):
            continue
        if "_archive" in str(p) or p.name == Path(__file__).name:
            continue
        try:
            t = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        for i, l in enumerate(t.splitlines(), 1):
            s = l.strip()
            if ("P_AT_1_BOUND = 0.285" in s or "P_AT_5_BOUND = 0.640" in s
                    or "0.285" in s and "RETIRED" not in s and "#" not in s[:1]
                    and "not in src" not in s):
                if "RETIRED" not in t[max(0, t.find(l) - 200):t.find(l)]:
                    offenders.append(f"{p.name}:{i}: {s[:70]}")
    rec["remaining_legacy_enforcement"] = offenders
    rec["verdict"] = ("BOUNDS_RETIRED" if not offenders
                      else "RETIREMENT_INCOMPLETE")

    OUT.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(json.dumps(rec, indent=2))
    print(f"\nWROTE {OUT} sha256={sha(OUT)[:16]}")


if __name__ == "__main__":
    main()
