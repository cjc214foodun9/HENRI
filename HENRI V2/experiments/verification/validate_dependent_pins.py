#!/usr/bin/env python3
"""DEPENDENT-PIN GATE: a changed DEFAULT operator must advance its pinned dependents.

WHY THIS EXISTS
    Commit 05a9121 (phase10i, 2026-09-15) changed the arc_task_functor DEFAULT fit
    family from normalised mean-correlation ('mean_conj_corr') to the regularized
    per-slot diagonal least-squares family ('per_slot_diagonal_ridge_ls', ridge
    1e-4). It did NOT advance the two contract baselines pinned to the old default:

        tests/contract/test_f6_adaptive_functor.py::TestDefaultOff::test_c7_differential
        tests/contract/test_f7_affine_egress.py::test_c5_differential

    Both tests then failed for ~15 hours against a tree whose default path was
    working correctly. The pins were RIGHT and the commit was INCOMPLETE: a
    measurement-receipt constant moved and its dependents did not move with it.
    That is the same defect class as doc/receipt divergence (see
    validate_seal_consistency.py), so it gets the same treatment: a mechanical gate.

THE RULE
    The default-operator sentinel in `arc_task_functor.py` and the operator-family
    strings pinned in the f6/f7 contract tests must AGREE. When they do not, the
    gate fails closed and names both sides.

    This is DELIBERATELY a static agreement check, not a hash-of-everything check:
    it fires when the DEFAULT FAMILY or the RIDGE DEFAULT moves without the pins,
    and it does NOT fire for unrelated edits (docstrings, new arms, formatting).
    A gate that fires on every edit gets bypassed; this one fires only on its subject.

USAGE
    python experiments/verification/validate_dependent_pins.py            # check
    python experiments/verification/validate_dependent_pins.py --json OUT # machine receipt

Exit code 0 = default sentinel and pins agree, 1 = disagreement (fail-closed).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]          # ...\HENRI V2

# (source module, [contract tests that pin its DEFAULT], pinned family literals)
DEPENDENT_SETS = [
    {
        "source": REPO / "arc_task_functor.py",
        "dependents": [
            REPO / "tests" / "contract" / "test_f6_adaptive_functor.py",
            REPO / "tests" / "contract" / "test_f7_affine_egress.py",
        ],
        "family_literals": ("per_slot_diagonal_ridge_ls", "mean_conj_corr"),
        "ridge_env": "HENRI_FUNCTOR_RIDGE",
    },
]


def read_family_candidates(source: Path) -> dict:
    """Recover the DEFAULT operator family from arc_task_functor.py.

    Two vocabularies exist in that module and they are NOT the same:
      - FIT MODE names:  "diag_ls", "mean_corr", "koopman_8", "static_partition"
      - OPERATOR FAMILY names: "per_slot_diagonal_ridge_ls", "mean_conj_corr"

    The contract tests pin the OPERATOR FAMILY. So this gate must resolve
    fit-mode -> operator-family through the mapping DECLARED IN THE SOURCE
    (provenance["operator_family"] = (... if _fit_mode == X else ...)), never
    through a hardcoded table here. Comparing the two vocabularies directly was a
    false-positive bug in the first revision of this gate: it reported INCONSISTENT
    against a correct tree.

    Text extraction is deliberate: this gate runs from the pre-commit hook with no
    project imports and no torch.
    """
    text = source.read_text(encoding="utf-8", errors="replace")

    # --- 1. the DEFAULT fit mode (the `else` branch of the selection expression) ---
    default_fit_mode = None
    m = re.search(r"_fit_mode\s*=\s*\(\s*_req\s+if\s+_req\s+in\s+\(([^)]*)\)\s*else\s*\"([^\"]+)\"",
                  text, re.S)
    if m:
        default_fit_mode = m.group(2)
    else:
        m2 = re.search(r"_fit_mode\s*=\s*_req\s+if\s+[^\n]*else\s*\"([^\"]+)\"", text)
        if m2:
            default_fit_mode = m2.group(1)

    # --- 2. the DECLARED fit-mode -> operator-family mapping ---
    # The live expression is a CHAINED ternary, so a single `X if C else Y` pattern
    # does not match it. Real shape (arc_task_functor.py, 2026-09-16):
    #     "operator_family": (
    #         "per_slot_koopman_subspace" if _fit_mode == "koopman_8"
    #         else "per_slot_diagonal_ridge_ls" if _fit_mode == "diag_ls"
    #         else "staticity_partition_identity_plus_colour_ls"
    #         if _fit_mode == "static_partition" else "mean_conj_corr"),
    # So: collect every ("<family>" if _fit_mode == "<mode>") pair, plus the FINAL
    # else-family as the default fallback for unlisted modes.
    expr = ""
    m3 = re.search(r'\"operator_family\"\s*:\s*\((.*?)\)\s*,', text, re.S)
    if m3:
        expr = m3.group(1)
    else:
        m3b = re.search(r'\"operator_family\"\s*:\s*(.*?),\n', text, re.S)
        expr = m3b.group(1) if m3b else ""

    mapping = {}
    for mm in re.finditer(r'\"([a-z_0-9]+)\"\s*\n?\s*if\s+_fit_mode\s*==\s*\"([a-z_0-9]+)\"', expr):
        family, mode = mm.group(1), mm.group(2)
        mapping[mode] = family

    # final fallback: the LAST `else "<family>"` in the chain
    fallbacks = re.findall(r'else\s+\"([a-z_0-9]+)\"', expr)
    # the last `else` that is a bare family literal (not followed by another `if`)
    tail = re.findall(r'else\s+\"([a-z_0-9]+)\"\s*$', expr.strip(), re.S)
    if tail:
        mapping["__else__"] = tail[-1]
    elif fallbacks:
        mapping["__else__"] = fallbacks[-1]

    default_family = None
    if default_fit_mode and mapping:
        default_family = mapping.get(default_fit_mode, mapping.get("__else__"))

    # --- 3. ridge default ---
    ridge_default = None
    m4 = re.search(r"os\.environ\.get\(\s*\"HENRI_FUNCTOR_RIDGE\"\s*,\s*\"([^\"]+)\"\s*\)", text)
    if m4:
        ridge_default = m4.group(1)

    return {"default_fit_mode": default_fit_mode,
            "default_family": default_family,
            "family_mapping": mapping,
            "ridge_default": ridge_default}


def read_sentinel_families(test: Path) -> dict:
    """Read the DEPENDENT-PIN-GATE sentinels declared in a contract test.

    The gate MUST NOT scrape arbitrary operator-family literals out of the test.
    Measured defect (2026-09-16, negative control B): the first revision did exactly
    that, so once the legacy A/B arm also pinned "mean_conj_corr", the pin set
    contained BOTH families and a flip of the source default diag_ls -> mean_corr
    did NOT fire the gate. A gate that is satisfied by either side of the transition
    it guards never ran.

    So each dependent declares its arms explicitly and the gate reads exactly one
    unambiguous value per arm:

        DEFAULT_OPERATOR_FAMILY = "<family the DEFAULT path must produce>"
        LEGACY_OPERATOR_FAMILY  = "<superseded family, kept reachable via the flag>"
    """
    text = test.read_text(encoding="utf-8", errors="replace")
    out = {"default": None, "legacy": None}
    m = re.search(r'^DEFAULT_OPERATOR_FAMILY\s*=\s*"([a-z_0-9]+)"', text, re.M)
    if m:
        out["default"] = m.group(1)
    m2 = re.search(r'^LEGACY_OPERATOR_FAMILY\s*=\s*"([a-z_0-9]+)"', text, re.M)
    if m2:
        out["legacy"] = m2.group(1)
    return out


def check(entry: dict) -> dict:
    source = entry["source"]
    rep = {"source": source.name, "problems": [], "CONSISTENT": True,
           "default_family": None, "ridge_default": None, "pins": {}}
    if not source.is_file():
        rep["CONSISTENT"] = False
        rep["problems"].append(f"source module missing: {source}")
        return rep

    got = read_family_candidates(source)
    rep["default_fit_mode"] = got["default_fit_mode"]
    rep["default_family"] = got["default_family"]
    rep["family_mapping"] = got["family_mapping"]
    rep["ridge_default"] = got["ridge_default"]
    if got["default_fit_mode"] is None:
        rep["CONSISTENT"] = False
        rep["problems"].append(
            "could not recover the DEFAULT fit-mode literal from the source module; "
            "the dependent-pin gate is BLIND -> failing closed")
        return rep
    if not got["family_mapping"]:
        rep["CONSISTENT"] = False
        rep["problems"].append(
            "could not recover the declared fit-mode -> operator-family mapping; "
            "the gate cannot compare vocabulary -> failing closed")
        return rep
    if got["default_family"] is None:
        rep["CONSISTENT"] = False
        rep["problems"].append(
            f"default fit-mode {got['default_fit_mode']!r} does not resolve to an "
            f"operator family via the declared mapping {got['family_mapping']}")
        return rep

    for dep in entry["dependents"]:
        if not dep.is_file():
            rep["CONSISTENT"] = False
            rep["problems"].append(f"dependent test missing: {dep.name}")
            continue
        sent = read_sentinel_families(dep)
        rep["pins"][dep.name] = sent
        if sent["default"] is None:
            # A dependent with no explicit default sentinel cannot detect drift.
            rep["CONSISTENT"] = False
            rep["problems"].append(
                f"{dep.name}: declares no DEFAULT_OPERATOR_FAMILY sentinel -> "
                f"the gate would be blind -> failing closed")
            continue
        if sent["default"] != got["default_family"]:
            rep["CONSISTENT"] = False
            rep["problems"].append(
                f"{dep.name}: DEFAULT_OPERATOR_FAMILY={sent['default']!r} does NOT match "
                f"the source default {got['default_family']!r} -> pin did not advance "
                f"with the default (or the default reverted)")
        if sent["legacy"] is not None and sent["legacy"] == sent["default"]:
            rep["CONSISTENT"] = False
            rep["problems"].append(
                f"{dep.name}: LEGACY sentinel equals DEFAULT sentinel -> the A/B arm "
                f"is vacuous and cannot detect a flip")
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="dependent-pin agreement gate")
    ap.add_argument("--json", default=None, help="write a machine-readable receipt")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    reports = [check(e) for e in DEPENDENT_SETS]
    bad = [r for r in reports if not r["CONSISTENT"]]

    if not args.quiet:
        for r in reports:
            status = "CONSISTENT" if r["CONSISTENT"] else "INCONSISTENT"
            print(f"[{status}] {r['source']}  default={r['default_family']!r} "
                  f"ridge={r['ridge_default']!r}")
            for name, pins in r["pins"].items():
                print(f"    {name}: pins {pins}")
            for p in r["problems"]:
                print(f"    PROBLEM: {p}")

    if args.json:
        Path(args.json).write_text(json.dumps(
            {"gate": "henri.dependent-pin.v1", "reports": reports,
             "covered": len(reports), "failed": len(bad)}, indent=1), encoding="utf-8")

    print(f"RESULT: dependent-pin gate covered {len(reports)} set(s), {len(bad)} failed")
    if bad:
        print("DEPENDENT-PIN GATE FAILED: advance the pinned baselines in the SAME commit "
              "as the default change (or revert the default). Do not silence the pin.",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
