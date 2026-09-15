#!/usr/bin/env python3
"""EXECUTABLE integrity gate: does the sealed DOC agree with the sealed RECEIPT?

WHY THIS EXISTS
    Commit 99fb88a sealed TWO descriptions of the same Phase 10.1 measurement:
      - references/henri_phase10_1_operator_gap_adjudication.md   (human-readable)
      - experiments/verification/evaluate_60_task_koopman_gap_observed.json
    A post-commit edit left HEAD carrying a doc that contradicted HEAD's OWN
    receipt (doc cited koopman_named6 / koopman_no_refl / 480 solves / 5.75 /
    6.0 s; the receipt has koopman_diagonly arms / n_solves 360 / mean_rank_G
    5.667). The doc READS fine, so the mismatch survived visual review. This gate
    makes self-consistency mechanical instead of a matter of care.

    This is an instance of a general rule for this project: a seal is not one
    artifact, it is a SET of artifacts that must agree. Consistency inside the set
    is a testable property, so it must be a test.

USAGE
    python experiments/verification/validate_seal_consistency.py [--doc P --receipt Q] [--json OUT]
    Defaults to the in-repo doc and receipt. --doc/--receipt allow validating
    extracted HEAD copies (see experiments/performance/resolve_seal_mismatch2.sh).

Exit code 0 = consistent, 1 = inconsistent (fail-closed).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]          # ...\HENRI V2
DOC_DEFAULT = REPO / "references" / "henri_phase10_1_operator_gap_adjudication.md"
RECEIPT_DEFAULT = (
    REPO / "experiments" / "verification" / "evaluate_60_task_koopman_gap_observed.json"
)

# Measured arm citations carry a lambda: `koopman_diagonly@0.01`. Prose mentions
# of a DELETED arm (no lambda) are legitimate and must not trip the gate.
ARM_RE = re.compile(r"`((?:legacy|mean_corr|diag_ls|koopman_[a-z0-9_]+@[0-9.]+))`")


def check(doc_p: Path, receipt_p: Path) -> dict:
    problems: list[str] = []
    notes: list[str] = []

    if not doc_p.exists():
        return {"problems": [f"doc missing: {doc_p}"], "notes": [], "CONSISTENT": False}
    if not receipt_p.exists():
        return {"problems": [f"receipt missing: {receipt_p}"], "notes": [], "CONSISTENT": False}

    doc = doc_p.read_text(encoding="utf-8")
    rc = json.loads(receipt_p.read_text(encoding="utf-8"))

    doc_arms = set(ARM_RE.findall(doc))              # lambda-bearing citations only
    receipt_arms = set(rc["arms"].keys())
    receipt_at = {a for a in receipt_arms if "@" in a}

    # ---- 1. lambda-bearing (measured) arms must match in BOTH directions --
    # Only @lambda citations are compared set-wise. ARM_RE also matches the
    # unparameterised baselines (`legacy`, `diag_ls`), which have NO @ form, so
    # comparing all of doc_arms would report them absent from a receipt that
    # cannot contain them. Those two are checked by mention in 1b instead.
    doc_at = {a for a in doc_arms if "@" in a}
    only_doc = sorted(doc_at - receipt_at)
    only_receipt = sorted(receipt_at - doc_at)
    if only_doc:
        problems.append(f"doc cites measured arms ABSENT from receipt: {only_doc}")
    if only_receipt:
        problems.append(f"receipt has arms ABSENT from doc: {only_receipt}")
    notes.append(f"doc_at_arms={len(doc_at)} receipt_at_arms={len(receipt_at)}")

    # ---- 1b. non-parameterised arms must at least be mentioned ------------
    for plain in sorted(receipt_arms - receipt_at):
        if plain in doc:
            notes.append(f"baseline arm `{plain}` mentioned")
        else:
            problems.append(f"receipt arm `{plain}` NEVER mentioned in doc")

    # ---- 2. identity is a scalar baseline, not an @-arm -------------------
    idm = rc.get("identity_mean")
    if idm is None:
        problems.append("receipt lacks identity_mean")
    else:
        s = format(idm, "+.4f")
        if s in doc:
            notes.append(f"identity_mean={s} present")
        else:
            notes.append(f"identity_mean={s} NOT printed in doc (advisory)")

    # ---- 3. per-arm held-out figure must appear verbatim (sign, 4 dp) -----
    for name, a in rc["arms"].items():
        held = format(a["held_out_mean"], "+.4f")
        if held in doc:
            notes.append(f"arm {name} held={held} present")
        else:
            problems.append(f"arm `{name}`: receipt held={held} NOT found in doc")

    # ---- 4. scalar diagnostics that silently drifted ----------------------
    diag = rc.get("koopman_diagnostics") or {}
    for label, value, fmt in (
        ("n_solves", diag.get("n_solves"), "d"),
        ("mean_rank_G", diag.get("mean_rank_G"), ".3f"),
    ):
        if value is None:
            problems.append(f"receipt lacks {label}")
            continue
        s = format(value, fmt)
        if s in doc:
            notes.append(f"{label}={s} present")
        else:
            problems.append(f"receipt {label}={s} NOT found in doc")

    # ---- 5. no @-form arm in the doc may be absent from the receipt ------
    for stale in sorted(ARM_RE.findall(doc)):
        if stale not in receipt_arms:
            problems.append(f"doc cites stale measured arm `{stale}` (absent from receipt)")

    # ---- 6. a VOID run must not be described as valid --------------------
    if not rc.get("baseline_replica_reproduces"):
        problems.append("baseline_replica_reproduces is False -> run VOID")

    # ---- 7. gate outcome in the doc must match the receipt's verdict -----
    v = rc.get("verdict", {})
    if v.get("ACCEPT") is False and "FALSIFIED" not in doc.upper():
        problems.append("receipt ACCEPT=False but doc does not state the gate FALSIFIED")
    if v.get("gate_held_out_met") is False and format(v.get("best_koopman_held_out", 0.0), "+.4f") not in doc:
        notes.append("best_koopman_held_out not printed verbatim (advisory)")

    def rel(p: Path) -> str:
        try:
            return str(p.relative_to(REPO))
        except ValueError:
            return str(p)

    return {
        "doc": rel(doc_p),
        "receipt": rel(receipt_p),
        "doc_arms": sorted(doc_arms),
        "receipt_arms": sorted(receipt_arms),
        "problems": problems,
        "notes": notes,
        "CONSISTENT": not problems,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc", default=str(DOC_DEFAULT))
    ap.add_argument("--receipt", default=str(RECEIPT_DEFAULT))
    ap.add_argument("--json", default=None, help="write the report here")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    rep = check(Path(args.doc), Path(args.receipt))
    if args.json:
        Path(args.json).write_text(json.dumps(rep, indent=1), encoding="utf-8")

    print(f"doc     : {rep['doc']}")
    print(f"receipt : {rep['receipt']}")
    print(f"doc_arms={len(rep['doc_arms'])}  receipt_arms={len(rep['receipt_arms'])}")
    if not args.quiet:
        for p in rep["problems"]:
            print(f"  PROBLEM: {p}")
    for n in rep["notes"]:
        print(f"  ok     : {n}")
    print()
    print("RESULT:", "PASS - doc and receipt are consistent" if rep["CONSISTENT"]
          else f"FAIL - {len(rep['problems'])} inconsistency(ies)")
    return 0 if rep["CONSISTENT"] else 1


if __name__ == "__main__":
    sys.exit(main())
