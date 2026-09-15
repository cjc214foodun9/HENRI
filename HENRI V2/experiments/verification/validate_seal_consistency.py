#!/usr/bin/env python3
"""EXECUTABLE integrity gate: does each sealed DOC agree with its RECEIPT?

WHY THIS EXISTS
    Commit 99fb88a sealed TWO descriptions of one Phase 10.1 measurement:
      - references/henri_phase10_1_operator_gap_adjudication.md   (human-readable)
      - experiments/verification/evaluate_60_task_koopman_gap_observed.json
    A post-commit edit left HEAD carrying a doc that contradicted HEAD's OWN
    receipt (doc cited koopman_named6 / koopman_no_refl / 480 solves / 5.75 /
    6.0 s; the receipt has koopman_diagonly arms / n_solves 360 / mean_rank_G
    5.667). The doc READS fine, so the mismatch survived visual review. This gate
    makes self-consistency mechanical instead of a matter of care.

    General rule for this project: a seal is not one artifact, it is a SET of
    artifacts that must agree. Consistency inside the set is a testable property,
    so it must be a test.

GENERALISATION (Phase 10.3)
    Phase 10.3 adds a second sealed pair. Rather than bolt on a second script, the
    gate now iterates a PAIRS list and adapts to the receipt schema:
      schema A (Phase 10.1): rc["arms"] + rc["koopman_diagnostics"]
      schema B (Phase 10.3): rc["arms_common"] (the comparable set) +
                             rc["arms_per_arm_scored_set"]
    Both schemas are checked with the SAME rules, so a doc can never be validated
    by a weaker path than its sibling.

USAGE
    python experiments/verification/validate_seal_consistency.py            # all pairs
    python .../validate_seal_consistency.py --doc P --receipt Q              # one pair
    python .../validate_seal_consistency.py --all --json OUT

Exit code 0 = every pair consistent, 1 = at least one inconsistent (fail-closed).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]          # ...\HENRI V2
VERIF = REPO / "experiments" / "verification"

# (doc, receipt) pairs sealed by this repository. NEW seals MUST be added here,
# otherwise the pre-commit hook silently stops covering them.
PAIRS = [
    (REPO / "references" / "henri_phase10_1_operator_gap_adjudication.md",
     VERIF / "evaluate_60_task_koopman_gap_observed.json"),
    (REPO / "references" / "henri_phase10_3_staticity_partition.md",
     VERIF / "evaluate_60_task_static_partition_observed.json"),
]

# Measured arm citations carry a lambda: `koopman_diagonly@0.01`. A prose mention
# of a DELETED arm (no lambda) is legitimate and must not trip the gate.
_ARM_NAME = r"(?:legacy|mean_corr|diag_ls|identity|wave_[a-z0-9_]+|grid_[a-z0-9_]+|sp_[a-z0-9_]+|koopman_[a-z0-9_]+)"
ARM_RE = re.compile(r"`(" + _ARM_NAME + r"(?:@[0-9.]+)?)`")


def extract_arms(rc: dict):
    """Return (arm_name -> record, schema_label). Schema-adaptive."""
    for key, label in (("arms", "A:rc['arms']"),
                       ("arms_common", "B:rc['arms_common']")):
        d = rc.get(key)
        if isinstance(d, dict) and d:
            return d, label
    return {}, "none"


def check(doc_p: Path, receipt_p: Path) -> dict:
    problems: list[str] = []
    notes: list[str] = []

    if not doc_p.exists():
        return {"problems": [f"doc missing: {doc_p}"], "notes": [], "CONSISTENT": False,
                "doc": str(doc_p), "receipt": str(receipt_p), "doc_arms": [],
                "receipt_arms": []}
    if not receipt_p.exists():
        return {"problems": [f"receipt missing: {receipt_p}"], "notes": [],
                "CONSISTENT": False, "doc": str(doc_p), "receipt": str(receipt_p),
                "doc_arms": [], "receipt_arms": []}

    doc = doc_p.read_text(encoding="utf-8")
    rc = json.loads(receipt_p.read_text(encoding="utf-8"))

    arm_map, schema = extract_arms(rc)
    notes.append(f"schema {schema}")
    if not arm_map:
        problems.append("receipt exposes no arm table (neither 'arms' nor 'arms_common')")
        arm_map = {}
    receipt_arms = set(arm_map)
    receipt_at = {a for a in receipt_arms if "@" in a}

    doc_arms = set(ARM_RE.findall(doc))
    doc_at = {a for a in doc_arms if "@" in a}

    # ---- 1. lambda-bearing (measured) arms must match in BOTH directions --
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

    # ---- 2. per-arm held-out figure must appear verbatim (sign, 4 dp) -----
    for name, a in arm_map.items():
        hv = a.get("held_out_mean")
        if hv is None:
            problems.append(f"arm `{name}` has no held_out_mean in receipt")
            continue
        held = format(hv, "+.4f")
        if held in doc:
            notes.append(f"arm {name} held={held} present")
        else:
            problems.append(f"arm `{name}`: receipt held={held} NOT found in doc")

    # ---- 3. a scalar identity baseline, if the schema exposes one ---------
    idm = rc.get("identity_mean")
    if idm is not None:
        s = format(idm, "+.4f")
        notes.append(f"identity_mean={s} {'present' if s in doc else 'NOT printed (advisory)'}")
    elif "identity" in arm_map:
        s = format(arm_map["identity"]["held_out_mean"], "+.4f")
        notes.append(f"identity arm held={s} {'present' if s in doc else 'NOT printed (advisory)'}")

    # ---- 4. scalar diagnostics that silently drifted (schema A only) ------
    diag = rc.get("koopman_diagnostics") or {}
    for label, value, fmt in (("n_solves", diag.get("n_solves"), "d"),
                              ("mean_rank_G", diag.get("mean_rank_G"), ".3f")):
        if value is None:
            continue
        s = format(value, fmt)
        if s in doc:
            notes.append(f"{label}={s} present")
        else:
            problems.append(f"receipt {label}={s} NOT found in doc")

    # ---- 5. no @-form arm in the doc may be absent from the receipt ------
    for stale in sorted(doc_at):
        if stale not in receipt_arms:
            problems.append(f"doc cites stale measured arm `{stale}` (absent from receipt)")

    # ---- 6. a VOID run must not be described as valid --------------------
    brr = rc.get("baseline_replica_reproduces")
    if brr is None:
        brr = (rc.get("baseline_replica_over_60") or {}).get("reproduces_recorded_baseline")
    if brr is None:
        problems.append("receipt lacks a baseline-replica flag")
    elif not brr:
        problems.append("baseline replica does not reproduce -> run VOID")

    # ---- 7. gate outcome in the doc must match the receipt's verdict -----
    v = rc.get("verdict", {})
    if v.get("ACCEPT") is False and "FALSIFIED" not in doc.upper():
        problems.append("receipt ACCEPT=False but doc does not state the gate FALSIFIED")

    # ---- 8. a receipt that scored arms on DIFFERENT task sets must say so -
    sizes = rc.get("size_of_each_arm_scored_set")
    if isinstance(sizes, dict) and len(set(sizes.values())) > 1:
        if "common subset" not in doc.lower() and "common_subset" not in doc.lower():
            problems.append("receipt scored arms on unequal task sets but the doc does "
                            "not disclose the common-subset comparison")

    return {"doc": str(doc_p), "receipt": str(receipt_p),
            "schema": schema,
            "doc_arms": sorted(doc_arms), "receipt_arms": sorted(receipt_arms),
            "problems": problems, "notes": notes, "CONSISTENT": not problems}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc", default=None)
    ap.add_argument("--receipt", default=None)
    ap.add_argument("--all", action="store_true", help="check every sealed pair")
    ap.add_argument("--json", default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if args.doc or args.receipt:
        if not (args.doc and args.receipt):
            print("--doc and --receipt must be given together", file=sys.stderr)
            return 2
        pairs = [(Path(args.doc), Path(args.receipt))]
    else:
        pairs = [(d, r) for d, r in PAIRS]

    reports, bad = [], 0
    for doc_p, rec_p in pairs:
        rep = check(doc_p, rec_p)
        reports.append(rep)
        if not rep["CONSISTENT"]:
            bad += 1
        if not args.quiet:
            print(f"doc     : {rep['doc']}")
            print(f"receipt : {rep['receipt']}")
            print(f"schema  : {rep.get('schema')}  doc_arms={len(rep['doc_arms'])}"
                  f"  receipt_arms={len(rep['receipt_arms'])}")
            for p in rep["problems"]:
                print(f"  PROBLEM: {p}")
            for n in rep["notes"]:
                print(f"  ok     : {n}")
            print()

    if args.json:
        Path(args.json).write_text(json.dumps(reports, indent=1), encoding="utf-8")

    if bad:
        print(f"RESULT: FAIL - {bad} of {len(pairs)} pair(s) inconsistent")
        return 1
    print(f"RESULT: PASS - {len(pairs)} sealed pair(s) consistent")
    return 0


if __name__ == "__main__":
    sys.exit(main())
