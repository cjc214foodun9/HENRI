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
    # Phase 10.4 -- ONE doc, FOUR receipts. The UWSH receipt carries the ARM TABLE (so
    # rules 1/2 apply); the other three are MEASUREMENT tables whose declared scalars
    # must appear verbatim in the same doc. A sealed claim covered by no rule is not
    # sealed, so every load-bearing number in that doc is bound to a receipt here.
    (REPO / "references" / "henri_phase10_4_uwsh_and_adjoint.md",
     VERIF / "uwsh_subspace_60_observed.json"),
    (REPO / "references" / "henri_phase10_4_uwsh_and_adjoint.md",
     VERIF / "torus_encoder_adjoint_observed.json"),
    (REPO / "references" / "henri_phase10_4_uwsh_and_adjoint.md",
     VERIF / "phase10_4_followup_observed.json"),
    (REPO / "references" / "henri_phase10_4_uwsh_and_adjoint.md",
     VERIF / "uwsh_zone_arithmetic_observed.json"),
    # Phase 7.8 P0-A1 -- the encoder-basis promotion. This pair is a
    # MEASUREMENT receipt (basis degeneracy + fractional recovery), so it is
    # attested by SCALAR_SPECS below rather than by an arm table.
    (REPO / "references" / "henri_phase_map_basis_promotion.md",
     VERIF / "phase_map_basis_observed.json"),
]

# Receipts with no arm table are still SEALABLE, if they declare which of their scalars
# must appear in the doc. Without this the generator of the doc and the checker of the
# doc are the only agreement, which is precisely the Phase 10.1 failure (a doc that reads
# fine while contradicting its own receipt). Keyed by receipt FILENAME.
SCALAR_SPECS = {
    "torus_encoder_adjoint_observed.json": [
        ("encoder-vs-explicit-algebra max abs err", "verification_max_err", ".3e"),
        ("decode cases", "decode_summary.n_cases", "d"),
        ("decode exact count", "decode_summary.n_exact", "d"),
        ("decode exact rate", "decode_summary.exact_rate", ".4f"),
        ("decode mean cell accuracy", "decode_summary.mean_cell_accuracy", ".4f"),
        ("distinct nonzero (kx,ky) pairs", "frequency_census.distinct_pairs_nonzero", "d"),
        ("complete lattice size (S-1)^2", "frequency_census.full_lattice_1_to_Sminus1", "d"),
    ],
    "phase10_4_followup_observed.json": [
        ("oracle rank available", "oracle_rank_available", "d"),
        ("best oracle ceiling at max constructible rank",
         "required_rank.best_oracle_ceiling_achieved", "+.4f"),
        ("incumbent diag_ls ceiling", "required_rank.incumbent_ceiling", "+.4f"),
        ("first k reaching 95pct of incumbent ceiling",
         "required_rank.first_k_at_95pct", "d"),
    ],
    "uwsh_zone_arithmetic_observed.json": [
        ("k required for 64 bytes", "claim_A_zone_a_bytes.64_bytes_requires_k", "d"),
        ("basis bytes at k=16", "basis_footprint.bytes", "d"),
        ("Zone C data floor bytes", "claim_B_zone_c_budget.total_bytes_floor", "d"),
        ("basis MiB at k=16", "basis_footprint.MiB", ".1f"),
        ("incumbent obs per complex slot",
         "claim_C_gamma.per_slot_obs_per_complex_slot", ".1f"),
        ("directive baseline gamma", "claim_C_gamma.directive_baseline_value", ".3e"),
    ],
    # Phase 7.8 P0-A1 -- encoder-basis promotion. The load-bearing claim is that
    # the production basis SEPARATES same-sum positions while the legacy basis
    # cannot, and that fractional recovery is exact on the non-default basis.
    # Every number below is regenerated by gen_phase_map_basis_receipt.py.
    "phase_map_basis_observed.json": [
        ("legacy collinear same-sum cosine",
         "per_basis.default.same_sum_cos", "+.6f"),
        ("incommensurate same-sum cosine",
         "per_basis.incommensurate.same_sum_cos", "+.6f"),
        ("random same-sum cosine", "per_basis.random.same_sum_cos", "+.6f"),
        ("fractional recovery cases", "recovery_incommensurate.n_cases", "d"),
        ("fractional recovery exact", "recovery_incommensurate.n_exact", "d"),
        ("fractional recovery exact rate",
         "recovery_incommensurate.exact_rate", ".4f"),
    ],
}


def dig(rc: dict, path: str):
    """Resolve a dotted path in a receipt; None if absent."""
    cur = rc
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur

# Measured arm citations carry a lambda: `koopman_diagonly@0.01`. A prose mention
# of a DELETED arm (no lambda) is legitimate and must not trip the gate.
_ARM_NAME = (r"(?:legacy|mean_corr|diag_ls|identity|mean_only|wave_[a-z0-9_]+|"
               r"grid_[a-z0-9_]+|sp_[a-z0-9_]+|koopman_[a-z0-9_]+|uwsh|oracle|random)")
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
    scal = SCALAR_SPECS.get(receipt_p.name, [])
    notes.append(f"schema {schema}" + (f" + {len(scal)} declared scalar(s)" if scal else ""))
    if not arm_map and not scal:
        problems.append("receipt exposes no arm table (neither 'arms' nor 'arms_common') "
                        "and declares no sealed scalars in SCALAR_SPECS")
        arm_map = {}
    receipt_arms = set(arm_map)
    receipt_at = {a for a in receipt_arms if "@" in a}

    doc_arms = set(ARM_RE.findall(doc))
    doc_at = {a for a in doc_arms if "@" in a}

    # ---- ARM RULES APPLY ONLY TO AN ARM-BEARING RECEIPT ------------------
    # A doc may bind several receipts (Phase 10.4 binds four). Measurement receipts
    # legitimately have NO arm table, so demanding that they attest the doc's arms would
    # fail on a correct doc. Each receipt is therefore checked for what it CAN attest:
    # the arm table by the receipt that has one, declared scalars by each measurement
    # receipt. Coverage grows (scalars are new checks); nothing is weakened, because the
    # doc's arms are still fully covered by the arm-bearing receipt.
    if not arm_map:
        notes.append(f"no arm table in this receipt -> arm rules 1/1b/2/5 SKIPPED "
                     f"(attested by the arm-bearing receipt in this doc); "
                     f"{len(scal)} scalar rule(s) apply instead")
    else:
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

    # ---- 5. no @-form arm in the doc may be absent from the ARM TABLE -----
    # Scoped to arm-bearing receipts for the same reason as rule 1: a measurement receipt
    # cannot attest arms it does not contain.
    if arm_map:
        for stale in sorted(doc_at):
            if stale not in receipt_arms:
                problems.append(f"doc cites stale measured arm `{stale}` (absent from receipt)")

    # ---- 6. a VOID run must not be described as valid --------------------
    brr = rc.get("baseline_replica_reproduces")
    if brr is None:
        brr = (rc.get("baseline_replica_over_60") or {}).get("reproduces_recorded_baseline")
    if brr is None:
        # A measurement receipt (adjoint / rank sweep / arithmetic) has no baseline to
        # replicate; its claims are covered by SCALAR_SPECS instead.
        if arm_map:
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

    # ---- 9. declared scalars must appear verbatim in the doc --------------
    for label, path, fmt in scal:
        val = dig(rc, path)
        if val is None:
            problems.append(f"declared scalar `{label}` ({path}) MISSING from receipt")
            continue
        try:
            s = format(val, fmt)
        except (TypeError, ValueError):
            problems.append(f"declared scalar `{label}`={val!r} unformattable with {fmt!r}")
            continue
        if s in doc:
            notes.append(f"scalar {label}={s} present")
        else:
            problems.append(f"receipt scalar `{label}`={s} NOT found in doc")

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
