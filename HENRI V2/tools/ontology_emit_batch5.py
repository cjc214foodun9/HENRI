#!/usr/bin/env python
"""Ontology batch 5 — ACTION 2 re-scoped A/B verdict + the identity-attractor finding.

Appends to: C:\\Users\\chan\\henri-telemetry\\ontology\\objects.jsonl
Schema: henri-ontology/references/ontology-schema.md
Rule: no probe_ref, no commit.
"""
from __future__ import annotations

import hashlib
import json
import os
import time

STORE = os.path.expanduser("~") + r"\henri-telemetry\ontology\objects.jsonl"
CODE = r"C:\Users\chan\henri-worktrees\zone-a-selfplay\HENRI V2"
RECEIPT = os.path.join(CODE, "experiments", "verification",
                       "action2_resonator_paired_ab_observed.json")
HARNESS = os.path.join(CODE, "tools", "action2_resonator_paired_ab.py")
UTS = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


d = json.load(open(RECEIPT, encoding="utf-8"))
h2, h1 = d["h2_decision"], d["h1_mechanism"]
rh, hh = sha(RECEIPT), sha(HARNESS)
n = d["coverage"]["n_scored"]
ident_n = sum(1 for r in d["per_task"] if r.get("identity_triple"))


def rec(rtype, term, value, probe, ev="OBSERVED", extra=None):
    o = {
        "record_id": f"action2-{rtype}-{term}",
        "record_type": rtype,
        "term": term,
        "value": value,
        "evidence_class": ev,
        "source_ref": "commit carrier/zone-a-selfplay + ACTION 2 paired A/B",
        "probe_ref": probe,
        "created_utc": UTS,
        "harness_sha256": hh,
        "receipt_sha256": rh,
    }
    if extra:
        o.update(extra)
    return o


recs = [
    rec("Evidence", "ACTION2_PAIRED_AB_VERDICT", {
            "verdict": h2["verdict"], "control_mean_cos": h2["control_mean_cos"],
            "treatment_mean_cos": h2["treatment_mean_cos"],
            "identity_mean_cos": h2["identity_mean_cos"],
            "delta": h2["delta"], "tau": h2["tau"], "n_scored": n,
            "treatment_beats_control": sum(1 for r in d["per_task"]
                                           if r["treatment_cos"] > r["control_cos"]),
            "control_beats_identity_margin":
                d["h3_identity"]["control_beats_identity"],
        },
        f"tools/action2_resonator_paired_ab.py --limit 60 -> {RECEIPT}"),

    rec("Evidence", "ACTION2_H1_MECHANISM_PASS", {
            "exact": h1["exact"], "trials": h1["trials"], "rate": h1["rate"],
            "gate": h1["gate"], "pass": h1["pass"],
            "meaning": ("the argmax recovers a KNOWN synthesized composite triple; "
                        "the resonator CLASS is sound on solvable-by-construction data"),
        },
        "H1 synthesized-composite arm of the same run",
        extra={"note": "mechanism VALID, decision FALSIFIED -- the two are independent"}),

    rec("Evidence", "IDENTITY_ATTRACTOR_DOMINATES", {
            "identity_triple_tasks": ident_n, "n_scored": n,
            "fraction": round(ident_n / n, 4),
            "identity_triple": [12, 3, 0],
            "decomposition": "shift(0,0) x BASE_COLOUR x solid -- every factor at its identity index",
            "distinct_triples": len({tuple(r["triple"]) for r in d["per_task"]}),
        },
        "per_task[].identity_triple in the receipt"),

    rec("Mapping", "RESONATOR_HYPOTHESIS_CLASS_INSUFFICIENT", {
            "claim": ("a real ARC task operator is NOT in {one roll x one colour x one topology}"),
            "why_the_bound_is_valid": ("exhaustive argmax over the 25x10x2=500 codebook product IS "
                                       "the upper bound of the class, so iteration order / anneal / "
                                       "beta cannot rescue it"),
            "cross_confirms": ("carrier/aaii-v43 @ 16d573c 'VOID on real ARC' real_arc 0.26858; "
                               "its kill record measured mask NOT diagonal (|ratio| std 14975.8) "
                               "and colour NOT a per-slot diagonal op (std 37.94)"),
        },
        "delta -0.017296 vs tau 0.01 on 60 real ARC-AGI-2 training tasks"),

    rec("Constraint", "ACTION2_LITERAL_EXCISION_FORBIDDEN", {
            "directive": "excise W=(X^T X)^-1 X^T Y and instantiate the resonator; verify 16/16",
            "measurement": ("the live functor family is a per-slot DIAGONAL ridge measured at "
                            "held-out 0.4368 / in-sample ceiling 0.7645; the resonator class scores "
                            "0.413310 held-out and is beaten by identity on 54/60 tasks"),
            "ruling": ("do NOT excise. The literal directive would REDUCE ARC accuracy. "
                       "Recorded FALSIFIED; do not retry this family blind."),
            "note": "the directive's '16/16' IS reproduced -- on solvable-by-construction data only (H1 16/16)",
        },
        "same receipt; H1 pass + H2 FALSIFIED together"),

    rec("Evidence", "LEAKAGE_CONTROL_FACTORIZE_NOT_CALLED", {
            "risk": "TripartiteResonator.factorize(target, reference) consumes the TARGET wave",
            "control": ("factorize() deliberately NOT called; the triple is chosen by DEMO-ONLY fit; "
                        "the held target enters at scoring only"),
            "gate": "same demos, same held pair, same decoding budget for both arms",
        },
        "tools/action2_resonator_paired_ab.py leakage_control section"),
]

os.makedirs(os.path.dirname(STORE), exist_ok=True)
with open(STORE, "a", encoding="utf-8") as fh:
    for r in recs:
        fh.write(json.dumps(r) + "\n")

total = sum(1 for _ in open(STORE, encoding="utf-8"))
print(f"appended {len(recs)} records -> store now {total} lines")
print(f"receipt sha256 {rh}")
print(f"harness sha256 {hh}")
print(f"verdict {h2['verdict']}  delta {h2['delta']:+.6f}  identity_triple {ident_n}/{n}")
