#!/usr/bin/env python
"""Ontology batch 2 — the ratified gates and the dreamer gate constants.

Appends to the same append-only store as batch 1:
  C:\\Users\\chan\\henri-telemetry\\ontology\\objects.jsonl

Covers:
  * GATE1 protocol authority: the .md milestone protocol governs; the PDF sprint
    list is advisory. The .md's own FALSIFIED facts do NOT resurrect.
  * GATE2 fresh RTX 5090 provisioning approval (scoped).
  * GATE3 / M3-ACT two-sided action-change gate (ratified).
  * Dream-gate roles: entry trigger vs wake setpoint (hysteresis, two-stage).
  * Defect found in the .md reference snippet: the Sagnac stress metric cannot
    pass its own 0.0431 gate at D=65536 (D-SAGNAC defect class).

Run:  python tools/ontology_emit_zone_a_selfplay_gates.py [--dry-run]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os

STORE = os.path.join(os.path.expanduser("~"), "henri-telemetry", "ontology", "objects.jsonl")
BANK = "ca4bb787"
CREATED = "2026-09-26T21:45:00+00:00"
AUDIT = "1616 records, chain intact (head 1a5736a354a74642)"


def rid(kind: str, key: str) -> str:
    return f"ont-{kind}-{hashlib.sha256(f'{kind}|{key}'.encode()).hexdigest()[:12]}"


def build() -> list[dict]:
    recs: list[dict] = []

    # ---- GATE 1: protocol authority
    e_g1 = rid("evidence", "gate1 protocol authority sealed")
    recs.append({
        "record_id": e_g1,
        "kind": "evidence",
        "supports": [],
        "probe": {
            "tool": "henri_audit",
            "query": "GATE1_PROTOCOL_AUTHORITY",
            "argv": ["python", "henri_audit.py", "record", "arbiter", "GATE1_PROTOCOL_AUTHORITY", "{...}"],
            "artifact_ref": "AppData/Local/hermes/audit/henri_audit_chain.jsonl",
            "artifact_sha256": None,
            "exit_code": 0,
            "observed_at_utc": CREATED,
        },
        "note": ("Sealed #5a08a62a792016f2. The document "
                 "HENRI-ARCH-2026-SELFPLAY-DREAMING-V1 (.md) is the governing training "
                 "protocol. The Zone A PDF sprint list is ADVISORY ONLY. Critical boundary: "
                 "adopting the .md protocol adopts its MILESTONE STRUCTURE, not its falsified "
                 "facts. '192 SMs' and the HOPE/Behrouz citation stay FALSIFIED."),
        "evidence_class": "OBSERVED",
        "probe_ref": "audit://GATE1_PROTOCOL_AUTHORITY",
        "created_utc": CREATED,
        "status": "active",
    })

    c_authority = rid("constraint", "md protocol governs but falsified facts stay falsified")
    recs.append({
        "record_id": c_authority,
        "kind": "constraint",
        "rule": ("Do NOT import a source document's numbers into code as measured values merely "
                 "because the document governs the protocol. Protocol authority and factual "
                 "authority are separate. The .md governs the milestone order; its unverified "
                 "constants (14.2 us JVP timing, 19.28 GB VRAM, 10B tokens, r>=0.95, lambda_dark) "
                 "remain HYPOTHESIS until measured on the target hardware."),
        "applies_when": "any implementation derived from a supplied specification document",
        "evidence_class": "OBSERVED",
        "probe_ref": e_g1,
        "created_utc": CREATED,
        "status": "active",
    })

    # ---- GATE 2: provisioning approval
    e_g2 = rid("evidence", "gate2 fresh 5090 approved")
    recs.append({
        "record_id": e_g2,
        "kind": "evidence",
        "supports": [],
        "probe": {
            "tool": "henri_audit",
            "query": "GATE2_FRESH_5090_APPROVED",
            "argv": ["python", "henri_audit.py", "record", "arbiter", "GATE2_FRESH_5090_APPROVED", "{...}"],
            "artifact_ref": "AppData/Local/hermes/audit/henri_audit_chain.jsonl",
            "artifact_sha256": None,
            "exit_code": 0,
            "observed_at_utc": CREATED,
        },
        "note": ("Sealed #8e374b8be50882fa. APPROVED for PROVISIONING ONLY. An unbounded run is "
                 "NOT approved: 10B tokens is a scheduled job requiring a measured throughput "
                 "extrapolation and a cost cap first. Destroying the uhr03 pair is NOT approved."),
        "evidence_class": "OBSERVED",
        "probe_ref": "audit://GATE2_FRESH_5090_APPROVED",
        "created_utc": CREATED,
        "status": "active",
    })

    # ---- GATE 3 / M3-ACT
    t_gate = rid("term", "M3-ACT two-sided action-change gate")
    recs.append({
        "record_id": t_gate,
        "kind": "term",
        "label": "M3-ACT gate — two-sided action-change gate for the dream engine",
        "definition": (
            "On a held-out task set where the frozen pre-dream policy selects an INCORRECT action, "
            "a full dream cycle must change the SELECTED ACTION to the correct one. A score-only "
            "change with an unchanged selected action is a FAIL. Pass requires "
            "flip_rate(wrong->correct) >= theta1 AND flip_rate(correct->wrong) <= theta2, at fixed "
            "seeds, with the pre-dream baseline reproduced within the C7 flake tolerance. Both "
            "sides are required: a one-sided 'did the score move' test is passable by a random "
            "action-flipper and is therefore vacuous."
        ),
        "bank": BANK,
        "semantic_relations": [],
        "evidence_class": "OBSERVED",
        "probe_ref": "audit://GATE3_M3_ACT_RATIFIED",
        "created_utc": CREATED,
        "status": "active",
    })

    c_gate = rid("constraint", "M3-ACT fail means diagnostic only with a real latch")
    recs.append({
        "record_id": c_gate,
        "kind": "constraint",
        "term_id": t_gate,
        "rule": ("On FAIL the dreamer is DIAGNOSTIC_ONLY and a REAL code-level latch (raise or a "
                 "None return on the egress path) must prevent it from touching egress. A comment "
                 "or an unread flag is not a latch: trace flag -> field -> branch -> output. On "
                 "PASS, egress wiring is PERMITTED but NOT AUTOMATIC: it still requires the FUWT "
                 "transducer, which at d97ddd9 exists only as SPEC-2026-09-24-FUWT-EGRESS-V1."),
        "applies_when": "any dream-engine integration into an action-selection path",
        "evidence_class": "OBSERVED",
        "probe_ref": "audit://GATE3_M3_ACT_RATIFIED",
        "created_utc": CREATED,
        "status": "active",
    })

    # ---- dream gate roles (hysteresis, honours the two-stage constraint)
    t_dream = rid("term", "dream entry trigger vs wake setpoint")
    recs.append({
        "record_id": t_dream,
        "kind": "term",
        "label": "Dream entry trigger vs wake setpoint (hysteresis)",
        "definition": (
            "The .md uses 0.0431 for BOTH the dream entry trigger and the wake condition. Under the "
            "ratified two-stage Sagnac constraint those are different gates. Adopted HENRI design: "
            "ENTRY (novelty detection, 'this topology is unseen') uses the SEARCH-class gate "
            "(the 0.35 family, arc_sagnac_veto) and WAKE/LOCK (crystallization) uses the 0.0431 "
            "setpoint. This yields proper hysteresis: enter when stress > 0.35, wake when stress "
            "<= 0.0431. Reusing one constant for both makes entry and exit conditions identical and "
            "conflates the two stages."
        ),
        "bank": BANK,
        "semantic_relations": [
            {"type": "derivation", "target": "ont-term-66266aafa4a3"},
        ],
        "evidence_class": "INFERRED",
        "probe_ref": "mcp__notebooklm__notebook_query@ca4bb787",
        "created_utc": CREATED,
        "status": "active",
    })

    c_dream = rid("constraint", "entry and wake must not share one constant")
    recs.append({
        "record_id": c_dream,
        "kind": "constraint",
        "term_id": t_dream,
        "rule": ("The dream entry trigger must not silently reuse the 0.0431 crystallization "
                 "setpoint. Entry = search-class gate; wake = crystallization setpoint. Log both "
                 "roles explicitly so a later refactor cannot 'unify' them."),
        "applies_when": "henri_latent_dreamer entry/wake logic",
        "evidence_class": "INFERRED",
        "probe_ref": "mcp__notebooklm__notebook_query@ca4bb787",
        "created_utc": CREATED,
        "status": "active",
    })

    # ---- defect in the .md reference snippet
    e_defect = rid("evidence", "md dream snippet sagnac metric cannot pass its own gate")
    recs.append({
        "record_id": e_defect,
        "kind": "evidence",
        "supports": [t_dream],
        "probe": {
            "tool": "file-read",
            "query": "static analysis of FocusedLatentDreamer.forward_dream_step in the .md",
            "argv": ["analysis", "winding = sum(d_phi)/(2*pi) compared against 0.0431"],
            "artifact_ref": "Downloads/project_henri_universal_self_play_pretraining_focused_latent_"
                            "dreaming_and_test_time_topological_adaptation.md",
            "artifact_sha256": None,
            "exit_code": 0,
            "observed_at_utc": CREATED,
        },
        "note": ("Two OBSERVED defects in the .md reference snippet, both of which would produce a "
                 "dead system. (1) sagnac_stress = |sum(d_phi)/(2*pi)| over D=65536 typically reaches "
                 "order sqrt(D) >> 0.0431, so 'passed' is essentially never true and the dream never "
                 "wakes. This is the D-SAGNAC defect class: the metric must be dimension-normalized. "
                 "(2) torch.autograd.grad(sagnac_stress, adapter_params) has no gradient path because "
                 "adapter_params never enters psi_dream, so the SGLD creep is dead. Also: unbounded "
                 "Brainfuck loops need a max_steps budget and a TIMEOUT output symbol."),
        "evidence_class": "OBSERVED",
        "probe_ref": "static-analysis://md-focused-latent-dreamer",
        "created_utc": CREATED,
        "status": "active",
    })

    return recs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    recs = build()
    existing: set[str] = set()
    if os.path.exists(STORE):
        with open(STORE, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        existing.add(json.loads(line)["record_id"])
                    except (json.JSONDecodeError, KeyError):
                        pass

    new = [r for r in recs if r["record_id"] not in existing]
    required = ("record_id", "kind", "evidence_class", "probe_ref", "created_utc", "status")
    for r in new:
        for f in required:
            assert r.get(f), f"{r['record_id']} missing {f}"

    print(f"built={len(recs)} new={len(new)} present={len(recs) - len(new)}")
    for r in new:
        print(f"  {r['kind']:10s} {r['record_id']}  [{r['evidence_class']}]")

    if args.dry_run:
        print("DRY RUN — nothing written")
        return 0

    with open(STORE, "a", encoding="utf-8") as fh:
        for r in new:
            fh.write(json.dumps(r, sort_keys=True) + "\n")
    with open(STORE, encoding="utf-8") as fh:
        total = sum(1 for ln in fh if ln.strip())
    print(f"APPENDED {len(new)} records (total now {total})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
