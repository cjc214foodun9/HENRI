#!/usr/bin/env python
"""Append the Zone A self-play sprint records to the HENRI ontology store.

Store:  C:\\Users\\chan\\henri-telemetry\\ontology\\objects.jsonl   (append-only)
Schema: henri-ontology/references/ontology-schema.md

Rules enforced here:
  * every record carries evidence_class + probe_ref (schema rule 1)
  * record_id is deterministic from record content: ont-<kind>-<12 hex>
  * append-only; nothing is rewritten in place
  * no benchmark target or evaluation answer enters a record (schema rule 3)

Run:  python tools/ontology_emit_zone_a_selfplay.py [--dry-run]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone

STORE = os.path.join(os.path.expanduser("~"), "henri-telemetry", "ontology", "objects.jsonl")
BANK = "ca4bb787"
CREATED = "2026-09-26T21:10:00+00:00"  # sprint session
AUDIT_COMMIT = "d97ddd9"


def rid(kind: str, key: str) -> str:
    """Deterministic record id: ont-<kind>-<12 hex>."""
    digest = hashlib.sha256(f"{kind}|{key}".encode("utf-8")).hexdigest()[:12]
    return f"ont-{kind}-{digest}"


def build_records() -> list[dict]:
    recs: list[dict] = []

    # ---------------------------------------------------------------- Evidence
    e_orx_paper = rid("evidence", "orx paper 2609.30063 exists")
    recs.append({
        "record_id": e_orx_paper,
        "kind": "evidence",
        "supports": [],
        "probe": {
            "tool": "orx",
            "query": "--paper 2609.30063 (Self-Play Pretraining with Zero Data)",
            "argv": ["python", "-m", "agentic_graph.autoresearch_cli", "--paper", "2609.30063"],
            "artifact_ref": "HENRI V2/.ar/cowsik_2609.30063.txt",
            "artifact_sha256": None,
            "exit_code": 0,
            "observed_at_utc": CREATED,
        },
        "note": ("status=pass. Real authors: Cowsik, Dolev, Cohen, Levine, Li, Goodman, "
                 "De Luca (TAU / Stanford / LAPTh). The paper EXISTS. The retrieved report is "
                 "abstract-level: it does NOT contain the preconditioned gradient-alignment "
                 "formula, so the formula's provenance stays HYPOTHESIS."),
        "evidence_class": "OBSERVED",
        "probe_ref": "orx://paper/2609.30063",
        "created_utc": CREATED,
        "status": "active",
    })

    e_orx_hope = rid("evidence", "orx HOPE attribution falsified")
    recs.append({
        "record_id": e_orx_hope,
        "kind": "evidence",
        "supports": [],
        "probe": {
            "tool": "orx",
            "query": "nested learning HOPE hierarchical optimization predictive evolution Behrouz",
            "argv": ["python", "-m", "agentic_graph.autoresearch_cli", "--query",
                     "nested learning HOPE hierarchical optimization predictive evolution Behrouz",
                     "--limit", "5"],
            "artifact_ref": "cache/spillover/orx_hope_query.txt",
            "artifact_sha256": None,
            "exit_code": 0,
            "observed_at_utc": CREATED,
        },
        "note": ("FALSIFIED attribution: the retrieved literature contains 'Nested Learning: The "
                 "Illusion of Deep Learning Architectures' (2512.24695) and 'Language Models Need "
                 "Sleep: Learning to Self-Modify and Consolidate Memories' (2606.03979), but NO "
                 "work named 'HOPE = Hierarchical Optimization and Predictive Evolution' by "
                 "Behrouz et al. The only HOPE hit is 'Hilbert Operator for Progressive Encoding' "
                 "(2607.21366), a different work. The synthesis document's HOPE citation does not "
                 "verify. Treat its dreaming-schedule claims as HYPOTHESIS."),
        "evidence_class": "FALSIFIED",
        "probe_ref": "orx://query/hope-behrouz",
        "created_utc": CREATED,
        "status": "active",
    })

    e_gpu = rid("evidence", "RTX 5090 ships 170 SMs not 192")
    recs.append({
        "record_id": e_gpu,
        "kind": "evidence",
        "supports": [],
        "probe": {
            "tool": "web-search",
            "query": "RTX 5090 GB202 SM count 170 vs 192 streaming multiprocessors enabled",
            "argv": ["web_search", "RTX 5090 GB202 SM count"],
            "artifact_ref": "https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/"
                            "nvidia-rtx-blackwell-gpu-architecture.pdf",
            "artifact_sha256": None,
            "exit_code": 0,
            "observed_at_utc": CREATED,
        },
        "note": ("GB202 die = 192 SMs; the RTX 5090 ships 170 of them enabled. The source blueprint's "
                 "'192 SMs' is the die, not the shipping card. Its SM-partition budgets are ~13% "
                 "optimistic. Correct before any per-SM partition arithmetic."),
        "evidence_class": "OBSERVED",
        "probe_ref": "web://nvidia-rtx-blackwell-architecture-pdf",
        "created_utc": CREATED,
        "status": "active",
    })

    e_m1 = rid("evidence", "M1 kernel tests 28 of 28 pass")
    recs.append({
        "record_id": e_m1,
        "kind": "evidence",
        "supports": [],
        "probe": {
            "tool": "file-read",
            "query": "pytest tests/unit/test_gradient_alignment_reward.py",
            "argv": ["python", "-m", "pytest", "tests/unit/test_gradient_alignment_reward.py",
                     "-q", "--no-header"],
            "artifact_ref": "HENRI V2/tests/unit/test_gradient_alignment_reward.py",
            "artifact_sha256": None,
            "exit_code": 0,
            "observed_at_utc": CREATED,
        },
        "note": ("28 passed in 1.05s on branch carrier/zone-a-selfplay at audit base d97ddd9. "
                 "Local run only: interpreter is torch 2.13.0+CPU, cuda=False. This is a unit-level "
                 "result, NOT CUDA verification and NOT a task outcome."),
        "evidence_class": "OBSERVED",
        "probe_ref": "pytest://tests/unit/test_gradient_alignment_reward.py",
        "created_utc": CREATED,
        "status": "active",
    })

    # ------------------------------------------------------------------- Term
    t_reward = rid("term", "preconditioned gradient-alignment reward")
    recs.append({
        "record_id": t_reward,
        "kind": "term",
        "label": "Preconditioned gradient-alignment reward",
        "definition": (
            "Self-play generator reward r_i = |< grad_theta L(y_i; theta_e), P_e @ delta_theta_e >| "
            "where P_e = diag(eta_lr / (sqrt(v_e) + eps)) is the AdamW diagonal step operator and "
            "delta_theta_e = theta_{p(e)} - theta_e is the parameter displacement over a growing "
            "lookback p(e) = floor(e/2). It is zero for mastered data (gradient -> 0) and zero for "
            "gradients orthogonal to the coherent consolidation direction; it is maximal only at "
            "the epistemic frontier. Formula provenance: HENRI synthesis document "
            "HENRI-ARCH-2026-SELFPLAY-DREAMING-V1; the primary paper exists but the retrieved text "
            "does not contain the formula. HYPOTHESIS-FORM."
        ),
        "bank": BANK,
        "semantic_relations": [
            {"type": "derivation", "target": e_orx_paper},
        ],
        "evidence_class": "HYPOTHESIS",
        "probe_ref": e_m1,
        "created_utc": CREATED,
        "status": "active",
    })

    # ---------------------------------------------------------------- Mapping
    m_kernel = rid("mapping", "reward kernel locator")
    recs.append({
        "record_id": m_kernel,
        "kind": "mapping",
        "term_id": t_reward,
        "target": {
            "system": "repo",
            "locator": "HENRI V2/henri_gradient_alignment_reward.py#alignment_reward",
        },
        "verified_by_probe": e_m1,
        "evidence_class": "OBSERVED",
        "probe_ref": e_m1,
        "created_utc": CREATED,
        "status": "active",
    })

    # -------------------------------------------------------------- Constraint
    c_disc = rid("constraint", "A4 discrimination rule")
    recs.append({
        "record_id": c_disc,
        "kind": "constraint",
        "term_id": t_reward,
        "rule": ("The preconditioner must change the candidate RANKING when the second moment v is "
                 "non-uniform. Under a uniform v it may only rescale all candidates. A kernel whose "
                 "preconditioner only rescales is tautological and must be rejected."),
        "applies_when": "any self-play reward used to select the next generated program",
        "evidence_class": "OBSERVED",
        "probe_ref": e_m1,
        "created_utc": CREATED,
        "status": "active",
    })

    c_scale = rid("constraint", "no local CPU run is CUDA verification")
    recs.append({
        "record_id": c_scale,
        "kind": "constraint",
        "term_id": t_reward,
        "rule": ("A unit-test pass on the local +CPU interpreter is a unit-level result only. It is "
                 "never CUDA verification, never a performance result, and never a task outcome."),
        "applies_when": "reporting any HENRI kernel result",
        "evidence_class": "OBSERVED",
        "probe_ref": e_m1,
        "created_utc": CREATED,
        "status": "active",
    })

    # ------------------------------------------- Sagnac stage-separation records
    t_sagnac = rid("term", "two-stage Sagnac gating")
    recs.append({
        "record_id": t_sagnac,
        "kind": "term",
        "label": "Two-stage Sagnac gating (search veto vs crystallization setpoint)",
        "definition": (
            "The normalized Sagnac delta is used by TWO gates at two stages, so the two published "
            "values are not in conflict. (1) Epistemic search veto tau_veto = 0.35: upper bound of "
            "permissible phase mismatch during candidate tree search; a branch above it is pruned "
            "(Q -> -inf). Equivalent to normalized homodyne transmission S >= 0.65. "
            "(2) Noetherian physical admissibility S <= 0.0431: fine-grained conservation check and "
            "thermodynamic crystallization setpoint before a write to Zone C; as S -> 0.0431 the "
            "Langevin thermostat cools to T_floor."
        ),
        "bank": BANK,
        "semantic_relations": [
            {"type": "association", "target": "ont-term-de9f6bf2f1a4"},
        ],
        "evidence_class": "INFERRED",
        "probe_ref": "mcp__notebooklm__notebook_query@ca4bb787",
        "created_utc": CREATED,
        "status": "active",
    })

    c_stage = rid("constraint", "do not apply crystallization setpoint to search veto")
    recs.append({
        "record_id": c_stage,
        "kind": "constraint",
        "term_id": t_sagnac,
        "rule": ("Do NOT replace the search-stage veto threshold in arc_sagnac_veto.py "
                 "(DEFAULT_EPSILON_HARD = 0.35) with the 0.0431 crystallization setpoint. They govern "
                 "different stages. The search veto keeps the search-stage threshold; 0.0431 belongs "
                 "on the pre-Zone-C-write admissibility path."),
        "applies_when": "Sprint 3 wave-interferometric Sagnac kernel work",
        "evidence_class": "INFERRED",
        "probe_ref": "mcp__notebooklm__notebook_query@ca4bb787",
        "created_utc": CREATED,
        "status": "active",
    })

    m_veto = rid("mapping", "search veto locator")
    recs.append({
        "record_id": m_veto,
        "kind": "mapping",
        "term_id": t_sagnac,
        "target": {
            "system": "repo",
            "locator": "HENRI V2/arc_sagnac_veto.py#DEFAULT_EPSILON_HARD,_sagnac_similarity",
        },
        "verified_by_probe": "ont-evidence-c45cfb79e4be",
        "evidence_class": "OBSERVED",
        "probe_ref": "ont-evidence-c45cfb79e4be",
        "created_utc": CREATED,
        "status": "active",
    })

    # block, per schema: no benchmark target may enter a record
    for r in recs:
        assert "benchmark" not in json.dumps(r).lower() or "not a task outcome" in json.dumps(r).lower()
    return recs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    recs = build_records()

    existing_ids: set[str] = set()
    if os.path.exists(STORE):
        with open(STORE, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    existing_ids.add(json.loads(line)["record_id"])
                except (json.JSONDecodeError, KeyError):
                    pass

    new = [r for r in recs if r["record_id"] not in existing_ids]
    dup = len(recs) - len(new)

    # validity check per schema rule 1
    required = ("record_id", "kind", "evidence_class", "probe_ref", "created_utc", "status")
    for r in new:
        for field in required:
            assert r.get(field), f"{r['record_id']} missing required field {field}"

    print(f"built={len(recs)} new={len(new)} already-present={dup}")
    for r in new:
        print(f"  {r['kind']:10s} {r['record_id']}  [{r['evidence_class']}]")

    if args.dry_run:
        print("DRY RUN — nothing written")
        return 0

    with open(STORE, "a", encoding="utf-8") as fh:
        for r in new:
            fh.write(json.dumps(r, sort_keys=True) + "\n")

    with open(STORE, "r", encoding="utf-8") as fh:
        total = sum(1 for ln in fh if ln.strip())
    print(f"APPENDED {len(new)} records to {STORE} (total now {total})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
