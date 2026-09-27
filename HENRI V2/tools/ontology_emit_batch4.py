#!/usr/bin/env python
"""Ontology batch 4 — the Five-Gaps premise audit and the resonator prior art.

Appends to: C:\\Users\\chan\\henri-telemetry\\ontology\\objects.jsonl
Schema: henri-ontology/references/ontology-schema.md
  * every record carries evidence_class + probe_ref (rule 1)
  * no benchmark target / no evaluation answer enters a record (rule 3)
  * rejected candidates are logged so the loop does not re-propose them blind

Run:  python tools/ontology_emit_batch4.py [--dry-run]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os

STORE = os.path.join(os.path.expanduser("~"), "henri-telemetry", "ontology", "objects.jsonl")
BANK = "ca4bb787"
CREATED = "2026-09-27T08:10:00+00:00"
HEAD = "3ff5cb5"


def rid(kind: str, key: str) -> str:
    return f"ont-{kind}-{hashlib.sha256(f'{kind}|{key}'.encode()).hexdigest()[:12]}"


def build() -> list[dict]:
    recs: list[dict] = []

    # ---------------------------------------------------------------- EVIDENCE
    e_taskfunctor = rid("evidence", "arc_task_functor is diagonal ridge not matrix inverse")
    recs.append({
        "record_id": e_taskfunctor,
        "kind": "evidence",
        "supports": [],
        "probe": {
            "tool": "file-read",
            "query": "arc_task_functor.py compute_optimal_task_functor body",
            "argv": ["read_file", "HENRI V2/arc_task_functor.py", "offset=111", "limit=40"],
            "artifact_ref": "HENRI V2/arc_task_functor.py",
            "artifact_sha256": None,
            "exit_code": 0,
            "observed_at_utc": CREATED,
        },
        "note": (
            f"FALSIFIES a supplied-document claim at {HEAD}. The document asserted "
            "arc_task_functor.py holds W = (X^T X)^(-1) X^T Y (a matrix inverse). The live "
            "code is `numerator / (denom + reg_lambda)` -- an elementwise PER-SLOT DIAGONAL "
            "ridge solve reducing only over the demo axis. The module docstring records the "
            "FFT/circulant family was FALSIFIED and that THIS diagonal family reaches "
            "held-out 0.4368 / in-sample 0.7645 on 60 real ARC tasks, vs ~0.00 for the FFT "
            "family. Therefore 'excise the one-shot linear regression' would target a family "
            "that is already the measured-best one."
        ),
        "evidence_class": "FALSIFIED",
        "probe_ref": "repo://arc_task_functor.py#compute_optimal_task_functor",
        "created_utc": CREATED,
        "status": "active",
    })

    e_resonator = rid("evidence", "tripartite resonator exists and is VOID on real ARC")
    recs.append({
        "record_id": e_resonator,
        "kind": "evidence",
        "supports": [],
        "probe": {
            "tool": "file-read",
            "query": "git log -1 16d573c on carrier/aaii-v43 (resonator verdict)",
            "argv": ["git", "show", "16d573c", "--stat"],
            "artifact_ref": "carrier/aaii-v43:HENRI V2/experiments/verification/"
                            "KILL_PREREGISTRATION_resonator_carrier.md",
            "artifact_sha256": None,
            "exit_code": 0,
            "observed_at_utc": CREATED,
        },
        "note": (
            "PRIOR ART, own probe. A tripartite VSA resonator ALREADY EXISTS on "
            "carrier/aaii-v43 @ 16d573c: henri_resonator.py (TripartiteResonator, "
            "FactorCodebook, ResonatorConfig, ResonatorResult), arc_tripartite_resonator.py, "
            "test_tripartite_resonator.py, test_resonator_kill_gates.py, and a pre-registered "
            "KILL file. Commit verdict: instrument VALIDATED (28/28, mutation gate 4/4 -> 4 "
            "mutations CAUGHT, 2 controls NOT_CAUGHT), VOID on real ARC. Measured scenario "
            "table: solvable 1.0000001 vs shuffled +0.3672; identity_truth VOID (tie, correct); "
            "impossible VOID (cap hit, 4 controls beat); REAL_ARC treatment 0.26858 LOSES to "
            "identity and diag_ls, iteration cap hit (non-convergent). So the same instrument "
            "already returned an honest negative; re-running it unchanged would be a blind retry."
        ),
        "evidence_class": "OBSERVED",
        "probe_ref": "repo://carrier/aaii-v43@16d573c/resonator-verdict",
        "created_utc": CREATED,
        "status": "active",
    })

    e_nondiag = rid("evidence", "mask and colour are not per-slot diagonal")
    recs.append({
        "record_id": e_nondiag,
        "kind": "evidence",
        "supports": [e_resonator],
        "probe": {
            "tool": "file-read",
            "query": "KILL_PREREGISTRATION_resonator_carrier.md measured-facts table",
            "argv": ["read_file", "KILL_PREREGISTRATION_resonator_carrier.md"],
            "artifact_ref": "carrier/aaii-v43:HENRI V2/experiments/verification/"
                            "KILL_PREREGISTRATION_resonator_carrier.md",
            "artifact_sha256": None,
            "exit_code": 0,
            "observed_at_utc": CREATED,
        },
        "note": (
            "Measured on the live encoder: colour change 3->5 gives |z_B/z_A| mean 5.2552 with "
            "std 37.94 and angle std 0.928 -> colour is NOT a per-slot diagonal operator. "
            "Enclosure (solid vs ring) gives |ratio| mean 2193.7 with std 14975.8 -> the mask is "
            "NOT diagonal and must be SEARCHED, not solved. Only roll is provably diagonal "
            "(exact to ~3e-4). CONSEQUENCE: any 'solve for the factors' design is forbidden by "
            "measurement; the resonator must search over explicit codebooks."
        ),
        "evidence_class": "OBSERVED",
        "probe_ref": "repo://carrier/aaii-v43@16d573c/kill-prereg-measured-facts",
        "created_utc": CREATED,
        "status": "active",
    })

    e_sagnac = rid("evidence", "arc_sagnac_veto is 0.35 not cosine 0.95")
    recs.append({
        "record_id": e_sagnac,
        "kind": "evidence",
        "supports": [],
        "probe": {
            "tool": "file-read",
            "query": "grep -nE 'cosine|0.95|DEFAULT_EPSILON' arc_sagnac_veto.py",
            "argv": ["grep", "-nE", "cosine|0\\.95|DEFAULT_EPSILON", "arc_sagnac_veto.py"],
            "artifact_ref": "HENRI V2/arc_sagnac_veto.py",
            "artifact_sha256": None,
            "exit_code": 0,
            "observed_at_utc": CREATED,
        },
        "note": (
            f"FALSIFIES a supplied-document claim at {HEAD}. The document asserted "
            "arc_sagnac_veto.py uses `cosine_similarity > 0.95`. The live file has "
            "DEFAULT_EPSILON_HARD = 0.35 and the string '0.95' occurs NOWHERE in it. The live "
            "gate is advisory and its contract is (delta, coherence, vetoed, status). "
            "This is consistent with the recorded two-stage split: 0.35 = search veto, "
            "0.0431 = pre-Zone-C crystallization setpoint."
        ),
        "evidence_class": "FALSIFIED",
        "probe_ref": "repo://arc_sagnac_veto.py#DEFAULT_EPSILON_HARD",
        "created_utc": CREATED,
        "status": "active",
    })

    e_bench = rid("evidence", "coding benchmark declares no cli flags")
    recs.append({
        "record_id": e_bench,
        "kind": "evidence",
        "supports": [],
        "probe": {
            "tool": "file-read",
            "query": "count add_argument + scicode mentions in the coding benchmark",
            "argv": ["grep", "-cE", "add_argument", "execute_authentic_coding_benchmark.py"],
            "artifact_ref": "HENRI V2/execute_authentic_coding_benchmark.py",
            "artifact_sha256": None,
            "exit_code": 0,
            "observed_at_utc": CREATED,
        },
        "note": (
            "FALSIFIES a supplied-document command. The document directs re-running "
            "`execute_authentic_coding_benchmark.py --benchmark scicode --window 48`. That file "
            "declares ZERO argparse arguments and mentions 'scicode' once; its docstring targets "
            "OpenAI HumanEval. The command cannot run as written. Related recorded ceiling: "
            "UHR05_egress_emittable_ceiling.md, UHR05_no_henri_code_generator.md."
        ),
        "evidence_class": "FALSIFIED",
        "probe_ref": "repo://execute_authentic_coding_benchmark.py#argparse",
        "created_utc": CREATED,
        "status": "active",
    })

    e_shards = rid("evidence", "heldout gate and shards smoke verified")
    recs.append({
        "record_id": e_shards,
        "kind": "evidence",
        "supports": [],
        "probe": {
            "tool": "file-read",
            "query": "stage0_seeding_run.py smoke 50k executions, held-out + shards",
            "argv": ["python", "stage0_seeding_run.py", "--n-executions", "50000",
                     "--batch-size", "512", "--heldout-samples", "128", "--shard-dir", "..."],
            "artifact_ref": "HENRI V2/telemetry/smoke_heldout/summary.json",
            "artifact_sha256": None,
            "exit_code": 0,
            "observed_at_utc": CREATED,
        },
        "note": (
            "The driver previously could NOT satisfy the ratified gate (promotion on HELD-OUT "
            "curriculum progress): it had no held-out set. build_heldout() now gives a batch that "
            "is disjoint on three axes (different generator seed, different program length, "
            "materialised once). Smoke: IDENTITY 49,964 x 33 = 1,648,812 == budget_learner_tokens "
            "EXACT; shard bytes == tokens IDENTICAL; held-out loss 5.5809 -> 0.3305 monotone "
            "(4.22/1.60/0.52/0.33 at rounds 20/40/60/80); 16,658 exec/s at 16 threads. "
            "Summary keys are named final_loss_NOT_PROMOTION / reward_mean_NOT_PROMOTION so a "
            "training-loss or reward gate cannot be mistaken for the promotion signal."
        ),
        "evidence_class": "OBSERVED",
        "probe_ref": "local://stage0_seeding_run/heldout-smoke",
        "created_utc": CREATED,
        "status": "active",
    })

    # -------------------------------------------------------- CONSTRAINT
    t_res = rid("term", "tripartite VSA resonator")
    c_noblink = rid("constraint", "do not retry the resonator blind")
    recs.append({
        "record_id": c_noblink,
        "kind": "constraint",
        "term_id": t_res,
        "rule": (
            "The tripartite resonator returned VOID on real ARC at carrier/aaii-v43@16d573c "
            "(0.26858, losing to identity, iteration cap hit). Do NOT re-run it unchanged and do "
            "NOT present it as a new remedy. It may be re-entered ONLY as a default-OFF third arm "
            "under a pre-registered A/B against the live diagonal-ridge control (held-out 0.4368 "
            "on 60 real ARC tasks), with acceptance hit_rate - identity_rate >= 0.30 and "
            "convergence below the iteration cap. A repeat VOID is logged, not retried."
        ),
        "applies_when": "any proposal to replace fac-arc_task_functor's operator",
        "evidence_class": "OBSERVED",
        "probe_ref": e_resonator,
        "created_utc": CREATED,
        "status": "active",
    })

    c_promote = rid("constraint", "promotion signal is held-out only")
    recs.append({
        "record_id": c_promote,
        "kind": "constraint",
        "rule": (
            "Stage-0 promotion is gated on heldout_progress ONLY. final_loss and reward_mean are "
            "recorded for diagnosis and are NEVER promotion signals: a training-loss or reward "
            "gate is satisfiable by memorisation, which is exactly how the two earlier reward "
            "gates failed. Any report that cites a falling training loss as evidence of learning "
            "is rejected."
        ),
        "applies_when": "any Stage-0 run report or promotion decision",
        "evidence_class": "OBSERVED",
        "probe_ref": e_shards,
        "created_utc": CREATED,
        "status": "active",
    })

    c_datapath = rid("constraint", "verify the file before refactoring toward it")
    recs.append({
        "record_id": c_datapath,
        "kind": "constraint",
        "rule": (
            "A supplied document's FILE-LEVEL claims must be verified against the live file "
            "before any refactor. Measured this sprint: 3 of 4 structural claims were "
            "misattributed (a diagonal ridge solve described as a matrix inverse; a 0.35 "
            "advisory gate described as a 0.95 cosine check; a CLI invocation that the file "
            "cannot accept). Refactoring toward a misdescribed target destroys working code."
        ),
        "applies_when": "any directive that names a file and an equation to replace",
        "evidence_class": "OBSERVED",
        "probe_ref": e_taskfunctor,
        "created_utc": CREATED,
        "status": "active",
    })

    # -------------------------------------------------------- TERM
    recs.append({
        "record_id": t_res,
        "kind": "term",
        "label": "Tripartite VSA resonator (T_task = R_torus x M_mask x Spin(3))",
        "definition": (
            "Iterative factorized-search operator that estimates three factors by unbinding the "
            "others from the target and snapping each estimate to its nearest codebook entry. "
            "Implemented and validated as an INSTRUMENT at carrier/aaii-v43@16d573c "
            "(28/28 tests, mutation gate CAUGHT 4/6 with 2 NOT_CAUGHT controls). Its measured "
            "verdict on a real ARC task is VOID: treatment 0.26858 loses to identity, and the "
            "relaxation hit its iteration cap. Structural reason from the same record: the mask "
            "factor is NOT diagonal (|ratio| std 14975.8) and colour is NOT a per-slot diagonal "
            "op (std 37.94), so those factors must be searched over explicit codebooks, never "
            "solved. Status: instrument VALIDATED, mechanism VOID on this substrate."
        ),
        "bank": BANK,
        "semantic_relations": [{"type": "derivation", "target": e_resonator}],
        "evidence_class": "OBSERVED",
        "probe_ref": e_resonator,
        "created_utc": CREATED,
        "status": "active",
    })

    return recs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

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
    if a.dry_run:
        print("DRY RUN - nothing written")
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
