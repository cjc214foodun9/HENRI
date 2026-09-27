#!/usr/bin/env python
"""Ontology batch 3 — the Stage-0 CUDA results and the device-parity defect class.

Appends to the append-only store:
  C:\\Users\\chan\\henri-telemetry\\ontology\\objects.jsonl

Covers:
  * the decisive RAW-vs-COS-vs-COSV result, now CUDA-observed
  * the Stage-0 bounded-seeding throughput measurement with its THREE budgets
  * the "local CPU smoke cannot verify device placement" defect class
  * the scale-conflation constraint that governs the cost extrapolation

Every record carries evidence_class + probe_ref (schema rule 1). No benchmark
target and no evaluation answer enters any record (schema rule 3).

Run:  python tools/ontology_emit_batch3.py [--dry-run]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os

STORE = os.path.join(os.path.expanduser("~"), "henri-telemetry", "ontology", "objects.jsonl")
BANK = "ca4bb787"
CREATED = "2026-09-27T07:20:00+00:00"

RAW_FACTS = {
    "instance": 52826640,
    "head": "d056e22",
    "vm_executions": 10_000_172,
    "reward_evaluations": 1_250_048,
    "learner_tokens": 330_005_676,
    "wall_seconds": 1067.36,
    "exec_per_sec": 9369.1,
    "first_loss": 5.580404,
    "final_loss": 0.099681,
    "timeout_rate": 0.0291,
    "bank_size": 4096,
    "distinct_outputs": 18811,
    "seeding_tar_sha256": "fcd566c10bd5e97727e43a206309c46cfae5221e5b2f7055eefd3da08b91ed8c",
    "reward_norm_tar_sha256": "f9241aeb45e6d87f4f4cc55ec8b131672072cf49a161ff2af038fe1eac299989",
}


def rid(kind: str, key: str) -> str:
    return f"ont-{kind}-{hashlib.sha256(f'{kind}|{key}'.encode()).hexdigest()[:12]}"


def build() -> list[dict]:
    recs: list[dict] = []

    # ---------------------------------------------------------------- Evidence
    e_raw = rid("evidence", "RAW separates on CUDA")
    recs.append({
        "record_id": e_raw,
        "kind": "evidence",
        "supports": [],
        "probe": {
            "tool": "file-read",
            "query": "probe_reward_normalisation on the CUDA host, seeds 0/1/2",
            "argv": ["python3", "tools/probe_reward_normalisation.py", "--seeds", "0,1,2",
                     "--base-steps", "40", "--n-probe", "10", "--progress-steps", "10"],
            "artifact_ref": "telemetry/vast_52826640/henri_final.tar.gz",
            "artifact_sha256": RAW_FACTS["reward_norm_tar_sha256"],
            "exit_code": 0,
            "observed_at_utc": CREATED,
        },
        "note": (
            "CUDA-observed. RAW |<g, P_e dtheta>| SEPARATES on every seed: "
            "frontier>mastered TRUE and frontier>noise TRUE. COS and COSV both return "
            "FALSE/FALSE: NORMALISING DESTROYS the separation. heldout_progress_axis_exists "
            "TRUE (base learner trained on MASTERED only, so FRONTIER is genuinely novel). "
            "seed 0: MASTERED 1.411e-03, FRONTIER 1.157e-01, NOISE 8.832e-02."
        ),
        "evidence_class": "OBSERVED",
        "probe_ref": "cuda://52826640/probe_reward_normalisation",
        "created_utc": CREATED,
        "status": "active",
    })

    t_reward = rid("term", "preconditioned gradient-alignment reward")
    c_unnorm = rid("constraint", "do not normalise the M1 reward")
    recs.append({
        "record_id": c_unnorm,
        "kind": "constraint",
        "term_id": t_reward,
        "rule": (
            "Do NOT normalise the M1 reward by ||grad|| or ||grad||*||v||. Measured on CUDA "
            "over seeds 0/1/2: the RAW absolute inner product separates frontier from mastered "
            "and from noise; COS and COSV both fail to separate. Normalisation is a "
            "regression, not a fix."
        ),
        "applies_when": "any change to henri_gradient_alignment_reward.py",
        "evidence_class": "OBSERVED",
        "probe_ref": e_raw,
        "created_utc": CREATED,
        "status": "active",
    })

    # ---- Stage-0 seeding measurement
    e_seed = rid("evidence", "stage0 bounded seeding 1e7 measured")
    recs.append({
        "record_id": e_seed,
        "kind": "evidence",
        "supports": [],
        "probe": {
            "tool": "file-read",
            "query": "stage0_seeding_run.py 10^7 executions on instance 52826640",
            "argv": ["python3", "stage0_seeding_run.py", "--n-executions", "10000000",
                     "--batch-size", "512", "--seed", "0"],
            "artifact_ref": "telemetry/vast_52826640/telemetry/stage0_seeding/summary.json",
            "artifact_sha256": RAW_FACTS["seeding_tar_sha256"],
            "exit_code": 0,
            "observed_at_utc": CREATED,
        },
        "note": (
            f"THREE SEPARATE BUDGETS, never conflated: vm_executions={RAW_FACTS['vm_executions']}, "
            f"reward_evaluations={RAW_FACTS['reward_evaluations']}, "
            f"learner_tokens={RAW_FACTS['learner_tokens']}. "
            f"wall={RAW_FACTS['wall_seconds']} s, {RAW_FACTS['exec_per_sec']} exec/s, "
            f"loss {RAW_FACTS['first_loss']} -> {RAW_FACTS['final_loss']}, "
            f"timeout_rate={RAW_FACTS['timeout_rate']}, bank={RAW_FACTS['bank_size']}, "
            f"distinct_outputs={RAW_FACTS['distinct_outputs']}. "
            "EGRESS_VERIFIED: local sha256 == remote sha256. "
            "NOTE: two advisor blocks quoted DIFFERENT numbers for this same file "
            "(10000000 / 0.061398 / 13133); the own SSH read of the file wins."
        ),
        "evidence_class": "OBSERVED",
        "probe_ref": "cuda://52826640/stage0_seeding_summary",
        "created_utc": CREATED,
        "status": "active",
    })

    c_scale = rid("constraint", "CPU VM throughput is not GPU kernel throughput")
    recs.append({
        "record_id": c_scale,
        "kind": "constraint",
        "rule": (
            "The measured 9369 exec/s is the CPU REFERENCE VM (pure Python + CPU tensors; the "
            "seeding path issues no CUDA calls). The .md asks for a CUDA shared-memory VM; that "
            "kernel DOES NOT EXIST. Never present the CPU throughput as GPU throughput, and never "
            "extrapolate a cost from CPU numbers as if the GPU contributed. VM executions, reward "
            "evaluations and learner tokens are three different budgets."
        ),
        "applies_when": "any Stage-0 scaling decision or cost extrapolation",
        "evidence_class": "DERIVED",
        "probe_ref": e_seed,
        "created_utc": CREATED,
        "status": "active",
    })

    # ---- device-parity defect class
    e_dev = rid("evidence", "device parity defect found only on cuda host")
    recs.append({
        "record_id": e_dev,
        "kind": "evidence",
        "supports": [],
        "probe": {
            "tool": "file-read",
            "query": "test_zone_a_dreamer.py on the CUDA host before/after the fixture fix",
            "argv": ["python3", "-m", "pytest", "tests/unit/test_zone_a_dreamer.py", "-q"],
            "artifact_ref": "HENRI V2/tests/unit/test_zone_a_dreamer.py",
            "artifact_sha256": None,
            "exit_code": 0,
            "observed_at_utc": CREATED,
        },
        "note": (
            "OBSERVED defect class: 'RuntimeError: Expected all tensors to be on the same device, "
            "but found at least two devices, cuda:0 and cpu!'. TWO instances were found. "
            "(1) ZoneACore built HENRIVisionEncoder with device=cuda while "
            "HenriSwarmOrchestrator takes NO device argument and stays on CPU -> fixed by "
            "self.orch.to(self.dev). (2) the TEST FIXTURE built its reference with "
            "torch.randn(...) on CPU -> fixed by device=core.dev. GPU-host suite: 4 failed / 11 "
            "passed BEFORE, 107 passed across the sprint suites AFTER. Local CPU smoke could not "
            "see either instance because there every tensor is CPU. The live CUDA assertion now "
            "prints core.dev=cuda, orch=cuda:0, wave=cuda:0, adapter=cuda:0, CUDA_DEVICE_PARITY_OK."
        ),
        "evidence_class": "OBSERVED",
        "probe_ref": "cuda://52826640/test_zone_a_dreamer",
        "created_utc": CREATED,
        "status": "active",
    })

    c_dev = rid("constraint", "assert device parity on the remote preflight")
    recs.append({
        "record_id": c_dev,
        "kind": "constraint",
        "rule": (
            "Every facade must place ALL components on ONE device, and every test fixture must "
            "build its tensors on the device under test. A local CPU pass cannot verify CUDA "
            "device placement: assert tensor devices on the remote preflight before a run."
        ),
        "applies_when": "any Zone A facade, dreamer, or CUDA harness change",
        "evidence_class": "OBSERVED",
        "probe_ref": e_dev,
        "created_utc": CREATED,
        "status": "active",
    })

    # ---- the earlier gate FAILs, recorded so they are not re-proposed
    e_gate = rid("evidence", "two reward gates failed on gate defects")
    recs.append({
        "record_id": e_gate,
        "kind": "evidence",
        "supports": [t_reward],
        "probe": {
            "tool": "file-read",
            "query": "stage0_reward_discrimination_gate / stage0_reward_validity_gate",
            "argv": ["python3", "stage0_reward_discrimination_gate.py", "--seeds", "0,1,2"],
            "artifact_ref": "HENRI V2/stage0_reward_discrimination_gate.py",
            "artifact_sha256": None,
            "exit_code": 1,
            "observed_at_utc": CREATED,
        },
        "note": (
            "Two gates FAILED and BOTH failures were GATE defects, not mechanism verdicts. "
            "D1 the NOISE control was not noise (VMConfig input_stream default (0,)); D2 the "
            "reward was applied PER BATCH instead of per sample r_i; D3 no vacuousness control; "
            "D4 no consolidated lookback checkpoint; D5 CircularTapeVM resets in_idx per "
            "execute(), so a fixed random input_stream still emitted IDENTICAL bytes - a "
            "constant stream, the opposite of noise; D6 the discrimination gate trained the "
            "learner ON the frontier family, so the frontier was already mastered. "
            "RECORDED so the same designs are not re-proposed."
        ),
        "evidence_class": "FALSIFIED",
        "probe_ref": "local://stage0_reward_gates",
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
