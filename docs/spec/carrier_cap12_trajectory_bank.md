# Carrier CAP12 — Fresh 12-Env K3-Cohort Trajectory Bank (deficit #2 remedy)

Document Identifier: `HENRI-SPEC-2026-09-V3-CARRIER-CAP12-BANK`
Parent: audit doc `HENRI_Architecture_Audit__SOTA_VLA_Deficits_and_RTX_5090_Execution_Analysis.md` (SHA `4c88bdcc…`, disposition #2 = `BLOCKED_MISSING_PREMISE`; the f3v2 bank covers only 7 of the 12 K3-cohort envs, so the K3 KG2/KG6 seal basis was 7 envs and `cn04 dc22 lf52 ls20 m0r0` ran goal-unavailable).
Date: 2026-09-02. Branch: `feat/carrier-k3-empirical-koopman`. Base: `591b526`.

## 1. Authorization

User approval (verbatim, 2026-09-02): "Approved to execute — Fresh 12-env trajectory capture on vast-5090 — closes audit deficit #2 and restores a full 12-env seal basis."
This document IS the sealed instrument: `HENRI-AUTH-2026-09-V3-CARRIER-CAP12-DISPATCH` (APPROVE_REMOTE_RUN by the quoted message). No push; branch stays local.

## 2. Premise (OBSERVED)

- Live f3v2 bank (`/root/f3-run/telemetry/f3_bank_capture_v2`) covers 12 envs: ar25 bp35 cd82 ft09 g50t ka59 lp85 sb26 sc25 sk48 tr87 wa30 — only 7 overlap the sealed K3 dispatch cohort.
- The sealed K3 dispatch cohort (receipt `ecb01252…`) = ar25-0c556536 bp35-0a0ad940 cd82-fb555c5d cn04-2fe56bfb dc22-fdcac232 ft09-0d8bbf25 g50t-5849a774 ka59-38d34dbb lf52-271a04aa lp85-305b61c3 ls20-9607627b m0r0-492f87ba. Five of them (cn04 dc22 lf52 ls20 m0r0) have ZERO bank rows → excluded from the KG2/KG6 seal basis.
- Environment version caches for all 12 cohort envs exist in the dispatch worktree `/workspace/henri-k3-dispatch/environment_files/` (25 dirs; the 5 void envs present with the version ids above).

## 3. Scope (bounded; capture ONLY; no engine, no policy change, no K3 hooks)

Reuse the existing, SHA-pinned f3 capture machinery (already on this branch at base `2f9bc57`, identical bytes across /root/f3-run and /workspace/henri-k3-dispatch):

| Artifact | SHA-256 (canonical LF) | Role |
|---|---|---|
| `HENRI V2/experiments/verification/f3_capture_driver.py` | `d9c80d2cdb199236da1bd3795eed4f44958c9dbb95471d04440eacbe4e4738e1` | per-env bounded attempts (floor 100, cap 150, max 5) |
| `HENRI V2/experiments/verification/f3_merge_banks.py` | `ee0733e305f8b91f016102759b7a962a52848cb487f162224710a70ecf4aaf46` | concat/trim/realign, authorized-only |
| `HENRI V2/experiments/verification/f3_capture_finalize.py` | `6bc780d8f2390cad84dac16f5865264ed5fd546e1d65978f355915036a80bb34` | entropy/diversity/record gates |

No new capture code is written. CAP12 launches the EXISTING driver against the sealed 12-env K3 cohort into a NEW bank directory (`f3_bank_capture_v3`), then finalizes.

Dispatch command (remote, from `/workspace/henri-k3-dispatch`, env from `/workspace/zonec_prod.env`, `ZONE_C_ENV=prod`, `HENRI_ARC_ACTION_PAYLOADS=1`, `PYTHONPATH="HENRI V2"`):

```bash
/venv/main/bin/python "HENRI V2/experiments/verification/f3_capture_driver.py" \
  --attempts-dir /root/f3-run/capture_attempts_v3 \
  --out /root/f3-run/telemetry/f3_bank_capture_v3 \
  --run-id production_run_k3cohort \
  --envs ar25-0c556536 bp35-0a0ad940 cd82-fb555c5d cn04-2fe56bfb dc22-fdcac232 \
         ft09-0d8bbf25 g50t-5849a774 ka59-38d34dbb lf52-271a04aa lp85-305b61c3 \
         ls20-9607627b m0r0-492f87ba \
  --steps 150 --floor 100 --env-cap 150 --max-attempts 5 --seed 20260903
```

The runner (`production_arc_run.py`) captures with `HENRI_ARC_TRAJECTORY_BANK=1` (authorized live arcade tuples; zero-pretraining invariant: no evaluation caches, no demos, no synthesized rows — the driver asserts `data_source=authorized` and digest parity in the merge).

## 4. Verification gates (CAP12)

| Gate | Metric (f3_capture_receipt.json) | Bound |
|---|---|---|
| CAP12-REC | per-env counts over the sealed 12 cohort | ≥ 100 records/env for ALL 12 (incl. the 5 formerly void) |
| CAP12-DIV | `bank_entropy_nats` + per-action support | ≥ 1.70 nats and ≥ 30 rows per union-vocab action (finalize gate) |
| CAP12-INT | driver/merge/finalize digests on the remote | equal to the pinned SHAs above |
| CAP12-SHA | npz/jsonl/manifest SHA-256 recorded + copied locally | full hashes preserved in the sealed results doc |

Verdict classes: `ENTROPY_GATE_PASS` (then CAP12 COMPLETE) | `BLOCKED_ENTROPY_GATE` | `BLOCKED_RECORD_FLOOR` (driver exits 2 fail-loud, artifacts preserved) | `BLOCKED_DIGEST_DRIFT` (abort, no bank use).

## 5. Relationship to the K3 chain

CAP12 does NOT reopen the sealed K3 verdict (`K3_GATE_KG5_LATENCY_FAILED`, `591b526`). It produces the premise for a FUTURE full-basis K3 re-dispatch: the bank at `f3_bank_capture_v3` replaces `f3v2` as the goal source for the next instrument. Whether the 5 newly captured envs become goal-available is MEASURED at that dispatch (same `p1_bind_env_goal` mechanism), never assumed.

## 6. Disclosures

- GPU exclusivity: the capture runs when no other CUDA job is active (observed idle: 2 MiB used of 32,607 MiB). The A1 Triton measurement run is scheduled AFTER capture completes (never concurrent — competing VRAM produces misleading failures in untouched tests).
- Fresh seed 20260903 (not the f3v2 seed 20260830); per-(env,attempt) seeds derived by the driver (`seed + env_index*1000 + attempt`).
- Runs are bounded: 12 envs × ≤5 attempts × 150 steps; driver exits 2 if any env misses the floor.
