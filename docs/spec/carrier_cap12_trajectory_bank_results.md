# Carrier CAP12 — Fresh 12-Env K3-Cohort Trajectory Bank: Sealed Results

Document Identifier: `HENRI-SPEC-2026-09-V3-CARRIER-CAP12-SEALED-RESULTS`
Instrument: `HENRI-SPEC-2026-09-V3-CARRIER-CAP12-BANK` (commit `3f1927b`)
Branch: `feat/carrier-k3-empirical-koopman`. Local only; no push.
Date: 2026-09-03. Remote: vast-5090. Generator: `f3_capture_finalize.py@6bc780d8f2390cad` (pinned digest parity verified on `/workspace/henri-k3-dispatch`).

## 1. Run identity (OBSERVED)

- Launcher: `/tmp/cap12_launch.sh` (LF-normalized; SHA verified before launch).
- Driver PID 920587; per-env runner attempts spawned per (env, attempt); watchdog `proc_a8f78a154152` fired on `CAP12_FINALIZE_DONE rc=0`.
- Capture: 2026-09-03 06:09–09:12 UTC (finalize receipt `utc: 2026-09-03T09:12:23Z`).
- Zone C prod env sourced (`/workspace/zonec_prod.env`); GPU idle 2 MiB before launch; exclusivity held through the run (no concurrent CUDA jobs observed).

## 2. Gate results

| Gate | Metric | Bound | Result |
|---|---|---|---|
| CAP12-REC | per-env counts, all 12 envs (incl. 5 formerly void) | ≥ 100/env | **PASS** — ar25 133, bp35 149, cd82 100, cn04 150, dc22 128, ft09 150, g50t 130, ka59 100, lf52 128, lp85 150, ls20 129, m0r0 150; total 1,597 |
| CAP12-DIV | `bank_entropy_nats` ≥ 1.70; per-action rows ≥ 30 | H ≥ 1.70, n_a ≥ 30 | **FAIL** — H = 1.6807 (max 1.9459); ACTION5 = 0 rows |
| CAP12-INT | driver/merge/finalize digests on remote | equal to sealed pins | **PASS** — `d9c80d2c…` / `ee0733e3…` / `6bc780d8…` (sha256sum on dispatch tree) |
| CAP12-SHA | npz/jsonl/manifest hashes recorded | full hashes in this doc | **PASS** — see §3 |

Verdict: **BLOCKED_ENTROPY_GATE** (sealed class; artifacts preserved).

## 3. Artifacts (remote `/root/f3-run/telemetry/f3_bank_capture_v3/`)

| Artifact | Size | SHA-256 |
|---|---|---|
| `trajectories_production_run_k3cohort.npz` | 418,656,369 B | `36064f5156055cb461ce9859484352c782cef23b7b0689af9b86271bc301ccb8` |
| `trajectories_production_run_k3cohort.jsonl` | 140,173 B | `5b3d812791d8eb7b75d5961782f12d0b21a767f6f7bf89b29e04acad82501db6` |
| `trajectories_production_run_k3cohort_manifest.json` | 996 B | (recorded in receipt) |
| `f3_capture_receipt.json` | 1,945 B | schema `f3-capture-receipt.v1`, `record_count: 1597` |
| `f3_merge_receipt.json` | 1,174 B | merge digest receipt |

Local mirror (Temp): receipt, merge receipt, manifest, and jsonl; jsonl SHA verified equal (`5b3d8127…`). npz not mirrored (418 MB; remote SHA recorded above).

## 4. Interpretation

- The 5 formerly void envs (cn04 dc22 lf52 ls20 m0r0) each captured ≥ 100 records — the premise of "no records" is closed. A full 12-env basis now EXISTS as raw records.
- The bank FAILS the diversity gate: H = 1.6807 < 1.70 and ACTION5 has zero rows across all 1,597 records. Per-env frame-delta means reproduce the known behavioral reality: ft09 and lp85 = 0.0, dc22 0.00143, lf52 0.00024 (ACTION6-masked / payload-driven envs; low policy-induced frame displacement), ls20 0.0411 highest.
- Consequence: `f3_bank_capture_v3` cannot serve as an authorized K3 goal source in its present state. Whether per-env subsets of the raw bank satisfy the goal-availability predicate (`p1_bind_env_goal`) is a MEASURED question for a future instrument, never assumed — but the sealed bank-level gate is FAIL.
- CAP12 does not reopen the K3 verdict (`591b526`, KG5 latency FAIL + KG2/KG6 FALSIFIED). Deficit #2 (audit doc disposition) is re-scoped: raw 12-env records EXIST; a diversity-qualified bank does NOT.

## 5. Governance

- Sealed instrument `3f1927b`; sealed results this commit. No capability claim. Negative/gated result recorded as a governance win with artifact path.
- Next action (unapproved): either (a) diagnose ACTION5 absence (policy/action-space reality vs capture defect) and re-capture with a diversity-directed bound, or (b) defer to A1 Triton measurement (Track A) which proceeds independently on the now-idle GPU.
