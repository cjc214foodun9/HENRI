# HOPS-VSA Production Proxy Gate — Pre-Registration (Class 4.5)

**Doc ID:** HENRI-CLASS45-HOPS-PROXY-GATE-2026-08-21
**Date:** 2026-08-21
**Branch:** `accuracy/fidelity-remediation` @ `5652c41` (HOPS core default-OFF; Path B2 sealed `479dba4`)
**Source spec:** `HENRI_Ground-Up_VSA_Holographic_Model_Specification.md` (SHA `c304d30d…`)
**Design gates:** `experiments/verification/hops_vsa_design.md` G1–G5 (unchanged; G5 remains the only external gate)
**References:** `hops-vsa-reference-core-lessons.md`, `accuracy-first-fidelity-remediation.md`, `representation-core-audit.md`

## 1. Purpose and scope

Measure whether the HOPS `P_null = I − V Vᵀ` skeleton basis engages the **live production
qFHRR carrier** before any CUDA suite or external gate cost. This proxy is NOT acceptance.
The decisive evidence remains paired rank/margin improvement, then the paired external
HumanEval gate (G5).

## 2. Anchors (frozen BEFORE results, no post-hoc thresholds)

| Anchor | Value | Source |
|---|---|---|
| B2 Gate A /23 | rank 5/71, margin −0.0218 | sealed `479dba4` |
| B2 Gate A /35 | rank 13/71, margin −0.0342 | sealed `479dba4` |
| Random-wave removed-energy baseline | E[‖Vᵀx‖²/‖x‖²] = k/D = 1.22e-4 (D=65,536, k=8) | DERIVED |
| Acceptance standard | rank ≤ 5 AND margin ≥ 0.25 (both targets) | B2 packet, unchanged |

## 3. Frozen experimental set (paired control/treatment)

- Dataset: `data/HumanEval.jsonl.gz` SHA `b796127e635a67f9` (same bytes as `HENRI V2/data/HumanEval.jsonl.gz`).
- Targets: `HumanEval/23` (oracle body `return len(string)`), `HumanEval/35` (`return max(l)`).
- Pool: `WaveASTDecoder(qFHRREpistemicCodec).decode(...)` — must be exactly 71 candidates with exactly one oracle each (assert, fail-closed otherwise).
- Waves: `decoder._wave(...)` — float32 unit-normalized continuous wave; the
  uint8 Z_256 ring boundary is crossed inside `_wave` (encode_text ring ->
  `(c/(k_bins-1))*2-1` -> `F.normalize`). `.to(device)` before HOPS scoring
  (runner-mirror). No second ring conversion at the probe boundary.
- **Control** = rank by raw real-wave cosine vs goal. **Treatment** = rank by P_null-projected cosine vs goal (vetoed candidates sink, runner-mirror). Same waves, same seed, same hardware, same item order.

## 4. Pre-registered gates (conjunctive PASS)

| Gate | Criterion | Fail ⇒ |
|---|---|---|
| **P1 carrier engagement** | mean removed-energy fraction ‖Vᵀx‖²/‖x‖² ≥ 1e-3 across production goal+candidate waves (8× random baseline) AND mean residual norm fraction ≥ 0.5 (no collapse) | FALSIFIED |
| **P2 paired rank/margin** | for BOTH targets: rank_treatment ≤ 5; margin_treatment ≥ 0.25; no regression vs control (rank_t ≤ rank_c, margin_t ≥ margin_c) | FALSIFIED |
| **P3 veto discrimination** | veto fraction across the 71-pool ∈ [0.05, 0.95]; oracle NOT vetoed on either target | FALSIFIED |
| **P4 invariants** | projector gram_error ≤ 1e-6; all waves finite; V thin [D,k] (never [D,D]); pool == 71; one oracle per target; ring crossed only via `ring_to_real_wave`; scratch from input device | FALSIFIED/BLOCKED |

Any gate fail ⇒ seal `HOPS_PROXY_FALSIFIED`, keep `--hops-vsa-rank` default-OFF, preserve
receipts, halt. No CUDA suite, no external gate.

## 5. Execution ladder

1. Local smoke: probe `--smoke --d-model 4096 --device cpu` (mechanism, invariants, finite, veto machinery) + runner branch smoke `--hops-vsa-rank --limit 1 --attempts 1 --smoke-dim 4096 --device cpu`.
2. Commit ONLY packet + probe; push branch.
3. Remote exact-SHA worktree on Vast (RTX 5090, GPU idle); CUDA-visible focused run.
4. Production proxy D=65,536 on targets /23 and /35 → JSON receipt (dataset SHA, commit, device, per-candidate cos/projected/carrier/veto/rank).
5. Verdict per §4. PASS ⇒ isolated CUDA suite at exact SHA, then paired external HumanEval gate (G5). FAIL ⇒ seal + halt.

## 6. Governance

- Main promotion is NOT part of this phase (approval boundary; clarification deferred).
- Path B2 stays CLOSED: Gate A FALSIFIED, Gate B skipped, revert/seal `479dba4`. Reopens only with a NEW pre-registered packet + user approval.
- Supplied `Universal_Zone_C_TimescaleDB_Seeding_Module.py`: audited only, never executed, credentials REQUIRES_APPROVAL (unchanged).
