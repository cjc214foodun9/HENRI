# System-1 Stage-0c-rev3 — r=16 Reduced-Koopman Spectral Evaluation — PRE-REGISTRATION

**Date:** 2026-08-24 · **Reference 3 (gpt-5.6-sol) binding** · Status: SEALED BEFORE K CONSTRUCTION

## Prior verdicts (preserved, never relabeled)
- Stage-0c-rev (RFF, PR≥16 gate): `IDENTIFIABILITY_BLOCKED` (audit `5f44c3c8…`, commit `a7d621d`).
- Stage-0c-rev2 (r∈{4,8}): `CONTRACT_FAILED` (C8 calib projected ε 0.145–0.181 > 0.05; commit `754ffb3`).

## Upload (proposal artifact, audited)
- `Stage-0c-rev3_Evaluation_and_Authorization.md`, 739 B, SHA-256 `99d28342e31e21e327bca0b2c837ba78984d267d097e57b9af453cd8f2099fd1`.
- 15-line diagram: top-16 share > 92%, orthogonal residual < 8%, κ16 = 5.49, floor < 0.065,
  SSR target < 0.350; user protocol thresholds: **SSR_eval ≤ 0.40, ε_rollout,5 ≤ 0.35**.
- C7–C10 text, SSR definition, rollout semantics, spectral-radius gate, verdict names NOT specified
  → authored + sealed here.

## Read-only spectral probe (OBSERVED, sha `a34b50f9…`; no operator fitted)
| Claim | Probe a0 / a1 | Disposition |
|---|---|---|
| top-16 share > 92% | 0.9341 / 0.9414 | PASS (gate ≥ 0.92) |
| κ16 = 5.49 | 5.4948 / 5.9771 | PASS (gate ≤ 10.0; upload value = a0) |
| orthogonal residual < 8% | 6.6% / 5.9% (X) | PASS on X |
| **absolute error floor < 0.065** | **0.3347 / 0.3670 (Y-target); 0.2567 / 0.2421 (X-same-space)** | **FALSIFIED** — 0.065 is the variance fraction (1−0.9341); the true full-space floor is ≥ 0.24. Same convention error as rev2. Sealed metric = PROJECTED (coefficient-space) SSR; full-space floors diagnostic only. |
| N_a ≥ 4r = 64 | 102 / 69 | PASS (a1 margin 5) |

## Data
- **Calibration (fitting):** the ESTABLISHED 171-record split (lexicographic filename order;
  ids 101,1010,1111,1212,1313,1414,1515,202,303,404; manifest `54b7350a…`). Unchanged from rev/rev2.
- **Evaluation (fresh, disjoint):** `vla_stage0c_rev3_eval_corpus` — 220 records, 10 new seeds
  2101–3010 (one verified Stage-0a wrapper per episode, ledger-serialized), manifest
  `f0c9a7624f26bf70…`; **raw-obs overlap vs calibration = 0/181/230 (OBSERVED)**.
- The rev2 133-record evaluation split is NOT reused as fresh evidence (adaptive-development
  exposure); it appears nowhere in rev3 gates.

## Frozen input
- `vla_stage0b_rev_params.npz`; runtime loader and script both ASSERT full SHA-256
  `766e607ad0bc739ea0a139172dd34e16d01a268cca80e990af5aab01006cfcd7`. No retuning.

## Operators (fixed, no selection on evaluation)
- r = 16 FIXED. Separate per-action bases and operators: V16^(a) = top-16 right singular vectors of
  calibration X^(a) (raw SVD, no regularization); K16^(a) = lstsq(XV16^(a), YV16^(a), rcond=None).
  Never shared across actions.
- Implicit projection: P16^(a) = V16^(a) V16^(a)ᵀ, applied as (x @ V) @ V.T. NO dense 6144×6144
  matrices materialized (two float64 dense P would be ~576 MiB; Reference 3).

## Metrics (all reported separately)
- full-space error `||Y − (X·V)·K·Vᵀ||_F / ||Y||_F` — DIAGNOSTIC (floors 0.24–0.37 make gates
  infeasible)
- projected (coefficient-space) one-step `||(Y·V) − (X·V)·K||_F / ||Y·V||_F`
- persistence baseline `||(Y − X)·V||_F / ||Y·V||_F` (projected); calibration-mean baseline
- **SSR_eval = mean over actions of (projected eval one-step ε) / (projected persistence eval ε)**
- 5-step open-loop rollout: maintain the PREDICTED FULL lifted state; at each step project into the
  CURRENT action's V16, apply that K16, reconstruct full state; on action switch, re-project the
  full state into the new action's basis. Error at horizon 5: projected error
  `||(x̂−x)·V16^(a_last)|| / ||x·V16^(a_last)||`, averaged over all valid windows and episodes,
  grouped by final action.
- spectral radius ρ(K16^(a)) for each action.

## Gates C7–C12 and verdict chain (failure precedence)
- **C7** κ16 ≤ 10.0 AND top-16 share ≥ 0.92, both actions.
- **C8** calibration projected one-step ε ≤ 0.05, both actions (sanity; interpolation near 0).
- **C9** SSR_eval ≤ 0.40 (aggregate over actions).
- **C10** ρ(K16^(a)) ≤ 1.05, both actions, AND 5-step rollout projected error ≤ 0.35 (aggregate).
- **C11** determinism: telemetry JSON + operators NPZ byte-identical SHA-256 across ≥ 2 processes.
- **C12** baselines reported (persistence, calib-mean) for calibration and evaluation.
- Verdict: all of C7–C11 pass → `REDUCED_KOOPMAN_SPECTRAL_VERIFIED`; otherwise
  `CONTRACT_FAILED` at the first failing gate. Kill criteria: no post-result tuning of r, ridge,
  basis, or metric within this carrier.

## Boundaries
- CartPole dynamics result only. **VLA gate 0/12; AAII v4.1.1 0/9 BLOCKED** (no causal path from
  this spectral carrier to any index component). No SOTA claim.
