# Phase 8.39-V1 — Spec-target ranking verdict (HENRI-SYNTHESIS-PHASE0-AUDIT Stage 1)

Status: **FALSIFIED (pre-registered kill: solved < 3/50)**
Branch: `phase839/humaneval-wave-ast` @ `3078c90`
Date: 2026-08-20

## Result (OBSERVED, CUDA RTX 5090, n=50, d=65,536)

| Arm | solved | expressible | docstring targets | infra_errors |
|---|---|---|---|---|
| Control (no flag, `0066e9a`) | 2/50 (/23, /35) | 50 | 0 | 0 |
| `--spec-rank` (`3078c90`) | **1/50 (/34 only)** | 50 | **50/50** | 0 |

Scorecard: `experiments/verification/humaneval_specrank_839_scorecard.json`.

## Mechanism evidence (OBSERVED / DERIVED)

- Decoder path confirmed: `decode(pred, prompt)` computes `pn = normalize(pred − prompt)`. Legacy call `decode(prompt_wave, prompt_wave)` → `pn = 0` → sim ≡ 0 → enumeration order (zero-target variance measured 0.0 at d=2048). This matches Lens A.1 of the synthesis doc.
- `--spec-rank` passes the docstring-derived wave as `pred`: variance restored (0.019–0.022 at d=2048), 50/50 items engaged at production D.
- Oracle at d=2048: correct body ranked 3/71 (/35, in window) and 34/71 (/23, out of window). At production D the docstring direction is character-overlap reshuffle, not semantic prediction: both baseline items fell out of the 12-attempt window and a new item (/34) entered it.

## Pre-registered gate

Kill: solved >= 3/50 → PASS. OBSERVED 1/50 → **KILL FIRED**. `--spec-rank` stays default-OFF, unpromoted.

## Ranking-lever class verdict (aggregate)

| Lever | commit | score | class |
|---|---|---|---|
| control | `0066e9a` | 2/50 | OBSERVED baseline |
| reward-rank | `3654b60` | 1/50 | FALSIFIED |
| decoder-rank | `2b048d2` | 0/50 | FALSIFIED (oracle 49/71, 68/71) |
| spec-rank (V1) | `3078c90` | 1/50 | FALSIFIED |

Three independent re-ranking mechanisms (exemplar transfer, trained-unbinder entropy, docstring target) all degrade or fail to beat enumeration at this codec fidelity. The wave-geometric ranking class is closed at the current qFHRR random-ring representation for full-program waves.

## Disposition of the synthesis doc stages

- Stage 1 (non-zero target): **IMPLEMENTED via bounded substitute (docstring, not Wave_JEPA) — FALSIFIED at production.** Literal `Wave_JEPA_Predict` remains `BLOCKED_MISSING_PREMISE` (no live caller; no authorized training pairs; zero-pretraining invariant).
- Stage 2 (Zone C lexical/action partition): **BLOCKED_REQUIRES_APPROVAL** — prod DDL policy (CHECKPOINT/VACUUM only). Probe confirmed only `zone_c_engrams` + `phylogenetic_engrams_65536` exist; no lexical table.
- Stage 3 (non-parametric ranking): **executed as V1 — FALSIFIED**; parametric heads already sealed OFF (decoder-rank verdict).

## Standing scores (all OBSERVED)

HumanEval **2/50** · GPQA Diamond 0.298 · MMLU 0.2598.

## Next options (require approval; philosophy-conflicting)

1. Offline program-wave unbinder retraining (trained on code pairs) — changes the representation, not just ranking.
2. Zone C lexical engram partition + structured-codec phrase ingestion (doc Stage 2) — bounded prod DDL design.
3. Accept the current representation ceiling and pivot effort to the structured-codec text path (run20 finding).
