# System-1 v0.6.3a — Pre-Reasoning Entropy Instrumentation (CORRECTED scope)

**Date:** 2026-08-24 (before any CUDA launch)
**Reference 3 (gpt-5.6-sol) + corpus consult #20 (INFERRED)**
**Upload audited:** `Analysis___Strategic_Decision_Blueprint.md` sha
`7d126068…` (1161 B) — v0.6.0.1 diagram re-uploaded; its v0.6.3 directive
is corrected below.

## Claim audit (upload vs measured telemetry)

| Upload claim | Disposition | Evidence |
|---|---|---|
| Search space saturated (H→0, correct at rank 1) | **FALSIFIED** | dev9_v0601 per-task: first-pass ranks uniform 1–9 {1:10,2:10,3:10,4:10,5:5,6:5,7:5,8:5,9:5}, mean 4.231, rank-1 rate 0.154 |
| ~90% verifier-call reduction via H-gate | **FALSIFIED** | Safe gate (stop after first verified pass) = current behavior = 0 savings. Unsafe gate (accept rank-1 unverified) = 15.4% outcome pass, not 1.00. Max possible safe margin 0. |
| H<0.40 threshold | **BLOCKED_MISSING_PREMISE** | Corpus #20: low H = concentration only; 56% accuracy ceiling at H≈0 in the forced-answer study; "confidently wrong" documented |
| V(p₁)=0 failure hook | **BLOCKED** | No such signal exists in live code |
| R-EDMD/Wave-JEPA continuous learning attached | **CONFLICTS_WITH_SCOPE** | No bundling; separate carrier (v0.6.3c) only after margin is demonstrated |
| Reuse dev9_v0601 for calibration | **FALSIFIED** | Consumed split; never replay. Fresh disposable splits only |

## Corrected scope: v0.6.3a — entropy instrumentation ONLY

- No behavioral change. No gating. No learning. Default OFF.
- Measure per task, pre-verifier: Shannon entropy H and H/logK of the
  candidate cosine-score distribution (bridge sims, real pre-verifier
  signal), first-passing rank, verifier calls, outcome, family.
- Answer ONE question: does pre-verifier entropy predict first-pass rank
  or outcome? (Saving score distributions is required — v0.6.0.1 telemetry
  did not save them.)
- Fresh disposable splits, n % 13 == 0, seeds disjoint from all consumed.
- Verdict chain (Reference 3): `ENTROPY_PREDICTIVE_ONLY` /
  `COST_EFFECTIVE_PRESERVED` / `NO_EFFECT` / `UNSAFE_BYPASS` / `REGRESSION`.

## Gates

- G1 integrity: no consumed digest (incl. dev9_v0601 a8a2d7a7…, heldout
  87390286…, a09bf275…) staged.
- G2 no behavioral change: R0 identity — gated arm must produce
  byte-identical pool order and identical outcome to baseline.
- G3 predictive value: entropy (or H/logK) has nonzero correlation with
  first-pass rank or outcome (report Spearman ρ + CI; gate at |ρ| > 0.2
  with CI excluding 0 for `ENTROPY_PREDICTIVE_ONLY`).
- G4 default-off: ENV flag `HENRI_V063A_ENABLE=1` required; absent → no
  entropy code path executes.

## Frozen manifest (filled after hash freeze)

- `v063_entropy_gate_carrier.py`: `64cdf1a6ac9632726b53a79bddc17189c1d88daf7a22fd951e47780ac10f6941`
- `contract_v063.py`: `436d50e0720a7e99c06594d608e56bda7fde2eca4645b7b32cd3757fc4e89e84`
- `eval_v063_dev.py`: `e9d37ded11b1719bd457fed5874349d9e777c2a7d5ec8172284b46a35f39cb87`
- carrier kernel `system1_kernel_v055_ast_skeleton.py` (unchanged): `d9a976adff4146a11950c51218ca32af1cde4b3db59431ce370a45913ff8d870`
- checkpoint `v041_energy_checkpoint.pt` (unchanged): `11d56121e4b091e2162078eb4cae71ce213dacc01397d8f8209bc9e2152a8f4d`
- split `dev10_v063` (sealed): `41f283f42006b8584d1dd72a34bae67f4dee2eb04df8808d57e588c9e6d1b7fc` (seed 70707, n=65, 13×5, single_use)

## Splits (this cycle)

- seal: fresh, seed 70707, n=65, single_use, generation-only.
- smoke: disposable, seed 76126, n=13.
- dev9_v0601 (a8a2d7a7…) NEVER loaded.
