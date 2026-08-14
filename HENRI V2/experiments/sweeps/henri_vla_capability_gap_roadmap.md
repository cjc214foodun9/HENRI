# HENRI VLA Capability-Gap Roadmap (grounded 2026-08-14)

Source: /henri-research leaf audit `deleg_f79b40b0` (arXiv API, titles/abstracts
verified in-transcript) + sealed Phase 8.1/8.2 verdicts. Citations below are
OBSERVED (title/ID verified via arXiv API 2026-08-14); implications are
HYPOTHESIS until a live consumer proves them.

## Measured position (sealed)

- 7.9e/7.9f: 0 progress events over ~155k steps; SANS rows = 0; held-out sealed.
- Demo ingress: 20/20 envs `BLOCKED_NO_DEMONSTRATIONS` (`325a2db5…`, `ff5fbec7…`).
- Phase 8.1 D4 orbit goal FALSIFIED (`5a15a9d5…`): orbit-norm 0.980 → carrier-dominated.
- Phase 8.2 in-context functor FALSIFIED at K1 (`f978428c…`): held-out recovery works
  (K2 +0.1136), but true action ranks 37/128 sim / 120/128 EFE, margin −0.02875.
- Throughput is NOT the gap (vmap 21.9×, agreement 2.4e-07); directionality is.

## Capability-gap matrix (verified SOTA anchors)

| Capability | SOTA anchor (arXiv, OBSERVED) | HENRI state | Missing proof / feature |
|---|---|---|---|
| Latent world model | I-JEPA 2301.08243; V-JEPA-2 2506.09985; Next-LAT 2511.05963 | UWE waves [8192,8] | No predictive next-state objective on the live loop |
| VSA operator math | Frady 2109.03429 (VSA/FHRR) | compile_functor_wave exists | Operator recovery ≠ action selection (K1 failure) |
| Active-inference action selection | Tschantz 2002.12636 | EFEPlanner argmin | Goal wave is carrier-dominated; no candidate-specific action semantics |
| Object-centric grounding | Slot Attention / 2511.11478 | CC-OS segmenter | Object records not yet bound into action payloads |
| Continual learning | EWC 1612.00796; GEM 1706.08840 | EDMD + SGLD | No evaluated transfer metric |
| Program synthesis (ARC) | Abductive solver 2411.18158; Generalized planning 2401.07426 | CEGIS macro engine | No authentic task pairs to fit/validate against |
| Egress calibration | — | Decoder checkpoint LOADED | `(GameAction, data)` payload path not task-validated |

## Root-cause chain (each link measured or directly derived)

1. Carrier dominance: color-0 background constant mass dominates UWE similarity
   (orbit norm 0.980; goal_sim_obs 0.4096 ≈ baseline) → any goal wave is
   nearly equidistant → no ranking signal (K1 failures 8.1 AND 8.2).
2. No authentic supervision: 20/20 envs expose no provenance-bearing pairs →
   W_task can only be compiled from synthetic or reconstructed pairs → no
   task-directional operator.
3. No outcome attribution: 0 progress rows → EFE and W_task quality are
   unmeasurable → every downstream "improvement" is unverifiable.

## Bounded plan (ordered; each step has a kill)

### R1 — Representation-discrimination kill (NEXT, before any functor retest)
- Variant: foreground-masked + independent/incommensurate x/y ramps vs legacy
  encoder (default path byte-identical when OFF).
- Gate: known-transform rank ≤ 2 AND margin ≥ +0.05, tested SEPARATELY for
  translate / rotate / color at D=65,536 on CUDA. Any fail → kill.
- If R1 passes → retest in-context functor K1 with the repaired encoder.

### R2 — Action-semantic payload calibration
- Calibrate `(GameAction, data)` generation from the policy representation;
  ACTION6 coordinate-bearing payloads validated against the action schema.
- Kill: semantic action-head probe (wave→logits→payload) must show calibrated
  ranking on legal action sets before any score eligibility.

### R3 — Predictive world-model objective on the live loop
- Add next-state prediction loss (JEPA-class) over the UWE wave sequence in
  EFEPlanner; measure held-out prediction agreement.
- Kill: prediction agreement must beat identity/statistics baseline on
  synthetic grids before claiming world-model capability.

### R4 — Authenticated supervision ingress (enables everything downstream)
- Options: official ARC-AGI-3 training split (if the benchmark license allows),
  or a benchmark-compatible demo API. Do NOT reconstruct labels from game
  logic/recordings (boundary sealed).
- Gate: provenance-carrying pairs with digest + split identity.

### R5 — Externally grounded progress attribution
- Matched-counterfactual first-action branches; progress rows require
  reproducible strict `levels_completed` increase with attribution.
- SANS task validation only after ≥50 authentic progress rows.

### R6 — Production VLA composition
- Typed multimodal ingress (vision/text) → UWE; calibrated action egress;
  Zone C frozen axioms; release gates (checkpoint provenance, evidence chain,
  score-eligibility single-source rule). All phases default-OFF until their
  kill passes.

## Sequencing rule

R1 → R2 → (R3 ∥ R4) → R5 → R6. No functor tuning, no throughput work, no
rollout before R1 passes. Score/SOTA claims stay BLOCKED until R5 produces a
reproducible strict progress increase with matched-counterfactual attribution.
