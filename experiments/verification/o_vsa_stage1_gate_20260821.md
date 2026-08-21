# O-VSA Stage 1 Gate — Pre-Registration and Decision Matrix (Class 4.6)

**Doc ID:** HENRI-CLASS46-O-VSA-STAGE1-GATE-2026-08-21
**Spec source:** `HENRI_Ontological_Phase_Manifold_Remedy_Specification.md`
  SHA-256 `f9cef399082e90419683b7fd2fdf716491aa7d8f181506b57aa14509b20738ee`, 9,743 B,
  `G:\My Drive\HENRI_Inbox\`; frozen copy
  `experiments/verification/o_vsa_remedy_spec/HENRI_Ontological_Phase_Manifold_Remedy_Specification_frozen_f9cef399.md`.
**Baseline:** branch `accuracy/fidelity-remediation` @ `f535137` (HOPS_PROXY_FALSIFIED sealed, clean,
  local == origin). main `849c65d` untouched; promotion deferred at approval boundary.
**Audit anchor:** random-ring confirmed `zone_c_epistemic_axiom_harness.py:58-63` (SHA-256 seed →
  `torch.randint` → Z_256); `WaveASTDecoder._wave` = ring→real→unit-normalize
  (`wave_ast_decoder.py:176-179`); `compile_functor` NOT wired in `humaneval_wave_ast_runner.py`
  (grep 0 hits); B2 Gate A probe recipe recovered (`gate_a_b2_recovered.py`, 157 lines).

## Decision matrix (three remedies)

| Criterion | R1 O-VSA ingress | R2 Lan_K FunctorFlow | R3 Hopfield energy egress |
|---|---|---|---|
| Mechanism | harmonic sub-band phase comb over ontology hierarchy | Left Kan extension context aggregation | continuous Hopfield settling over Zone C attractors |
| Bottleneck addressed | random-ring non-compositional substrate (root cause) | context aggregation (downstream of R1) | egress ambiguity (downstream of R1/R2) |
| Supervision | none (static curated ontology map) | none (algebraic) | attractor bank needed |
| Checkpoint | NO | NO | attractor codebook (Zone C) |
| Leakage risk | low; map is static, NOT derived from HumanEval | low | moderate (attractor source) |
| Memory D=65,536 | O(D) per token, band-block patterns | O(D) | O(M·D) codebook |
| Cheapest discriminator | cos-engagement probe + paired Gate A | commutativity check (internal) | energy-margin probe |
| External gate | oracle rank/margin 71-pool | none until R1 passes | delta-E margin |
| Kill rule | T1 or T2 fail → seal, halt | not gated standalone | delta-E > -0.5 → kill |
| **Verdict** | **SELECT (root cause)** | defer until R1 passes | defer until R1 passes |

R1 is the only remedy that adds semantic structure AT INGRESS; R2/R3 operate on a substrate that
remains random-ring unless R1 lands. Selecting R1 only; R2/R3 require a new approved packet.

## Stage 1 implementation (one bounded default-OFF change)

- New module `HENRI V2/o_vsa_harmonic_encoder.py`: `OVSAHarmonicEncoder.encode_text(text) -> uint8 [D]`
  matching `qFHRREpistemicCodec` interface (drop-in for `WaveASTDecoder`).
- 16 sub-bands × 4,096 dims per spec layout:
  - dims 0–4,095: Ω_root domain anchor (domain-seeded harmonic ramp; domain = code_ast);
  - dims 4,096–16,383: ω_category, 12 disjoint node-type blocks (module/definition/flow/name/
    call/operator/signature/constant/iteration/attr/condition/keyword);
  - dims 16,384–65,535: φ_instance, token-family coarse hash (6 bands) + token fine hash (6 bands);
    curated static family map (len/count/strlen→length, max/maximum→maximum, ...) documented in-module.
- AST parse with regex-token fallback; phase accumulation via cos/sin bundling per band; quantize
  `round(θ/2π·256) % 256` → uint8 Z_256 (same family as legacy, structured phases).
- Runner flag `--o-vsa-harmonic-ingress` default OFF: swaps the codec instance passed to
  `WaveASTDecoder` (flag → codec object → `_wave` → candidate/goal waves → ranking).

## Pre-registered gates (frozen BEFORE any probe results)

**T1 — semantic engagement (spec §Stage-1 Metric 1):**
- Related pair cos ≥ 0.40: `return len(x)` vs `return count(x)`; `return max(l)` vs `return maximum(l)`.
- Unrelated pair cos < 0.15 (spec failure threshold; target ≤ 0.05, reported separately):
  `len`-node vs binary-op node; `len` vs `grid` token.

**T2 — external rank/margin (spec §Stage-3 Gate A_O-VSA, same 71-pool recipe as B2):**
- BOTH HumanEval/23 (true `return len(string)`) AND HumanEval/35 (true `return max(l)`)
  oracle rank ≤ 5 / 71 under O-VSA scoring; control arm = live random-ring path.
- Margin gate (AMENDED pre-result): true-vs-best **cross-family** lookalike margin ≥ 0.25
  (spec "lookalike" = semantically distinct program; same-family candidates such as
  len-containing bodies are structurally related by design of T1, so same-family margin is
  reported separately as `same_family_margin` and does not gate).
- Control and treatment measured at the SAME SHA, same pool, same hardware (RTX 5090, D=65,536).
- Any execution error in either arm → `BLOCKED_INFRASTRUCTURE`, no verdict.

**Kill rule:** T1 fail OR T2 fail ⇒ `O_VSA_INGRESS_FALSIFIED` sealed, `--o-vsa-harmonic-ingress`
stays default-OFF, no Stage 2/3, no CUDA suite beyond the probe, halt.

**Boundaries:** no dataset-derived ontology (static curated map); no evaluator ingress; no [D,D]
allocation (assert max tensor ≤ [D]); oracle used ONLY for ranking measurement, never in candidate
generation or egress; ring boundary explicit (encoder output IS the Z_256 ring consumed by `_wave`).

## Verification ladder

py_compile → RED contract (T1 pairs, band layout, ring family, no-dense) → local D=4096 probe both
arms → commit+push exact SHA → remote clean worktree → D=65,536 paired probe alone on GPU →
verdict table → seal/advance.
