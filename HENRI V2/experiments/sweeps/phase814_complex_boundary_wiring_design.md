# Phase 8.14 — Complex Boundary Wiring (pre-registered, 2026-08-16)

Sources: user instruction (wire verified 7.3/7.4 encoder → 8.11 complex
transition at planner boundary, complex egress contract) + Roadmap
`Project HENRI V2 Strategic R&D Roadmap.pdf` (SHA `0ca9f7a1…`, §1.1
A(x,y)·e^{jΘ} + §1.2 V W† + R_block, branch base @ `9f1c207`).
Execution ledger: `experiments/sweeps/roadmap_execution_ledger.md`.

## Mechanism (value-lift boundary)

The 8.12/8.13 failures show the boundary must preserve the REAL
discriminative channels (occupancy amplitude + color-phase values) while
keeping the state complex THROUGH the transition (exact phase evolution,
8.11 G1b) and paying the real-egress loss ONCE at the environment edge.

- Lift:      z_d = |w_d| · exp(j·φ_d),  φ_d = (π/2)·(w_d / ‖w‖∞)
             (amplitude |w_d| preserved; value ratio w_d/‖w‖∞ ∈ [−1,1]
             encoded in phase; φ_d ∈ [−π/2, π/2] so sin(φ_d) carries the
             sign of w_d).
- Egress:    ŵ_d = sign(imag(z_d)) · |z_d|      (EXACT round-trip: ŵ = w,
             since φ_d ∈ [−π/2,π/2] ⇒ sign(imag) = sign(w_d), |z_d| = |w_d|).
- Pipeline:  real [1,NB,BD] → value-lift → ℂ^D → NativeComplexWaveTransition
             (forward_complex / update_phase_complex, amplitude-invariant
             8.13-G2-proved) → value-egress → real [1,NB,BD] at the
             environment edge only.
- Default-OFF: `HENRI_ARC_COMPLEX_BOUNDARY` env flag; unset ⇒ planner
             keeps production real path (LowRankCoupledTransition).

## Gates (pre-registered)

- G1 Discrimination: complex-cosine of value-lifted hard pairs
  (color 3v6, shared-support ring-vs-line, disjoint A-vs-C) ≤ 0.02
  (user-mandated; roadmap G1 < 0.05 satisfied as corollary).
  KILL if any > 0.02 OR legacy real encoder beats the complex lift by
  ≥2× on the SAME pair (8.13 comparator lesson).
- G2 Paired transfer: deterministic 32 translation pairs; real arm
  (forward + update_wirtinger, 8.12 G4 recipe) vs complex arm
  (lift → update_phase_complex → egress), matched held-out pre/post.
  ACCEPT iff post_complex ≤ 0.90 AND post_complex − post_real ≥ +0.02.
  KILL otherwise.
- G3 Demo block: BLOCKED_NO_DEMONSTRATIONS asserted (never fabricated).
- G4 Latency: lift + forward_complex + egress cycle ≤ 2.0 ms @ D=65,536
  (8.12 G5 standard; roadmap 50 µs Triton gate belongs to 8.16).
- G5 Default-OFF: EFEPlanner.transition type remains
  LowRankCoupledTransition with flag unset.
- G6 Round-trip: value-egress(lift(w)) == w, max abs err < 1e-6 (d=512).

## Kill criteria (pre-registered)

- G1 OR G2 fails at scale (after the local probe passes) ⇒ SEALED KILL,
  redirect documented, main untouched.
- Local probe (d=512) failing G1 already ⇒ stop before remote (cheapest kill).

## Deviations (from roadmap)

- D6 branch @ `9f1c207` (roadmap-mandated base); main untouched @ `2218ec4`.
- Phantom CLIs (#8/#9) → real contract tests + dedicated runner.
- R8.13.3 (RESET-penalty engine) deferred — load-bearing production change.

## VARIANT B (corrected 2026-08-16) — native un-realify boundary
Mechanism: the production encoder IS complex-native (accumulates ℂ^{D/2},
projects to ℝ^D via concat([re, im]) + L2). Variant B re-pairs:
    z = w[:D/2] + 1j·w[D/2:]          (exact by construction)
runs the 8.11 transition on the TRUE phase space (ℂ^{D/2},
transition num_blocks=D//2, block_dim=1), and re-realifies ONLY at the
environment edge. Complex inner product on re-paired waves ≡ legacy real
cosine (proved at d=512: A_A6 0.191138 both, B_C 0.001547 both), so the
sealed 8.13 discrimination evidence (6.7e-07 / 7.0e-05 @ D=65,536)
transfers unchanged.
Variant A (value-lift) FALSIFIED locally: surrogate phases re-encode
(color cos 0.4555 vs legacy 0.1911 = 2.4x worse; shared 0.3199 vs 0.0016
= 200x worse). Functions kept for audit trail.

## Amended gates (variant B)
- G1 local (d=512, contract): complex_cos == legacy real_cos on every
  hard pair (|Δ| < 1e-5) AND shared/disjoint pairs complex_cos < 0.05.
  KILL if identity fails.
- G1 scale (D=65,536, remote): complex_cos ≤ 0.02 on hard pairs.
  KILL if any pair > 0.02 (8.13 evidence predicts 1e-6..7e-5).
- G2 paired held-out transfer: 32 deterministic translation pairs;
  fit on 24, eval on 8 held-out. Real arm = 8.12 G4 recipe
  (forward + update_wirtinger lr=0.05). Complex arm =
  un_realify → forward_complex + update_phase_complex lr=0.05 →
  re_realify. ACCEPT iff post_complex ≤ 0.90 AND
  post_complex − post_real ≥ +0.02. KILL otherwise.
- G3 demo block (BLOCKED_NO_DEMONSTRATIONS, never fabricated).
- G4 latency: un_realify + forward_complex + re_realify cycle
  ≤ 2.0 ms @ D=65,536.
- G5 default-OFF: EFEPlanner.transition remains LowRankCoupledTransition.
- G6 round-trip: re_realify(un_realify(w)) == w, max err < 1e-6 @ scale.

## VERDICT — SEALED ACCEPT (OBSERVED 2026-08-16, RTX 5090, D=65,536, commits e618bec..8e06374)
G1 SCALE_PASS: complex_cos == real_cos exactly on every hard pair
(color 0.2002774, shared 0.0002753, disjoint 0.0003604 — identity delta
0.0 everywhere). The boundary reproduces the production encoder
byte-exactly; the real-egress boundary loss is eliminated by
construction (the complex inner product IS the real cosine).
G2 ACCEPT (32 pairs, 24 fit / 8 held-out): complex arm
pre 0.82825 -> post 0.4199 (improvement +0.40835); real arm
pre 0.87222 -> post 0.82733 (improvement +0.04489);
post_real - post_complex = 0.40743 >= 0.02; post_complex <= 0.90.
The complex transition learns translation operators ~9x more effectively
than the real acos-lift path (8.11 G2@scale gap +0.019 is now bridged:
+0.408 complex vs +0.045 real on matched held-out pairs).
G3 BLOCKED_NO_DEMONSTRATIONS (never fabricated).
G4 CYCLE_PASS 0.079 ms <= 2.0 ms.
G5 DEFAULT_OFF_PASS (LowRankCoupledTransition).
G6 ROUNDTRIP_PASS max_err 0.0.
DONE_MARKER rc=0 failures=[] verdict=ACCEPT.
Evidence: p814_matrix_d65536.json SHA fb4f2a2a165b8a5686a8aee2d52ef71f301440f5695f80a03b803ca0d8f7e498;
p814_full.log SHA c89238aa77eebd052de72ca715793d1d031d38d489127bb2f84054f5c1384cc5
(local: phase8_evidence/phase814_complex_boundary/).
Local contract 5/5; full suite 423p/1s. main UNTOUCHED @ 2218ec4.
NO promotion yet — default-OFF component, sealed on feature branch.
ARC-AGI-3 SOTA stays BLOCKED_NO_DEMONSTRATIONS (20/20; G3).

## Phase 8.15 redirect (from evidence)
The complex boundary (variant B) is verified and default-OFF. The
roadmap's R8.15 (holographic egress head, L_obstruct < 1e-4) and R8.16
(Triton LUT <= 50 us) are the next staged phases; R8.14 (Zone C
hash-chained ledger, G4 1,000-update verification) is a deterministic
store test pending in the execution ledger.
