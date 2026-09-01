# Carrier F16 — Active Lie Generator Warping & Adaptive Affordance Re-Ranking — Pre-Registration

- Directive: `Project_HENRI_F15_Post-Mortem_Audit___Adaptive_Lie_Re-Ranking_Directive.md`
- Directive SHA-256: `c5d14170b2e33d342409b5b55e873a2692afb800840d77ce935abbf3bb7d703c` (21,716 B, 249 lines)
- Directive ID: `HENRI-DIR-2026-08-F15-POSTMORTEM-ADAPTIVE-RERANKING-ORDER`
- Branch: `carrier/f16-adaptive-reranking` (base: F15 head `b574db4` — the directive's evaluated commit)
- Governance: ledger target record 1,099+ (current head 1,101; seal lands >= 1,102); prior seal `F15_GATES_VERDICT #0fb9c7be…` @ 1,098 RATIFIED by directive

## 1. Mechanism (directive §1.2, §3)

Replaces F15's static Lie generators + Slerp-waypoint beam with goal-conditioned Hamiltonian warping:

1. **Goal extraction**: identical to F15 (trajectory-bank terminal state, `resolve_trajectory_goal`, PG1 <= 0.90 pre-flight) — the verified ingress pathway is retained unchanged.
2. **Tier 1 — anti-symmetric goal projection tensor** (per live step):
   `Omega_goal(t) = Psi_goal Psi_t^T - Psi_t Psi_goal^T in so(D)`.
   At the D=64 verification substrate this is the full so(64) rank-2 skew tensor (the production M=8192 per-block so(8) form is dimensionally identical per 8-block).
3. **Tier 2 — warped generators** (per live step, per action):
   `D~_a = D_a + alpha_steer * Omega_goal(t)`, `alpha_steer = 0.35`;
   `Psi_hat_{t+1}(a) = exp(D~_a) Psi_t` (torch.linalg.matrix_exp; normalized to S^{63}).
4. **Tier 3 — vectorized K=8 beam, goal-direct objective** (Slerp waypoint REMOVED):
   `J(a_{1:8}) = |<Psi_hat_{t+8}, Psi_goal>| - beta_Sagnac * sum_{k=1..8} Delta_damped(k)`, `beta_Sagnac = 0.05` (F15 beam alpha carried);
   `Delta_damped(k) = Delta_Sagnac(k) + gamma_damp * ||Psi_{t+k} - Psi_t||^2_2`, `gamma_damp = 0.10`;
   `a* = argmax_a J(a)`.
5. **Valence**: `Delta_nu_goal(t) = |<Psi_{t+1}, Psi_goal>| - |<Psi_t, Psi_goal>|` (goal valence; F15 measured vs waypoint — superseded).
6. **Tier 4 — Hebbian creep** retained from F15 (eta=0.05, zero-valence guard), goal-valence-driven: proven rank-inert in F15 (did not alter selection); retention cannot confound a positive steering effect, which must be attributed to generator warping.

## 2. Gates (directive §4)

| Gate | Criterion | Failure action |
|---|---|---|
| PG1 | `max_env |<Psi_0, Psi_goal>| <= 0.90` | PRE-FLIGHT KILL (degenerate trajectory goal) |
| G1 | 1,800 steps / 12 envs, latency <= 5.0 ms/step | KILL K1 |
| G2 | >= 1 of 12 solved (Score > 0.0%) | KILL K2 |
| G3 | mean Delta_nu_goal >= +0.0200 | KILL K3 (rank invariance persists) |
| G4 | single-pass K=8 horizon coherence (damped) <= 0.0500 | KILL K4 |

## 3. Pre-registered deviations (disclosed, not silent)

- **D1 (Omega at verification substrate)**: full so(64) `Omega = Psi_goal Psi_t^T - Psi_t Psi_goal^T` per directive §3.1 Tier 1 formula; the per-block so(8) form (M=8192) is the production-scale instantiation, dimensionally identical per 8-block. No explicit Cartan-Killing basis projection is applied: Omega in so(D) already lies in the algebra (directive §3.1 is the operative blueprint; §1.2's Killing form is the derivation).
- **D2 (beam target)**: J uses `Psi_goal` directly (directive Tier 3); F15's Slerp waypoint is removed. G3 valence likewise vs goal.
- **D3 (G4 instrument)**: G4 measured on `Delta_damped` (directive §3.3); raw `Delta_Sagnac` also reported for F15 cross-comparability (F15 G4 raw = 0.9607).
- **D4 (creep)**: retained from F15, goal-valence-driven, same eta=0.05/zero-valence guard — inert-by-proof in F15, no attribution risk.
- **D5 (seed/params)**: directive command values — seed `20260915`, horizon 8, beam 8, alpha_steer 0.35, gamma_damp 0.10, beta_Sagnac 0.05 (directive leaves beta unnamed; pre-registered as the F15 beam alpha).
- **D6 (bank/goal source)**: identical to F15 (npz key `psi`, JSONL field `env`, bridge block-mean -> K=64, PatchIngress) — verified pathway, unchanged.

## 4. Execution order

1. TDD: `tests/contract/test_f16_warped_engine.py` (C1–C14: Omega skew, warped D~ skew, exp orthogonal, geodesic rotation law, rank-break anti-lock, steering gain, damping non-negative, beam determinism, bank/PG1 fail-closed degenerate, flag fail-closed, no-bank fail-closed, valence semantics, module-constant guard (C13 lesson), substrate-constructor reachability (C14 lesson — catches harness defects invisible to unit tests)).
2. Implement `experiments/verification/arc_f16_warped_engine.py` (flag `HENRI_F16_WARPED=1` fail-closed; canonical imports from `arc_f15_trajectory_engine` / `arc_f10_live_engine` / `arc_f11_plasticity_engine` only; no local shadowing).
3. Local suite + full regression from repo root.
4. Commit + push; remote detached-worktree CUDA verify @ exact SHA.
5. Live gauntlet per directive command: 12 envs x 150 = 1,800 steps, seed `20260915`, K=8, alpha_steer 0.35, gamma_damp 0.10, bank `/root/f3-run/telemetry/f3_bank_capture_v2/`.
6. Seal `F16_GATES_VERDICT`; deliver scorecard + lessons.

## 5. Kill experiment / cheapest falsification

If the C5 rank-break contract fails locally (goal-direct J(a) still uniform across actions on a PG1-valid synthetic pair) or the C6 steering gain fails (warped alignment <= static alignment for a goal-aligned generator) -> the warping does not break rank invariance on the D=64 substrate -> KILL before launch. If live G3 <= 0 and G2 = 0 with warping engaged -> goal-warped generators are falsified against the trajectory goal (same class as F15's rank-invariance lock).

## 6. Falsifiable claim

"HYPOTHESIS: goal-warped Lie generators (D~_a = D_a + 0.35 * Omega_goal) break candidate rank invariance and produce positive goal-directed valence (G3 >= +0.02) with >= 1 live solve (G2 >= 1) under the verified bank ingress on the 12-env live gauntlet." PG1 is pre-verified on the real bank; G1–G4 measured live on vast-5090.
