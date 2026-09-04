# Carrier F17 — Candidate-Differential Killing-Form Steering — Pre-Registration

- Directive: `Project_HENRI_F16_Post-Mortem_Audit___Differential_Killing-Form_Steering_Directive.md`
- Directive SHA-256: `9962037e2555d77e15b3f78dc7edbad0c01345cd7859f43ee8bdfc8f9b797edf` (22,080 B, 261 lines)
- Directive ID: `HENRI-DIR-2026-08-F16-POSTMORTEM-DIFFERENTIAL-STEERING`
- Branch: `carrier/f17-differential-steering` (base: F16 head `423a028` — the directive's evaluated commit)
- Governance: ledger head 1,106 at auth; prereg seal lands >= 1,107; prior seal `F16_GATES_VERDICT #bb6797be…` @ 1,103 RATIFIED by directive

## 1. Mechanism (directive §1.2, §3.1, Tier-3 blueprint)

Replaces F16's additive common-mode warp `D~_a = D_a + alpha*Omega` (FALSIFIED: BCH first-order term common to all candidates; goal alignment 0.0 -> 0.0 over 1,800 steps; G3 +0.000214) with a candidate-DIFFERENTIAL Killing-form modulation:

1. **Goal extraction**: identical to F15/F16 (trajectory-bank terminal state, `resolve_trajectory_goal`, PG1 <= 0.90 pre-flight) — verified ingress pathway retained unchanged.
2. **Tier 1 — anti-symmetric goal projection tensor** (per live step):
   `Omega_goal(t) = Psi_goal Psi_t^T - Psi_t Psi_goal^T in so(D)` (unchanged from F16).
3. **Tier 2 — Killing-form projection (candidate-differential)**:
   `gamma_a = -1/2 Tr(D_a Omega_goal(t))` (per action, all 8 in parallel);
   by skew-D algebra `gamma_a = <Psi_goal, D_a Psi_t>` — the projection of the goal onto the direction each generator would displace the state.
   `D~_a = (1 + kappa_diff * tanh(gamma_a)) * D_a`, `kappa_diff = 0.75` (§3.1 upgrade 2: tanh velocity gating, bounded `D~_a in [1-kappa, 1+kappa] D_a`).
   `Psi_hat_{t+1}(a) = exp(D~_a) Psi_t` (torch.linalg.matrix_exp; normalized to S^{63}).
4. **Tier 3 — Lyapunov-damped vectorized K=8 beam** (Slerp waypoint REMOVED, F16 objective carried with the Lyapunov penalty replacing the F16 displacement penalty):
   `J(a_{1:8}) = |<Psi_hat_{t+8}, Psi_goal>| - beta_Sagnac * sum_{k=1..8} Delta_Lyapunov(k)`, `beta_Sagnac = 0.05` (carried from F16 — unnamed in directive, D4);
   `Delta_Lyapunov(k) = Delta_Sagnac(k) + mu_damp * ||D~_a||_F^2`, `mu_damp = 0.15` (§3.1 upgrade 3; norm is per-action constant over the horizon);
   `a* = argmax_a J(a)`.
5. **Valence**: `Delta_nu_goal(t) = |<Psi_{t+1}, Psi_goal>| - |<Psi_t, Psi_goal>|` (unchanged from F16).
6. **Tier 4 — Hebbian creep** retained from F15/F16 (eta=0.05, zero-valence guard), goal-valence-driven; proven rank-inert in F15 and non-attributable in F16 — retention cannot confound a positive steering effect, which must be attributed to the differential Killing warp (D5).

## 2. Gates (directive §4)

| Gate | Criterion | Failure action |
|---|---|---|
| PG1 | `max_env |<Psi_0, Psi_goal>| <= 0.90` | PRE-FLIGHT KILL (degenerate trajectory goal) |
| G1 | 1,800 steps / 12 envs, latency <= 5.0 ms/step | KILL K1 |
| G2 | >= 1 of 12 solved (Score > 0.0%) | KILL K2 |
| G3 | mean Delta_nu_goal >= +0.0200 | KILL K3 (differential steering failure / invariance persists) |
| G4 | mean Delta_Lyapunov <= 0.0500 | KILL K4 (waveform coherence degradation) |
| E1 (added, pre-registered) | mean over steps of `std_a(gamma_a)` > 1e-6 | `F17_FALSIFIED_NO_ENGAGEMENT` — the warp never differentiated candidates (subsumes K2/K3 attribution) |

## 3. Pre-registered deviations (disclosed, not silent)

- **D1 (alpha_trans term NOT implemented)**: the schematic figure (line 67) shows an extra `+ alpha_trans * gamma_a * Omega_goal` term. §3.1 (the operative mechanism text, upgrades 1–3) and the §5 command block (no `--alpha-trans`) do not include it. The executable spec is the §3.1 + §5 command; the schematic additive term is omitted and recorded here.
- **D2 (tanh gating)**: `tanh(gamma_a)` per §3.1 upgrade 2 (bounded `[1-kappa, 1+kappa]`); §1.2's linear `(1 + kappa*gamma_a)` form is the derivation, not the operative formula.
- **D3 (G4 instrument)**: G4 measured on `Delta_Lyapunov = Delta_Sagnac + mu_damp * ||D~_a||_F^2` (Tier-3 box, §3.1 upgrade 3) for the SELECTED action's warped generator; raw `Delta_Sagnac` also reported for F16 cross-comparability (F16 G4 damped = 0.9477, raw 0.9607).
- **D4 (beta_Sagnac)**: directive leaves the beam's Sagnac weight unnamed; carried from F16 at 0.05.
- **D5 (creep)**: retained from F15/F16, goal-valence-driven, eta=0.05/zero-valence guard — non-attributable in both prior carriers.
- **D6 (seed/params)**: directive command values — seed `20260916`, horizon 8, beam 8, kappa_diff 0.75, mu_damp 0.15, 150 steps/env x 12 envs = 1,800.
- **D7 (flag/schema)**: engine gated by `HENRI_F17_DIFFERENTIAL=1` (mirrors F15/F16); receipt schema `f17-differential-engine.v1`.

## 4. Execution order

1. TDD: `tests/contract/test_f17_differential_engine.py` (C1–C15: Killing-form identity gamma = <Psi_goal, D Psi_t>; aligned/opposed sign split; tanh bound; exp orthogonality; rank-break anti-lock + F15/F16 degenerate lock reproduction; steering gain (aligned boosted, opposed damped); Lyapunov damping monotonicity; beam determinism; bank/PG1 fail-closed degenerate; flag fail-closed; no-bank fail-closed; valence semantics; module-constant guard; substrate-constructor reachability; engagement-gate sensitivity + verdict wiring).
2. Implement `experiments/verification/arc_f17_differential_engine.py` (flag fail-closed; canonical imports from `arc_f15_trajectory_engine` / `arc_f10_live_engine` / `arc_f11_plasticity_engine` only; no local shadowing).
3. Local suite + full regression from repo root.
4. Commit + push; remote detached-worktree CUDA verify @ exact SHA.
5. Live gauntlet per directive command: 12 envs x 150 = 1,800 steps, seed `20260916`, K=8, kappa_diff 0.75, mu_damp 0.15, bank `/root/f3-run/telemetry/f3_bank_capture_v2/`.
6. Seal `F17_GATES_VERDICT`; deliver scorecard + lessons.

## 5. Kill experiment / cheapest falsification

If the C5 rank-break contract fails locally (goal-direct J(a) still uniform across actions on a PG1-valid synthetic pair) or C6 steering gain fails (aligned generator not boosted / opposed not damped) -> the Killing warp does not break rank invariance on the D=64 substrate -> KILL before launch. If live engagement fails (mean std_a(gamma_a) <= 1e-6) -> `F17_FALSIFIED_NO_ENGAGEMENT`. If live G3 <= 0 and G2 = 0 with engagement confirmed -> candidate-differential Killing steering is falsified against the trajectory goal (rank invariance persists at live scale).

## 6. Falsifiable claim

"HYPOTHESIS: candidate-differential Killing-form modulation (D~_a = (1 + 0.75*tanh(gamma_a)) D_a, gamma_a = -1/2 Tr(D_a Omega_goal)) breaks candidate rank invariance and produces positive goal-directed valence (G3 >= +0.02) with >= 1 live solve (G2 >= 1) under the verified bank ingress on the 12-env live gauntlet, while Lyapunov damping holds coherence at G4 <= 0.05." PG1 is pre-verified on the real bank; G1–G4 measured live on vast-5090.
