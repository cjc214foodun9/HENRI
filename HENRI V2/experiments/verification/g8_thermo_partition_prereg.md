# G8 pre-registration — timescale-separated partition-function calibration
Date: 2026-09-08. Carrier: carrier/g8-thermo-calibration (from origin/main f4dcb69).
Source: Corberi, dello Russo, Messuti, Scarpetta, Smaldone, "Thermodynamic
learning", arXiv:2609.04732v1 (4 Sep 2026). Paper text extracted and read at
primary source: henri-telemetry/corberi_2609_04732.txt (6 pages).

## What is transferred (and what is NOT)
TRANSFERRED (structural, falsifiable):
  1. Three widely separated timescales: fast sigma (Zone A / candidate settling),
     intermediate S^alpha (Zone B / policy exposure), slow j (Zone C / axioms).
  2. The partition-function hierarchy Eqs (5)-(8): integrate out fast variables
     -> Z_{j,S} / free energy F_{j,S}; integrate exposure -> F_j; P(j) ~ e^{-beta_j F_j}.
  3. The ORDER OF LIMITS beta -> beta_0: prior (beta_S -> 0 FIRST) vs post
     (beta_S -> 0 LAST). Paper: prior realizes global optimization over ALL
     inputs and is expected to learn better; post optimizes per current belief.
  4. Ratios n = beta_S/beta_sigma, m = beta_S/beta_j -> 0 encode the separation.
NOT TRANSFERRED: Ising spin dynamics, NP-hard spin-glass ground-state search,
Hadamard-orthogonal pattern capacity, any claim of thermodynamic-device parity.
HENRI keeps continuous wave geometry; the transfer is the CALIBRATION LAW.

## Zone mapping (wire targets, live code verified)
  Zone A: observation codec settling (telemetry of beta ratios only).
  Zone B: candidate selection in the planner (efe_planner.pragmatic_value
          surprise = min over axioms; selection currently argmin of EFE).
  Zone C: boundary axioms reweighting (load_boundary_axioms ->
          USE_ZONE_C_AXIOMS=1; verify_wave_integrity acceptance).

## Mechanism (hypothesis to falsify)
At fixed exposure count e, the selection temperature follows the canonical
three-scale schedule:
  beta_sigma(e) = 4.0 * e^0.5, beta_S(e) = 0.25 * e^-1.0 (prior) or e^-0.25
  (post), beta_j(e) = 6.0 * e^0.75; n, m -> 0.
Zone B: policy P(a) = e^{-beta_sigma E(a)} / Z (Gibbs), deterministic under
  fixed rng seed. As beta_sigma -> inf this recovers current argmin.
Zone C: axiom influence reweighted P(j) = e^{-beta_j F_j} / Z_j; the surprise
  term is softmin(-log sum exp) whose beta_j -> inf limit recovers the current
  hard min over axioms.
ANTI-RESCALE requirement: a signal that only rescales all candidates is
NO evidence; the partition policy must be able to CHANGE selection rank
(measured by Gibbs entropy > 0 at finite beta and rank-change count > 0 on
synthetic near-tied scores). Any claim that "v8 changes ranking" requires
rank-change evidence — a rescaled argmin is FALSIFIED.

## Pre-registered acceptance (default-OFF carrier, numpy/torch self-contained)
T1 softmin hard-min limit: softmin(v, beta=1e6) == min(v) within 1e-4.
T2 softmin zero-beta drift: beta=1e-6 -> softmin ~ -log(n)/beta (uniform).
T3 Gibbs concentration: entropy(gibbs(v, beta)) strictly decreasing in beta;
  argmax(gibbs(v, beta)) == argmin(v) for all beta > 0.
T4 Selection rank change: on near-tied scores, a fixed-seed Gibbs draw can
  select a non-argmin candidate (measured rank_change_count >= 1 on the
  synthetic set), proving the policy is not a rescale of the comparison.
T5 Schedule separation: for all e in [1,64]: beta_sigma > beta_S,
  beta_j > beta_S, beta_S > 0; n,m -> 0 as e -> infinity.
T6 Prior vs post: at same e, prior path has SMALLER n (faster beta_S decay);
  exposure-free-energy(uniform) != exposure-free-energy(peak belief) on a
  crafted free-energy vector.
T7 Determinism: identical inputs -> identical outputs (same seed).
T8 Axiom reweight concentration: as beta_j -> inf, axiom_weights -> delta on
  lowest free energy (recovers hard-min acceptance).
Kill: any of T1-T8 FAIL under the default-OFF carrier, OR the wiring probe
fails to reproduce a measured rank change on the real EFE candidate scores
(the anti-rescale gate) — then v8 is a sealed negative (default-OFF, no main).

## Evidence class
CALIBRATION_LAW_EVIDENCE (canonical-stat-mech schedule on wave selection).
NOT a task score, NOT an AAII score, NOT production egress. Carrier stays
default-OFF; main promotion requires a separate approval gate.

## Deferred (explicitly out of scope)
Spin-glass equivalence, any claim of Ising-ground-state parity, replacement
of the current anneal schedule on the default path, Zone A/B/C schema changes.
