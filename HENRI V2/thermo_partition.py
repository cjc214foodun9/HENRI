"""G8 — Corberi et al. (arXiv:2609.04732) timescale-separated partition hierarchy.

PURPOSE (bounded, default-OFF, zero trainable parameters, no DB writes):
make the calibration path canonical statistical mechanics instead of ad-hoc
schedules. Transferable, NOT claimed-equivalent: the paper's Ising spins are
mapped to HENRI wave states ONLY through the *hierarchy of timescales* and the
*order of limits*; no spin-glass equivalence is asserted.

Mapping (Zone A / Zone B / Zone C == fast / intermediate / slow):
  Zone A (fast, sigma): observation codec settling (telemetry guard only).
  Zone B (intermediate, S^alpha): candidate/policy exposure; the fast free
      energy is Z_{j,S} = sum_a exp(-beta_sigma * E(a)) over candidate waves;
      the policy is the Gibbs measure P(a) = e^{-beta_sigma E(a)} / Z.
  Zone C (slow, j): boundary axioms / couplings; P(j) = e^{-beta_j F_j} / Z_j
      reweights axiom influence; the fast free energy at the axiom scale is a
      SOFTMIN surprise whose hard-min limit recovers the current behavior.

Equation map (Corberi Eqs 5-8):
  Z_{j,S}   = sum_{sigma} e^{-beta_sigma (H_int+H_in+H_out)}   -> self.policy_z
  F_{j,S}   = -(1/beta_sigma) log Z_{j,S}                      -> softmin surprise
  e^{-beta_S F_j} = sum_alpha e^{-beta_S (F_{j,S}+H_S)}        -> exposure aggregation
  P({j})    = Z^{-1} e^{-beta_j F_j}                           -> self.axiom_weights
  n = beta_S/beta_sigma, m = beta_S/beta_j -> 0
  beta->beta_0^prior: beta_S -> 0 FIRST (uniform a priori data; global
      optimization over ALL inputs; paper: expected better).
  beta->beta_0^post: beta_S -> 0 LAST (inputs accord to current belief).

The ad-hoc schedule T(t_hat)=T_base*(1-t_hat)^alpha is left untouched on the
default path; under HENRI_THERMO_PARTITION=1 the canonical ratio schedule
replaces it for the selection temperature.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# --- canonical beta ratio schedule (prior path; deterministic, closed-form) ---
BETA_S0 = 0.25      # exposure inverse temperature (small: inputs near-free)
BETA_SIGMA0 = 4.0   # fast inverse temperature (large: candidates near-ground)
BETA_J0 = 6.0       # slow inverse temperature (largest: axioms near-frozen)
P_EXP = 0.5         # growth exponent for beta_sigma with exposure count
Q_EXP = 0.75        # growth exponent for beta_j


@dataclass
class BetaRatios:
    """Timescale-separated inverse temperatures at exposure count e."""
    e: int
    beta_sigma: float
    beta_s: float
    beta_j: float
    n: float
    m: float
    limit_order: str  # "prior" or "post"

    def as_dict(self) -> dict:
        return {
            "e": int(self.e),
            "beta_sigma": round(float(self.beta_sigma), 6),
            "beta_s": round(float(self.beta_s), 6),
            "beta_j": round(float(self.beta_j), 6),
            "n": round(float(self.n), 6),
            "m": round(float(self.m), 6),
            "limit_order": self.limit_order,
        }


def schedule(e: int, *, limit_order: str = "prior") -> BetaRatios:
    """Canonical three-scale inverse-temperature schedule.

    Prior path: beta_S decays to 0 (inputs become free); beta_sigma and beta_j
    grow to infinity. Ratios n=beta_S/beta_sigma, m=beta_S/beta_j -> 0.
    Post path (for comparison): beta_S decays SLOWLY (stays finite longer),
    i.e. the input scale feels the system's belief (beta_S decays only after
    beta_sigma, beta_j have grown; exponent ordering flipped).
    """
    e = max(1, int(e))
    if limit_order == "prior":
        beta_s = BETA_S0 * (e ** -1.0)          # -> 0 fast
        beta_sigma = BETA_SIGMA0 * (e ** P_EXP)  # -> inf
        beta_j = BETA_J0 * (e ** Q_EXP)          # -> inf
    elif limit_order == "post":
        beta_s = BETA_S0 * (e ** -0.25)          # -> 0 slowly (belief-coupled)
        beta_sigma = BETA_SIGMA0 * (e ** P_EXP)
        beta_j = BETA_J0 * (e ** Q_EXP)
    else:
        raise ValueError(f"unknown limit_order {limit_order!r}")
    # order-of-limits invariants
    if not (beta_sigma > beta_s and beta_j > beta_s and beta_s > 0.0):
        raise ValueError(f"schedule violates timescale separation: "
                         f"beta_sigma={beta_sigma}, beta_s={beta_s}, beta_j={beta_j}")
    return BetaRatios(e=e, beta_sigma=float(beta_sigma), beta_s=float(beta_s),
                      beta_j=float(beta_j), n=float(beta_s / beta_sigma),
                      m=float(beta_s / beta_j), limit_order=limit_order)


def softmin(values: np.ndarray, beta: float) -> float:
    """-log sum_i exp(-beta * v_i) / beta = -(1/beta) log Z (exact).

    At beta -> infinity recovers min(values) (hard-min limit);
    at beta -> 0 recovers -log(n)/beta drift to -mean-ish (uniform).
    """
    v = np.asarray(values, dtype=np.float64)
    beta = max(float(beta), 1e-12)
    m = float(np.min(v))
    z = float(np.sum(np.exp(-beta * (v - m))))
    return m - (1.0 / beta) * math.log(z)


def gibbs_weights(values: np.ndarray, beta: float) -> np.ndarray:
    """P(a) = e^{-beta E(a)} / Z — the Gibbs (Boltzmann) policy weights."""
    v = np.asarray(values, dtype=np.float64)
    beta = max(float(beta), 1e-12)
    m = float(np.min(v))
    w = np.exp(-beta * (v - m))
    return w / (w.sum() + 1e-300)


def gibb_select(values: np.ndarray, beta: float, rng: np.random.Generator) -> int:
    """Sample an index from the Gibbs measure; deterministic given rng seed."""
    w = gibbs_weights(values, beta)
    return int(rng.choice(len(w), p=w))


def rank_change_count(baseline_order: list[int], thermo_order: list[int]) -> int:
    """Count positions whose selected index differs (anti-rescale check).

    A signal that only rescales all candidates keeps the same argmin/top-1;
    a real rank change must move at least one position.
    """
    n = min(len(baseline_order), len(thermo_order))
    return int(sum(1 for i in range(n) if baseline_order[i] != thermo_order[i]))


def entropy(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    p = p / (p.sum() + 1e-300)
    nz = p[p > 0.0]
    return float(-(nz * np.log(nz)).sum())


def axiom_weights(free_energies: np.ndarray, beta_j: float) -> np.ndarray:
    """P(j) = e^{-beta_j F_j} / Z_j over axiom free energies (Zone C)."""
    return gibbs_weights(np.asarray(free_energies, dtype=np.float64), beta_j)


def exposure_free_energy(free_energies: np.ndarray, beta_s: float,
                         beliefs: np.ndarray | None = None) -> float:
    """F_j = -(1/beta_S) log sum_alpha e^{-beta_S F_{j,S}} (Eq 6).

    beta->beta_0^prior: beliefs=None -> uniform a priori weights (global
    optimization over ALL exposures; the paper's preferred learning path).
    beta->beta_0^post: beliefs given -> exposure reweighted by current belief
    (inputs accord to the system's current state; per-single-input optimum).
    """
    v = np.asarray(free_energies, dtype=np.float64)
    if beliefs is None:
        return softmin(v, beta_s)
    b = np.asarray(beliefs, dtype=np.float64)
    b = b / (b.sum() + 1e-300)
    return float(np.sum(b * v))


@dataclass
class ThermoCalibrator:
    """Closed-form calibration object (no state, no parameters)."""

    num_axioms: int = 0
    num_candidates: int = 0
    exposure_e: int = 1
    limit_order: str = "prior"

    def ratios(self) -> BetaRatios:
        return schedule(self.exposure_e, limit_order=self.limit_order)

    def policy_weights(self, candidate_scores: np.ndarray) -> np.ndarray:
        """Gibbs policy over per-candidate EFE scores (Zone B selection)."""
        r = self.ratios()
        return gibbs_weights(candidate_scores, r.beta_sigma)

    def axiom_reweight(self, axiom_scores: np.ndarray) -> np.ndarray:
        """P(j) reweights per-axiom energies before the softmin surprise."""
        r = self.ratios()
        return axiom_weights(axiom_scores, r.beta_j)

    def surprise(self, candidate_scores: np.ndarray,
                 axiom_energies: np.ndarray | None = None) -> float:
        """Fast free energy: softmin over axiom energies, hard-min limit
        recovers EFEPlanner's current min-over-axioms surprise."""
        if axiom_energies is None or axiom_energies.size == 0:
            return float(np.min(candidate_scores))
        r = self.ratios()
        return softmin(np.asarray(axiom_energies, dtype=np.float64), r.beta_j)
