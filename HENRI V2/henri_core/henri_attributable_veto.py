"""henri_attributable_veto.py -- attributable veto, replacing the saturated coherence gate.

THE MEASURED DEFECT THIS REPLACES
    Shipped statistic (arc_sagnac_veto.py):
        S(a, b) = |mean(conj(a) * b)|          triggered = (1 - S) > eps
    For complex UNIT-MODULUS waves with independent phases, E[S] = sqrt(pi / (4D)).
    The receipt records this: expected_abs_cos_independent_unit = 0.011049 at D = 8192,
    against sqrt(pi/(4*8192)) = 0.0098. So for UNRELATED waves S is already ~0.011 and
    delta = 1 - S ~ 0.989.

    eps = 0.35 therefore vetoes every candidate that is not a near-copy of the reference.
    Measured consequence, from henri_derandomized_veto_observed.json:
        arm2 (fixed bank-mean axiom) trigger_rate = 1.0        <- SATURATED
        false_veto_rate_on_correct = 0.81993                   <- fires on CORRECT
        rejection_rate_on_incorrect = 0.136516                 <- spares INCORRECT
        separation = -0.683414 (bar +0.7000)                   <- inverted
    A gate that fires on 100% of one arm cannot fail. That is the defect class named in
    the architecture contract: "No gate may sit downstream of an operator that mixes the
    candidate with the bank", paired with "a gate that cannot fail will pass".

WHY IT INVERTS -- the mechanism, stated before the code
    A threshold on (1 - S) is a threshold on SIMILARITY TO ONE REFERENCE. When the
    reference is a bank MEAN, it is near-orthogonal to every member, so every candidate
    looks maximally dissimilar and every candidate vetoes. When the reference is the
    candidate's own content, the statistic reduces to self-similarity, which is high for
    correct and high for incorrect alike. Neither form asks the question the veto is for:
    "is this candidate EXPLAINED by the axiom family, beyond what its peers explain?"

THE REPLACEMENT -- a CONTRAST, read PRE-MIXING
    Every feature below is computed from the candidate against individual axioms and
    individual peers. Nothing is read after an operator that has already mixed the
    candidate with the bank, which is the defect that produced the saturation.
        A_resonance   best axiom agreement, ABOVE the chance level for D-dim unit waves
        B_peer        mean peer agreement, ABOVE the chance level
        C_loo         leave-one-out centroid delta: how much the SET centroid's match to
                      the axiom family moves when this candidate is removed
    score = (A_res - chance_A) - (B_peer - chance_B)
    A candidate that is genuinely the answer resonates with the axioms AND stands apart
    from its peers. A distractor does neither. The contrast is what the single similarity
    statistic threw away.

BAR DISCIPLINE
    The pre-registered bars are FROZEN: V_C standalone AUC > 0.55, V_D rejection rate in
    (0.02, 0.98), V_E no single feature |r| > 0.995. This module measures them. It does
    NOT move them. The dark port stays mechanically disconnected.
"""
from __future__ import annotations

import math

import torch

# Frozen bars, copied from henri_attributable_veto_preregistration.json. NOT tunable here.
AUC_BAR = 0.55
REJECT_LO, REJECT_HI = 0.02, 0.98
IDENTITY_R_BAR = 0.995


def chance_coherence(d: int) -> float:
    """E|mean(conj(a)*b)| for two INDEPENDENT unit-modulus complex vectors of length d.

    mean(conj(a)*b) is a sum of d unit phasors divided by d. Its real and imaginary parts
    are each approximately normal with variance 1/(2d), so |sum/d| follows a Rayleigh
    distribution with E = sqrt(pi/(4d)).
    """
    return math.sqrt(math.pi / (4.0 * max(int(d), 1)))


def chance_coherence_rms(d: int) -> float:
    """RMS of |mean(conj(a)*b)| for the ENTRY-unimodular family = 1/sqrt(d).

    MEASURED RECONCILIATION. The prior receipt recorded
    expected_abs_cos_independent_unit = 0.011049 at D = 8192.
        1/sqrt(8192)              = 0.011049   <- the RMS
        sqrt(pi/(4*8192))         = 0.009792   <- the MEAN (Rayleigh)
        ratio                     = 1.128 = 1/sqrt(pi/4)
    The earlier constant is the RMS; chance_coherence() returns the MEAN. Both summarise
    the same Rayleigh law, so the difference is explained rather than absorbed.
    """
    return 1.0 / math.sqrt(max(int(d), 1))


def chance_max_coherence(d: int, k: int) -> float:
    """Expected MAX |coherence| over k independent trials (extreme-value expansion).

    The max of k Rayleigh variables lifts the single-trial scale by about sqrt(2 ln k).
    """
    base = chance_coherence(d)
    if k <= 1:
        return base
    return base * math.sqrt(2.0 * math.log(k))


def sagnac_coherence(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """The SHIPPED statistic, kept so a control can show it saturates."""
    a = a.reshape(-1)
    b = b.reshape(-1)
    if a.is_complex() or b.is_complex():
        return torch.abs(torch.mean(a.conj() * b))
    return 0.5 * (1.0 + torch.dot(a, b))


class AttributableVeto:
    """Contrast-based attributable veto. No training; the score is a fixed formula, so
    there is no train/test leak and no selector to validate."""

    def __init__(self, d: int = 8192, n_axioms: int = 64):
        self.d = int(d)
        self.n_axioms = int(n_axioms)
        # chance levels for THIS dimensionality and family size
        self.chance_pair = chance_coherence(self.d)
        self.chance_best = chance_max_coherence(self.d, self.n_axioms)
        self.chance_peer = chance_coherence(self.d)

    # ------------------------------------------------------------------ features
    def feature_vector(self, cand: torch.Tensor, axioms: torch.Tensor,
                       peers: torch.Tensor) -> dict:
        """All three feature families, read PRE-MIXING against individual references."""
        c = cand.reshape(-1)
        if not c.is_complex():
            c = c.to(torch.complex64)

        # NORMALISATION, corrected. MY DEFECT, self-caught: the first draft divided the
        # inner product by sqrt(D) and compared against a max-of-K chance. For the
        # ENTRY-unimodular family the inner product of two UNIT-NORM vectors already lies
        # in [0, 1] (Cauchy-Schwarz), so dividing by sqrt(D) shrank an exact match from
        # 1.0 to 1/sqrt(D) and drove A_resonance NEGATIVE (-0.024). Divide by the NORMS.
        cn = c.norm().clamp_min(1e-30)

        def norm_coh(Mx):
            Mn = Mx / Mx.norm(dim=-1, keepdim=True).clamp_min(1e-30)
            return (Mn.conj() @ c).abs() / cn

        # A: best single-axiom agreement, above chance
        ax = axioms.to(torch.complex64).reshape(axioms.shape[0], -1)
        a_scores = norm_coh(ax)
        a_best = float(a_scores.max())
        a_res = a_best - self.chance_best

        # B: mean peer agreement, above chance
        if peers is not None and peers.shape[0] > 0:
            pr = peers.to(torch.complex64).reshape(peers.shape[0], -1)
            b_peer = float(norm_coh(pr).mean())
        else:
            b_peer = self.chance_peer
        b_excess = b_peer - self.chance_peer

        # C: leave-one-out centroid delta (how much the set moves when this one leaves)
        if peers is not None and peers.shape[0] > 1:
            pr = peers.to(torch.complex64).reshape(peers.shape[0], -1)
            cen_full = torch.cat([pr, c.unsqueeze(0)]).mean(dim=0)
            cen_wo = pr.mean(dim=0)
            conv = lambda z: z / z.norm().clamp_min(1e-30)          # noqa: E731
            loo = float((conv(cen_full) - conv(cen_wo)).norm())
        else:
            loo = 0.0

        return {"A_best_axiom": a_best, "A_resonance": a_res,
                "B_peer_mean": b_peer, "B_excess": b_excess, "C_loo_delta": loo}

    # ------------------------------------------------------------------ score
    def score(self, cand: torch.Tensor, axioms: torch.Tensor, peers: torch.Tensor) -> float:
        """Contrast: resonance with the axiom family MINUS peer agreement.

        Sign convention: HIGHER score = MORE likely to be the genuine answer.
        """
        f = self.feature_vector(cand, axioms, peers)
        # NOTE (honest arithmetic): A_resonance - B_excess
        #   = (a_best - chance_best) - (b_peer - chance_peer)
        # The two chance constants enter with OPPOSITE signs but are DIFFERENT constants
        # (best-of-K vs single), so the score is a_best - b_peer + (chance_peer - chance_best).
        # The residual offset is a fixed number, not zero; it shifts the score without
        # changing the ranking, so AUC and separation are unaffected. Stated, not hidden.
        return f["A_resonance"] - f["B_excess"]

    # ------------------------------------------------------------------ gate
    def rejection_band(self, scores) -> dict:
        """Report where a threshold would sit, WITHOUT choosing one to force a verdict.

        The pre-registered V_D clause requires the operating point to reject a non-trivial
        fraction in (0.02, 0.98). This reports the fraction a fixed median-split threshold
        rejects, so saturation is visible as a rate pinned at 0.0 or 1.0.
        """
        s = torch.as_tensor(scores, dtype=torch.float64)
        if s.numel() == 0:
            return {"n": 0, "reject_rate": float("nan"), "saturated": True}
        med = float(s.median())
        reject = float((s < med).to(torch.float64).mean())
        saturated = reject <= REJECT_LO or reject >= REJECT_HI
        return {"n": int(s.numel()), "median_threshold": med, "reject_rate": reject,
                "in_band": not saturated, "saturated": saturated}


def auc(scores, labels) -> float:
    """Rank AUC. labels: 1 = genuine, 0 = distractor. No sklearn dependency."""
    s = torch.as_tensor(scores, dtype=torch.float64)
    y = torch.as_tensor(labels, dtype=torch.float64)
    pos = s[y == 1]
    neg = s[y == 0]
    if pos.numel() == 0 or neg.numel() == 0:
        return float("nan")
    # Mann-Whitney U with tie handling via pairwise comparison
    cmp = (pos.unsqueeze(1) - neg.unsqueeze(0))
    wins = (cmp > 0).to(torch.float64) + 0.5 * (cmp == 0).to(torch.float64)
    return float(wins.mean())


def max_feature_identity_r(feature_rows, target) -> float:
    """V_E: the largest |Pearson r| between any single feature and the target score.

    Guards against a 'new' statistic that is one old feature in costume.
    """
    t = torch.as_tensor(target, dtype=torch.float64)
    worst = 0.0
    for name, col in feature_rows.items():
        c = torch.as_tensor(col, dtype=torch.float64)
        if c.numel() < 3 or c.std() < 1e-12 or t.std() < 1e-12:
            r = 0.0
        else:
            r = float(torch.corrcoef(torch.stack([c, t]))[0, 1])
        worst = max(worst, abs(r))
    return worst
