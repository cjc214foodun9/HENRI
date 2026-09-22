"""Tau derivation contract: must work at the DEPLOYMENT lattice, not just test sizes.

WHY THIS FILE EXISTS
    `derive_tau` was written and tested only at n_slots <= 900. At the lattice size
    that actually matters -- 4096, measured from a live ARC-AGI-3 frame of 64x64 --
    it raised:

        OverflowError: int too large to convert to float

    because it computed `math.comb(4096, k)` directly; math.comb(4096, 2048) is an
    integer with ~1230 digits. So the threshold could not be derived where it is used.

    That is the SAME class of defect this whole module exists to prevent: a threshold
    that works on the test case and fails at the operating point. It is pinned here.

INVARIANTS
    T1  the DEPLOYMENT lattice (64x64, 11 values) derives without error
    T2  the log-space rewrite agrees with the exact combinatorial form wherever the
        exact form is computable (regression: no silent change to known answers)
    T3  tau decreases as alpha decreases (a stricter alpha admits fewer candidates)
    T4  tau rises toward 1 - 1/n_values as the lattice grows -- the metric's null
    T5  Monte-Carlo null agrees with the exact binomial at the deployment lattice
    T6  the derived tau is NOT the inherited 0.35
    T7  the strictness is DOCUMENTED, not discovered: at 4096 slots a candidate may
        get thousands of cells wrong and still pass, because the metric's null is
        weak there. The waveform cosine does the ranking; tau does the veto.
"""
from __future__ import annotations

import math
import sys
from math import comb
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from henri_grid_observable import derive_tau  # noqa: E402

DEPLOY_SLOTS = 64 * 64          # measured: live ARC-AGI-3 frame is 64x64
DEPLOY_VALUES = 11              # measured: frame values 0..10


def exact_tau(n_slots: int, n_values: int, alpha: float) -> float:
    """The direct combinatorial form. Only usable for small n_slots."""
    p = 1.0 / n_values
    pmf = [comb(n_slots, k) * (p ** k) * ((1 - p) ** (n_slots - k))
           for k in range(n_slots + 1)]
    s = sum(pmf)
    pmf = [x / s for x in pmf]
    cum = 0.0
    for k in range(n_slots + 1):
        cum += pmf[k]
        if cum >= 1.0 - alpha:
            return 1.0 - k / n_slots
    return 0.0


# ------------------------------------------------------------------ T1
def test_deployment_lattice_derives_without_overflow():
    """T1. The regression. This raised OverflowError before the log-space rewrite."""
    for alpha in (0.05, 0.01, 0.001):
        t = derive_tau(DEPLOY_SLOTS, DEPLOY_VALUES, alpha)
        assert 0.0 < t["tau_observational"] < 1.0
        assert t["q_1_minus_alpha_matched"] > DEPLOY_SLOTS / DEPLOY_VALUES


def test_large_lattice_also_works():
    """T1b. 30x30 = 900 (the largest ARC eval grid) and beyond must also derive."""
    for n in (900, 4096, 16_384):
        t = derive_tau(n, DEPLOY_VALUES, 0.01)
        assert 0.0 < t["tau_observational"] < 1.0, f"failed at n_slots={n}"


# ------------------------------------------------------------------ T2
def test_log_space_matches_exact_form_where_exact_is_computable():
    """T2. No silent change to answers the old form could produce."""
    for n in (4, 16, 64, 256, 900):
        for alpha in (0.05, 0.01, 0.001):
            new = derive_tau(n, DEPLOY_VALUES, alpha)["tau_observational"]
            old = exact_tau(n, DEPLOY_VALUES, alpha)
            assert abs(new - old) < 1e-9, (
                f"n={n} alpha={alpha}: log-space {new} != exact {old}")


# ------------------------------------------------------------------ T3
def test_tau_is_monotone_in_alpha():
    """T3. A stricter alpha must not admit MORE candidates."""
    for n in (64, 4096):
        t05 = derive_tau(n, DEPLOY_VALUES, 0.05)["tau_observational"]
        t01 = derive_tau(n, DEPLOY_VALUES, 0.01)["tau_observational"]
        t001 = derive_tau(n, DEPLOY_VALUES, 0.001)["tau_observational"]
        assert t05 >= t01 >= t001, f"n={n}: {t05} {t01} {t001}"


# ------------------------------------------------------------------ T4
def test_null_tag_converges_to_one_minus_one_over_n_values():
    """T4. The metric's null is 1 - 1/n_values; tau must approach it from below."""
    target = 1.0 - 1.0 / DEPLOY_VALUES
    small = derive_tau(16, DEPLOY_VALUES, 0.01)["tau_observational"]
    large = derive_tau(16_384, DEPLOY_VALUES, 0.01)["tau_observational"]
    assert small < large < target, (
        f"tau did not converge toward the null from below: 16->{small}, "
        f"16384->{large}, null->{target}")


# ------------------------------------------------------------------ T5
def test_monte_carlo_null_agrees_at_the_deployment_lattice():
    """T5. Exact binomial vs measured null, at the real size."""
    t = derive_tau(DEPLOY_SLOTS, DEPLOY_VALUES, 0.01)["tau_observational"]
    g = torch.Generator().manual_seed(0)
    trials = 40_000
    dec = torch.randint(0, DEPLOY_VALUES, (trials, DEPLOY_SLOTS), generator=g)
    ref = torch.randint(0, DEPLOY_VALUES, (1, DEPLOY_SLOTS), generator=g)
    stress = 1.0 - (dec == ref).float().mean(dim=1)
    assert (dec == ref).float().mean().item() == pytest.approx(1.0 / DEPLOY_VALUES,
                                                              abs=0.01)
    mc = float(stress.quantile(0.01).item())
    assert abs(mc - t) <= 0.01, f"MC q01 {mc:.4f} vs exact tau {t:.4f}"


# ------------------------------------------------------------------ T6
def test_derived_tau_is_not_the_inherited_waveform_constant():
    """T6. The r = 0.2682 lesson."""
    t = derive_tau(DEPLOY_SLOTS, DEPLOY_VALUES, 0.01)["tau_observational"]
    assert abs(t - 0.35) > 0.1, f"tau {t:.4f} is suspiciously near 0.35"
    assert t > 0.85, (
        f"tau {t:.4f} is unexpectedly low for a match-rate metric at 4096 slots; "
        f"the null is 1 - 1/11 = 0.909")


# ------------------------------------------------------------------ T7
def test_strictness_is_documented_and_measured():
    """T7. The veto is STRICT at 4096 slots. That must be recorded, not discovered.

    At the deployment lattice a candidate may have thousands of cells wrong and still
    pass, because one random decode already matches ~1/11 of cells. This is a property
    of the metric, not a bug, and the division of labour is deliberate:
        waveform cosine -> ranks among partially-correct candidates
        observational tau -> vetoes candidates no better than random
    """
    t = derive_tau(DEPLOY_SLOTS, DEPLOY_VALUES, 0.01)
    tau = t["tau_observational"]
    cells_wrong_allowed = DEPLOY_SLOTS - t["q_1_minus_alpha_matched"]
    assert cells_wrong_allowed > 1000, (
        f"expected a strict gate (many cells tolerated), got "
        f"{cells_wrong_allowed} of {DEPLOY_SLOTS}")
    # The contrast with the inherited constant.
    correct_needed_derived = t["q_1_minus_alpha_matched"]
    correct_needed_inherited = 0.65 * DEPLOY_SLOTS
    assert correct_needed_inherited / correct_needed_derived > 5.0, (
        f"inheriting 0.35 would require {correct_needed_inherited:.0f} correct cells "
        f"vs {correct_needed_derived} derived; expected a >5x difference")
