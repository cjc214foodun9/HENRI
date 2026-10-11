"""Discriminating tests for the cross-fold ENSEMBLE SPREAD evidence, and for the SCOPE
refinement it forces on my own prior claim.

WHY THIS FILE EXISTS
    My earlier cross-fold receipt recorded "scale hypothesis FALSIFIED" and left it there.
    The probe also computed an ensemble-spread arm (STEP 3) that my first log read truncated.
    That arm is now verified present in the committed receipt, and it forces a SCOPE
    refinement of my own claim:

        "scale hypothesis FALSIFIED" is correct FOR MEDIAN FILTRATION SCALE.
        It is SILENT ABOUT DISPERSION, which differs 2.3x.

    Those are two DISTINCT hypotheses and I conflated their scope. Pinning the distinction
    here prevents a later reader from over-reading the prior claim in either direction.

MEASURED, from henri_crossfold_fullwidth_observed.json
    1D3Z  n=10   pairs=45    mean RMSD 1.158 A   max 2.067 A   p95 1.914 A
    2K39  n=116  pairs=6670  mean RMSD 2.662 A   max 6.941 A   p95 4.024 A
    spread ratio 2K39/1D3Z = 2.30x
    median CA distance 1D3Z 15.127 A   2K39 15.127 A   EQUAL

THE MECHANISM THE REFINEMENT EXPOSES
    Wider dispersion lengthens the distance distribution's tails. Longer tails push the
    CROSS minimum down, which SHRINKS min_cross -- the same min-order-statistic effect that
    made the extremum bar pair-count dependent (0.922 -> 1.609 across 100 -> 1160 pairs).
    Two observations, one mechanism. The statistic is not sensitive to dispersion in the way
    the 0.8000 bar demands.

SCOPE. Superposed pairwise RMSD is a PROXY, not the cited 3.29 A statistic. Single seed.
CPU. No training. No bar moved. No CUDA. No AAII claim.
"""
import json
import os

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
V = os.path.abspath(os.path.join(HERE, "..", ".."))
REPO = os.path.abspath(os.path.join(V, ".."))
RECEIPT = os.path.join(REPO, "design", "zone_a", "evidence",
                       "henri_crossfold_fullwidth_observed.json")

BAR = 0.8000


def _load():
    if not os.path.exists(RECEIPT):
        pytest.skip("cross-fold full-width receipt not produced yet")
    return json.load(open(RECEIPT, encoding="utf-8"))


# ============================================== THE EVIDENCE IS IN THE COMMITTED RECEIPT
def test_committed_receipt_carries_the_ensemble_spread_arm():
    """STEP 3 must be present. It was computed but not displayed by my first log read, so
    its presence is asserted rather than assumed."""
    R = _load()
    assert "ensemble_spread" in R, "the spread arm must be recorded"
    for pid in ("1D3Z", "2K39"):
        s = R["ensemble_spread"][pid]
        assert s["n_models"] > 0 and s["n_pairs"] > 0
        assert s["mean_rmsd_A"] > 0 and s["max_rmsd_A"] >= s["mean_rmsd_A"]
        # the proxy must label itself, so it is never read as the cited statistic
        assert "proxy" in s["note"].lower()
        assert "not the cited" in s["note"].lower()
    # pair counts match the combinatorics n(n-1)/2 -- a real constraint, so this can fail
    for pid in ("1D3Z", "2K39"):
        n = R["ensemble_spread"][pid]["n_models"]
        assert R["ensemble_spread"][pid]["n_pairs"] == n * (n - 1) // 2, \
            f"{pid}: n_pairs must equal n(n-1)/2"


# ============================================== THE SCOPE REFINEMENT, AS A TESTABLE CLAIM
def test_median_scale_is_equal_so_scale_cannot_explain_a_difference():
    """HALF THE REFINEMENT. If the medians are equal, a global scale effect cannot explain
    any observed separability gap. Asserted numerically, not narrated."""
    R = _load()
    m = R["truncation_test"]["median_full_A"]
    assert m["1D3Z"] == pytest.approx(m["2K39"], abs=0.01), \
        f"medians {m['1D3Z']} vs {m['2K39']} are NOT equal; the scope claim would be wrong"
    assert m["1D3Z"] == pytest.approx(15.127, abs=0.01)


def test_dispersion_differs_materially_so_spread_is_a_distinct_hypothesis():
    """THE OTHER HALF, and the reason my prior claim needed scoping. Spread differs 2.3x
    while the median is equal. Two distinct hypotheses; conflating them was my error."""
    R = _load()
    a = R["ensemble_spread"]["1D3Z"]["mean_rmsd_A"]
    b = R["ensemble_spread"]["2K39"]["mean_rmsd_A"]
    ratio = b / a
    assert ratio == pytest.approx(2.30, abs=0.05), f"spread ratio {ratio:.3f} != 2.30"
    assert b > a, "2K39 must be the more disperse ensemble"
    # and the medians are equal in the same receipt, so the two facts coexist
    m = R["truncation_test"]["median_full_A"]
    assert m["1D3Z"] == pytest.approx(m["2K39"], abs=0.01)


def test_cited_rmsd_sits_between_proxy_mean_and_p95():
    """The cited full-ensemble rmsd 3.29 A must be CONSISTENT with, but not equal to, my
    proxy. Asserting the bracketing keeps the proxy honest: it is same-order evidence."""
    R = _load()
    s = R["ensemble_spread"]["2K39"]
    assert s["mean_rmsd_A"] < 3.29 < s["p95_rmsd_A"], \
        (f"cited 3.29 A must lie between the proxy mean {s['mean_rmsd_A']:.3f} "
         f"and p95 {s['p95_rmsd_A']:.3f}")


# ============================================== THE VERDICT IS UNCHANGED
def test_verdict_and_bar_are_unchanged_by_the_refinement():
    """A refinement of scope must NOT move a bar. The extremum bar stays 0.8000 and stays
    FAIL, and the truncation outcome stays representative."""
    R = _load()
    assert R["verdict"] == "CROSSFOLD_STILL_NOT_SEPARATED_AT_FULL_WIDTH"
    assert R["preregistered"]["bar_extremum_ratio"] == BAR
    assert R["truncation_test"]["outcome"] == "H_representative"
    for tag in ("truncated", "full"):
        for arm in ("A_raw_betti", "C_landscape"):
            assert R["arms"][tag][arm]["passes_bar"] is False, \
                f"{tag}/{arm} must not clear the bar"


def test_the_min_order_statistic_mechanism_is_recorded():
    """The refinement exposes ONE mechanism behind two observations: a shrinking minimum
    under a growing pair count. The receipt must name it, so the link is not lost."""
    R = _load()
    t = R["truncation_test"]
    assert t["representative"] is True
    # the extremum ratio moved materially while the median scale did not
    tr_ext = R["arms"]["truncated"]["C_landscape"]["ratio_within_over_cross"]
    fu_ext = R["arms"]["full"]["C_landscape"]["ratio_within_over_cross"]
    assert fu_ext > tr_ext, "the extremum ratio must degrade as the cross set grows"
    tr_mean = R["arms"]["truncated"]["C_landscape"]["mean_ratio"]
    fu_mean = R["arms"]["full"]["C_landscape"]["mean_ratio"]
    assert abs(fu_mean - tr_mean) < 0.05, \
        f"the MEAN ratio must stay stable, got {tr_mean:.3f} -> {fu_mean:.3f}"
