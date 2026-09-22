"""Grid-observable readout contract: decode, derived tau, fail-closed.

WHY THIS FILE EXISTS
    The approved step was "wire delta_sagnac_observational into search()". Two
    measured facts constrain what that can honestly mean:

      * `search()` candidates already have a LITERAL grid
        (`child_grid = child_ast.execute(input_grid)`), so the observational veto
        there is an EXACT cell comparison and needs no decode at all. Using a lossy
        decode when ground truth exists is a gratuitous failure mode.
      * The scoring REFERENCE is a wave from `compile_functor` +
        `single_pass_associative_retrieval` (a k_bins = 256 phase algebra). It was NOT
        built by binding role-filler pairs, so decoding it with an independently
        seeded codebook recovers nothing. Measured: a wrong codebook matched 0.125 of
        cells, and the round trip only became exact once the frequency-domain
        conjugate was used (see below).

    So this module is proven on waves IT encoded, and the observable veto is only
    applied where an observable reference genuinely exists.

INVARIANTS ENFORCED
    H1  single bound pair round-trips EXACTLY (separates algebra from capacity)
    H2  an 8x8 lattice round-trips exactly
    H3  decoding with a DIFFERENT codebook does not recover the lattice
    H4  tau is DERIVED from the match-rate null, and is NOT 0.35
    H5  the derived tau separates identical from unrelated (spread > 0.6)
    H6  zero-energy input fails closed, no NaN
    H7  exact grid-vs-grid stress, including shape mismatch
    H8  REGRESSION: the frequency-domain conjugate. A spatial-domain conjugate
        (`ifft(fft(psi) * fft(conj(role)))`) is correlation against a TIME-REVERSED
        role and round-tripped only 25% of cells. It is not a stylistic choice.
    H9  no hardcoded 0.35 anywhere in the module
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from henri_grid_observable import GridObservableReadout, derive_tau  # noqa: E402

DIM = 1024


@pytest.fixture(scope="module")
def ro() -> GridObservableReadout:
    return GridObservableReadout(shape=(8, 8), dim=DIM, n_values=11, seed=20261012)


def make_grid(seed: int = 0) -> np.ndarray:
    g = np.zeros((8, 8), dtype=np.int64)
    g[1:4, 1:4] = 3
    g[5:7, 5:7] = 7
    return g


def perturb(grid: np.ndarray, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = grid.copy()
    idx = rng.choice(out.size, size=n, replace=False)
    for i in idx:
        r, c = divmod(int(i), out.shape[1])
        out[r, c] = int(rng.integers(0, 10))
    return out


# ------------------------------------------------------------------ H1
def test_single_bound_pair_round_trips_exactly():
    """H1. M=1 has no crosstalk, so any failure here is an ALGEBRA bug.

    This gate is what localized the first failure: the 8x8 round trip decoded only
    25% of cells, and M=1 exactness proved the cause was the unbinding form, not
    superposition capacity.
    """
    r1 = GridObservableReadout(shape=(1, 1), dim=DIM, n_values=11, seed=5)
    dec = r1.decode(r1.encode(np.array([[7]], dtype=np.int64)))
    assert dec.valid and int(dec.values[0, 0]) == 7, (
        f"a single binding decoded to {int(dec.values[0,0])}, expected 7; the "
        f"unbinding algebra is wrong so no capacity result is meaningful")


# ------------------------------------------------------------------ H2
def test_lattice_round_trips_exactly(ro):
    """H2."""
    base = make_grid()
    dec = ro.decode(ro.encode(base))
    assert dec.valid
    assert np.array_equal(dec.values, base), (
        f"round trip differed at {int((dec.values != base).sum())} cells")
    assert abs(float(ro.encode(base).norm().item()) - 1.0) < 1e-5


def test_round_trip_margin_is_recorded_not_assumed(ro):
    """The decode is EXACT at dim=1024 but the per-cell margin is thin.

    Measured min cell quality 0.0627. Recording it as an assertion stops a later
    author from mistaking "exact" for "high confidence", and makes a drift in margin
    visible. At the deployment dim the margin is larger; this pins the CPU-test fact.
    """
    dec = ro.decode(ro.encode(make_grid()))
    assert dec.quality.min() == dec.quality.min()  # not NaN
    assert dec.quality.min() < 0.30, (
        "per-cell quality rose above 0.30; the module header claims a thin margin at "
        "dim=1024, so either the measurement or the claim must be updated")


# ------------------------------------------------------------------ H3
def test_wrong_codebook_does_not_recover_the_lattice(ro):
    """H3. Proves the shared-codebook dependency is real, not decorative."""
    other = GridObservableReadout(shape=(8, 8), dim=DIM, n_values=11, seed=999)
    assert ro.codebook_fingerprint() != other.codebook_fingerprint()
    dec = other.decode(ro.encode(make_grid()))
    match = float((dec.values == make_grid()).mean()) if dec.valid else 0.0
    assert match < 0.5, (
        f"a DIFFERENT codebook recovered {match:.2%} of cells, so the decode is "
        f"trivially insensitive to its codebook and the guard is vacuous")


# ------------------------------------------------------------------ H4
def test_tau_is_derived_and_is_not_the_inherited_constant(ro):
    """H4. The r = 0.2682 lesson: a threshold belongs to a METRIC."""
    info = derive_tau(64, 11, 0.01)
    assert abs(ro.tau - info["tau_observational"]) < 1e-9
    assert abs(ro.tau - 0.35) > 0.1, (
        f"tau is {ro.tau:.4f}, suspiciously near the inherited 0.35 waveform constant")
    assert info["null_match_rate"] == pytest.approx(1.0 / 11.0)
    # tau MUST grow with lattice size: a bigger lattice has more chance matches.
    t = [derive_tau(n, 11, 0.01)["tau_observational"] for n in (16, 64, 256)]
    assert t[0] < t[1] < t[2], f"tau did not increase with n_slots: {t}"


def test_derived_tau_rejects_the_036_trap_for_partially_correct_candidates(ro):
    """H4b. 0.35 requires a 65% exact match rate; the derived tau requires ~19%."""
    assert 1.0 - 0.35 == pytest.approx(0.65)
    assert 1.0 - ro.tau < 0.25, (
        f"derived tau {ro.tau:.4f} implies a match-rate floor of "
        f"{1 - ro.tau:.3f}; expected < 0.25 for an 8x8 lattice")


# ------------------------------------------------------------------ H5
def test_observational_stress_separates_and_is_monotone(ro):
    """H5."""
    base = make_grid()
    same = ro.observational_stress(ro.encode(base), base)
    assert same["stress"] < 1e-6, f"identical stress {same['stress']}"
    unrelated = np.random.default_rng(99).integers(0, 10, size=(8, 8)).astype(np.int64)
    unr = ro.observational_stress(ro.encode(unrelated), base)
    spread = unr["stress"] - same["stress"]
    assert spread > 0.6, f"separation only {spread:.4f}"
    # Quasi-monotone under damage. Damage LOCATION matters, so this is a floor not a
    # law; the assertion is that heavy damage exceeds light damage.
    light = ro.observational_stress(ro.encode(perturb(base, 1, 1)), base)["stress"]
    heavy = ro.observational_stress(ro.encode(perturb(base, 16, 4)), base)["stress"]
    assert light < heavy, f"1-cell {light:.4f} not less than 16-cell {heavy:.4f}"


# ------------------------------------------------------------------ H6
def test_zero_energy_wave_fails_closed(ro):
    """H6."""
    z = torch.zeros(DIM, dtype=torch.complex64)
    out = ro.observational_stress(z, make_grid())
    assert out["stress"] == 1.0 and out["valid"] is False
    assert not np.isnan(out["stress"])


# ------------------------------------------------------------------ H7
def test_direct_grid_stress_is_exact(ro):
    """H7. The preferred path whenever a literal observable exists."""
    base = make_grid()
    assert ro.grid_stress(base, base) == 0.0
    assert ro.grid_stress(perturb(base, 1, 1), base) == pytest.approx(1.0 / 64)
    assert ro.grid_stress(np.zeros((2, 2), dtype=np.int64), base) == 1.0


# ------------------------------------------------------------------ H8
def test_unbinding_conjugates_in_the_frequency_domain():
    """H8. REGRESSION. A spatial-domain conjugate is a time reversal, not correlation.

        fft(conj(R))_k = conj(fft(R)_{(-k) mod D})

    so `ifft(fft(psi) * fft(conj(role)))` correlates against a TIME-REVERSED role.
    Measured: that form round-tripped only 25% of cells (min quality 0.036, the noise
    floor). The module must conjugate the SPECTRUM. Checked on the source so the
    wrong form cannot be reintroduced as an "equivalent" rewrite.
    """
    src = (ROOT / "henri_grid_observable.py").read_text(encoding="utf-8")
    assert "conj(torch.fft.fft(roles" in src, (
        "unbinding no longer conjugates the SPECTRUM; a spatial-domain conjugate is "
        "a time reversal and recovers nothing")
    # And the delta identity must hold: role (*) conj(role) is delta-like.
    r1 = GridObservableReadout(shape=(1, 1), dim=DIM, n_values=11, seed=5)
    z = r1._unbind(r1.role_keys[0], r1.role_keys[:1])
    peak_over_mean = float((z.abs().max() / z.abs().mean().clamp_min(1e-30)).item())
    assert peak_over_mean > 10.0, (
        f"role (*) conj(role) peak/mean {peak_over_mean:.3f} is not delta-like")


# ------------------------------------------------------------------ H9
def test_module_hardcodes_no_inherited_threshold():
    """H9. AST-level: 0.35 must not appear as a numeric constant in the module."""
    tree = ast.parse((ROOT / "henri_grid_observable.py").read_text(encoding="utf-8"))
    hits = [n.lineno for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, float)
            and abs(n.value - 0.35) < 1e-12]
    # 0.35 IS allowed inside the documented as-string mention of the trap; only a
    # numeric literal used as a threshold is a violation.
    assert not hits, (
        f"a numeric 0.35 literal appears at lines {hits}; the observational veto "
        f"threshold must be derived (derive_tau), never inherited")
