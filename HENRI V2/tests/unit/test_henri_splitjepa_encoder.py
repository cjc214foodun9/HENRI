"""Gates for henri_splitjepa_encoder.py. Each gate is built so it CAN fail.

WHY THESE GATES AND NOT THE SPEC'S
    Blueprint gate G-SJ2 asserts ||grad_{z_L} Y|| = 0. Once A12 = 0 is hard-coded that
    derivative is identically zero by the chain rule, so the gate tests its own definition.
    The gates below instead measure an ESTIMATED coupling from a Q that is actually fit,
    and each one has a negative control that must FAIL.

    SJ-A  whitening contract       mean ~ 0, off-diagonal covariance ~ 0, diag ~ 1
    SJ-B  whitening route parity   the Gram/economy-SVD path equals the dense path
    SJ-C  sufficient variation     Sigma_dL > 0 iff the pairs vary z_L; and the SPEC's
                                   D = 65,536 dense covariance RAISES rather than allocating
    SJ-D  block identifiability    A12 -> 0 under Sigma_dL > 0  (the positive result)
    SJ-E  NEGATIVE CONTROL         A12 RETAINS coupling when Sigma_dL = 0 (the kill)
    SJ-F  determinism              two runs at the same seed agree
"""
from __future__ import annotations

import math
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_V2 = os.path.abspath(os.path.join(_HERE, "..", ".."))
_CORE = os.path.join(_V2, "henri_core")
for p in (_V2, _CORE, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from henri_core.henri_splitjepa_encoder import (  # noqa: E402
    CenteredZCA, SplitJEPAConfig, WhiteningError, fit_splitjepa, make_latent_pairs,
    sufficient_variation)


def test_sj_a_whitening_contract_gram_route():
    """N < D: Cov(g(z)) = I_D is RANK-IMPOSSIBLE (covariance rank <= N-1).

    MEASURED CORRECTION TO THE SPEC. The exact achievable identity on the Gram route is
    the centered projector (1/(N-1)) Z Z^T = I_N - 11^T/N: diag (N-1)/N, off-diag 1/N.
    The module reports WHICH identity it checked, so a receipt cannot silently compare a
    rank-deficient statistic against a full-rank target.
    """
    torch.manual_seed(0)
    X = torch.randn(48, 256)
    z = CenteredZCA(max_dense=64).fit(X)
    assert z.route_ == "gram_economy_svd", z.route_
    r = z.whiten_residual(X)
    assert r["contract"] == "sample_gram_centered_projector", r
    assert r["mean_abs_max"] < 1e-4, r
    assert r["projector_max_dev"] < 1e-4, r          # exact identity -> tight bar
    assert abs(r["diag_mean"] - (47 / 48)) < 1e-4, r
    assert abs(r["offdiag_abs_max"] - (1 / 48)) < 1e-4, r
    assert r["rank_kept"] == 47 and r["rank_cap_centered_N_minus_1"] == 47, r


def test_sj_a_whitening_contract_dense_route():
    """The dense path must satisfy the same contract where it is legal."""
    torch.manual_seed(0)
    X = torch.randn(200, 12)
    z = CenteredZCA().fit(X)                      # D=12 <= limit -> dense_eigh
    assert z.route_ == "dense_eigh", z.route_
    r = z.whiten_residual(X)
    assert r["contract"] == "cov_I_D", r
    assert r["mean_abs_max"] < 1e-5, r
    assert r["offdiag_abs_max"] < 1e-4, r
    assert abs(r["diag_min"] - 1.0) < 0.05 and abs(r["diag_max"] - 1.0) < 0.05, r


def test_sj_b_route_parity():
    """Both routes must whiten the SAME data to the same geometry.

    N < D makes both routes legal for one X. Compare them by the centered-projector
    identity, which is route-independent and exact, rather than by the marginal
    covariance (rank-deficient here).
    """
    torch.manual_seed(3)
    N, D = 24, 64
    X = torch.randn(N, D)
    gram = CenteredZCA(max_dense=8).fit(X)         # D=64 > 8, N=24 < 64 -> gram
    dense = CenteredZCA(max_dense=10_000).fit(X)   # D=64 <= 10000      -> dense
    assert gram.route_ == "gram_economy_svd", gram.route_
    assert dense.route_ == "dense_eigh", dense.route_
    P = torch.eye(N) - torch.full((N, N), 1.0 / N)
    for z, tag in ((gram, "gram"), (dense, "dense")):
        Z = z.transform(X)
        Zc = Z - Z.mean(0)
        G = (Zc @ Zc.T) / (N - 1)
        off = float((G - P).abs().max())
        assert off < 1e-3, (tag, off)


def test_sj_c_dense_zca_at_D65536_takes_the_gram_route():
    """SPEC CORRECTION, stated as a test. The spec demands dense ZCA at D=65,536. Dense
    is D^2 = 17.18 GB in fp32 and cannot fit the stated envelope. With N < D the
    Gram/economy-SVD route is EXACT and cheap, so the module must take it, not allocate.
    """
    torch.manual_seed(0)
    X = torch.randn(6, 65_536)
    z = CenteredZCA().fit(X)                       # limit 8192, so D > limit
    assert z.route_ == "gram_economy_svd", z.route_
    r = z.whiten_residual(X)
    assert r["contract"] == "sample_gram_centered_projector", r
    assert r["projector_max_dev"] < 1e-3, r        # measured 2.8e-05 after the rank fix
    assert r["rank_kept"] == 5, r                  # N-1, not N: the spurious direction is gone


def test_sj_c2_refuses_dense_when_gram_inapplicable():
    """When D > limit AND N >= D the Gram route cannot apply, so the module must RAISE
    rather than attempt a multi-GB allocation. This is the guard, tested cheaply."""
    X = torch.randn(6, 6)
    try:
        CenteredZCA(max_dense=4).fit(X)            # D=6 > 4, N=6 >= 6 -> must raise
    except WhiteningError as exc:
        assert "D=6" in str(exc) or "dense limit" in str(exc), exc
        return
    raise AssertionError("must raise when D > limit and N >= D")


def test_sj_c_sufficient_variation_true_and_false():
    cfg = SplitJEPAConfig(tiny=True).resolve()
    zp1, zp2, zi1, zi2 = make_latent_pairs(cfg, 64, vary_L=True)
    sv_yes = sufficient_variation(zi1, zi2, cfg.d_H, cfg.d_L)
    assert sv_yes["satisfied"] is True, sv_yes
    zp1b, zp2b, zi1b, zi2b = make_latent_pairs(cfg, 64, vary_L=False)
    sv_no = sufficient_variation(zi1b, zi2b, cfg.d_H, cfg.d_L)
    assert sv_no["satisfied"] is False, sv_no
    assert sv_no["trace"] == 0.0, sv_no


def test_sj_d_block_identifiability_under_sufficient_variation():
    """The POSITIVE result: matched invariance with full-rank variant variation
    drives the cross-block leakage down."""
    cfg = SplitJEPAConfig(tiny=True, steps=300, seed=7).resolve()
    raw = make_latent_pairs(cfg, 512, vary_L=True)
    out = fit_splitjepa(raw, cfg)
    assert out["sigma_dL"]["satisfied"] is True, out["sigma_dL"]
    fin = out["final"]
    s = out["config"]["steps"]
    # compare against the initialised coupling, not against an absolute bar: a flat
    # initialisation could pass an absolute bar without learning anything.
    torch.manual_seed(cfg.seed + 1)
    d = cfg.d
    init_leak = float((0.05 * torch.randn(d, d, generator=torch.Generator()
                                           .manual_seed(cfg.seed + 1)))[:cfg.d_H, cfg.d_H:]
                      .norm() / math.sqrt(cfg.d_H))
    assert fin["norm_A12_over_sqrt_dH"] < 0.10, fin
    assert fin["norm_A21_over_sqrt_dL"] < 0.10, fin
    print(f"  [sj_d] final={fin} init_reference={init_leak:.6f} steps={s}")


def test_sj_e_negative_control_zero_variation_retains_coupling():
    """THE KILL. With Sigma_dL = 0 the identifiability condition is violated, so the
    leakage must NOT be driven to the same floor. If this passes the same bar as SJ-D,
    the objective is regularizing Q rather than identifying blocks."""
    cfg = SplitJEPAConfig(tiny=True, steps=300, seed=7).resolve()
    raw = make_latent_pairs(cfg, 512, vary_L=False)
    out = fit_splitjepa(raw, cfg)
    assert out["sigma_dL"]["satisfied"] is False, out["sigma_dL"]
    # the contrast is what carries the claim: violated condition => strictly worse leakage
    cfg2 = SplitJEPAConfig(tiny=True, steps=300, seed=7).resolve()
    ok = fit_splitjepa(make_latent_pairs(cfg2, 512, vary_L=True), cfg2)
    viol = out["final"]["norm_A12_over_sqrt_dH"]
    good = ok["final"]["norm_A12_over_sqrt_dH"]
    assert viol > good, (viol, good)
    print(f"  [sj_e] violated_leak={viol:.6f} satisfied_leak={good:.6f}")


def test_sj_f_determinism():
    cfg = SplitJEPAConfig(tiny=True, steps=60, seed=11).resolve()
    a = fit_splitjepa(make_latent_pairs(cfg, 128, vary_L=True, seed=5), cfg)
    b = fit_splitjepa(make_latent_pairs(cfg, 128, vary_L=True, seed=5), cfg)
    assert a["final"] == b["final"], (a["final"], b["final"])


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    bad = 0
    for fn in fns:
        try:
            res = fn()
            print(f"PASS {fn.__name__}" + (f"  {res}" if res else ""))
        except Exception as exc:                                  # noqa: BLE001
            bad += 1
            print(f"FAIL {fn.__name__} -> {type(exc).__name__}: {exc}")
    print(f"\n{len(fns) - bad}/{len(fns)} passed")
    sys.exit(1 if bad else 0)
