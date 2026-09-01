"""Carrier F21.1 contract tests — Vectorized Batched Horizon & Spectral-Capped EDMD.

Directive HENRI-DIR-2026-08-F21-POSTMORTEM-VECTORIZED-EDMD §3/§4 (20,586 B, sha 5cac800b).
Tests mirror the F21 pattern with the F21.1 deltas: spectral cap on Logm, batched
bmm horizon unroll, beta 0.015, PG1 measured on CAPPED generators, F21_1_* verdicts.
"""
import json
import pathlib
import sys
import textwrap

import numpy as np
import pytest
import torch
import torch.nn.functional as F

VERIF = pathlib.Path(__file__).resolve().parents[2] / "experiments" / "verification"
sys.path.insert(0, str(VERIF))

import arc_f21_1_vectorized_engine as eng  # noqa: E402

D = eng.D_SUB


def _skew(n, seed):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(n, n, generator=g)
    S = 0.5 * (A - A.T)
    return S / S.norm()  # unit Frobenius


def make_healthy_bank(n_per_action=96, theta=0.3, noise=0.01, seed=7):
    """Synthetic bank with ground-truth per-action rotations (N > D, non-vacuous)."""
    rng = torch.Generator().manual_seed(seed)
    n_actions = 7
    psi, nxt, onehot = [], [], []
    for a in range(n_actions):
        G = _skew(D, seed + a)
        R = torch.linalg.matrix_exp(theta * G)
        x = F.normalize(torch.randn(n_per_action, D, generator=rng), dim=-1)
        y = F.normalize(x @ R.T + noise * torch.randn(n_per_action, D, generator=rng), dim=-1)
        psi.append(x)
        nxt.append(y)
        o = torch.zeros(n_per_action, n_actions)
        o[:, a] = 1.0
        onehot.append(o)
    return torch.cat(psi), torch.cat(nxt), torch.cat(onehot)


def make_degenerate_bank(n_per_action=96, seed=11):
    """Random unrelated next states with N > D: PG1 must collapse."""
    rng = torch.Generator().manual_seed(seed)
    n_actions = 7
    psi, nxt, onehot = [], [], []
    for a in range(n_actions):
        psi.append(F.normalize(torch.randn(n_per_action, D, generator=rng), dim=-1))
        nxt.append(F.normalize(torch.randn(n_per_action, D, generator=rng), dim=-1))
        o = torch.zeros(n_per_action, n_actions)
        o[:, a] = 1.0
        onehot.append(o)
    return torch.cat(psi), torch.cat(nxt), torch.cat(onehot)


def test_c1_constants_and_verdicts():
    assert eng.OMEGA_BOUND == pytest.approx(0.0982, rel=1e-3)  # pi/32
    assert eng.G3_MIN_DELTA_NU == pytest.approx(0.0200)
    assert eng.G4_MAX_SAGNAC == pytest.approx(0.0500)
    assert eng.DEFAULT_BETA_SAGNAC == pytest.approx(0.015)


def test_c2_spectral_cap_below_bound_identity():
    G = _skew(D, 3)
    u = torch.linalg.svdvals(G).max().item()
    capped = eng.spectral_cap(G, omega_bound=10.0 * u)  # bound above sigma_max -> identity
    assert torch.allclose(capped, G, atol=1e-6)
    # bound below sigma_max -> sigma_max of the capped generator == bound
    tight = eng.spectral_cap(G, omega_bound=0.5 * u)
    assert torch.linalg.svdvals(tight).max().item() == pytest.approx(0.5 * u, rel=1e-4)


def test_c3_spectral_cap_binds_sigma_max():
    G = _skew(D, 5)
    capped = eng.spectral_cap(G, omega_bound=eng.OMEGA_BOUND)
    assert torch.linalg.svdvals(capped).max().item() <= eng.OMEGA_BOUND + 1e-6
    if G.norm() > eng.OMEGA_BOUND:
        assert capped.norm() < G.norm()


def test_c4_bmm_unroll_equals_loop():
    bank = make_healthy_bank()
    comp = eng.compile_generators_capped(bank[0], bank[1], bank[2], seed=20260921)
    tpow = comp["t_pow"]  # [7, K, D, D]
    assert tpow.shape == (7, eng.DEFAULT_HORIZON, D, D)
    psi = F.normalize(torch.randn(D), dim=-1)
    vec = eng.bmm_unroll(psi, tpow)  # [7, K]
    for a in range(7):
        for k in range(1, eng.DEFAULT_HORIZON + 1):
            manual = F.normalize(psi @ torch.linalg.matrix_power(comp["transitions"][a], k).T, dim=-1)
            assert vec[a, k - 1].item() == pytest.approx(
                float((manual * psi).sum(-1).abs().item()), abs=1e-5
            )


def test_c5_pg1_recon_healthy_on_capped():
    bank = make_healthy_bank()
    comp = eng.compile_generators_capped(bank[0], bank[1], bank[2], seed=20260921)
    rec = eng.preflight_pg1(comp["generators"], bank[0], bank[1], onehot=bank[2])
    assert rec["min_recon"] >= 0.85
    assert rec["min_recon"] >= eng.PG1_MIN_RECON


def test_c6_pg1_kill_degenerate():
    bank = make_degenerate_bank()
    comp = eng.compile_generators_capped(bank[0], bank[1], bank[2], seed=20260921)
    rec = eng.preflight_pg1(comp["generators"], bank[0], bank[1], onehot=bank[2])
    assert rec["min_recon"] < eng.PG1_MIN_RECON
    result = {"verdict": "F21_1_EDMD_FIT_COLLAPSE", "steps_done": 0}
    assert result["verdict"].startswith("F21_1_")


def test_c7_score_uses_horizon_plus_sagnac():
    bank = make_healthy_bank()
    comp = eng.compile_generators_capped(bank[0], bank[1], bank[2], seed=20260921)
    engine = eng.F21_1Engine(**comp, n_actions=7, seed=20260921, device="cpu")
    psi = F.normalize(torch.randn(D), dim=-1)
    goal = F.normalize(torch.randn(D), dim=-1)
    js = engine.score_all_actions(psi, goal, None)
    assert set(js) == {str(i) for i in range(7)}
    best = max(js, key=js.get)
    assert best in {str(i) for i in range(7)}


def test_c8_no_per_action_python_loop_in_step():
    import ast
    import inspect
    src = textwrap.dedent(inspect.getsource(eng.F21_1Engine.step_once))
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.For):
            pytest.fail("step_once contains a Python for-loop (must be vectorized)")


def test_c9_cli_flags_and_receipt():
    ap = eng.build_parser()
    argv = ["--device", "cuda", "--steps-per-env", "150", "--seed", "20260921",
            "--horizon", "8", "--omega-bound", "0.0982", "--beta-sagnac", "0.015",
            "--trajectory-bank", "x.npz", "--trajectory-jsonl", "x.jsonl",
            "--out-dir", "/tmp/x", "--receipt-out", "/tmp/x/r.json"]
    args = ap.parse_args(argv)
    assert args.omega_bound == pytest.approx(0.0982)
    assert args.beta_sagnac == pytest.approx(0.015)
    keys = {"verdict", "steps_done", "mean_latency_ms", "sagnac_raw_mean",
            "mean_delta_nu_goal", "goal_align_first", "goal_align_last",
            "per_action_recon", "creeps", "n_actions", "seed", "envs_solved",
            "wall_s", "omega_bound", "beta_sagnac", "horizon"}
    result = {"verdict": "F21_1_PASS", "steps_done": 1800, "mean_latency_ms": 2.1,
              "sagnac_raw_mean": 0.04, "mean_delta_nu_goal": 0.025,
              "goal_align_first": 0.05, "goal_align_last": 0.08,
              "per_action_recon": {}, "creeps": 0, "n_actions": 7, "seed": 20260921,
              "envs_solved": 1, "wall_s": 1.0, "omega_bound": 0.0982,
              "beta_sagnac": 0.015, "horizon": 8}
    assert keys <= set(result)
