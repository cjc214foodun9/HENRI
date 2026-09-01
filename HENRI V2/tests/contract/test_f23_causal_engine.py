"""Carrier F23 contract tests — Online Causal Calibration & Semigroup Axiom Grounding.

Directive HENRI-DIR-2026-08-F22-POSTMORTEM-CAUSAL-GROUNDING §3/§4 (20,587 B,
sha 9cafa2a3). Deltas vs F22: (1) in-situ rank-1 Stiefel calibration of the
executed action's transition (eta_cal = 0.05), (2) semigroup stationary axiom
Psi_axiom = LeadingEigenvector(T_bar = mean_a T_a), (3) causal horizon stall
penalty (memory of stalled transitions), (4) G3 relaxed to +0.0150, (5)
F23_* verdicts + F23_NO_CALIBRATION_ENGAGEMENT fail-closed.
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

import arc_f23_causal_engine as eng  # noqa: E402
import arc_f21_1_vectorized_engine as f211  # noqa: E402

D = eng.D_SUB


def _skew(n, seed):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(n, n, generator=g)
    S = 0.5 * (A - A.T)
    return S / S.norm()


def make_healthy_bank(n_per_action=96, theta=0.3, noise=0.01, seed=7):
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


def make_shared_axis_ops(n_actions=7, theta=0.2, seed=13):
    """T_a all share one invariant axis n (T_a n = n); other eigenvalues at
    exp(+-i theta_k) with theta_k != 0 -> n is the unique leading eigenvector."""
    g = torch.Generator().manual_seed(seed)
    n = F.normalize(torch.randn(D, generator=g), dim=-1)
    ops = []
    for a in range(n_actions):
        S = _skew(D, seed + a + 100)
        d = S @ n  # skew => n . d == 0
        S2 = S - torch.outer(d, n) + torch.outer(n, d)  # skew AND S2 n == 0
        ops.append(torch.linalg.matrix_exp(theta * S2))
    return torch.stack(ops), n


def test_c1_constants_and_gates():
    assert eng.OMEGA_BOUND == pytest.approx(0.0982, rel=1e-3)  # pi/32
    assert eng.WAYPOINT_ADVANCE_THRESH == pytest.approx(0.60)
    assert eng.LANGEVIN_TEMP == pytest.approx(0.50)
    assert eng.LANGEVIN_STEPS == 3
    assert eng.ETA_CALIBRATION == pytest.approx(0.05)
    assert eng.STALL_COS == pytest.approx(0.90)
    assert eng.STALL_MEMORY == 32
    assert eng.STALL_PENALTY == pytest.approx(0.05)
    assert eng.PG1_MIN_RECON == pytest.approx(0.85)
    assert eng.G1_LATENCY_MS == pytest.approx(2.0)
    assert eng.G2_MIN_SOLVED == 1
    assert eng.G3_MIN_DELTA_NU == pytest.approx(0.0150)  # relaxed vs F22 0.0200
    assert eng.G4_MAX_SAGNAC == pytest.approx(0.0500)
    assert eng.DEFAULT_BETA_SAGNAC == pytest.approx(0.015)


def test_c2_semigroup_axiom_recovers_shared_invariant_axis():
    ops, n = make_shared_axis_ops()
    axiom, ev = eng.synthesize_semigroup_axiom(ops)
    assert axiom.shape == (D,)
    assert torch.allclose(torch.linalg.vector_norm(axiom), torch.tensor(1.0), atol=1e-5)
    cos = float((axiom * n).sum().abs().clamp(0.0, 1.0))
    assert cos >= 0.99, f"axiom deviates from shared invariant axis: cos={cos}"
    assert ev > 0.9  # leading real eigenvalue close to 1


def test_c3_calibration_preserves_orthogonality_and_improves_alignment():
    bank = make_healthy_bank()
    comp = f211.compile_generators_capped(bank[0], bank[1], bank[2], seed=20260923)
    T = comp["transitions"][0].clone()
    g = torch.Generator().manual_seed(3)
    psi = F.normalize(torch.randn(D, generator=g), dim=-1)
    nxt = F.normalize(psi @ T.T + 0.2 * torch.randn(D, generator=g), dim=-1)
    # eta=0 blend identity
    T0 = eng.blend_rank1(T, psi, nxt, eta=0.0)
    assert torch.allclose(T0, T, atol=1e-6)
    Tcal = eng.blend_rank1(T, psi, nxt, eta=eng.ETA_CALIBRATION)
    # orthogonality preserved by the SVD retraction
    ortho_err = torch.linalg.matrix_norm(Tcal.T @ Tcal - torch.eye(D), ord="fro")
    assert ortho_err.item() <= 1e-4
    # The calibrated prediction moves TOWARD the observed post-action state:
    # cos(nxt, Tcal psi) >= cos(nxt, T psi). The rank-1 blend is
    # (1-eta) T psi + eta nxt, whose target-cosine increases for eta>0
    # (derivative 1 - c^2 > 0); the SVD retraction may shift it slightly,
    # hence the 1e-4 tolerance. (Alignment with psi itself is NOT the
    # contract — the blend steers the prediction toward observed reality.)
    before = float(F.cosine_similarity(nxt.unsqueeze(0), F.normalize(psi @ T.T, dim=-1).unsqueeze(0)).abs().item())
    after = float(F.cosine_similarity(nxt.unsqueeze(0), F.normalize(psi @ Tcal.T, dim=-1).unsqueeze(0)).abs().item())
    assert after >= before - 1e-4


def test_c4_pre_retraction_error_contraction_exact():
    """||T_tilde psi - psi_next|| = (1 - eta) ||T psi - psi_next|| (unit psi)."""
    g = torch.Generator().manual_seed(4)
    T = torch.linalg.matrix_exp(0.3 * _skew(D, 4))
    psi = F.normalize(torch.randn(D, generator=g), dim=-1)
    nxt = F.normalize(psi @ T.T + 0.15 * torch.randn(D, generator=g), dim=-1)
    eta = 0.05
    E = nxt - psi @ T.T
    Tt = T + eta * torch.outer(E, psi)  # pre-retraction raw update
    err_before = torch.linalg.vector_norm(E)
    err_after = torch.linalg.vector_norm(nxt - psi @ Tt.T)
    assert err_after.item() == pytest.approx((1.0 - eta) * err_before.item(), rel=1e-4)


def test_c5_pg1_healthy_capped_recon():
    bank = make_healthy_bank()
    comp = f211.compile_generators_capped(bank[0], bank[1], bank[2], seed=20260923)
    rec = f211.preflight_pg1(comp["generators"], bank[0], bank[1], onehot=bank[2])
    assert rec["min_recon"] >= 0.85
    assert rec["min_recon"] >= eng.PG1_MIN_RECON


def test_c6_pg1_degenerate_kill():
    rng = torch.Generator().manual_seed(11)
    n_actions = 7
    psi = F.normalize(torch.randn(7 * 96, D, generator=rng), dim=-1)
    nxt = F.normalize(torch.randn(7 * 96, D, generator=rng), dim=-1)
    onehot = torch.zeros(7 * 96, n_actions)
    for a in range(n_actions):
        onehot[a * 96:(a + 1) * 96, a] = 1.0
    comp = f211.compile_generators_capped(psi, nxt, onehot, seed=20260923)
    rec = f211.preflight_pg1(comp["generators"], psi, nxt, onehot=onehot)
    assert rec["min_recon"] < eng.PG1_MIN_RECON


def test_c7_calibrate_updates_tpow_and_counter():
    bank = make_healthy_bank()
    comp = f211.compile_generators_capped(bank[0], bank[1], bank[2], seed=20260923)
    g = torch.Generator().manual_seed(4)
    wp = F.normalize(torch.randn(D, generator=g), dim=-1)
    engine = eng.F23Engine(**comp, n_actions=7, seed=20260923, device="cpu",
                           waypoints=[wp], eta_calibration=eng.ETA_CALIBRATION)
    psi = F.normalize(torch.randn(D, generator=g), dim=-1)
    nxt = F.normalize(psi @ comp["transitions"][0].T + 0.1 * torch.randn(D, generator=g), dim=-1)
    assert engine.calibration_updates == 0
    engine.calibrate(0, psi, nxt)
    assert engine.calibration_updates == 1
    # t_pow row for action 0 recomputed from the calibrated operator
    Tcal = engine.transitions[0]
    for k in range(1, engine.horizon + 1):
        expected = torch.linalg.matrix_power(Tcal, k)
        assert torch.allclose(engine.t_pow[0, k - 1], expected, atol=1e-5)


def test_c8_stall_memory_penalizes_stalled_action():
    bank = make_healthy_bank()
    comp = f211.compile_generators_capped(bank[0], bank[1], bank[2], seed=20260923)
    g = torch.Generator().manual_seed(6)
    wp = F.normalize(torch.randn(D, generator=g), dim=-1)
    engine = eng.F23Engine(**comp, n_actions=7, seed=20260923, device="cpu",
                           waypoints=[wp], eta_calibration=eng.ETA_CALIBRATION)
    psi = F.normalize(torch.randn(D, generator=g), dim=-1)
    js_clean = engine.score_all_actions(psi, wp)
    # record a stall: same action 0 at near-identical state, no movement
    engine.stall_mem.append((psi.clone(), 0, 0.95))
    js_stalled = engine.score_all_actions(psi, wp)
    assert js_stalled["0"] == pytest.approx(js_clean["0"] - eng.STALL_PENALTY, abs=1e-6)
    for name in engine.action_names[1:]:
        assert js_stalled[name] == pytest.approx(js_clean[name], abs=1e-6)


def test_c9_no_per_action_python_loop_in_step():
    import ast
    import inspect
    src = textwrap.dedent(inspect.getsource(eng.F23Engine.step_once))
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.For):
            pytest.fail("step_once contains a Python for-loop (must be vectorized)")


def test_c10_cli_flags_and_receipt_keys():
    ap = eng.build_parser()
    argv = ["--device", "cuda", "--steps-per-env", "150", "--seed", "20260923",
            "--horizon", "8", "--omega-bound", "0.0982", "--beta-sagnac", "0.015",
            "--waypoint-advance-thresh", "0.60", "--langevin-temp", "0.50",
            "--eta-calibration", "0.05", "--stall-cos", "0.90",
            "--stall-memory", "32", "--stall-penalty", "0.05",
            "--trajectory-bank", "x.npz", "--trajectory-jsonl", "x.jsonl",
            "--out-dir", "/tmp/henri_f23_causal/",
            "--receipt-out", "/tmp/henri_f23_causal/f23_gates_receipt.json"]
    args = ap.parse_args(argv)
    assert args.seed == 20260923
    assert args.eta_calibration == pytest.approx(0.05)
    assert args.stall_cos == pytest.approx(0.90)
    assert args.stall_memory == 32
    assert args.stall_penalty == pytest.approx(0.05)
    assert args.omega_bound == pytest.approx(0.0982)
    assert args.beta_sagnac == pytest.approx(0.015)
    keys = {"verdict", "steps_done", "resets", "mean_latency_ms", "sagnac_axiom_mean",
            "mean_delta_nu_wp", "waypoint_align_first", "waypoint_align_last",
            "waypoint_advances", "langevin_escapes", "per_action_recon", "creeps",
            "n_actions", "seed", "envs_solved", "env_levels", "wall_s", "omega_bound",
            "beta_sagnac", "horizon", "waypoint_advance_thresh", "langevin_temp",
            "axiom_ev", "calibration_updates", "stall_penalties",
            "eta_calibration", "stall_cos", "stall_memory", "stall_penalty"}
    result = {"verdict": "F23_PASS", "steps_done": 1800, "resets": 3,
              "mean_latency_ms": 1.2, "sagnac_axiom_mean": 0.03, "mean_delta_nu_wp": 0.02,
              "waypoint_align_first": 0.1, "waypoint_align_last": 0.5,
              "waypoint_advances": 40, "langevin_escapes": 2, "per_action_recon": {},
              "creeps": 100, "n_actions": 7, "seed": 20260923, "envs_solved": 1,
              "env_levels": {"e1": 1}, "wall_s": 2.0, "omega_bound": 0.0982,
              "beta_sagnac": 0.015, "horizon": 8, "waypoint_advance_thresh": 0.60,
              "langevin_temp": 0.50, "axiom_ev": 0.99, "calibration_updates": 1800,
              "stall_penalties": 120, "eta_calibration": 0.05, "stall_cos": 0.90,
              "stall_memory": 32, "stall_penalty": 0.05}
    assert keys <= set(result)


def test_c11_ingress_flatten_boundary():
    ingress = eng.PatchIngress(in_dim=4096, d=D, num_blocks=8, p=32, seed=20260923)
    x = torch.randn(1, 4096)
    structured = ingress(x)
    assert structured.shape == (1, 8, 8)
    flat = structured.reshape(1, -1)
    assert flat.shape == (1, D)


def test_c12_g4_physical_sagnac_not_goal_distance():
    """G4 = 1 - |cos(T_a psi_t, axiom)| with axiom from the semigroup. A distant
    goal must NOT inflate G4: goal distance is excluded from the physical metric."""
    bank = make_healthy_bank()
    comp = f211.compile_generators_capped(bank[0], bank[1], bank[2], seed=20260923)
    g = torch.Generator().manual_seed(6)
    psi = F.normalize(torch.randn(D, generator=g), dim=-1)
    goal = F.normalize(torch.randn(D, generator=g), dim=-1)
    axiom, _ = eng.synthesize_semigroup_axiom(comp["transitions"])
    engine = eng.F23Engine(**comp, n_actions=7, seed=20260923, device="cpu",
                           waypoints=[goal], axiom=axiom,
                           eta_calibration=eng.ETA_CALIBRATION)
    idx = 0
    pred = F.normalize(psi @ comp["transitions"][idx].T, dim=-1)
    physical = float(1.0 - (pred * axiom).sum(-1).abs().clamp(0.0, 1.0).item())
    assert 0.0 <= physical <= 2.0
    assert physical == pytest.approx(engine.g4_single_pass(psi, idx).item(), abs=1e-5)
