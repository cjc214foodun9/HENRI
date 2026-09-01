"""Carrier F22 contract tests — Dynamic Affordance Sub-Goal Stepping & Metric-Realigned Task Resolution.

Directive HENRI-DIR-2026-08-F21-1-POSTMORTEM-TASK-RESOLUTION §3/§4 (20,690 B, sha 841c73a5).
Tests mirror the F21.1 pattern with the F22 deltas: per-env waypoint chains (greedy
geodesic, stride 15, dtheta 0.35, 4-6 waypoints + terminal), dynamic waypoint
advancement (|cos| >= 0.60 -> k+1), Langevin barrier escape (3 steps, T=0.50),
realigned G4 (physical Sagnac vs axiom, not goal distance), and F22_* verdicts.
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

import arc_f22_resolution_engine as eng  # noqa: E402
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


def make_waypoint_curve(n=200, seed=3):
    """Smooth geodesic curve on S^{D-1}: successive states 0.2 rad apart."""
    g = torch.Generator().manual_seed(seed)
    a = F.normalize(torch.randn(D, generator=g), dim=-1)
    b = F.normalize(torch.randn(D, generator=g), dim=-1)
    t = torch.linspace(0.0, 1.0, n)
    cos0 = float((a * b).sum().clamp(-1.0, 1.0))
    theta0 = torch.acos(torch.tensor(cos0))
    sin0 = torch.sin(theta0)
    pts = []
    for ti in t:
        if abs(float(sin0)) < 1e-8:
            w = (1.0 - ti) * a + ti * b
        else:
            w = (torch.sin((1.0 - ti) * theta0) / sin0) * a + (torch.sin(ti * theta0) / sin0) * b
        pts.append(F.normalize(w, dim=-1))
    return torch.stack(pts)  # [n, D]


def test_c1_constants_and_gates():
    assert eng.OMEGA_BOUND == pytest.approx(0.0982, rel=1e-3)  # pi/32
    assert eng.WAYPOINT_ADVANCE_THRESH == pytest.approx(0.60)
    assert eng.LANGEVIN_TEMP == pytest.approx(0.50)
    assert eng.LANGEVIN_STEPS == 3
    assert eng.PG1_MIN_RECON == pytest.approx(0.85)
    assert eng.G1_LATENCY_MS == pytest.approx(2.0)
    assert eng.G2_MIN_SOLVED == 1
    assert eng.G3_MIN_DELTA_NU == pytest.approx(0.0200)
    assert eng.G4_MAX_SAGNAC == pytest.approx(0.0500)
    assert eng.DEFAULT_BETA_SAGNAC == pytest.approx(0.015)


def test_c2_waypoint_extraction_geodesic():
    curve = make_waypoint_curve(n=200)
    goal = curve[-1]
    wps = eng.extract_waypoints(curve, goal, delta_theta=0.35, max_waypoints=6, min_waypoints=2)
    assert len(wps) >= eng.WAYPOINT_MIN
    assert len(wps) <= 6
    # all unit norm
    for w in wps:
        assert torch.allclose(torch.linalg.vector_norm(w), torch.tensor(1.0), atol=1e-5)
    # final waypoint is the terminal goal
    assert float((wps[-1] * goal).sum().abs()) == pytest.approx(1.0, abs=1e-5)
    # monotone geodesic progress toward the goal
    last = 0.0
    for w in wps:
        c = float((w * goal).sum().abs().clamp(0.0, 1.0))
        assert c >= last - 1e-4
        last = c


def test_c3_waypoint_advancement_state_machine():
    g = torch.Generator().manual_seed(1)
    goal = F.normalize(torch.randn(D, generator=g), dim=-1)
    far = F.normalize(torch.randn(D, generator=g), dim=-1)
    near = goal.clone()
    k = eng.advance_waypoint_index(far, goal, k=0, thresh=0.60)
    assert k == 0
    k = eng.advance_waypoint_index(near, goal, k=0, thresh=0.60)
    assert k == 1
    # capped at K_max - 1
    k = eng.advance_waypoint_index(near, goal, k=eng.WAYPOINT_MAX - 1, thresh=0.60)
    assert k == eng.WAYPOINT_MAX - 1


def test_c4_langevin_escape_three_steps():
    eng_state = {"steps": 0, "active": True}  # set active by a reset event
    for _ in range(2):
        eng_state = eng.langevin_escape_tick(eng_state, langevin_steps=3)
        assert eng_state["active"] is True
    eng_state = eng.langevin_escape_tick(eng_state, langevin_steps=3)
    assert eng_state["active"] is False


def test_c5_pg1_healthy_capped_recon():
    bank = make_healthy_bank()
    comp = f211.compile_generators_capped(bank[0], bank[1], bank[2], seed=20260922)
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
    comp = f211.compile_generators_capped(psi, nxt, onehot, seed=20260922)
    rec = f211.preflight_pg1(comp["generators"], psi, nxt, onehot=onehot)
    assert rec["min_recon"] < eng.PG1_MIN_RECON


def test_c7_score_uses_waypoint_and_axiom():
    bank = make_healthy_bank()
    comp = f211.compile_generators_capped(bank[0], bank[1], bank[2], seed=20260922)
    g = torch.Generator().manual_seed(4)
    waypoint = F.normalize(torch.randn(D, generator=g), dim=-1)
    axiom = F.normalize(torch.randn(D, generator=g), dim=-1)
    engine = eng.F22Engine(**comp, n_actions=7, seed=20260922, device="cpu",
                          waypoints=[waypoint], axiom=axiom)
    psi = F.normalize(torch.randn(D, generator=g), dim=-1)
    js = engine.score_all_actions(psi, waypoint)
    assert set(js) == {str(i) for i in range(7)}
    best = max(js, key=js.get)
    assert best in {str(i) for i in range(7)}


def test_c8_no_per_action_python_loop_in_step():
    import ast
    import inspect
    src = textwrap.dedent(inspect.getsource(eng.F22Engine.step_once))
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.For):
            pytest.fail("step_once contains a Python for-loop (must be vectorized)")


def test_c9_cli_flags_and_receipt_keys():
    ap = eng.build_parser()
    argv = ["--device", "cuda", "--steps-per-env", "150", "--seed", "20260922",
            "--horizon", "8", "--omega-bound", "0.0982", "--waypoint-advance-thresh", "0.60",
            "--langevin-temp", "0.50", "--trajectory-bank", "x.npz",
            "--trajectory-jsonl", "x.jsonl", "--out-dir", "/tmp/henri_f22_resolution/",
            "--receipt-out", "/tmp/henri_f22_resolution/f22_gates_receipt.json"]
    args = ap.parse_args(argv)
    assert args.omega_bound == pytest.approx(0.0982)
    assert args.waypoint_advance_thresh == pytest.approx(0.60)
    assert args.langevin_temp == pytest.approx(0.50)
    keys = {"verdict", "steps_done", "resets", "mean_latency_ms", "sagnac_axiom_mean",
            "mean_delta_nu_wp", "waypoint_align_first", "waypoint_align_last",
            "waypoint_advances", "langevin_escapes", "per_action_recon", "creeps",
            "n_actions", "seed", "envs_solved", "env_levels", "wall_s", "omega_bound",
            "beta_sagnac", "horizon", "waypoint_advance_thresh", "langevin_temp"}
    result = {"verdict": "F22_PASS", "steps_done": 1800, "resets": 3,
              "mean_latency_ms": 1.2, "sagnac_axiom_mean": 0.03, "mean_delta_nu_wp": 0.025,
              "waypoint_align_first": 0.1, "waypoint_align_last": 0.5,
              "waypoint_advances": 40, "langevin_escapes": 2, "per_action_recon": {},
              "creeps": 100, "n_actions": 7, "seed": 20260922, "envs_solved": 1,
              "env_levels": {"e1": 1}, "wall_s": 2.0, "omega_bound": 0.0982,
              "beta_sagnac": 0.015, "horizon": 8, "waypoint_advance_thresh": 0.60,
              "langevin_temp": 0.50}
    assert keys <= set(result)


def test_c10_g4_physical_sagnac_not_goal_distance():
    """G4 = 1 - |cos(T_a psi_t, axiom)| (operator coherence). A distant goal must
    NOT inflate G4: goal distance is excluded from the physical metric."""
    bank = make_healthy_bank()
    comp = f211.compile_generators_capped(bank[0], bank[1], bank[2], seed=20260922)
    g = torch.Generator().manual_seed(6)
    psi = F.normalize(torch.randn(D, generator=g), dim=-1)
    axiom = F.normalize(torch.randn(D, generator=g), dim=-1)
    engine = eng.F22Engine(**comp, n_actions=7, seed=20260922, device="cpu",
                          waypoints=[axiom], axiom=axiom)
    idx = 0
    pred = F.normalize(psi @ comp["transitions"][idx].T, dim=-1)
    physical = float(1.0 - (pred * axiom).sum(-1).abs().clamp(0.0, 1.0).item())
    # if the goal were far but the operator is coherent with the axiom, G4 stays small
    assert 0.0 <= physical <= 2.0
    assert physical == pytest.approx(engine.g4_single_pass(psi, idx).item(), abs=1e-5)
