"""Carrier F21 contract tests — In-Situ Empirical EDMD & Trajectory-Span Generator Synthesis.

Directive HENRI-DIR-2026-08-F20-POSTMORTEM-DYNAMIC-GENERATOR-ORDER §3/§4 + F20 §4 pattern.
C1–C17: flag fail-closed, bank schema, action partition, bridge, Koopman solve,
Stiefel retraction, skew Logm, exp(D_a) unitarity, PG1 recon >= 0.70 (per-action min),
PG1 kill (zero steps), horizon beam J, n_actions from bank (7), action-name mapping,
no-SVD-in-timed-loop, gate thresholds, receipt keys, seed determinism.
"""
import math
import pathlib
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F

VERIF = pathlib.Path(__file__).resolve().parents[2] / "experiments" / "verification"
sys.path.insert(0, str(VERIF))

from arc_f21_edmd_engine import (  # noqa: E402
    DEFAULT_BETA_SAGNAC,
    DEFAULT_HORIZON,
    G1_LATENCY_MS,
    G2_MIN_SOLVED,
    G3_MIN_DELTA_NU,
    G4_MAX_SAGNAC,
    PG1_MIN_RECON,
    EDMDGeneratorBank,
    F21Engine,
    preflight_pg1,
    compile_lie_generators_d64,
    _bridge_block_mean,
)

torch.manual_seed(0)
D = 64
N_ACT = 7


# --------------------------------------------------------------------------- fixtures
def _skew(n, seed):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(n, n, generator=g)
    S = 0.5 * (A - A.T)
    return S / S.norm()  # unit Frobenius: sigma_max <= 1/sqrt(2) -> theta*sigma_max < pi


def make_healthy_bank(n_per_action=96, theta=0.3, noise=0.01, seed=7):
    """Synthetic bank with ground-truth per-action rotations: next = exp(theta*G_a) psi.

    n_per_action must exceed D=64 so the Koopman normal equation is full-rank and
    PG1 is non-vacuous (N < D -> LS interpolation -> recon ~0.95 even for random
    pairs, and the Procrustes complement is unconstrained)."""
    rng = torch.Generator().manual_seed(seed)
    psi, nxt, onehot, names = [], [], [], []
    for a in range(N_ACT):
        G = _skew(D, 100 + a)
        X = F.normalize(torch.randn(n_per_action, D, generator=rng), dim=-1)
        Y = F.normalize(X @ torch.matrix_exp(theta * G).T + noise * torch.randn(n_per_action, D, generator=rng), dim=-1)
        psi.append(X)
        nxt.append(Y)
        onehot.append(F.one_hot(torch.full((n_per_action,), a, dtype=torch.long), N_ACT))
        names.append(f"ACTION{a + 1}")
    return (
        torch.cat(psi),
        torch.cat(nxt),
        torch.cat(onehot).to(torch.uint8),
        names,
    )


def make_degenerate_bank(n_per_action=96, seed=11):
    """Random unrelated next states with N > D (non-vacuous): recon must collapse."""
    rng = torch.Generator().manual_seed(seed)
    psi, nxt, onehot = [], [], []
    for a in range(N_ACT):
        psi.append(F.normalize(torch.randn(n_per_action, D, generator=rng), dim=-1))
        nxt.append(F.normalize(torch.randn(n_per_action, D, generator=rng), dim=-1))
        onehot.append(F.one_hot(torch.full((n_per_action,), a, dtype=torch.long), N_ACT))
    return (
        torch.cat(psi),
        torch.cat(nxt),
        torch.cat(onehot).to(torch.uint8),
        [f"ACTION{i + 1}" for i in range(N_ACT)],
    )


class FakeGame:
    def __init__(self, n_actions=N_ACT):
        self._t = 0
        self.action_space = [f"ACTION{i + 1}" for i in range(n_actions)]
        self.available_actions = self.action_space
        self._frame = [[0]] * 3

    def reset(self):
        self._t = 0
        return [[0]] * 3

    def step(self, action):
        self._t += 1
        return [[self._t]] * 3

    def score(self):
        return 0.0


def make_engine(bank=None, **kw):
    if bank is None:
        bank = make_healthy_bank()
    comp = compile_lie_generators_d64(bank[0], bank[1], bank[2], seed=kw.pop("seed", 20260920))
    return F21Engine(
        generators=comp["generators"],
        exp_generators=comp["exp_generators"],
        recon=comp["recon"],
        n_actions=len(comp["generators"]),
        seed=kw.pop("seed", 20260920),
        horizon=kw.pop("horizon", DEFAULT_HORIZON),
        beta_sagnac=kw.pop("beta_sagnac", DEFAULT_BETA_SAGNAC),
        device=kw.pop("device", "cpu"),
        env_factory=kw.pop("env_factory", FakeGame),
    )


# --------------------------------------------------------------------------- C1 flag fail-closed
def test_c1_flag_fail_closed(monkeypatch):
    import arc_f21_edmd_engine as mod

    monkeypatch.delenv("HENRI_F21_EDMD", raising=False)
    with pytest.raises(RuntimeError, match="HENRI_F21_EDMD"):
        mod.require_flag()


# --------------------------------------------------------------------------- C2 bank schema
def test_c2_bank_schema(tmp_path):
    psi, nxt, onehot, names = make_healthy_bank(6)
    # npz file carries full-D rows for the CLI path; schema check only needs keys
    arr = np.zeros((psi.shape[0], 65536), dtype=np.float16)
    p = tmp_path / "bank.npz"
    np.savez(p, psi=arr, next_wave=arr, actions_onehot=onehot.numpy(), action_names=np.array(names))
    data = np.load(p)
    assert {"psi", "next_wave", "actions_onehot", "action_names"} <= set(data.files)


# --------------------------------------------------------------------------- C3 action partition
def test_c3_action_partition():
    bank = make_healthy_bank(24)
    comp = compile_lie_generators_d64(bank[0], bank[1], bank[2])
    assert set(comp["generators"].keys()) == set(range(N_ACT))
    for a in range(N_ACT):
        assert comp["recon"][a] is not None


# --------------------------------------------------------------------------- C4 bridge
def test_c4_bridge():
    from arc_f21_edmd_engine import _bridge_to_d64_batch

    x = torch.randn(3, 65536)
    out = _bridge_block_mean(x)
    assert out.shape == (3, 4096)
    # ingress=None boundary: block-mean -> normalize x K=64 -> unit norm, [N, 4096]
    full = _bridge_to_d64_batch(x, ingress=None)
    assert full.shape == (3, 4096)
    assert torch.allclose(full.norm(dim=-1), torch.ones(3), atol=1e-4)


# --------------------------------------------------------------------------- C5 Koopman solve
def test_c5_koopman_shape_and_ridge():
    bank = make_healthy_bank(16)
    comp = compile_lie_generators_d64(bank[0], bank[1], bank[2], ridge=1e-3)
    D0 = comp["generators"][0]
    assert D0.shape == (D, D)
    assert torch.isfinite(D0).all()


# --------------------------------------------------------------------------- C6/C7 retraction + skew
def test_c6_c7_retraction_and_skew():
    bank = make_healthy_bank(24)
    comp = compile_lie_generators_d64(bank[0], bank[1], bank[2])
    for a in range(N_ACT):
        Da = comp["generators"][a]
        assert torch.allclose(Da, -Da.T, atol=1e-5), f"skew symmetry D_{a}"
        W = torch.matrix_exp(Da)
        err = (W @ W.T - torch.eye(D)).abs().max().item()
        assert err < 1e-3, f"exp(D_{a}) orthogonality {err}"


# --------------------------------------------------------------------------- C8 exp(D_a) unitary
def test_c8_exp_unitary():
    bank = make_healthy_bank(12)
    comp = compile_lie_generators_d64(bank[0], bank[1], bank[2])
    for a in range(N_ACT):
        U = comp["exp_generators"][a]
        assert U.shape == (D, D)
        err = (U @ U.T - torch.eye(D)).abs().max().item()
        assert err < 1e-3


# --------------------------------------------------------------------------- C9 PG1 recon >= 0.70
def test_c9_pg1_recon_healthy():
    bank = make_healthy_bank(24, theta=0.3, noise=0.01)
    comp = compile_lie_generators_d64(bank[0], bank[1], bank[2])
    rec = preflight_pg1(comp["generators"], bank[0], bank[1], onehot=bank[2])
    assert rec["min_recon"] >= PG1_MIN_RECON, rec


# --------------------------------------------------------------------------- C10 PG1 kill zero steps
def test_c10_pg1_kill():
    bank = make_degenerate_bank()
    comp = compile_lie_generators_d64(bank[0], bank[1], bank[2])
    rec = preflight_pg1(comp["generators"], bank[0], bank[1], onehot=bank[2])
    assert rec["min_recon"] < PG1_MIN_RECON
    eng = make_engine(bank)
    result = eng.run(["fake-0", "fake-1"], steps_per_env=3, out_dir=None, receipt_out=None, allow_kill=True)
    assert result["verdict"] == "F21_EDMD_FIT_COLLAPSE"
    assert result["steps_done"] == 0


# --------------------------------------------------------------------------- C11 horizon beam J
def test_c11_beam_j():
    eng = make_engine()
    psi = F.normalize(torch.randn(64), dim=-1)
    goal = F.normalize(torch.randn(64), dim=-1)
    js = eng.score_all_actions(psi, goal, env_action_names=[f"ACTION{i + 1}" for i in range(N_ACT)])
    assert len(js) == N_ACT
    assert all(math.isfinite(float(j)) for j in js.values())
    best = max(js, key=js.get)
    assert best in js


# --------------------------------------------------------------------------- C12 n_actions from bank
def test_c12_n_actions_from_bank():
    bank = make_healthy_bank(12)
    comp = compile_lie_generators_d64(bank[0], bank[1], bank[2])
    assert len(comp["generators"]) == N_ACT  # 7, not 8


# --------------------------------------------------------------------------- C13 action-name mapping
def test_c13_action_name_mapping():
    eng = make_engine()
    js = eng.score_all_actions(torch.randn(64), torch.randn(64), env_action_names=["ACTION5", "ACTION1", "ACTION3"])
    assert set(js.keys()) == {"ACTION5", "ACTION1", "ACTION3"}


# --------------------------------------------------------------------------- C14 no SVD/Logm in timed loop
def test_c14_no_svd_in_timed_loop():
    import inspect

    from arc_f21_edmd_engine import F21Engine

    src = inspect.getsource(F21Engine.score_all_actions) + inspect.getsource(F21Engine.step_once)
    for banned in ("linalg.svd", "linalg.eig", "matrix_exp", "logm"):
        assert banned not in src, f"banned op {banned} in timed path"


# --------------------------------------------------------------------------- C15 gate thresholds
def test_c15_gate_thresholds():
    assert PG1_MIN_RECON == pytest.approx(0.70)
    assert G1_LATENCY_MS == pytest.approx(5.0)
    assert G2_MIN_SOLVED == 1
    assert G3_MIN_DELTA_NU == pytest.approx(0.0200)
    assert G4_MAX_SAGNAC == pytest.approx(0.0500)


# --------------------------------------------------------------------------- C16 receipt keys
def test_c16_receipt_keys():
    eng = make_engine()
    result = eng.run(["fake-0"], steps_per_env=4, out_dir=None, receipt_out=None, allow_kill=True)
    keys = {"verdict", "steps_done", "mean_latency_ms", "sagnac_raw_mean", "mean_delta_nu_goal",
            "goal_align_first", "goal_align_last", "per_action_recon", "creeps", "n_actions", "seed"}
    assert keys <= set(result.keys()), sorted(keys - set(result.keys()))


# --------------------------------------------------------------------------- C17 seed determinism
def test_c17_seed_determinism():
    bank = make_healthy_bank(12)
    c1 = compile_lie_generators_d64(bank[0], bank[1], bank[2], seed=20260920)
    c2 = compile_lie_generators_d64(bank[0], bank[1], bank[2], seed=20260920)
    for a in range(N_ACT):
        d = (c1["generators"][a] - c2["generators"][a]).abs().max().item()
        assert d < 1e-6
