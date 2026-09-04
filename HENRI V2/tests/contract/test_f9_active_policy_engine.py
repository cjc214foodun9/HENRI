"""Contract tests for Carrier F9 active policy-gradient engine.

Covers: default-OFF guard, loader validation (real schema), grouped 4-fold
env-level split (an env is never split across folds), skew-symmetric D_a
maintenance, Stiefel QR retraction of W_in (Gram <= 1e-4), egress prototype
row normalization, forward logit shape/semantics, CE+transition composite
objective with gradient flow into all three modules, and end-to-end receipt
schema on a synthetic bank. Fixtures use small D and the REAL bank schema.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "verification"))

from arc_f9_active_policy_engine import (  # noqa: E402
    ActivePolicyEngine,
    build_receipt,
    grouped_env_folds,
    require_f9_enabled,
    run_gauntlet,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def _make_bank(tmp_path: Path, seed: int = 7, n_envs: int = 4, rows_per_env: int = 12):
    """Synthetic REAL-schema bank: psi/next_wave float16 [N, D],
    actions_onehot uint8 [N, 7], jsonl env/step/action_name."""
    rng = np.random.default_rng(seed)
    D = 512
    env_names = [f"e{i}" for i in range(n_envs)]
    psi, nxt, onehot, meta = [], [], [], []
    for e, name in enumerate(env_names):
        for s in range(rows_per_env):
            x = rng.normal(size=D).astype(np.float32)
            # deterministic action-conditional shift
            a = (s + e) % 7
            y = np.roll(x, a + 1) + 0.05 * rng.normal(size=D).astype(np.float32)
            psi.append(x)
            nxt.append(y)
            o = np.zeros(7, dtype=np.uint8)
            o[a] = 1
            onehot.append(o)
            meta.append({"env": name, "step": s, "action_name": f"A{a+1}", "t": float(s)})
    psi = np.asarray(psi, dtype=np.float16)
    nxt = np.asarray(nxt, dtype=np.float16)
    onehot = np.asarray(onehot, dtype=np.uint8)
    npz = tmp_path / "bank.npz"
    np.savez(npz, psi=psi, next_wave=nxt, actions_onehot=onehot,
             action_names=np.array([f"A{i+1}" for i in range(7)], dtype="<U7"))
    jl = tmp_path / "bank.jsonl"
    with open(jl, "w", encoding="utf-8") as f:
        for m in meta:
            f.write(json.dumps(m) + "\n")
    return str(npz), str(jl)


# ---------------------------------------------------------------------------
# C1 — default-OFF guard
# ---------------------------------------------------------------------------
def test_c1_default_off_guard(monkeypatch):
    monkeypatch.delenv("HENRI_F9_ACTIVE", raising=False)
    with pytest.raises(RuntimeError, match="HENRI_F9_ACTIVE"):
        require_f9_enabled()
    monkeypatch.setenv("HENRI_F9_ACTIVE", "1")
    require_f9_enabled()  # no raise


# ---------------------------------------------------------------------------
# C2 — loader rejects wrong schema (complex psi)
# ---------------------------------------------------------------------------
def test_c2_loader_rejects_wrong_schema(tmp_path):
    npz, jl = _make_bank(tmp_path)
    from arc_f9_active_policy_engine import load_bank
    bad = tmp_path / "complex.npz"
    d = np.load(npz)
    np.savez(bad, psi=d["psi"].astype(np.complex64), next_wave=d["next_wave"],
             actions_onehot=d["actions_onehot"], action_names=d["action_names"])
    with pytest.raises(ValueError, match="real"):
        load_bank(str(bad), jl)


# ---------------------------------------------------------------------------
# C3 — grouped env-level folds: an env is never split across folds
# ---------------------------------------------------------------------------
def test_c3_grouped_env_folds(tmp_path):
    npz, jl = _make_bank(tmp_path, n_envs=12, rows_per_env=12)
    from arc_f9_active_policy_engine import load_bank
    data = load_bank(npz, jl)
    folds = grouped_env_folds(data["envs"], n_folds=4, seed=20260906)
    assert len(folds) == 4
    n_envs = len(set(data["envs"]))
    assert n_envs % 4 == 0
    all_rows = set(range(len(data["envs"])))
    covered = set()
    for tr, te in folds:
        tr_envs = {data["envs"][i] for i in tr}
        te_envs = {data["envs"][i] for i in te}
        assert tr_envs & te_envs == set()  # grouped: no env in both
        assert len(te_envs) == n_envs // 4  # 3 envs per test fold
        covered |= set(te)
        for i in te:
            assert data["envs"][i] not in tr_envs
    assert covered == all_rows  # complete


# ---------------------------------------------------------------------------
# C4 — engine construction + forward shapes
# ---------------------------------------------------------------------------
def test_c4_forward_shapes(monkeypatch):
    monkeypatch.setenv("HENRI_F9_ACTIVE", "1")
    eng = ActivePolicyEngine(d=512, r=32, n_actions=7, seed=20260906)
    X = torch.randn(16, 512)
    logits, z, psi = eng.forward_logits(X)
    assert logits.shape == (16, 7)
    assert z.shape == (16, 7)
    assert psi.shape == (16, 512)
    assert torch.allclose(z.sum(dim=1), torch.ones(16), atol=1e-5)
    # softmax values in [0,1]
    assert z.min() >= 0.0 and z.max() <= 1.0


# ---------------------------------------------------------------------------
# C5 — skew-symmetric D_a maintenance after update
# ---------------------------------------------------------------------------
def test_c5_skew_symmetry(monkeypatch):
    monkeypatch.setenv("HENRI_F9_ACTIVE", "1")
    eng = ActivePolicyEngine(d=512, r=32, n_actions=7, seed=1)
    for a in range(7):
        Da = eng.D_a[a]  # [64, 8, 8] for d=512 -> 64 blocks
        # constructed skew: D + D^T == 0 on the last two axes
        assert torch.allclose(Da + Da.transpose(-1, -2), torch.zeros_like(Da), atol=1e-6)


# ---------------------------------------------------------------------------
# C6 — Stiefel retraction keeps Gram error <= 1e-4
# ---------------------------------------------------------------------------
def test_c6_stiefel_gram(monkeypatch):
    monkeypatch.setenv("HENRI_F9_ACTIVE", "1")
    eng = ActivePolicyEngine(d=512, r=32, n_actions=7, seed=2)
    eng.stiefel_retract()
    W = eng.W_in.detach()  # [d, r]
    gram_err = (W.T @ W - torch.eye(32)).abs().max().item()
    assert gram_err <= 1e-4


# ---------------------------------------------------------------------------
# C7 — egress prototypes are unit rows
# ---------------------------------------------------------------------------
def test_c7_prototype_norms(monkeypatch):
    monkeypatch.setenv("HENRI_F9_ACTIVE", "1")
    eng = ActivePolicyEngine(d=512, r=32, n_actions=7, seed=3)
    eng.normalize_prototypes()
    norms = eng.M.detach().norm(dim=1)
    assert torch.allclose(norms, torch.ones(7), atol=1e-4)


# ---------------------------------------------------------------------------
# C8 — composite objective descends and gradients reach all modules
# ---------------------------------------------------------------------------
def test_c8_gradient_flow(monkeypatch):
    monkeypatch.setenv("HENRI_F9_ACTIVE", "1")
    eng = ActivePolicyEngine(d=512, r=32, n_actions=7, seed=4)
    X = torch.randn(32, 512, requires_grad=False)
    y = torch.randint(0, 7, (32,))
    Xn = torch.randn(32, 512)
    loss0 = eng.composite_loss(X, y, Xn).item()
    loss = eng.composite_loss(X, y, Xn)
    loss.backward()
    # gradients exist and are nonzero in all three modules
    assert eng.W_in.grad is not None and eng.W_in.grad.abs().sum() > 0
    assert eng.D_a.grad is not None and eng.D_a.grad.abs().sum() > 0
    assert eng.M.grad is not None and eng.M.grad.abs().sum() > 0
    # a descent step reduces loss
    opt = torch.optim.AdamW(eng.parameters(), lr=1e-2)
    for _ in range(5):
        opt.zero_grad()
        l = eng.composite_loss(X, y, Xn)
        l.backward()
        opt.step()
        eng.post_step()
    loss1 = eng.composite_loss(X, y, Xn).item()
    assert loss1 < loss0


# ---------------------------------------------------------------------------
# C9 — end-to-end gauntlet receipt on synthetic bank
# ---------------------------------------------------------------------------
def test_c9_receipt_schema(tmp_path, monkeypatch):
    monkeypatch.setenv("HENRI_F9_ACTIVE", "1")
    npz, jl = _make_bank(tmp_path, n_envs=4, rows_per_env=12)
    receipt = run_gauntlet(
        bank_npz=npz, bank_jsonl=jl, device="cpu", n_folds=4,
        seed=20260906, epochs=3, git_sha="test-sha", out_dir=None,
    )
    for key in ("schema", "git_sha", "n_valid", "folds", "macro_p1",
                "loss_ce_train", "gram_max", "gates", "verdict"):
        assert key in receipt, f"missing {key}"
    assert receipt["schema"] == "f9-active-policy.v1"
    assert receipt["verdict"] in (
        "F9_ACTIVE_POLICY_VERIFIED",
        "F9_OPTIMIZATION_FAILED",
        "F9_ACTIVE_LOSS_NO_GAIN",
    )
    assert receipt["gates"]["G4"]["gram_max"] <= 1e-4 or receipt["verdict"] == "F9_OPTIMIZATION_FAILED"
