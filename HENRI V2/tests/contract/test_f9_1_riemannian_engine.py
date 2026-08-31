"""Contract tests for Carrier F9.1 Riemannian optimization engine.

Covers: default-OFF guard, loader validation (real schema), grouped 4-fold
env-level split, unconstrained adapter forward (LayerNorm + residual + L2
sphere: output unit norm), skew-symmetric D_a, egress prototype row norms,
gradient clipping bound, cosine-anneal + warmup schedule sanity, joint
loss gradient flow into all modules, and end-to-end receipt schema.
Fixtures use small D and the REAL bank schema.
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

from arc_f9_1_riemannian_engine import (  # noqa: E402
    RiemannianPolicyEngine,
    build_receipt,
    cosine_anneal_lr,
    grouped_env_folds,
    require_f9_1_enabled,
    run_gauntlet,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def _make_bank(tmp_path: Path, seed: int = 7, n_envs: int = 4, rows_per_env: int = 12):
    rng = np.random.default_rng(seed)
    D = 512
    env_names = [f"e{i}" for i in range(n_envs)]
    psi, nxt, onehot, meta = [], [], [], []
    for e, name in enumerate(env_names):
        for s in range(rows_per_env):
            x = rng.normal(size=D).astype(np.float32)
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
    monkeypatch.delenv("HENRI_F9_1_ACTIVE", raising=False)
    with pytest.raises(RuntimeError, match="HENRI_F9_1_ACTIVE"):
        require_f9_1_enabled()
    monkeypatch.setenv("HENRI_F9_1_ACTIVE", "1")
    require_f9_1_enabled()  # no raise


# ---------------------------------------------------------------------------
# C2 — loader rejects wrong schema (complex psi)
# ---------------------------------------------------------------------------
def test_c2_loader_rejects_wrong_schema(tmp_path):
    npz, jl = _make_bank(tmp_path)
    from arc_f9_1_riemannian_engine import load_bank
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
    from arc_f9_1_riemannian_engine import load_bank
    data = load_bank(npz, jl)
    folds = grouped_env_folds(data["envs"], n_folds=4, seed=20260907)
    assert len(folds) == 4
    n_envs = len(set(data["envs"]))
    assert n_envs % 4 == 0
    all_rows = set(range(len(data["envs"])))
    covered = set()
    for tr, te in folds:
        tr_envs = {data["envs"][i] for i in tr}
        te_envs = {data["envs"][i] for i in te}
        assert tr_envs & te_envs == set()
        assert len(te_envs) == n_envs // 4
        covered |= set(te)
        for i in te:
            assert data["envs"][i] not in tr_envs
    assert covered == all_rows


# ---------------------------------------------------------------------------
# C4 — forward produces unit-norm wave on S^(D-1)
# ---------------------------------------------------------------------------
def test_c4_forward_unit_norm(monkeypatch):
    monkeypatch.setenv("HENRI_F9_1_ACTIVE", "1")
    eng = RiemannianPolicyEngine(d=512, r=64, n_actions=7, seed=20260907)
    X = torch.randn(16, 512)
    logits, z, psi = eng.forward_logits(X)
    assert logits.shape == (16, 7)
    assert z.shape == (16, 7)
    assert psi.shape == (16, 512)
    norms = psi.norm(dim=1)
    assert torch.allclose(norms, torch.ones(16), atol=1e-5)
    assert torch.allclose(z.sum(dim=1), torch.ones(16), atol=1e-5)


# ---------------------------------------------------------------------------
# C5 — skew-symmetric D_a and unit egress prototypes
# ---------------------------------------------------------------------------
def test_c5_skew_and_prototypes(monkeypatch):
    monkeypatch.setenv("HENRI_F9_1_ACTIVE", "1")
    eng = RiemannianPolicyEngine(d=512, r=64, n_actions=7, seed=1)
    for a in range(7):
        Da = eng.D_a[a]
        assert torch.allclose(Da + Da.transpose(-1, -2), torch.zeros_like(Da), atol=1e-6)
    eng.normalize_prototypes()
    norms = eng.M.detach().norm(dim=1)
    assert torch.allclose(norms, torch.ones(7), atol=1e-4)


# ---------------------------------------------------------------------------
# C6 — cosine anneal + warmup schedule sanity
# ---------------------------------------------------------------------------
def test_c6_lr_schedule():
    lrs = [cosine_anneal_lr(ep, 40, lr_max=0.01, lr_min=1e-5, warmup=2) for ep in range(40)]
    # warmup: rises from small to lr_max at end of warmup
    assert lrs[0] < lrs[1] < lrs[2] or lrs[0] <= lrs[1]  # non-decreasing during warmup
    assert max(lrs) == pytest.approx(0.01, abs=1e-4)
    assert lrs[39] == pytest.approx(1e-5, rel=0.1)
    # anneal decays after warmup (mostly decreasing tail)
    assert lrs[35] > lrs[39]


# ---------------------------------------------------------------------------
# C7 — joint loss descends and gradients reach all modules, clipped
# ---------------------------------------------------------------------------
def test_c7_gradient_flow_and_clip(monkeypatch):
    monkeypatch.setenv("HENRI_F9_1_ACTIVE", "1")
    eng = RiemannianPolicyEngine(d=512, r=64, n_actions=7, seed=4)
    X = torch.randn(32, 512)
    y = torch.randint(0, 7, (32,))
    Xn = torch.randn(32, 512)
    loss0 = eng.composite_loss(X, y, Xn).item()
    loss = eng.composite_loss(X, y, Xn)
    loss.backward()
    assert eng.W_down.grad is not None and eng.W_down.grad.abs().sum() > 0
    assert eng.W_up.grad is not None and eng.W_up.grad.abs().sum() > 0
    assert eng.D_a.grad is not None and eng.D_a.grad.abs().sum() > 0
    assert eng.M.grad is not None and eng.M.grad.abs().sum() > 0
    # clip bound
    torch.nn.utils.clip_grad_norm_(eng.parameters(), 1.0)
    tot = 0.0
    for p in eng.parameters():
        if p.grad is not None:
            tot += p.grad.norm().item() ** 2
    assert tot ** 0.5 <= 1.0 + 1e-4
    opt = torch.optim.AdamW(eng.parameters(), lr=1e-2)
    for _ in range(5):
        opt.zero_grad()
        l = eng.composite_loss(X, y, Xn)
        l.backward()
        torch.nn.utils.clip_grad_norm_(eng.parameters(), 1.0)
        opt.step()
        eng.post_step()
    loss1 = eng.composite_loss(X, y, Xn).item()
    assert loss1 < loss0


# ---------------------------------------------------------------------------
# C8 — end-to-end gauntlet receipt on synthetic bank
# ---------------------------------------------------------------------------
def test_c8_receipt_schema(tmp_path, monkeypatch):
    monkeypatch.setenv("HENRI_F9_1_ACTIVE", "1")
    npz, jl = _make_bank(tmp_path, n_envs=4, rows_per_env=12)
    receipt = run_gauntlet(
        bank_npz=npz, bank_jsonl=jl, device="cpu", n_folds=4,
        seed=20260907, epochs=3, git_sha="test-sha", out_dir=None,
        r=64,
    )
    for key in ("schema", "git_sha", "n_valid", "folds", "macro_p1",
                "loss_ce_train", "p1_train", "min_fold_p1", "gates", "verdict"):
        assert key in receipt, f"missing {key}"
    assert receipt["schema"] == "f9-1-riemannian.v1"
    assert receipt["verdict"] in (
        "F9_1_RIEMANNIAN_VERIFIED",
        "F9_1_OPTIMIZATION_FAILED",
        "F9_1_ACTIVE_NO_GAIN",
    )
