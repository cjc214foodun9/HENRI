"""Contract tests for Carrier F10 live interactive world-model engine.

Covers: default-OFF guard, patch-ingress shapes/norms, single-pass K=8 horizon,
Sagnac delta bounds + veto, synthetic in-sample descent (the K2 pre-flight that
would have caught both F9/F9.1 K1s), and receipt schema.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "HENRI V2" / "experiments" / "verification"))

from arc_f10_live_engine import (
    PatchIngress,
    SinglePassHorizon,
    efe_select,
    run_gauntlet,
    sagnac_delta,
    veto,
)

NUM_BLOCKS = 8
D = 64  # reduced scale for contracts; production D=65536 on CUDA
RANK = 8
P = 32
K = 8


def _synthetic_stream(n=256, n_classes=4, seed=7, dirs=None, scale=3.0, dim=64):
    """Class-conditional separable observation stream (raw grid patches).

    Class label indexes the class direction applied (labels aligned with the
    shift — the earlier fixture drew labels after the shift, mislabeling it).
    Shared `dirs` across train/test splits: the test measures in-sample
    learnability (fresh noise, same class directions), which is the K2
    pre-flight intent.
    """
    g = torch.Generator().manual_seed(seed)
    if dirs is None:
        dirs = torch.nn.functional.normalize(
            torch.randn(n_classes, dim, generator=torch.Generator().manual_seed(99)), dim=-1)
    y = torch.randint(0, n_classes, (n,), generator=g)
    noise = torch.randn(n, dim, generator=g)
    X = noise + dirs[y] * scale
    return X, y, dirs


def test_c1_default_off_fail_closed(tmp_path):
    with pytest.raises(RuntimeError, match="HENRI_F10_LIVE"):
        run_gauntlet(
            env_names=["fake"], steps=2, seed=1, out_dir=str(tmp_path),
            receipt_out=str(tmp_path / "r.json"), _force_enabled=False,
        )


def test_c2_patch_ingress_shapes_and_norms():
    m = PatchIngress(in_dim=64, d=16, num_blocks=NUM_BLOCKS, seed=3)
    X = torch.randn(4, 64)
    psi = m(X)  # [B, num_blocks, 8]
    assert psi.shape == (4, NUM_BLOCKS, 8)
    assert torch.isfinite(psi).all()
    norms = torch.linalg.vector_norm(psi, dim=-1)  # [B, num_blocks]
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)


def test_c3_patch_semantics_contiguous():
    m = PatchIngress(in_dim=64, d=16, num_blocks=NUM_BLOCKS, seed=3)
    X = torch.zeros(1, 64)
    X[0, 0:32] = 1.0  # first patch hot
    Y = torch.zeros(1, 64)
    Y[0, 32:64] = 1.0  # second patch hot
    a = m(X).detach()
    b = m(Y).detach()
    # different patch occupancy must change the wave beyond noise
    assert torch.norm(a - b) > 1e-3


def test_c4_single_pass_horizon_shapes():
    h = SinglePassHorizon(d=D, rank=RANK, K=K, num_blocks=NUM_BLOCKS, seed=5)
    psi = torch.randn(2, NUM_BLOCKS, 8)
    psi = psi / torch.linalg.vector_norm(psi, dim=-1, keepdim=True)
    out = h(psi)  # [B, K, num_blocks, 8]
    assert out.shape == (2, K, NUM_BLOCKS, 8)
    assert torch.isfinite(out).all()


def test_c5_sagnac_delta_bounds():
    a = torch.randn(64)
    b = torch.randn(64)
    d_same = sagnac_delta(a, a)
    assert abs(d_same) < 1e-5
    d_orth = sagnac_delta(a, -a)
    assert abs(d_orth - 2.0) < 1e-3
    d = sagnac_delta(a, b)
    assert 0.0 <= d <= 2.0 + 1e-5


def test_c6_veto_threshold():
    # veto fires strictly above tau=0.35
    assert veto(0.36) is True
    assert veto(0.35) is False
    assert veto(0.1) is False


def test_c7_synthetic_in_sample_descent():
    """K2 pre-flight: separable stream must descend in-sample.

    If this fails, a live K1 is guaranteed for the same reason — the engine
    cannot learn any in-sample signal. Fixture: shared class directions
    across train/test, labels aligned with the shift (verified recipe:
    in_dim 128, d 64, scale 3.0, Adam 1e-2, 30 epochs -> CE 0.0104, P@1 0.89).
    """
    m = PatchIngress(in_dim=128, d=64, num_blocks=NUM_BLOCKS, p=P, seed=11)
    head = torch.nn.Linear(64, 4)
    opt = torch.optim.Adam(list(m.parameters()) + list(head.parameters()), lr=1e-2)
    lossf = torch.nn.CrossEntropyLoss()

    X, y, dirs = _synthetic_stream(n=512, n_classes=4, scale=3.0, dim=128)
    Xt, yt, _ = _synthetic_stream(n=128, n_classes=4, seed=13, dirs=dirs, scale=3.0, dim=128)

    for epoch in range(30):
        opt.zero_grad()
        psi = m(X).reshape(X.shape[0], -1)
        logits = head(psi)
        loss = lossf(logits, y)
        loss.backward()
        opt.step()

    with torch.no_grad():
        psi_t = m(Xt).reshape(Xt.shape[0], -1)
        logits_t = head(psi_t)
        p1 = (logits_t.argmax(dim=1) == yt).float().mean().item()

    assert loss.item() < 0.5, f"CE did not descend: {loss.item():.4f}"
    assert p1 >= 0.85, f"P@1 did not reach 0.85: {p1:.4f}"


def test_c8_receipt_schema(tmp_path):
    m = PatchIngress(in_dim=64, d=16, num_blocks=NUM_BLOCKS, seed=3)
    h = SinglePassHorizon(d=D, rank=RANK, K=K, num_blocks=NUM_BLOCKS, seed=5)
    X = torch.randn(2, 64)
    psi = m(X)
    roll = h(psi)
    sel = efe_select(roll, torch.zeros_like(psi[0]))
    assert sel.shape == (2,)

    # receipt writer must exist with expected keys
    from arc_f10_live_engine import write_receipt
    path = tmp_path / "r.json"
    write_receipt(
        path,
        gates={"G1": True, "G2": False, "G3": True, "G4": True},
        telemetry={"steps": 10, "solved": 0},
        meta={"seed": 20260908, "K": K, "p": P},
    )
    data = json.loads(path.read_text())
    for k in ("schema", "gates", "telemetry", "verdict", "created_utc"):
        assert k in data, f"missing {k}"
    assert data["schema"] == "f10-live-engine.v1"
