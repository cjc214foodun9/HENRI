"""Unit tests for the Phase 8.33 non-linear macro-option Wave-JEPA module.

Reduced-dimension tests (CPU). Production D=65,536 runs on the remote CUDA
kill experiment only. Default-OFF: nothing in the production path activates
this module.
"""

import torch

from henri_nonlinear_wavejepa import (
    MacroOptionAttractorBank,
    NonLinearWaveJEPA,
    NonLinearWaveTransitionBlock,
)

D = 512
L = 64
K = 8
OPT = 16
BLOCKS = 8
BDIM = 8


def make_model(seed: int = 0):
    torch.manual_seed(seed)
    return NonLinearWaveJEPA(
        full_dim=D, compressed_dim=L, num_options=K, opt_dim=OPT,
        sagnac_lambda=0.15, device="cpu",
    )


def test_construction_frozen_compress():
    model = make_model()
    # W_compress is a FIXED Stiefel buffer: no gradient, no optimizer param.
    assert isinstance(model.W_compress, torch.Tensor)
    assert not model.W_compress.requires_grad
    # Column orthonormality (Stiefel).
    W = model.W_compress.detach()
    gram = W.t() @ W
    err = (gram - torch.eye(L)).abs().max().item()
    assert err < 1e-4, f"W_compress not orthonormal: {err}"


def test_compress_wave_real_input():
    model = make_model()
    psi = torch.randn(4, D)
    out = model.compress_wave(psi)
    assert out.shape == (4, L, 2)
    # Unit modulus per component (S^1 per component).
    mod = torch.norm(out, dim=-1)
    assert torch.allclose(mod, torch.ones_like(mod), atol=1e-5)


def test_compress_wave_complex_input():
    model = make_model()
    psi = torch.randn(3, D, 2)
    out = model.compress_wave(psi)
    assert out.shape == (3, L, 2)
    mod = torch.norm(out, dim=-1)
    assert torch.allclose(mod, torch.ones_like(mod), atol=1e-5)


def test_attractor_bank_unit_modulus():
    bank = MacroOptionAttractorBank(num_options=K, opt_dim=OPT)
    idx = torch.arange(K)
    out = bank(idx)
    assert out.shape == (K, OPT, 2)
    mod = torch.norm(out, dim=-1)
    assert torch.allclose(mod, torch.ones_like(mod), atol=1e-5)


def test_predict_next_state_shape_and_norm():
    model = make_model()
    psi = torch.randn(2, D)
    opt = torch.tensor([0, 3])
    pred = model.predict_next_state(psi, opt)
    assert pred.shape == (2, L, 2)
    mod = torch.norm(pred, dim=-1)
    assert torch.allclose(mod, torch.ones_like(mod), atol=1e-5)


def test_select_option_in_range():
    model = make_model()
    act = torch.randn(4, D)
    opt = model.select_option(act)
    assert opt.shape == (4,)
    assert opt.dtype == torch.int64
    assert (opt >= 0).all() and (opt < K).all()


def test_predict_full_wave_planner_contract():
    model = make_model()  # full_dim=512
    psi = torch.randn(2, D)
    opt = torch.tensor([1, 1])
    # num_blocks * block_dim must equal full_dim (production: 8192*8=65536).
    wave = model.predict_full_wave(psi, opt, num_blocks=64, block_dim=8)
    assert wave.shape == (2, 64, 8)
    assert wave.dtype == torch.float32
    norms = torch.norm(wave, p=2, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_forward_loss_metrics():
    model = make_model()
    psi = torch.randn(4, D)
    target = torch.randn(4, D)
    opt = torch.tensor([0, 1, 2, 3])
    out = model(psi, opt, target)
    for key in ("loss", "jepa_loss", "sagnac_stress", "psi_pred"):
        assert key in out
    assert torch.isfinite(torch.tensor(out["loss"]))
    assert 0.0 <= out["sagnac_stress"] <= 1.0 + 1e-6
    assert out["jepa_loss"] >= 0.0


def test_predict_full_wave_single_wave():
    """1D per-record input (kill-runner held-out path) must work."""
    model = make_model()
    psi = torch.randn(D)  # [D], no batch dim
    opt = torch.tensor([2])
    wave = model.predict_full_wave(psi, opt, num_blocks=64, block_dim=8)
    assert wave.shape == (1, 64, 8)
    norms = torch.norm(wave, p=2, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_sagnac_identical_waves_zero():
    model = make_model()
    psi = torch.randn(2, L, 2)
    d = model.compute_sagnac_delta(psi, psi)
    assert d.abs().max().item() < 1e-4


def test_engagement_training_reduces_loss():
    """Cheap kill: with a fixed option and paired targets, the transition
    core must reduce the JEPA loss below its untrained value."""
    torch.manual_seed(7)
    model = make_model()
    opt = torch.zeros(8, dtype=torch.long)
    psi = torch.randn(8, D)
    target = psi + 0.05 * torch.randn(8, D)
    # Only core + codebook params train (W_compress is a frozen buffer).
    params = [p for p in model.parameters() if p.requires_grad]
    assert len(params) >= 2
    optm = torch.optim.Adam(params, lr=1e-3)
    before = model(psi, opt, target)["loss"]
    for _ in range(120):
        optm.zero_grad()
        loss = model(psi, opt, target)["loss"]
        loss.backward()
        optm.step()
    after = model(psi, opt, target)["loss"]
    assert after < before, f"no training engagement: {before:.4f} -> {after:.4f}"
    assert torch.isfinite(torch.tensor(after))
