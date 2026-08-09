"""Phase 5 P2 — AdaptiveViscoelasticThermostat wavelet wait-for-signal tests.

Software-health evidence only (not CUDA, not task capability). Covers:
- Haar forward/inverse round-trip (orthonormal reconstruction).
- default-OFF flag preserves the legacy isotropic noise path and does not
  emit wavelet telemetry.
- dominance/lock counter mechanics: sustained strong-signal dominance
  increments the counter, lock fires at signal_lock_steps, and a locked
  thermostat silences thermal noise.
- null-band gradient (uniform energy) never locks and keeps noise.
- gating reduces injected noise on strong-signal bands vs isotropic.
"""

import math

import pytest
import torch

from adaptive_viscoelastic_thermostat import (
    AdaptiveViscoelasticThermostat,
    _haar_forward,
    _haar_inverse,
)


def _smooth_grad(n: int, seed: int = 7) -> torch.Tensor:
    """Low-frequency-structured gradient (strong coarse-band energy)."""
    g = torch.Generator().manual_seed(seed)
    t = torch.linspace(0, 1, n)
    # Smooth ramp + gentle sinusoid: energy concentrates in coarse Haar bands.
    wave = (torch.sin(2 * math.pi * t) + t * 2.0).unsqueeze(0)
    return (wave.T @ wave) + 0.05 * torch.randn(n, n, generator=g)


def _single_band_grad(n: int) -> torch.Tensor:
    """Gradient equal to a single finest-level Haar basis vector.

    Haar-forward of [+1,-1]/sqrt(2) at positions (0,1) yields exactly ONE
    nonzero coefficient in the finest detail band. 100% of gradient energy
    sits in that band, so its noise gate collapses to 1/(1+kappa) while all
    other bands keep gate 1.0 — the discriminating wait-for-signal regime.
    """
    x = torch.zeros(n, n)
    flat = x.reshape(-1)
    flat[0] = 1.0 / math.sqrt(2.0)
    flat[1] = -1.0 / math.sqrt(2.0)
    return x


class TestHaarRoundTrip:
    def test_haar_round_trip_power_of_two(self):
        x = torch.randn(256)
        coarse, detail = _haar_forward(x)
        rec = _haar_inverse(coarse, detail)
        assert torch.allclose(rec, x, atol=1e-6)

    def test_haar_orthonormal_preserves_energy(self):
        x = torch.randn(128)
        coarse, detail = _haar_forward(x)
        e_in = x.pow(2).sum()
        e_out = coarse.pow(2).sum() + sum(d.pow(2).sum() for d in detail)
        assert torch.allclose(e_in, e_out, atol=1e-4)


class TestThermostatWaveletGating:
    def test_default_off_preserves_legacy_path(self):
        """Flag OFF: wavelet keys absent, isotropic noise path unchanged."""
        th = AdaptiveViscoelasticThermostat(d_model=4096, use_wavelet_gating=False)
        W = torch.eye(64) + 0.01 * torch.randn(64, 64)
        grad = torch.randn(64, 64) * 0.5
        W_up, telem = th.step_viscoelastic_creep(
            W, grad, lambda_active=0.08, sagnac_delta=0.04, temperature=1e-4)
        assert "wavelet_dominance" not in telem
        assert "wavelet_locked" not in telem
        assert W_up.shape == W.shape
        assert torch.isfinite(W_up).all()

    def test_signal_lock_fires_within_bound(self):
        """Preregistered lock criterion: lock <= signal_lock_steps when the
        gradient is strongly band-concentrated."""
        th = AdaptiveViscoelasticThermostat(
            d_model=4096, use_wavelet_gating=True,
            signal_lock_steps=12, signal_dominance_threshold=0.5)
        W = torch.eye(64)
        grad = _smooth_grad(64)
        seen_locked = None
        for step in range(20):
            _, telem = th.step_viscoelastic_creep(
                W, grad, lambda_active=0.3, sagnac_delta=0.4, temperature=1e-3)
            if telem["wavelet_locked"]:
                seen_locked = step + 1
                break
        assert seen_locked is not None, "never locked"
        assert seen_locked <= 12

    def test_locked_thermostat_silences_noise(self):
        """Once locked, the gated noise norm collapses vs the isotropic path."""
        th_g = AdaptiveViscoelasticThermostat(
            d_model=4096, use_wavelet_gating=True,
            signal_lock_steps=2, signal_dominance_threshold=0.3)
        th_i = AdaptiveViscoelasticThermostat(
            d_model=4096, use_wavelet_gating=False)
        W = torch.eye(64)
        grad = _smooth_grad(64)
        # Drive the gated thermostat into lock.
        for _ in range(3):
            th_g.step_viscoelastic_creep(
                W, grad, lambda_active=0.3, sagnac_delta=0.4, temperature=1e-2)
        gated_noise, dom, locked = th_g.compute_wavelet_gated_noise(
            W, grad, temperature=1e-2, effective_lr=1e-3)
        assert locked is True
        iso_noise = torch.randn_like(W) * math.sqrt(2.0 * 1e-2 * 1e-3)
        assert torch.norm(gated_noise) < 0.05 * torch.norm(iso_noise)

    def test_null_band_gradient_never_locks(self):
        """White-noise gradient has flat band energy -> dominance low -> no lock."""
        th = AdaptiveViscoelasticThermostat(
            d_model=4096, use_wavelet_gating=True,
            signal_lock_steps=3, signal_dominance_threshold=0.8)
        g = torch.Generator().manual_seed(11)
        W = torch.eye(64)
        grad = torch.randn(64, 64, generator=g)
        for _ in range(6):
            _, telem = th.step_viscoelastic_creep(
                W, grad, lambda_active=0.3, sagnac_delta=0.4, temperature=1e-3)
        assert telem["wavelet_locked"] is False
        assert telem["wavelet_dominance"] < 0.8

    def test_gating_reduces_noise_on_strong_bands(self):
        """For a single-band (finest-level) gradient, the injected gated noise
        norm is robustly below isotropic: gate 1/(1+kappa) on the signal band,
        1.0 elsewhere -> expected ratio sqrt((0.04*L + L)/2L) ~= 0.72."""
        th_g = AdaptiveViscoelasticThermostat(
            d_model=4096, use_wavelet_gating=True, signal_lock_steps=999)
        W = torch.eye(64)
        grad = _single_band_grad(64)
        gated_noise, _, _ = th_g.compute_wavelet_gated_noise(
            W, grad, temperature=1e-3, effective_lr=1e-3)
        iso_noise = torch.randn_like(W) * math.sqrt(2.0 * 1e-3 * 1e-3)
        # Margin 0.8: expectation 0.72; robust against RNG draw differences.
        assert torch.norm(gated_noise) < 0.8 * torch.norm(iso_noise)

    def test_gradient_amplitude_does_not_change_noise(self):
        """Regression (P2 fix): gated noise depends only on the fresh draw
        and the gate RATIOS. Scaling the gradient amplitude leaves the
        single-band energy ratio unchanged, so the gated noise must be
        identical. The old construction (gradient coarse coefficient reused
        as noise) scaled with gradient amplitude — a deterministic leak."""
        th = AdaptiveViscoelasticThermostat(
            d_model=4096, use_wavelet_gating=True, signal_lock_steps=999)
        W = torch.eye(64)
        grad = _single_band_grad(64)
        base = torch.randn(64, 64)
        n1, _, _ = th.compute_wavelet_gated_noise(
            W, grad, temperature=1e-3, effective_lr=1e-3, base_noise=base)
        n2, _, _ = th.compute_wavelet_gated_noise(
            W, grad * 7.0, temperature=1e-3, effective_lr=1e-3, base_noise=base)
        assert torch.allclose(n1, n2, atol=1e-12)
        # The gated reconstruction genuinely differs from the raw paired draw
        # (gating modified the noise, not merely rescaled it identically).
        raw = base.reshape(-1) * math.sqrt(2.0 * 1e-3 * 1e-3)
        assert not torch.allclose(n1.reshape(-1), raw, atol=1e-3)

    def test_zero_temperature_silences_gated_noise(self):
        """T=0 must inject exactly zero noise (scale factor is zero)."""
        th = AdaptiveViscoelasticThermostat(
            d_model=4096, use_wavelet_gating=True, signal_lock_steps=999)
        W = torch.eye(64)
        grad = _single_band_grad(64)
        nz, _, _ = th.compute_wavelet_gated_noise(
            W, grad, temperature=0.0, effective_lr=1e-3)
        assert torch.norm(nz) == 0.0

    def test_gated_noise_seeded_reproducible(self):
        """Same paired draw + same gates -> identical gated noise."""
        th = AdaptiveViscoelasticThermostat(
            d_model=4096, use_wavelet_gating=True, signal_lock_steps=999)
        W = torch.eye(64)
        grad = _single_band_grad(64)
        torch.manual_seed(0)
        base1 = torch.randn(64, 64)
        torch.manual_seed(0)
        base2 = torch.randn(64, 64)
        n1, _, _ = th.compute_wavelet_gated_noise(
            W, grad, temperature=1e-3, effective_lr=1e-3, base_noise=base1)
        n2, _, _ = th.compute_wavelet_gated_noise(
            W, grad, temperature=1e-3, effective_lr=1e-3, base_noise=base2)
        assert torch.allclose(n1, n2)
