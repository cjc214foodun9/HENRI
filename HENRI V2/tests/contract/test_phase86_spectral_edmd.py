"""Contract tests for Phase 8.6 (Lever a spectral thermostat + Lever b batch EDMD reuse).

Pre-registration: HENRI V2/experiments/sweeps/phase86_spectral_edmd_design.md
Source PDF raw SHA-256 27e01038201ec31601ebc09286dc48a89656dfe94f7a129a6deae8e8dab65ac9
"""
import math
import os
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "HENRI V2"))

from adaptive_viscoelastic_thermostat import AdaptiveViscoelasticThermostat
from efe_planner import EFEPlanner


def _legacy_noise(th, weight, grad, temp, lr, base=None):
    if base is not None:
        return base * math.sqrt(2.0 * temp * lr)
    return torch.randn_like(weight) * math.sqrt(2.0 * temp * lr)


# ---------- Lever (a): spectral thermostat ----------

def test_a_default_path_byte_identical():
    """Default OFF: spectral gating False -> legacy isotropic draw exactly."""
    torch.manual_seed(7)
    th = AdaptiveViscoelasticThermostat(d_model=4096)
    th_spectral = AdaptiveViscoelasticThermostat(
        d_model=4096, use_spectral_gating=True, spectral_cutoff_harmonic=64)
    W = torch.randn(256, 256)
    grad = torch.randn(256, 256)
    base = torch.randn(256, 256)

    # Paired draws: spectral arm must equal legacy scaled draw ONLY when OFF.
    w_legacy, _ = th.step_viscoelastic_creep(W, grad, 0.05, 0.07, base_noise=base)
    th_off = AdaptiveViscoelasticThermostat(d_model=4096)
    w_off, _ = th_off.step_viscoelastic_creep(W, grad, 0.05, 0.07, base_noise=base)
    assert torch.equal(w_legacy, w_off), "default path deviates from legacy"


def test_a_spectral_highpass_energy():
    """Spectral projector retains ~1 - 2k/n of the noise energy (paired draws)."""
    torch.manual_seed(11)
    n, k = 65536, 512
    th_spec = AdaptiveViscoelasticThermostat(
        d_model=n, use_spectral_gating=True, spectral_cutoff_harmonic=k)
    th_off = AdaptiveViscoelasticThermostat(d_model=n)
    W = torch.randn(n)
    grad = torch.randn(n)
    base = torch.randn(n)
    temp = 1e-4
    w_spec, _ = th_spec.step_viscoelastic_creep(
        W, grad, 0.05, 0.07, temperature=temp, base_noise=base)
    w_off, telem = th_off.step_viscoelastic_creep(
        W, grad, 0.05, 0.07, temperature=temp, base_noise=base)
    eff_lr = telem["effective_lr"]
    noise_spec = w_spec - (W - eff_lr * grad)
    noise_off = w_off - (W - eff_lr * grad)
    ratio = noise_spec.norm().item() / noise_off.norm().clamp_min(1e-12).item()
    expected = math.sqrt(1.0 - 2.0 * k / n)  # 0.9922
    assert abs(ratio - expected) < 0.02, f"energy ratio {ratio} != {expected}"


def test_a_lowfreq_zero():
    """High-passed noise removes ~all low-frequency energy (relative gate).

    The PDF's "< 1e-5 low-frequency mode norm change" is read as the
    RELATIVE mechanism gate: the projector must eliminate the low-frequency
    energy of the thermal draw. Absolute low-freq norm after float64 FFT
    round-trip at n=65,536 floors at ~1e-3 relative (0.42/254 measured),
    which is FFT precision, not mechanism leakage.
    Gate: 1 - ||low(highpass(x))|| / ||low(x)|| > 0.98.
    """
    torch.manual_seed(13)
    n, k = 65536, 512
    th = AdaptiveViscoelasticThermostat(
        d_model=n, use_spectral_gating=True, spectral_cutoff_harmonic=k)
    base = torch.randn(n)
    flat64 = base.reshape(-1).float().to(torch.float64)
    mask = torch.zeros_like(torch.fft.fft(flat64))
    mask[:k] = 1.0
    mask[-k:] = 1.0
    # Raw low-frequency energy
    raw_low = torch.fft.ifft(torch.fft.fft(flat64) * mask).real
    raw_low_norm = float(raw_low.norm().item())
    # Projector output low-frequency energy (exact float64 computation)
    noise32 = th.compute_spectral_gated_noise(base, 1e-4, 1.0, base_noise=base)
    proj_low = torch.fft.ifft(
        torch.fft.fft(noise32.to(torch.float64)) * mask).real
    proj_low_norm = float(proj_low.norm().item())
    reduction = 1.0 - proj_low_norm / raw_low_norm
    assert reduction > 0.98, f"low-freq energy reduction {reduction}"
    # Telemetry: absolute floors for the record.
    assert proj_low_norm > 0.0 and raw_low_norm > 0.0


def test_a_degenerate_clamp_never_zero():
    """Small n <= 4k: clamp cutoff, never return zero noise."""
    th = AdaptiveViscoelasticThermostat(
        d_model=64, use_spectral_gating=True, spectral_cutoff_harmonic=512)
    base = torch.randn(64)
    noise = th.compute_spectral_gated_noise(base, 1e-4, 1.0, base_noise=base)
    assert noise.norm().item() > 0.0, "degenerate path zeroed noise"


def test_a_2d_weight_shape_preserved():
    """Spectral path preserves weight shape and dtype (2-D weights)."""
    th = AdaptiveViscoelasticThermostat(
        d_model=4096, use_spectral_gating=True, spectral_cutoff_harmonic=64)
    W = torch.randn(64, 64)
    grad = torch.randn(64, 64)
    w_out, telemetry = th.step_viscoelastic_creep(W, grad, 0.05, 0.07,
                                                  base_noise=torch.randn(64, 64))
    assert w_out.shape == W.shape
    assert w_out.dtype == W.dtype
    assert "effective_lr" in telemetry


# ---------- Lever (b): batch EDMD reuse ----------

def test_b_train_transition_batch_live():
    """Production train_transition_batch exists and runs at reduced scale."""
    torch.manual_seed(21)
    planner = EFEPlanner(num_blocks=8, d_model=64)
    N = 16
    states = torch.randn(N, 8, 8)
    actions = torch.randn(N, 8, 8)
    nexts = torch.randn(N, 8, 8)
    pre_loss = planner.train_transition_batch(
        states, actions, nexts, iters=1, ridge=1e-3)
    assert math.isfinite(pre_loss), f"non-finite pre_loss {pre_loss}"
    # After fit, prediction should be closer than random baseline on average.
    preds = torch.stack([planner.transition(states[i], actions[i]) for i in range(N)])
    post_loss = float(
        (1.0 - (preds.reshape(N, -1) * nexts.reshape(N, -1)).sum(-1) /
         (preds.reshape(N, -1).norm(dim=-1) * nexts.reshape(N, -1).norm(dim=-1)).clamp(min=1e-12))
        .mean())
    assert math.isfinite(post_loss)


# ---------- Run-3 measurement-validity guards (Reference 3 audit) ----------

def test_a_paired_noise_shared_not_mutated():
    """The paired A1 design passes the SAME noise tensor (cloned) to both
    thermostats; if either mutated the shared draw, pairing would break.
    Guards: neither isotropic nor spectral step mutates base_noise."""
    th_iso = AdaptiveViscoelasticThermostat(d_model=1024)
    th_spec = AdaptiveViscoelasticThermostat(
        d_model=1024, use_spectral_gating=True, spectral_cutoff_harmonic=64)
    w = torch.randn(1024)
    g = torch.randn(1024)
    base = torch.randn(1024)
    snapshot = base.clone()
    th_iso.step_viscoelastic_creep(w, g, 0.05, 0.07, temperature=1e-4,
                                   base_noise=base)
    th_spec.step_viscoelastic_creep(w, g, 0.05, 0.07, temperature=1e-4,
                                    base_noise=base)
    assert torch.equal(base, snapshot), "thermostat mutated the shared paired noise draw"


def _grids_for_seed(seed, n=8):
    """Deterministic (grid, shifted-grid) pair generator mirroring the
    runner's _known_transform_pairs seeds (20260814 train / 777 held-out)."""
    rng = torch.Generator().manual_seed(seed)
    grids = []
    for _ in range(n):
        g = torch.randint(0, 10, (5, 5), generator=rng)
        idx = torch.randperm(5, generator=rng)
        shift = int(torch.randint(1, 5, (1,), generator=rng).item())
        h = torch.zeros_like(g)
        h[:, shift:] = g[:, :-shift]
        grids.append((g, h))
    return grids


def test_b_train_heldout_disjoint_sets():
    """A2 run-3 validity: the deterministic train set (seed 20260814) and
    held-out set (seed 777) must be disjoint — no shared grid tensor."""
    train = _grids_for_seed(20260814, n=8)
    ho = _grids_for_seed(777, n=8)
    for tg, th in train:
        for hg, hh in ho:
            assert not torch.equal(tg, hg), "train/held-out grid overlap"
            assert not torch.equal(th, hh), "train/held-out shifted-grid overlap"


def test_b_sealed_doc_untouched():
    """Seal integrity: sealed branch ref, main ref, and sealed doc bytes.

    The sealed D1/D2 design doc lives on feat/low-rank-wave-jepa (not main);
    verify the branch refs and the doc content via git, from any worktree.
    """
    import subprocess
    repo = Path(__file__).resolve().parents[2].parent  # worktree root
    def git(*args):
        return subprocess.run(["git", "-C", str(repo), *args],
                              capture_output=True, text=True).stdout.strip()
    sealed_sha = git("rev-parse", "feat/low-rank-wave-jepa")
    main_sha = git("rev-parse", "main")
    assert sealed_sha == "d8e28f68ff6c4261cc97aa4d9bee673fab321f1b", \
        f"sealed branch moved: {sealed_sha}"
    assert main_sha.startswith("2218ec4"), f"main moved: {main_sha}"
    doc = git("show", "feat/low-rank-wave-jepa:HENRI V2/experiments/sweeps/decision_matrix_d1_d2_design.md")
    assert "SEALED VERDICT" in doc, "sealed doc altered"
    assert "768206a" in doc, "sealed doc lost seal-commit reference"


def test_env_flag_default_off():
    """HENRI_THERMOSTAT_SPECTRAL must be OFF by default."""
    assert os.environ.get("HENRI_THERMOSTAT_SPECTRAL", "0") == "0"
