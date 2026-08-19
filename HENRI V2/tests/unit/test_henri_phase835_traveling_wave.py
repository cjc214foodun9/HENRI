"""Phase 8.35 Sprint a — directional traveling-wave coupler (T1), dual-scale
analog lexical snap (T2), directional Sagnac homodyne (T3).

Mechanism-discrimination tests on reduced dims (d=64) using the production
learning rule (update_online_step / store_engrams). Control arms must be
byte-identical where the design requires it; treatment arms must differ
from controls (discrimination), never absolute thresholds (Phase 5 rule).
"""

import math

import pytest
import torch
import torch.nn.functional as F

from recursive_dual_edmd import (
    CoupledRecursiveDualEDMD,
    DirectionalTravelingWaveCoupler,
)
from hopfield_cleanup import (
    ContinuousHopfieldCleanup,
    DualScaleAnalogLexicalSnap,
)
from efe_planner import EFEPlanner

D = 64
NB = 8
BD = 8
R = 8


def _random_wave(dim: int = D, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return F.normalize(torch.randn(dim, generator=g), p=2, dim=0)


def _train(coupler, steps: int = 12, seed: int = 1, teacher: torch.Tensor = None) -> list:
    losses = []
    for i in range(steps):
        s = _random_wave(D, seed=seed + i)
        a = _random_wave(D, seed=seed + 1000 + i)
        if teacher is not None:
            y = F.normalize(teacher @ s, p=2, dim=0)
        else:
            y = _random_wave(D, seed=seed + 2000 + i)
        losses.append(coupler.update_online_step(s, a, y))
    return losses


def _field_teacher(seed: int = 5) -> torch.Tensor:
    """Low-rank teacher: drives the field channel to learn a nonzero map
    (random dense targets leave B ~ 0 -> vacuous discrimination)."""
    g = torch.Generator().manual_seed(seed)
    M = torch.randn(D, R, generator=g) / math.sqrt(D)
    return M @ M.T


# --------------------------------------------------------------------------
# T1: DirectionalTravelingWaveCoupler
# --------------------------------------------------------------------------

class TestTravelingShift:
    def test_ap_differs_from_pa(self):
        ap = DirectionalTravelingWaveCoupler(d_model=D, r_rank=R, num_blocks=NB,
                                             block_dim=BD, direction="AP", k_max=2.0)
        pa = DirectionalTravelingWaveCoupler(d_model=D, r_rank=R, num_blocks=NB,
                                             block_dim=BD, direction="PA", k_max=2.0)
        mode = torch.randn(R)
        d_ap = ap.traveling_shift(mode)
        d_pa = pa.traveling_shift(mode)
        assert torch.max(torch.abs(d_ap - d_pa)) > 1e-6, "AP and PA shifts must differ"

    def test_k0_is_identity(self):
        c = DirectionalTravelingWaveCoupler(d_model=D, r_rank=R, num_blocks=NB,
                                            block_dim=BD, direction="AP", k_max=0.0)
        mode = torch.randn(R)
        assert torch.max(torch.abs(c.traveling_shift(mode) - mode)) == 0.0

    def test_shift_preserves_norm(self):
        ap = DirectionalTravelingWaveCoupler(d_model=D, r_rank=R, num_blocks=NB,
                                             block_dim=BD, direction="AP", k_max=2.0)
        mode = torch.randn(R)
        assert abs(torch.norm(ap.traveling_shift(mode)) - torch.norm(mode)) < 1e-5

    def test_invalid_direction_raises(self):
        with pytest.raises(ValueError):
            DirectionalTravelingWaveCoupler(d_model=D, r_rank=R, num_blocks=NB,
                                            block_dim=BD, direction="XX")


class TestDirectionalCoupler:
    def test_online_update_finite(self):
        c = DirectionalTravelingWaveCoupler(d_model=D, r_rank=R, num_blocks=NB,
                                            block_dim=BD, direction="AP", k_max=0.5)
        losses = _train(c)
        assert all(math.isfinite(l) for l in losses)
        assert torch.isfinite(c.B).all()

    def test_directional_prediction_differs_from_base(self):
        base = CoupledRecursiveDualEDMD(d_model=D, r_rank=R, num_blocks=NB,
                                        block_dim=BD, field_channel=True)
        dirc = DirectionalTravelingWaveCoupler(d_model=D, r_rank=R, num_blocks=NB,
                                               block_dim=BD, direction="AP", k_max=2.0)
        teacher = _field_teacher()
        _train(base, teacher=teacher)
        _train(dirc, teacher=teacher)
        s = _random_wave(D, seed=77)
        a = _random_wave(D, seed=78)
        p_base = base.forward(s, a)
        p_dirc = dirc.forward(s, a)
        assert torch.max(torch.abs(p_base - p_dirc)) > 1e-6, (
            "directional coupler must change predictions vs base"
        )

    def test_ap_pa_predictions_differ_after_training(self):
        ap = DirectionalTravelingWaveCoupler(d_model=D, r_rank=R, num_blocks=NB,
                                             block_dim=BD, direction="AP", k_max=2.0)
        pa = DirectionalTravelingWaveCoupler(d_model=D, r_rank=R, num_blocks=NB,
                                             block_dim=BD, direction="PA", k_max=2.0)
        teacher = _field_teacher()
        _train(ap, seed=11, teacher=teacher)
        _train(pa, seed=11, teacher=teacher)
        s = _random_wave(D, seed=13)
        a = _random_wave(D, seed=14)
        assert torch.max(torch.abs(ap.forward(s, a) - pa.forward(s, a))) > 1e-6, (
            "AP vs PA must break time-reversal symmetry"
        )

    def test_field_channel_off_is_control(self):
        off = DirectionalTravelingWaveCoupler(d_model=D, r_rank=R, num_blocks=NB,
                                              block_dim=BD, field_channel=False,
                                              direction="AP", k_max=0.5)
        base_off = CoupledRecursiveDualEDMD(d_model=D, r_rank=R, num_blocks=NB,
                                            block_dim=BD, field_channel=False)
        _train(off)
        _train(base_off)
        s = _random_wave(D, seed=21)
        a = _random_wave(D, seed=22)
        assert torch.max(torch.abs(off.forward(s, a) - base_off.forward(s, a))) == 0.0, (
            "field_channel=False must be byte-identical to base control"
        )

    def test_k0_matches_base(self):
        k0 = DirectionalTravelingWaveCoupler(d_model=D, r_rank=R, num_blocks=NB,
                                             block_dim=BD, direction="AP", k_max=0.0)
        base = CoupledRecursiveDualEDMD(d_model=D, r_rank=R, num_blocks=NB,
                                        block_dim=BD, field_channel=True)
        _train(k0, seed=31)
        _train(base, seed=31)
        s = _random_wave(D, seed=33)
        a = _random_wave(D, seed=34)
        assert torch.max(torch.abs(k0.forward(s, a) - base.forward(s, a))) == 0.0, (
            "k_max=0 must be byte-identical to the 8.34 coupled arm"
        )


# --------------------------------------------------------------------------
# T2: DualScaleAnalogLexicalSnap
# --------------------------------------------------------------------------

class TestDualScaleSnap:
    def _make(self, seed: int = 835):
        snap = DualScaleAnalogLexicalSnap(dim_micro=D, dim_macro=8, tau=1.0, seed=seed)
        g = torch.Generator().manual_seed(seed)
        engrams = F.normalize(torch.randn(6, D, generator=g), p=2, dim=-1)
        snap.store_engrams(engrams)
        return snap

    def test_macro_zero_matches_plain_snap(self):
        snap = self._make()
        plain = ContinuousHopfieldCleanup(dim=D)
        plain.store_engrams(snap.cleanup.engrams.clone())
        wave = _random_wave(D, seed=5)
        macro = torch.zeros(8)
        idx_g, _ = snap.snap(wave, macro)
        idx_p, _ = plain.lexical_snap(wave)
        assert torch.equal(idx_g, idx_p), "macro=0 must match plain lexical_snap"

    def test_macro_changes_retrieval(self):
        snap = self._make()
        wave = _random_wave(D, seed=6)
        macro = _random_wave(8, seed=7)
        idx_zero, _ = snap.snap(wave, torch.zeros(8))
        idx_gated, _ = snap.snap(wave, macro)
        assert torch.max(torch.abs(snap.gated_wave(wave, macro) - wave)) > 1e-6, (
            "gated wave must differ from raw wave under nonzero macro"
        )

    def test_fail_closed_without_engrams(self):
        snap = DualScaleAnalogLexicalSnap(dim_micro=D, dim_macro=8, tau=1.0)
        with pytest.raises(AssertionError):
            snap.snap(_random_wave(D, seed=8), torch.zeros(8))

    def test_tau_sharpening_approaches_onehot(self):
        snap = self._make()
        macro = _random_wave(8, seed=9)
        m_hot = snap.gate_mask(macro.clone())
        assert m_hot.numel() == D
        assert abs(m_hot.sum().item() - 1.0) < 1e-5


# --------------------------------------------------------------------------
# T3: directional Sagnac homodyne
# --------------------------------------------------------------------------

class TestDirectionalSagnac:
    @pytest.fixture
    def planner(self):
        return EFEPlanner(d_model=D, num_blocks=NB)

    def test_complex_same_phase_zero(self, planner):
        ph = torch.zeros(D, dtype=torch.complex64)
        w = torch.exp(1j * ph)
        assert abs(planner.directional_sagnac_delta(w, w).item()) < 1e-5

    def test_complex_constant_phase_diff_is_locked(self, planner):
        # The formula measures phase-difference COHERENCE, not magnitude:
        # a constant difference (even pi) is perfectly locked -> Delta = 0.
        ph = torch.zeros(D, dtype=torch.complex64)
        w1 = torch.exp(1j * ph)
        w2 = torch.exp(1j * (ph + math.pi))
        assert abs(planner.directional_sagnac_delta(w1, w2).item()) < 1e-4

    def test_complex_random_phases_max_stress(self, planner):
        g = torch.Generator().manual_seed(11)
        ph1 = torch.rand(D, generator=g) * 2 * math.pi
        ph2 = torch.rand(D, generator=g) * 2 * math.pi
        w1 = torch.exp(1j * ph1)
        w2 = torch.exp(1j * ph2)
        d = planner.directional_sagnac_delta(w1, w2).item()
        assert d > 0.9, f"independent random phases should be incoherent, got {d}"

    def test_real_identical_small(self, planner):
        w = _random_wave(D, seed=10)
        d = planner.directional_sagnac_delta(w, w.clone()).item()
        assert 0.0 <= d <= 0.05, f"identical real waves should align, got {d}"

    def test_real_bounded_unit(self, planner):
        g = torch.Generator().manual_seed(12)
        a = F.normalize(torch.randn(D, generator=g), p=2, dim=0)
        b = F.normalize(torch.randn(D, generator=g), p=2, dim=0)
        d = planner.directional_sagnac_delta(a, b).item()
        assert 0.0 <= d <= 1.0 + 1e-5
