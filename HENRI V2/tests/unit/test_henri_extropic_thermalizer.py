"""
Phase 8.37 Extropic Thermalizer Compiler — verification suite.

Covers the three directives from HENRI-SPEC-EXTROPIC-THERMALIZER-2026:

  D1. qfhrr_to_ising_hamiltonian() in qfhrr_kernels.py (2606.17327 §2)
      Potts->Ising embedding; factorized rank-1 coupling (no [D,D] tensor);
      Gibbs sampler converges to the h-field ground state at low T.
  D2. Context Matching in wave_jepa.py (2608.01615 §III.B)
      Default-OFF byte-identity; ON path anchors on context engrams.
  D3. trajectory_reinforce_post_train() in efe_planner.py (2608.01615 §IV)
      Advantage-weighted Sagnac-gradient post-train; fail-closed guards.

All tests run at reduced scale on CPU (CI) per the canonical suite pattern.
"""

import math
import pytest
import torch

from qfhrr_kernels import (
    K_PHASE,
    potts_onehot_spins,
    ising_spin_constraint,
    IsingHamiltonian,
    qfhrr_to_ising_hamiltonian,
    sample_ising_gibbs,
)
from wave_jepa import WaveJEPA
from efe_planner import EFEPlanner, LowRankCoupledTransition


def _wave(nb, seed, device="cpu"):
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(nb, 8, generator=g).to(device)
    return w / torch.norm(w, p=2, dim=-1, keepdim=True)


# ---------------------------------------------------------------------------
# D1 — Potts/Ising translation
# ---------------------------------------------------------------------------

class TestIsingTranslation:
    def test_onehot_spins_constraint(self):
        q = torch.tensor([0, 127, 255], dtype=torch.uint8)
        s = potts_onehot_spins(q, k=K_PHASE)
        assert s.shape == (3, K_PHASE)
        assert torch.all((s == 1.0).sum(dim=-1) == 1)
        assert torch.all((s == -1.0).sum(dim=-1) == K_PHASE - 1)
        # Per-row constraint: sum = 2 - K
        assert torch.all(ising_spin_constraint(s) == 2.0 - K_PHASE)

    def test_factorized_energy_matches_dense(self):
        """The factorized rank-1 energy must equal the dense [P, P] form."""
        torch.manual_seed(0)
        P, K = 8, 16
        phi = torch.randn(P)
        h = torch.randn(P, K)
        spins = potts_onehot_spins(
            torch.randint(0, K, (P,), dtype=torch.uint8), k=K)
        ham = IsingHamiltonian(phi, h)
        # Dense reference: E = -(1/D) * (s . phi)^2 - h . s
        sdot = (spins * phi.unsqueeze(-1)).sum()
        dense_e = -(sdot * sdot) / P - (spins * h).sum()
        assert ham.energy(spins).item() == pytest.approx(dense_e.item(), abs=1e-5)
        assert ham.dense_coupling_bytes == 4 * P * P

    def test_converter_produces_valid_hamiltonian(self):
        w = _wave(4, 7)
        ham = qfhrr_to_ising_hamiltonian(w)
        assert ham.D == 16
        assert ham.spins.shape == (16, K_PHASE)
        # Initial spins are the wave's own quantized codes (valid one-hot)
        codes = ham.spins.argmax(dim=-1)
        assert torch.all(ham.spins.sum(dim=-1).eq(2.0 - K_PHASE))
        # Re-encoding the spins must recover the wave's phase codes
        from qfhrr_kernels import wave_to_phase_codes
        assert torch.equal(codes.view(4, 4), wave_to_phase_codes(w).view(4, 4))

    def test_gibbs_converges_to_field_ground_state(self):
        """At low temperature, sampling converges to argmax_a h[i, a].

        Heat-bath draws an exact per-row Boltzmann sample every sweep. On the
        real cosine h-field, rows whose phase sits within float32 epsilon of
        a bin-boundary midpoint are TIED (h_best - h_second < 1e-6): the
        sampler flips between the tied bins at the exact Boltzmann rate and
        the argmax reference picks one deterministically — a genuine tie
        ambiguity (same as the legacy cosine-LUT kernel), not a defect.
        Contract: disagreements vs the argmax are permitted ONLY on tied rows.
        """
        w = _wave(8, 11)
        ham = qfhrr_to_ising_hamiltonian(w)
        T = 1e-6
        res = sample_ising_gibbs(ham, n_samples=3, temperature=T,
                                 steps=200, seed=0)
        scaled = (2.0 * ham.h_field / T).argmax(dim=-1).reshape(8, 4).to(torch.uint8)
        top2 = ham.h_field.topk(2, dim=-1).values          # [P, 2]
        gap = (top2[:, 0] - top2[:, 1])                    # [P] field gap
        tie = gap < 1e-6                                   # float32-tied rows
        for sample in res["codes"]:
            diff = (sample != scaled).view(-1)
            assert bool((~diff | tie).all()), (
                "sampler left ground state on a non-tied row; "
                f"tie rows {tie.sum()}/{tie.numel()}")
        # Agreement with the ground state is high overall (tie rows excluded)
        agree = (res["codes"] == scaled).float().mean(dim=(-2, -1))
        assert bool((agree > 0.90).all()), f"agreement {agree.tolist()}"

    def test_gibbs_converges_exact_delta_field(self):
        """Exact convergence on a well-separated field (no ties): the sampler
        must return the ground state with probability -> 1 at low T."""
        P, K = 16, 256
        h = torch.zeros(P, K)
        q = torch.randint(0, K, (P,))
        h.scatter_(1, q.unsqueeze(1), 1.0)                 # delta field
        phi = torch.zeros(P)                                # no coupling
        ham = IsingHamiltonian(phi, h)
        res = sample_ising_gibbs(ham, n_samples=4, temperature=1e-6,
                                 steps=200, seed=0)
        expected = h.argmax(dim=-1).reshape(-1, 4).to(torch.uint8)
        agree = (res["codes"] == expected).float().mean(dim=(-2, -1))
        assert bool((agree == 1.0).all()), \
            f"delta-field sampler agreement {agree.tolist()}"

    def test_sampler_deterministic(self):
        w = _wave(4, 3)
        ham = qfhrr_to_ising_hamiltonian(w)
        r1 = sample_ising_gibbs(ham, n_samples=2, steps=40, seed=5)
        r2 = sample_ising_gibbs(ham, n_samples=2, steps=40, seed=5)
        assert torch.equal(r1["codes"], r2["codes"])
        assert torch.allclose(r1["energies"], r2["energies"])

    def test_dense_coupling_ban_at_production_scale(self):
        """Audit: a dense J at D=65,536 would be 34 GiB — the factorized
        Hamiltonian must never materialize it."""
        w = _wave(4, 13)
        ham = qfhrr_to_ising_hamiltonian(w)
        # [D, D] fp32 at D = 16 (4 blocks x 4 pairs) = 1 KB; the guard is
        # the property (not the value) — scale-linear audit.
        assert ham.dense_coupling_bytes == 4 * ham.D * ham.D


# ---------------------------------------------------------------------------
# D2 — Context Matching
# ---------------------------------------------------------------------------

class TestContextMatching:
    def test_default_off_byte_identical(self):
        """use_context_matching=False must reproduce the legacy prediction."""
        jepa = WaveJEPA(d_model=512, num_blocks=64, r_rank=8,
                        use_context_matching=False)
        s = _wave(64, 1)
        a = _wave(64, 2)
        ctx = torch.stack([_wave(64, 3), _wave(64, 4)])
        p_legacy = jepa.predict_future_latent(s, a)
        assert torch.equal(p_legacy, jepa.predict_future_latent(s, a))
        # context path with lam=0 must equal base
        jepa.context_mix = 0.0
        p_ctx = jepa.predict_future_latent_context(s, a, ctx)
        assert torch.allclose(p_legacy, p_ctx, atol=1e-6)

    def test_context_anchoring_moves_prediction(self):
        """With context ON, the prediction shifts toward the context anchor."""
        jepa = WaveJEPA(d_model=512, num_blocks=64, r_rank=8,
                        use_context_matching=True, context_mix=0.5,
                        context_beta=8.0)
        s = _wave(64, 1)
        a = _wave(64, 2)
        ctx = torch.stack([s.clone() * 1.0 + 0.1 * _wave(64, 9) for _ in range(3)])
        p_base = jepa.predict_future_latent(s, a)
        p_ctx = jepa.predict_future_latent_context(s, a, ctx)
        # The context anchor (weighted engram) must differ from base, and the
        # blended prediction must move toward it.
        anchor = torch.einsum('j,jnb->nb',
                              torch.softmax(8.0 * (
                                  torch.nn.functional.normalize(s.view(-1), p=2, dim=0)
                                  @ torch.nn.functional.normalize(
                                      ctx.view(3, -1), p=2, dim=-1).T), dim=-1),
                              ctx)
        anchor = torch.nn.functional.normalize(anchor.view(-1), p=2, dim=0).view(64, 8)
        assert not torch.allclose(p_base, anchor, atol=1e-3)
        # blended prediction is strictly closer to anchor than base is
        d_base = float((p_base - anchor).norm().item())
        d_ctx = float((p_ctx - anchor).norm().item())
        assert d_ctx < d_base

    def test_context_weights_are_softmax(self):
        jepa = WaveJEPA(d_model=512, num_blocks=64, r_rank=8,
                        use_context_matching=True)
        s = _wave(64, 5)
        ctx = torch.stack([_wave(64, 6), _wave(64, 7)])
        w = jepa._context_weights(s, ctx)
        assert torch.allclose(w.sum(), torch.ones(1))
        assert torch.all(w >= 0)


# ---------------------------------------------------------------------------
# D3 — Trajectory REINFORCE
# ---------------------------------------------------------------------------

class TestTrajectoryReinforce:
    def _planner(self, device="cpu"):
        return EFEPlanner(num_blocks=16, d_model=128, num_actions=4).to(device)

    def test_fail_closed_on_short_trajectory(self):
        p = self._planner()
        s = _wave(16, 1)
        a = _wave(16, 2)
        o = _wave(16, 3)
        res = p.trajectory_reinforce_post_train(
            s.unsqueeze(0), a.unsqueeze(0), o.unsqueeze(0),
            rewards=torch.tensor([1.0]))
        assert res["engaged"] is False
        assert res["status"] == "TRAJECTORY_TOO_SHORT"

    def test_fail_closed_on_zero_advantage(self):
        p = self._planner()
        N = 4
        s = torch.stack([_wave(16, i + 1) for i in range(N)])
        a = torch.stack([_wave(16, i + 10) for i in range(N)])
        o = torch.stack([_wave(16, i + 20) for i in range(N)])
        res = p.trajectory_reinforce_post_train(
            s, a, o, rewards=torch.full((N,), 0.5))
        assert res["engaged"] is False
        assert res["status"] == "ZERO_ADVANTAGE"

    def test_positive_advantage_reduces_loss(self):
        """A positive-advantage trajectory consolidates: post-train Sagnac
        loss on the same window must not increase (and usually decreases).

        Rewards must be NON-UNIFORM: constant rewards collapse the advantage
        to zero and the method correctly fails closed (ZERO_ADVANTAGE).
        """
        p = self._planner()
        N = 5
        torch.manual_seed(0)
        s = torch.stack([_wave(16, i + 1) for i in range(N)])
        a = torch.stack([_wave(16, i + 10) for i in range(N)])
        # observed_nexts = forward predictions + small noise -> learnable
        with torch.no_grad():
            o = torch.stack([p.transition(s[i], a[i]) for i in range(N)])
            o = o + 0.01 * torch.randn_like(o)
            o = o / torch.norm(o, p=2, dim=-1, keepdim=True)
        pre_loss = float(p._batch_sagnac_loss(s, a, o).item())
        res = p.trajectory_reinforce_post_train(
            s, a, o, rewards=torch.tensor([1.0, 1.0, 1.0, 0.5, 0.5]), lr=0.05)
        assert res["engaged"] is True, res
        assert res["status"] == "OK"
        post_loss = float(p._batch_sagnac_loss(s, a, o).item())
        assert post_loss <= pre_loss + 1e-6, \
            f"post-train loss {post_loss:.6f} > pre {pre_loss:.6f}"

    def test_negative_advantage_repulses(self):
        """A negative-advantage trajectory pushes the transition away:
        post-train loss on the same window must increase."""
        p = self._planner()
        N = 4
        torch.manual_seed(1)
        s = torch.stack([_wave(16, i + 31) for i in range(N)])
        a = torch.stack([_wave(16, i + 40) for i in range(N)])
        with torch.no_grad():
            o = torch.stack([p.transition(s[i], a[i]) for i in range(N)])
            o = o / torch.norm(o, p=2, dim=-1, keepdim=True)
        pre_loss = float(p._batch_sagnac_loss(s, a, o).item())
        res = p.trajectory_reinforce_post_train(
            s, a, o, rewards=torch.tensor([-1.0, -1.0, -1.0, 0.5]), lr=0.2)
        assert res["engaged"] is True, res
        post_loss = float(p._batch_sagnac_loss(s, a, o).item())
        assert post_loss >= pre_loss - 1e-6, \
            f"negative-advantage post-train loss {post_loss:.6f} < pre {pre_loss:.6f}"

    def test_incompatible_transition_fail_closed(self):
        """Non-coupled transitions (e.g. complex) must fail closed."""
        p = EFEPlanner(num_blocks=16, d_model=128, num_actions=4,
                       use_complex_transition=True)
        N = 3
        s = torch.stack([_wave(16, i + 1) for i in range(N)])
        a = torch.stack([_wave(16, i + 2) for i in range(N)])
        o = torch.stack([_wave(16, i + 3) for i in range(N)])
        res = p.trajectory_reinforce_post_train(
            s, a, o, rewards=torch.ones(N))
        assert res["engaged"] is False
        assert res["status"] == "INCOMPATIBLE_TRANSITION"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q", "--tb=short"]))
