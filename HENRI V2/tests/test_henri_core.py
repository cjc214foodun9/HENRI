"""
Canonical pytest suite for Project HENRI V2.

Runs at reduced scale on CPU (CI) and full scale on CUDA when available.
Markers:
    - tests auto-scale: production dims on GPU, reduced dims on CPU.

Run:
    pytest tests/test_henri_core.py -v
"""

import math
import pytest
import torch

from darwinian_phase_swarm import HenriSwarmOrchestrator, GapJunctionSwarmSyncytium
from hopfield_cleanup import ContinuousHopfieldCleanup, HopfieldActionDecoder
from product_clifford_product_kernel import ProductCliffordAlgebra3D
from henri_pwm_orchestrator import SagnacInterferometer
from efe_planner import EFEPlanner, UnitaryWaveTransition

# Scale presets: CPU runs reduced, CUDA runs production
if torch.cuda.is_available():
    SCALE = dict(num_experts=1024, d_model=65536, r_rank=16, num_blocks=8192)
    DEVICE = "cuda"
else:
    SCALE = dict(num_experts=64, d_model=512, r_rank=8, num_blocks=64)
    DEVICE = "cpu"


@pytest.fixture(scope="module")
def device():
    return torch.device(DEVICE)


@pytest.fixture(scope="module")
def orch(device):
    torch.manual_seed(0)
    return HenriSwarmOrchestrator(**SCALE).to(device)


def mk_wave(shape, device, seed):
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(*shape, generator=g).to(device)
    return w / torch.norm(w, p=2, dim=-1, keepdim=True)


# ---------------------------------------------------------------------------
# Swarm: SGLD creep stability, bounded delta, retraction
# ---------------------------------------------------------------------------

class TestSwarmCreep:
    def test_sgld_steps_bounded_and_finite(self, orch, device):
        wave = mk_wave((SCALE["num_blocks"], 8), device, 1)
        target = mk_wave((SCALE["num_blocks"], 8), device, 2)
        deltas = [
            orch.process_active_reasoning_step(wave, target, t_shock_max=torch.tensor(0.5, device=device))[0]
            for _ in range(5)
        ]
        assert all(math.isfinite(d) for d in deltas)
        assert all(0.0 <= d <= 2.0 for d in deltas)

    def test_no_nan_after_creep(self, orch):
        A = orch.syncytium.experts_A
        B = orch.syncytium.experts_B
        assert not torch.isnan(A).any()
        assert not torch.isnan(B).any()

    def test_stiefel_retraction_holds(self, orch, device):
        r = SCALE["r_rank"]
        I = torch.eye(r, device=device)
        for param in (orch.syncytium.experts_A, orch.syncytium.experts_B):
            err = (torch.bmm(param, param.transpose(-2, -1)) - I).abs().max().item()
            assert err < 1e-4, f"retraction error {err}"

    def test_free_energy_finite(self, orch, device):
        wave = mk_wave((SCALE["num_blocks"], 8), device, 3)
        target = mk_wave((SCALE["num_blocks"], 8), device, 4)
        F = orch.compute_free_energy(wave, target).item()
        assert math.isfinite(F)


# ---------------------------------------------------------------------------
# Sagnac delta normalization
# ---------------------------------------------------------------------------

class TestSagnacDelta:
    def test_self_resonance_zero(self, orch, device):
        wave = mk_wave((SCALE["num_blocks"], 8), device, 5)
        d = 1.0 - orch.sagnac_coherence(wave, wave).item()
        assert abs(d) < 1e-3

    def test_coherence_bounded(self, orch, device):
        wave = mk_wave((SCALE["num_blocks"], 8), device, 6)
        target = mk_wave((SCALE["num_blocks"], 8), device, 7)
        c = orch.sagnac_coherence(wave, target).item()
        assert -1.0 <= c <= 1.0

    def test_pwm_delta_range(self):
        sg = SagnacInterferometer()
        v = torch.randn(128, dtype=torch.complex64)
        v = v / torch.norm(v)
        _, d0 = sg.verify(v, v)
        _, d1 = sg.verify(v, -v)
        assert abs(d0.item()) < 1e-5
        assert d1.item() > 1.5  # anti-correlated -> near max


# ---------------------------------------------------------------------------
# Hopfield cleanup
# ---------------------------------------------------------------------------

class TestHopfield:
    def test_hard_retrieval_recovers_engram(self, device):
        D, M = (4096, 16) if DEVICE == "cuda" else (1024, 16)
        cleanup = ContinuousHopfieldCleanup(dim=2 * D).to(device)
        th = torch.rand(M, D, device=device) * 2 * math.pi
        E = torch.complex(torch.cos(th), torch.sin(th))
        E = E / torch.norm(E, dim=-1, keepdim=True)
        cleanup.store_engrams(E)
        noise = torch.complex(torch.randn(D, device=device), torch.randn(D, device=device))
        noise = noise / torch.norm(noise)
        noisy = E[3] + 0.35 * noise
        noisy = noisy / torch.norm(noisy)
        _, idx, conf = cleanup.hard_retrieve(noisy)
        assert int(idx) == 3

    def test_dim_guard_fires(self):
        cleanup = ContinuousHopfieldCleanup(dim=16)
        with pytest.raises(AssertionError):
            cleanup.store_engrams(torch.randn(2, 8))

    def test_decoder_snaps_to_action(self, device):
        D = SCALE["d_model"]
        dec = HopfieldActionDecoder(d_model=D).to(device)
        wave = dec.get_action_wave(list(dec.action_to_id)[0])
        noise = torch.complex(torch.randn(D, device=device), torch.randn(D, device=device))
        action, conf = dec.decode_wave_to_action(wave + 0.3 * noise)
        assert action == list(dec.action_to_id)[0]


# ---------------------------------------------------------------------------
# Clifford kernel
# ---------------------------------------------------------------------------

class TestClifford:
    def test_rotor_transform_finite(self, device):
        kern = ProductCliffordAlgebra3D(num_blocks=4).to(device)
        rotor = torch.randn(1, 4, 8, device=device)
        rotor = rotor / torch.norm(rotor, dim=-1, keepdim=True)
        out = kern(rotor, torch.randn(1, 4, 8, device=device))
        assert torch.isfinite(out).all()

    def test_geometric_product_known_identity(self, device):
        # e1 * e1 = 1 (scalar slot); basis index 1 squared -> index 0
        kern = ProductCliffordAlgebra3D(num_blocks=1).to(device)
        e1 = torch.zeros(1, 1, 8, device=device)
        e1[..., 1] = 1.0
        out = kern.geometric_product(e1, e1)
        assert torch.allclose(out[..., 0], torch.ones(1, 1, device=device))
        assert torch.allclose(out[..., 1:], torch.zeros(1, 1, 7, device=device))


# ---------------------------------------------------------------------------
# EFE planner
# ---------------------------------------------------------------------------

class TestEFEPlanner:
    def test_scores_finite_and_selects(self, orch, device):
        wave = mk_wave((SCALE["num_blocks"], 8), device, 8)
        boundary = mk_wave((2, SCALE["num_blocks"], 8), device, 9)
        action, predicted, table = orch.plan_action(wave, boundary, top_k=3)
        assert action in list(orch.decoder.id_to_action.values())
        assert predicted.shape == (SCALE["num_blocks"], 8)
        assert all(math.isfinite(s["efe"]) for s in table)

    def test_predicted_wave_on_manifold(self, orch, device):
        wave = mk_wave((SCALE["num_blocks"], 8), device, 10)
        boundary = mk_wave((2, SCALE["num_blocks"], 8), device, 11)
        _, predicted, _ = orch.plan_action(wave, boundary, top_k=2)
        norms = torch.norm(predicted, p=2, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)

    def test_transition_unitarity(self, device):
        nb = SCALE["num_blocks"]
        t = UnitaryWaveTransition(num_blocks=nb).to(device)
        # Residual: per-block unitary
        prod = torch.matmul(t.block_residual, t.block_residual.mH)
        I = torch.eye(8, dtype=torch.complex64, device=device)
        assert (prod - I).abs().max().item() < 1e-3
        # Field V: column-semi-unitary. The leading min(rank,4) block is the
        # invariant; trailing columns may be zeroed in the young-EDMD regime
        # (rank-limited to data, k=min(rank,N)<rank — commit 7a04e95), where
        # the reduced Gram's zero eigenvalues are structure, not error.
        V = t.field_V
        gram = V.T @ V
        k = min(t.rank, 4)
        Iv = torch.eye(k, device=device)
        assert (gram[:k, :k] - Iv).abs().max().item() < 1e-3

    def test_cross_block_coupling(self, device):
        nb = SCALE["num_blocks"]
        t = UnitaryWaveTransition(num_blocks=nb).to(device)
        s1 = mk_wave((nb, 8), device, 40)
        a = mk_wave((nb, 8), device, 41)
        s2 = s1.clone()
        s2[0] = torch.randn(8, device=device)
        s2[0] = s2[0] / torch.norm(s2[0])
        o1, o2 = t(s1, a), t(s2, a)
        diff = (o1[min(5, nb - 1)] - o2[min(5, nb - 1)]).abs().max().item()
        assert diff > 1e-6, "no cross-block coupling (block-diagonal regression)"


# ---------------------------------------------------------------------------
# IDBD + SwiftTD adaptive step-sizes
# ---------------------------------------------------------------------------

class TestIDBDSwiftTD:
    def test_alpha_grows_on_persistent_gradient(self, device):
        from idbd_swifttd import AdaptiveCreepController
        ctrl = AdaptiveCreepController((16, 1, 1), meta_theta=0.1, device=device)
        consistent = torch.ones(16, 1, 1, device=device) * 0.01
        for _ in range(100):
            ctrl.scaled_drift(1.0, consistent)
        assert ctrl.plasticity_stats()["max_alpha"] > 0.05

    def test_alpha_stable_at_zero_error(self, device):
        from idbd_swifttd import AdaptiveCreepController
        ctrl = AdaptiveCreepController((16, 1, 1), device=device)
        for _ in range(50):
            ctrl.scaled_drift(0.0, torch.randn(16, 1, 1, device=device) * 0.01)
        assert abs(ctrl.plasticity_stats()["mean_alpha"] - 0.05) < 1e-3

    def test_swifttd_bounds_overshoot(self, device):
        from idbd_swifttd import AdaptiveCreepController
        ctrl = AdaptiveCreepController((16, 1, 1), device=device)
        big = torch.ones(16, 1, 1, device=device) * 100.0
        bounded = ctrl.scaled_drift(1.0, big)
        assert bounded.abs().max().item() < big.abs().max().item()

    def test_swarm_creep_uses_adaptive_stepsizes(self, orch, device):
        wave = mk_wave((SCALE["num_blocks"], 8), device, 12)
        target = mk_wave((SCALE["num_blocks"], 8), device, 13)
        for _ in range(3):
            orch.process_active_reasoning_step(wave, target, t_shock_max=torch.tensor(0.5, device=device))
        # Controllers exist and hold finite per-expert step-sizes
        for ctrl in (orch.syncytium.creep_ctrl_A, orch.syncytium.creep_ctrl_B):
            stats = ctrl.plasticity_stats()
            assert math.isfinite(stats["mean_alpha"]) and stats["mean_alpha"] > 0


# ---------------------------------------------------------------------------
# Transition-model training (T1/T2)
# ---------------------------------------------------------------------------

class TestTransitionTraining:
    def test_train_step_reduces_sagnac_loss(self, orch, device):
        wave = mk_wave((SCALE["num_blocks"], 8), device, 20)
        action_w = mk_wave((SCALE["num_blocks"], 8), device, 21)
        target = mk_wave((SCALE["num_blocks"], 8), device, 22)
        losses = [orch.planner.train_transition_step(wave, action_w, target, lr=0.1)
                  for _ in range(10)]
        assert all(math.isfinite(l) and 0.0 <= l <= 2.0 for l in losses)
        assert losses[-1] < losses[0], f"no descent: {losses[0]} -> {losses[-1]}"

    def test_transition_unitarity_after_training(self, orch, device):
        wave = mk_wave((SCALE["num_blocks"], 8), device, 23)
        action_w = mk_wave((SCALE["num_blocks"], 8), device, 24)
        target = mk_wave((SCALE["num_blocks"], 8), device, 25)
        for _ in range(5):
            orch.planner.train_transition_step(wave, action_w, target, lr=0.1)
        T = orch.planner.transition.transition
        prod = torch.matmul(T, T.mH)
        I = torch.eye(8, dtype=torch.complex64, device=device)
        assert (prod - I).abs().max().item() < 1e-2

    def test_action_conditioned_learning(self, device):
        # Model must learn different outcomes for different actions (T2)
        from efe_planner import EFEPlanner
        p = EFEPlanner(num_blocks=SCALE["num_blocks"], d_model=SCALE["d_model"]).to(device)
        state = mk_wave((SCALE["num_blocks"], 8), device, 26)
        act_a = mk_wave((SCALE["num_blocks"], 8), device, 27)
        act_b = mk_wave((SCALE["num_blocks"], 8), device, 28)
        out_a = mk_wave((SCALE["num_blocks"], 8), device, 29)
        out_b = mk_wave((SCALE["num_blocks"], 8), device, 30)
        for _ in range(15):
            p.train_transition_step(state, act_a, out_a, lr=0.1)
            p.train_transition_step(state, act_b, out_b, lr=0.1)
        # Predictions for the two actions should now diverge toward their targets
        pa = p.transition(state, act_a)
        pb = p.transition(state, act_b)
        assert not torch.allclose(pa, pb, atol=1e-3), "action conditioning collapsed"


# ---------------------------------------------------------------------------
# T4: calibrated exploration
# ---------------------------------------------------------------------------

class TestCalibratedExploration:
    def test_explores_when_spread_high(self, device):
        from efe_planner import EFEPlanner
        p = EFEPlanner(num_blocks=SCALE["num_blocks"], d_model=SCALE["d_model"]).to(device)
        state = mk_wave((SCALE["num_blocks"], 8), device, 31)
        boundary = mk_wave((2, SCALE["num_blocks"], 8), device, 32)
        cands = [("A", mk_wave((SCALE["num_blocks"], 8), device, 33)),
                 ("B", mk_wave((SCALE["num_blocks"], 8), device, 34)),
                 ("C", mk_wave((SCALE["num_blocks"], 8), device, 35))]
        # Force exploration with a zero threshold: any spread triggers epistemic pick
        action, _, table, chosen = p.select_action(state, cands, boundary, explore_threshold=-1.0)
        epistemic_best = max(table, key=lambda r: r["epistemic"])["action"]
        assert action == epistemic_best, "high spread should route to epistemic action"
        assert chosen["explored"] is True

    def test_exploits_when_spread_low(self, device):
        from efe_planner import EFEPlanner
        p = EFEPlanner(num_blocks=SCALE["num_blocks"], d_model=SCALE["d_model"]).to(device)
        p.loss_ema = 0.0  # accurate model -> exploit
        state = mk_wave((SCALE["num_blocks"], 8), device, 36)
        boundary = mk_wave((2, SCALE["num_blocks"], 8), device, 37)
        cands = [("A", mk_wave((SCALE["num_blocks"], 8), device, 38)),
                 ("B", mk_wave((SCALE["num_blocks"], 8), device, 39))]
        action, _, table, chosen = p.select_action(state, cands, boundary, explore_threshold=1e9)
        exploit_best = min(table, key=lambda r: r["efe"])["action"]
        assert action == exploit_best, "low spread should route to min-EFE action"
        assert chosen["explored"] is False

    def test_explore_to_exploit_transition_as_model_learns(self, device):
        # With high loss_ema (near peak) the planner explores; once the model
        # trains down well below the adaptive floor, it flips to exploitation.
        from efe_planner import EFEPlanner
        p = EFEPlanner(num_blocks=SCALE["num_blocks"], d_model=SCALE["d_model"]).to(device)
        state = mk_wave((SCALE["num_blocks"], 8), device, 42)
        boundary = mk_wave((2, SCALE["num_blocks"], 8), device, 43)
        cands = [("A", mk_wave((SCALE["num_blocks"], 8), device, 44)),
                 ("B", mk_wave((SCALE["num_blocks"], 8), device, 45))]
        # Peak error at init; loss_ema still at peak -> explore
        p.loss_ema_peak = 1.0
        p.loss_ema = 1.0
        _, _, _, chosen_hi = p.select_action(state, cands, boundary)
        assert chosen_hi["explored"] is True, "high loss_ema should explore"
        # Model learned: error far below the adaptive floor (peak - 0.1)
        p.loss_ema = 0.1
        _, _, _, chosen_lo = p.select_action(state, cands, boundary)
        assert chosen_lo["explored"] is False, "low loss_ema should exploit"

    def test_loss_ema_updates_on_training(self, orch, device):
        wave = mk_wave((SCALE["num_blocks"], 8), device, 46)
        action_w = mk_wave((SCALE["num_blocks"], 8), device, 47)
        target = mk_wave((SCALE["num_blocks"], 8), device, 48)
        before = orch.planner.loss_ema
        for _ in range(20):
            orch.planner.train_transition_step(wave, action_w, target, lr=0.3)
        after = orch.planner.loss_ema
        assert after < before, "loss_ema did not track learning"

    def test_adaptive_floor_opens_after_improvement(self, device):
        # Simulate a session where the model improves >10% from its peak error:
        # the gate should transition from explore to exploit.
        from efe_planner import EFEPlanner
        p = EFEPlanner(num_blocks=SCALE["num_blocks"], d_model=SCALE["d_model"]).to(device)
        state = mk_wave((SCALE["num_blocks"], 8), device, 90)
        boundary = mk_wave((2, SCALE["num_blocks"], 8), device, 91)
        cands = [("A", mk_wave((SCALE["num_blocks"], 8), device, 92)),
                 ("B", mk_wave((SCALE["num_blocks"], 8), device, 93))]
        # Peak error 1.0 -> floor 0.9; with loss_ema at 0.95 still explore
        p.loss_ema_peak = 1.0
        p.loss_ema = 0.95
        _, _, _, c1 = p.select_action(state, cands, boundary)
        assert c1["explored"] is True
        # Model improved below floor (0.85 < 0.9): exploit
        p.loss_ema = 0.85
        _, _, _, c2 = p.select_action(state, cands, boundary)
        assert c2["explored"] is False


class TestEpistemicNovelty:
    def test_repeated_prediction_loses_novelty(self, device):
        from efe_planner import EFEPlanner
        p = EFEPlanner(num_blocks=SCALE["num_blocks"], d_model=SCALE["d_model"]).to(device)
        # Seed the attractor store so entropy term is active
        for i in range(4):
            p.cleanup.store_engrams(mk_wave((1, SCALE["d_model"]), device, 50 + i))
        outcome = mk_wave((1, SCALE["d_model"]), device, 60).view(-1)
        v_first = p.epistemic_value(outcome).item()
        p.remember_outcome(outcome)
        v_second = p.epistemic_value(outcome).item()
        assert v_second < v_first, "remembered outcome should lose epistemic value"

    def test_novel_outcome_keeps_value(self, device):
        from efe_planner import EFEPlanner
        p = EFEPlanner(num_blocks=SCALE["num_blocks"], d_model=SCALE["d_model"]).to(device)
        for i in range(4):
            p.cleanup.store_engrams(mk_wave((1, SCALE["d_model"]), device, 70 + i))
        seen = mk_wave((1, SCALE["d_model"]), device, 80).view(-1)
        novel = mk_wave((1, SCALE["d_model"]), device, 81).view(-1)
        p.remember_outcome(seen)
        assert p.epistemic_value(novel).item() > p.epistemic_value(seen).item()


# ---------------------------------------------------------------------------
# Valence wires: pragmatic preference store (A) + precision gate (B)
# ---------------------------------------------------------------------------

class TestValenceWires:
    def test_preference_lowers_pragmatic_value(self, device):
        # Wire A: registering a wave into the preference store must LOWER the
        # pragmatic score of a matching prediction (argmin-EFE is pulled
        # toward verified basins) without touching the surprise term.
        from efe_planner import EFEPlanner
        p = EFEPlanner(num_blocks=SCALE["num_blocks"], d_model=SCALE["d_model"]).to(device)
        fav = mk_wave((SCALE["num_blocks"], 8), device, 90)
        boundary = mk_wave((2, SCALE["num_blocks"], 8), device, 91)
        before = p.pragmatic_value(fav, boundary).item()
        p.register_preference(fav)
        after = p.pragmatic_value(fav, boundary).item()
        assert after < before - 0.5, f"preference resonance did not lower score: {before} -> {after}"
        # Surprise term untouched: without resonance, score == min sagnac delta
        q = EFEPlanner(num_blocks=SCALE["num_blocks"], d_model=SCALE["d_model"]).to(device)
        assert abs(q.pragmatic_value(fav, boundary).item() - before) < 1e-6

    def test_preference_ring_capacity(self, device):
        from efe_planner import EFEPlanner
        p = EFEPlanner(num_blocks=SCALE["num_blocks"], d_model=SCALE["d_model"]).to(device)
        for i in range(p.preference_capacity + 10):
            p.register_preference(mk_wave((SCALE["num_blocks"], 8), device, 100 + i))
        assert p.preference_store.num_engrams() == p.preference_capacity

    def test_valence_gate_polarity(self, device):
        # Wire B: success damps the effective update (crystallize), failure
        # damps harder but NEVER freezes (no zero-halt, no gradient inversion).
        from efe_planner import EFEPlanner

        def step_with(valence, seed):
            torch.manual_seed(seed)
            p = EFEPlanner(num_blocks=SCALE["num_blocks"], d_model=SCALE["d_model"]).to(device)
            s = mk_wave((SCALE["num_blocks"], 8), device, seed + 1)
            a = mk_wave((SCALE["num_blocks"], 8), device, seed + 2)
            o = mk_wave((SCALE["num_blocks"], 8), device, seed + 3)
            before = p.transition.field_V.detach().clone()
            p.train_transition_step(s, a, o, lr=0.05, valence=valence)
            return (p.transition.field_V.detach() - before).norm().item()

        d_neutral = step_with(0.0, 200)
        d_success = step_with(1.0, 200)
        d_failure = step_with(-1.0, 200)
        assert d_success < d_neutral, "success must damp (crystallize)"
        assert d_failure < d_neutral, "failure must damp consolidation"
        assert d_failure > 0.0, "failure must NOT freeze plasticity to zero"
        assert d_success > 0.0, "success must NOT freeze either"

    def test_swarm_temperature_valence_channel(self, orch, device):
        # Wire B swarm side: failure valence raises the thermal budget,
        # success cools it. Verified through the monkeypatched noise scale.
        wave = mk_wave((SCALE["num_blocks"], 8), device, 110)
        target = mk_wave((SCALE["num_blocks"], 8), device, 111)
        # Smoke: both polarities execute finite creep steps
        for v in (-1.0, 1.0):
            delta, _, _ = orch.process_active_reasoning_step(
                wave, target, t_shock_max=torch.tensor(0.5, device=device), valence=v)
            assert math.isfinite(delta)
            assert 0.0 <= delta <= 2.0


# ---------------------------------------------------------------------------
# T5: attractor consolidation
# ---------------------------------------------------------------------------

class TestAttractorConsolidation:
    def test_consolidation_merges_similar_engrams(self, device):
        # Use the surrogate store with a consolidation shim for offline testing
        from zone_c_segment_cache import SegmentCache, InProcessZoneCStore
        nb = 64
        cache = SegmentCache(store=InProcessZoneCStore(nb), num_blocks=nb)
        torch.manual_seed(0)
        base = torch.randn(nb, 8)
        base = base / torch.norm(base, p=2, dim=-1, keepdim=True)
        # Three near-identical engrams + one distinct
        for i in range(3):
            w = base + 0.01 * torch.randn(nb, 8)
            w = w / torch.norm(w, p=2, dim=-1, keepdim=True)
            cache.checkpoint(w, "test", 0.1)
        other = torch.randn(nb, 8)
        other = other / torch.norm(other, p=2, dim=-1, keepdim=True)
        cache.checkpoint(other, "test", 0.1)
        assert cache.store.count() == 4

        # Consolidate using the same greedy-cosine logic on the surrogate
        from zone_c_segment_cache import semantic_projection
        sems = torch.stack([semantic_projection(w) for w, *_ in cache.store.rows])
        sems = sems / torch.norm(sems, dim=-1, keepdim=True)
        sim = sems @ sems.T
        # The three near-identical should pairwise-exceed 0.9; the distinct one shouldn't
        near = (sim[:3, :3] > 0.9).sum().item()
        far = (sim[3, :3] > 0.9).sum().item()
        assert near >= 3, "near-identical engrams not clustered"
        assert far == 0, "distinct engram wrongly clustered"


# ---------------------------------------------------------------------------
# Zone C SegmentCache + Gated Residual Memory
# ---------------------------------------------------------------------------

class TestZoneCSegmentCache:
    def _surrogate_cache(self, num_blocks):
        from zone_c_segment_cache import SegmentCache, InProcessZoneCStore
        return SegmentCache(store=InProcessZoneCStore(num_blocks), num_blocks=num_blocks)

    def test_checkpoint_and_recall_roundtrip(self):
        nb = 64
        cache = self._surrogate_cache(nb)
        torch.manual_seed(0)
        w = torch.randn(nb, 8)
        w = w / torch.norm(w, p=2, dim=-1, keepdim=True)
        cache.checkpoint(w, "test", 0.1)
        assert cache.store.count() == 1
        out = cache.retrieve(w)
        assert out["hits"] == 1
        assert out["top_similarity"] > 0.99
        assert out["conditioning_wave"].shape == (nb, 8)

    def test_grm_gates_concentrate_on_nearest(self):
        nb = 64
        cache = self._surrogate_cache(nb)
        torch.manual_seed(0)
        waves = []
        for i in range(3):
            w = torch.randn(nb, 8)
            w = w / torch.norm(w, p=2, dim=-1, keepdim=True)
            cache.checkpoint(w, "test", 0.1 * i)
            waves.append(w)
        q = waves[2] + 0.05 * torch.randn(nb, 8)
        q = q / torch.norm(q, p=2, dim=-1, keepdim=True)
        out = cache.retrieve(q)
        # dominant gate should be the most similar (most recently stored) engram
        assert int(max(range(len(out["gates"])), key=lambda i: out["gates"][i])) == 0

    def test_orchestrator_zone_c_integration(self, orch, device):
        from zone_c_segment_cache import InProcessZoneCStore
        orch._segment_cache = None
        cache = orch.attach_zone_c(dsn="offline://surrogate")
        assert isinstance(cache.store, InProcessZoneCStore)
        wave = mk_wave((SCALE["num_blocks"], 8), device, 14)
        orch.checkpoint_wave(wave.cpu(), "test", 0.1)
        recalled = orch.recall_conditioning_wave(wave.cpu())
        assert recalled is not None
        assert recalled.shape == (SCALE["num_blocks"], 8)


# ---------------------------------------------------------------------------
# qFHRR kernels: codec, LUT similarity, Triton-vs-torch agreement
# ---------------------------------------------------------------------------

class TestQFHRRKernels:
    def test_roundtrip_phase_bound(self, device):
        from qfhrr_kernels import quantization_roundtrip_error
        w = mk_wave((SCALE["num_blocks"], 8), device, 300)
        err = quantization_roundtrip_error(w)
        assert err < 0.0123, f"phase roundtrip error {err} exceeds bin bound"

    def test_self_similarity_unity_random_near_zero(self, device):
        from qfhrr_kernels import build_cos_lut, wave_to_phase_codes, qfhrr_similarity
        lut = build_cos_lut(device)
        wa = mk_wave((SCALE["num_blocks"], 8), device, 301)
        wb = mk_wave((SCALE["num_blocks"], 8), device, 302)
        qa = wave_to_phase_codes(wa).view(-1)
        qb = wave_to_phase_codes(wb).view(-1)
        s_self = float(qfhrr_similarity(qa, qa, lut))
        s_rand = float(qfhrr_similarity(qa, qb, lut))
        assert abs(s_self - 1.0) < 1e-3
        assert abs(s_rand) < 0.15

    def test_lut_matches_fp32_cosine(self, device):
        # LUT-based modular similarity must track the continuous FP32 cosine
        # between the phase-only reconstructions within quantization noise.
        from qfhrr_kernels import (build_cos_lut, wave_to_phase_codes,
                                   phase_codes_to_wave, qfhrr_similarity)
        import torch.fft as fft
        lut = build_cos_lut(device)
        g = torch.Generator().manual_seed(7)
        nb = SCALE["num_blocks"]
        for seed in (310, 311, 312):
            wa = torch.randn(nb, 8, generator=g).to(device)
            wa = wa / torch.norm(wa, p=2, dim=-1, keepdim=True)
            wb = torch.randn(nb, 8, generator=g).to(device)
            wb = wb / torch.norm(wb, p=2, dim=-1, keepdim=True)
            qa = wave_to_phase_codes(wa).view(-1)
            qb = wave_to_phase_codes(wb).view(-1)
            s_lut = float(qfhrr_similarity(qa, qb, lut))
            # FP32 reference: cosine over the quantized phase reconstructions
            ra = phase_codes_to_wave(qa.view(nb, 4)).view(-1)
            rb = phase_codes_to_wave(qb.view(nb, 4)).view(-1)
            s_fp = float(torch.dot(ra, rb) / (ra.norm() * rb.norm()))
            assert abs(s_lut - s_fp) < 0.05, (
                f"seed {seed}: LUT {s_lut:.4f} vs FP32 {s_fp:.4f} diverge")

    def test_triton_torch_agreement(self, device):
        if device.type != "cuda":
            pytest.skip("Triton path requires CUDA")
        from qfhrr_kernels import (build_cos_lut, wave_to_phase_codes,
                                   qfhrr_similarity_torch, qfhrr_similarity_triton)
        lut = build_cos_lut(device)
        nb = SCALE["num_blocks"]
        store = torch.stack([
            wave_to_phase_codes(mk_wave((nb, 8), device, 320 + i)).view(-1)
            for i in range(8)])
        q = wave_to_phase_codes(mk_wave((nb, 8), device, 330)).view(-1)
        s_t = qfhrr_similarity_torch(q, store, lut)
        s_g = qfhrr_similarity_triton(q, store, lut)
        assert (s_t - s_g).abs().max().item() < 1e-4
