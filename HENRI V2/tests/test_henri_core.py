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
        # Each block matrix should be (near-)unitary after retraction
        prod = torch.matmul(t.transition, t.transition.mH)
        I = torch.eye(8, dtype=torch.complex64, device=device)
        err = (prod - I).abs().max().item()
        assert err < 1e-3
