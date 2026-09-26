"""Tests for the ZoneACore facade and the FocusedLatentDreamer.

Gates under test:
  * facade: pure delegation; deltas stay DIFFERENTIABLE (the graph must survive)
  * GATE-A hysteresis: entry and wake are different thresholds; wake < entry
  * GATE-B family: the gating metric is the live [0,2] family, not the [0,1] one
  * GATE-C live gradient path: dream steps actually move adapter parameters
  * GATE-D M3-ACT latch: guarded_egress raises until a real receipt is supplied
"""

from __future__ import annotations

import pytest
import torch

from henri_latent_dreamer import (
    ENTRY_THRESHOLD,
    WAKE_THRESHOLD,
    DreamConfig,
    DreamEgressBlocked,
    FocusedLatentDreamer,
)

# the live orchestrator needs num_experts >= 5 (m=4 in the BA skeleton)
D_MODEL, N_BLOCKS, N_EXPERTS, R_RANK = 128, 16, 8, 4


@pytest.fixture(scope="module")
def core():
    from henri_zone_a_core import ZoneACore

    torch.manual_seed(0)
    return ZoneACore(d_model=D_MODEL, num_blocks=N_BLOCKS, num_experts=N_EXPERTS, r_rank=R_RANK)


# ================================================================= A: hysteresis
def test_thresholds_are_the_ratified_pair():
    assert ENTRY_THRESHOLD == 0.35
    assert WAKE_THRESHOLD == 0.0431


def test_config_enforces_hysteresis():
    with pytest.raises(ValueError, match="hysteresis"):
        DreamConfig(entry_threshold=0.0431, wake_threshold=0.0431).validate()
    with pytest.raises(ValueError, match="hysteresis"):
        DreamConfig(entry_threshold=0.1, wake_threshold=0.2).validate()
    assert DreamConfig().validate() is not None


def test_config_rejects_out_of_range_thresholds():
    with pytest.raises(ValueError):
        DreamConfig(entry_threshold=0.0).validate()
    with pytest.raises(ValueError):
        DreamConfig(entry_threshold=3.0).validate()


def test_entry_and_wake_use_their_own_thresholds():
    class _Stub:
        num_blocks = 4

        def plan(self, *a, **k):  # pragma: no cover - not reached
            raise AssertionError

    d = FocusedLatentDreamer(_Stub())
    assert d.should_enter(0.40) is True
    assert d.should_enter(0.30) is False
    assert d.should_wake(0.04, consensus=0.99) is True
    assert d.should_wake(0.04, consensus=0.50) is False   # consensus required
    assert d.should_wake(0.05, consensus=0.99) is False   # stress required


# ================================================================= D: the latch
class _StubCore:
    num_blocks = 4

    def plan(self, tokens, **kwargs):
        return tokens + 1


def test_latch_blocks_egress_before_ratification():
    d = FocusedLatentDreamer(_StubCore())
    assert d.egress_permitted() is False
    with pytest.raises(DreamEgressBlocked, match="DIAGNOSTIC_ONLY"):
        d.guarded_egress(torch.zeros(3))


def test_latch_requires_a_real_m3_act_receipt():
    d = FocusedLatentDreamer(_StubCore())
    with pytest.raises(ValueError, match="M3-ACT"):
        d.ratify_for_egress("")
    with pytest.raises(ValueError, match="M3-ACT"):
        d.ratify_for_egress("an unrelated receipt")
    assert d.egress_permitted() is False


def test_latch_opens_only_with_the_receipt():
    d = FocusedLatentDreamer(_StubCore())
    d.ratify_for_egress("M3-ACT PASS flip_up>=t1 flip_down<=t2")
    assert d.egress_permitted() is True
    assert torch.equal(d.guarded_egress(torch.zeros(3)), torch.ones(3))


def test_adapter_is_identity_at_construction():
    """Default-OFF shape: up is zero-init, so the adapter changes nothing."""
    from henri_latent_dreamer import LowRankDreamAdapter

    a = LowRankDreamAdapter(num_blocks=4, width=8, rank=4)
    x = torch.randn(4, 8)
    assert torch.allclose(a(x), x, atol=1e-12)
    assert a.delta_norm() == 0.0


# =========================================================== facade + GATE-B/C
def test_facade_delegates_and_shapes(core):
    grid = torch.randint(0, 10, (5, 5)).tolist()
    wave = core.encode(grid)
    assert tuple(wave.shape) == (N_BLOCKS, 8)
    assert float(wave.norm()) == pytest.approx(1.0, abs=1e-4)

    cs = core.candidate_set(wave, wave, top_k=4)
    assert len(cs) == 4
    assert tuple(cs.deltas.shape) == (4,)
    # GATE-B: the live family is bounded [0, 2]
    assert float(cs.deltas.abs().max()) <= 2.0 + 1e-5


def test_facade_deltas_preserve_the_graph(core):
    """The deltas must be a DIFFERENTIABLE tensor, not a Python float list.

    This is the defect that broke the first revision: converting to float()
    severed autograd and the dream raise 'does not require grad'.
    """
    grid = torch.randint(0, 10, (5, 5)).tolist()
    wave = core.encode(grid)
    ref = torch.randn(N_BLOCKS, 8, requires_grad=True)
    cs = core.candidate_set(wave, ref, top_k=3)
    assert cs.deltas.requires_grad, "deltas lost the autograd graph"
    g = torch.autograd.grad(cs.deltas.sum(), ref, allow_unused=True)[0]
    assert g is not None and float(g.norm()) > 0.0


def _high_stress_pair(core):
    """Anti-align the reference: delta = 1 - Re(<c,-c>)/(|c||c|) = 2 > entry."""
    grid = torch.randint(0, 10, (5, 5)).tolist()
    psi = core.encode(grid)
    return psi, -psi


def test_gate_c_gradient_path_is_live_with_creep_enabled(core):
    """GATE-C: with the creep flag ON, the SGLD loss must reach the adapter.

    Two independent liveness assertions:
      1. dream() calls autograd.grad(..., allow_unused=False), which RAISES if any
         adapter parameter is absent from the graph - so reaching the movement
         check at all already proves the path exists.
      2. with the flag ON the parameters must also MOVE, proving the update is
         APPLIED, not merely computed.

    The creep is default-OFF (load-bearing test-time weight adaptation), so the
    DEFAULT path deliberately applies no change; that is covered by a separate
    test. Asserting movement on the default path would assert a false premise.
    """
    import os

    os.environ["HENRI_DREAM_CREEP"] = "1"
    try:
        d = FocusedLatentDreamer(core, DreamConfig(max_dream_steps=3, consensus_r=1e-9))
        psi, ref = _high_stress_pair(core)
        before = [p.detach().clone() for p in d.adapter.parameters()]
        res = d.dream(psi, ref, max_steps=2)
        moved = any(
            float((a - b).norm()) > 0.0
            for a, b in zip(list(d.adapter.parameters()), before)
        )
    finally:
        del os.environ["HENRI_DREAM_CREEP"]

    assert res.entered, f"high-stress fixture did not trigger entry (delta={res.initial_delta})"
    assert res.steps_taken > 0
    assert res.creep_enabled is True
    assert moved, "creep enabled but the adapter did not move - GATE-C is dead"
    assert abs(res.adapter_delta_norm) > 0.0, "adapter displacement reported as zero"


def test_default_path_applies_no_weight_change(core):
    """Default-OFF differential: gradients are computed, weights do NOT move."""
    d = FocusedLatentDreamer(core, DreamConfig(max_dream_steps=2, consensus_r=1e-9))
    psi, ref = _high_stress_pair(core)
    before = [p.detach().clone() for p in d.adapter.parameters()]
    res = d.dream(psi, ref, max_steps=2)
    after = list(d.adapter.parameters())
    moved = any(float((a - b).norm()) > 0.0 for a, b in zip(after, before))
    assert res.entered is True
    assert res.creep_enabled is False
    assert moved is False, "default path must not change weights (flag-gated OFF)"


def test_dream_returns_telemetry_never_an_action(core):
    d = FocusedLatentDreamer(core, DreamConfig(max_dream_steps=2))
    psi, ref = _high_stress_pair(core)
    res = d.dream(psi, ref)
    assert not hasattr(res, "action")
    assert not hasattr(res, "tokens")
    assert isinstance(res.adapter_delta_norm, float)
    assert res.entered is True


def test_dream_does_not_enter_on_a_resonant_context(core):
    """A perfectly resonant reference must NOT trigger a dream.

    Measured fact (2026-09-26): candidates are ACTION engrams, so candidate-vs-
    CONTEXT always scores high (delta ~0.98). The resonant regime is reached when
    the reference IS one of the candidates: that candidate then scores ~0.0.
    An earlier revision of this test compared candidate-vs-context and wrongly
    expected resonance - it asserted a false premise, not a code defect.
    """
    d = FocusedLatentDreamer(core, DreamConfig(max_dream_steps=2))
    active = core.encode(torch.randint(0, 16, (2, 6)))
    _, resonant_ref = core.orch.candidate_action_waves(top_k=1)[0]
    res = d.dream(active, resonant_ref)
    assert res.initial_delta <= WAKE_THRESHOLD
    assert res.entered is False
    assert res.steps_taken == 0


def test_creep_is_flag_gated_off_by_default():
    import os

    assert FocusedLatentDreamer.creep_enabled() is False
    os.environ["HENRI_DREAM_CREEP"] = "1"
    try:
        assert FocusedLatentDreamer.creep_enabled() is True
    finally:
        del os.environ["HENRI_DREAM_CREEP"]
