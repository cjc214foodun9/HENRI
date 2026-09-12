"""Unit and contract tests for the basal-cognition boundary engine bundle.

Spec: HENRI-ARCH-2026-BASAL-COGNITION-AND-BOUNDARY-ENGINEERING

Modules under test:
    basal_boundary_engine.py    Zone A blanket, Cl(3,0) tiling, Kuramoto syncytium
    epsilon_band_gate.py        Defect D1 symmetrical epsilon-band contract
    zone_bc_engram_sync.py      Gap 4 decoupled engrammatic memory sync
    koopman_action_ledger.py    Gap 3 action-conditioned transition identification
    unified_henri_vla_engine.py UnifiedHENRIVLAEngine composition + gates

These tests are fail-closed: a defect must produce a FAIL, not a warning. Each
test names the defect or gap it covers so a failure maps to a mandate item.

Device: CPU is used for the FAST contract tests. The full 65536-dimension gate
battery lives in unified_henri_vla_engine.verify_engine() and is exercised on the
CUDA target by experiments/verification/verify_basal_boundary_gates.py.
"""

from __future__ import annotations

import math
import os
import sys

import pytest
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from basal_boundary_engine import (
    CLIFFORD_REVERSION_MASK,
    SPEC_CLIFFORD_BLOCK_SIZE,
    SPEC_DIMENSION_D,
    SPEC_MEASURED_MIN_LEAKAGE_FRACTION,
    SPEC_NUM_TILES,
    SPEC_R_GATE,
    SPEC_SHUTTER_HZ,
    SPEC_SLOT_SECONDS,
    BasalBoundaryError,
    DynamicMarkovBlanket,
    EvanescentKuramotoSyncytium,
    ShutterClock,
    clifford_bivector_tiles,
    evanescent_kernel,
    leakage_fraction,
    recommended_leakage_length,
    tile_descriptor,
    tile_grade_decompose,
)
from epsilon_band_gate import (
    BAND_ASYMMETRIC,
    BAND_SYMMETRIC,
    evaluate_band,
    one_sided_band,
    probe_gate,
    two_sided_band,
    zlib_warmup_n_min,
)
from koopman_action_ledger import (
    STATUS_ILL_CONDITIONED,
    STATUS_INSUFFICIENT,
    STATUS_NO_PAYLOADS,
    STATUS_OK,
    LedgerCorpus,
    identify_action_conditioned,
    quadratic_observables,
)
from zone_bc_engram_sync import (
    STATUS_QUEUED,
    STATUS_THROTTLED,
    DecoupledEngramSync,
    QueueOverflowError,
    make_envelope,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SMALL_TILES = 64
SMALL_CHANNELS = 8


class _SlowStore:
    """A store whose commit path is deliberately expensive."""

    def __init__(self, latency_s: float):
        self.latency_s = latency_s
        self.committed = []

    def commit(self, env) -> str:
        t0 = __import__("time").perf_counter()
        while __import__("time").perf_counter() - t0 < self.latency_s:
            pass
        self.committed.append(env)
        return env.engram_id


# ---------------------------------------------------------------------------
# Mandate 2 - Zone A temporal boundary (the 20 kHz time-slot shutter)
# ---------------------------------------------------------------------------

class TestTemporalBoundary:
    def test_slot_is_fifty_microseconds(self):
        """G1: 20 kHz => exactly 5e-5 s = 50 us."""
        c = ShutterClock()
        assert c.slot_hz == SPEC_SHUTTER_HZ == 20000.0
        assert c.slot_seconds == pytest.approx(1.0 / 20000.0)
        assert c.slot_seconds * 1e6 == pytest.approx(50.0)
        assert SPEC_SLOT_SECONDS == pytest.approx(c.slot_seconds)

    def test_slot_index_is_monotonic(self):
        c = ShutterClock()
        idx = [c.advance().slot_index for _ in range(5)]
        assert idx == [0, 1, 2, 3, 4]

    def test_ingress_is_throttled_and_counted_not_silently_dropped(self):
        """Sensory flooding must be RECORDED. A silent drop is invisible."""
        c = ShutterClock(max_ingress_per_slot=8)
        s = c.advance(ingress_count=100)
        assert s.ingress_admitted == 8
        assert s.ingress_throttled == 92
        assert s.isolated is True

    def test_non_monotonic_wall_clock_is_rejected(self):
        """Isolation budget cannot be trusted if slots are not in time order."""
        c = ShutterClock()
        c.advance(wall_now=10.0)
        with pytest.raises(BasalBoundaryError):
            c.advance(wall_now=5.0)

    def test_invalid_construction_is_rejected(self):
        with pytest.raises(BasalBoundaryError):
            ShutterClock(slot_hz=0.0)
        with pytest.raises(BasalBoundaryError):
            ShutterClock(ticks_per_slot=0)
        with pytest.raises(BasalBoundaryError):
            ShutterClock(max_ingress_per_slot=0)


# ---------------------------------------------------------------------------
# Mandate 2 - Zone A spatial boundary (Cl(3,0) bivector tiling, M = 8192)
# ---------------------------------------------------------------------------

class TestSpatialBoundary:
    def test_tile_arithmetic_is_exact(self):
        """M = D / block_size must be exact: a dropped remainder is silent."""
        assert SPEC_DIMENSION_D // SPEC_CLIFFORD_BLOCK_SIZE == SPEC_NUM_TILES
        assert SPEC_NUM_TILES == 8192
        assert SPEC_NUM_TILES * SPEC_CLIFFORD_BLOCK_SIZE == SPEC_DIMENSION_D

    def test_tiling_shape_and_locality(self):
        w = torch.arange(SPEC_DIMENSION_D, dtype=torch.float32)
        t = clifford_bivector_tiles(w)
        assert t.shape == (SPEC_NUM_TILES, SPEC_CLIFFORD_BLOCK_SIZE)
        # Row k holds exactly the k-th contiguous block: no cross-tile mixing.
        assert torch.equal(t[0], w[0:8])
        assert torch.equal(t[1], w[8:16])
        assert torch.equal(t[-1], w[-8:])

    def test_wrong_length_fails_closed(self):
        with pytest.raises(BasalBoundaryError):
            clifford_bivector_tiles(torch.zeros(SPEC_DIMENSION_D - 1))

    def test_reversion_mask_matches_live_clifford_kernel(self):
        """The bivector/pseudoscalar reversion sign must match the live kernel.

        ProductCliffordAlgebra3D.reversion_mask is (1,1,1,1,-1,-1,-1,-1). If
        these drift apart, the two Cl(3,0) implementations disagree.
        """
        assert CLIFFORD_REVERSION_MASK == (1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0)
        try:
            from product_clifford_product_kernel import ProductCliffordAlgebra3D
        except Exception:
            pytest.skip("live Clifford kernel not importable in this environment")
        live = ProductCliffordAlgebra3D(num_blocks=2).reversion_mask
        assert list(live.tolist()) == list(CLIFFORD_REVERSION_MASK), (
            "local Cl(3,0) reversion mask diverged from the live kernel"
        )

    def test_grade_decomposition_partitions_the_tile(self):
        t = clifford_bivector_tiles(torch.arange(SPEC_DIMENSION_D, dtype=torch.float32))
        g = tile_grade_decompose(t)
        assert g["scalar"].shape == (SPEC_NUM_TILES, 1)
        assert g["vector"].shape == (SPEC_NUM_TILES, 3)
        assert g["bivector"].shape == (SPEC_NUM_TILES, 3)
        assert g["pseudoscalar"].shape == (SPEC_NUM_TILES, 1)
        # 1 + 3 + 3 + 1 == 8: the grades partition the tile exactly.
        assert 1 + 3 + 3 + 1 == SPEC_CLIFFORD_BLOCK_SIZE

    def test_metric_locality_witness_discriminates(self):
        """THE discriminating test for the metric-locality claim.

        A fixture can put signal on the axis the operator PRESERVES, which makes
        the gate unfailable. This test uses the minimal discriminating case: a
        descriptor must DETECT a tile permutation, while mean pooling must NOT.
        """
        n = SMALL_TILES
        torch.manual_seed(0)
        w = torch.randn(n * SMALL_CHANNELS)
        t = clifford_bivector_tiles(w, n, SMALL_CHANNELS)
        perm = torch.roll(torch.arange(n), shifts=n // 2)

        d_a, d_b = tile_descriptor(t).reshape(-1), tile_descriptor(t[perm]).reshape(-1)
        cos_desc = float(
            torch.dot(d_a, d_b) / (torch.linalg.vector_norm(d_a) * torch.linalg.vector_norm(d_b))
        )
        m_a, m_b = t.mean(dim=0), t[perm].mean(dim=0)
        cos_pool = float(
            torch.dot(m_a, m_b) / (torch.linalg.vector_norm(m_a) * torch.linalg.vector_norm(m_b))
        )
        assert cos_desc < 0.95, "descriptor failed to see a tile permutation"
        assert cos_pool == pytest.approx(1.0, abs=1e-6), "mean pooling control broken"

    def test_mean_pooling_control_is_permutation_invariant(self):
        """The control: summation is permutation-invariant, so it CANNOT see it."""
        n = SMALL_TILES
        torch.manual_seed(1)
        w = torch.randn(n * SMALL_CHANNELS)
        t = clifford_bivector_tiles(w, n, SMALL_CHANNELS)
        perm = torch.roll(torch.arange(n), shifts=7)
        assert torch.allclose(t.mean(dim=0), t[perm].mean(dim=0), atol=1e-6)


# ---------------------------------------------------------------------------
# Mandate 1 - Zone B acoustic boundary (evanescent leakage -> Kuramoto)
# ---------------------------------------------------------------------------

class TestAcousticBoundary:
    def test_kernel_is_row_sum_one_and_local(self):
        k = evanescent_kernel(1024, 16.0)
        assert float(k.sum()) == pytest.approx(1.0, abs=1e-6)
        # An exponential kernel concentrates mass near zero distance; a constant
        # (all-to-all) kernel would not.
        assert float(k[0]) > float(k[512])
        assert float(k[0]) > 1.0 / 1024.0

    def test_kernel_rejects_degenerate_arguments(self):
        with pytest.raises(BasalBoundaryError):
            evanescent_kernel(1, 1.0)
        with pytest.raises(BasalBoundaryError):
            evanescent_kernel(16, 0.0)

    def test_leakage_is_scale_free_by_ring_size(self):
        """The threshold is a FRACTION of the ring, not an absolute count."""
        assert leakage_fraction(8192, 504.0) == pytest.approx(504.0 / 8192)
        small = recommended_leakage_length(256, margin=3.0)
        large = recommended_leakage_length(8192, margin=3.0)
        assert small / 256 == pytest.approx(large / 8192)
        assert small == pytest.approx(256 * SPEC_MEASURED_MIN_LEAKAGE_FRACTION * 3.0)

    def test_margin_below_one_is_rejected(self):
        """A margin < 1 places the working point below the measured threshold."""
        with pytest.raises(BasalBoundaryError):
            recommended_leakage_length(8192, margin=0.5)

    def test_order_parameter_is_bounded_and_exact_on_sync(self):
        syn = EvanescentKuramotoSyncytium(num_channels=64, coupling_K=8.0, decay_length=64.0)
        syn.phases = torch.zeros(64)
        assert syn.order_parameter(syn.phases) == pytest.approx(1.0, abs=1e-6)
        syn.phases = torch.linspace(-math.pi, math.pi, 65)[:64]
        assert syn.order_parameter(syn.phases) < 0.1

    def test_all_to_all_limit_reaches_full_sync(self):
        """CONTROL that proves the integrator is correct.

        With zero natural frequencies and an effectively global kernel, K > K_c
        must drive r -> 1. If this fails, the integrator (not the physics) is
        wrong, and every local-kernel result is untrustworthy.
        """
        syn = EvanescentKuramotoSyncytium(
            num_channels=256, coupling_K=32.0, decay_length=1e5, dt=0.01,
            natural_frequency_scale=0.0, noise_temperature=0.0, seed=0,
        )
        syn.random_phases(seed=0)
        out = syn.relax(500)
        assert out["finite"] is True
        assert float(out["r"]) > 0.99, "all-to-all control failed: integrator suspect"

    def test_local_only_coupling_does_not_globally_lock(self):
        """MEASURED PROPERTY, encoded as a regression guard.

        On a 1-D ring an exponentially local kernel yields dense LOCAL order
        (r_local -> ~1) without global locking. This is the finding that made
        the mandate's r >= 0.93 unreachable at the spec's implied local leakage.
        If a future change makes local coupling lock globally, that is a real
        change in behaviour and this test must be revisited deliberately.
        """
        n = 1024
        syn = EvanescentKuramotoSyncytium(
            num_channels=n, coupling_K=32.0, decay_length=2.0, dt=0.01,
            natural_frequency_scale=0.0, noise_temperature=0.0, seed=0,
        )
        syn.random_phases(seed=0)
        out = syn.relax(1000)
        assert float(out["r"]) < SPEC_R_GATE
        assert float(out["r_local_mean"]) > 0.9, (
            "expected dense local order without global order"
        )

    def test_configured_working_point_clears_the_gate(self):
        """The measured operating point must actually clear r >= 0.93."""
        n = 1024
        leak = recommended_leakage_length(n, margin=3.0)
        syn = EvanescentKuramotoSyncytium(
            num_channels=n, coupling_K=2.45, decay_length=leak, dt=0.01,
            natural_frequency_scale=0.0, noise_temperature=0.0, seed=0,
        )
        syn.random_phases(seed=0)
        out = syn.relax(1024)
        assert float(out["r"]) >= SPEC_R_GATE

    def test_subcritical_control_stays_below_the_gate(self):
        """A gate that passes here is measuring something other than locking."""
        syn = EvanescentKuramotoSyncytium(
            num_channels=1024, coupling_K=2.45, decay_length=8.0, dt=0.01,
            natural_frequency_scale=1.0, noise_temperature=0.0, seed=0,
        )
        r = syn.subcritical_control(steps=512, K=2.45)
        assert r < SPEC_R_GATE

    def test_theoretical_k_c_matches_the_uniform_distribution(self):
        syn = EvanescentKuramotoSyncytium(num_channels=64)
        assert syn.theoretical_k_c() == pytest.approx(4.0)
        # The spec default K=2.45 sits BELOW K_c = 4; the measured lock at that
        # K depends on the wide-leakage working point, not on K alone.
        assert 2.45 < syn.theoretical_k_c()


class TestSagnacInterpretant:
    def test_identical_waves_give_zero_delta(self):
        """D-SAGNAC: the canonical metric gives 0 for identical unit waves."""
        a = torch.nn.functional.normalize(torch.randn(4096), p=2, dim=0)
        assert EvanescentKuramotoSyncytium.sagnac_delta(a, a) == pytest.approx(0.0, abs=1e-6)

    def test_opposite_waves_give_one(self):
        a = torch.nn.functional.normalize(torch.randn(4096), p=2, dim=0)
        assert EvanescentKuramotoSyncytium.sagnac_delta(a, -a) == pytest.approx(1.0, abs=1e-6)

    def test_spec_form_would_veto_every_valid_candidate(self):
        """The defect, demonstrated: 1 - <a,b>/D is ~1 for unit-norm waves.

        This is the FALSIFIED false-veto class documented live in
        arc_sagnac_veto.py (2026-08-12). The canonical metric fixes it.
        """
        d = SPEC_DIMENSION_D
        a = torch.nn.functional.normalize(torch.randn(d), p=2, dim=0)
        spec_delta = float(1.0 - torch.dot(a, a).item() / d)
        canonical = EvanescentKuramotoSyncytium.sagnac_delta(a, a)
        assert spec_delta > 0.9999, "spec form should be ~1 for identical unit waves"
        assert canonical < 1e-6, "canonical metric must be ~0 for identical waves"

    def test_veto_threshold_semantics(self):
        s = EvanescentKuramotoSyncytium
        assert s.is_vetoed(0.40, 0.35) is True
        assert s.is_vetoed(0.30, 0.35) is False

    def test_degenerate_input_is_vetoed_not_passed(self):
        """No reference => delta 1.0 => veto. Unknown must never pass."""
        assert EvanescentKuramotoSyncytium.sagnac_delta(torch.zeros(8), torch.zeros(8)) == 1.0


# ---------------------------------------------------------------------------
# Mandate 2 - the blanket as a whole
# ---------------------------------------------------------------------------

class TestDynamicMarkovBlanket:
    def _blanket(self, **kw):
        base = dict(num_tiles=SMALL_TILES, channels=SMALL_CHANNELS,
                    coupling_K=8.0, decay_length=64.0, ticks_per_slot=16, seed=0)
        base.update(kw)
        return DynamicMarkovBlanket(**base)

    def test_step_is_typed_finite_and_bounded(self):
        b = self._blanket()
        torch.manual_seed(0)
        w = torch.randn(SMALL_TILES * SMALL_CHANNELS)
        st = b.forward(w, None, ingress_count=1, slots=1)
        assert st.finite is True
        assert 0.0 <= st.order_parameter_r <= 1.0
        assert 0.0 <= st.sagnac_delta <= 1.0
        assert st.tiles == SMALL_TILES and st.channels == SMALL_CHANNELS
        assert st.relaxed_steps == 16
        assert st.boundary_violation is True, "no engram reference => veto"

    def test_absent_reference_vetoes_and_never_crystallizes(self):
        b = self._blanket()
        w = torch.randn(SMALL_TILES * SMALL_CHANNELS)
        st = b.forward(w, None, slots=1)
        assert st.boundary_violation is True
        assert st.is_crystallized is False

    def test_locality_witness_reported_alongside_its_control(self):
        b = self._blanket()
        w = torch.randn(SMALL_TILES * SMALL_CHANNELS)
        st = b.forward(w, None, slots=1)
        assert st.telemetry["mean_pooling_locality_cosine"] == pytest.approx(1.0, abs=1e-6)

    def test_phase_initialization_preserves_tile_structure(self):
        syn = EvanescentKuramotoSyncytium(num_channels=SMALL_TILES)
        torch.manual_seed(0)
        ph = syn.seed_phases_from_wave(torch.randn(SMALL_TILES * SMALL_CHANNELS))
        assert ph.shape == (SMALL_TILES,)
        assert bool(torch.isfinite(ph).all())

    def test_relax_before_seeding_fails_closed(self):
        syn = EvanescentKuramotoSyncytium(num_channels=SMALL_TILES)
        with pytest.raises(BasalBoundaryError):
            syn.relax(4)


# ---------------------------------------------------------------------------
# Gap - defect D1: the symmetrical epsilon-band gate
# ---------------------------------------------------------------------------

class TestEpsilonBandGate:
    def test_precondition_is_checked_before_scoring(self):
        """Order matters: scoring before the pre-condition IS the D1 defect."""
        assert two_sided_band(0.5, 0.02, n=10, n_min=20000) == "BLOCKED_SUB_MINIMUM_N"

    def test_two_sided_band_classifies_both_signs_as_control(self):
        assert two_sided_band(-0.0031, 0.02, 20000, 20000) == "ACCEPT_CONTROL_BAND"
        assert two_sided_band(0.0180, 0.02, 20000, 20000) == "ACCEPT_CONTROL_BAND"
        assert two_sided_band(1.82, 0.02, 20000, 20000) == "ACCEPT_STRUCTURED"

    def test_two_sided_gate_is_symmetric(self):
        streams = [
            ("dead", "control", 20000, -0.0031),
            ("noise", "control", 20000, 0.0180),
            ("struct", "signal", 20000, 1.82),
        ]
        rep = probe_gate(lambda g, n: two_sided_band(g, 0.02, n, 20000), streams, 0.02, 20000)
        assert rep.status == BAND_SYMMETRIC
        assert rep.ok is True
        assert rep.controls_inside == 2 and rep.signals_outside == 1

    def test_one_sided_gate_is_detected_as_asymmetric(self):
        """THE regression guard for defect D1.

        The FALSIFIED one-sided rule rejects its own negative controls (their
        statistic approaches zero from below). The probe must catch that, or the
        probe is not discriminating.
        """
        streams = [
            ("dead", "control", 20000, -0.0031),
            ("noise", "control", 20000, 0.0180),
            ("struct", "signal", 20000, 1.82),
        ]
        rep = probe_gate(lambda g, n: one_sided_band(g, 0.02, n, 20000), streams, 0.02, 20000)
        assert rep.status == BAND_ASYMMETRIC
        assert "dead" in rep.rejected_controls
        assert rep.ok is False

    def test_underdetermined_without_both_roles(self):
        only_controls = [("dead", "control", 20000, 0.0)]
        rep = evaluate_band(
            [__import__("epsilon_band_gate").StreamObservation("dead", "control", 20000, 0.0, "ACCEPT_CONTROL_BAND")],
            eps=0.02, n_min=20000,
        )
        assert rep.status != BAND_SYMMETRIC

    def test_admitted_signal_is_asymmetric(self):
        """A gate that lets a structured stream into the control band is broken."""
        obs = [
            __import__("epsilon_band_gate").StreamObservation("dead", "control", 20000, 0.0, "ACCEPT_CONTROL_BAND"),
            __import__("epsilon_band_gate").StreamObservation("struct", "signal", 20000, 0.005, "ACCEPT_CONTROL_BAND"),
        ]
        rep = evaluate_band(obs, eps=0.02, n_min=20000)
        assert rep.status == BAND_ASYMMETRIC
        assert "struct" in rep.admitted_signals

    def test_n_min_is_derived_from_overhead_and_band(self):
        """The pre-condition is DERIVED, not chosen."""
        assert zlib_warmup_n_min(0.02) == 20000
        assert zlib_warmup_n_min(0.04) == 10000
        with pytest.raises(ValueError):
            zlib_warmup_n_min(0.0)

    def test_invalid_eps_rejected(self):
        with pytest.raises(ValueError):
            evaluate_band([], eps=0.0, n_min=1)


# ---------------------------------------------------------------------------
# Gap 4 - decoupled engrammatic memory synchronization
# ---------------------------------------------------------------------------

class TestDecoupledEngramSync:
    def test_publish_does_not_block_on_store(self):
        """THE decoupling claim: publish latency independent of store latency."""
        slow = _SlowStore(latency_s=0.004)          # 4 ms per commit
        sync = DecoupledEngramSync(slow, capacity=64, enabled=True)
        w = torch.randn(8, 8)
        t0 = __import__("time").perf_counter()
        for k in range(32):
            r = sync.publish(make_envelope(w, "d", 0.1 * k))
            assert r["status"] == STATUS_QUEUED
        elapsed = __import__("time").perf_counter() - t0
        assert len(slow.committed) == 0, "publish must NOT touch the store"
        assert elapsed < 0.004 * 8, f"publish blocked: {elapsed:.4f}s"
        assert sync.depth() == 32

    def test_drain_commits_everything_and_verifies_digests(self):
        slow = _SlowStore(latency_s=0.0)
        sync = DecoupledEngramSync(slow, capacity=64, enabled=True)
        w = torch.randn(8, 8)
        for k in range(16):
            sync.publish(make_envelope(w, "d", 0.0))
        d = sync.drain()
        assert d["committed"] == 16 and d["failed"] == 0
        assert len(slow.committed) == 16
        assert all(env.verify() for env in slow.committed)

    def test_envelope_owns_its_bytes(self):
        """A queued envelope must not alias Zone B working memory."""
        w = torch.randn(4, 8)
        env = make_envelope(w, "d", 0.0)
        before = env.payload
        w.add_(1000.0)                       # mutate the source wave in place
        assert env.payload == before
        assert env.verify() is True

    def test_overflow_is_counted_not_silent(self):
        sync = DecoupledEngramSync(_SlowStore(0.0), capacity=4, enabled=True)
        w = torch.randn(2, 8)
        for _ in range(10):
            sync.publish(make_envelope(w, "d", 0.0))
        assert sync.published == 4
        assert sync.throttled == 6, "overflow must be accounted"
        assert sync.telemetry()["throttled"] == 6

    def test_fail_closed_mode_raises_on_overflow(self):
        sync = DecoupledEngramSync(
            _SlowStore(0.0), capacity=2, fail_closed_on_overflow=True, enabled=True
        )
        w = torch.randn(2, 8)
        sync.publish(make_envelope(w, "d", 0.0))
        sync.publish(make_envelope(w, "d", 0.0))
        with pytest.raises(QueueOverflowError):
            sync.publish(make_envelope(w, "d", 0.0))

    def test_disabled_sync_is_inert(self):
        sync = DecoupledEngramSync(_SlowStore(0.0), capacity=4, enabled=False)
        r = sync.publish(make_envelope(torch.randn(2, 8), "d", 0.0))
        assert r["status"] == "SYNC_DISABLED"
        assert sync.depth() == 0

    def test_lag_is_reported(self):
        sync = DecoupledEngramSync(_SlowStore(0.0), capacity=4, enabled=True)
        sync.publish(make_envelope(torch.randn(2, 8), "d", 0.0))
        assert sync.lag_ms() is not None and sync.lag_ms() >= 0.0
        sync.drain()
        assert sync.lag_ms() == 0.0

    def test_digest_round_trips_through_a_live_store_contract(self):
        """The commit path must reproduce the recorded digest."""
        committed = {}

        class _Store:
            def write_engram(self, wave, domain, sagnac_stress):
                import hashlib
                b = wave.detach().cpu().contiguous().to(torch.float32).numpy().tobytes()
                committed["digest"] = hashlib.sha256(b).hexdigest()
                return "ok"

        sync = DecoupledEngramSync(_Store(), capacity=4, enabled=True)
        w = torch.randn(8, 8)
        env = make_envelope(w, "d", 0.0)
        sync.publish(env)
        sync.drain()
        assert committed["digest"] == env.digest


# ---------------------------------------------------------------------------
# Gap 3 - action-conditioned Koopman transition ledger
# ---------------------------------------------------------------------------

class TestKoopmanActionLedger:
    def test_digest_only_corpus_is_blocked_not_silently_scored(self):
        """Stage 4: digest-only rows recover 0 triples. Must be BLOCKED."""
        empty = LedgerCorpus(
            n_rows=100, n_usable=0, n_unusable=100, n_digest_mismatch=0,
            X=torch.zeros(0, 0).numpy(), A=__import__("numpy").array([], dtype=int),
            Y=torch.zeros(0, 0).numpy(), episode_ids=[], steps=[],
        )
        res = identify_action_conditioned(empty, min_pairs=10)
        assert res.status == STATUS_NO_PAYLOADS
        assert res.gate_pass is False
        assert "no (state, action, next_state) triple" in res.reason

    def test_insufficient_pairs_is_blocked(self):
        """The n=1500 failure was ill-conditioning, not estimator error."""
        import numpy as np
        n, d = 200, 4
        rng = np.random.default_rng(0)
        corpus = LedgerCorpus(
            n_rows=n, n_usable=n, n_unusable=0, n_digest_mismatch=0,
            X=rng.standard_normal((n, d)), A=rng.integers(0, 2, n),
            Y=rng.standard_normal((n, d)),
            episode_ids=[f"ep{i%20}" for i in range(n)], steps=list(range(n)),
        )
        res = identify_action_conditioned(corpus, min_pairs=12000)
        assert res.status == STATUS_INSUFFICIENT
        assert res.gate_pass is False

    def test_well_conditioned_linear_system_is_identified(self):
        """Linear dynamics: a quadratic dictionary represents them exactly."""
        import numpy as np
        d, n_ep, steps = 4, 60, 40
        rng = np.random.default_rng(11)
        ops = []
        for _ in range(2):
            M = rng.standard_normal((d, d)) / math.sqrt(d)
            ops.append(M * (0.9 / np.abs(np.linalg.eigvals(M)).max()))
        X, A, Y, eps, st = [], [], [], [], []
        for e in range(n_ep):
            x = rng.standard_normal(d)
            for s in range(steps):
                a = int(rng.integers(0, 2))
                xn = ops[a] @ x + 0.02 * rng.standard_normal(d)
                X.append(x); A.append(a); Y.append(xn); eps.append(f"ep{e}"); st.append(s)
                x = xn
        corpus = LedgerCorpus(
            n_rows=len(X), n_usable=len(X), n_unusable=0, n_digest_mismatch=0,
            X=np.asarray(X), A=np.asarray(A, dtype=np.int64), Y=np.asarray(Y),
            episode_ids=eps, steps=st,
        )
        res = identify_action_conditioned(
            corpus, n_actions=2, n_test_episodes=10, min_pairs=100, max_kappa=1e9,
            gate=0.15, ground_truth=ops,
        )
        assert res.status == STATUS_OK
        # The fitted operator must match the truth far better than the truth's
        # own residual noise, given clean conditioning.
        assert res.delta_frobenius is not None and res.delta_frobenius < 0.15
        assert res.delta_fitted_over_truth is not None
        assert res.delta_fitted_over_truth < 2.0, "estimator diverged from ground truth"

    def test_ill_conditioning_is_reported_not_hidden(self):
        import numpy as np
        d, n = 6, 400
        rng = np.random.default_rng(3)
        # A near-degenerate design: all rows nearly equal -> huge kappa.
        X = np.tile(rng.standard_normal(d), (n, 1)) + 1e-12 * rng.standard_normal((n, d))
        corpus = LedgerCorpus(
            n_rows=n, n_usable=n, n_unusable=0, n_digest_mismatch=0,
            X=X, A=np.zeros(n, dtype=int), Y=rng.standard_normal((n, d)),
            episode_ids=[f"ep{i%30}" for i in range(n)], steps=list(range(n)),
        )
        res = identify_action_conditioned(
            corpus, n_actions=1, n_test_episodes=5, min_pairs=100, max_kappa=5.0
        )
        assert res.status == STATUS_ILL_CONDITIONED
        assert res.kappa_max is not None and res.kappa_max > 5.0
        assert res.gate_pass is False

    def test_primary_metric_is_labelled_and_secondary_retained(self):
        """Both metrics are reported: the audit showed they disagree."""
        import numpy as np
        d, n = 4, 2400
        rng = np.random.default_rng(5)
        M = rng.standard_normal((d, d)) / math.sqrt(d)
        M *= 0.9 / np.abs(np.linalg.eigvals(M)).max()
        X, Y, eps, st = [], [], [], []
        for e in range(30):
            x = rng.standard_normal(d)
            for s in range(80):
                xn = M @ x + 0.02 * rng.standard_normal(d)
                X.append(x); Y.append(xn); eps.append(f"ep{e}"); st.append(s); x = xn
        corpus = LedgerCorpus(
            n_rows=len(X), n_usable=len(X), n_unusable=0, n_digest_mismatch=0,
            X=np.asarray(X), A=np.zeros(len(X), dtype=int), Y=np.asarray(Y),
            episode_ids=eps, steps=st,
        )
        res = identify_action_conditioned(
            corpus, n_actions=1, n_test_episodes=5, min_pairs=100, max_kappa=1e9, gate=0.15
        )
        assert res.status == STATUS_OK
        assert res.metric == "noise_normalized_frobenius"
        assert res.delta_frobenius is not None
        assert res.delta_per_sample is not None

    def test_quadratic_dictionary_shape(self):
        import numpy as np
        x = np.zeros(4)
        psi = quadratic_observables(x)
        assert psi.shape == (1 + 4 + 4 * 5 // 2,)

    def test_prediction_requires_a_fitted_action(self):
        import numpy as np
        from koopman_action_ledger import predict_action_conditioned
        corpus = LedgerCorpus(
            n_rows=0, n_usable=0, n_unusable=0, n_digest_mismatch=0,
            X=torch.zeros(0, 0).numpy(), A=np.array([], dtype=int),
            Y=torch.zeros(0, 0).numpy(), episode_ids=[], steps=[],
        )
        res = identify_action_conditioned(corpus, min_pairs=10)
        assert predict_action_conditioned(res, np.zeros(4), 0) is None


# ---------------------------------------------------------------------------
# Mandate 4 - the unified engine
# ---------------------------------------------------------------------------

class TestUnifiedEngine:
    def _engine(self, **cfg_kw):
        from unified_henri_vla_engine import (
            MarkovBlanketSpec,
            UnifiedHENRIVLAConfig,
            UnifiedHENRIVLAEngine,
        )
        # D=1024 is the smallest valid tiling (1024/8 = 128 tiles), which keeps
        # these contract tests fast while exercising the real code path. The
        # production value is D=65536 -> 8192 tiles.
        spec = MarkovBlanketSpec(
            dimension_D=1024, clifford_block_size=8,
            kuramoto_coupling_K=8.0, ticks_per_slot=16,
            auto_leakage_from_ring=True, evanescent_leakage_margin=3.0,
            lock_horizon_steps=2048,
        )
        spec = spec.model_copy(update=cfg_kw)
        cfg = UnifiedHENRIVLAConfig(blanket=spec)
        return UnifiedHENRIVLAEngine(cfg)

    def test_factory_is_default_off(self):
        from unified_henri_vla_engine import get_unified_basal_engine
        old = os.environ.pop("HENRI_BASAL_ENGINE", None)
        try:
            assert get_unified_basal_engine() is None
            os.environ["HENRI_BASAL_ENGINE"] = "1"
            assert get_unified_basal_engine() is not None
        finally:
            if old is None:
                os.environ.pop("HENRI_BASAL_ENGINE", None)
            else:
                os.environ["HENRI_BASAL_ENGINE"] = old

    def test_config_rejects_inexact_tiling(self):
        """A non-divisible tiling would silently drop a remainder."""
        from unified_henri_vla_engine import MarkovBlanketSpec
        with pytest.raises(Exception):
            MarkovBlanketSpec(dimension_D=1000, clifford_block_size=8)

    def test_config_is_frozen(self):
        from unified_henri_vla_engine import MarkovBlanketSpec
        s = MarkovBlanketSpec()
        with pytest.raises(Exception):
            s.dimension_D = 999

    def test_spec_tiling_drift_is_rejected(self):
        """D=65536 must tile to exactly 8192; a drift must fail loudly."""
        from unified_henri_vla_engine import MarkovBlanketSpec
        with pytest.raises(Exception):
            MarkovBlanketSpec(dimension_D=65536, clifford_block_size=4)

    def test_leakage_resolves_scale_free(self):
        eng = self._engine()
        n_tiles = eng.cfg.blanket.num_tiles
        assert eng.leakage_length == pytest.approx(
            recommended_leakage_length(n_tiles, margin=3.0)
        )
        assert eng.leakage_fraction_of_ring == pytest.approx(
            SPEC_MEASURED_MIN_LEAKAGE_FRACTION * 3.0, rel=1e-9
        )

    def test_bind_reports_the_norm_change_it_actually_makes(self):
        """D-ENGINE-1: the spec calls this norm-preserving; it is not."""
        eng = self._engine()
        torch.manual_seed(0)
        a = torch.randn(512)
        b = torch.randn(512)
        r = eng.circular_convolution_bind(a, b)
        assert r.normalized is True
        assert r.post_norm == pytest.approx(1.0, abs=1e-4)
        # The pre-normalize norm differs: circular convolution does not conserve
        # norm in the spatial domain. A claim of conservation would be false.
        assert r.post_norm_before_normalize != pytest.approx(r.pre_norm, rel=1e-3)
        assert math.isfinite(r.norm_change_ratio)

    def test_unlabeled_bind_does_not_normalize(self):
        eng = self._engine()
        r = eng.circular_convolution_bind(torch.randn(64), torch.randn(64),
                                         normalize_after_bind=False)
        assert r.normalized is False

    def test_lexical_snap_reports_measured_entropy(self):
        eng = self._engine()
        torch.manual_seed(0)
        engrams = torch.nn.functional.normalize(torch.randn(32, eng.egress_dim), p=2, dim=-1)
        eng.register_engrams(engrams)
        out = eng.lexical_snap(engrams[3], top_k=1)
        assert out["status"] == "SNAPPED"
        assert out["indices"][0] == 3, "an exact engram must snap to itself"
        assert out["logit_entropy_bits"] < 1.2, "exact match must snap sharply"

    def test_snap_without_codebook_fails_closed(self):
        eng = self._engine()
        out = eng.lexical_snap(torch.randn(eng.egress_dim))
        assert out["status"] == "REJECTED"

    def test_step_is_causal_and_leaves_zone_c_inert_when_disabled(self):
        eng = self._engine()
        w = torch.randn(eng.egress_dim)
        res = eng.step(w, engram_reference=None, slots=1)
        assert res.status == "OK"
        assert res.blanket["boundary_violation"] is True
        assert res.zonec["status"] == "SYNC_DISABLED"

    def test_bounded_light_cone_never_exceeds_the_unbounded_index(self):
        """D-ENGINE-2: r/delta is unbounded; r*(1-delta) is not."""
        eng = self._engine()
        res = eng.step(torch.randn(eng.egress_dim), slots=1)
        assert 0.0 <= res.light_cone_bounded <= 1.0
        assert res.light_cone_unbounded_index >= 0.0

    def test_nonfinite_relaxation_fails_closed(self):
        eng = self._engine()
        eng.blanket.syncytium.coupling_K = float("inf")
        res = eng.step(torch.randn(eng.egress_dim), slots=1)
        assert res.status == "FAIL_CLOSED_NONFINITE"

    def test_describe_exposes_every_mandated_constant(self):
        d = self._engine().describe()
        assert d["zone_a"]["tiles"] * d["zone_a"]["channels"] == d["zone_a"]["dimension_D"]
        assert d["zone_a"]["shutter_hz"] == 20000.0
        assert d["clifford_reversion_mask"] == [1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0]
        assert d["config_digest"]

    def test_engine_verification_gates_all_pass(self):
        """The holonic verification entry point must be green."""
        from unified_henri_vla_engine import verify_engine
        eng = self._engine()
        rep = verify_engine(eng, quiet=True, enforce_spec_tiling=False)
        failed = [k for k, g in rep.gates.items() if not g.get("pass")]
        assert rep.ok, f"failed gates: {failed}"

    def test_g2_integrity_gate_can_fail(self):
        """NEGATIVE CONTROL: a gate that cannot fail is not a gate.

        The tiling is corrupted behind the config's back (the validator would
        reject it at construction, so the corruption is injected directly).
        G2 must report a FAIL verdict - not raise, and not pass.
        """
        from unified_henri_vla_engine import verify_engine
        eng = self._engine()
        # Corrupt the tiling so D != tiles * channels.
        object.__setattr__(eng.cfg.blanket, "clifford_block_size", 7)
        rep = verify_engine(eng, quiet=True, enforce_spec_tiling=False)
        g2 = rep.gates["G2_tiling_integrity"]
        assert g2["pass"] is False
        assert g2["integrity_ok"] is False
        assert g2["tiling_error"], "a corrupted tiling must name its failure"
        assert rep.ok is False

    def test_spec_tiling_gate_separates_compliance_from_integrity(self):
        """A reduced-scale config is internally sound but not spec-compliant."""
        from unified_henri_vla_engine import verify_engine
        eng = self._engine()           # D=1024, not the spec D=65536
        rep = verify_engine(eng, quiet=True, enforce_spec_tiling=False)
        assert rep.gates["G2_tiling_integrity"]["pass"] is True
        assert rep.gates["G2b_spec_tiling"]["pass"] is False
        # With spec enforcement on, the same engine must be rejected.
        rep2 = verify_engine(eng, quiet=True, enforce_spec_tiling=True)
        assert rep2.ok is False

    def test_g4_lock_horizon_is_not_the_slot_tick_budget(self):
        """Conflating the slot with the lock horizon is the error to avoid."""
        eng = self._engine()
        b = eng.cfg.blanket
        assert b.lock_horizon_steps == 2048
        assert b.ticks_per_slot == 16
        assert b.slots_to_lock() == pytest.approx(2048 / 16)
        # The slot is 50 us and the gate ran 2048 steps: they are different
        # quantities and must be reported separately.
        assert b.slot_seconds == pytest.approx(5e-5)
        assert b.lock_horizon_steps != b.ticks_per_slot

    def test_hopfield_beta_defaults_to_the_spec_value(self):
        """Gap 2: beta == 8.0 by default, with the auto path available."""
        eng = self._engine()
        assert eng.describe()["zone_b"]["beta"] == pytest.approx(8.0)
        assert eng.describe()["zone_b"]["auto_beta_from_dim"] is False

    def test_hopfield_auto_beta_uses_sqrt_dim(self):
        from unified_henri_vla_engine import (
            HopfieldSpec, MarkovBlanketSpec, UnifiedHENRIVLAConfig,
            UnifiedHENRIVLAEngine,
        )
        cfg = UnifiedHENRIVLAConfig(
            blanket=MarkovBlanketSpec(dimension_D=1024, clifford_block_size=8),
            hopfield=HopfieldSpec(auto_beta_from_dim=True),
        )
        eng = UnifiedHENRIVLAEngine(cfg)
        assert eng.describe()["zone_b"]["beta"] == pytest.approx(math.sqrt(1024))
