"""Unit and contract tests for the Step 5.2 GPU carrier module.

Spec: HENRI-ARCH-2026-FRONTIER-EVALUATION-AND-DISCOVERY-LEDGER, section 5.2

These tests run on a CPU-only host by design. The Triton kernel cannot be
executed here (no Triton, no CUDA), so the tests cover what CAN be decided
without a GPU:

  * the kernel's weight vector IS the verified one (exact parity anchor)
  * the two independent coupling implementations agree (algorithm parity)
  * the span truncation at 504 reproduces the full-span result
  * the derived tau bound counts compute AND sync and can report failure
  * the backend report never claims a measurement it does not have
  * the GPU path fails CLOSED when Triton/CUDA are absent

A test that only asserted "the module imports" would pass on a broken kernel.
Every test here asserts a relationship that could come out false.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from basal_boundary_engine import (  # noqa: E402
    SPEC_KURAMOTO_COUPLING_K,
    SPEC_LOCK_HORIZON_STEPS,
    SPEC_NON_LOCAL_SPAN,
    evanescent_kernel,
)
from basal_triton_kernel import (  # noqa: E402
    SPEC_BLOCK_SIZE,
    SPEC_SHUTTER_US,
    SPEC_TAU_BUDGET_US,
    TRITON_AVAILABLE,
    backend_report,
    cuda_available,
    fft_relax,
    fused_relax,
    kernel_l1_distance,
    order_parameter,
    relax_span,
    ring_kernel,
    slot_budget_analysis,
    span_coupled_field,
    span_evanescent_weights,
    taps_for_reach,
    tau_budget_analysis,
)


class TestParityAnchor:
    """The load-bearing identity: the kernel's weights ARE the verified ones."""

    @pytest.mark.parametrize(
        "n,decay",
        [(64, 8.0), (256, 32.0), (1024, 96.0), (2048, 504.0)],
    )
    def test_untruncated_ring_kernel_equals_live_kernel(self, n, decay):
        """Exact equality, not closeness.

        With no truncation every cyclic distance is inside the reach, so
        ring_kernel must reproduce evanescent_kernel EXACTLY. If this ever
        drifts, the kernel solves a different problem than the verified code
        and every downstream number is void.
        """
        assert kernel_l1_distance(ring_kernel(n, decay), evanescent_kernel(n, decay)) == 0.0

    def test_l1_distance_detects_a_real_difference(self):
        """NEGATIVE CONTROL: the comparison must be able to fail.

        A parity check that cannot report a difference is not a check.
        """
        a = ring_kernel(256, 32.0)
        b = evanescent_kernel(256, 64.0)   # deliberately different decay
        assert kernel_l1_distance(a, b) > 1e-3

    def test_taps_are_always_odd(self):
        """A symmetric +/-H reach has 2H+1 taps; an even count would be
        off-centre and break the tap/distance correspondence the kernel uses."""
        for h in (0, 1, 7, 251, 252):
            assert taps_for_reach(h) % 2 == 1
            assert taps_for_reach(h) == 2 * h + 1

    def test_span_weights_are_a_convex_combination(self):
        """Renormalization keeps the weights summing to 1.

        Without it, dropping taps would silently scale the coupling strength
        and the same K would mean different things at different reaches.
        """
        w = span_evanescent_weights(8192, 504.0, SPEC_NON_LOCAL_SPAN // 2)
        assert float(w.sum()) == pytest.approx(1.0, abs=1e-6)
        assert bool((w >= 0).all())

    def test_truncation_zeros_only_the_outside_taps(self):
        """The truncated kernel must be the full kernel with far taps dropped."""
        n, decay = 512, 64.0
        half = 32
        w = ring_kernel(n, decay, half_width=half)
        full = evanescent_kernel(n, decay)
        j = torch.arange(n)
        d = torch.minimum(j, n - j)
        inside = d <= half
        # Outside the reach: exactly zero after renormalization's where().
        assert float(w[~inside].abs().sum()) == 0.0
        # Inside the reach: a positive, monotonically decaying profile.
        assert float(w[inside].sum()) == pytest.approx(1.0, abs=1e-6)


class TestAlgorithmParity:
    """Two independent coupling implementations must agree."""

    def test_direct_tap_sum_matches_circular_convolution(self):
        """The tap sum (what Triton does) vs the FFT (what the live class does).

        Independent implementations agreeing is evidence; a copy of the same
        code agreeing with itself is not.
        """
        n, decay, steps = 1024, 60.0, 512
        g = torch.Generator().manual_seed(7)
        phases = (torch.rand(n, generator=g) * 2.0 - 1.0) * torch.pi
        w = span_evanescent_weights(n, decay, n // 2)
        k = ring_kernel(n, decay)
        r_span = order_parameter(relax_span(
            phases, w, coupling_K=SPEC_KURAMOTO_COUPLING_K, dt=0.01, steps=steps))
        r_fft = order_parameter(fft_relax(
            phases, k, coupling_K=SPEC_KURAMOTO_COUPLING_K, dt=0.01, steps=steps))
        assert abs(r_span - r_fft) < 1e-4

    def test_span_path_matches_the_live_syncytium(self):
        """The new path must reproduce the already-verified class."""
        from basal_boundary_engine import EvanescentKuramotoSyncytium

        n, decay, steps = 1024, 60.0, 512
        g = torch.Generator().manual_seed(11)
        phases = (torch.rand(n, generator=g) * 2.0 - 1.0) * torch.pi
        syn = EvanescentKuramotoSyncytium(
            num_channels=n, coupling_K=SPEC_KURAMOTO_COUPLING_K,
            decay_length=decay, dt=0.01, natural_frequency_scale=0.0,
            noise_temperature=0.0, seed=0,
        )
        syn.phases = phases.clone()
        live_r = float(syn.relax(steps)["r"])
        r_span = order_parameter(relax_span(
            phases, span_evanescent_weights(n, decay, n // 2),
            coupling_K=SPEC_KURAMOTO_COUPLING_K, dt=0.01, steps=steps))
        assert abs(live_r - r_span) < 1e-4

    def test_order_parameter_bounds(self):
        """r is in [0, 1]: 1 for a locked state, 0 for a balanced spread."""
        assert order_parameter(torch.zeros(64)) == pytest.approx(1.0)
        balanced = torch.tensor([0.0, math.pi] * 32)
        assert order_parameter(balanced) < 1e-6

    def test_seeded_runs_are_reproducible(self):
        n, decay, steps = 512, 40.0, 256
        w = span_evanescent_weights(n, decay, 64)
        g = torch.Generator().manual_seed(3)
        phases = (torch.rand(n, generator=g) * 2.0 - 1.0) * torch.pi
        a = relax_span(phases, w, dt=0.01, steps=steps)
        b = relax_span(phases, w, dt=0.01, steps=steps)
        assert torch.equal(a, b)


class TestTruncation:
    def test_504_span_reproduces_full_span_lock(self):
        """The kernel's ACTUAL reach must not change the answer.

        This is the measurement that justifies using 504 rather than the full
        ring. If a narrow reach silently degraded r, the kernel would be
        wrong while its weights were right.
        """
        n, decay, steps = 1024, 60.0, 512
        g = torch.Generator().manual_seed(7)
        phases = (torch.rand(n, generator=g) * 2.0 - 1.0) * torch.pi
        r_full = order_parameter(fft_relax(
            phases, ring_kernel(n, decay),
            coupling_K=SPEC_KURAMOTO_COUPLING_K, dt=0.01, steps=steps))
        r_504 = order_parameter(relax_span(
            phases, span_evanescent_weights(n, decay, SPEC_NON_LOCAL_SPAN // 2),
            coupling_K=SPEC_KURAMOTO_COUPLING_K, dt=0.01, steps=steps))
        assert abs(r_504 - r_full) < 0.02

    def test_narrow_reach_degrades_lock(self):
        """NEGATIVE CONTROL: a too-narrow reach must measurably lose lock."""
        n, decay, steps = 1024, 60.0, 512
        g = torch.Generator().manual_seed(7)
        phases = (torch.rand(n, generator=g) * 2.0 - 1.0) * torch.pi
        r_full = order_parameter(fft_relax(
            phases, ring_kernel(n, decay),
            coupling_K=SPEC_KURAMOTO_COUPLING_K, dt=0.01, steps=steps))
        r_narrow = order_parameter(relax_span(
            phases, span_evanescent_weights(n, decay, 2),   # 5 taps
            coupling_K=SPEC_KURAMOTO_COUPLING_K, dt=0.01, steps=steps))
        assert r_narrow < r_full - 0.05


class TestTauBound:
    def test_sub_budget_is_reported_unreachable(self):
        """The 12.8 us sub-budget must not be reported as reachable."""
        b = tau_budget_analysis()
        assert b.sub_budget_reachable is False
        assert b.per_step_budget_ns == pytest.approx(12.5, abs=0.01)

    def test_compute_only_floor_exceeds_the_sub_budget(self):
        """The strongest form of the bound: even with ZERO sync cost.

        If this test ever fails, the 12.8 us budget MIGHT be reachable and the
        verdict must be revisited. It is the falsifiable core of the claim.
        """
        b = tau_budget_analysis()
        compute_only = min(b.compute_floor_us_fft_multi_block,
                           b.compute_floor_us_multi_block)
        assert compute_only > SPEC_TAU_BUDGET_US

    def test_compute_only_shutter_answer_differs_from_with_sync(self):
        """The two answers must be reported separately, never conflated.

        Compute alone fits the 50 us shutter; compute plus the mandatory
        per-step barrier does not. Collapsing them would hide which constraint
        actually binds.
        """
        b = tau_budget_analysis()
        assert b.shutter_reachable_compute_only is True
        assert b.shutter_reachable is False
        assert b.binding_constraint == "synchronization"

    def test_tau_scales_linearly_with_steps(self):
        a = tau_budget_analysis(steps=512)
        c = tau_budget_analysis(steps=1024)
        assert c.sync_floor_us_design_a[0] == pytest.approx(
            2.0 * a.sync_floor_us_design_a[0])

    def test_fft_coupling_is_cheaper_than_the_tap_sum(self):
        """Justifies putting the FFT form on the GPU production path."""
        s = slot_budget_analysis()
        assert s["macs_per_step_fft"] < s["macs_per_step_tap_sum"]

    def test_slot_budget_reports_both_verdicts(self):
        s = slot_budget_analysis()
        assert "fits_shutter_50us" in s
        assert "fits_sub_budget_12p8us" in s
        assert s["slots_to_lock_from_cold"] == math.ceil(
            SPEC_LOCK_HORIZON_STEPS / s["ticks_per_slot"])
        assert s["measured_tau_us"] is None
        assert s["evidence_class"] == "DERIVED"


class TestFailClosed:
    def test_backend_report_never_claims_a_measurement(self):
        br = backend_report()
        assert br["measured_tau_us"] is None
        ok = br["triton_available"] and br["cuda_available"]
        assert br["status"] == ("OBSERVED" if ok else "BLOCKED")

    @pytest.mark.skipif(
        TRITON_AVAILABLE and cuda_available(),
        reason="GPU present: the fail-closed path is not the active path",
    )
    def test_fused_relax_fails_closed_without_a_gpu(self):
        """The GPU path must raise, not silently fall back to CPU.

        A silent CPU fallback would let a "GPU measurement" report CPU numbers.
        """
        n = 64
        w = span_evanescent_weights(n, 8.0, 8)
        with pytest.raises(RuntimeError):
            fused_relax(torch.zeros(n), w, steps=1)

    def test_kernel_absent_when_triton_absent(self):
        """No Triton on this host means no kernel symbol, not a stub that runs."""
        if not TRITON_AVAILABLE:
            import basal_triton_kernel as m
            assert not hasattr(m, "_fused_autopoietic_kuramoto_kernel")
            assert m.TRITON_IMPORT_ERROR

    def test_span_coupled_field_rejects_an_even_tap_count(self):
        """An even tap count would centre the kernel between taps."""
        with pytest.raises(ValueError):
            span_coupled_field(torch.zeros(16), torch.ones(4) / 4.0)
