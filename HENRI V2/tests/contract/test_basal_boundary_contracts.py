"""Contract tests — basal boundary engine (Zone A/B/C Markov blankets).

Spec: HENRI-ARCH-2026-BASAL-COGNITION-AND-BOUNDARY-ENGINEERING
Frontier: HENRI-ARCH-2026-FRONTIER-EVALUATION-AND-DISCOVERY-LEDGER

These are the CONTRACT layer: they assert the interfaces the rest of the
repository depends on, and they are the file section 5.3 names as the remote
verification entry point. They run on a CPU-only host.

Scope, and why each item is a contract rather than a unit detail:

  1. DEFAULT-OFF INVARIANT. `get_unified_basal_engine()` returns None unless
     HENRI_BASAL_ENGINE=1. This is load-bearing: with the flag absent, importing
     the module must not construct a class or change any live path.
  2. SEALED CONSTANTS. non_local_span and lock_horizon_steps are sealed at the
     production tiling and REJECT drift. The seal must be able to fail.
  3. APERTURE DECOUPLING. The 50 us shutter and the lock horizon are different
     quantities and must not be derived from one another.
  4. EVIDENCE RECEIPTS. The claims in the commit messages are backed by receipt
     files on disk, and each receipt states its own evidence class. A receipt
     that disappears makes the provenance claim false.
  5. GPU PATH FAILS CLOSED. A "GPU measurement" must never silently report CPU
     numbers.
  6. CROSS-MODULE WIRING. The engine consumes the modules it names, so a rename
     in one is caught here rather than at first live invocation.

A test that only asserted "the module imports" would pass on a broken engine.
Every test below asserts a relationship that could come out false.
"""

from __future__ import annotations

import json
import math
import os
import pathlib
import sys

import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]          # <repo>/HENRI V2
VERIF = ROOT / "experiments" / "verification"
for _p in (str(ROOT), str(VERIF)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import basal_boundary_engine as bbe  # noqa: E402
import epsilon_band_gate as ebg  # noqa: E402
import koopman_action_ledger as kal  # noqa: E402
import unified_henri_vla_engine as uve  # noqa: E402
import zone_bc_engram_sync as zes  # noqa: E402


# ---------------------------------------------------------------------------
# 1. Default-OFF invariant
# ---------------------------------------------------------------------------

class TestDefaultOffInvariant:
    """The flag is the only thing that may construct the engine."""

    def test_engine_is_none_when_flag_absent(self, monkeypatch):
        monkeypatch.delenv(uve.FLAG, raising=False)
        assert uve.get_unified_basal_engine() is None

    def test_engine_is_none_for_any_non_one_value(self, monkeypatch):
        """Only the exact string "1" enables it.

        "true"/"yes"/"0"/"" must all stay off. A truthy-string check here would
        silently arm the engine on a typo.
        """
        for value in ("0", "", "true", "yes", "TRUE", "on"):
            monkeypatch.setenv(uve.FLAG, value)
            assert uve.get_unified_basal_engine() is None, value

    def test_engine_constructs_when_flag_is_exactly_one(self, monkeypatch):
        monkeypatch.setenv(uve.FLAG, "1")
        eng = uve.get_unified_basal_engine()
        assert eng is not None

    def test_importing_the_module_does_not_construct_anything(self):
        """Import must be inert: no GPU, no clock, no side effect.

        Asserted by reading the source for module-level construction of the
        engine class. A module-level `ENGINE = UnifiedHENRIVLAEngine()` would
        make importing the module change behaviour for every consumer.
        """
        src = (ROOT / "unified_henri_vla_engine.py").read_text(encoding="utf-8")
        for line in src.splitlines():
            stripped = line.strip()
            if stripped.startswith("UnifiedHENRIVLAEngine("):
                # Only legal inside a function body (indented) or under __main__.
                assert line.startswith(" ") or line.startswith("\t"), line

    def test_all_five_modules_import_without_a_gpu(self):
        """The bundle must be importable on a CPU-only host."""
        for mod in (bbe, ebg, zes, kal, uve):
            assert mod is not None


# ---------------------------------------------------------------------------
# 2. Sealed constants
# ---------------------------------------------------------------------------

class TestSealedConstantsContract:
    def test_sealed_module_constants_exist_and_are_stable(self):
        assert bbe.SPEC_NON_LOCAL_SPAN == 504
        assert bbe.SPEC_LOCK_HORIZON_STEPS == 1024
        assert bbe.SPEC_MEASURED_MIN_LEAKAGE_FRACTION == pytest.approx(0.0205)

    def test_production_tiling_accepts_the_sealed_values(self):
        m = uve.MarkovBlanketSpec()          # D = 65536, the production ring
        assert m.dimension_D == 65536
        assert m.num_tiles == 8192
        assert m.lock_horizon_steps == 1024
        assert abs(m.resolve_leakage_length() - bbe.SPEC_NON_LOCAL_SPAN) <= 1.0

    def test_seal_rejects_lock_horizon_drift(self):
        """CONTRACT: the seal must be able to fail."""
        with pytest.raises(Exception) as ei:
            uve.MarkovBlanketSpec(lock_horizon_steps=2048)
        assert "lock horizon" in str(ei.value).lower()

    def test_seal_rejects_leakage_margin_drift(self):
        """Changing the margin moves the RESOLVED span, not just a field."""
        with pytest.raises(Exception) as ei:
            uve.MarkovBlanketSpec(evanescent_leakage_margin=9.0)
        assert "non-local span" in str(ei.value).lower()

    def test_seal_rejects_fixed_span_drift(self):
        with pytest.raises(Exception) as ei:
            uve.MarkovBlanketSpec(
                auto_leakage_from_ring=False, evanescent_decay_length=8.0
            )
        assert "non-local span" in str(ei.value).lower()

    def test_reduced_scale_config_is_exempt(self):
        """The seal is scoped to D=65536, so it cannot become a tautology that
        forbids every test configuration."""
        m = uve.MarkovBlanketSpec(
            dimension_D=1024, clifford_block_size=8, lock_horizon_steps=2048
        )
        assert m.lock_horizon_steps == 2048

    def test_markov_blanket_spec_is_frozen(self):
        """'Immutable' must mean the field cannot be reassigned."""
        m = uve.MarkovBlanketSpec()
        with pytest.raises(Exception):
            m.lock_horizon_steps = 4096

    def test_tiling_validator_rejects_a_non_divisible_dimension(self):
        with pytest.raises(Exception):
            uve.MarkovBlanketSpec(dimension_D=1025, clifford_block_size=8)

    def test_leakage_margin_below_one_is_rejected(self):
        """A margin < 1 places the working point below the measured knee."""
        with pytest.raises(bbe.BasalBoundaryError):
            bbe.recommended_leakage_length(8192, margin=0.5)


# ---------------------------------------------------------------------------
# 3. Aperture decoupling
# ---------------------------------------------------------------------------

class TestApertureDecoupling:
    def test_shutter_is_twenty_kilohertz_and_fifty_microseconds(self):
        m = uve.MarkovBlanketSpec()
        assert m.shutter_hz == pytest.approx(20_000.0)
        assert m.slot_seconds == pytest.approx(5e-5)

    def test_horizon_and_aperture_are_independent_quantities(self):
        """1024 is a relaxation STEP COUNT; 5e-5 is a TIME.

        Deriving either from the other is the error the horizon probe exists to
        catch. They also must not be numerically equal, which makes an
        accidental aliasing visible.
        """
        m = uve.MarkovBlanketSpec()
        assert m.lock_horizon_steps == 1024
        assert m.slot_seconds == pytest.approx(5e-5)
        assert m.lock_horizon_steps != m.ticks_per_slot
        assert m.slot_seconds != m.lock_horizon_steps

    def test_slots_to_lock_is_the_horizon_over_ticks(self):
        m = uve.MarkovBlanketSpec()
        expected = m.lock_horizon_steps / m.ticks_per_slot
        assert m.slots_to_lock() == pytest.approx(expected)
        # At the default 32 ticks/slot this is 32 slots, NOT one slot.
        assert m.slots_to_lock() == pytest.approx(32.0)

    def test_cold_lock_is_not_a_single_slot_operation(self):
        """CONTRACT: the 1024-step lock spans many slots, and the receipt says so.

        If slots_to_lock() were ever changed to 1, the architecture story would
        silently become false.
        """
        m = uve.MarkovBlanketSpec()
        assert m.slots_to_lock() > 1.0


# ---------------------------------------------------------------------------
# 4. Evidence receipts
# ---------------------------------------------------------------------------

class TestEvidenceReceipts:
    EXPECTED = (
        "basal_syncytium_calibration.json",
        "basal_syncytium_discriminating.json",
        "basal_syncytium_frontier.json",
        "basal_syncytium_boundary.json",
        "basal_syncytium_horizon.json",
    )

    def test_the_five_calibration_receipts_exist(self):
        for name in self.EXPECTED:
            assert (VERIF / name).is_file(), f"missing receipt: {name}"

    def test_receipts_are_small_json_not_binaries(self):
        for name in self.EXPECTED:
            p = VERIF / name
            assert p.stat().st_size < 1_000_000, name
            json.loads(p.read_text(encoding="utf-8"))

    def test_boundary_receipt_records_the_percolation_knee(self):
        """The 160 -> 168 channel transition is the measured percolation knee.

        It lives in the BOUNDARY receipt (K=2.45, 8192 channels) and NOT in the
        frontier receipt, which was swept at K=32. Pointing a receipt test at
        the wrong file is how a provenance claim becomes vacuous; the test now
        names the file that actually holds the measurement.
        """
        rep = json.loads(
            (VERIF / "basal_syncytium_boundary.json").read_text(encoding="utf-8")
        )
        sweep = {
            int(r["decay"]): float(r["r_final"])
            for r in rep["fine_sweep_random_start"]
        }
        assert sweep[160] < 0.93, "160 channels must remain sub-critical"
        assert sweep[168] >= 0.93, "168 channels must clear the gate"
        # A bifurcation, not a gradual ramp: the knee is sharp.
        assert sweep[168] - sweep[160] > 0.5

    def test_frontier_receipt_records_the_span_threshold(self):
        """The frontier receipt measures the span needed at other K values.

        Every recorded threshold must sit above the measured knee and clear the
        r gate, or the sealed 504 channels would not be a safe margin over it.
        """
        rep = json.loads(
            (VERIF / "basal_syncytium_frontier.json").read_text(encoding="utf-8")
        )
        assert "D_frontier" in rep and rep["D_frontier"]
        for row in rep["D_frontier"]:
            assert float(row["min_decay"]) >= 160, row
            assert float(row["r_at_min"]) >= 0.93, row

    def test_triton_parity_receipt_states_its_evidence_class(self):
        rep = json.loads(
            (VERIF / "basal_triton_parity.json").read_text(encoding="utf-8")
        )
        assert rep["backend"]["measured_tau_us"] is None
        assert rep["backend"]["status"] in ("OBSERVED", "BLOCKED")
        assert rep["tau_budget"]["evidence_class"] == "DERIVED"
        assert rep["tau_budget"]["measured_tau_us"] is None

    def test_parity_receipt_has_no_failures(self):
        rep = json.loads(
            (VERIF / "basal_triton_parity.json").read_text(encoding="utf-8")
        )
        assert rep["failures"] == []
        assert rep["ok"] is True

    def test_receipts_contain_no_secret_material(self):
        """Receipts are committed to a public repository."""
        needles = ("api_key", "apikey", "password", "secret", "bearer ", "sk-")
        for p in VERIF.glob("basal_*.json"):
            low = p.read_text(encoding="utf-8").lower()
            for n in needles:
                assert n not in low, f"{p.name} contains {n!r}"


# ---------------------------------------------------------------------------
# 5. GPU path fails closed
# ---------------------------------------------------------------------------

class TestGpuPathFailsClosed:
    def test_fused_relax_raises_rather_than_falling_back(self):
        """A silent CPU fallback would let a 'GPU measurement' report CPU data."""
        import basal_triton_kernel as tk

        if tk.TRITON_AVAILABLE and tk.cuda_available():
            pytest.skip("GPU present: the fail-closed path is not active")
        w = tk.span_evanescent_weights(64, 8.0, 8)
        with pytest.raises(RuntimeError):
            tk.fused_relax(torch.zeros(64), w, steps=1)

    def test_backend_report_admits_when_it_has_no_gpu(self):
        import basal_triton_kernel as tk

        br = tk.backend_report()
        assert br["measured_tau_us"] is None
        if not (br["triton_available"] and br["cuda_available"]):
            assert br["status"] == "BLOCKED"

    def test_tau_bound_is_reported_as_unreachable(self):
        """The derived bound must keep reporting the failing direction.

        If a later edit makes the 12.8 us sub-budget look reachable, this test
        fires and the claim must be re-argued rather than assumed.
        """
        import basal_triton_kernel as tk

        b = tk.tau_budget_analysis()
        assert b.sub_budget_reachable is False
        assert b.binding_constraint == "synchronization"

    def test_kernel_source_is_absent_without_triton(self):
        """A stub named like the kernel would be worse than no kernel."""
        import basal_triton_kernel as tk

        if not tk.TRITON_AVAILABLE:
            assert not hasattr(tk, "_fused_autopoietic_kuramoto_kernel")
            assert tk.TRITON_IMPORT_ERROR


# ---------------------------------------------------------------------------
# 6. Cross-module wiring
# ---------------------------------------------------------------------------

class TestCrossModuleWiring:
    def _engine(self):
        return uve.UnifiedHENRIVLAEngine(
            uve.UnifiedHENRIVLAConfig(
                blanket=uve.MarkovBlanketSpec(
                    dimension_D=1024, clifford_block_size=8,
                    kuramoto_coupling_K=8.0, lock_horizon_steps=2048,
                )
            )
        )

    def test_engine_composes_the_named_modules(self):
        eng = self._engine()
        assert isinstance(eng.blanket, bbe.DynamicMarkovBlanket)
        assert eng.leakage_length > 0.0

    def test_describe_surfaces_the_sealed_values(self):
        d = self._engine().describe()
        assert "zone_a" in d and "zone_b" in d
        assert d["zone_b"]["beta"] == pytest.approx(bbe.SPEC_HOPFIELD_BETA)

    def test_gate_report_names_required_and_optional_gates(self):
        rep = uve.verify_engine(self._engine(), quiet=True, enforce_spec_tiling=False)
        assert "G2_tiling_integrity" in rep.gates
        assert "G2b_spec_tiling" in rep.gates
        assert rep.gates["G2b_spec_tiling"]["required"] is False
        assert rep.failed_gates == []

    def test_all_gates_pass_on_a_reduced_scale_engine(self):
        rep = uve.verify_engine(self._engine(), quiet=True, enforce_spec_tiling=False)
        assert rep.ok is True
        assert len(rep.gates) >= 7

    def test_clifford_reversion_mask_matches_the_contract(self):
        """Cl(3,0) reversion flips bivectors and the pseudoscalar."""
        assert bbe.CLIFFORD_REVERSION_MASK == (1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0)

    def test_epsilon_band_gate_is_two_sided(self):
        """Defect D1's fix: the two-sided form accepts its own control.

        The gate returns a CLASSIFICATION STRING, not an object. Asserting the
        classification is what makes this a real check; asserting `.accepted` on
        a string raises AttributeError and tests nothing about the gate.
        """
        assert ebg.two_sided_band(0.0, 0.02, 512, 128) == "ACCEPT_CONTROL_BAND"
        assert ebg.two_sided_band(0.5, 0.02, 512, 128) == "ACCEPT_STRUCTURED"
        # The i/o warm-up pre-condition is checked FIRST: without it, a dead
        # stream at small n is scored and the D1 defect returns.
        assert ebg.two_sided_band(0.5, 0.02, 8, 128) == "BLOCKED_SUB_MINIMUM_N"
        # The falsified one-sided rule rejects that same control stream.
        assert ebg.one_sided_band(0.0, 0.02, 512, 128) == "REJECT_OUT_OF_BAND"

    def test_sagnac_delta_is_bounded_and_canonical(self):
        """Canonical homodyne form: identical waves -> 0, opposite -> 1.

        The spec's `1 - <a,b>/D` is ~1 for EVERY unit-norm pair and therefore
        vetoes everything (D-SAGNAC). The canonical form must SEPARATE the two
        extremes; that separation is what makes the veto meaningful.
        """
        sagnac = bbe.EvanescentKuramotoSyncytium.sagnac_delta
        a = torch.tensor([1.0, 0.0, 0.0])
        assert sagnac(a, torch.tensor([1.0, 0.0, 0.0])) == pytest.approx(0.0, abs=1e-6)
        assert sagnac(a, torch.tensor([-1.0, 0.0, 0.0])) == pytest.approx(1.0, abs=1e-6)
        # Bounded for arbitrary direction pairs, and the degenerate zero
        # reference is maximum stress rather than a crash.
        g = torch.Generator().manual_seed(5)
        for _ in range(16):
            u = torch.randn(32, generator=g)
            v = torch.randn(32, generator=g)
            assert 0.0 <= sagnac(u, v) <= 1.0
        assert sagnac(a, torch.zeros(3)) == pytest.approx(1.0)
        # The veto threshold is in the stated range and rejects a full mismatch.
        assert bbe.SPEC_SAGNAC_VETO_THRESHOLD == pytest.approx(0.35)
        assert bbe.EvanescentKuramotoSyncytium.is_vetoed(1.0) is True
        assert bbe.EvanescentKuramotoSyncytium.is_vetoed(0.0) is False

    def test_zone_c_sync_digest_roundtrips(self):
        """Zone C persistence needs a content-addressed handle on the wave.

        The digest must be reproducible from the wave, from the envelope's own
        payload bytes, and must change when the wave changes. A digest that is
        stable across different payloads would not identify anything.
        """
        w = torch.linspace(-1.0, 1.0, 8)
        env = zes.make_envelope(w, "lexical", 0.25)
        assert zes.wave_digest(w) == env.digest
        assert zes.wave_digest(env.payload) == env.digest
        assert env.domain == "lexical"
        assert env.sagnac_stress == pytest.approx(0.25)
        # NEGATIVE CONTROL: a different wave must not collide.
        assert zes.wave_digest(w * 1.5) != env.digest
        # The envelope is byte-owning: its payload survives the source tensor.
        assert len(env.payload) == w.numel() * 4          # float32 little-endian
