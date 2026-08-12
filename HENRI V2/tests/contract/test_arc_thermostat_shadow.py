"""Phase 7.5 CONN Module B contracts: read-only thermostat shadow.

Pre-registered contracts (manifest_CONN Module B):
- C1 thermostat None -> UNAVAILABLE (never crashes)
- C2 None lambda_active / sagnac_delta -> UNAVAILABLE
- C3 exception -> UNAVAILABLE (fail-closed)
- C4 clean eval below stiffness threshold -> friction 1.0 (production math)
- C5 high stiffness -> friction < 1.0 (anisotropic damping)
- C6 effective_lr = (base_lr / friction) * (1 + sagnac_delta)
- C7 read_only flag True (no weight/policy mutation surface)
- C8 real production thermostat instance evaluates cleanly (scalar-only ctor)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest

from arc_thermostat_shadow import (
    THERMO_OK,
    THERMO_UNAVAILABLE,
    evaluate_thermostat_shadow,
)


class _FakeThermo:
    base_lr = 1e-3

    def __init__(self, friction=1.0, raise_exc=False):
        self._friction = friction
        self._raise = raise_exc

    def compute_anisotropic_friction(self, lambda_active, sagnac_delta):
        if self._raise:
            raise RuntimeError("boom")
        return self._friction


def test_none_thermostat_fail_closed():
    out, status = evaluate_thermostat_shadow(None, 0.1, 0.3)
    assert status == THERMO_UNAVAILABLE
    assert out["status"] == THERMO_UNAVAILABLE


def test_none_signals_fail_closed():
    t = _FakeThermo()
    out, status = evaluate_thermostat_shadow(t, None, 0.3)
    assert status == THERMO_UNAVAILABLE
    out2, status2 = evaluate_thermostat_shadow(t, 0.1, None)
    assert status2 == THERMO_UNAVAILABLE


def test_exception_fail_closed():
    t = _FakeThermo(raise_exc=True)
    out, status = evaluate_thermostat_shadow(t, 0.1, 0.3)
    assert status == THERMO_UNAVAILABLE


def test_low_stiffness_friction_one():
    t = _FakeThermo(friction=1.0)
    out, status = evaluate_thermostat_shadow(t, 0.005, 0.07)
    assert status == THERMO_OK
    assert out["langevin_friction"] == pytest.approx(1.0)


def test_high_stiffness_damping():
    t = _FakeThermo(friction=0.554)
    out, status = evaluate_thermostat_shadow(t, 2.0, 0.5)
    assert status == THERMO_OK
    assert out["langevin_friction"] < 1.0


def test_effective_lr_formula():
    t = _FakeThermo(friction=0.5)
    out, status = evaluate_thermostat_shadow(t, 1.0, 1.0)
    assert status == THERMO_OK
    # effective_lr = (1e-3 / 0.5) * (1 + 1.0) = 4e-3
    assert out["effective_lr"] == pytest.approx(4e-3)


def test_read_only_flag():
    t = _FakeThermo()
    out, status = evaluate_thermostat_shadow(t, 0.1, 0.3)
    assert status == THERMO_OK
    assert out["read_only"] is True


def test_production_thermostat_evaluates():
    from adaptive_viscoelastic_thermostat import AdaptiveViscoelasticThermostat

    t = AdaptiveViscoelasticThermostat(d_model=64, device="cpu")
    out, status = evaluate_thermostat_shadow(t, 0.05, 0.1)
    assert status == THERMO_OK
    assert out["langevin_friction"] == pytest.approx(1.0)
    assert out["read_only"] is True


def test_runner_flag_default_off_and_telemetry_only():
    """C9 source inspection: flag defaults OFF, shadow output is written ONLY
    into the telemetry emit (no policy/action/optimizer consumer), and an
    ON-init failure still emits a typed UNAVAILABLE status (never silent)."""
    runner = Path(__file__).resolve().parents[2] / "production_arc_run.py"
    src = runner.read_text(encoding="utf-8")
    assert 'os.environ.get("HENRI_ARC_THERMOSTAT", "0") == "1"' in src
    # The only occurrence of the shadow info key must be the telemetry emit.
    assert src.count('"thermo_shadow": thermo_shadow_info') == 1
    # Every line touching the shadow must be in its eval/emit block; none may
    # reference the decision path (chosen/action/efe_table/policy/rank/step).
    decision_terms = ("chosen", "action", "efe_table", "policy", "rank", "step")
    for i, line in enumerate(src.splitlines(), 1):
        if "thermo_shadow_info" in line:
            for term in decision_terms:
                assert term not in line, (
                    f"line {i} couples shadow to decision term '{term}': {line.strip()}"
                )
    # ON-init failure must still emit typed UNAVAILABLE per step: the eval
    # guard must not require _thermo_shadow to be non-None.
    assert "if HENRI_ARC_THERMOSTAT:" in src
    assert "if HENRI_ARC_THERMOSTAT and _thermo_shadow is not None:" not in src
    assert "THERMO_SHADOW_UNAVAILABLE" in src
    # compute-then-assign guard: shadow evaluation is after predicted_prior.
    emit_idx = src.index('"thermo_shadow": thermo_shadow_info')
    assert src.index("predicted_prior = predicted_wave.detach()") < emit_idx
