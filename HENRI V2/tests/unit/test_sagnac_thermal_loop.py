"""Contract tests for the Pillar 3 Sagnac veto -> Langevin thermalisation loop.

These pin the properties `henri_sagnac_thermal_loop` CLAIMS, so the module cannot
drift into claiming more than it does.

CLAIM MAP (each test is able to FAIL):
  T1  flag OFF            -> theta returned unchanged, DISABLED, zero energy
  T2  VETO_UNAVAILABLE    -> quiescent; a failed measurement is NEVER a veto
  T3  coherent wave       -> quiescent, zero energy injected
  T4  hard veto           -> fires, bounded creep, steps == max_steps
  T5  DISCRIMINATING      -> kT strictly increases in delta AND exceeds kT_base.
      A flat kT would make the loop a constant-noise injector carrying no
      information from the veto; this test is what separates coupling from theatre.
  T6  seed reproducibility -> bit-identical output for a fixed seed
  T7  bounded             -> kT <= kT_max for any delta
  T8  norm preserved      -> ||theta|| restored after creep
  T9  SCOPE LIMIT (honest) -> a ONE-HOT wave FIRES even when self-compared,
      because the metric is a MEAN over D components. This documents the
      module's domain rather than hiding it: eps_hard=0.35 is meaningful only
      for near-unit-modulus waves.
  T10 status surface is compact and names the flag

METRIC NOTE (measured, not assumed): for complex waves `_sagnac_similarity`
returns |mean(conj(a)*b)|. Matching phases -> 1.0 -> delta 0. A one-hot -> 1/D.
"""

import os

import pytest
import torch

import arc_sagnac_veto as ASV
import henri_sagnac_thermal_loop as TL

D = 64


def _onehot(i):
    v = torch.zeros(D, dtype=torch.complex64)
    v[i] = 1.0 + 0.0j
    return v


def _phasor():
    return torch.ones(D, dtype=torch.complex64)      # all phases 0 -> sim 1.0


@pytest.fixture()
def enabled(monkeypatch):
    monkeypatch.setenv(TL.FLAG_ENV, "1")
    return TL.SagnacThermalLoop(TL.ThermalLoopConfig(seed=7))


@pytest.fixture()
def disabled(monkeypatch):
    monkeypatch.delenv(TL.FLAG_ENV, raising=False)
    return TL.SagnacThermalLoop(TL.ThermalLoopConfig(seed=7))


def test_t1_flag_off_preserves_production_path(disabled):
    theta = torch.randn(D) * 0.1
    out, ev = disabled.step(_onehot(1), _onehot(0), _onehot(0), theta)
    assert ev.fired is False
    assert ev.status == "DISABLED"
    assert ev.energy_injected == 0.0
    assert torch.equal(out, theta), "flag OFF must return theta unchanged"


def test_t2_unavailable_veto_never_fires(enabled):
    theta = torch.randn(D) * 0.1
    out, ev = enabled.step(None, _phasor(), _phasor(), theta)
    assert ev.fired is False
    assert ev.status == ASV.VETO_UNAVAILABLE, "a failed measurement is not a veto"
    assert torch.equal(out, theta)


def test_t3_coherent_wave_injects_nothing(enabled):
    """A coherent candidate must not be heated. Uses a unit-modulus fixture."""
    theta = torch.randn(D) * 0.1
    out, ev = enabled.step(_phasor(), _phasor(), _phasor(), theta)
    assert ev.fired is False
    assert ev.energy_injected == 0.0
    assert torch.equal(out, theta)


def test_t4_hard_veto_fires_with_bounded_creep(enabled):
    theta = torch.randn(D) * 0.1
    out, ev = enabled.step(_onehot(1), _onehot(0), _onehot(0), theta)
    assert ev.fired is True
    assert ev.status == ASV.VETO_OK
    assert ev.delta_axiom > TL.DEFAULT_TAU_VETO
    assert ev.energy_injected > 0.0
    assert ev.steps == enabled.cfg.max_steps
    assert not torch.equal(out, theta), "a fired veto must move theta"


def test_t5_kT_is_strictly_increasing_in_delta():
    """DISCRIMINATING. A flat kT would make the loop noise-carrying-not-information."""
    cfg = TL.ThermalLoopConfig()
    ds = [0.40, 0.70, 1.00, 1.50]
    ks = [TL.kT_from_delta(d, cfg) for d in ds]
    assert all(ks[i] < ks[i + 1] for i in range(len(ks) - 1)), \
        "kT must strictly increase in delta; a flat kT is theatre"
    assert ks[0] > cfg.kT_base, "an active veto must exceed the quiescent budget"


def test_t6_seed_reproducibility():
    th = torch.randn(D) * 0.1
    a = TL.SagnacThermalLoop(TL.ThermalLoopConfig(seed=7))
    b = TL.SagnacThermalLoop(TL.ThermalLoopConfig(seed=7))
    oa, _ = a.step(_onehot(1), _onehot(0), _onehot(0), th)
    ob, _ = b.step(_onehot(1), _onehot(0), _onehot(0), th)
    assert torch.equal(oa, ob), "same seed must give bit-identical creep"


def test_t7_kT_is_bounded():
    cfg = TL.ThermalLoopConfig()
    for d in (3.0, 10.0, 1e6):
        assert TL.kT_from_delta(d, cfg) <= cfg.kT_max
    assert TL.kT_from_delta(float("nan"), cfg) == 0.0
    assert TL.kT_from_delta(0.0, cfg) == 0.0


def test_t8_norm_preserved(enabled):
    theta = torch.randn(D) * 0.1
    n0 = float(theta.norm())
    out, ev = enabled.step(_onehot(1), _onehot(0), _onehot(0), theta)
    assert ev.fired is True
    assert abs(float(out.norm()) - n0) < 1e-4


def test_t9_scope_limit_onehot_always_fires():
    """HONEST SCOPE. The similarity metric is a MEAN, so a sparse wave self-compares
    to 1/D and the hard veto fires regardless of actual coherence. eps_hard=0.35 is
    meaningful only for near-unit-modulus waves. Documented, not hidden."""
    da_self, _, trig_self, st = ASV.evaluate_veto(_onehot(0), _onehot(0), _onehot(0))
    assert st == ASV.VETO_OK
    assert abs((1.0 - da_self) - 1.0 / D) < 1e-6, "one-hot self-similarity must be 1/D"
    assert trig_self is True, "a one-hot self-compare FIRES under this metric"


def test_t10_status_surface(monkeypatch):
    monkeypatch.delenv(TL.FLAG_ENV, raising=False)
    s = TL.loop_status()
    assert s["module"] == "henri_sagnac_thermal_loop"
    assert s["enabled"] is False
    assert s["tau_veto"] == TL.DEFAULT_TAU_VETO
    monkeypatch.setenv(TL.FLAG_ENV, "1")
    assert TL.loop_status()["enabled"] is True


def test_t11_constructor_rejects_bad_config():
    with pytest.raises(ValueError):
        TL.SagnacThermalLoop(TL.ThermalLoopConfig(kT_max=0.0))
    with pytest.raises(ValueError):
        TL.SagnacThermalLoop(TL.ThermalLoopConfig(max_steps=-1))
