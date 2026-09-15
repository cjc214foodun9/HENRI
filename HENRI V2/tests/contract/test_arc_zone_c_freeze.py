"""Contract tests: Phase 10.1 Zone C freeze on the PSG engine (fail-closed).

Covers:
- the freeze guard exists in the engine (source inspection);
- requesting the PSG path while frozen RAISES before any compute;
- the guard fires BEFORE the encoder is touched (no work, no leaked result);
- the explicit development override HENRI_ZONE_C_FREEZE=0 permits compute;
- flag OFF (the default) leaves the engine inert rather than "frozen";
- the two naive operator sites are UNMODIFIED. This is load-bearing: a
  governance freeze that silently rewrote the frozen code would be a defect.
"""
import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]      # ...\HENRI V2
ENGINE = REPO / "progressive_semantic_grounding_engine.py"
ENGINE_TEXT = ENGINE.read_text(encoding="utf-8", errors="replace")


def _clean_env(**over):
    for k in ("HENRI_ARC_PSG", "HENRI_ZONE_C_FREEZE"):
        os.environ.pop(k, None)
    os.environ.update(over)


def _engine():
    sys.path.insert(0, str(REPO))
    import progressive_semantic_grounding_engine as psg
    return psg


def test_freeze_guard_present():
    assert "def psg_frozen(" in ENGINE_TEXT
    assert "def assert_psg_not_frozen(" in ENGINE_TEXT
    assert "BLOCKED_ZONE_C_FROZEN" in ENGINE_TEXT
    assert "assert_psg_not_frozen(" in ENGINE_TEXT.split(
        "def compile_functor_wave(")[1]


def test_naive_operator_sites_unmodified():
    # Frozen, NOT refactored: both naive operator sites read exactly as before.
    assert "w_task = w_task + torch.conj(wx) * wy" in ENGINE_TEXT
    assert "w_task = F.normalize(w_task, p=2, dim=-1)" in ENGINE_TEXT
    assert "w_task, goal_c, res = compile_functor_wave(" in ENGINE_TEXT


def test_frozen_raise_when_psg_requested():
    _clean_env(HENRI_ARC_PSG="1")  # freeze default ON
    psg = _engine()
    assert psg.psg_frozen() is True
    with pytest.raises(RuntimeError) as ei:
        psg.assert_psg_not_frozen("unit-test", task_id="t")
    assert "BLOCKED_ZONE_C_FROZEN" in str(ei.value)
    assert "unit-test" in str(ei.value)
    assert "t" in str(ei.value)


def test_frozen_does_not_fire_when_flag_off():
    _clean_env()
    psg = _engine()
    assert psg.psg_frozen() is False
    psg.assert_psg_not_frozen("unit-test")  # must NOT raise


def test_explicit_override_permits_compute():
    _clean_env(HENRI_ARC_PSG="1", HENRI_ZONE_C_FREEZE="0")
    psg = _engine()
    assert psg.psg_frozen() is False
    psg.assert_psg_not_frozen("unit-test")  # must NOT raise
    _clean_env()


def test_compile_functor_wave_raises_before_compute():
    """The guard must fire BEFORE encoding: no work done, no result leaked."""
    _clean_env(HENRI_ARC_PSG="1")
    psg = _engine()

    class _ExplodingTokenizer:
        def encode_spatial_grid(self, grid):  # pragma: no cover
            raise AssertionError("encoder was called: guard fired too late")

    with pytest.raises(RuntimeError) as ei:
        psg.compile_functor_wave([[1]], _ExplodingTokenizer(), task_id="t")
    assert "BLOCKED_ZONE_C_FROZEN" in str(ei.value)
    _clean_env()


def test_class_method_guarded():
    _clean_env(HENRI_ARC_PSG="1")
    psg = _engine()
    eng = object.__new__(psg.ProgressiveSemanticGroundingEngine)
    with pytest.raises(RuntimeError) as ei:
        eng.compile_task_functor([], task_id="t2")
    assert "BLOCKED_ZONE_C_FROZEN" in str(ei.value)
    _clean_env()
