"""Contract tests: Phase 10.2 typed action gate (default-OFF, fail-closed).

Run locally (CPU) from the repo root:
    env -u VIRTUAL_ENV -u PYTHONPATH -u PYTHONHOME \\
      PYTHONPATH="HENRI V2" /c/Python314/python.exe -m pytest \\
      "HENRI V2/tests/contract/test_action_gate.py" -q --tb=short
"""
import enum
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from henri_action_gate import (
    FLAG,
    REASON_ACTION_NOT_LEGAL,
    REASON_DECODE_FAILED,
    REASON_PAYLOAD_MALFORMED,
    TypedAction,
    TypedActionRejection,
    TypedActionGate,
    get_action_gate,
)
from hopfield_cleanup import HopfieldActionDecoder


class _Actions(enum.Enum):
    """Hashable action stand-in mirroring arcengine.GameAction semantics."""

    ACTION1 = 1
    ACTION6 = 6


def _act(name: str):
    return _Actions[name]


def _decoder():
    """Hopfield decoder over 2 actions (ACTION1 simple, ACTION6 coordinate)."""
    return HopfieldActionDecoder(
        d_model=64,
        action_enum_class=[_Actions.ACTION1, _Actions.ACTION6],
        seed=1234,
    )


def _grid_with_object() -> list:
    g = [[0] * 4 for _ in range(4)]
    g[0][0] = 1  # single object at top-left
    return g


def _grid_empty() -> list:
    return [[0] * 4 for _ in range(4)]


def test_c1_default_off_factory(monkeypatch):
    """C1: absent flag -> factory returns None (module never constructed)."""
    monkeypatch.delenv(FLAG, raising=False)
    assert get_action_gate(_decoder()) is None


@pytest.fixture
def flag_on(monkeypatch):
    monkeypatch.setenv(FLAG, "1")


def test_c2_simple_action_legal(flag_on):
    """C2: legal simple action -> TypedAction with data=None."""
    dec = _decoder()
    gate = get_action_gate(dec)
    assert isinstance(gate, TypedActionGate)
    wave = dec.get_action_wave(_act("ACTION1"))
    out = gate.gate(wave, _grid_empty(), [_act("ACTION1"), _act("ACTION6")])
    assert isinstance(out, TypedAction)
    assert out.action.name == "ACTION1"
    assert out.data is None
    assert out.payload_complete is False  # simple actions carry no payload


def test_c3_coordinate_action_payload(flag_on):
    """C3: legal coordinate action -> typed (GameAction, {"x","y"}) payload."""
    dec = _decoder()
    gate = get_action_gate(dec)
    wave = dec.get_action_wave(_act("ACTION6"))
    out = gate.gate(wave, _grid_with_object(), [_act("ACTION1"), _act("ACTION6")])
    assert isinstance(out, TypedAction)
    assert out.action.name == "ACTION6"
    assert out.payload_complete is True
    assert isinstance(out.data, dict) and "x" in out.data and "y" in out.data
    assert out.source in ("object_centroid", "fallback_grid")


def test_c4_not_legal_fail_closed(flag_on):
    """C4: decoded action outside the legal subset -> fail-closed No-Op."""
    dec = _decoder()
    gate = get_action_gate(dec)
    wave = dec.get_action_wave(_act("ACTION6"))
    out = gate.gate(wave, _grid_empty(), [_act("ACTION1")])
    assert isinstance(out, TypedActionRejection)
    assert out.reason == REASON_ACTION_NOT_LEGAL
    assert out.decoded_action_name == "ACTION6"
    assert out.legal_actions == ("ACTION1",)


def test_c5_payload_malformed_fail_closed(flag_on, monkeypatch):
    """C5: coordinate action with a broken payload builder -> rejection."""
    import henri_action_gate as hag

    dec = _decoder()
    gate = get_action_gate(dec)
    wave = dec.get_action_wave(_act("ACTION6"))

    def _boom(*args, **kwargs):
        raise RuntimeError("payload builder unavailable")

    monkeypatch.setattr(hag, "build_payload_candidates", _boom)
    out = gate.gate(wave, _grid_empty(), [_act("ACTION1"), _act("ACTION6")])
    assert isinstance(out, TypedActionRejection)
    assert out.reason == REASON_PAYLOAD_MALFORMED


def test_c6_decode_failure_fail_closed(flag_on):
    """C6: decoder raise -> typed rejection, never a crash or silent action."""
    dec = _decoder()
    gate = get_action_gate(dec)

    class _BrokenDecoder:
        def decode_wave_to_action(self, wave):
            raise RuntimeError("snap failed")

    gate.decoder = _BrokenDecoder()
    out = gate.gate(torch.zeros(64), _grid_empty(), [_act("ACTION1")])
    assert isinstance(out, TypedActionRejection)
    assert out.reason == REASON_DECODE_FAILED


def test_c7_confidence_gate(flag_on):
    """C7: confidence below threshold -> rejection (optional gate)."""
    dec = _decoder()
    gate = get_action_gate(dec, confidence_threshold=1.5)  # impossible threshold
    wave = dec.get_action_wave(_act("ACTION1"))
    out = gate.gate(wave, _grid_empty(), [_act("ACTION1"), _act("ACTION6")])
    assert isinstance(out, TypedActionRejection)
    assert out.reason == REASON_DECODE_FAILED
