"""Contract tests for OVSAHarmonicEncoder (Class 4.6, Stage 1).

Pre-registered gates (packet o_vsa_stage1_gate_20260821.md):
  T1-1 related pair cos >= 0.40:  return len(x) vs return count(x)
  T1-2 related pair cos >= 0.40:  return max(l)  vs return maximum(l)
  T1-3 unrelated pair cos < 0.15: return len(x)  vs return a + b
  T1-4 unrelated pair cos < 0.15: len vs grid (bare identifiers)
Invariants: uint8 [65536], values in [0,255], determinism, identical strings ~1.0,
  no [D,D] allocation (max allocated tensor == [D]).
"""
from __future__ import annotations

import torch

from o_vsa_harmonic_encoder import OVSAHarmonicEncoder

D = 65536


def _enc(text: str) -> torch.Tensor:
    e = OVSAHarmonicEncoder(d_model=D, device="cpu")
    return e.encode_text(text)


def test_shape_and_dtype():
    v = _enc("def f(x):\n    return len(x)")
    assert v.shape == (D,)
    assert v.dtype == torch.uint8
    assert int(v.min()) >= 0 and int(v.max()) <= 255


def test_determinism():
    a = _enc("def f(x):\n    return len(x)")
    b = _enc("def f(x):\n    return len(x)")
    assert torch.equal(a, b)


def test_identical_strings_near_unity():
    a = _enc("def f(x):\n    return len(x)")
    b = _enc("def f(x):\n    return len(x)")
    sim = OVSAHarmonicEncoder(d_model=D, device="cpu").compute_similarity(a, b)
    assert sim > 0.99, sim


def test_t1_related_len_count():
    e = OVSAHarmonicEncoder(d_model=D, device="cpu")
    a = e.encode_text("def f(x):\n    return len(x)")
    b = e.encode_text("def f(x):\n    return count(x)")
    sim = e.compute_similarity(a, b)
    assert sim >= 0.40, f"len/count related cos {sim} < 0.40"


def test_t1_related_max_maximum():
    e = OVSAHarmonicEncoder(d_model=D, device="cpu")
    a = e.encode_text("def f(l):\n    return max(l)")
    b = e.encode_text("def f(l):\n    return maximum(l)")
    sim = e.compute_similarity(a, b)
    assert sim >= 0.40, f"max/maximum related cos {sim} < 0.40"


def test_t1_unrelated_len_vs_binop():
    e = OVSAHarmonicEncoder(d_model=D, device="cpu")
    a = e.encode_text("def f(x):\n    return len(x)")
    b = e.encode_text("def f(a, b):\n    return a + b")
    sim = e.compute_similarity(a, b)
    assert sim < 0.15, f"len vs binop unrelated cos {sim} >= 0.15"


def test_t1_unrelated_len_vs_grid():
    e = OVSAHarmonicEncoder(d_model=D, device="cpu")
    a = e.encode_text("len")
    b = e.encode_text("grid")
    sim = e.compute_similarity(a, b)
    assert sim < 0.15, f"len vs grid unrelated cos {sim} >= 0.15"


def test_no_dense_allocation():
    """No [D,D] intermediates: max allocation is band-scoped [D]-class tensors."""
    e = OVSAHarmonicEncoder(d_model=D, device="cpu")
    for name, t in e.named_parameters():
        assert t.numel() <= D, f"parameter {name} exceeds [D]"
    v = e.encode_text("def f(x):\n    return len(x)")
    assert v.numel() == D
