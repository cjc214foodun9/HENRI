"""Contract tests for henri_topological_encoder.py (directive 4).

Every declared property gets a test; every gate gets a DEAD-INPUT NEGATIVE
CONTROL that must fail it. An untested gate is not a gate.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from henri_topological_encoder import (  # noqa: E402
    MultiscaleTopologicalEncoder,
    TopologicalEncoderError,
    cosine,
)

D = 4096


def _ring(n=9, ring_col=3, inner=7, top=2, left=2, h=5, w=5, bg=0):
    g = [[bg] * n for _ in range(n)]
    bot, right = top + h - 1, left + w - 1
    for j in range(left, right + 1):
        g[top][j] = ring_col
        g[bot][j] = ring_col
    for i in range(top, bot + 1):
        g[i][left] = ring_col
        g[i][right] = ring_col
    y = [r[:] for r in g]
    for i in range(top + 1, bot):
        for j in range(left + 1, right):
            y[i][j] = inner
    return g, y


def test_interior_and_boundary_markers_are_computed():
    enc = MultiscaleTopologicalEncoder(d_model=D, n_levels=3)
    g, _ = _ring()
    _, f = enc.encode(g)
    n_int = sum(sum(r) for r in f.interior_mask)
    n_bnd = sum(sum(r) for r in f.boundary_mask)
    # ring 5x5 => interior is 3x3 = 9 cells
    assert n_int == 9, n_int
    assert n_bnd > 0
    assert not any(f.interior_mask[i][j] and f.boundary_mask[i][j]
                   for i in range(9) for j in range(9))


def test_interior_marker_is_position_dependent_not_global():
    """The whole point: two grids with the SAME colour multiset but different
    ring POSITION must produce different interior masks."""
    enc = MultiscaleTopologicalEncoder(d_model=D, n_levels=3)
    g1, _ = _ring(top=1, left=1)
    g2, _ = _ring(top=3, left=3)
    _, f1 = enc.encode(g1)
    _, f2 = enc.encode(g2)
    assert f1.interior_mask != f2.interior_mask


def test_multiscale_levels_are_reported():
    enc = MultiscaleTopologicalEncoder(d_model=D, n_levels=5)
    g, _ = _ring()
    _, f = enc.encode(g)
    assert f.n_levels >= 2


def test_wave_contract_shape_and_dtype():
    enc = MultiscaleTopologicalEncoder(d_model=D, n_levels=3)
    g, _ = _ring()
    w, _ = enc.encode(g)
    assert len(w) == D
    assert all(isinstance(x, float) for x in w[:8])


def test_disabled_returns_the_flat_base_only():
    g, _ = _ring()
    on = MultiscaleTopologicalEncoder(d_model=D, n_levels=4, enabled=True)
    off = MultiscaleTopologicalEncoder(d_model=D, n_levels=4, enabled=False)
    w_on, f_on = on.encode(g)
    w_off, f_off = off.encode(g)
    assert f_off.n_levels == 1
    assert not any(any(r) for r in f_off.interior_mask)
    assert w_on != w_off          # multi-level must differ from flat


def test_live_encoder_separates_two_different_grids():
    enc = MultiscaleTopologicalEncoder(d_model=D, n_levels=3)
    g1, _ = _ring(ring_col=3)
    g2, _ = _ring(ring_col=4)
    w1, _ = enc.encode(g1)
    w2, _ = enc.encode(g2)
    assert cosine(w1, w2) < 0.999


def test_dead_input_control_cannot_separate_two_different_grids():
    """NEGATIVE CONTROL: content_blind derives every phase from the cell INDEX
    only, so two different grids must be INDISTINGUISHABLE."""
    blind = MultiscaleTopologicalEncoder(d_model=D, n_levels=3, content_blind=True)
    g1, _ = _ring(ring_col=3)
    g2, y2 = _ring(ring_col=4)
    w1, _ = blind.encode(g1)
    w2, _ = blind.encode(g2)
    assert cosine(w1, w2) > 0.9999, "dead-input control separated the inputs"
    live = MultiscaleTopologicalEncoder(d_model=D, n_levels=3)
    lw1, _ = live.encode(g1)
    lw2, _ = live.encode(g2)
    assert cosine(lw1, lw2) < cosine(w1, w2), "live must separate more than the control"


def test_not_position_invariant():
    """A grid and its own colour-swap must differ: the encoder reads content."""
    enc = MultiscaleTopologicalEncoder(d_model=D, n_levels=3)
    g, _ = _ring()
    swapped = [[(9 - c) for c in row] for row in g]
    w1, _ = enc.encode(g)
    w2, _ = enc.encode(swapped)
    assert cosine(w1, w2) < 0.999


def test_deterministic():
    enc = MultiscaleTopologicalEncoder(d_model=D, n_levels=3)
    g, _ = _ring()
    a, _ = enc.encode(g)
    b, _ = enc.encode(g)
    assert a == b


def test_component_count():
    enc = MultiscaleTopologicalEncoder(d_model=D, n_levels=3)
    g, _ = _ring()
    _, f = enc.encode(g)
    assert f.n_components == 1


def test_fail_closed_on_bad_input():
    enc = MultiscaleTopologicalEncoder(d_model=D, n_levels=3)
    with pytest.raises(TopologicalEncoderError):
        enc.encode([])
    with pytest.raises(TopologicalEncoderError):
        enc.encode([[1, 2], [3]])            # ragged
    with pytest.raises(TopologicalEncoderError):
        MultiscaleTopologicalEncoder(d_model=4095)   # odd


# END
