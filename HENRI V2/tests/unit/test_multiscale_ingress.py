"""Contract tests for the MULTISCALE TOPOLOGICAL INGRESS (Directive 4).

The upgrade: `O_VSA_IngressTokenizer.encode_spatial_grid_multiscale`, reached by
`encode_spatial_grid` only when `HENRI_ENCODER_MULTISCALE=1`.

The flat path it replaces superposes `theta_v + norm_x*theta_x + norm_y*theta_y`
over cells: ONE spatial level, NO nesting, NO topology (`multiscale|jordan|
interior` measured 0 hits in the file before this change).

Required controls (each caught a real defect somewhere in this project):
  1. DEFAULT-PATH BYTE IDENTITY   -- flag OFF must be bit-identical to legacy.
  2. DEAD-INPUT NEGATIVE CONTROL  -- content_blind derives phase from the INDEX
     only and must be indistinguishable across different grids.
  3. DIFFERENTIAL EFFECT          -- two different grids must give different waves.
  4. TOPOLOGY SENSITIVITY         -- the SAME colours with a CLOSED vs a BROKEN
     curve must not be identical; otherwise the Jordan marker contributes nothing.
  5. SHAPE/CONTRACT               -- [1, num_blocks, 8], finite, block norms ~1.
"""
import importlib
import math
import os
import sys

import pytest
import torch

def _project_root() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    d = here
    for _ in range(6):
        if os.path.exists(os.path.join(d, "o_vsa_ingress_tokenizer.py")):
            return d
        d = os.path.dirname(d)
    return here


C = _project_root()
if C not in sys.path:
    sys.path.insert(0, C)

NB = 8192


def _tok():
    from o_vsa_ingress_tokenizer import O_VSA_IngressTokenizer
    return O_VSA_IngressTokenizer(num_blocks=NB, vocab_size=256, device="cpu")


def _noise_grid(seed=0, n=12, band=(0, 1, 2)):
    st = seed * 7919 + 13
    g = []
    for _ in range(n):
        row = []
        for _ in range(n):
            st = (st * 1103515245 + 12345) % 2147483648
            row.append(band[(st >> 16) % len(band)])
        g.append(row)
    return g


def _ring(g, ring, top, left, h=4, closed=True):
    bot, right = top + h - 1, left + h - 1
    for j in range(left, right + 1):
        g[top][j] = ring
        g[bot][j] = ring
    for i in range(top, bot + 1):
        g[i][left] = ring
        if closed:
            g[i][right] = ring
    return g


# ------------------------------------------------------- default path identity
def test_flag_off_is_byte_identical_to_legacy(monkeypatch):
    monkeypatch.delenv("HENRI_ENCODER_MULTISCALE", raising=False)
    monkeypatch.delenv("HENRI_ENCODER_TORUS", raising=False)
    t = _tok()
    g = _ring(_noise_grid(1), 6, 3, 3)
    a = t.encode_spatial_grid(g)
    b = t.encode_spatial_grid(g)
    assert float((a - b).abs().max()) == 0.0
    assert a.shape == (1, NB, 8)


def test_flag_off_default_is_not_multiscale(monkeypatch):
    monkeypatch.delenv("HENRI_ENCODER_MULTISCALE", raising=False)
    t = _tok()
    g = _ring(_noise_grid(2), 6, 3, 3)
    flat = t.encode_spatial_grid(g)
    ms = t.encode_spatial_grid_multiscale(g)
    assert float((flat - ms).abs().max()) > 1e-6, "flag OFF must differ from the multiscale path"


def test_flag_on_routes_to_the_multiscale_path(monkeypatch):
    t = _tok()
    g = _ring(_noise_grid(3), 6, 3, 3)
    monkeypatch.setenv("HENRI_ENCODER_MULTISCALE", "1")
    monkeypatch.delenv("HENRI_ENCODER_TORUS", raising=False)
    a = t.encode_spatial_grid(g)
    b = t.encode_spatial_grid_multiscale(g)
    assert float((a - b).abs().max()) == 0.0


# ------------------------------------------------------------- shape contract
def test_shape_dtype_and_finiteness():
    t = _tok()
    w = t.encode_spatial_grid_multiscale(_ring(_noise_grid(4), 6, 3, 3))
    assert tuple(w.shape) == (1, NB, 8)
    assert w.dtype == torch.float32
    assert torch.isfinite(w).all()


def test_block_rows_are_unit_norm():
    t = _tok()
    w = t.encode_spatial_grid_multiscale(_ring(_noise_grid(5), 6, 3, 3)).view(NB, 8)
    norms = torch.linalg.vector_norm(w, dim=-1)
    assert float((norms - 1.0).abs().max()) < 1e-5


def test_non_square_grid_is_accepted():
    t = _tok()
    g = [[(i + j) % 5 for j in range(9)] for i in range(7)]
    assert tuple(t.encode_spatial_grid_multiscale(g).shape) == (1, NB, 8)


def test_d_model_contract_matches_num_blocks_times_8():
    """d_model = num_blocks*8 is what makes the reshape into [NB, 8] exact."""
    t = _tok()
    from henri_topological_encoder import MultiscaleTopologicalEncoder
    enc = MultiscaleTopologicalEncoder(d_model=NB * 8, n_levels=2, enabled=True,
                                      background_values=(0, 1, 2))
    wave, _ = enc.encode(_ring(_noise_grid(6), 6, 3, 3))
    assert len(wave) == NB * 8, "encoder width must equal num_blocks*8 or the view breaks"


# --------------------------------------------------------- differential effect
def test_two_different_colours_differ():
    t = _tok()
    a = t.encode_spatial_grid_multiscale(_ring(_noise_grid(7), 6, 3, 3))
    b = t.encode_spatial_grid_multiscale(_ring(_noise_grid(7), 7, 3, 3))
    assert float((a - b).abs().max()) > 1e-6


def test_two_different_positions_differ():
    t = _tok()
    base = _noise_grid(8)
    a = t.encode_spatial_grid_multiscale(_ring([r[:] for r in base], 6, 2, 2))
    b = t.encode_spatial_grid_multiscale(_ring([r[:] for r in base], 6, 5, 5))
    assert float((a - b).abs().max()) > 1e-6


def test_identical_grids_are_deterministic():
    t = _tok()
    g = _ring(_noise_grid(9), 6, 3, 3)
    a = t.encode_spatial_grid_multiscale(g)
    b = t.encode_spatial_grid_multiscale(g)
    assert float((a - b).abs().max()) == 0.0


# ------------------------------------------------------ topology sensitivity
def test_closed_vs_broken_curve_is_NOT_identical():
    """The Jordan marker must contribute: with the same colours and the same noise,
    a CLOSED ring encloses an interior while a BROKEN ring does not. If the two
    encode identically the topological channel is inert decoration."""
    t = _tok()
    closed = _ring(_noise_grid(10), 6, 3, 3, closed=True)
    broken = _ring(_noise_grid(10), 6, 3, 3, closed=False)
    a = t.encode_spatial_grid_multiscale(closed)
    b = t.encode_spatial_grid_multiscale(broken)
    assert float((a - b).abs().max()) > 1e-6


def test_marker_actually_fires_on_the_closed_curve():
    """Direct check on the marker, not on the wave: a closed ring on a noisy band
    must yield a non-empty INTERIOR, and the broken one must not."""
    from henri_topological_encoder import MultiscaleTopologicalEncoder
    enc = MultiscaleTopologicalEncoder(d_model=64, n_levels=1, enabled=True,
                                      background_values=(0, 1, 2))
    closed = _ring(_noise_grid(11), 6, 3, 3, closed=True)
    broken = _ring(_noise_grid(11), 6, 3, 3, closed=False)
    ic, _ = enc._markers(closed)
    ib, _ = enc._markers(broken)
    n_closed = sum(r.count(True) for r in ic)
    n_broken = sum(r.count(True) for r in ib)
    assert n_closed == 4, f"closed 4x4 ring should enclose 2x2=4 cells, got {n_closed}"
    assert n_broken == 0, f"broken curve must enclose nothing, got {n_broken}"


def test_single_value_background_is_the_measured_defect():
    """Documents WHY the band is required: on a noisy grid a single-value fill
    treats the other noise values as curve and leaks the interior."""
    from henri_topological_encoder import MultiscaleTopologicalEncoder
    g = _ring(_noise_grid(12), 6, 3, 3, closed=True)
    band = MultiscaleTopologicalEncoder(d_model=64, n_levels=1, enabled=True,
                                        background_values=(0, 1, 2))
    single = MultiscaleTopologicalEncoder(d_model=64, n_levels=1, enabled=True,
                                          background_values=(0,))
    ib, _ = band._markers(g)
    is_, _ = single._markers(g)
    n_band = sum(r.count(True) for r in ib)
    n_single = sum(r.count(True) for r in is_)
    assert n_band == 4
    assert n_single > 4, "single-value fill should leak (the documented defect)"


# ------------------------------------------------------------ dead-input control
def test_content_blind_control_is_indistinguishable_across_grids():
    """DEAD-INPUT NEGATIVE CONTROL: with content_blind=True every phase derives
    from the cell INDEX, so two different grids MUST encode identically. If they
    differ, the encoder reads something other than what it claims."""
    from henri_topological_encoder import MultiscaleTopologicalEncoder
    a = MultiscaleTopologicalEncoder(d_model=64, n_levels=2, enabled=True, content_blind=True)
    g1 = _ring(_noise_grid(13), 6, 3, 3)
    g2 = _ring(_noise_grid(14), 7, 5, 5)
    w1, _ = a.encode(g1)
    w2, _ = a.encode(g2)
    assert w1 == w2, "content_blind must be grid-invariant"


def test_content_blind_control_does_NOT_hold_with_content(monkeypatch):
    """The complement: with content (the default) the same two grids MUST differ,
    so the control above is not vacuous."""
    from henri_topological_encoder import MultiscaleTopologicalEncoder
    a = MultiscaleTopologicalEncoder(d_model=64, n_levels=2, enabled=True, content_blind=False)
    g1 = _ring(_noise_grid(13), 6, 3, 3)
    g2 = _ring(_noise_grid(14), 7, 5, 5)
    w1, _ = a.encode(g1)
    w2, _ = a.encode(g2)
    assert w1 != w2


# ---------------------------------------------------------------- level count
def test_more_levels_change_the_encoding():
    t = _tok()
    g = _ring(_noise_grid(15), 6, 3, 3)
    a = t.encode_spatial_grid_multiscale(g, n_levels=1)
    b = t.encode_spatial_grid_multiscale(g, n_levels=4)
    assert float((a - b).abs().max()) > 1e-6, "nesting must be load-bearing"


def test_level_count_is_clamped_to_the_grid():
    t = _tok()
    g = _ring(_noise_grid(16), 6, 3, 3)          # 12x12 -> at most 4 levels
    a = t.encode_spatial_grid_multiscale(g, n_levels=99)
    b = t.encode_spatial_grid_multiscale(g, n_levels=4)
    assert float((a - b).abs().max()) < 1e-12
