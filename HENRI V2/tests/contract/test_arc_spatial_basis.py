"""Phase 7.8 P0-A1 contracts: production encoder-basis defaults (G1 ACCEPTED).

Pins:
- production default resolver = ("incommensurate", True) with no env;
- legacy opt-in = ("default", False) via BOTH explicit env vars;
- invalid kind fails closed;
- the production-default combination is causally consumed by encode_grid
  (distinct y ramp; output differs from legacy; legacy explicit combination
  remains byte-identical).
"""
import pytest
import torch

from arc_spatial_basis import resolve_spatial_basis
from henri_vision_encoder import HENRIVisionEncoder

D_SMALL = 1024
K_SMALL = 128


def test_production_default_is_incommensurate_with_bg_mask(monkeypatch):
    monkeypatch.delenv("HENRI_ARC_SPATIAL_BASIS", raising=False)
    monkeypatch.delenv("HENRI_ARC_BG_MASK", raising=False)
    assert resolve_spatial_basis() == ("incommensurate", True)


def test_legacy_opt_in_via_both_explicit_env(monkeypatch):
    monkeypatch.setenv("HENRI_ARC_SPATIAL_BASIS", "default")
    monkeypatch.setenv("HENRI_ARC_BG_MASK", "0")
    assert resolve_spatial_basis() == ("default", False)


def test_invalid_kind_fails_closed(monkeypatch):
    monkeypatch.setenv("HENRI_ARC_SPATIAL_BASIS", "bogus")
    with pytest.raises(ValueError):
        resolve_spatial_basis()


def _encode(enc, grid):
    with torch.no_grad():
        return enc.encode_grid(grid).to(torch.float32)


def _grid(*rows):
    return [list(r) for r in rows]


def test_production_default_combo_changes_output_and_consumes_bg():
    legacy = HENRIVisionEncoder(d_model=D_SMALL, k_blocks=K_SMALL, device="cpu")
    prod = HENRIVisionEncoder(
        d_model=D_SMALL, k_blocks=K_SMALL, device="cpu",
        spatial_basis_kind="incommensurate", bg_mask=True,
    )
    # flag -> field: y ramp differs from x ramp (collinear degeneracy broken)
    assert not torch.allclose(prod.spatial_basis_y[1], legacy.spatial_basis_y[1])
    grid = _grid((0, 0, 5, 0), (3, 0, 0, 0), (0, 0, 12, 0), (0, 0, 0, 0))
    w_legacy = _encode(legacy, grid)
    w_prod = _encode(prod, grid)
    # bg exclusion is consumed: output changes on grids with color-0 cells
    assert torch.max(torch.abs(w_legacy - w_prod)).item() > 1e-6


def test_legacy_explicit_still_byte_identical():
    legacy = HENRIVisionEncoder(d_model=D_SMALL, k_blocks=K_SMALL, device="cpu")
    opt = HENRIVisionEncoder(
        d_model=D_SMALL, k_blocks=K_SMALL, device="cpu",
        spatial_basis_kind="default", bg_mask=False,
    )
    grid = _grid((0, 0, 5, 0), (3, 0, 0, 0), (0, 0, 12, 0), (0, 0, 0, 0))
    assert torch.equal(legacy.spatial_basis_x, opt.spatial_basis_x)
    assert torch.equal(legacy.spatial_basis_y, opt.spatial_basis_y)
    assert torch.max(torch.abs(_encode(legacy, grid) - _encode(opt, grid))).item() == 0.0


def test_empty_foreground_fails_closed():
    prod = HENRIVisionEncoder(
        d_model=D_SMALL, k_blocks=K_SMALL, device="cpu",
        spatial_basis_kind="incommensurate", bg_mask=True,
    )
    with pytest.raises(ValueError):
        _encode(prod, [[0, 0], [0, 0]])
