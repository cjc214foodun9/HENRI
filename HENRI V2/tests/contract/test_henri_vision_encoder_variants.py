"""Phase 7.3 G1 split-candidate contracts: encoder-basis variants (default-OFF).

Deterministic contracts for the accepted invertibility mechanism
(release/phase73-g1-invertibility):
- default constructor == legacy byte identity;
- flag -> field -> computational branch -> changed output (causal chain);
- color-0 exclusion is actually consumed by encode_grid (no dead flag);
- same-sum coordinates become distinct under incommensurate/random ramps;
- invalid spatial_basis_kind fails closed;
- empty foreground under bg_mask fails closed (no NaN waves).

Note: the D=65,536 gate values are measured by the remote real-encoder probe;
these contracts run at reduced scale and assert structure, not gate numbers.
"""
import pytest
import torch

from henri_vision_encoder import HENRIVisionEncoder

D_SMALL = 1024
K_SMALL = 128


def _encode(enc, grid):
    with torch.no_grad():
        return enc.encode_grid(grid).to(torch.float32)


def _single_pixel(rows, cols, r, c, color=5):
    g = [[0] * cols for _ in range(rows)]
    g[r][c] = color
    return g


def _grid(*rows):
    return [list(r) for r in rows]


def test_default_constructor_byte_identity():
    legacy = HENRIVisionEncoder(d_model=D_SMALL, k_blocks=K_SMALL, device="cpu")
    defaulted = HENRIVisionEncoder(
        d_model=D_SMALL, k_blocks=K_SMALL, device="cpu",
        spatial_basis_kind="default", bg_mask=False,
    )
    grid = _grid((0, 0, 5, 0), (3, 0, 0, 0), (0, 0, 12, 0), (0, 0, 0, 0))
    w_legacy = _encode(legacy, grid)
    w_default = _encode(defaulted, grid)
    assert torch.equal(legacy.spatial_basis_x, defaulted.spatial_basis_x)
    assert torch.equal(legacy.spatial_basis_y, defaulted.spatial_basis_y)
    assert torch.max(torch.abs(w_legacy - w_default)).item() == 0.0


def test_flag_to_field_to_branch_to_output():
    base = HENRIVisionEncoder(d_model=D_SMALL, k_blocks=K_SMALL, device="cpu")
    inc = HENRIVisionEncoder(
        d_model=D_SMALL, k_blocks=K_SMALL, device="cpu",
        spatial_basis_kind="incommensurate", bg_mask=True,
    )
    # flag -> field: y ramp differs from x ramp
    assert not torch.allclose(base.spatial_basis_y[1], inc.spatial_basis_y[1])
    # branch -> changed output on a backgrounded grid
    grid = _grid((0, 5, 0), (0, 0, 0), (0, 0, 12))
    w_base = _encode(base, grid)
    w_inc = _encode(inc, grid)
    assert torch.max(torch.abs(w_base - w_inc)).item() > 1e-6


def test_bg_mask_consumed_by_encode_grid():
    enc = HENRIVisionEncoder(
        d_model=D_SMALL, k_blocks=K_SMALL, device="cpu", bg_mask=True,
    )
    unmasked = HENRIVisionEncoder(d_model=D_SMALL, k_blocks=K_SMALL, device="cpu")
    no_bg = _grid((5, 5), (5, 5))
    with_bg = _grid((0, 5), (0, 5))
    # Foreground-only grid: mask is a no-op, byte-identical to unmasked.
    assert torch.max(torch.abs(_encode(enc, no_bg) - _encode(unmasked, no_bg))).item() == 0.0
    # Backgrounded grid: mask changes the output (DC offset removed).
    assert torch.max(torch.abs(_encode(enc, with_bg) - _encode(unmasked, with_bg))).item() > 1e-6


def test_same_sum_coordinates_distinct_under_variant_bases():
    for kind in ("incommensurate", "random"):
        enc = HENRIVisionEncoder(
            d_model=D_SMALL, k_blocks=K_SMALL, device="cpu",
            spatial_basis_kind=kind, bg_mask=True,
        )
        w12 = _encode(enc, _single_pixel(6, 6, 1, 2))
        w21 = _encode(enc, _single_pixel(6, 6, 2, 1))
        cos = float(torch.dot(w12, w21).item())
        assert cos < 0.5, f"{kind}: same-sum cos {cos:.4f} not distinct"


def test_invalid_basis_kind_fails_closed():
    with pytest.raises(ValueError):
        HENRIVisionEncoder(
            d_model=D_SMALL, k_blocks=K_SMALL, device="cpu",
            spatial_basis_kind="bogus",
        )


def test_empty_foreground_under_bg_mask_fails_closed():
    enc = HENRIVisionEncoder(
        d_model=D_SMALL, k_blocks=K_SMALL, device="cpu", bg_mask=True,
    )
    with pytest.raises(ValueError):
        _encode(enc, _grid((0, 0), (0, 0)))
