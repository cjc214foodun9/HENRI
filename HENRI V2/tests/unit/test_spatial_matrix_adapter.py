"""Directive 1 contract tests: SpatialMatrixAdapter (typed 2D-grid ingress).

THE VERIFICATION THE DIRECTIVE ASKS FOR
=======================================
"Ensure that training batches consume both sequential strings and 2D spatial
matrices with zero type errors."

That is tested directly: `collate_mixed` takes one iterable holding token
sequences AND 2D grids, and must return a batch whose counts, kinds, and tensor
dtypes are all correct. The load-bearing negative control is that a MALFORMED
item must RAISE -- a "zero type errors" claim is vacuous unless the adapter also
rejects bad input.

HONEST SCOPE
============
CPU only (this host has no CUDA). The adapter reuses HENRIVisionEncoder for the
wave construction; these tests check the ADAPTER's validation, dispatch, and
shape contract, not the encoder's physics (covered by its own suite).
"""

import os
import sys

import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import henri_spatial_matrix_adapter as A  # noqa: E402


@pytest.fixture(scope="module")
def adapter():
    # reduced scale for test speed; shape contract identical at D=65536.
    # GEOMETRY (measured): d_model == num_blocks * 8  =>  512 == 64 * 8
    return A.SpatialMatrixAdapter(d_model=512, num_blocks=64, device="cpu")


# ------------------------------------------------------------ geometry guards
def test_d_model_must_equal_num_blocks_times_8():
    with pytest.raises(A.SpatialIngressError, match="num_blocks"):
        A.SpatialMatrixAdapter(d_model=512, num_blocks=63, device="cpu")


def test_production_geometry_is_accepted():
    """65536 == 8192 * 8, the measured planner-boundary contract."""
    ad = A.SpatialMatrixAdapter(d_model=65536, num_blocks=8192, device="cpu")
    assert ad.d_model == ad.num_blocks * 8


def test_d_model_must_be_even():
    with pytest.raises(A.SpatialIngressError, match="even"):
        A.SpatialMatrixAdapter(d_model=513, num_blocks=64, device="cpu")


# --------------------------------------------------------------- single grid
def test_single_grid_returns_flat_unit_wave(adapter):
    w = adapter.to_wave([[0, 1], [2, 3]])
    assert w.shape == (512,)
    assert w.dtype == torch.float32
    assert abs(float(w.norm()) - 1.0) < 1e-4


def test_planner_view_is_num_blocks_by_8(adapter):
    w = adapter.to_planner_wave([[0, 1], [2, 3]])
    assert w.shape == (64, 8)
    assert abs(float(w.norm()) - 1.0) < 1e-4


def test_accepts_list_ndarray_and_tensor_equivalently(adapter):
    grid = [[1, 2, 3], [4, 5, 6]]
    a = adapter.to_wave(grid)
    b = adapter.to_wave(np.asarray(grid))
    c = adapter.to_wave(torch.tensor(grid))
    assert torch.allclose(a, b, atol=1e-6)
    assert torch.allclose(a, c, atol=1e-6)


# ------------------------------------------------------------- fail-closed
def test_raises_on_3d_input(adapter):
    with pytest.raises(A.SpatialIngressError, match="2-D"):
        adapter.to_wave(torch.zeros(2, 3, 4, dtype=torch.long))


def test_raises_on_non_integral_colours(adapter):
    with pytest.raises(A.SpatialIngressError, match="integral"):
        adapter.to_wave(np.array([[0.5, 1.0], [2.0, 3.0]]))


def test_raises_on_out_of_palette_colour(adapter):
    with pytest.raises(A.SpatialIngressError, match="colours must lie"):
        adapter.to_wave([[0, 16], [1, 2]])


def test_raises_on_negative_colour(adapter):
    with pytest.raises(A.SpatialIngressError, match="colours must lie"):
        adapter.to_wave([[0, -1], [1, 2]])


def test_raises_on_non_finite(adapter):
    with pytest.raises(A.SpatialIngressError, match="non-finite"):
        adapter.to_wave(np.array([[0.0, np.nan], [1.0, 2.0]]))


def test_raises_on_empty_grid(adapter):
    with pytest.raises(A.SpatialIngressError, match="non-empty"):
        adapter.to_wave(np.zeros((0, 3), dtype=np.int64))


def test_raises_on_grid_exceeding_arc_bound(adapter):
    with pytest.raises(A.SpatialIngressError, match="exceeds"):
        adapter.to_wave(np.zeros((31, 4), dtype=np.int64))


def test_raises_on_wrong_type(adapter):
    with pytest.raises(A.SpatialIngressError, match="list/tuple/ndarray/Tensor"):
        adapter.to_wave("not a grid")


# ------------------------------------------------------ MIXED BATCH (the D1 test)
def test_mixed_batch_consumes_strings_and_grids_with_zero_type_errors(adapter):
    """THE directive criterion: one batch, both kinds, no type errors."""
    items = [
        [7, 8, 9, 10],                      # 1-D token sequence
        [[0, 1], [2, 3]],                   # 2-D grid
        [1, 2, 3],                          # 1-D token sequence
        [[4, 5, 6], [7, 8, 9]],             # 2-D grid
    ]
    batch = adapter.collate_mixed(items)
    assert len(batch) == 4
    assert batch.counts() == {"seq": 2, "grid": 2}
    assert batch.kinds == ["seq", "grid", "seq", "grid"]

    tok = batch.to_token_batch()
    assert tok.dtype == torch.int64
    assert tok.shape == (2, 4)              # padded to the longest sequence
    assert tok[1].tolist() == [1, 2, 3, 0]

    waves = batch.to_wave_batch()
    assert waves.dtype == torch.float32
    assert waves.shape == (2, 512)


def test_mixed_batch_is_dispatch_by_ndim_not_by_guesswork(adapter):
    """A 1-D item is NEVER treated as a grid and vice versa."""
    batch = adapter.collate_mixed([[0, 1, 2], [[9, 9], [9, 9]]])
    assert batch.counts() == {"seq": 1, "grid": 1}
    assert batch.grids[0].shape == (2, 2)
    assert batch.seqs[0].shape == (3,)


def test_mixed_batch_rejects_a_3d_item(adapter):
    with pytest.raises(A.SpatialIngressError, match="neither a 1-D sequence nor a 2-D grid"):
        adapter.collate_mixed([[1, 2, 3], torch.zeros(2, 2, 2, dtype=torch.long)])


def test_sequence_validation_is_fail_closed(adapter):
    with pytest.raises(A.SpatialIngressError, match="non-negative"):
        adapter.collate_mixed([[1, -5, 3]])
    with pytest.raises(A.SpatialIngressError, match="non-empty"):
        adapter.collate_mixed([[]])


def test_token_batch_dtype_is_int64_for_uint8_input(adapter):
    batch = adapter.collate_mixed([np.array([1, 2, 3], dtype=np.uint8)])
    assert batch.to_token_batch().dtype == torch.int64


def test_describe_states_what_it_does_not_claim(adapter):
    d = adapter.describe()
    assert d["schema"] == "henri.spatial-matrix-adapter.v1"
    assert "does NOT raise" in d["does_not_claim"]


# ------------------------------------------------ production-scale smoke
def test_production_scale_adapter_shape():
    """D=65536, 8192 blocks: the real planner boundary, one grid."""
    ad = A.SpatialMatrixAdapter(d_model=65536, num_blocks=8192, device="cpu")
    w = ad.to_wave([[0, 1, 2], [3, 4, 5]])
    assert w.numel() == 65536
    assert ad.to_planner_wave([[0, 1, 2], [3, 4, 5]]).shape == (8192, 8)
