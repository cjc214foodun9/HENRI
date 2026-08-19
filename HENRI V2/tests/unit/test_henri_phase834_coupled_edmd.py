"""Phase 8.34 unit tests: CoupledRecursiveDualEDMD + lexical_snap (reduced dims).

Toy-gate design follows Phase 5 P1 lessons: production learning rule
(update_online_step / store_engrams), mechanism discrimination (field-dominant
teacher the block-diag control structurally cannot represent), held-out split,
cross-block Jacobian proof, rank contract.
"""

import math

import pytest
import torch
import torch.nn.functional as F

from hopfield_cleanup import ContinuousHopfieldCleanup
from recursive_dual_edmd import CoupledRecursiveDualEDMD, RecursiveDualEDMD


# ---------------- CoupledRecursiveDualEDMD ----------------

def _teacher_f(d: int, rank: int = 8, seed: int = 3) -> torch.Tensor:
    """Rank-8 field-dominant teacher: dense low-rank coupling the per-block
    control cannot represent (Phase 5: field-DOMINANT teacher)."""
    g = torch.Generator().manual_seed(seed)
    U = torch.randn(d, rank, generator=g)
    V = torch.randn(d, rank, generator=g)
    return (U @ V.T) / math.sqrt(d)


def _make_data(d: int, n_train: int, n_held: int, seed: int = 11):
    g = torch.Generator().manual_seed(seed)
    Fmat = _teacher_f(d)
    xs = F.normalize(torch.randn(n_train + n_held, d, generator=g), p=2, dim=-1)
    ys = F.normalize(xs @ Fmat.T, p=2, dim=-1)
    return xs[:n_train], ys[:n_train], xs[n_train:], ys[n_train:]


def _fit(arm: CoupledRecursiveDualEDMD, xs, ys, blocks: int, bdim: int):
    for x, y in zip(xs, ys):
        arm.update_online_step(
            x.view(blocks, bdim), torch.zeros(blocks, bdim), y.view(blocks, bdim)
        )


def _heldout_loss(arm, xs, ys, blocks: int, bdim: int) -> float:
    with torch.no_grad():
        losses = []
        for x, y in zip(xs, ys):
            pred = arm.forward(x.view(blocks, bdim), torch.zeros(blocks, bdim))
            losses.append(
                1.0 - F.normalize(pred.view(-1), p=2, dim=0)
                @ F.normalize(y.view(-1), p=2, dim=0)
            )
        return float(sum(losses) / len(losses))


def test_coupled_edmd_rank_contract():
    with pytest.raises(TypeError):
        CoupledRecursiveDualEDMD(d_model=64, r_rank=True)
    with pytest.raises(TypeError):
        CoupledRecursiveDualEDMD(d_model=64, r_rank=8.5)
    with pytest.raises(ValueError):
        CoupledRecursiveDualEDMD(d_model=64, r_rank=0)
    m = CoupledRecursiveDualEDMD(d_model=64, r_rank=128)
    assert m.requested_rank == 128
    assert m.r_rank == 64  # effective rank clamp
    assert m.V.shape == (64, 64)


def test_coupled_edmd_cross_block_jacobian():
    d, blocks, bdim, r = 64, 8, 8, 8
    xs_tr, ys_tr, xs_ho, _ = _make_data(d, 60, 20)
    coupled = CoupledRecursiveDualEDMD(d_model=d, r_rank=r, num_blocks=blocks,
                                       block_dim=bdim, field_channel=True)
    control = CoupledRecursiveDualEDMD(d_model=d, r_rank=r, num_blocks=blocks,
                                       block_dim=bdim, field_channel=False)
    _fit(coupled, xs_tr, ys_tr, blocks, bdim)
    _fit(control, xs_tr, ys_tr, blocks, bdim)
    s = xs_ho[0].view(blocks, bdim)
    a = torch.zeros(blocks, bdim)
    j_full = coupled.cross_block_jacobian(s, a, block_a=0, block_b=7, include_field=True)
    j_nofield = coupled.cross_block_jacobian(s, a, block_a=0, block_b=7, include_field=False)
    # Dense-V baseline couples globally even without the field channel.
    assert j_nofield > 1e-6, f"dense-V baseline off-block Jacobian {j_nofield} missing"
    # The field channel must add measurable coupling beyond dense V.
    assert j_full - j_nofield > 1e-6, f"field-attributable coupling {j_full - j_nofield:.3e} not engaged"
    # Control arm (no field channel) equals the dense-V baseline magnitude.
    j_ctrl = control.cross_block_jacobian(s, a, block_a=0, block_b=7)
    assert abs(j_ctrl - j_nofield) < 0.2 * max(j_ctrl, 1e-9), (
        f"control {j_ctrl:.4f} diverges from no-field baseline {j_nofield:.4f}")


def test_coupled_edmd_mechanism_discrimination():
    """Field-dominant teacher: coupled held-out loss < block-diag control."""
    d, blocks, bdim, r = 64, 8, 8, 8
    xs_tr, ys_tr, xs_ho, ys_ho = _make_data(d, 60, 20, seed=17)
    coupled = CoupledRecursiveDualEDMD(d_model=d, r_rank=r, num_blocks=blocks,
                                       block_dim=bdim, field_channel=True)
    control = CoupledRecursiveDualEDMD(d_model=d, r_rank=r, num_blocks=blocks,
                                       block_dim=bdim, field_channel=False)
    _fit(coupled, xs_tr, ys_tr, blocks, bdim)
    _fit(control, xs_tr, ys_tr, blocks, bdim)
    l_c = _heldout_loss(coupled, xs_ho, ys_ho, blocks, bdim)
    l_ctrl = _heldout_loss(control, xs_ho, ys_ho, blocks, bdim)
    # Mechanism criterion (Phase 5): coupled < block-diag control. No absolute
    # floor — absolute thresholds on toy harnesses produce spurious KILLs.
    assert l_c < l_ctrl, f"coupled {l_c:.4f} not < control {l_ctrl:.4f}"


# ---------------- lexical_snap ----------------

def test_lexical_snap_single_and_batch():
    dim, m = 128, 6
    g = torch.Generator().manual_seed(5)
    engrams = F.normalize(torch.randn(m, dim, generator=g), p=2, dim=-1)
    cleanup = ContinuousHopfieldCleanup(dim=dim)
    cleanup.store_engrams(engrams)
    idx, conf = cleanup.lexical_snap(engrams[3])
    assert int(idx) == 3
    assert abs(float(conf) - 1.0) < 1e-3
    # Batch
    idx_b, conf_b = cleanup.lexical_snap(engrams[:4])
    assert idx_b.tolist() == [0, 1, 2, 3]
    assert conf_b.min().item() > 0.99


def test_lexical_snap_top_k():
    dim, m = 64, 5
    g = torch.Generator().manual_seed(6)
    engrams = F.normalize(torch.randn(m, dim, generator=g), p=2, dim=-1)
    cleanup = ContinuousHopfieldCleanup(dim=dim)
    cleanup.store_engrams(engrams)
    idx, conf = cleanup.lexical_snap(engrams[0], top_k=3)
    assert idx.shape == (3,)
    assert int(idx[0]) == 0
    assert conf[0] >= conf[1] >= conf[2]


def test_lexical_snap_requires_engrams():
    cleanup = ContinuousHopfieldCleanup(dim=32)
    with pytest.raises(AssertionError):
        cleanup.lexical_snap(torch.randn(32))
