"""Phase 8.7 contract tests — typed action embeddings (8.7-A) + valence-free
pre-training (8.7-B). Default-OFF diagnostics; production path untouched.
Guards: typed waves on-manifold, deterministic, quasi-orthogonal carriers;
Clifford bind NON-commutative; production FHRR bind UNCHANGED (commutative)."""

import torch
import torch.nn.functional as F

from henri_typed_actions import (
    TypedActionEmbedding,
    clifford_bind,
    CliffordTransition,
    _torque_level,
)
from efe_planner import LowRankCoupledTransition


def test_typed_embedding_shape_and_unit_norm():
    emb = TypedActionEmbedding(num_actions=8, num_blocks=64, block_dim=8)
    w = emb.embed(torch.tensor([0, 1, 2]))
    assert w.shape == (3, 64, 8), f"shape {w.shape}"
    norms = w.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), "per-block unit norm"


def test_typed_embedding_deterministic():
    emb = TypedActionEmbedding(num_actions=4, num_blocks=32, block_dim=8)
    a = emb.embed(torch.tensor([3]))
    b = emb.embed(torch.tensor([3]))
    assert torch.equal(a, b), "typed action wave must be deterministic per token"


def test_typed_embedding_carriers_quasi_orthogonal():
    emb = TypedActionEmbedding(num_actions=4, num_blocks=128, block_dim=8)
    w0 = emb.embed(torch.tensor([0])).reshape(-1)
    w1 = emb.embed(torch.tensor([1])).reshape(-1)
    cos = F.cosine_similarity(w0, w1, dim=0).item()
    assert abs(cos) < 0.3, f"action carriers not quasi-orthogonal: {cos:.4f}"


def test_clifford_bind_noncommutative():
    s = F.normalize(torch.randn(4, 8), dim=-1)
    a = F.normalize(torch.randn(4, 8), dim=-1)
    ab = clifford_bind(s, a)
    ba = clifford_bind(a, s)
    assert not torch.allclose(ab, ba, atol=1e-3), "Clifford bind must be NON-commutative"


def test_clifford_bind_unit_norm():
    s = F.normalize(torch.randn(16, 8), dim=-1)
    a = F.normalize(torch.randn(16, 8), dim=-1)
    b = clifford_bind(s, a)
    assert torch.allclose(b.norm(dim=-1), torch.ones(16), atol=1e-5)


def test_clifford_transition_forward():
    tr = CliffordTransition(num_blocks=8, block_dim=8, rank=2)
    s = F.normalize(torch.randn(8, 8), dim=-1)
    a = F.normalize(torch.randn(8, 8), dim=-1)
    pred = tr.forward(s, a)
    assert pred.shape == (8, 8), f"shape {pred.shape}"
    assert torch.allclose(pred.norm(dim=-1), torch.ones(8), atol=1e-4)
    assert torch.isfinite(pred).all()


def test_production_fhrr_bind_commutative_unchanged():
    """Guard: the production LowRankCoupledTransition.bind is FHRR circular
    convolution (COMMUTATIVE). The 8.7 Clifford bind must NOT have touched it."""
    tr = LowRankCoupledTransition(num_blocks=8, block_dim=8, rank=2)
    s = F.normalize(torch.randn(8, 8), dim=-1)
    a = F.normalize(torch.randn(8, 8), dim=-1)
    ab = tr.bind(s, a)
    ba = tr.bind(a, s)
    assert torch.allclose(ab, ba, atol=1e-4), "production FHRR bind must remain commutative"


def test_torque_level_bounded():
    for k in range(8):
        t = _torque_level(k, 8, max_torque=10.0)
        assert -10.0 <= t <= 10.0, f"torque {t} out of bounds"
    assert _torque_level(0, 8) < 0.0 and _torque_level(7, 8) > 0.0
