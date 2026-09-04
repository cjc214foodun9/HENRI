"""Contracts for the frozen semantic-backbone wave boundary.

This suite uses a tiny local fixture. It proves software invariants only. It
is not evidence that a backbone contains useful world knowledge.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from henri_semantic_backbone import (
    FactorizedSemanticWaveAdapter,
    FrozenBackboneProvenance,
    FrozenSemanticBackbone,
    SemanticContractError,
)


class TinyBackbone(nn.Module):
    def __init__(self, width=8):
        super().__init__()
        self.proj = nn.Linear(width, width, bias=False)

    def forward(self, x, attention_mask=None):
        return SimpleNamespace(last_hidden_state=self.proj(x))


def provenance():
    return FrozenBackboneProvenance(
        model_id="test/tiny-semantic-backbone",
        revision="0123456789abcdef0123456789abcdef01234567",
        artifact_sha256="a" * 64,
        artifact_bytes=123,
        hidden_size=8,
        source="test-fixture",
    )


def test_factorized_adapter_has_bounded_storage_and_effective_rank():
    adapter = FactorizedSemanticWaveAdapter(
        semantic_dim=8, num_blocks=4, block_dim=8, rank=64
    )
    assert adapter.effective_rank == 8
    assert adapter.projection_bytes < 4 * 8 * (4 * 8) * 4
    assert adapter.projection_bytes == (8 * 8 + 32 * 8) * 4


def test_adapter_is_zero_trainable():
    adapter = FactorizedSemanticWaveAdapter(
        semantic_dim=8, num_blocks=4, block_dim=8, rank=4
    )
    assert list(adapter.parameters()) == []
    assert all(not b.requires_grad for b in adapter.buffers())


def test_adapter_emits_canonical_shape_and_global_unit_norm():
    adapter = FactorizedSemanticWaveAdapter(
        semantic_dim=8, num_blocks=4, block_dim=8, rank=4
    )
    out = adapter(torch.randn(3, 8))
    assert out.shape == (3, 4, 8)
    assert torch.allclose(out.flatten(1).norm(dim=1), torch.ones(3), atol=1e-5)


def test_adapter_is_deterministic_for_same_seed():
    a = FactorizedSemanticWaveAdapter(8, 4, 8, rank=4, seed=19)
    b = FactorizedSemanticWaveAdapter(8, 4, 8, rank=4, seed=19)
    assert torch.equal(a.left_basis, b.left_basis)
    assert torch.equal(a.right_basis, b.right_basis)


def test_adapter_rejects_wrong_feature_shape():
    adapter = FactorizedSemanticWaveAdapter(8, 4, 8, rank=4)
    with pytest.raises(SemanticContractError):
        adapter(torch.randn(2, 7))


def test_provenance_rejects_unpinned_metadata():
    with pytest.raises(SemanticContractError):
        FrozenBackboneProvenance(
            model_id="test/tiny",
            revision="main",
            artifact_sha256="a" * 64,
            artifact_bytes=1,
            hidden_size=8,
            source="test",
        )


def test_frozen_backbone_requires_no_trainable_parameters():
    backbone = TinyBackbone()
    with pytest.raises(SemanticContractError):
        FrozenSemanticBackbone(backbone, provenance(), wave_rank=4, num_blocks=4)


def test_frozen_backbone_masks_padding_before_wave_projection():
    backbone = TinyBackbone()
    for p in backbone.parameters():
        p.requires_grad_(False)
    wrapper = FrozenSemanticBackbone(
        backbone, provenance(), wave_rank=4, num_blocks=4
    )
    x = torch.randn(2, 3, 8)
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.long)
    out = wrapper(x, attention_mask=mask)
    assert out.shape == (2, 4, 8)
    assert torch.isfinite(out).all()


def test_frozen_backbone_is_eval_and_zero_grad():
    backbone = TinyBackbone()
    for p in backbone.parameters():
        p.requires_grad_(False)
    wrapper = FrozenSemanticBackbone(
        backbone, provenance(), wave_rank=4, num_blocks=4
    )
    assert not wrapper.backbone.training
    assert all(not p.requires_grad for p in wrapper.backbone.parameters())
    out = wrapper(torch.randn(2, 1, 8), attention_mask=torch.ones(2, 1))
    assert not out.requires_grad


def test_provenance_file_receipt(tmp_path):
    payload = b"immutable-backbone-fixture"
    path = tmp_path / "weights.bin"
    path.write_bytes(payload)
    import hashlib

    receipt = FrozenBackboneProvenance(
        model_id="test/tiny",
        revision="0123456789abcdef0123456789abcdef01234567",
        artifact_sha256=hashlib.sha256(payload).hexdigest(),
        artifact_bytes=len(payload),
        hidden_size=8,
        source="test",
    )
    assert receipt.verify_artifact(path) is True
    assert receipt.artifact_bytes == len(payload)
