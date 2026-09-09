"""Contract tests for E2 k-NN softmax egress probe (carrier/e2-egress-knn-scale).

TDD order: tests first. E2 module is default-OFF and never imported by
production_arc_run.py / henri_decoder.py / henri_egress.py.

Covers: probe geometry (k, tau frozen), zero-trainable, softmax normalization,
dense [D,D] ban, margin semantics (+0.05 gate), negative control (shuffled
targets must lose), empty-input rejection, default-OFF non-import.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]  # <worktree>/HENRI V2
sys.path.insert(0, str(REPO))

from e2_egress_knn import (  # noqa: E402
    E2Config, KNNSoftmaxProbe, e2_alignment_margin, e2_negative_control,
    knn_softmax_probe,
)


@pytest.fixture(scope="module")
def config() -> E2Config:
    return E2Config(d_model=65536, num_blocks=8192, block_dim=8,
                    d_bottleneck=256, d_target=896, vocab_size=151936,
                    k=16, tau=0.07, seed=20260909, device="cpu",
                    dtype=torch.float32)


def test_probe_boundary_shapes(config):
    probe = KNNSoftmaxProbe(config)
    feats = torch.randn(4, config.d_target)
    teacher = torch.randn(512, config.d_target)
    z_hat, w = probe(feats, teacher)
    assert z_hat.shape == (4, config.d_target)
    assert w.shape == (4, min(config.k, 512))


def test_softmax_weights_sum_to_one(config):
    probe = KNNSoftmaxProbe(config)
    feats = torch.randn(3, config.d_target)
    teacher = torch.randn(64, config.d_target)
    _, w = probe(feats, teacher)
    assert torch.allclose(w.sum(dim=-1), torch.ones(3), atol=1e-5)


def test_probe_zero_trainable(config):
    probe = KNNSoftmaxProbe(config)
    assert sum(p.numel() for p in probe.parameters()) == 0


def test_no_dense_dd(config):
    probe = KNNSoftmaxProbe(config)
    for p in probe.parameters():
        assert not (p.shape[0] == config.d_model and p.shape[1] == config.d_model)


def test_margin_semantics_positive_when_aligned(config):
    torch.manual_seed(config.seed)
    feats = torch.randn(8, config.d_target)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    teacher = torch.randn(512, config.d_target)
    # ALIGNED CLUSTER (batch of 8): rows 0..7 = feats exactly; rows 8..8+2k-1 =
    # noisy copies, so the top-k centroid lands near feats for EACH input.
    g = torch.Generator().manual_seed(config.seed + 1)
    teacher[:8] = feats
    n_copies = 2 * config.k  # 32 rows of noisy aligned copies (k=16)
    base = feats.repeat((n_copies + 7) // 8, 1)[:n_copies]
    teacher[8:8 + n_copies] = (
        base + 0.02 * torch.randn(n_copies, config.d_target, generator=g))
    probe = KNNSoftmaxProbe(config)
    z_hat, _ = probe(feats, teacher)
    rand_teacher = torch.randn(512, config.d_target)
    z_untrained, _ = probe(feats, rand_teacher)
    ctrl = e2_negative_control(feats, feats, teacher, k=config.k, tau=config.tau)
    margin = e2_alignment_margin(z_hat, feats, z_untrained, ctrl)
    align_trained = float((F.normalize(z_hat, dim=-1) * feats).sum(dim=-1).mean())
    assert align_trained > 0.9, f"cluster centroid not aligned: {align_trained:.4f}"
    assert margin > 0.05, f"alignment margin {margin:.4f} <= 0.05"


def test_margin_metric_rejects_vacuous_zero():
    with pytest.raises((ValueError, IndexError, RuntimeError)):
        e2_alignment_margin([], [], [], [])


def test_negative_control_shuffled_targets_lose(config):
    """Aligned pairing must beat a deterministic shuffled pairing."""
    torch.manual_seed(config.seed)
    feats = torch.randn(8, config.d_target)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    teacher = torch.randn(256, config.d_target)
    teacher[:8] = feats  # rows 0..7 align exactly with feats
    probe = KNNSoftmaxProbe(config)
    z_ok, _ = probe(feats, teacher)
    z_untrained, _ = probe(feats, torch.randn(256, config.d_target))
    ctrl = e2_negative_control(feats, feats, teacher, k=config.k, tau=config.tau, seed=1)
    margin_ok = e2_alignment_margin(z_ok, feats, z_untrained, ctrl)
    # shuffled pairing: inputs permuted, targets kept in original order
    perm = torch.randperm(8, generator=torch.Generator().manual_seed(1))
    z_shuf, _ = probe(feats[perm], teacher)
    margin_bad = e2_alignment_margin(z_shuf, feats, z_untrained, ctrl)
    assert ctrl < 0.5, f"negative control not collapsed: {ctrl:.4f}"
    assert margin_ok > 0.05, f"aligned margin {margin_ok:.4f} <= 0.05"
    assert margin_ok > margin_bad, f"shuffled beat aligned: {margin_ok} vs {margin_bad}"


def test_default_off_non_import():
    prod_files = ["production_arc_run.py", "henri_decoder.py", "henri_egress.py"]
    for f in prod_files:
        src = (REPO / f).read_text(encoding="utf-8")
        assert "e2_egress_knn" not in src, f"{f} imports E2 module"
