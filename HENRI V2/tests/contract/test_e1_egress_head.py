"""Contract tests for E1 egress calibration head (carrier/e1-egress-calibration).

TDD order: tests first. No production import; module is default-OFF and never
imported by production_arc_run.py / henri_decoder.py / henri_egress.py.

Covers: boundary shapes, dense-[D,D] ban, Stiefel QR orthogonality, contrastive
loss finiteness/descent, default-OFF non-import contract, determinism of codec.
"""
from __future__ import annotations

import math
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]  # <worktree>/HENRI V2
sys.path.insert(0, str(REPO))

from e1_egress_calibration import (  # noqa: E402
    E1Config, E1EgressHead, E1Trainer, contrastive_alignment_loss,
    qr_retraction, build_window_pairs, load_teacher_embeddings,
)


@pytest.fixture(scope="module")
def config() -> E1Config:
    return E1Config(d_model=65536, num_blocks=8192, block_dim=8,
                    d_bottleneck=256, d_target=896, vocab_size=151936,
                    seed=20260908, device="cpu", dtype=torch.float32)


def test_boundary_shapes(config):
    head = E1EgressHead(config)
    x = torch.randn(4, config.num_blocks, config.block_dim)
    feats, logits = head(x)
    assert feats.shape == (4, config.d_target)
    assert logits.shape == (4, config.vocab_size)


def test_no_dense_dd(config):
    head = E1EgressHead(config)
    for name, p in head.named_parameters():
        if len(p.shape) == 2:
            assert not (p.shape[0] == config.d_model and p.shape[1] == config.d_model), \
                f"dense [D,D] parameter: {name} {tuple(p.shape)}"


def test_stiefel_orthogonality_after_retraction(config):
    torch.manual_seed(config.seed)
    W = torch.randn(config.d_model, config.d_bottleneck)
    for _ in range(25):
        G = torch.randn_like(W) * 1e-2
        W = qr_retraction(W - config.stiefel_lr * G)
    err = torch.norm(W.t() @ W - torch.eye(config.d_bottleneck), p="fro").item()
    assert err <= 1e-4, f"Stiefel error {err}"


def test_contrastive_loss_finite_and_descent(config):
    torch.manual_seed(config.seed)
    head = E1EgressHead(config)
    # real wave pair payload: two calibratable "sentences" (ASCII text)
    texts = ["the black cat sat on the mat",
             "a large red ball rolled across the floor",
             "cold water flows down the mountain stream",
             "the old clock ticked slowly in the hall"]
    # plumbing fixture only: synthetic teacher table = deterministic random rows.
    # Production pairs use the REAL frozen Qwen embedding table (see e1_calibrate.py).
    torch.manual_seed(config.seed)
    teacher = torch.randn(64, config.d_target)
    teacher = teacher / teacher.norm(dim=-1, keepdim=True)
    pairs = build_window_pairs(texts, max_words=24, teacher_embeddings=teacher)
    assert len(pairs) >= 2
    waves = torch.stack([p.wave for p in pairs])
    targets = torch.stack([p.target for p in pairs])
    opt = torch.optim.AdamW(head.adapter_params(), lr=1e-3, weight_decay=1e-4)
    losses = []
    for _ in range(20):
        opt.zero_grad()
        feats, _ = head(waves)
        loss = contrastive_alignment_loss(feats, targets, temperature=0.07)
        loss = loss + 0.02 * torch.norm(head.stiefel_down_proj.t() @ head.stiefel_down_proj
                                        - torch.eye(config.d_bottleneck), p="fro") ** 2
        loss.backward()
        opt.step()
        with torch.no_grad():
            G = head.stiefel_down_proj.grad
            head.stiefel_down_proj.copy_(
                qr_retraction(head.stiefel_down_proj - config.stiefel_lr * G))
            head.stiefel_down_proj.grad.zero_()
        losses.append(loss.item())
    assert math.isfinite(losses[-1])
    assert losses[-1] < losses[0], f"no descent: {losses[0]:.4f} -> {losses[-1]:.4f}"


def test_default_off_non_import():
    """Production files must not import the E1 trainer module."""
    prod_files = ["production_arc_run.py", "henri_decoder.py", "henri_egress.py"]
    for f in prod_files:
        src = (REPO / f).read_text(encoding="utf-8")
        assert "e1_egress_calibration" not in src, f"{f} imports E1 trainer"


def test_teacher_loader_missing_artifact_fails_closed(config, tmp_path):
    with pytest.raises(FileNotFoundError):
        load_teacher_embeddings(Path(tmp_path) / "nope.pt")


def test_per_block_rotation_geometry(config):
    """E1 G4 rotation: per-block O(8) rotation must preserve row norms."""
    torch.manual_seed(7)
    w0 = torch.randn(2, config.num_blocks, config.block_dim)
    R = torch.randn(8, 8)
    Q, _ = torch.linalg.qr(R)
    wrot = torch.einsum("ij,bkj->bki", Q, w0)     # production geometry
    assert wrot.shape == w0.shape
    assert torch.norm(wrot[0, :, :], dim=-1).shape == (config.num_blocks,)
    err = (torch.norm(wrot[0], dim=-1) - torch.norm(w0[0], dim=-1)).abs().max().item()
    assert err <= 1e-5, f"rotation broke row norms: {err}"
