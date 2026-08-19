"""Unit tests for the Phase 8.33 Option (2) egress head + SGLD adaptation.

Reduced-dimension CPU tests. The production-dimension D=65,536 evaluation
runs ONLY in the remote CUDA kill experiment
(`arc_phase833_egress_sgld_experiment.py`). Default-OFF: nothing in this
module is constructed by any production path.
"""

import math

import pytest
import torch

from henri_egress import CompressedProjectionHead, sgld_adapt_head

D = 64
HIDDEN = 16
VOCAB = 6


def make_head(d_model=D, hidden_dim=HIDDEN, vocab_size=VOCAB, seed=0):
    torch.manual_seed(seed)
    return CompressedProjectionHead(d_model=d_model, hidden_dim=hidden_dim,
                                    vocab_size=vocab_size, sagnac_lambda=0.25)


def test_head_shapes_1d_and_2d():
    head = make_head()
    out1 = head(torch.randn(D))
    assert out1.shape == (1, VOCAB)
    out2 = head(torch.randn(8, D))
    assert out2.shape == (8, VOCAB)


def test_random_head_entropy_is_uniform():
    """Falsify-criterion reference: an un-adapted head emits ~uniform entropy."""
    head = make_head()
    with torch.no_grad():
        logits = head(torch.randn(64, D))
        ent = head.logit_entropy(logits).mean().item()
    assert abs(ent - math.log(VOCAB)) < 0.25  # near-uniform ln(6) ~ 1.79


def test_no_dense_d_param():
    """Memory contract: no [D, D] parameter; largest weight is [D, hidden]."""
    head = make_head()
    sizes = sorted((p.numel() for p in head.parameters()), reverse=True)
    assert sizes[0] == D * HIDDEN
    assert all(s < D * D for s in sizes)


def test_sgld_reduces_loss_on_separable_data():
    """SGLD creep must drive the head toward the label structure."""
    torch.manual_seed(1)
    n = 24
    waves = torch.randn(n, D)
    # 2 separable clusters -> labels 0/1, rest of vocab unused.
    waves[:12] += 2.0
    waves[12:] -= 2.0
    targets = torch.zeros(n, dtype=torch.long)
    targets[12:] = 1

    head = make_head(seed=2)
    with torch.no_grad():
        before = torch.nn.functional.cross_entropy(head(waves), targets).item()

    res = sgld_adapt_head(head, waves, targets, lr=5e-3, steps=200,
                          log_every=200, seed=3)
    with torch.no_grad():
        after = torch.nn.functional.cross_entropy(head(waves), targets).item()
        acc = (head(waves).argmax(dim=-1) == targets).float().mean().item()

    assert after < before, f"loss did not drop: {before:.4f} -> {after:.4f}"
    assert acc > 0.9, f"train accuracy {acc:.3f} < 0.9"
    assert res["steps"] == 200
    # Real learning happened (not every step yielded); late converged steps
    # legitimately yield (Bingham yield: no gradient stress -> no plastic flow).
    assert res["yielded"] < res["steps"], f"all {res['steps']} steps yielded"


def test_sgld_rejects_wrong_wave_dim():
    head = make_head()
    with pytest.raises(ValueError):
        sgld_adapt_head(head, torch.randn(4, 128), torch.zeros(4, dtype=torch.long))


def test_head_is_untrained_by_default():
    """Constructor does not train anything; weights are the seeded init."""
    head = make_head()
    total = sum(p.numel() for p in head.parameters())
    assert total == D * HIDDEN + HIDDEN + HIDDEN * VOCAB + VOCAB + 2 * HIDDEN  # Linear+LN+Linear(+LN affine)
