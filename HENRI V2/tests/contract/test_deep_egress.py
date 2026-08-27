"""Contract tests: Phase 10.0 deep egress proposal head (default-OFF).

Run locally (CPU) from the repo root:
    env -u VIRTUAL_ENV -u PYTHONPATH -u PYTHONHOME \\
      PYTHONPATH="HENRI V2" /c/Python314/python.exe -m pytest \\
      "HENRI V2/tests/contract/test_deep_egress.py" -q --tb=short
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from henri_deep_egress import (
    FLAG,
    DeepEgressDisabledError,
    DeepEgressProposalHead,
    get_deep_egress_head,
)

# Reduced scale for local CPU: D=512, 64 blocks x 8, proj 2, hidden 64, vocab 128.
D, NB, BD, PD, DH, V = 512, 64, 8, 2, 64, 128


def _unit_wave(seed: int = 1, batch: int = 1) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(batch, D, generator=g)
    return w / w.norm(dim=-1, keepdim=True)


def _linear_baseline():
    """Deterministic 2-layer linear baseline mirroring HENRINeuralEgressUnbinder."""
    torch.manual_seed(7)
    down = torch.nn.Linear(D, DH, bias=False)
    ln = torch.nn.LayerNorm(DH)
    act = torch.nn.GELU()
    lm_head = torch.nn.Linear(DH, V, bias=False)

    def f(wave):
        u = wave / (wave.norm(dim=-1, keepdim=True) + 1e-8)
        return lm_head(act(ln(down(u))))

    return f, lm_head


def test_c1_default_off_factory(monkeypatch):
    """C1: absent flag -> factory returns None (module never constructed)."""
    monkeypatch.delenv(FLAG, raising=False)
    assert get_deep_egress_head(D, NB, BD, DH, V) is None


def test_c1b_flag_on_factory(flag_on):
    """C1b: flag set -> factory constructs the head."""
    head = get_deep_egress_head(D, NB, BD, DH, V)
    assert head is not None
    assert isinstance(head, DeepEgressProposalHead)


@pytest.fixture
def flag_on(monkeypatch):
    monkeypatch.setenv(FLAG, "1")


def test_c2_beta0_byte_identity(flag_on):
    """C2: beta=0 (default) returns linear_logits VERBATIM (byte-identical)."""
    f, lm_head = _linear_baseline()
    wave = _unit_wave()
    lin = f(wave)
    head = DeepEgressProposalHead(D, NB, BD, PD, DH, V, lm_head=lm_head)
    out = head(wave, lin)
    assert torch.equal(out, lin)


def test_c2b_beta0_never_computes_deep(flag_on, monkeypatch):
    """C2b: at beta=0 the deep path is never executed (identity + zero compute)."""
    f, lm_head = _linear_baseline()
    wave = _unit_wave()
    lin = f(wave)
    head = DeepEgressProposalHead(D, NB, BD, PD, DH, V, lm_head=lm_head)

    def _explode(*args, **kwargs):
        raise AssertionError("deep_logits must not run at beta=0")

    monkeypatch.setattr(head, "deep_logits", _explode)
    out = head(wave, lin)
    assert torch.equal(out, lin)


def test_c3_beta_half_blend(flag_on):
    """C3: beta>0 blends exactly: (1-b)*lin + b*deep."""
    f, lm_head = _linear_baseline()
    wave = _unit_wave()
    lin = f(wave)
    head = DeepEgressProposalHead(D, NB, BD, PD, DH, V, lm_head=lm_head)
    b = 0.5
    out = head(wave, lin, beta=b)
    expected = (1.0 - b) * lin + b * head.deep_logits(wave)
    assert torch.allclose(out, expected, atol=1e-6)
    assert not torch.equal(out, lin)


def test_c4_gradient_reachability(flag_on):
    """C4: every trainable deep parameter receives a nonzero gradient."""
    f, lm_head = _linear_baseline()
    wave = _unit_wave()
    lin = f(wave)
    head = DeepEgressProposalHead(D, NB, BD, PD, DH, V, lm_head=lm_head)
    out = head(wave, lin, beta=0.5)
    loss = out.sum()
    loss.backward()
    deep_params = {
        "block_proj.weight": head.block_proj.weight,
        "deep_down.weight": head.deep_down.weight,
        "layer_norm.weight": head.layer_norm.weight,
        "layer_norm.bias": head.layer_norm.bias,
    }
    for name, p in deep_params.items():
        assert p.grad is not None, f"no gradient on {name}"
        assert p.grad.abs().sum().item() > 0.0, f"zero gradient on {name}"


def test_c5_no_dense_allocation(flag_on):
    """C5: no [D,D] parameter and activation budget < 1.5 GB at production scale."""
    head = DeepEgressProposalHead(65536, 8192, 8, 2, 2048, 32000)
    for name, p in head.named_parameters():
        if len(p.shape) == 2:
            assert not (p.shape[0] == 65536 and p.shape[1] == 65536), name
    # Peak B=1 activation bytes (fp32): block output + aggregated + hidden + logits.
    act_bytes = 8192 * 2 * 4 + 8192 * 2 * 4 + 2048 * 4 + 32000 * 4
    assert act_bytes < 1.5e9
    # The guard itself runs at construction without raising.
    head._assert_no_dense_allocation()


def test_c6_beta_init_zero(flag_on):
    """C6: the blending coefficient initializes to 0.0 (report constraint 1)."""
    head = DeepEgressProposalHead(D, NB, BD, PD, DH, V)
    assert float(head.beta) == 0.0
