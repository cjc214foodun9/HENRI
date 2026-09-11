"""Contract tests — E3 egress-head reform (carrier/e3-egress-reform).

Prereg: experiments/verification/e3_egress_reform_prereg.md (sealed).
Arm A: probe-distribution readout (zero training) over the frozen E2 checkpoint.
Arm B: generative CE head warm-started from E2 with a FROZEN tied teacher
readout (lm_head excluded from training, never called, kept equal to E).

TDD order: this file is written before e3_egress_reform.py / e3_calibrate.py.
"""

import sys
from pathlib import Path

import torch

HENRI2 = Path(__file__).resolve().parents[2]  # <wt>/HENRI V2
sys.path.insert(0, str(HENRI2))

from e1_egress_calibration import E1Config, E1EgressHead  # noqa: E402
from e3_egress_reform import (  # noqa: E402
    E3GenerativeHead,
    probe_token_distribution,
)

# Reduced-scale fixture geometry (same class as the E1/E2 contract fixtures)
NB, BD, DB, DT, V = 8, 8, 16, 32, 64


def _tiny_cfg(**kw):
    base = dict(d_model=NB * BD, num_blocks=NB, block_dim=BD,
                d_bottleneck=DB, d_target=DT, vocab_size=V,
                seed=20260908, device="cpu")
    base.update(kw)
    return E1Config(**base)


def _tiny_ckpt(tmp_path):
    """E2-format payload {config, model_state, telemetry} at reduced scale."""
    head = E1EgressHead(_tiny_cfg())
    payload = {"config": {k: v for k, v in vars(head.config).items()},
               "model_state": head.state_dict(),
               "telemetry": []}
    p = tmp_path / "e2_tiny.pt"
    torch.save(payload, str(p))
    return p


def test_probe_token_distribution_shapes_and_finiteness():
    torch.manual_seed(2)
    E = torch.randn(V, DT)
    feats = torch.randn(3, DT)
    scores, z_hat = probe_token_distribution(feats, E, k=4, tau=0.07)
    assert scores.shape == (3, V)
    assert z_hat.shape == (3, DT)
    assert torch.isfinite(scores).all() and torch.isfinite(z_hat).all()
    # k-NN centroid is L2-normalized
    assert torch.allclose(z_hat.norm(dim=-1), torch.ones(3), atol=1e-5)


def test_e3_head_loads_e2_checkpoint_format(tmp_path):
    ck = _tiny_ckpt(tmp_path)
    E = torch.randn(V, DT)
    head = E3GenerativeHead.from_e2_checkpoint(ck, E_table=E, device="cpu")
    feats, _ = head(torch.randn(2, NB, BD))
    assert feats.shape == (2, DT)
    # tied frozen readout equals the teacher table after the warm start
    assert torch.equal(head.head.lm_head.weight.data, E)


def test_lm_head_frozen_and_excluded_from_trainables():
    torch.manual_seed(0)
    E = torch.randn(V, DT)
    head = E3GenerativeHead(_tiny_cfg(), E_table=E)
    assert head.head.lm_head.weight.requires_grad is False
    assert torch.equal(head.head.lm_head.weight.data, E)
    n_all = len(list(head.head.parameters()))
    n_train = len(head.trainable_parameters())
    # exactly two parameters excluded: stiefel_down_proj + lm_head.weight
    assert n_train == n_all - 2


def test_ce_loss_finite_and_descends_tiny():
    torch.manual_seed(1)
    E = torch.randn(V, DT)
    head = E3GenerativeHead(_tiny_cfg(), E_table=E)
    waves = torch.randn(24, NB, BD)
    golds = torch.randint(0, V, (24,))
    E_before = head.head.lm_head.weight.detach().clone()
    losses = head.fit_ce(waves, golds, epochs=6, batch=8, lr=3e-3)
    assert len(losses) >= 2
    assert all(float(torch.isfinite(torch.tensor(l))) for l in losses)
    assert losses[-1] < losses[0]
    # the frozen tied readout must be untouched by training
    assert torch.equal(head.head.lm_head.weight.detach(), E_before)


def test_isometry_error_reported_small_after_construction():
    torch.manual_seed(3)
    E = torch.randn(V, DT)
    head = E3GenerativeHead(_tiny_cfg(), E_table=E)
    # constructor applies the QR retraction; error must be tiny
    assert head.isometry_error() < 1e-5


def test_runner_pins_bounds_and_artifacts():
    src = (HENRI2 / "e3_calibrate.py").read_text(encoding="utf-8")
    # RETIRED 2026-09-11: the legacy 0.285/0.640 bound sits below the
    # measured trivial baseline and must never be reinstated.
    assert "P_AT_1_BOUND = 0.285" not in src
    assert "P_AT_5_BOUND = 0.640" not in src
    assert "load_registered_bounds" in src
    assert "LEGACY_BOUNDS_RETIRED" in src
    assert "08747c70" in src and "e83889ba" in src and "48b174e9" in src
    assert 'os.environ.get("P_AT_1_BOUND"' not in src
    assert 'os.environ.get("P_AT_5_BOUND"' not in src
    assert "GOLD_TOKEN_PREFIX_MISMATCH" in src
    assert "CONDITIONAL_SAME_CORPUS_HELDOUT" in src
    assert "build_pairs_ordered" in src
    assert "E3_VERDICT=" in src


def test_no_dense_d_by_d_allocation():
    src = (HENRI2 / "e3_egress_reform.py").read_text(encoding="utf-8")
    assert "torch.eye(65536" not in src
    assert "65536, 65536" not in src
    assert "151936, 151936" not in src
