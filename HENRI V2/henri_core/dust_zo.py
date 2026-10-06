"""Zeroth-order (node-perturbation) descent estimator for Project HENRI.

STATUS: DEFAULT OFF. Nothing here executes unless a caller opts in through
DustZOConfig + dust_zo_enabled(). The default HENRI path is byte-identical.

Clean-room implementation. No upstream code is copied or vendored. The estimator
is implemented from the published mathematics in the supplied specification
(HENRI-SPEC-2026-DUST-ZEROTH-ORDER-01) and cross-checked against the behaviour of
the MIT-licensed reference implementation:

    repo    https://github.com/qlabs-eng/dust
    commit  2fdb01ba91ca38368a2b6b31a2697d77512ad175
    file    dust.py  sha256 154a03a77cc4b02c9040ef5f375a869fe8c612654c7fe7890a37cc4a74b718a7
    license MIT

Estimator, per token t, on a layer output y_t = W x_t:

    a_k   ~ N(0, I_D)                          virtual perturbation (activation space)
    c_k    = L_bar - L(y_t + sigma * a_k)      one-sided loss drop, CENTRED across draws
    r_t    = sum_{s >= t} gamma^(s-t) c_s      causal (discounted) credit
    g_hat  = (1 / (K * sigma)) * sum_k r_k a_k -> estimates -grad_y L   (descent at y)
    G_W    = sum_t g_hat_t (x) x_t^T           -> estimates -grad_W L   (descent at W)

Sign convention: this module returns the DESCENT direction (-grad L), which is
what an optimizer consumes. Verified against autograd by exp_dust_g1.py.

Scope limits (disclosed, not hidden):
  * Estimates a descent direction for a LINEAR layer output. It does not replace
    BPTT for the whole model. No claim of backpropagation removal.
  * Cost is K forward evaluations per step. K=256 costs ~K/2 forward-equivalents
    of a single backward pass on digital hardware. The photonic argument in the
    specification is a HYPOTHESIS about future hardware, not a measured result.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable

import torch

ENV_FLAG = "HENRI_DUST_ZO"


def dust_zo_enabled(cli_flag: bool = False) -> bool:
    """True only when the caller opts in. Default OFF, both env and CLI."""
    if cli_flag:
        return True
    return os.environ.get(ENV_FLAG, "0").strip().lower() not in ("", "0", "false", "no")


@dataclass(frozen=True)
class DustZOConfig:
    """Frozen estimator hyperparameters. Pin these in any pre-registration."""

    K: int = 256          # virtual population size (draws per step)
    sigma: float = 0.05   # perturbation scale
    gamma: float = 0.98   # causal credit decay per token of lag
    seed: int = 20261005  # RNG pin (reproducibility)
    chunk: int = 32       # draws evaluated per forward chunk (memory guard)


def _discounted_credit(c: torch.Tensor, gamma: float) -> torch.Tensor:
    """r_t = sum_{s >= t} gamma^(s-t) c_s along the last axis."""
    t_len = c.shape[-1]
    out = torch.empty_like(c)
    run = torch.zeros_like(c[..., 0])
    for t in range(t_len - 1, -1, -1):
        run = c[..., t] + gamma * run
        out[..., t] = run
    return out


def node_perturbation_descent(
    y: torch.Tensor,
    x: torch.Tensor,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    labels: torch.Tensor,
    cfg: DustZOConfig,
) -> torch.Tensor:
    """Estimate the descent direction -dL/dW for a layer with output y = W x.

    y:       [N, T, D]     nominal layer output (no grad)
    x:       [N, T, d_in]  cached layer input
    loss_fn: (y, labels) -> [M, T] per-token loss, no cross-token coupling
    labels:  [N, T]
    returns: [D, d_in] estimated descent direction. Caller applies W += eta * G.
    """
    if y.dim() != 3 or x.dim() != 3:
        raise ValueError("y and x must be [N, T, *]")
    # C3 review fix: this module is CPU-only. The generator is pinned to CPU and
    # the accumulator/perturbation tensors omit a device argument, so a CUDA
    # input would either raise at the first add (cpu + cuda) or silently shift
    # work to the host. Refuse loudly instead of half-working.
    if y.is_cuda or x.is_cuda or labels.is_cuda:
        raise RuntimeError(
            "dust_zo is CPU-only: y/x/labels must be on CPU. henri_core has no "
            "device plumbing; thread a device argument before enabling CUDA here.")
    n_batch, t_len, dim = y.shape
    if cfg.K < 1 or cfg.sigma <= 0:
        raise ValueError("require K >= 1 and sigma > 0")
    if cfg.chunk < 1:
        raise ValueError("require chunk >= 1")

    gen = torch.Generator(device="cpu").manual_seed(int(cfg.seed))
    with torch.no_grad():
        clean = loss_fn(y, labels)                      # [N, T]
        acc = torch.zeros(n_batch, t_len, dim, dtype=y.dtype)
        done = 0
        while done < cfg.K:
            n = min(cfg.chunk, cfg.K - done)
            a = torch.randn(n, n_batch, t_len, dim, generator=gen, dtype=y.dtype)
            lab = labels.unsqueeze(0).expand(n, -1, -1).reshape(n * n_batch, t_len)
            yp = (y.unsqueeze(0) + cfg.sigma * a).reshape(n * n_batch, t_len, dim)
            lp = loss_fn(yp, lab).reshape(n, n_batch, t_len)
            c = clean.unsqueeze(0) - lp                 # [n, N, T] one-sided drop
            c = c - c.mean(dim=0, keepdim=True)         # CENTRE across draws
            r = _discounted_credit(c, cfg.gamma)        # [n, N, T]
            # sum_k r_k a_k  ->  [N, T, D]
            acc = acc + torch.einsum("nit,nitd->itd", r, a)
            done += n
        acc = acc / (cfg.K * cfg.sigma)
        grad = torch.einsum("ntr,nti->ri", acc, x)      # [D, d_in]
    return grad


def apply_descent_(weight: torch.Tensor, grad: torch.Tensor, eta: float) -> None:
    """In-place weight step. weight and grad share shape [D, d_in]."""
    if weight.shape != grad.shape:
        raise ValueError(f"shape mismatch {tuple(weight.shape)} vs {tuple(grad.shape)}")
    with torch.no_grad():
        weight.add_(grad, alpha=float(eta))


def matcher_loss_fn(readout: torch.Tensor, scale: float = 1.0):
    """A fixed readout + cross-entropy loss, mirroring a downstream head.

    readout: [C, D]. Returns loss_fn(y, labels) -> [M, T].
    """
    C = readout.shape[0]

    def loss_fn(y: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        m, t, _ = y.shape
        logits = (y @ readout.t()) * scale
        return torch.nn.functional.cross_entropy(
            logits.reshape(m * t, C), labels.reshape(m * t), reduction="none"
        ).reshape(m, t)

    return loss_fn
